# backend/train.py
"""
Two-phase training (PyTorch).
Phase 1: frozen backbone, train adapter + head (~20 epochs, LR from config)
Phase 2: unfreeze last backbone blocks, fine-tune (up to EPOCHS, low LR)

Saves:
  - best checkpoint → backend/models/hemtrace_best.pt
  - confusion_matrix.png, training_history.png, classification_report.txt
"""

import os
import sys
import copy
import random

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from config import (
    CLASSES,
    MODEL_PATH,
    HISTORY_PATH,
    CM_PATH,
    RESULTS_DIR,
    BATCH_SIZE,
    EPOCHS,
    SEED,
    LR_PHASE1,
    LR_PHASE2,
    NUM_CLASSES,
)
from dataset import build_datasets
from model import HemaTraceNet, build_model, print_model_summary


def _require_gpu():
    """Require CUDA unless HEMATRACE_ALLOW_CPU=1 (dev / no GPU wheel)."""
    if os.environ.get("HEMATRACE_ALLOW_CPU", "").strip().lower() in ("1", "true", "yes"):
        print("\n⚠️  HEMATRACE_ALLOW_CPU is set — training may run on CPU.\n")
        return "cpu"
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Install PyTorch with a CUDA-enabled wheel "
            "(see https://pytorch.org/get-started/locally/ ) and an NVIDIA driver.\n"
            "For CPU-only training, set: HEMATRACE_ALLOW_CPU=1"
        )
    print(f"\n🖥️  Training on GPU: {torch.cuda.get_device_name(0)}")
    return "cuda"


def _set_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def _np_to_torch_xy(X_np, y_onehot_np):
    """X: (N,H,W,1) float32 → (N,1,H,W) CPU tensor; y: one-hot → class indices."""
    X = torch.from_numpy(np.transpose(X_np.astype(np.float32), (0, 3, 1, 2)))
    y = torch.from_numpy(np.argmax(y_onehot_np, axis=1).astype(np.int64))
    return X, y


def compute_class_weights_tensor(y_onehot):
    """Class index counts → tensor weights for CrossEntropyLoss (same formula as before)."""
    y_int = np.argmax(y_onehot, axis=1)
    counts = np.bincount(y_int, minlength=len(CLASSES))
    total = counts.sum()
    w = np.array(
        [total / (len(CLASSES) * c) if c > 0 else 1.0 for c in counts],
        dtype=np.float32,
    )
    print("\n⚖️  Class weights:")
    for i, cls in enumerate(CLASSES):
        print(f"   {cls:4s}: {w[i]:.3f}  (n={counts[i]})")
    return torch.from_numpy(w)


def _train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        bs = xb.size(0)
        total_loss += loss.item() * bs
        correct += (logits.argmax(dim=1) == yb).sum().item()
        n += bs
    return total_loss / max(n, 1), correct / max(n, 1)


@torch.no_grad()
def _evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        bs = xb.size(0)
        total_loss += loss.item() * bs
        correct += (logits.argmax(dim=1) == yb).sum().item()
        n += bs
    return total_loss / max(n, 1), correct / max(n, 1)


def _save_checkpoint(model, val_acc, epoch, path):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "val_acc": float(val_acc),
            "epoch": int(epoch),
            "num_classes": NUM_CLASSES,
        },
        path,
    )


def _run_phase(
    model,
    train_loader,
    val_loader,
    device,
    criterion,
    lr,
    max_epochs,
    patience,
    scheduler_factor,
    scheduler_patience,
    scheduler_min_lr,
    phase_tag,
):
    """Returns history dict with keys accuracy, val_accuracy, loss, val_loss (Keras-style names)."""
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=scheduler_min_lr,
    )

    history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}
    best_val_acc = -1.0
    best_state = None
    bad_epochs = 0

    for epoch in range(1, max_epochs + 1):
        tr_loss, tr_acc = _train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = _evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["accuracy"].append(tr_acc)
        history["val_accuracy"].append(val_acc)
        history["loss"].append(tr_loss)
        history["val_loss"].append(val_loss)

        print(
            f"{phase_tag} Epoch {epoch:3d}/{max_epochs}  "
            f"loss={tr_loss:.4f} acc={tr_acc:.4f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
            _save_checkpoint(model, val_acc, epoch, MODEL_PATH)
            print(f"   → saved best checkpoint (val_acc={val_acc:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"   Early stopping ({patience} epochs without val_acc improvement)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


class HistoryBundle:
    """Wrap phase histories like Keras History for plot_history."""

    def __init__(self, history_dict):
        self.history = history_dict


def plot_history(h1, h2=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("HemaTrace Training History", fontweight="bold")

    def _plot_phase(ax_acc, ax_loss, hist, label_prefix, offset=0):
        h = hist.history
        epochs = range(offset + 1, offset + len(h["accuracy"]) + 1)
        ax_acc.plot(epochs, h["accuracy"], label=f"{label_prefix} train")
        ax_acc.plot(epochs, h["val_accuracy"], label=f"{label_prefix} val", linestyle="--")
        ax_loss.plot(epochs, h["loss"], label=f"{label_prefix} train")
        ax_loss.plot(epochs, h["val_loss"], label=f"{label_prefix} val", linestyle="--")

    _plot_phase(axes[0], axes[1], h1, "Ph1", offset=0)
    offset = len(h1.history["accuracy"])
    if h2 is not None:
        _plot_phase(axes[0], axes[1], h2, "Ph2", offset=offset)
        axes[0].axvline(x=offset, color="gray", linestyle=":", label="phase boundary")
        axes[1].axvline(x=offset, color="gray", linestyle=":")

    axes[0].set_title("Accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title("Loss")
    axes[1].set_ylabel("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(HISTORY_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n📊 Training history saved → {HISTORY_PATH}")


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("HemaTrace — Confusion Matrix (Test Set)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Confusion matrix saved → {CM_PATH}")


@torch.no_grad()
def _evaluate_test(model, X_te, y_te_idx, criterion, device, batch_size):
    model.eval()
    ds = TensorDataset(X_te, y_te_idx)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    correct = 0
    n = 0
    all_pred = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        bs = xb.size(0)
        total_loss += loss.item() * bs
        pred = logits.argmax(dim=1)
        all_pred.append(pred.cpu().numpy())
        correct += (pred == yb).sum().item()
        n += bs
    y_pred = np.concatenate(all_pred, axis=0)
    return total_loss / max(n, 1), correct / max(n, 1), y_pred


def train():
    print("=" * 60)
    print("🩸 HemaTrace — Training Pipeline (PyTorch)")
    print("=" * 60)

    device_str = _require_gpu()
    device = torch.device(device_str)
    _set_seeds()

    (X_tr, y_tr, X_val, y_val, X_te, y_te, _info) = build_datasets(verbose=True)

    weight_t = compute_class_weights_tensor(y_tr).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_t, label_smoothing=0.1)

    X_tr_t, y_tr_t = _np_to_torch_xy(X_tr, y_tr)
    X_val_t, y_val_t = _np_to_torch_xy(X_val, y_val)
    X_te_t, y_te_t = _np_to_torch_xy(X_te, y_te)

    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # ── Phase 1 ─────────────────────────────────────────────────────────────
    print("\n🚀 Phase 1: Training head (backbone frozen) ...")
    model, _ = build_model(phase=1)
    model = model.to(device)
    print_model_summary(model)

    hist1_dict = _run_phase(
        model,
        train_loader,
        val_loader,
        device,
        criterion,
        lr=LR_PHASE1,
        max_epochs=20,
        patience=10,
        scheduler_factor=0.3,
        scheduler_patience=4,
        scheduler_min_lr=1e-8,
        phase_tag="Ph1",
    )
    h1 = HistoryBundle(hist1_dict)
    best_p1 = max(hist1_dict["val_accuracy"])
    print(f"\n✅ Phase 1 complete — best val accuracy: {best_p1 * 100:.2f}%")

    # ── Phase 2 ─────────────────────────────────────────────────────────────
    print("\n🚀 Phase 2: Fine-tuning (backbone partial unfreeze) ...")
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = HemaTraceNet().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.set_phase(2)
    print_model_summary(model)

    hist2_dict = _run_phase(
        model,
        train_loader,
        val_loader,
        device,
        criterion,
        lr=LR_PHASE2,
        max_epochs=EPOCHS,
        patience=15,
        scheduler_factor=0.3,
        scheduler_patience=5,
        scheduler_min_lr=1e-9,
        phase_tag="Ph2",
    )
    h2 = HistoryBundle(hist2_dict)
    best_p2 = max(hist2_dict["val_accuracy"])
    print(f"\n✅ Phase 2 complete — best val accuracy: {best_p2 * 100:.2f}%")

    # ── Test set ────────────────────────────────────────────────────────────
    print("\n📋 Evaluating on held-out test set ...")
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_loss, test_acc, y_pred = _evaluate_test(
        model, X_te_t, y_te_t, criterion, device, BATCH_SIZE
    )
    y_true = y_te_t.cpu().numpy()

    print(f"\n🎯 TEST ACCURACY : {test_acc * 100:.2f}%")
    print(f"   TEST LOSS     : {test_loss:.4f}")

    report = classification_report(y_true, y_pred, target_names=CLASSES)
    print(f"\n📋 Per-class report:\n{report}")

    report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Test Accuracy: {test_acc * 100:.2f}%\n\n{report}")
    print(f"📄 Report saved → {report_path}")

    plot_history(h1, h2)
    plot_confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 60)
    print("🏁 Training complete!")
    print(f"   Best model  : {MODEL_PATH}")
    print(f"   Test acc    : {test_acc * 100:.2f}%")
    print(f"   Val acc ph1 : {best_p1 * 100:.2f}%   ph2: {best_p2 * 100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    train()
