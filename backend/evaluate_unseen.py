# backend/evaluate_unseen.py
"""Run trained model on images under ROOT/unseen dataset/<class>/. Ground truth = folder name."""
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from config import ROOT, CLASSES
from predict import predict_from_path

UNSEEN_ROOT = os.path.join(ROOT, "unseen dataset")
EXTS = (".bmp", ".png", ".jpg", ".jpeg")


def main():
    if not os.path.isdir(UNSEEN_ROOT):
        print(f"Missing folder: {UNSEEN_ROOT}")
        sys.exit(1)

    rows = []
    for cls_dir in sorted(os.listdir(UNSEEN_ROOT)):
        folder = os.path.join(UNSEEN_ROOT, cls_dir)
        if not os.path.isdir(folder):
            continue
        true_label = cls_dir if cls_dir in CLASSES else None
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith(EXTS):
                continue
            path = os.path.join(folder, fname)
            r = predict_from_path(path, tta_steps=10)
            if r.get("error"):
                rows.append((path, true_label, None, None, None, r["error"]))
                continue
            pred = r["predicted_class"]
            conf = r["confidence"]
            ok = true_label is not None and pred == true_label
            rows.append((path, true_label, pred, conf, ok, None))

    # Print table
    print("=" * 100)
    print(f"Unseen evaluation: {UNSEEN_ROOT}")
    print(f"Model: hemtrace_best.pt | Classes: {CLASSES}")
    print("=" * 100)
    print(f"{'File':<50} {'True':<6} {'Pred':<6} {'Conf%':>7} {'Match':>5}")
    print("-" * 100)
    correct = 0
    total_labeled = 0
    for path, true_l, pred, conf, ok, err in rows:
        rel = os.path.relpath(path, UNSEEN_ROOT)
        if err:
            print(f"{rel:<50} {'?':<6} {'?':<6} {'':>7} {'ERR':>5}  {err}")
            continue
        m = "yes" if ok else "no"
        if true_l is not None:
            total_labeled += 1
            if ok:
                correct += 1
        tl = true_l or "—"
        print(f"{rel:<50} {str(tl):<6} {pred:<6} {conf:>7.2f} {m:>5}")

    print("-" * 100)
    if total_labeled:
        acc = 100.0 * correct / total_labeled
        print(f"Labeled images (folder in {CLASSES}): {total_labeled}")
        print(f"Correct: {correct}  |  Accuracy: {acc:.2f}%")
    else:
        print("No subfolders matched CLASSES for accuracy; predictions above are still valid.")

    # Per-class breakdown for labeled
    from collections import defaultdict
    by_true = defaultdict(lambda: [0, 0])
    for _, true_l, pred, _, ok, err in rows:
        if err or true_l is None:
            continue
        by_true[true_l][1] += 1
        if ok:
            by_true[true_l][0] += 1
    if by_true:
        print("\nPer ground-truth folder (when folder name is a class label):")
        for c in sorted(by_true.keys()):
            hit, tot = by_true[c]
            print(f"  {c}: {hit}/{tot} correct ({100*hit/tot:.1f}%)")

    out = os.path.join(ROOT, "backend", "results", "unseen_evaluation.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"Unseen root: {UNSEEN_ROOT}\n")
        if total_labeled:
            f.write(f"Accuracy (labeled): {100*correct/total_labeled:.2f}% ({correct}/{total_labeled})\n\n")
        for path, true_l, pred, conf, ok, err in rows:
            rel = os.path.relpath(path, UNSEEN_ROOT)
            if err:
                f.write(f"{rel}\tERROR\t{err}\n")
            else:
                f.write(f"{rel}\ttrue={true_l}\tpred={pred}\tconf={conf}\tmatch={ok}\n")
    print(f"\nSaved detail → {out}")


if __name__ == "__main__":
    main()
