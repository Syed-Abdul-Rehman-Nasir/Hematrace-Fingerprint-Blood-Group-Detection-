# backend/dataset.py
"""
Loads images from BOTH datasets (developed SecuGen set + Kaggle),
performs person-aware train/val/test split (no data leakage),
and returns ready-to-train numpy arrays.
"""

import os, random
from collections import defaultdict
import numpy as np

from config import (CLASSES, NUM_CLASSES, INPUT_SIZE, PRINTS_PER_PERSON,
                    TRAIN_RATIO, VAL_RATIO, SEED,
                    DEVELOPED_DATA_PATH, KAGGLE_DATA_PATH, KAGGLE_CLASSES)
from preprocess import preprocess_single_image, augment_image


def _to_categorical(y_int, num_classes):
    """One-hot encode integer labels (replaces tensorflow.keras.utils.to_categorical)."""
    y_int = np.asarray(y_int, dtype=np.int64).reshape(-1)
    return np.eye(num_classes, dtype=np.float32)[y_int]


# ── Low-level loaders ─────────────────────────────────────────────────────────

def _load_folder(folder_path, label_idx, source_tag):
    """
    Load all .bmp/.png images from folder_path.
    Returns list of (arr_float32_224x224, label_idx, source_tag, filename).
    Skips unreadable files silently.
    """
    records = []
    if not os.path.isdir(folder_path):
        print(f"  ⚠️  Missing folder: {folder_path}")
        return records
    exts = ('.bmp', '.png', '.jpg', '.jpeg')
    files = sorted([f for f in os.listdir(folder_path)
                    if f.lower().endswith(exts)])
    for fname in files:
        arr = preprocess_single_image(os.path.join(folder_path, fname))
        if arr is not None:
            records.append((arr, label_idx, source_tag, fname))
    return records


def load_all_data(verbose=True):
    """
    Load FYP dataset + Kaggle dataset.
    Returns dict: { class_label: [ (arr, label_idx, source, fname), ... ] }
    """
    data_by_class = defaultdict(list)

    if verbose:
        print("📂 Loading developed dataset (SecuGen)...")
    for cls in CLASSES:
        folder = os.path.join(DEVELOPED_DATA_PATH, cls)
        recs   = _load_folder(folder, CLASSES.index(cls), "developed")
        data_by_class[cls].extend(recs)
        if verbose:
            print(f"   {cls:4s}: {len(recs):4d} images  [{folder}]")

    if verbose:
        print("\n📂 Loading Kaggle data...")
    for kaggle_folder, target_cls in KAGGLE_CLASSES.items():
        if target_cls is None:
            continue
        folder = os.path.join(KAGGLE_DATA_PATH, kaggle_folder)
        recs   = _load_folder(folder, CLASSES.index(target_cls), "kaggle")
        data_by_class[target_cls].extend(recs)
        if verbose:
            print(f"   {kaggle_folder} → {target_cls}: {len(recs):4d} images")

    if verbose:
        print(f"\n📊 Total per class:")
        grand_total = 0
        for cls in CLASSES:
            n = len(data_by_class[cls])
            grand_total += n
            print(f"   {cls:4s}: {n:5d}")
        print(f"   {'TOTAL':4s}: {grand_total:5d}")

    return data_by_class


# ── Person-aware split ────────────────────────────────────────────────────────

def _person_aware_split(records, cls_name):
    """
    Groups images into persons (every PRINTS_PER_PERSON images = 1 person),
    then splits persons 70/15/15 into train/val/test.
    This prevents the same person's fingerprints appearing in multiple splits.
    """
    random.seed(SEED)
    n = len(records)

    # Assign each image to a person group
    person_groups = defaultdict(list)
    for i, rec in enumerate(records):
        pid = f"{cls_name}_P{i // PRINTS_PER_PERSON:04d}"
        person_groups[pid].append(rec)

    persons = list(person_groups.keys())
    random.shuffle(persons)

    n_p     = len(persons)
    n_train = max(1, int(TRAIN_RATIO * n_p))
    n_val   = max(1, int(VAL_RATIO   * n_p))
    # Guarantee at least 1 person per split
    if n_train + n_val >= n_p:
        n_val   = max(1, n_p - n_train - 1)
    n_test = n_p - n_train - n_val

    train_p = persons[:n_train]
    val_p   = persons[n_train:n_train + n_val]
    test_p  = persons[n_train + n_val:]

    train_recs = [r for p in train_p for r in person_groups[p]]
    val_recs   = [r for p in val_p   for r in person_groups[p]]
    test_recs  = [r for p in test_p  for r in person_groups[p]]

    return train_recs, val_recs, test_recs, \
           (len(train_p), len(val_p), len(test_p))


# ── Augment training set ──────────────────────────────────────────────────────

def _augment_train(train_recs, aug_fraction=0.40):
    """
    Adds augmented copies to training records.
    aug_fraction=0.40 → add 40% extra samples (randomly sampled from train set).
    Augmented images carry the same label as their source.
    """
    random.seed(SEED)
    n_aug   = int(len(train_recs) * aug_fraction)
    sources = random.choices(train_recs, k=n_aug)
    extra   = []
    for (arr, lbl, src, fname) in sources:
        versions = augment_image(arr)
        chosen   = random.choice(versions[1:])   # skip index 0 = original
        extra.append((chosen, lbl, src + "_aug", fname))
    return train_recs + extra


# ── Public API ────────────────────────────────────────────────────────────────

def build_datasets(aug_fraction=0.40, verbose=True):
    """
    Full pipeline: load → split → augment → return numpy arrays.

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test  (all numpy)
        y_*_cat   : one-hot versions
        split_info: dict with counts
    """
    data_by_class = load_all_data(verbose=verbose)

    all_train, all_val, all_test = [], [], []
    person_splits = {}

    for cls in CLASSES:
        recs = data_by_class[cls]
        random.seed(SEED)
        random.shuffle(recs)
        tr, vl, te, p_counts = _person_aware_split(recs, cls)
        person_splits[cls] = p_counts

        # Augment only training records
        tr_aug = _augment_train(tr, aug_fraction)
        all_train.extend(tr_aug)
        all_val.extend(vl)
        all_test.extend(te)

    # Shuffle
    random.seed(SEED)
    random.shuffle(all_train)

    H, W = INPUT_SIZE

    def to_arrays(recs):
        X = np.array([r[0].reshape(H, W, 1) for r in recs], dtype=np.float32)
        y = np.array([r[1] for r in recs],                  dtype=np.int32)
        return X, y

    X_tr,  y_tr  = to_arrays(all_train)
    X_val, y_val = to_arrays(all_val)
    X_te,  y_te  = to_arrays(all_test)

    y_tr_cat  = _to_categorical(y_tr,  NUM_CLASSES)
    y_val_cat = _to_categorical(y_val, NUM_CLASSES)
    y_te_cat  = _to_categorical(y_te,  NUM_CLASSES)

    if verbose:
        aug_pct = (len(all_train) - sum(len(data_by_class[c]) * TRAIN_RATIO
                                        for c in CLASSES)) / len(all_train) * 100
        print(f"\n📊 Final dataset split:")
        print(f"   Train : {len(X_tr):5d}  (includes ~{aug_fraction*100:.0f}% augmented)")
        print(f"   Val   : {len(X_val):5d}  (original only)")
        print(f"   Test  : {len(X_te):5d}  (original only)")
        print(f"\n   Person-level splits:")
        for cls in CLASSES:
            tr_p, vl_p, te_p = person_splits[cls]
            print(f"   {cls:4s}: {tr_p} train / {vl_p} val / {te_p} test persons")
        print(f"\n   Pixel range → min:{X_tr.min():.3f}  max:{X_tr.max():.3f}")
        print(f"   Input shape : {X_tr.shape[1:]}")

    split_info = {
        "n_train": len(X_tr), "n_val": len(X_val), "n_test": len(X_te),
        "person_splits": person_splits
    }

    return (X_tr, y_tr_cat, X_val, y_val_cat, X_te, y_te_cat, split_info)


if __name__ == "__main__":
    build_datasets(verbose=True)
