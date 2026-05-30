# HemaTrace — Project Documentation

**Single reference** for project structure, datasets, PyTorch ML backend, Electron desktop app, configuration, evaluation metrics, UI behaviour, and known limitations.

**Project root:** auto-detected as the parent of `backend/` (see `backend/config.py` → `ROOT`). Override with env var `HEMATRACE_ROOT`.

>

## Table of Contents

1. [High-level overview](#1-high-level-overview)
2. [Directory structure](#2-directory-structure)
3. [Data sources](#3-data-sources)
4. [Preprocessing pipeline](#4-preprocessing-pipeline)
5. [Model & training](#5-model--training)
6. [Flask API server](#6-flask-api-server)
7. [Electron desktop app](#7-electron-desktop-app)
8. [How to run](#8-how-to-run)
9. [Environment variables](#9-environment-variables)
10. [Artifacts on disk](#10-artifacts-on-disk)
11. [Limitations & known issues](#11-limitations--known-issues)
12. [Troubleshooting](#12-troubleshooting)
13. [Changelog](#13-changelog)
14. [Publishing to GitHub](#14-publishing-to-github)

---

## 1. High-level overview

| Layer | Role |
|-------|------|
| **Python backend** (`backend/`) | Load fingerprint images; train a **4-class** Rh-positive classifier (**A+, B+, AB+, O+**); serve Flask HTTP API; save PyTorch checkpoints (`.pt`) |
| **HemaTrace** (`HemaTrace/`) | Electron desktop app — staff login, patient test workflow, simulated scan UI, predictions via IPC → HTTP; spawns `server.py` on launch |
| **Research notebook** | `BloodGroup_Complete_Final_*.ipynb` — original TensorFlow/Colab pipeline, **not** used at runtime |
| **Datasets** | SecuGen captures under `blood group positives\`; Kaggle Rh± under `Kaggle dataset\` (only Rh+ folders used for training) |

**Scope:** Research / hospital-style prototype — not validated for clinical use. The deployed model predicts four Rh-positive groups only.

```
Electron HemaTrace
  renderer/index.html
        │  IPC predictBloodGroup
  preload.js
        │
   main.js ──── POST /predict ──→ Flask server.py :5000
                                        │
                                   predict.py
                                        │
                               hemtrace_best.pt
```

---

## 2. Directory structure

```
d:\sara pro\
├── .venv\                             # Python virtualenv (torch, flask, opencv …)
├── requirements.txt
├── README.md
├── PROJECT_DOCUMENTATION.md
│
├── backend\
│   ├── config.py                      # ROOT, paths, hyperparameters, Flask host/port
│   ├── preprocess.py                  # CLAHE, Gabor, 224×224, augmentation
│   ├── dataset.py                     # Developed + Kaggle loader, person-aware split
│   ├── model.py                       # HemaTraceNet (EfficientNet-B0)
│   ├── train.py                       # Two-phase training + test evaluation
│   ├── predict.py                     # TTA inference (CLI + importable library)
│   ├── server.py                      # Flask: /health, /predict
│   ├── evaluate_unseen.py             # Eval on unseen dataset\ per folder
│   ├── models\
│   │   ├── hemtrace_best.pt           # Active checkpoint (~17 MB)
│   │   └── hemtrace_best.keras        # Legacy TF (optional, unused)
│   └── results\
│       ├── classification_report.txt
│       ├── training_history.png
│       ├── confusion_matrix.png
│       └── unseen_evaluation.txt
│
├── HemaTrace\
│   ├── main.js
│   ├── preload.js
│   ├── package.json
│   ├── renderer\index.html
│   └── assets\icons\
│
├── blood group positives\developed dataset\   # Primary SecuGen data
├── Kaggle dataset\dataset_blood_group\
└── unseen dataset\                             # Hold-out eval
```

---

## 3. Data sources

### 3.1 Primary — developed dataset

Path: `DEVELOPED_DATA_PATH` in `config.py` (`blood group positives\developed dataset\`).

| Subfolder | Format | Approx. count |
|-----------|--------|---------------|
| `A+`  | `.bmp` | 401 |
| `B+`  | `.bmp` | 299 |
| `AB+` | `.bmp` | 287 |
| `O+`  | `.bmp` | 427 |
| **Total** | | **1,414** |

### 3.2 Secondary — Kaggle dataset

| Folder | Approx. files | Used in training |
|--------|---------------|------------------|
| `A+`, `B+`, `AB+`, `O+` | 565 / 652 / 708 / 852 | **Yes** |
| `A-`, `B-`, `AB-`, `O-` | 1009 / 741 / 761 / 712 | **No** (`KAGGLE_CLASSES → None`) |

**Combined Rh+ (developed + Kaggle):** ~4,191 images before split. Run `python dataset.py` for exact counts.

### 3.3 Loader & split details

| Setting | Value |
|---------|-------|
| Accepted extensions | `.bmp`, `.png`, `.jpg`, `.jpeg` (case-insensitive) |
| Failed images | Skipped silently |
| Person assumption | `PRINTS_PER_PERSON = 10` — 10 consecutive sorted files = one person |
| Split ratio | **70% train / 15% val / 15% test** (by person count, per class) |
| Train augmentation | `aug_fraction = 0.40` extra augmented copies |

---

## 4. Preprocessing pipeline

All images pass through `preprocess_single_image()` in `preprocess.py`:

1. **Load** grayscale (from path, ndarray, or PIL image)
2. **Portrait fix** — rotate if width > height
3. **CLAHE** — clip limit 2.0, tile grid 8×8
4. **Gabor filter bank** — 8 orientations (0°–157.5°), max response per pixel
5. **Gaussian blur** — 3×3 kernel
6. **Resize** to 224×224 (Lanczos), normalize to [0, 1]
7. **Augmentation** (train & TTA only) — ±10° rotation, brightness jitter, ±5% zoom; no horizontal flip

---

## 5. Model & training

### 5.1 Architecture (`model.py`)

**HemaTraceNet** wraps EfficientNet-B0 for single-channel fingerprint input:

| Component | Detail |
|-----------|--------|
| Input adapter | `Conv2d(1→3, 1×1)` + `BatchNorm2d` |
| Backbone | `efficientnet_b0` feature extractor (ImageNet pretrained) |
| Pooling | Global average pooling |
| Head | BN → Linear(256) → Dropout(0.5) → Linear(128) → Dropout(0.4) → 4 logits |

### 5.2 Training strategy (`train.py`)

| Phase | Frozen layers | LR | Max epochs | Patience |
|-------|--------------|-----|-----------|---------|
| Phase 1 | Full backbone | 1e-3 | 20 | 10 |
| Phase 2 | All except last 2 blocks | 1e-5 | 80 | 15 |

| Setting | Value |
|---------|-------|
| Loss | `CrossEntropyLoss` with class weights, label smoothing 0.1 |
| Optimizer | AdamW (`weight_decay=1e-4`) |
| Scheduler | `ReduceLROnPlateau` on validation loss |
| Batch size | 32 |
| Requires GPU | Yes (CUDA); override with `HEMATRACE_ALLOW_CPU=1` |

### 5.3 Hyperparameters (`config.py`)

| Symbol | Value |
|--------|-------|
| `CLASSES` | `['A+', 'B+', 'AB+', 'O+']` |
| `INPUT_SIZE` | `(224, 224)` |
| `BATCH_SIZE` | 32 |
| `SEED` | 42 |
| `TRAIN_RATIO` / `VAL_RATIO` | 0.70 / 0.15 |
| `SERVER_HOST` / `SERVER_PORT` | `127.0.0.1` / `5000` |

### 5.4 Inference (`predict.py`)

- **TTA:** 10 augmented views, mean softmax → class + confidence
- **Response fields:** `predicted_class`, `confidence` (0–100), `all_scores`, `error` (`null` on success)
- **Entry points:** `predict_from_path(path)`, `predict_from_bytes(bytes)`, or CLI: `python predict.py <image>`

### 5.5 Model performance (current checkpoint)

**Test set — 659 held-out samples:**

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| A+    | 0.49 | 0.50 | 0.49 |
| B+    | 0.59 | 0.58 | 0.59 |
| AB+   | 0.58 | 0.62 | 0.60 |
| O+    | 0.66 | 0.63 | 0.64 |
| **Overall** | | | **58.42% accuracy** |

**Unseen folder eval — 10 O+ samples (different capture conditions):**

| Metric | Value |
|--------|-------|
| Accuracy | 30% (3/10) |
| Note | Low confidence (~34–65%); likely domain shift vs. training mix |

Re-run unseen evaluation:

```powershell
cd "d:\sara pro\backend"
& "..\.venv\Scripts\python.exe" evaluate_unseen.py
```

> Remember to update §5.5 and §10 after each training run.

---

## 6. Flask API server

Bound to `127.0.0.1:5000` (loopback only). CORS allows `localhost`, `null`, and `file://` (Electron).

| Endpoint | Method | Body | Response |
|----------|--------|------|----------|
| `/health` | GET | — | `{ "status": "ok", "model_loaded": true }` |
| `/predict` | POST | `multipart/form-data`, field **`image`** | Prediction JSON; HTTP 400/500 on errors |

**Startup:** `load_model()` runs in `__main__` — if `.pt` is missing, the process exits immediately.

> Note: `model_loaded` is static `true` and does not reflect runtime load failures.

---

## 7. Electron desktop app

### 7.1 Runtime flow

1. `main.js` spawns `.venv\Scripts\python.exe backend\server.py`
2. Polls `GET /health` (up to 30 × 1 s)
3. Opens frameless window → `renderer/index.html`
4. Staff workflow: **login → dashboard → new test → scan → result**
5. **Scan:** ~3.5 s UI animation; `showResult()` sends image bytes via IPC → main process → POST `/predict`
6. **Image source:** file picker (`.bmp`/`.png`); `window._lastFingerprintBytes` reserved for future SDK integration
7. **Records:** in-memory only (lost on restart)

### 7.2 Security model

| Setting | Value |
|---------|-------|
| `contextIsolation` | `true` |
| `nodeIntegration` | `false` |
| API exposure | Loopback only |
| Login | Local/demo — name + role, no password |

### 7.3 IPC API (`preload.js`)

```javascript
window.electronAPI = {
  minimize(),
  maximize(),
  close(),
  predictBloodGroup(imageArrayBuffer)  // → POST /predict → prediction JSON
}
```

### 7.4 UI notes

| Area | Current state |
|------|--------------|
| Branding | "Non-Invasive Blood Group Detection System"; no university/FYP text |
| Login | "Staff Sign In"; placeholder name "Abdul Rehman" |
| About | Version 1.0.0; no development team or advisor section |
| Sensor | "SecuGen Hamster U20" status is **simulated** (no physical SDK wired) |
| Scope label | About page states "4 Rh-positive groups (A+, B+, AB+, O+)" |

---

## 8. How to run

### 8.1 First-time setup

```powershell
cd "d:\HemaTrace"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt

cd HemaTrace
npm install
```

Ensure `backend\models\hemtrace_best.pt` exists (included in repo, or train to regenerate).

**GPU support (training):**

```powershell
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available())"
```

### 8.2 Option A — Desktop app (backend auto-started)

```powershell
cd "d:\HemaTrace"
npm start
```

### 8.3 Option B — Separate terminals

**Terminal 1 — backend:**

```powershell
cd "d:\backend"
& "..\.venv\Scripts\python.exe" server.py
```

**Terminal 2 — frontend:**

```powershell
cd "d:\HemaTrace"
npm start
```

> `main.js` still spawns a second Python process on launch. If port 5000 is already taken, the child fails silently while Terminal 1 handles requests. To use a single backend, disable `startPythonServer()` in `main.js`.

**Health check:** `curl http://127.0.0.1:5000/health`

### 8.4 Common backend commands

| Task | Command (from `backend\`) |
|------|--------------------------|
| Train | `python train.py` |
| Dataset smoke test | `python dataset.py` |
| Model smoke test | `python model.py` |
| CLI predict | `python predict.py path\to\image.bmp` |
| Unseen evaluation | `python evaluate_unseen.py` |
| Unbuffered training logs | `$env:PYTHONUNBUFFERED="1"; python train.py` |
| CPU-only training | `$env:HEMATRACE_ALLOW_CPU="1"; python train.py` |

> **PowerShell tip:** Use `& "..\.venv\Scripts\python.exe" script.py` — the `&` call operator is required when executing from a quoted path.

---

## 9. Environment variables

| Variable | Effect |
|----------|--------|
| `HEMATRACE_ROOT` | Override auto-detected project root |
| `HEMATRACE_ALLOW_CPU` | `1` / `true` / `yes` → allow training without CUDA |
| `PYTHONUNBUFFERED` | `1` → line-buffered stdout during training |
| `PYTHONUTF8` | `1` → UTF-8 console on Windows |

---

## 10. Artifacts on disk

| Path | Notes |
|------|-------|
| `backend/models/hemtrace_best.pt` | **Required for inference** (~17 MB) |
| `backend/models/hemtrace_best.keras` | Legacy TF — safe to delete |
| `backend/results/classification_report.txt` | Test metrics (58.42% accuracy) |
| `backend/results/training_history.png` | Phase 1 + 2 loss/accuracy curves |
| `backend/results/confusion_matrix.png` | Test confusion matrix |
| `backend/results/unseen_evaluation.txt` | Unseen folder run log |

---

## 11. Limitations & known issues

| Topic | Detail |
|-------|--------|
| **Classes** | 4 Rh+ groups only; Rh− Kaggle folders excluded |
| **Accuracy** | ~58% test; ~30% unseen — not clinical-grade |
| **Person split** | Assumes 10 consecutive sorted files = one person; incorrect ordering → leakage |
| **Domain shift** | Mixing SecuGen + Kaggle sources hurts generalization on new captures |
| **Sensor** | SecuGen connection is UI simulation; inference uses file picker |
| **Persistence** | Patient records are in-memory in Electron only |
| **Flask** | Development server; single-process, not production WSGI |
| **`/health`** | `model_loaded` is static `true`; does not verify actual load |
| **Notebook** | TensorFlow/MobileNetV2 research path — not the deployment source |

---

## 12. Troubleshooting

| Problem | Fix |
|---------|-----|
| `The module '.venv' could not be loaded` | Venv is at project root, not `backend\`. Use `& "..\.venv\Scripts\python.exe"` from `backend\`. |
| `Unexpected token 'server.py'` | Missing `&` before quoted Python path in PowerShell. |
| Electron opens but predict fails | Start backend first; verify `hemtrace_best.pt` exists and `GET /health` returns 200. |
| CUDA OOM on GTX 1650 | Lower `BATCH_SIZE` in `config.py` to 16 or 8. |
| Training refuses CPU | Set `$env:HEMATRACE_ALLOW_CPU="1"`. |

---

## 13. Changelog

| Phase | Change |
|-------|--------|
| Original | Jupyter + TensorFlow/Keras notebook |
| Backend v1 | `config`, `preprocess`, `dataset`, `model`, `train`, `predict`, Flask `server` |
| Electron v1 | IPC + subprocess + `/predict` |
| Stack migration | TensorFlow → PyTorch (`.pt`); `requirements.txt` + CUDA wheels |
| Eval tooling | `evaluate_unseen.py`, `unseen_evaluation.txt` |
| UI cleanup (2026) | Removed university/FYP branding; login placeholder "Abdul Rehman" |
| Docs (2026) | Full refresh: run commands, metrics, UI state, separate terminals, troubleshooting |
| GitHub prep (2026) | `.gitignore`, portable `ROOT`, editor checkpoints excluded, `README.md`, `DATA_SETUP.md` |

---

