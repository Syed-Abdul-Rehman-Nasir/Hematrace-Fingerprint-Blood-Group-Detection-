# HemaTrace Backend

PyTorch + Flask service for fingerprint → blood group (A+, B+, AB+, O+).

## Modules

| File | Role |
|------|------|
| `config.py` | Paths, hyperparameters, server bind |
| `preprocess.py` | CLAHE, Gabor, 224×224, augmentation |
| `dataset.py` | Load developed dataset + Kaggle, person-aware split |
| `model.py` | `HemaTraceNet` (EfficientNet-B0) |
| `train.py` | Two-phase training |
| `predict.py` | TTA inference |
| `server.py` | `GET /health`, `POST /predict` |
| `evaluate_unseen.py` | Eval on `unseen dataset/` |

## API

**POST** `/predict` — `multipart/form-data`, field `image`

```json
{
  "predicted_class": "O+",
  "confidence": 72.5,
  "all_scores": { "A+": 5.1, "B+": 12.3, "AB+": 10.1, "O+": 72.5 },
  "error": null
}
```

## Checkpoint

- **Active:** `models/hemtrace_best.pt` (required for `server.py`)
- **Legacy:** `*.keras` is ignored by Git; do not use with this stack

## Commands

```powershell
python dataset.py          # smoke load
python train.py            # train (GPU recommended)
python predict.py img.bmp  # CLI predict
python server.py           # Flask on 127.0.0.1:5000
python evaluate_unseen.py
```

See [../PROJECT_DOCUMENTATION.md](../PROJECT_DOCUMENTATION.md) for metrics and details.
