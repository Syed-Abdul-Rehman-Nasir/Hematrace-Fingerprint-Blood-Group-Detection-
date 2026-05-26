# HemaTrace 

AI-based **fingerprint blood group classification** (4 Rh-positive classes: **A+, B+, AB+, O+**) with a **PyTorch** backend and **Electron** desktop UI.

> **Datasets are not in this repository.** You must place training data locally (see [DATA_SETUP.md](DATA_SETUP.md)). Image files are blocked by `.gitignore`.

## Repository layout

| Path | Description |
|------|-------------|
| [`backend/`](backend/) | ML pipeline, Flask API, training & inference |
| [`HemaTrace/`](HemaTrace/) | Electron desktop application |
| [`requirements.txt`](requirements.txt) | Python dependencies |
| [`PROJECT_DOCUMENTATION.md`](PROJECT_DOCUMENTATION.md) | Full technical reference |

## Quick start (backend)

```powershell
cd path\to\repo
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place [`hemtrace_best.pt`](backend/models/hemtrace_best.pt) in `backend/models/` (included in repo if present) **or** train:

```powershell
cd backend
python train.py
```

Run the API:

```powershell
cd backend
python server.py
```

Health: `http://127.0.0.1:5000/health`

## Quick start (desktop app)

```powershell
cd HemaTrace
npm install
npm start
```

## GPU training

Install CUDA-enabled PyTorch (see [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) §8.2). CPU-only: `$env:HEMATRACE_ALLOW_CPU="1"`.

## License & use

Research / prototype — **not for clinical diagnosis**.
