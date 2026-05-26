# Local data setup (not in Git)

Git **never** stores fingerprint datasets. After cloning, create this layout under your project root:

```
<project-root>/
├── blood group positives/
│   └── developed dataset/
│       ├── A+/
│       ├── B+/
│       ├── AB+/
│       └── O+/
├── Kaggle dataset/
│   └── dataset_blood_group/
│       ├── A+/
│       ├── B+/
│       ├── AB+/
│       └── O+/
└── unseen dataset/          # optional, for evaluate_unseen.py
    └── O+/
        └── *.bmp
```

Rh-negative Kaggle folders (`A-`, `B-`, …) may exist on disk but are **not used** for the 4-class model.

Verify loading:

```powershell
cd backend
python dataset.py
```

Paths are configured in [`backend/config.py`](backend/config.py) (`DEVELOPED_DATA_PATH`, `KAGGLE_DATA_PATH`).
