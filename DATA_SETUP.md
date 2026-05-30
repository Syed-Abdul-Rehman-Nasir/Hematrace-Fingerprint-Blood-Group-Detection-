# Local data setup (not in Git)

Git **never** stores fingerprint datasets. `.gitignore` blocks all dataset folders, `.bmp` images, archives (`.zip`, etc.), and the research notebook at the repo root.

**Before every push:** run `git status` and confirm you do **not** see `blood group positives/`, `Kaggle dataset/`, `unseen dataset/`, `Zips/`, or thousands of `.bmp` files. If any appear as “to be committed”, run `git reset` and do not use `git add -A` blindly.

After cloning, create this layout under your project root:

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
