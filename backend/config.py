# backend/config.py
import os

# ── Project root ──────────────────────────────────────────────
# Parent of backend/ (portable for any clone path). Override with HEMATRACE_ROOT.
ROOT = os.environ.get(
    "HEMATRACE_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)

# ── Dataset paths ─────────────────────────────────────────────
# Primary: real SecuGen sensor data (self-collected)
DEVELOPED_DATA_PATH = os.path.join(ROOT, "blood group positives", "developed dataset")
# Backward-compatible alias (deprecated name: FYP +)
FYP_DATA_PATH = DEVELOPED_DATA_PATH

# Secondary: Kaggle fingerprint dataset
KAGGLE_DATA_PATH = os.path.join(ROOT, "Kaggle dataset", "dataset_blood_group")

# Map Kaggle subfolder names → our 4 class labels
# Fill this in based on actual subfolder names found on disk:
KAGGLE_CLASSES = {
    "A+": "A+",
    "B+": "B+",
    "AB+": "AB+",
    "O+": "O+",
    # Rh-negative folders — not used in 4-class (A+/B+/AB+/O+) training; exclude or merge separately
    "A-": None,
    "B-": None,
    "AB-": None,
    "O-": None,
}

# ── Model & output paths ───────────────────────────────────────
MODELS_DIR   = os.path.join(ROOT, "backend", "models")
RESULTS_DIR  = os.path.join(ROOT, "backend", "results")
DATA_DIR     = os.path.join(ROOT, "backend", "data")
DB_PATH      = os.path.join(DATA_DIR, "hematrace.db")
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR,    exist_ok=True)

MODEL_PATH   = os.path.join(MODELS_DIR, "hemtrace_best.pt")
HISTORY_PATH = os.path.join(RESULTS_DIR, "training_history.png")
CM_PATH      = os.path.join(RESULTS_DIR, "confusion_matrix.png")

# ── Image config ───────────────────────────────────────────────
CLASSES       = ['A+', 'B+', 'AB+', 'O+']
NUM_CLASSES   = 4
INPUT_SIZE    = (224, 224)   # larger than before — preserves ridge detail
PRINTS_PER_PERSON = 10       # fingerprints per person in the dataset

# ── Training config ────────────────────────────────────────────
BATCH_SIZE    = 32
EPOCHS        = 80
LR_PHASE1     = 1e-3   # train head only (base frozen)
LR_PHASE2     = 1e-5   # fine-tune top layers (base partially unfrozen)
SEED          = 42

# ── Split ratios ───────────────────────────────────────────────
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15   # 70/15/15 person-aware split
# TEST_RATIO  = 0.15   (remainder)

# ── Flask server ───────────────────────────────────────────────
SERVER_HOST   = "127.0.0.1"
SERVER_PORT   = 5000
