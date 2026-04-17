from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_PATH      = PROJECT_ROOT / "Data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR    = PROJECT_ROOT / "results"

# ── Data categories ──────────────────────────────────────────────────────────
USE_MAIN    = True
USE_CRASHES = False
USE_OTHER   = False

# ── Features ─────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    'torDes', 'posDes', 'velDes',
    'posAct', 'velAct', 'accelAct',
    'i', 'torEst', 'posErr', 'velErr',
]
INCLUDE_I2T = False   # set True to add i2t as an additional input feature
TARGET_COL  = 'torAct'
N_FEATURES  = len(FEATURE_COLS) + int(INCLUDE_I2T)

# ── Sequence ──────────────────────────────────────────────────────────────────
SEQ_LEN = 30          # window / history length (timesteps)

# ── MLP hyperparameters ───────────────────────────────────────────────────────
MLP_HIDDEN_SIZE = 32
MLP_N_LAYERS    = 3

# ── GRU hyperparameters ───────────────────────────────────────────────────────
GRU_HIDDEN_SIZE = 64
GRU_N_LAYERS    = 4
GRU_DROPOUT     = 0.0

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE       = 64
LR               = 1e-4
ONE_CYCLE_MAX_LR = 1e-3   # peak LR for OneCycleLR
MAX_EPOCHS       = 200
PATIENCE         = 20     # early-stopping patience (epochs)
GRAD_CLIP_NORM   = 1.0
WEIGHT_DECAY     = 0.0

# ── Data split ────────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 1 - TRAIN_RATIO - VAL_RATIO  (implied)

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
