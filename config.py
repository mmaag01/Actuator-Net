from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_PATH      = PROJECT_ROOT / "Data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR    = PROJECT_ROOT / "results"

# ── Data categories ──────────────────────────────────────────────────────────
USE_MAIN    = True
USE_CRASHES = False
USE_OTHER   = False

USE_PLC             = True
USE_PMS             = True
USE_TMS             = True
USE_tStep           = True
USE_tStep_CmdErrs   = False
USE_Misc            = False
ZERO_OFFSET         = False

UNITS={'time':'s', 'torque':'Nm', 'velocity':'rad/s', 
               'position':'rad', 'current':'A'}

# ── Features ─────────────────────────────────────────────────────────────────
# V1: torDes, posDes, velDes, posErr, velErr, posAct, velAct, accelAct, i, torEst
# V3: torDes, posDes, velDes, posErr, velErr, posAct, velAct, accelAct
FEATURE_COLS = [
    'torDes', 'posDes', 'velDes', 'posAct', 'posErr', 'velErr',
    'velAct',  'accelAct', 'i', 'torEst',
    'torKdEst', 'kd', 'i2t', 't',
]
TARGET_COL  = 'torAct'
INCLUDE_I2T      = True   # i2t (thermal accumulation proxy)
INCLUDE_curr     = True
INCLUDE_torKdEst = True   # Kd contribution to torEst decomposition
INCLUDE_torEst   = True
INCLUDE_kd       = True   # derivative gain value
INCLUDE_posDes   = True   # desired position (always on in v1)
INCLUDE_accelAct = True   # actual acceleration (always on in v1)
INCLUDE_t        = True   # timestamp — adds absolute-time signal

N_FEATURES  = (len(FEATURE_COLS)
               - int(not INCLUDE_I2T)
               - int(not INCLUDE_curr)
               - int(not INCLUDE_torKdEst)
               - int(not INCLUDE_torEst)
               - int(not INCLUDE_kd)
               - int(not INCLUDE_posDes)
               - int(not INCLUDE_accelAct)
               - int(not INCLUDE_t))

# ── Feature Scaling ───────────────────────────────────────────────────────────
SCALER_TYPE = 'standard'  # Options: 'standard', 'minmax', 'robust', 'quantile', 'power', 'polynomial'

# Polynomial feature expansion (used when SCALER_TYPE = 'polynomial')
POLY_DEGREE = 2                # Degree of polynomial features (2 = quadratic)
POLY_INTERACTION_ONLY = False  # If True, only interaction terms (no x^2, x^3, etc.)
POLY_INCLUDE_BIAS = False      # Include bias column (constant term)

# Scaler-specific parameters (advanced)
SCALER_PARAMS = {
    'standard': {},
    'minmax': {'feature_range': (0, 1)},
    'robust': {'quantile_range': (25.0, 75.0)},  # Less sensitive to outliers
    'quantile': {'n_quantiles': 1000, 'output_distribution': 'uniform'},
    'power': {'method': 'yeo-johnson', 'standardize': True},  # Gaussianize features
}

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
RESIDUAL_TARGET  = False  # predict torAct-torEst; reconstruct at inference with +torEst

# ── Data split ────────────────────────────────────────────────────────────────
FIT_NEW_SCALERS = True
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 1 - TRAIN_RATIO - VAL_RATIO = 0.15  (implied)

# ── Acceleration smoothing ────────────────────────────────────────────────────
SMOOTH_ACCEL = True   # apply Savitzky-Golay smoothing to accelAct before windowing
SG_WINDOW    = 21     # must be odd; matches Test 3 analysis
SG_POLYORDER = 3

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42