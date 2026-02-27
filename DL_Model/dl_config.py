"""
dl_config.py
============
Central configuration for the Deep Learning Pothole Detection system.

All hyperparameters, paths, and constants live here so that every
other module imports from one place and stays in sync.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR    = os.path.join(BASE_DIR, "..", "dataset")
MODELS_DIR     = os.path.join(BASE_DIR, "models")
LOGS_DIR       = os.path.join(BASE_DIR, "logs")
EXPORTS_DIR    = os.path.join(BASE_DIR, "exports")

MODEL_SAVE_PATH     = os.path.join(MODELS_DIR, "pothole_dl_model.keras")
SCALER_SAVE_PATH    = os.path.join(MODELS_DIR, "dl_scaler.pkl")
HISTORY_SAVE_PATH   = os.path.join(LOGS_DIR,   "training_history.json")
REPORT_SAVE_PATH    = os.path.join(LOGS_DIR,   "classification_report.txt")
CONFUSION_SAVE_PATH = os.path.join(LOGS_DIR,   "confusion_matrix.png")

# ── Sensor / Signal constants ─────────────────────────────────────────────────
BASELINE_CM      = 1000.0   # Fixed sensor-to-road distance (cm)
BASELINE_MIN_CM  =   50.0   # Minimum baseline in synthetic data
BASELINE_MAX_CM  = 1000.0   # Maximum baseline in synthetic data
POTHOLE_THRESH   =    3.0   # Min +deviation to call a pothole (cm)
BUMP_THRESH      =    3.0   # Min -deviation to call a speed bump (cm)
NOISE_STD        =    1.5   # Sensor noise std at close range (cm)

# ── Adaptive Baseline constants ───────────────────────────────────────────────
# Moving-average window (readings).  20–50 is the recommended range.
# Larger = smoother baseline but slower to track real road-slope changes.
ADAPT_MA_WINDOW      = 30

# IIR high-pass filter coefficient  α ∈ (0,1).
# y[n] = α * (y[n-1] + x[n] - x[n-1])   (removes slow drift / road slope)
# 0.95 → passes events faster than ~3 s at 10 Hz
ADAPT_HP_ALPHA       = 0.95

# Derivative (velocity) thresholds (cm / reading).
# Positive vel = sensor reading increases (distance grows → pothole).
# We look for: big_pos_vel followed by big_neg_vel.
ADAPT_VEL_UP_THRESH  =  2.0   # cm/reading → start of a pothole
ADAPT_VEL_DOWN_THRESH= -2.0   # cm/reading → end of a pothole

# Depth-duration guard: how many *consecutive* readings the deviation must
# stay above POTHOLE_THRESH before we confirm a pothole.
# At 10 Hz: 3 → 300 ms, 5 → 500 ms  (rejects single-spike noise)
ADAPT_MIN_DURATION   =  3     # readings

# Feature vector produced by AdaptiveBaseline  → feed into LSTM model
# [dist, strength, ma_dev, hp_signal, velocity, above_thresh_flag]
ADAPT_N_FEATURES     =  6

# ── Window / Sequence ─────────────────────────────────────────────────────────
WINDOW_SIZE      = 30       # Raw readings per inference window  (≥20 for LSTM)
STRIDE           =  5       # Sliding window stride when building dataset
N_RAW_FEATURES   =  3       # distance_cm, strength, deviation_cm  (per timestep)
# When the adaptive pipeline is used, the feature tensor is wider:
# N_RAW_FEATURES → ADAPT_N_FEATURES (set via build_model(n_features=ADAPT_N_FEATURES))

# ── Classes ───────────────────────────────────────────────────────────────────
CLASS_NAMES = {
    0: "Flat Road",
    1: "Shallow Pothole",
    2: "Deep Pothole",
    3: "Speed Bump",
}
N_CLASSES = len(CLASS_NAMES)

# Severity thresholds (depth in cm)
SEVERITY_LEVELS = {
    "Noise"          : (0,   2),
    "Shallow"        : (2,   8),
    "Moderate"       : (8,  20),
    "Deep/Dangerous" : (20, 1e6),
}

# ── Training Hyperparameters ──────────────────────────────────────────────────
N_SYNTHETIC_SAMPLES = 20_000   # Synthetic rows per class (×4 = total)
REAL_AUGMENT_FACTOR =     50   # Each real row is augmented × this many times

BATCH_SIZE   =  64
EPOCHS       = 120
LEARNING_RATE = 1e-3
LR_PATIENCE  =  12    # ReduceLROnPlateau patience
ES_PATIENCE  =  25    # EarlyStopping patience
VALIDATION_SPLIT = 0.15
TEST_SPLIT       = 0.15

# ── Architecture ──────────────────────────────────────────────────────────────
#  CNN block
CNN_FILTERS    = [64, 128, 256]   # per conv block
CNN_KERNEL     = 3
POOL_SIZE      = 2
DROPOUT_CNN    = 0.25

#  BiLSTM block
LSTM_UNITS     = [128, 64]
DROPOUT_LSTM   = 0.30

#  Attention
ATTENTION_HEADS   = 4
ATTENTION_KEY_DIM = 32

#  Dense head
DENSE_UNITS    = [128, 64]
DROPOUT_DENSE  = 0.40

# ── Real-time detection ───────────────────────────────────────────────────────
RT_CONFIDENCE_THRESHOLD = 0.55   # Min softmax confidence to accept prediction
RT_ALERT_DEPTH_CM       = 5.0    # Alert threshold for pothole depth
RT_WINDOW_BUFFER        = WINDOW_SIZE  # readings to buffer before first pred.

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42
