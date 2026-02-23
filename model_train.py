"""
model_train.py
==============
Trains a pothole / road-anomaly classifier based on BASELINE-DEVIATION features.

Concept (as the user defined):
─────────────────────────────
  The LiDAR is mounted vertically, pointing DOWN at the road surface.

  • Calibrated baseline = expected distance to a flat road surface (cm).
    Example: sensor mounted at 180 cm → baseline ≈ 180 cm.

  • Pothole  → reading GREATER than baseline (ground is farther from sensor).
  • Speed bump→ reading LESS    than baseline (ground is closer  to sensor).

  The model works on DEVIATIONS from the baseline, not raw distances.
  This makes it invariant to mounting height and self-calibrating.

Classes
───────
  0 – Flat road        (deviation ≈ 0, just noise)
  1 – Shallow pothole  (deviation  +3 to  +8 cm)
  2 – Deep pothole     (deviation  +9 to +25 cm)
  3 – Speed bump/hump  (deviation  -3 to -15 cm)

Features (22 per window) — all computed on the DEVIATION array:
  mean_dev, std_dev, max_dev, min_dev, range_dev,
  median_dev, iqr_dev, skewness, kurtosis,
  frac_positive (fraction of points where deviation > 0),
  frac_pothole  (fraction where deviation > +POTHOLE_THRESH),
  frac_bump     (fraction where deviation < -BUMP_THRESH),
  peak_depth    (max positive deviation),
  max_dip       (max negative deviation, absolute),
  max_run_pos   (longest consecutive run of positive deviations),
  max_run_neg   (longest consecutive run of negative deviations),
  slope_entry, slope_exit, half_asymmetry,
  n_extrema     (count of direction changes),
  str_mean, str_std  (signal strength stats — 0 if unavailable)
"""

import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import warnings

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
WINDOW_SIZE      = 20          # Readings per inference window
N_SAMPLES        = 4000        # Total training windows
BASELINE_CM      = 180         # Default road surface distance (sensor height)
NOISE_STD        = 1.5         # Road surface noise (cm)
POTHOLE_THRESH   = 3.0         # Min deviation to count as "in pothole" (cm)
BUMP_THRESH      = 3.0         # Min upward deviation to count as "bump" (cm)
MODEL_PATH       = "pothole_model.pkl"
META_PATH        = "feature_meta.pkl"


# ── Feature extractor ─────────────────────────────────────────────────────────

def extract_features(raw_window: np.ndarray,
                     strength: np.ndarray = None,
                     baseline: float = BASELINE_CM) -> np.ndarray:
    """
    Convert a window of raw LiDAR distance readings to a deviation-based
    feature vector.

    Parameters
    ----------
    raw_window : 1-D array of distance readings in cm  (length = WINDOW_SIZE)
    strength   : 1-D array of signal strength values   (same length, optional)
    baseline   : Road surface distance in cm (default BASELINE_CM)

    Returns
    -------
    1-D numpy float64 array of length 22
    """
    w   = np.asarray(raw_window, dtype=float)
    dev = w - baseline          # Positive → pothole / Negative → bump

    n   = len(dev)

    # ── Basic stats on deviation ──────────────────────────────────────────
    mean_dev   = dev.mean()
    std_dev    = dev.std()
    max_dev    = dev.max()
    min_dev    = dev.min()
    range_dev  = max_dev - min_dev
    median_dev = float(np.median(dev))
    iqr_dev    = float(np.percentile(dev, 75) - np.percentile(dev, 25))
    skew_dev   = float(skew(dev))
    kurt_dev   = float(kurtosis(dev))

    # ── Pothole / bump zone fractions ─────────────────────────────────────
    frac_positive = float(np.mean(dev > 0))
    frac_pothole  = float(np.mean(dev >  POTHOLE_THRESH))
    frac_bump     = float(np.mean(dev < -BUMP_THRESH))
    peak_depth    = float(max(max_dev, 0))          # Max sag below road
    max_dip       = float(max(-min_dev, 0))         # Max rise above road

    # ── Longest consecutive runs ──────────────────────────────────────────
    def _max_run(mask):
        best = run = 0
        for v in mask:
            run = (run + 1) if v else 0
            best = max(best, run)
        return best

    max_run_pos = _max_run(dev >  POTHOLE_THRESH)
    max_run_neg = _max_run(dev < -BUMP_THRESH)

    # ── Gradient / asymmetry ──────────────────────────────────────────────
    half        = n // 2
    slope_entry = float((dev[half - 1] - dev[0]) / max(half, 1))
    slope_exit  = float((dev[-1] - dev[half])    / max(n - half, 1))
    half_asym   = float(dev[:half].mean() - dev[half:].mean())

    # ── Number of local extrema ───────────────────────────────────────────
    signs    = np.sign(np.diff(dev))
    n_extr   = int(np.sum(np.diff(signs) != 0))

    # ── Strength features ─────────────────────────────────────────────────
    if strength is not None and len(strength) > 0:
        s        = np.asarray(strength, dtype=float)
        str_mean = float(s.mean())
        str_std  = float(s.std())
    else:
        str_mean = str_std = 0.0

    return np.array([
        mean_dev, std_dev, max_dev, min_dev, range_dev,
        median_dev, iqr_dev, skew_dev, kurt_dev,
        frac_positive, frac_pothole, frac_bump,
        peak_depth, max_dip,
        max_run_pos, max_run_neg,
        slope_entry, slope_exit, half_asym,
        n_extr,
        str_mean, str_std,
    ], dtype=np.float64)


FEATURE_NAMES = [
    "mean_dev", "std_dev", "max_dev", "min_dev", "range_dev",
    "median_dev", "iqr_dev", "skewness", "kurtosis",
    "frac_positive", "frac_pothole", "frac_bump",
    "peak_depth_cm", "max_bump_cm",
    "max_run_pothole", "max_run_bump",
    "slope_entry", "slope_exit", "half_asymmetry",
    "n_extrema",
    "str_mean", "str_std",
]
N_FEATURES = len(FEATURE_NAMES)   # 22


# ── Synthetic data generator ──────────────────────────────────────────────────

def _make_window(baseline, noise_std, window_size,
                 event_type="flat", depth=0, start=5, width=4):
    """Generate one synthetic window of raw LiDAR readings."""
    w = np.random.normal(baseline, noise_std, window_size)  # Road noise

    if event_type == "pothole":
        # Distance INCREASES over the hole (ground is farther)
        end = min(start + width, window_size)
        w[start:end] += depth

    elif event_type == "bump":
        # Distance DECREASES over the bump (ground is closer)
        end = min(start + width, window_size)
        w[start:end] -= depth
        w = np.clip(w, 10, None)   # Physical floor

    return w


def generate_synthetic_data(n_samples=N_SAMPLES,
                             window_size=WINDOW_SIZE,
                             baseline=BASELINE_CM):
    """
    Generate labelled synthetic windows.
    Returns X (feature matrix), y (labels), raw_windows (for debug).
    """
    X_feat, y_list, X_raw = [], [], []
    per_class = n_samples // 4

    rng = np.random.default_rng(42)

    for cls, cfg in [
        (0, {"event_type": "flat"}),
        (1, {"event_type": "pothole", "depth_range": (3, 8)}),
        (2, {"event_type": "pothole", "depth_range": (9, 25)}),
        (3, {"event_type": "bump",    "depth_range": (3, 15)}),
    ]:
        for _ in range(per_class):
            depth = 0
            if cls in (1, 2, 3):
                lo, hi = cfg["depth_range"]
                depth  = rng.uniform(lo, hi)

            start = int(rng.integers(2, window_size - 7))
            width = int(rng.integers(2, 6))

            w = _make_window(
                baseline, NOISE_STD, window_size,
                event_type=cfg["event_type"],
                depth=depth, start=start, width=width
            )

            # Simulate realistic strength (asphalt range 20–80)
            base_str = rng.uniform(20, 80, window_size)
            # Strength drops slightly inside a pothole (more scatter)
            if cls in (1, 2):
                base_str[start:start + width] *= rng.uniform(0.5, 0.9)

            X_feat.append(extract_features(w, base_str, baseline))
            X_raw.append(w)
            y_list.append(cls)

    return np.array(X_feat), np.array(y_list), np.array(X_raw)


# ── Train ─────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  Pothole Detector — Model Training")
print(f"  Baseline road distance: {BASELINE_CM} cm")
print(f"  Window size: {WINDOW_SIZE} readings  |  Features: {N_FEATURES}")
print("=" * 60)

print(f"\n[1/4] Generating {N_SAMPLES} synthetic windows … ", end="", flush=True)
X, y, X_raw = generate_synthetic_data()
unique, counts = np.unique(y, return_counts=True)
print(f"done.  Shape: {X.shape}")
for cls, cnt in zip(unique, counts):
    labels = {0: "Flat road", 1: "Shallow pothole",
              2: "Deep pothole", 3: "Speed bump"}
    print(f"   class {cls} ({labels[cls]}): {cnt} samples")

print("\n[2/4] Split 80/20 stratified … ", end="", flush=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("done")

print("\n[3/4] Training RandomForest pipeline …")
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_f1 = cross_val_score(pipeline, X_train, y_train,
                         cv=cv, scoring="f1_weighted", n_jobs=-1)
print(f"   5-fold CV F1 (weighted): {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")

pipeline.fit(X_train, y_train)

print("\n[4/4] Evaluation on held-out test set:")
target_names = ["Flat road", "Shallow pothole", "Deep pothole", "Speed bump"]
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred, target_names=target_names))
print("Confusion matrix (rows=actual, cols=predicted):")
print(confusion_matrix(y_test, y_pred))

# ── Save ──────────────────────────────────────────────────────────────────────
joblib.dump(pipeline, MODEL_PATH)
print(f"\n✅  Model saved → {MODEL_PATH}")

joblib.dump({
    "feature_names" : FEATURE_NAMES,
    "n_features"    : N_FEATURES,
    "window_size"   : WINDOW_SIZE,
    "baseline_cm"   : BASELINE_CM,
    "pothole_thresh": POTHOLE_THRESH,
    "bump_thresh"   : BUMP_THRESH,
    "class_labels"  : {0: "Flat road", 1: "Shallow pothole",
                       2: "Deep pothole", 3: "Speed bump"},
}, META_PATH)
print(f"✅  Feature metadata saved → {META_PATH}")