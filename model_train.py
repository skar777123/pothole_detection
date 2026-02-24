"""
model_train.py
==============
Trains a pothole / road-anomaly classifier based on BASELINE-DEVIATION features.

Concept
───────
  LiDAR points DOWN at road.  Baseline = expected distance to flat road.

  • Pothole   → reading GREATER than baseline (ground is farther)
  • Speed bump → reading LESS    than baseline (ground is closer)

Classes
───────
  0 – Flat road
  1 – Shallow pothole  (+3 to  +8 cm deviation)
  2 – Deep pothole     (+9 to +25 cm deviation)
  3 – Speed bump/hump  (-3 to -15 cm deviation)

IMPORTANT
─────────
  The training code (generate / fit / save) is wrapped in:
      if __name__ == "__main__":
  so that importing this module (e.g. from dashboard.py) ONLY provides the
  helper functions and constants — it does NOT re-run training.
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

# ── Shared constants (imported by dashboard.py) ───────────────────────────────
WINDOW_SIZE      = 20          # Readings per inference window
N_SAMPLES        = 8000        # More samples for wider baseline coverage
BASELINE_CM      = 300         # Default mounting height (cm) — mid-range for 0.1–10 m
NOISE_STD        = 1.5         # Road surface noise at close range (cm)
BASELINE_MIN_CM  = 50          # Minimum baseline used during training
BASELINE_MAX_CM  = 1000        # Maximum baseline used during training (10 m)
POTHOLE_THRESH   = 3.0         # Minimum positive deviation = pothole zone (cm)
BUMP_THRESH      = 3.0         # Minimum negative deviation = bump zone (cm)
MODEL_PATH       = "pothole_model.pkl"
META_PATH        = "feature_meta.pkl"


def _noise_for_baseline(baseline_cm: float) -> float:
    """
    TF02-Pro noise scales with distance.
    Approx: ±1 cm at <2 m  →  ±4 cm at 10 m  (~0.4% of range).
    We add a small base floor of 1 cm.
    """
    return max(1.0, baseline_cm * 0.004)


# ── Feature extractor — importable by dashboard ───────────────────────────────

def extract_features(raw_window: np.ndarray,
                     strength: np.ndarray = None,
                     baseline: float = BASELINE_CM) -> np.ndarray:
    """
    Convert a window of LiDAR distance readings to a 22-element feature vector.

    All features are computed on DEVIATION = raw_window - baseline so the
    classifier is invariant to sensor mounting height.

    Parameters
    ----------
    raw_window : 1-D array, distance readings in cm
    strength   : 1-D array, signal strength values (optional, same length)
    baseline   : Road surface distance in cm

    Returns
    -------
    np.ndarray, shape (22,), dtype float64
    """
    w   = np.asarray(raw_window, dtype=float)
    dev = w - baseline       # + = pothole, - = bump

    n   = len(dev)

    # Basic stats
    mean_dev   = float(dev.mean())
    std_dev    = float(dev.std())
    max_dev    = float(dev.max())
    min_dev    = float(dev.min())
    range_dev  = max_dev - min_dev
    median_dev = float(np.median(dev))
    iqr_dev    = float(np.percentile(dev, 75) - np.percentile(dev, 25))
    skew_dev   = float(skew(dev))
    kurt_dev   = float(kurtosis(dev))

    # Zone fractions
    frac_positive = float(np.mean(dev > 0))
    frac_pothole  = float(np.mean(dev >  POTHOLE_THRESH))
    frac_bump     = float(np.mean(dev < -BUMP_THRESH))
    peak_depth    = float(max(max_dev, 0))
    max_dip       = float(max(-min_dev, 0))

    # Longest consecutive runs inside each zone
    def _max_run(mask):
        best = run = 0
        for v in mask:
            run = (run + 1) if v else 0
            best = max(best, run)
        return float(best)

    max_run_pos = _max_run(dev >  POTHOLE_THRESH)
    max_run_neg = _max_run(dev < -BUMP_THRESH)

    # Gradient / asymmetry
    half        = n // 2
    slope_entry = float((dev[half - 1] - dev[0]) / max(half, 1))
    slope_exit  = float((dev[-1] - dev[half])    / max(n - half, 1))
    half_asym   = float(dev[:half].mean() - dev[half:].mean())

    # Local direction changes
    signs  = np.sign(np.diff(dev))
    n_extr = float(np.sum(np.diff(signs) != 0))

    # Strength features
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
    w = np.random.normal(baseline, noise_std, window_size)
    if event_type == "pothole":
        end = min(start + width, window_size)
        w[start:end] += depth
    elif event_type == "bump":
        end = min(start + width, window_size)
        w[start:end] -= depth
        w = np.clip(w, 10, None)
    return w


def generate_synthetic_data(n_samples=N_SAMPLES, window_size=WINDOW_SIZE,
                             baseline=None):
    """
    Generate synthetic windows across the full baseline range (50–1000 cm).

    When `baseline` is None (default), each window is assigned a RANDOM
    baseline drawn uniformly from [BASELINE_MIN_CM, BASELINE_MAX_CM].
    This makes the trained model invariant to mounting height / distance.
    """
    X_feat, y_list, X_raw = [], [], []
    per_class = n_samples // 4
    rng = np.random.default_rng(42)

    class_configs = [
        (0, "flat",    None),
        (1, "pothole", (3,  8)),
        (2, "pothole", (9, 25)),
        (3, "bump",    (3, 15)),
    ]

    for cls, etype, depth_range in class_configs:
        for _ in range(per_class):
            # ── Pick a random baseline for this window ──────────────────────
            if baseline is None:
                b = float(rng.uniform(BASELINE_MIN_CM, BASELINE_MAX_CM))
            else:
                b = float(baseline)

            # Scale noise with distance (TF02-Pro degrades at range)
            noise = _noise_for_baseline(b)

            depth = float(rng.uniform(*depth_range)) if depth_range else 0.0
            start = int(rng.integers(2, window_size - 7))
            width = int(rng.integers(2, 6))

            w = _make_window(b, noise_std=noise,
                             window_size=window_size,
                             event_type=etype, depth=depth,
                             start=start, width=width)

            # Signal strength: weaker at long range
            str_base = float(rng.uniform(200, 800) * max(0.1, 1.0 - b / 1500))
            base_str = rng.uniform(str_base * 0.8, str_base * 1.2, window_size)
            if cls in (1, 2):
                base_str[start:start + width] *= rng.uniform(0.5, 0.9)

            X_feat.append(extract_features(w, base_str, b))
            X_raw.append(w)
            y_list.append(cls)

    return np.array(X_feat), np.array(y_list), np.array(X_raw)


# ── Training entry point — ONLY runs when executed directly ───────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Pothole Detector — Model Training")
    print(f"  Baseline range: {BASELINE_MIN_CM}–{BASELINE_MAX_CM} cm (random per window)")
    print(f"  Window: {WINDOW_SIZE}  |  Features: {N_FEATURES}")
    print("=" * 60)

    print(f"\n[1/4] Generating {N_SAMPLES} synthetic windows "
          f"(varied baselines {BASELINE_MIN_CM}–{BASELINE_MAX_CM} cm) …", flush=True)
    X, y, X_raw = generate_synthetic_data()   # baseline=None → random per window
    unique, counts = np.unique(y, return_counts=True)
    print(f"      Shape: {X.shape}")
    labels_map = {0: "Flat road", 1: "Shallow pothole",
                  2: "Deep pothole", 3: "Speed bump"}
    for cls, cnt in zip(unique, counts):
        print(f"   class {cls} ({labels_map[cls]}): {cnt} samples")

    print("\n[2/4] Splitting data 80/20 stratified …", flush=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

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

    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1  = cross_val_score(pipeline, X_train, y_train,
                              cv=cv, scoring="f1_weighted", n_jobs=-1)
    print(f"   5-fold CV F1 (weighted): {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")

    pipeline.fit(X_train, y_train)

    print("\n[4/4] Evaluation on held-out test set:")
    y_pred = pipeline.predict(X_test)
    target_names = [labels_map[i] for i in sorted(labels_map)]
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(pipeline, MODEL_PATH)
    print(f"\n✅  Model saved → {MODEL_PATH}")

    joblib.dump({
        "feature_names"  : FEATURE_NAMES,
        "n_features"     : N_FEATURES,
        "window_size"    : WINDOW_SIZE,
        "baseline_cm"    : BASELINE_CM,
        "baseline_min_cm": BASELINE_MIN_CM,
        "baseline_max_cm": BASELINE_MAX_CM,
        "pothole_thresh" : POTHOLE_THRESH,
        "bump_thresh"    : BUMP_THRESH,
        "class_labels"   : labels_map,
    }, META_PATH)
    print(f"✅  Metadata saved → {META_PATH}")