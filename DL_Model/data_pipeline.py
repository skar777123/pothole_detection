"""
data_pipeline.py
================
Loads real LiDAR export CSVs, generates physics-based synthetic data,
augments, combines, and packages everything into (X, y) tensors ready
for training the 1D-CNN + BiLSTM deep learning model.

X shape : (N, WINDOW_SIZE, N_RAW_FEATURES)
            features per timestep: [distance_cm, strength, deviation_cm]
y shape : (N,)  integer class labels  0-3
"""

import os
import glob
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

warnings.filterwarnings("ignore")

from dl_config import (
    DATASET_DIR, SCALER_SAVE_PATH, MODELS_DIR, LOGS_DIR,
    BASELINE_CM, BASELINE_MIN_CM, BASELINE_MAX_CM,
    POTHOLE_THRESH, BUMP_THRESH,
    WINDOW_SIZE, STRIDE, N_RAW_FEATURES,
    N_SYNTHETIC_SAMPLES, REAL_AUGMENT_FACTOR,
    VALIDATION_SPLIT, TEST_SPLIT,
    RANDOM_SEED, N_CLASSES,
    CLASS_NAMES,
    ADAPT_MA_WINDOW, ADAPT_HP_ALPHA, ADAPT_N_FEATURES,
)

# ── helpers ───────────────────────────────────────────────────────────────────

def _noise_for_baseline(baseline_cm: float) -> float:
    """TF02-Pro: noise ≈ 1 cm at <2 m → ≈4 cm at 10 m  (0.4 % of range)."""
    return max(1.0, baseline_cm * 0.004)


def _label_from_type(t: str, dev: float) -> int:
    """Map a raw 'Type' emoji string + deviation to an integer class."""
    t = str(t).lower()
    if "shallow" in t or "pothole" in t:
        if abs(dev) < POTHOLE_THRESH:
            return 0
        return 1
    if "deep" in t:
        return 2
    if "bump" in t or "speed" in t:
        return 3
    # fallback: sign of deviation
    if dev > POTHOLE_THRESH:
        return 1 if dev < 9 else 2
    if dev < -BUMP_THRESH:
        return 3
    return 0


def _severity_label(depth_cm: float) -> str:
    if depth_cm < 2:
        return "Noise"
    if depth_cm < 8:
        return "Shallow"
    if depth_cm < 20:
        return "Moderate"
    return "Deep/Dangerous"


# ── Real data loader ──────────────────────────────────────────────────────────

def load_real_data() -> pd.DataFrame:
    """
    Load all CSV export files from DATASET_DIR.
    Standardise column names and return a tidy DataFrame.
    """
    pattern = os.path.join(DATASET_DIR, "*.csv")
    files   = sorted(glob.glob(pattern))
    if not files:
        print(f"  [data_pipeline] No CSV files found in {DATASET_DIR}")
        return pd.DataFrame()

    frames = []
    for fp in files:
        try:
            df = pd.read_csv(fp, index_col=0)
            # Rename columns for consistency
            df.columns = [c.strip() for c in df.columns]
            rename = {
                "Dev (cm)"   : "dev_cm",
                "Type"       : "type",
                "Depth (cm)" : "depth_cm",
                "Length (cm)": "length_cm",
                "Width (cm)" : "width_cm",
                "Severity"   : "severity",
                "Conf."      : "confidence",
                "Strength"   : "strength",
                "Baseline"   : "baseline_cm",
                "Time"       : "time",
            }
            df.rename(columns=rename, inplace=True)
            # Parse dev_cm: strip "+" sign then coerce to float
            df["dev_cm"] = (
                df["dev_cm"].astype(str)
                .str.replace("+", "", regex=False)
                .pipe(pd.to_numeric, errors="coerce")
            )
            frames.append(df)
        except Exception as exc:
            print(f"  [data_pipeline] Warning – skipping {fp}: {exc}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Coerce numeric columns
    for col in ["dev_cm", "depth_cm", "length_cm", "width_cm", "strength", "baseline_cm"]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    combined.dropna(subset=["dev_cm", "baseline_cm"], inplace=True)

    # Assign integer labels
    combined["label"] = combined.apply(
        lambda r: _label_from_type(r.get("type", ""), r["dev_cm"]), axis=1
    )

    print(f"  [data_pipeline] Loaded {len(combined)} real rows from {len(files)} files")
    print(f"  [data_pipeline] Real class distribution:")
    for cls, cnt in combined["label"].value_counts().sort_index().items():
        print(f"    class {cls} ({CLASS_NAMES[cls]}): {cnt}")
    return combined


# ── Physics-based synthetic window generator ──────────────────────────────────

def _make_raw_window(
    baseline_cm: float,
    event: str = "flat",     # "flat" | "pothole" | "bump"
    depth_cm: float = 0.0,
    start_idx: int = 5,
    event_width: int = 4,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate one WINDOW_SIZE-length LiDAR measurement sequence.

    Returns
    -------
    distances : (WINDOW_SIZE,)  distance readings in cm
    strength  : (WINDOW_SIZE,)  signal-strength readings
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    noise = _noise_for_baseline(baseline_cm)
    dist  = rng.normal(baseline_cm, noise, WINDOW_SIZE)

    end_idx = min(start_idx + event_width, WINDOW_SIZE)
    if event == "pothole":
        dist[start_idx:end_idx] += depth_cm
    elif event == "bump":
        dist[start_idx:end_idx] -= depth_cm
        dist = np.clip(dist, 10, None)

    # Signal strength: weaker at long range, weaker inside potholes
    str_base  = float(rng.uniform(200, 800) * max(0.1, 1.0 - baseline_cm / 1500))
    strength  = rng.uniform(str_base * 0.85, str_base * 1.15, WINDOW_SIZE)
    if event == "pothole":
        strength[start_idx:end_idx] *= rng.uniform(0.45, 0.85)

    return dist, strength


def generate_synthetic_data(n_per_class: int = N_SYNTHETIC_SAMPLES,
                             rng: np.random.Generator = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate n_per_class windows per class using varied baselines,
    depths and event positions to maximise coverage.

    Returns
    -------
    X : (n_per_class*4, WINDOW_SIZE, N_RAW_FEATURES)
    y : (n_per_class*4,)
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    class_configs = [
        # (class_id, event_type, depth_range_cm)
        (0, "flat",    (0,    0)),
        (1, "pothole", (3,    8)),
        (2, "pothole", (9,   65)),
        (3, "bump",    (3,   50)),
    ]

    X_list, y_list = [], []

    for cls, etype, (d_min, d_max) in class_configs:
        for _ in range(n_per_class):
            baseline = float(rng.uniform(BASELINE_MIN_CM, BASELINE_MAX_CM))
            depth    = float(rng.uniform(d_min, d_max))
            start    = int(rng.integers(2, WINDOW_SIZE - 8))
            width    = int(rng.integers(2, 8))

            dist, strength = _make_raw_window(
                baseline_cm=baseline,
                event=etype,
                depth_cm=depth,
                start_idx=start,
                event_width=width,
                rng=rng,
            )

            dev = dist - baseline
            window = np.stack([dist, strength, dev], axis=-1)  # (T, 3)
            X_list.append(window)
            y_list.append(cls)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y


# ── Real-data augmentation into synthetic-style windows ──────────────────────

def augment_real_row(
    dev_cm: float,
    depth_cm: float,
    strength_val: float,
    baseline_cm: float,
    label: int,
    factor: int = REAL_AUGMENT_FACTOR,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    From a single real event row, create `factor` augmented windows
    by jittering depth, baseline, noise, and event position.
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    if np.isnan(baseline_cm) or baseline_cm < 10:
        baseline_cm = BASELINE_CM
    if np.isnan(depth_cm):
        depth_cm = abs(dev_cm)
    if np.isnan(strength_val):
        strength_val = 3000.0

    event_map = {0: "flat", 1: "pothole", 2: "pothole", 3: "bump"}
    etype = event_map.get(label, "flat")

    X_aug, y_aug = [], []
    for _ in range(factor):
        b     = float(rng.normal(baseline_cm, baseline_cm * 0.02))
        b     = np.clip(b, BASELINE_MIN_CM, BASELINE_MAX_CM)
        d     = float(abs(rng.normal(depth_cm, max(1.0, depth_cm * 0.15))))
        start = int(rng.integers(2, WINDOW_SIZE - 8))
        width = int(rng.integers(3, 9))

        dist, strength = _make_raw_window(
            baseline_cm=b,
            event=etype,
            depth_cm=d,
            start_idx=start,
            event_width=width,
            rng=rng,
        )
        # Blend signal strength from real sensor value
        s_scale = (strength_val / 3000.0) * rng.uniform(0.9, 1.1)
        strength = strength * s_scale

        dev    = dist - b
        window = np.stack([dist, strength, dev], axis=-1)
        X_aug.append(window)
        y_aug.append(label)

    return np.array(X_aug, dtype=np.float32), np.array(y_aug, dtype=np.int64)


# ── Main assembly ─────────────────────────────────────────────────────────────

def build_dataset() -> dict:
    """
    Full pipeline:
      1. Load real CSV rows
      2. Augment each real row × REAL_AUGMENT_FACTOR
      3. Generate synthetic windows
      4. Combine → fit scaler on train split → return dict of splits

    Returns
    -------
    dict with keys: X_train, X_val, X_test, y_train, y_val, y_test,
                    scaler, class_counts
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR,   exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)

    # ── 1. Real data ──────────────────────────────────────────────────────────
    real_df     = load_real_data()
    X_real_list, y_real_list = [], []

    if not real_df.empty:
        for _, row in real_df.iterrows():
            Xa, ya = augment_real_row(
                dev_cm       = float(row.get("dev_cm",      0)),
                depth_cm     = float(row.get("depth_cm",    0)),
                strength_val = float(row.get("strength", 3000)),
                baseline_cm  = float(row.get("baseline_cm", BASELINE_CM)),
                label        = int(row["label"]),
                factor       = REAL_AUGMENT_FACTOR,
                rng          = rng,
            )
            X_real_list.append(Xa)
            y_real_list.append(ya)

        X_real = np.concatenate(X_real_list, axis=0)
        y_real = np.concatenate(y_real_list, axis=0)
        print(f"  [data_pipeline] Augmented real data: {X_real.shape[0]} windows")
    else:
        X_real = np.empty((0, WINDOW_SIZE, N_RAW_FEATURES), dtype=np.float32)
        y_real = np.empty((0,), dtype=np.int64)

    # ── 2. Synthetic data ─────────────────────────────────────────────────────
    print(f"  [data_pipeline] Generating {N_SYNTHETIC_SAMPLES * N_CLASSES} synthetic windows …")
    X_synth, y_synth = generate_synthetic_data(n_per_class=N_SYNTHETIC_SAMPLES, rng=rng)
    print(f"  [data_pipeline] Synthetic data: {X_synth.shape[0]} windows")

    # ── 3. Combine ────────────────────────────────────────────────────────────
    X_all = np.concatenate([X_real, X_synth], axis=0)
    y_all = np.concatenate([y_real, y_synth], axis=0)

    # Shuffle
    idx   = rng.permutation(len(X_all))
    X_all = X_all[idx]
    y_all = y_all[idx]

    print(f"  [data_pipeline] Total dataset: {X_all.shape[0]} windows "
          f"  shape={X_all.shape}")

    # ── 4. Normalise per-feature (across timestep dimension) ─────────────────
    # Flatten time dimension for scaler fit, then restore
    N, T, F = X_all.shape
    X_flat  = X_all.reshape(-1, F)          # (N*T, F)

    # Train/val/test split BEFORE fitting scaler to avoid leakage
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_all, y_all,
        test_size    = TEST_SPLIT,
        stratify     = y_all,
        random_state = RANDOM_SEED,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size    = VALIDATION_SPLIT / (1 - TEST_SPLIT),
        stratify     = y_trainval,
        random_state = RANDOM_SEED,
    )

    # Fit scaler on training set only
    scaler = StandardScaler()
    Nt, Tt, Ft = X_train.shape
    scaler.fit(X_train.reshape(-1, Ft))

    def scale(X):
        n, t, f = X.shape
        return scaler.transform(X.reshape(-1, f)).reshape(n, t, f).astype(np.float32)

    X_train = scale(X_train)
    X_val   = scale(X_val)
    X_test  = scale(X_test)

    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"  [data_pipeline] Scaler saved → {SCALER_SAVE_PATH}")

    # Class counts for logging
    unique, counts = np.unique(y_train, return_counts=True)
    class_counts   = dict(zip(unique.tolist(), counts.tolist()))

    print(f"\n  Dataset splits:")
    print(f"    Train : {X_train.shape[0]:>6}  Val : {X_val.shape[0]:>6}  Test : {X_test.shape[0]:>6}")
    print(f"  Train class distribution:")
    for cls, cnt in class_counts.items():
        print(f"    class {cls} ({CLASS_NAMES[cls]}): {cnt}")

    return {
        "X_train"      : X_train,
        "X_val"        : X_val,
        "X_test"       : X_test,
        "y_train"      : y_train,
        "y_val"        : y_val,
        "y_test"       : y_test,
        "scaler"       : scaler,
        "class_counts" : class_counts,
    }


if __name__ == "__main__":
    print("=== Data Pipeline Self-Test ===")
    splits = build_dataset()
    print("\nX_train:", splits["X_train"].shape)
    print("y_train sample:", splits["y_train"][:10])


# ── Adaptive 6-feature dataset builder ───────────────────────────────────────────────

def _apply_adaptive_pipeline(dist_seq: np.ndarray,
                              strength_seq: np.ndarray,
                              ma_window: int = ADAPT_MA_WINDOW,
                              hp_alpha: float = ADAPT_HP_ALPHA
                             ) -> np.ndarray:
    """
    Convert a (WINDOW_SIZE,) distance + strength sequence into the
    6-feature adaptive representation used by AdaptiveDetector:
      [dist, strength, ma_dev, hp_signal, velocity, above_thresh_flag]
    Returns (WINDOW_SIZE, ADAPT_N_FEATURES) float32 array.
    """
    T = len(dist_seq)
    ma_window = min(ma_window, T)          # guard for short sequences

    # Moving-average deviation
    ma   = np.convolve(dist_seq, np.ones(ma_window) / ma_window, mode="same")
    # Fix edges: use expanding mean for first ma_window-1 samples
    for i in range(min(ma_window - 1, T)):
        ma[i] = dist_seq[:i+1].mean()
    ma_dev = dist_seq - ma

    # High-pass IIR
    hp = np.zeros(T, dtype=np.float32)
    for t in range(1, T):
        hp[t] = hp_alpha * (hp[t-1] + dist_seq[t] - dist_seq[t-1])

    # Velocity
    vel    = np.zeros(T, dtype=np.float32)
    vel[1:] = np.diff(dist_seq)

    # Above-threshold flag
    above  = (ma_dev > POTHOLE_THRESH).astype(np.float32)

    return np.stack(
        [dist_seq, strength_seq, ma_dev, hp, vel, above], axis=-1
    ).astype(np.float32)   # (T, 6)


def build_adaptive_dataset(n_per_class: int = N_SYNTHETIC_SAMPLES) -> dict:
    """
    Same as build_dataset() but produces ADAPT_N_FEATURES=6 feature windows.
    Saves scaler to  models/dl_scaler_adaptive.pkl.

    Use this when you want to retrain dl_model.py with adaptive features:
        model = build_model(n_features=ADAPT_N_FEATURES)
    """
    import joblib
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR,   exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)

    # Reuse synthetic generator from build_dataset()
    X_raw, y_raw = generate_synthetic_data(n_per_class=n_per_class, rng=rng)
    # X_raw: (N, WINDOW_SIZE, 3)  columns: [dist, strength, dev]

    # Convert to 6-feature via adaptive pipeline
    X6 = np.stack(
        [_apply_adaptive_pipeline(X_raw[i, :, 0], X_raw[i, :, 1])
         for i in range(len(X_raw))],
        axis=0
    )  # (N, WINDOW_SIZE, 6)

    print(f"  [data_pipeline] Adaptive dataset: {X6.shape[0]} windows  shape={X6.shape}")

    # Split BEFORE fitting scaler
    X_tv, X_test, y_tv, y_test = train_test_split(
        X6, y_raw, test_size=TEST_SPLIT,
        stratify=y_raw, random_state=RANDOM_SEED,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size=VALIDATION_SPLIT / (1 - TEST_SPLIT),
        stratify=y_tv, random_state=RANDOM_SEED,
    )

    scaler = StandardScaler()
    Nt, Tt, Ft = X_train.shape
    scaler.fit(X_train.reshape(-1, Ft))

    def _scale(X):
        n, t, f = X.shape
        return scaler.transform(X.reshape(-1, f)).reshape(n, t, f).astype(np.float32)

    X_train = _scale(X_train)
    X_val   = _scale(X_val)
    X_test  = _scale(X_test)

    adaptive_scaler_path = os.path.join(MODELS_DIR, "dl_scaler_adaptive.pkl")
    joblib.dump(scaler, adaptive_scaler_path)
    print(f"  [data_pipeline] Adaptive scaler saved → {adaptive_scaler_path}")

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val":   y_val, "y_test":  y_test,
        "scaler": scaler, "n_features": ADAPT_N_FEATURES,
    }
