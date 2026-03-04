"""
adaptive_detector.py
====================
Adaptive 1D LiDAR Pothole Detector for Raspberry Pi.

Replaces the naive  "if distance > 1000 cm → pothole"  logic with a
four-stage signal-processing pipeline that the DL (LSTM) model then
classifies using richer features.

Pipeline stages
───────────────
  Stage 1 │ Moving-Average Baseline
           │   baseline[t] = mean(last ADAPT_MA_WINDOW readings)
           │   ma_dev[t]   = dist[t] − baseline[t]
           │
  Stage 2 │ IIR High-Pass Filter  (removes slow road-slope drift)
           │   hp[t] = α · (hp[t-1] + dist[t] − dist[t-1])
           │
  Stage 3 │ Derivative / Velocity Check
           │   vel[t] = dist[t] − dist[t-1]
           │   Pothole signature: vel spikes UP then spikes DOWN.
           │
  Stage 4 │ Depth-Duration Guard
           │   Require deviation > POTHOLE_THRESH for ≥ ADAPT_MIN_DURATION
           │   consecutive readings before calling a pothole.
           │   Rejects single-spike noise.

Feature vector per timestep (feeds the LSTM)
─────────────────────────────────────────────
  [dist_cm, strength, ma_dev, hp_signal, velocity, above_thresh_flag]
  = ADAPT_N_FEATURES = 6

Class outputs (same as DL model)
────────────────────────────────
  0 → Flat Road
  1 → Shallow Pothole
  2 → Deep Pothole
  3 → Speed Bump

Usage – standalone rule-based mode (no trained model needed)
──────────────────────────────────────────────────────────────
  python adaptive_detector.py --demo

Usage – with trained LSTM model
────────────────────────────────
  python adaptive_detector.py --demo --use-model

Usage – real serial LiDAR
──────────────────────────
  python adaptive_detector.py --port /dev/ttyUSB0 --baud 115200 --use-model

Import as module
──────────────────
  from adaptive_detector import AdaptiveDetector
  det = AdaptiveDetector(use_model=False)   # rules only
  result = det.feed(distance_cm=312.5, strength=3200)
"""

from __future__ import annotations

import os, sys, time, argparse, threading, queue
from collections import deque
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from dl_config import (
    WINDOW_SIZE, N_CLASSES, CLASS_NAMES, SEVERITY_LEVELS,
    MODEL_SAVE_PATH, SCALER_SAVE_PATH,
    POTHOLE_THRESH, BUMP_THRESH,
    RT_CONFIDENCE_THRESHOLD, RT_ALERT_DEPTH_CM,
    ADAPT_MA_WINDOW, ADAPT_HP_ALPHA,
    ADAPT_VEL_UP_THRESH, ADAPT_VEL_DOWN_THRESH,
    ADAPT_MIN_DURATION, ADAPT_N_FEATURES,
)


# ── Lazy model loader ─────────────────────────────────────────────────────────

_model  = None
_scaler = None


def _load_model():
    global _model, _scaler
    if _model is not None:
        return
    import joblib
    from dl_model import load_model as _lm
    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_SAVE_PATH}.\n"
            "Run  python train.py  first."
        )
    _model  = _lm(MODEL_SAVE_PATH)
    _scaler = joblib.load(SCALER_SAVE_PATH)
    print(f"[AdaptiveDetector] DL model loaded ← {MODEL_SAVE_PATH}")


# ═══════════════════════════════════════════════════════════════════════════════
# Stage helpers
# ═══════════════════════════════════════════════════════════════════════════════

class _MovingAverageBaseline:
    """
    Stage 1 – Moving Average Baseline.

    Keeps a sliding ring-buffer of the last N raw readings and returns
    their mean as the 'expected flat-road distance' at this moment.

    Key insight: as the vehicle goes uphill / downhill the baseline drifts
    slowly, but a pothole causes a sudden spike. The MA baseline adapts to the
    slope while staying insensitive to the short spike.

    IMPORTANT: Once calibrated, readings that deviate more than
    `freeze_thresh` cm from the current baseline are NOT added to the
    buffer.  This prevents the baseline from 'chasing' a pothole or bump
    and keeps it locked to the normal road surface.
    """

    def __init__(self, window: int = ADAPT_MA_WINDOW,
                 freeze_thresh: float = None):
        self._buf = deque(maxlen=window)
        self._sum = 0.0
        # Default freeze threshold: max of POTHOLE_THRESH and BUMP_THRESH
        self._freeze_thresh = freeze_thresh or max(POTHOLE_THRESH, BUMP_THRESH)

    def update(self, dist_cm: float) -> float:
        """Push one reading; return current adaptive baseline.

        If the baseline is already calibrated (window full) and the new
        reading deviates by more than ``freeze_thresh`` from the current
        mean, it is *excluded* from the buffer so the baseline stays
        locked to the normal surface distance.
        """
        if self.ready:
            current_baseline = self._sum / len(self._buf)
            deviation = abs(dist_cm - current_baseline)
            if deviation > self._freeze_thresh:
                # Anomalous reading → freeze baseline, return current mean
                return current_baseline

        # Normal reading → update buffer
        if len(self._buf) == self._buf.maxlen:
            self._sum -= self._buf[0]        # remove oldest
        self._buf.append(dist_cm)
        self._sum += dist_cm
        return self._sum / len(self._buf)    # current mean

    @property
    def ready(self) -> bool:
        """True once the window is fully filled."""
        return len(self._buf) == self._buf.maxlen

    @property
    def baseline(self) -> float:
        return self._sum / max(1, len(self._buf))


class _HighPassFilter:
    """
    Stage 2 – IIR First-Order High-Pass Filter.

    Equation (difference form):
        y[n] = α · (y[n-1] + x[n] − x[n-1])

    Removes slow variations (e.g. road slope) — only sudden transients
    (potholes, bumps) pass through.  α close to 1 → longer time constant.
    """

    def __init__(self, alpha: float = ADAPT_HP_ALPHA):
        self._alpha   = alpha
        self._prev_x  = None   # previous raw input
        self._prev_y  = 0.0    # previous filtered output

    def update(self, x: float) -> float:
        if self._prev_x is None:
            self._prev_x = x
            return 0.0
        y = self._alpha * (self._prev_y + x - self._prev_x)
        self._prev_x = x
        self._prev_y = y
        return y


class _DerivativeChecker:
    """
    Stage 3 – Velocity / Derivative Check.

    vel[t] = dist[t] − dist[t-1]

    Pothole signature:
        vel spikes POSITIVE  (sensor distance increases abruptly → rim entry)
        then spikes NEGATIVE (sensor distance returns → rim exit)

    Speed-bump signature is the mirror image.

    We emit a boolean `spike_up` and `spike_down` flag plus the raw velocity
    so the downstream model gets the full picture.
    """

    def __init__(self,
                 up_thresh: float   = ADAPT_VEL_UP_THRESH,
                 down_thresh: float = ADAPT_VEL_DOWN_THRESH):
        self._up    = up_thresh
        self._down  = down_thresh
        self._prev  = None

    def update(self, dist_cm: float) -> tuple[float, bool, bool]:
        """
        Returns (velocity, spike_up, spike_down).
        velocity   : signed cm/reading
        spike_up   : velocity > up_thresh   (start of pothole / entry edge)
        spike_down : velocity < down_thresh (end of pothole  / exit edge)
        """
        if self._prev is None:
            self._prev = dist_cm
            return 0.0, False, False
        vel        = dist_cm - self._prev
        self._prev = dist_cm
        return vel, vel > self._up, vel < self._down


class _DepthDurationGuard:
    """
    Stage 4 – Depth-Duration Confirmation.

    Real pothole:  deviation stays large for ≥ ADAPT_MIN_DURATION readings.
    Noise spike:   one or two readings then vanishes.

    Maintains a consecutive-readings counter.  When the run reaches
    ADAPT_MIN_DURATION the guard fires (confirmed=True).  It resets as
    soon as deviation drops below threshold.
    """

    def __init__(self,
                 threshold: float = POTHOLE_THRESH,
                 min_duration: int = ADAPT_MIN_DURATION):
        self._thresh    = threshold
        self._min_dur   = min_duration
        self._run       = 0     # consecutive readings above threshold
        self.confirmed  = False # True while inside a confirmed pothole

    def update(self, deviation: float) -> tuple[bool, int]:
        """
        Returns (confirmed, run_length).
        confirmed  : deviation has been sustained for ≥ min_duration readings
        run_length : current consecutive-readings count above threshold
        """
        if deviation > self._thresh:
            self._run += 1
        else:
            self._run  = 0

        self.confirmed = (self._run >= self._min_dur)
        return self.confirmed, self._run


# ═══════════════════════════════════════════════════════════════════════════════
# Feature builder
# ═══════════════════════════════════════════════════════════════════════════════

class _FeatureBuffer:
    """
    Combines the outputs of all four stages into a 6-feature vector per
    timestep, and accumulates WINDOW_SIZE worth of them for the LSTM.

      Feature index  │ Value
      ───────────────┼──────────────────────────────────────────────────
          0          │ raw dist_cm
          1          │ signal strength
          2          │ ma_dev   = dist - moving_average_baseline
          3          │ hp       = high-pass filtered signal
          4          │ velocity = dist[t] - dist[t-1]
          5          │ above_thresh_flag  (1 if ma_dev > POTHOLE_THRESH)
    """

    def __init__(self):
        self._rows = deque(maxlen=WINDOW_SIZE)

    def push(self,
             dist: float, strength: float,
             ma_dev: float, hp: float,
             velocity: float, above_flag: float) -> None:
        self._rows.append([dist, strength, ma_dev, hp, velocity, above_flag])

    @property
    def ready(self) -> bool:
        return len(self._rows) == WINDOW_SIZE

    def as_array(self) -> np.ndarray:
        """Return (WINDOW_SIZE, ADAPT_N_FEATURES) float32 array."""
        return np.array(self._rows, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Rule-based pothole classifier (no model required)
# ═══════════════════════════════════════════════════════════════════════════════

def _rule_based_classify(
    ma_dev: float,
    velocity: float,
    hp_signal: float,
    confirmed: bool,
    run_length: int,
) -> tuple[int, float]:
    """
    Lightweight rule-based classifier.  Returns (class_id, confidence).

    Decision tree:
    ─────────────
    1. Sustained deviation  AND  velocity spike pattern → pothole or bump
    2. Duration confirmed   AND  ma_dev negative        → speed bump
    3. ma_dev large (deep)                               → Deep Pothole
    4. ma_dev moderate                                   → Shallow Pothole
    5. else                                              → Flat Road
    """
    # Speed bump: distance decreases (sensor closer to object)
    if confirmed and ma_dev < -BUMP_THRESH:
        confidence = min(1.0, abs(ma_dev) / (BUMP_THRESH * 4))
        return 3, round(confidence, 4)

    if confirmed and ma_dev > POTHOLE_THRESH:
        # Deep vs Shallow based on ma_dev magnitude
        if ma_dev >= 9.0:
            confidence = min(1.0, ma_dev / 65.0)
            return 2, round(confidence, 4)
        else:
            confidence = min(1.0, ma_dev / 9.0)
            return 1, round(confidence, 4)

    # Not confirmed but still has meaningful deviation → suspicious
    if ma_dev > POTHOLE_THRESH and run_length > 0:
        conf = run_length / ADAPT_MIN_DURATION * 0.5
        return 1, round(min(conf, 0.55), 4)

    return 0, round(max(0.0, 1.0 - abs(ma_dev) / POTHOLE_THRESH), 4)


# ═══════════════════════════════════════════════════════════════════════════════
# Main AdaptiveDetector
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptiveDetector:
    """
    End-to-end adaptive 1D LiDAR pothole detector.

    Parameters
    ----------
    use_model : bool
        True  → run the trained LSTM model on the 6-feature window.
        False → use rule-based logic only (fast, no TF required).
    ma_window : int
        Moving-average window size (overrides ADAPT_MA_WINDOW from config).
    hp_alpha : float
        High-pass IIR coefficient (overrides ADAPT_HP_ALPHA).
    min_duration : int
        Minimum consecutive readings for depth-duration confirmation.
    confidence_threshold : float
        Minimum LSTM softmax confidence to accept prediction.
    alert_depth_cm : float
        Depth above which the 'alert' flag is raised.

    Public interface
    ----------------
    result = det.feed(distance_cm, strength=3000.0)
        → dict (or None until the LSTM window is filled)
    det.baseline   → float : current adaptive baseline
    det.latest_result
    det.alert      → bool
    det.reset()    → clears all state
    """

    def __init__(self,
                 use_model:            bool  = False,
                 ma_window:            int   = ADAPT_MA_WINDOW,
                 hp_alpha:             float = ADAPT_HP_ALPHA,
                 min_duration:         int   = ADAPT_MIN_DURATION,
                 confidence_threshold: float = RT_CONFIDENCE_THRESHOLD,
                 alert_depth_cm:       float = RT_ALERT_DEPTH_CM):

        self._use_model     = use_model
        self._conf_thresh   = confidence_threshold
        self._alert_depth   = alert_depth_cm

        # Stage objects
        self._ma   = _MovingAverageBaseline(ma_window)
        self._hp   = _HighPassFilter(hp_alpha)
        self._drv  = _DerivativeChecker()
        self._dur  = _DepthDurationGuard(min_duration=min_duration)
        self._fbuf = _FeatureBuffer()

        # History for derivative pattern detection
        self._vel_history: deque[float] = deque(maxlen=10)

        self.latest_result: dict | None = None
        self.alert: bool                = False
        self._lock = threading.Lock()

        if use_model:
            _load_model()

    # ── public API ─────────────────────────────────────────────────────────────

    @property
    def baseline(self) -> float:
        return self._ma.baseline

    def feed(self, distance_cm: float, strength: float = 3000.0) -> dict | None:
        """
        Ingest one LiDAR reading.

        Returns
        -------
        dict   – inference result once enough data is buffered
        None   – if the baseline window / LSTM window is not yet full
        """
        with self._lock:
            return self._process(float(distance_cm), float(strength))

    def reset(self):
        """Clear all internal state (call after sensor reconnect)."""
        with self._lock:
            self._ma   = _MovingAverageBaseline(self._ma._buf.maxlen)
            self._hp   = _HighPassFilter(self._hp._alpha)
            self._drv  = _DerivativeChecker()
            self._dur  = _DepthDurationGuard()
            self._fbuf = _FeatureBuffer()
            self._vel_history.clear()
            self.latest_result = None
            self.alert         = False

    # ── internal pipeline ──────────────────────────────────────────────────────

    def _process(self, dist: float, strength: float) -> dict | None:
        # Stage 1 – Moving Average Baseline
        baseline = self._ma.update(dist)
        ma_dev   = dist - baseline

        # Stage 2 – High-Pass Filter
        hp = self._hp.update(dist)

        # Stage 3 – Derivative check
        vel, spike_up, spike_down = self._drv.update(dist)
        self._vel_history.append(vel)

        # Stage 4 – Depth-Duration guard  (uses absolute deviation for both
        # potholes and bumps)
        abs_dev = abs(ma_dev)
        confirmed, run_len = self._dur.update(abs_dev)

        # Push into feature buffer
        above_flag = 1.0 if ma_dev > POTHOLE_THRESH else 0.0
        self._fbuf.push(dist, strength, ma_dev, hp, vel, above_flag)

        # We need the baseline window filled before we trust ma_dev
        if not self._ma.ready:
            return None

        # ── Classify ─────────────────────────────────────────────────────────
        if self._use_model and self._fbuf.ready:
            result = self._model_infer(
                dist, strength, ma_dev, hp, vel,
                baseline, confirmed, run_len, spike_up, spike_down
            )
        else:
            # Rule-based (works before LSTM window is full too)
            class_id, confidence = _rule_based_classify(
                ma_dev, vel, hp, confirmed, run_len
            )
            result = self._build_result(
                dist, strength, ma_dev, baseline, vel,
                class_id, confidence, spike_up, spike_down,
                confirmed, run_len
            )

        # ── Sanity guard ─────────────────────────────────────────────────────
        # Override model/rule classification if the actual deviation is too
        # small — prevents LiDAR noise from being classified as anomalies.
        cls = result["class_id"]
        if cls in (1, 2) and ma_dev < POTHOLE_THRESH:
            result["class_id"]   = 0
            result["class_name"] = CLASS_NAMES[0]
            result["confidence"] = round(max(0.0, 1.0 - abs_dev / POTHOLE_THRESH), 4)
        elif cls == 3 and ma_dev > -BUMP_THRESH:
            result["class_id"]   = 0
            result["class_name"] = CLASS_NAMES[0]
            result["confidence"] = round(max(0.0, 1.0 - abs_dev / BUMP_THRESH), 4)

        self.latest_result = result
        self.alert = (
            result["class_id"] in (1, 2) and
            result["depth_cm"] >= self._alert_depth
        )
        result["alert"] = self.alert
        return result

    def _model_infer(self,
                     dist, strength, ma_dev, hp, vel,
                     baseline, confirmed, run_len,
                     spike_up, spike_down) -> dict:
        """Run the LSTM on the current 6-feature window."""
        window = self._fbuf.as_array()          # (WINDOW_SIZE, 6)

        # Determine how many features the model actually expects
        model_n_features = _model.input_shape[-1]   # e.g. 3 or 6

        # Normalise using saved scaler.
        # Scaler was fitted on 3-feature data; we handle mismatch gracefully:
        # if scaler has 3 features we scale only the first 3 columns.
        try:
            if _scaler.n_features_in_ == ADAPT_N_FEATURES:
                flat   = window.reshape(-1, ADAPT_N_FEATURES)
                flat   = _scaler.transform(flat).astype(np.float32)
                window = flat.reshape(WINDOW_SIZE, ADAPT_N_FEATURES)
            else:
                # Scaler was trained on 3-feature data – scale 3-feature slice
                flat3   = window[:, :3].reshape(-1, 3)
                flat3   = _scaler.transform(flat3).astype(np.float32)
                window[:, :3] = flat3.reshape(WINDOW_SIZE, 3)
        except Exception:
            pass

        # Slice to match the model's expected feature count
        if model_n_features < window.shape[1]:
            window = window[:, :model_n_features]

        tensor = window.reshape(1, WINDOW_SIZE, model_n_features)
        probs      = _model.predict(tensor, verbose=0)[0]   # (4,)
        class_id   = int(np.argmax(probs))
        confidence = float(probs[class_id])

        if confidence < self._conf_thresh:
            class_id   = 0
            confidence = float(probs[0])

        return self._build_result(
            dist, strength, ma_dev, baseline, vel,
            class_id, confidence, spike_up, spike_down,
            confirmed, run_len, probs=probs
        )

    def _build_result(self,
                      dist, strength, ma_dev, baseline, vel,
                      class_id, confidence,
                      spike_up, spike_down,
                      confirmed, run_len,
                      probs=None) -> dict:
        """Assemble the standardised result dict."""
        depth_cm = round(max(0.0, ma_dev), 1)
        severity = _severity_label(depth_cm)

        # Velocity pattern: look for historical spike-up then spike-down
        vel_arr = np.array(self._vel_history)
        has_up_down = (
            (vel_arr > ADAPT_VEL_UP_THRESH).any() and
            (vel_arr < ADAPT_VEL_DOWN_THRESH).any()
        )

        r = {
            "timestamp"       : time.time(),
            "class_id"        : class_id,
            "class_name"      : CLASS_NAMES[class_id],
            "confidence"      : round(confidence, 4),
            "depth_cm"        : depth_cm,
            "severity"        : severity,
            "baseline_cm"     : round(baseline, 1),
            "ma_deviation_cm" : round(ma_dev, 2),
            "hp_signal"       : round(float(
                                    np.array(self._vel_history).mean()
                                    if len(self._vel_history) else 0.0), 2),
            "velocity_cm"     : round(vel, 2),
            "spike_up"        : spike_up,
            "spike_down"      : spike_down,
            "vel_pattern"     : has_up_down,
            "duration_run"    : run_len,
            "duration_confirmed": confirmed,
            "alert"           : False,   # filled by caller
        }
        if probs is not None:
            r["probs"] = {CLASS_NAMES[i]: round(float(probs[i]), 4)
                          for i in range(N_CLASSES)}
        return r


def _severity_label(depth_cm: float) -> str:
    for label, (lo, hi) in SEVERITY_LEVELS.items():
        if lo <= depth_cm < hi:
            return label
    return "Deep/Dangerous"


# ═══════════════════════════════════════════════════════════════════════════════
# Console renderer
# ═══════════════════════════════════════════════════════════════════════════════

ANSI = {
    "reset"  : "\033[0m",  "bold"  : "\033[1m",
    "red"    : "\033[91m", "yellow": "\033[93m",
    "green"  : "\033[92m", "cyan"  : "\033[96m",
    "blue"   : "\033[94m", "grey"  : "\033[90m",
    "white"  : "\033[97m", "magenta": "\033[95m",
}

_CLS_COLOR = {0: "green", 1: "yellow", 2: "red", 3: "cyan"}


def _render_bar(v: float, max_v: float = 1.0, width: int = 20) -> str:
    filled = int(min(v / max(max_v, 1e-9), 1.0) * width)
    return "█" * filled + "░" * (width - filled)


def _print_result(r: dict, n: int):
    c    = ANSI[_CLS_COLOR.get(r["class_id"], "white")]
    rst  = ANSI["reset"]; bold = ANSI["bold"]
    grey = ANSI["grey"];  red  = ANSI["red"]
    mag  = ANSI["magenta"]

    alert_tag = f"  {red}{bold}⚠  POTHOLE ALERT!{rst}" if r["alert"] else ""
    dur_tag   = f"  {mag}✔ confirmed({r['duration_run']}){rst}" if r["duration_confirmed"] else ""
    vel_tag   = f"  {ANSI['blue']}↑↓ vel-pattern{rst}" if r["vel_pattern"] else ""
    bar       = _render_bar(r["confidence"])

    print(
        f"\r{grey}[#{n:05d}]{rst}  "
        f"{c}{bold}{r['class_name']:<18}{rst}  "
        f"{grey}conf {rst}{bar} {r['confidence']*100:5.1f}%  "
        f"base={r['baseline_cm']:6.1f}cm  dev={r['ma_deviation_cm']:+6.2f}cm  "
        f"depth={r['depth_cm']:5.1f}cm  sev={r['severity']:<15}"
        f"{dur_tag}{vel_tag}{alert_tag}",
        end="", flush=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Matplotlib live plot
# ═══════════════════════════════════════════════════════════════════════════════

def _run_live_plot(rq: queue.Queue, stop: threading.Event, max_pts: int = 300):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import Patch

    PALETTE = {0: "#2ea043", 1: "#f0e030", 2: "#f85149", 3: "#58a6ff"}
    LABELS  = {0: "Flat Road", 1: "Shallow Ph", 2: "Deep Ph", 3: "Speed Bump"}

    from collections import deque
    dists    = deque(maxlen=max_pts)
    baselines= deque(maxlen=max_pts)
    ma_devs  = deque(maxlen=max_pts)
    vels     = deque(maxlen=max_pts)
    classes  = deque(maxlen=max_pts)
    confs    = deque(maxlen=max_pts)
    dur_conf = deque(maxlen=max_pts)

    plt.style.use("dark_background")
    fig, axes = plt.subplots(4, 1, figsize=(15, 10),
                             gridspec_kw={"height_ratios": [3, 2, 1.5, 1]})
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
    ax1, ax2, ax3, ax4 = axes

    fig.suptitle(
        "🧠  Adaptive Baseline + DL Pothole Detector  —  1D LiDAR",
        fontsize=15, fontweight="bold", color="white", y=0.99,
    )

    def _upd(_):
        while True:
            try:
                r = rq.get_nowait()
                dists.append(r["baseline_cm"] + r["ma_deviation_cm"])
                baselines.append(r["baseline_cm"])
                ma_devs.append(r["ma_deviation_cm"])
                vels.append(r["velocity_cm"])
                classes.append(r["class_id"])
                confs.append(r["confidence"])
                dur_conf.append(1 if r["duration_confirmed"] else 0)
            except queue.Empty:
                break

        if len(dists) < 2:
            return

        xs = list(range(len(dists)))
        cs = [PALETTE[c] for c in classes]

        # ax1: distance + baseline overlay
        ax1.clear(); ax1.set_facecolor("#161b22")
        ax1.plot(xs, list(dists),    color="#8b949e", lw=1.2, alpha=0.6, label="Distance")
        ax1.plot(xs, list(baselines),color="#58a6ff", lw=1.5, alpha=0.8, label="Adaptive Baseline", linestyle="--")
        ax1.scatter(xs, list(dists), c=cs, s=12, zorder=5)
        ax1.set_ylabel("Distance (cm)", color="#8b949e")
        ax1.tick_params(colors="#8b949e")
        legend_handles = [Patch(color=PALETTE[i], label=LABELS[i]) for i in range(4)]
        legend_handles.append(Patch(color="#58a6ff", label="Baseline (MA)"))
        ax1.legend(handles=legend_handles, loc="upper left",
                   facecolor="#21262d", labelcolor="white", fontsize=8)
        ax1.set_xlim(0, max(max_pts, len(xs)))
        ax1.axhline(0, color="#30363d", lw=0.5)

        # ax2: MA deviation + threshold lines
        ax2.clear(); ax2.set_facecolor("#161b22")
        devs_l = list(ma_devs)
        ax2.fill_between(xs, devs_l, 0,
                         where=[d > 0 for d in devs_l],
                         color="#f85149", alpha=0.55, label="+ deviation (pothole)")
        ax2.fill_between(xs, devs_l, 0,
                         where=[d < 0 for d in devs_l],
                         color="#3fb950", alpha=0.55, label="− deviation (bump)")
        ax2.axhline( POTHOLE_THRESH, color="#f0e030", lw=1, linestyle="--",
                     alpha=0.7, label=f"Pothole thresh ({POTHOLE_THRESH} cm)")
        ax2.axhline(-BUMP_THRESH,    color="#3fb950", lw=1, linestyle="--",
                     alpha=0.7, label=f"Bump thresh (−{BUMP_THRESH} cm)")
        ax2.set_ylabel("MA Deviation (cm)", color="#8b949e")
        ax2.tick_params(colors="#8b949e")
        ax2.legend(facecolor="#21262d", labelcolor="white", fontsize=8)
        ax2.set_xlim(0, max(max_pts, len(xs)))

        # ax3: velocity
        ax3.clear(); ax3.set_facecolor("#161b22")
        vl = list(vels)
        ax3.plot(xs, vl, color="#d2a8ff", lw=1.2, label="Velocity (cm/reading)")
        ax3.axhline( ADAPT_VEL_UP_THRESH,   color="#f85149", lw=1, linestyle=":",
                     alpha=0.8, label=f"Up thresh (+{ADAPT_VEL_UP_THRESH})")
        ax3.axhline( ADAPT_VEL_DOWN_THRESH, color="#3fb950", lw=1, linestyle=":",
                     alpha=0.8, label=f"Down thresh ({ADAPT_VEL_DOWN_THRESH})")
        ax3.set_ylabel("Velocity (cm)", color="#8b949e")
        ax3.tick_params(colors="#8b949e")
        ax3.legend(facecolor="#21262d", labelcolor="white", fontsize=8)
        ax3.set_xlim(0, max(max_pts, len(xs)))

        # ax4: confidence + duration-confirmed shading
        ax4.clear(); ax4.set_facecolor("#161b22")
        ax4.plot(xs, list(confs), color="#58a6ff", lw=1.5, label="Confidence")
        ax4.fill_between(xs, list(dur_conf), 0,
                         color="#f0e030", alpha=0.3, label="Duration confirmed")
        ax4.axhline(RT_CONFIDENCE_THRESHOLD, color="#f0e030", lw=1, linestyle="--",
                    alpha=0.6, label=f"Min conf. ({RT_CONFIDENCE_THRESHOLD:.0%})")
        ax4.set_ylim(0, 1.05)
        ax4.set_ylabel("Confidence", color="#8b949e")
        ax4.set_xlabel("Reading #",  color="#8b949e")
        ax4.tick_params(colors="#8b949e")
        ax4.legend(facecolor="#21262d", labelcolor="white", fontsize=8)
        ax4.set_xlim(0, max(max_pts, len(xs)))

        plt.tight_layout(rect=[0, 0, 1, 0.98])

    ani = animation.FuncAnimation(fig, _upd, interval=150, cache_frame_data=False)
    plt.show()
    stop.set()


# ═══════════════════════════════════════════════════════════════════════════════
# Simulation / Demo
# ═══════════════════════════════════════════════════════════════════════════════

def _simulate_stream(baseline_cm: float = 300.0, hz: float = 10.0):
    """
    Yield (distance_cm, strength) at `hz` Hz, simulating a realistic road.
    Includes:
      – uphill / downhill slope (tests that adaptive baseline tracks it)
      – shallow and deep potholes
      – speed bumps
      – single-spike noise (should NOT trigger detection)
    """
    rng   = np.random.default_rng(42)
    noise = max(1.0, baseline_cm * 0.004)
    dt    = 1.0 / hz

    segments = [
        # (kind, n_readings, depth_cm, slope_cm_per_reading)
        ("flat",    50,  0,    0.0),      # flat approach
        ("slope",   30,  0,    0.5),      # road tilts up  (baseline drifts up)
        ("flat",    20,  0,    0.0),
        ("noise",    1, 15,    0.0),      # SINGLE spike – should NOT confirm
        ("flat",    15,  0,    0.0),
        ("pothole", 10,  6.0,  0.0),     # shallow pothole
        ("flat",    30,  0,    0.0),
        ("slope",   20,  0,   -0.8),     # road tilts back down
        ("flat",    20,  0,    0.0),
        ("pothole", 12, 22.0,  0.0),    # deep pothole
        ("flat",    35,  0,    0.0),
        ("bump",     8, 14.0,  0.0),    # speed bump
        ("flat",    40,  0,    0.0),
        ("pothole", 14, 48.0,  0.0),    # very deep pothole
        ("flat",    50,  0,    0.0),
    ]

    slope_offset = 0.0
    while True:
        for seg in segments:
            kind   = seg[0]
            count  = seg[1]
            depth  = seg[2]
            slope  = seg[3]
            for i in range(count):
                d = float(rng.normal(baseline_cm + slope_offset, noise))
                slope_offset += slope

                if kind == "pothole" and 2 <= i < count - 2:
                    d += depth * rng.uniform(0.85, 1.15)
                elif kind == "bump" and 2 <= i < count - 2:
                    d -= depth * rng.uniform(0.85, 1.15)
                    d  = max(d, 10)
                elif kind == "noise":
                    d += depth * rng.uniform(0.9, 1.1)

                s = float(rng.uniform(2200, 4000) * max(0.1, 1 - baseline_cm / 1500))
                yield d, s
                time.sleep(dt)


def run_demo(port: str | None = None, baud: int = 115200,
             baseline_cm: float = 300.0, hz: float = 10.0,
             use_model: bool = False, show_plot: bool = True):
    """
    Run the adaptive detector.
      port=None  → simulation  (no hardware needed)
      port       → real serial LiDAR
    """
    mode_str = "SIMULATION" if port is None else f"SERIAL {port}"
    model_str = "LSTM Model" if use_model else "Rule-Based"

    print("\n" + "=" * 70)
    print("  🧠  Adaptive 1D LiDAR Pothole Detector")
    print(f"  Mode      : {mode_str}")
    print(f"  Classifier: {model_str}")
    print(f"  Baseline  : moving-average over {ADAPT_MA_WINDOW} readings")
    print(f"  HP alpha  : {ADAPT_HP_ALPHA}")
    print(f"  Duration  : ≥{ADAPT_MIN_DURATION} readings to confirm")
    print(f"  Rate      : {hz} Hz")
    print("  Press Ctrl+C to stop.")
    print("=" * 70 + "\n")

    det        = AdaptiveDetector(use_model=use_model)
    rq         = queue.Queue(maxsize=500)
    stop_event = threading.Event()

    if show_plot:
        t = threading.Thread(target=_run_live_plot,
                             args=(rq, stop_event), daemon=True)
        t.start()

    # Data source
    if port is not None:
        try:
            import serial
            ser = serial.Serial(port, baud, timeout=1)
            def _serial_gen():
                buf = b""
                while not stop_event.is_set():
                    buf += ser.read(16)
                    while len(buf) >= 9:
                        if buf[0] == 0x59 and buf[1] == 0x59:
                            dist = (buf[3] << 8 | buf[2])
                            strn = (buf[5] << 8 | buf[4])
                            yield float(dist), float(strn)
                            buf  = buf[9:]
                        else:
                            buf = buf[1:]
            stream = _serial_gen()
        except Exception as e:
            print(f"  ⚠  Serial error: {e} – falling back to simulation")
            stream = _simulate_stream(baseline_cm, hz)
    else:
        stream = _simulate_stream(baseline_cm, hz)

    count = 0
    try:
        for dist, strength in stream:
            if stop_event.is_set():
                break
            result = det.feed(dist, strength)
            count += 1
            if result is not None:
                _print_result(result, count)
                try:
                    rq.put_nowait(result)
                except queue.Full:
                    pass
    except KeyboardInterrupt:
        print("\n\n  Stopped by user.")
    finally:
        stop_event.set()

    print(f"\n\n  [Summary]  Total readings: {count}")
    if det.latest_result:
        r = det.latest_result
        print(f"  Last class  : {r['class_name']}  (conf={r['confidence']:.2%})")
        print(f"  Last depth  : {r['depth_cm']} cm  ({r['severity']})")
        print(f"  Last baseline: {r['baseline_cm']} cm  (adaptive)")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Adaptive 1D LiDAR Pothole Detector")
    p.add_argument("--demo",      action="store_true",
                   help="Simulation mode (no hardware needed)")
    p.add_argument("--port",      default=None,
                   help="Serial port, e.g. /dev/ttyUSB0 or COM3")
    p.add_argument("--baud",      type=int,   default=115200)
    p.add_argument("--baseline",  type=float, default=300.0,
                   help="Initial baseline hint (cm); auto-adapts immediately")
    p.add_argument("--hz",        type=float, default=10.0,
                   help="LiDAR sample rate Hz (simulation only)")
    p.add_argument("--use-model", action="store_true",
                   help="Use trained LSTM model instead of rule-based logic")
    p.add_argument("--no-plot",   action="store_true",
                   help="Disable live matplotlib plot")
    args = p.parse_args()

    if not args.demo and args.port is None:
        print("No --port specified; running in simulation mode.")

    run_demo(
        port       = args.port,
        baud       = args.baud,
        baseline_cm= args.baseline,
        hz         = args.hz,
        use_model  = args.use_model,
        show_plot  = not args.no_plot,
    )
