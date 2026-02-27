"""
realtime_detector.py
====================
Real-time pothole detection engine using the trained DL model.

This module provides:
  • RealtimeDetector  – core class wrapping model inference + ring buffer
  • run_live_demo     – simulates a live LiDAR feed and renders live console output
  • LivePlotter       – matplotlib animation for visual monitoring

It integrates with the existing serial LiDAR driver (lidar_driver.py) so you can
drop this in as a replacement for the RandomForest inference in dashboard.py.

Usage (simulation mode – no hardware required)
───────────────────────────────────────────────
  python realtime_detector.py --demo

Usage (with real LiDAR serial port)
────────────────────────────────────
  python realtime_detector.py --port COM3 --baud 115200

Usage (import as module)
────────────────────────
  from realtime_detector import RealtimeDetector
  det = RealtimeDetector()
  det.feed(distance_cm=312.5, strength=3200)
  result = det.latest_result          # dict or None until window fills
"""

import os, sys, time, argparse, threading, queue
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import joblib

from dl_config import (
    WINDOW_SIZE, N_RAW_FEATURES, N_CLASSES,
    MODEL_SAVE_PATH, SCALER_SAVE_PATH,
    BASELINE_CM, POTHOLE_THRESH, BUMP_THRESH,
    RT_CONFIDENCE_THRESHOLD, RT_ALERT_DEPTH_CM,
    CLASS_NAMES, SEVERITY_LEVELS,
    ADAPT_MA_WINDOW, ADAPT_HP_ALPHA,
    ADAPT_VEL_UP_THRESH, ADAPT_VEL_DOWN_THRESH, ADAPT_MIN_DURATION,
)

# ── Lazy TF import so the module can be loaded quickly ────────────────────────

_model  = None
_scaler = None


def _load_inference_resources():
    global _model, _scaler
    if _model is not None:
        return
    from dl_model import load_model as _lm
    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_SAVE_PATH}.\n"
            "Run  python train.py  first to train and save the model."
        )
    _model  = _lm(MODEL_SAVE_PATH)
    _scaler = joblib.load(SCALER_SAVE_PATH)
    print(f"[RealtimeDetector] Model loaded from {MODEL_SAVE_PATH}")


# ── Severity helper ───────────────────────────────────────────────────────────

def _estimate_depth(raw_readings: np.ndarray, baseline: float) -> float:
    """Estimate peak depth from a raw window."""
    dev = raw_readings - baseline
    return float(max(dev.max(), 0))


def _severity_label(depth_cm: float) -> str:
    for label, (lo, hi) in SEVERITY_LEVELS.items():
        if lo <= depth_cm < hi:
            return label
    return "Deep/Dangerous"


def _estimate_length(deviation_series: np.ndarray,
                     threshold_cm: float = POTHOLE_THRESH,
                     speed_cms: float = 1400.0) -> float:
    """
    Estimate pothole length in cm.
    Counts consecutive positive-deviation readings above threshold and converts
    to distance using vehicle speed (default 50 km/h ≈ 1400 cm/s at 10 Hz).
    """
    above  = deviation_series > threshold_cm
    # Find longest consecutive run
    max_run = best = 0
    for v in above:
        max_run = (max_run + 1) if v else 0
        best    = max(best, max_run)
    # Each reading is ~100ms at 10 Hz from LiDAR at 10 fps
    # (speed_cms * 0.1s * n_readings)
    return float(best * (speed_cms * 0.1))


# ── Core detector class ───────────────────────────────────────────────────────

class RealtimeDetector:
    """
    Sliding-window real-time pothole detector.

    Call  .feed(distance_cm, strength)  every time a new LiDAR reading arrives.
    After WINDOW_SIZE readings have been buffered, inference runs on every call
    using a sliding window (new reading in → oldest reading out).

    Attributes
    ----------
    latest_result : dict | None
        The most recent inference result, or None if window not yet full.
    alert         : bool
        True when the last window detected a pothole above ALERT_DEPTH_CM.
    """

    def __init__(self,
                 baseline_cm: float = BASELINE_CM,
                 confidence_threshold: float = RT_CONFIDENCE_THRESHOLD,
                 alert_depth_cm: float = RT_ALERT_DEPTH_CM,
                 adaptive_baseline: bool = True,
                 ma_window: int = ADAPT_MA_WINDOW):
        _load_inference_resources()

        self.baseline_cm          = baseline_cm
        self.conf_threshold       = confidence_threshold
        self.alert_depth_cm       = alert_depth_cm
        self._adaptive            = adaptive_baseline
        self._ma_window           = ma_window

        # Ring buffers
        self._dist_buf    = np.full(WINDOW_SIZE, baseline_cm, dtype=np.float32)
        self._str_buf     = np.full(WINDOW_SIZE, 3000.0,      dtype=np.float32)
        self._count       = 0

        # Adaptive baseline ring buffer (separate from dist_buf)
        self._ma_buf      = np.full(ma_window, baseline_cm, dtype=np.float32)
        self._ma_ptr      = 0
        self._ma_sum      = float(baseline_cm * ma_window)
        self._ma_ready    = False
        self._ma_count    = 0

        self.latest_result: dict | None = None
        self.alert: bool = False

        self._lock = threading.Lock()

    # ── public API ────────────────────────────────────────────────────────────

    def feed(self, distance_cm: float, strength: float = 3000.0) -> dict | None:
        """
        Ingest one new LiDAR reading and run inference if window is full.

        Returns the result dict or None if the window is not yet filled.
        """
        with self._lock:
            # Shift buffer left, append new reading
            self._dist_buf = np.roll(self._dist_buf, -1)
            self._str_buf  = np.roll(self._str_buf,  -1)
            self._dist_buf[-1] = distance_cm
            self._str_buf[-1]  = strength
            self._count       += 1

            if self._count < WINDOW_SIZE:
                return None   # buffer not full yet

            result = self._infer()
            self.latest_result = result
            self.alert = (
                result["class_id"] in (1, 2) and
                result["depth_cm"] >= self.alert_depth_cm
            )
            return result

    def update_baseline(self, new_baseline_cm: float):
        """Manually override the baseline (only meaningful when adaptive=False)."""
        with self._lock:
            self.baseline_cm = float(new_baseline_cm)

    def reset(self):
        """Clear the ring buffer (e.g. after sensor reconnect)."""
        with self._lock:
            self._dist_buf[:] = self.baseline_cm
            self._str_buf[:]  = 3000.0
            self._count       = 0
            self.latest_result = None
            self.alert         = False
            self._ma_buf[:]   = self.baseline_cm
            self._ma_sum      = float(self.baseline_cm * self._ma_window)
            self._ma_ptr      = 0
            self._ma_ready    = False
            self._ma_count    = 0

    # ── internal inference ────────────────────────────────────────────────────

    def _update_adaptive_baseline(self, dist_cm: float) -> float:
        """Push one reading into the MA ring buffer; return current baseline."""
        oldest          = self._ma_buf[self._ma_ptr]
        self._ma_buf[self._ma_ptr] = dist_cm
        self._ma_ptr    = (self._ma_ptr + 1) % self._ma_window
        self._ma_sum   += dist_cm - oldest
        self._ma_count += 1
        if self._ma_count >= self._ma_window:
            self._ma_ready = True
        return self._ma_sum / self._ma_window

    def _infer(self) -> dict:
        dist   = self._dist_buf.copy()
        strn   = self._str_buf.copy()

        # ── Adaptive baseline: use MA over the last ADAPT_MA_WINDOW readings
        if self._adaptive and self._ma_ready:
            b = self._update_adaptive_baseline(float(dist[-1]))
            # Recompute deviation for every timestep using the single current
            # baseline. For a per-timestep MA you'd need a 2D history; this
            # single-value approach is the best approximation inside the
            # existing 3-feature slot.
        else:
            b = self.baseline_cm
        dev = dist - b

        # Build (1, T, F) tensor
        window = np.stack([dist, strn, dev], axis=-1).astype(np.float32)  # (T, 3)
        # Normalise with saved scaler
        flat   = window.reshape(-1, N_RAW_FEATURES)
        flat   = _scaler.transform(flat).astype(np.float32)
        tensor = flat.reshape(1, WINDOW_SIZE, N_RAW_FEATURES)

        probs      = _model.predict(tensor, verbose=0)[0]  # (4,)
        class_id   = int(np.argmax(probs))
        confidence = float(probs[class_id])

        # Low-confidence → fall back to "Flat Road"
        if confidence < self.conf_threshold:
            class_id   = 0
            confidence = float(probs[0])

        depth_cm   = _estimate_depth(dist, b)
        severity   = _severity_label(depth_cm)
        length_cm  = _estimate_length(dev) if class_id in (1, 2) else 0.0

        return {
            "timestamp"   : time.time(),
            "class_id"    : class_id,
            "class_name"  : CLASS_NAMES[class_id],
            "confidence"  : round(confidence, 4),
            "probs"       : {CLASS_NAMES[i]: round(float(probs[i]), 4)
                             for i in range(N_CLASSES)},
            "depth_cm"    : round(depth_cm, 1),
            "length_cm"   : round(length_cm, 1),
            "severity"    : severity,
            "baseline_cm" : round(b, 1),
            "baseline_mode": "adaptive-MA" if (self._adaptive and self._ma_ready) else "fixed",
            "mean_dist_cm": round(float(dist.mean()), 1),
            "max_dev_cm"  : round(float(dev.max()), 1),
            "alert"       : class_id in (1, 2) and depth_cm >= self.alert_depth_cm,
        }


# ── Live Console Renderer ─────────────────────────────────────────────────────

ANSI = {
    "reset"  : "\033[0m",
    "bold"   : "\033[1m",
    "red"    : "\033[91m",
    "yellow" : "\033[93m",
    "green"  : "\033[92m",
    "cyan"   : "\033[96m",
    "blue"   : "\033[94m",
    "grey"   : "\033[90m",
    "white"  : "\033[97m",
}


def _cls_color(class_id: int) -> str:
    return {0: "green", 1: "yellow", 2: "red", 3: "cyan"}.get(class_id, "white")


def _render_bar(value: float, max_val: float, width: int = 20) -> str:
    filled = int(min(value / max(max_val, 1e-6), 1.0) * width)
    return "█" * filled + "░" * (width - filled)


def _print_result(result: dict, reading_no: int):
    c    = ANSI[_cls_color(result["class_id"])]
    rst  = ANSI["reset"]
    bold = ANSI["bold"]
    grey = ANSI["grey"]
    red  = ANSI["red"]

    alert_tag = (f"  {red}{bold}⚠  POTHOLE ALERT!{rst}"
                 if result["alert"] else "")

    bar = _render_bar(result["confidence"], 1.0, 24)

    print(f"\r{grey}[#{reading_no:05d}]{rst}  "
          f"{c}{bold}{result['class_name']:<18}{rst}  "
          f"{grey}conf {rst}{bar} {result['confidence']*100:5.1f}%"
          f"  depth={result['depth_cm']:5.1f}cm"
          f"  sev={result['severity']:<15}"
          f"{alert_tag}", end="", flush=True)


# ── Matplotlib live plot ──────────────────────────────────────────────────────

def _run_live_plot(result_queue: queue.Queue, stop_event: threading.Event,
                   max_points: int = 200):
    """Run in a separate thread; reads from result_queue, draws animated plot."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import Patch
    from collections import deque

    # Colour palette per class
    PALETTE = {0: "#2ea043", 1: "#f0e030", 2: "#f85149", 3: "#58a6ff"}
    LABELS  = {0: "Flat Road", 1: "Shallow Ph", 2: "Deep Ph", 3: "Speed Bump"}

    dists   = deque(maxlen=max_points)
    classes = deque(maxlen=max_points)
    confs   = deque(maxlen=max_points)
    depths  = deque(maxlen=max_points)

    plt.style.use("dark_background")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 9),
                                         gridspec_kw={"height_ratios": [3, 1, 1]})
    fig.patch.set_facecolor("#0d1117")
    for ax in (ax1, ax2, ax3):
        ax.set_facecolor("#161b22")

    fig.suptitle("🚦  Real-Time Pothole Detector  —  DL Model",
                 fontsize=16, fontweight="bold", color="white", y=0.98)

    def _update(_frame):
        # Drain queue
        while True:
            try:
                r = result_queue.get_nowait()
                dists.append(r["mean_dist_cm"])
                classes.append(r["class_id"])
                confs.append(r["confidence"])
                depths.append(r["depth_cm"])
            except queue.Empty:
                break

        if len(dists) < 2:
            return

        xs = list(range(len(dists)))
        cs = [PALETTE[c] for c in classes]

        # --- ax1: distance + class colouring ---
        ax1.clear(); ax1.set_facecolor("#161b22")
        ax1.plot(xs, list(dists), color="#8b949e", lw=1, alpha=0.5, label="Distance (cm)")
        ax1.scatter(xs, list(dists), c=cs, s=14, zorder=5)
        ax1.set_ylabel("Distance (cm)", color="#8b949e")
        ax1.tick_params(colors="#8b949e")
        legend_handles = [Patch(color=PALETTE[i], label=LABELS[i]) for i in range(4)]
        ax1.legend(handles=legend_handles, loc="upper left",
                   facecolor="#21262d", labelcolor="white", fontsize=9)
        ax1.set_xlim(0, max(max_points, len(xs)))

        # --- ax2: depth ---
        ax2.clear(); ax2.set_facecolor("#161b22")
        ax2.fill_between(xs, list(depths), color="#f85149", alpha=0.6, label="Depth (cm)")
        ax2.set_ylabel("Depth (cm)", color="#8b949e")
        ax2.tick_params(colors="#8b949e")
        ax2.set_xlim(0, max(max_points, len(xs)))

        # --- ax3: confidence ---
        ax3.clear(); ax3.set_facecolor("#161b22")
        ax3.plot(xs, list(confs), color="#58a6ff", lw=1.5, label="Confidence")
        ax3.axhline(RT_CONFIDENCE_THRESHOLD, color="#f0e030", lw=1,
                    linestyle="--", alpha=0.7, label=f"Threshold {RT_CONFIDENCE_THRESHOLD:.0%}")
        ax3.set_ylim(0, 1.05)
        ax3.set_ylabel("Confidence", color="#8b949e")
        ax3.set_xlabel("Reading #",  color="#8b949e")
        ax3.tick_params(colors="#8b949e")
        ax3.legend(facecolor="#21262d", labelcolor="white", fontsize=9)
        ax3.set_xlim(0, max(max_points, len(xs)))

        plt.tight_layout(rect=[0, 0, 1, 0.97])

    ani = animation.FuncAnimation(fig, _update, interval=200, cache_frame_data=False)
    plt.show()
    stop_event.set()


# ── Simulation / Demo ─────────────────────────────────────────────────────────

def _simulate_lidar_stream(baseline_cm: float = 300.0, hz: float = 10.0):
    """
    Yield (distance_cm, strength) at the given rate, simulating:
      – Long flat stretches
      – Shallow potholes
      – Deep potholes
      – Speed bumps
    """
    rng = np.random.default_rng(0)
    noise = max(1.0, baseline_cm * 0.004)

    segments = [
        ("flat",    40),
        ("pothole", 8,  6.0),
        ("flat",    30),
        ("pothole", 10, 18.0),
        ("flat",    20),
        ("bump",    8,  12.0),
        ("flat",    25),
        ("pothole", 12, 45.0),
        ("flat",    50),
        ("bump",    10, 25.0),
    ]

    period = 1.0 / hz
    idx    = 0
    while True:
        for seg in segments:
            kind  = seg[0]
            count = seg[1]
            depth = seg[2] if len(seg) > 2 else 0.0

            for i in range(count):
                d = float(rng.normal(baseline_cm, noise))
                if kind == "pothole" and 2 <= i < count - 2:
                    d += depth * rng.uniform(0.8, 1.2)
                elif kind == "bump"    and 2 <= i < count - 2:
                    d -= depth * rng.uniform(0.8, 1.2)
                    d  = max(d, 10)

                s = float(rng.uniform(2500, 4000) * max(0.1, 1 - baseline_cm / 1500))
                yield d, s
                idx += 1
                time.sleep(period)


def run_live_demo(port: str = None, baud: int = 115200,
                  baseline_cm: float = 300.0, hz: float = 10.0,
                  show_plot: bool = True):
    """
    Run the real-time detector:
      - port=None  → simulation mode (no hardware needed)
      - port="COM3" → reads real LiDAR over serial
    """
    print("\n" + "="*60)
    print("  🚀  Real-Time Pothole Detector  (DL model)")
    print(f"  Mode    : {'SIMULATION' if port is None else f'SERIAL {port}'}")
    print(f"  Baseline: {baseline_cm} cm")
    print(f"  Rate    : {hz} Hz")
    print("  Press Ctrl+C to stop.")
    print("="*60 + "\n")

    detector = RealtimeDetector(baseline_cm=baseline_cm)

    rq         = queue.Queue(maxsize=500)
    stop_event = threading.Event()

    # Launch plot in background thread
    if show_plot:
        plot_thread = threading.Thread(
            target=_run_live_plot,
            args=(rq, stop_event),
            daemon=True
        )
        plot_thread.start()

    # Data source
    if port is not None:
        try:
            import serial
            ser = serial.Serial(port, baud, timeout=1)
            def _read():
                buf = b""
                while not stop_event.is_set():
                    buf += ser.read(16)
                    # Minimal TF-02 Pro frame parse (9 bytes)
                    while len(buf) >= 9:
                        if buf[0] == 0x59 and buf[1] == 0x59:
                            dist  = (buf[3] << 8 | buf[2])
                            strn  = (buf[5] << 8 | buf[4])
                            yield float(dist), float(strn)
                            buf  = buf[9:]
                        else:
                            buf = buf[1:]
            stream = _read()
        except Exception as e:
            print(f"  ⚠  Serial error: {e} – falling back to simulation")
            stream = _simulate_lidar_stream(baseline_cm, hz)
    else:
        stream = _simulate_lidar_stream(baseline_cm, hz)

    count = 0
    try:
        for dist, strength in stream:
            if stop_event.is_set():
                break
            result = detector.feed(dist, strength)
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

    # Summary
    print("\n\n  [Summary] Total readings processed:", count)
    if detector.latest_result:
        print(f"  Last result: {detector.latest_result}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Real-Time Pothole Detector")
    p.add_argument("--demo",      action="store_true", help="Simulation mode (no HW)")
    p.add_argument("--port",      default=None,        help="Serial port, e.g. COM3")
    p.add_argument("--baud",      type=int,  default=115200)
    p.add_argument("--baseline",  type=float,default=300.0,  help="Baseline distance cm")
    p.add_argument("--hz",        type=float,default=10.0,   help="LiDAR sample rate Hz")
    p.add_argument("--no-plot",   action="store_true",       help="Disable live plot")
    args = p.parse_args()

    if not args.demo and args.port is None:
        print("No --port specified; running in simulation mode (--demo).")

    run_live_demo(
        port       = args.port,
        baud       = args.baud,
        baseline_cm= args.baseline,
        hz         = args.hz,
        show_plot  = not args.no_plot,
    )
