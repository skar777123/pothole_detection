"""
dashboard.py
============
Real-time Pothole Detection Dashboard — Single-Point LiDAR (TF02-Pro)

Detection Concept
─────────────────
  The LiDAR points straight DOWN at the road.

  Baseline = expected distance to flat road (default 180 cm, tunable in sidebar).

  • Reading > baseline + POTHOLE_THRESH  →  ground is FARTHER → POTHOLE
  • Reading < baseline - BUMP_THRESH     →  ground is CLOSER  → SPEED BUMP
  • Otherwise                            →  FLAT ROAD

  Each reading goes through:
    1. LiDAR driver   : checksum + distance-range gate
    2. Median filter  : median of N_MEDIAN consecutive frames (noise reduction)
    3. Deviation check: compare against runtime-calibrated baseline
    4. ML classifier  : 22-feature RandomForest (confirms the deviation)
    5. Streak gate    : CONFIRM_FRAMES consecutive positives before alert
    6. Dimension calc : depth, length, width from the confirmed window
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
import logging
from collections import deque

from lidar_driver import TF02Pro, LiDARReadError
from model_train import extract_features, WINDOW_SIZE, POTHOLE_THRESH, BUMP_THRESH

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("dashboard")

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="LiDAR Pothole Detector",
    page_icon="🕳️",
)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_LABELS = {
    0: "🟢 Flat Road",
    1: "🟡 Shallow Pothole",
    2: "🔴 Deep Pothole",
    3: "🔶 Speed Bump",
}
IS_POTHOLE = {0: False, 1: True, 2: True, 3: False}

FRAME_RATE_HZ = 100          # TF02-Pro output rate

# Severity bands (depth in cm from baseline)
SEVERITY_BANDS = [
    (0,  3,   "None (noise)"),
    (3,  8,   "⚠️  Shallow"),
    (8,  15,  "🔶 Moderate"),
    (15, 999, "🔴 Deep / Dangerous"),
]


# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading pothole model …")
def load_model():
    try:
        m = joblib.load("pothole_model.pkl")
        logger.info("Model loaded from pothole_model.pkl")
        return m
    except FileNotFoundError:
        return None


# ── Session-state initialisation ──────────────────────────────────────────────
_state_defaults = {
    "dist_history"  : deque(maxlen=400),
    "str_history"   : deque(maxlen=400),
    "baseline_hist" : deque(maxlen=400),   # Running baseline for chart overlay
    "pothole_count" : 0,
    "bump_count"    : 0,
    "pothole_log"   : [],
    "running"       : False,
    "confirm_streak": 0,
    "last_label"    : "—",
    "last_depth"    : 0.0,
    "last_length"   : 0.0,
    "last_width"    : 0.0,
    "last_strength" : 0,
    "calib_readings": [],    # Readings during calibration phase
    "baseline_cm"   : 180.0, # Overwritten once calibrated
    "calibrated"    : False,
}
for k, v in _state_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Sidebar configuration ─────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Sensor Settings")

    lidar_port = st.text_input("Serial Port", value="/dev/ttyUSB0",
                               help="e.g. /dev/ttyUSB0 or COM3")
    lidar_baud = st.selectbox("Baud Rate", [115200, 9600], index=0)
    n_median   = st.slider("Median filter (frames)", 1, 7, 3,
                           help="Consecutive frames per reading; trades latency for noise")
    confirm_n  = st.slider("Confirm streak", 1, 5, 3,
                           help="Consecutive windows that must predict positive before alert")

    st.markdown("---")
    st.subheader("📏 Baseline")
    manual_baseline = st.number_input(
        "Default road distance (cm)", min_value=30, max_value=2000,
        value=180, step=5,
        help="Distance from sensor to flat road surface. "
             "Overridden automatically during calibration.",
    )
    calib_n = st.slider("Calibration samples", 10, 100, 30,
                        help="Readings averaged to compute baseline on-the-fly")

    st.markdown("---")
    st.subheader("🚗 Vehicle")
    speed_kmph = st.number_input("Vehicle speed (km/h)", 5, 120, 30, step=5)

    st.markdown("---")
    st.subheader("📊 Classes")
    for k, v in CLASS_LABELS.items():
        st.markdown(f"- `{k}` → {v}")


# ── Derived parameters ────────────────────────────────────────────────────────
speed_cm_s      = (speed_kmph * 1000 * 100) / 3600
dist_per_reading = speed_cm_s / FRAME_RATE_HZ   # cm of road per LiDAR reading


# ── Helper: severity label ────────────────────────────────────────────────────
def severity_label(depth_cm: float) -> str:
    for lo, hi, label in SEVERITY_BANDS:
        if lo <= depth_cm < hi:
            return label
    return "—"


# ── Helper: compute pothole dimensions ───────────────────────────────────────
def compute_dimensions(dist_buf: list, str_buf: list, baseline: float) -> dict:
    """
    Given a confirmed detection window, compute physical dimensions.

    depth  = max deviation above baseline  (cm)
    length = distance travelled across     (cm, based on vehicle speed)
    width  = estimated cross-width         (cm, geometric estimate)
    """
    arr      = np.array(dist_buf, dtype=float)
    dev      = arr - baseline

    # Depth = peak positive deviation (into pothole)
    depth_cm = float(max(dev.max(), 0))

    # Length = count of points with deviation > POTHOLE_THRESH × cm per reading
    points_in_hole = int(np.sum(dev > POTHOLE_THRESH))
    length_cm      = round(points_in_hole * dist_per_reading, 1)

    # Width estimate (potholes are roughly elliptical; conservative 0.8 factor)
    width_cm = round(length_cm * 0.80, 1)

    avg_str = float(np.mean(str_buf)) if str_buf else 0.0

    return {
        "depth_cm"   : round(depth_cm, 1),
        "length_cm"  : length_cm,
        "width_cm"   : width_cm,
        "severity"   : severity_label(depth_cm),
        "avg_strength": round(avg_str, 0),
    }


# ── Simple rule-based pre-classifier (runs BEFORE ML) ────────────────────────
def rule_classify(dist_cm: float, baseline: float) -> int | None:
    """
    Fast single-reading rule based on the user's baseline-deviation concept.

    Returns class int or None if the reading is within the noise band.

    Class 0 = flat road
    Class 1/2 = pothole (threshold determines shallow vs deep — ML resolves this)
    Class 3 = bump
    """
    dev = dist_cm - baseline
    if dev > POTHOLE_THRESH:
        return 1   # Possible pothole (ML will refine to 1 or 2)
    elif dev < -BUMP_THRESH:
        return 3   # Speed bump
    else:
        return 0   # Flat road


# ── Main detection loop ───────────────────────────────────────────────────────
def run_detection(model):
    st.markdown("---")
    stop_col, status_col = st.columns([1, 5])
    stop_btn = stop_col.button("⏹ Stop", type="secondary", key="stop_btn")

    # ── Open LiDAR ───────────────────────────────────────────────────────────
    try:
        lidar = TF02Pro(port=lidar_port, baudrate=lidar_baud)
    except Exception as exc:
        st.error(f"❌ Cannot open `{lidar_port}`: {exc}")
        logger.error("LiDAR open error: %s", exc)
        st.session_state.running = False
        return

    status_col.success(f"✅ LiDAR connected → `{lidar_port}` @ {lidar_baud} baud")

    # ── KPI placeholders ─────────────────────────────────────────────────────
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    ph_count   = k1.empty()
    ph_bump    = k2.empty()
    ph_depth   = k3.empty()
    ph_length  = k4.empty()
    ph_width   = k5.empty()
    ph_str     = k6.empty()

    st.markdown("---")

    chart_header = st.empty()
    chart_dist   = st.empty()
    chart_str    = st.empty()

    st.markdown("---")
    status_ph = st.empty()
    calib_ph  = st.empty()

    st.markdown("---")
    log_ph = st.empty()

    # ── Buffers ───────────────────────────────────────────────────────────────
    dist_buf = []
    str_buf  = []
    consec_err = 0

    logger.info("Detection loop started. Baseline=%.1f cm", st.session_state.baseline_cm)

    while not stop_btn:

        # ── READ ─────────────────────────────────────────────────────────────
        try:
            reading   = lidar.read_median(samples=n_median)
            consec_err = 0
        except LiDARReadError as exc:
            consec_err += 1
            logger.warning("Read error #%d: %s", consec_err, exc)
            if consec_err > 30:
                st.error("❌ Too many consecutive read errors. Check hardware.")
                break
            time.sleep(0.05)
            continue

        dist     = reading["distance_cm"]
        strength = reading["strength"]

        # ── BASELINE AUTO-CALIBRATION ─────────────────────────────────────
        if not st.session_state.calibrated:
            st.session_state.calib_readings.append(dist)
            remaining = calib_n - len(st.session_state.calib_readings)
            calib_ph.info(
                f"🔵 **Calibrating baseline** … collecting {remaining} more "
                f"flat-road readings. Keep sensor over flat ground."
            )
            if len(st.session_state.calib_readings) >= calib_n:
                # Use median of calibration readings as baseline (robust to outliers)
                cal = sorted(st.session_state.calib_readings)
                st.session_state.baseline_cm = float(cal[len(cal) // 2])
                st.session_state.calibrated  = True
                calib_ph.success(
                    f"✅ Baseline calibrated: **{st.session_state.baseline_cm:.1f} cm**"
                )
                logger.info("Baseline calibrated: %.1f cm", st.session_state.baseline_cm)
            continue   # Don't classify during calibration

        baseline = st.session_state.baseline_cm

        # ── UPDATE HISTORY ────────────────────────────────────────────────
        st.session_state.dist_history.append(dist)
        st.session_state.str_history.append(strength)
        st.session_state.baseline_hist.append(baseline)

        # ── SLIDING WINDOW ────────────────────────────────────────────────
        dist_buf.append(dist)
        str_buf.append(strength)
        if len(dist_buf) > WINDOW_SIZE:
            dist_buf.pop(0)
            str_buf.pop(0)

        # ── CLASSIFY (only when window is full) ───────────────────────────
        if len(dist_buf) == WINDOW_SIZE:
            # Step A: rule-based pre-screen on current reading only
            rule_cls = rule_classify(dist, baseline)

            # Step B: ML on the full window (22 deviation features)
            feats      = extract_features(
                np.array(dist_buf), np.array(str_buf), baseline
            ).reshape(1, -1)
            ml_cls     = int(model.predict(feats)[0])
            ml_proba   = model.predict_proba(feats)[0]
            ml_conf    = float(ml_proba[ml_cls])

            # Agreement: use ML class if confident; use rule as tie-breaker
            if ml_conf >= 0.60:
                final_cls = ml_cls
            else:
                final_cls = rule_cls

            label = CLASS_LABELS[final_cls]
            is_ph = IS_POTHOLE.get(final_cls, False)

            # Step C: Streak accumulator
            if is_ph:
                st.session_state.confirm_streak += 1
            else:
                st.session_state.confirm_streak = 0

            confirmed = st.session_state.confirm_streak >= confirm_n

            # ── CONFIRMED EVENT ───────────────────────────────────────────
            if confirmed:
                dims = compute_dimensions(dist_buf, str_buf, baseline)
                st.session_state.pothole_count += 1
                st.session_state.confirm_streak = 0

                st.session_state.last_label    = label
                st.session_state.last_depth    = dims["depth_cm"]
                st.session_state.last_length   = dims["length_cm"]
                st.session_state.last_width    = dims["width_cm"]
                st.session_state.last_strength = int(dims["avg_strength"])

                log_entry = {
                    "Time"       : time.strftime("%H:%M:%S"),
                    "Type"       : label,
                    "Depth (cm)" : dims["depth_cm"],
                    "Length (cm)": dims["length_cm"],
                    "Width (cm)" : dims["width_cm"],
                    "Severity"   : dims["severity"],
                    "ML Conf."   : f"{ml_conf:.0%}",
                    "Strength"   : int(dims["avg_strength"]),
                    "Baseline"   : f"{baseline:.0f} cm",
                }
                st.session_state.pothole_log.insert(0, log_entry)

                status_ph.error(
                    f"🔴 **{label} CONFIRMED** — "
                    f"Depth: **{dims['depth_cm']} cm** | "
                    f"Length: **{dims['length_cm']} cm** | "
                    f"Severity: {dims['severity']} | "
                    f"Confidence: {ml_conf:.0%}"
                )
                logger.info("DETECTED: %s", log_entry)

                # Clear buffer to reset for next event
                dist_buf.clear()
                str_buf.clear()

            else:
                # Real-time status when no event confirmed
                dev = dist - baseline
                dev_str = f"+{dev:.1f}" if dev >= 0 else f"{dev:.1f}"
                streak_bar = "█" * st.session_state.confirm_streak + \
                             "░" * (confirm_n - st.session_state.confirm_streak)

                if final_cls == 0:
                    status_ph.success(
                        f"🟢 **Flat Road** — "
                        f"Dist: {dist} cm | Dev: {dev_str} cm | "
                        f"Strength: {strength}"
                    )
                elif is_ph:
                    status_ph.warning(
                        f"🟡 **Possible {label}** — "
                        f"Dist: {dist} cm | Dev: {dev_str} cm | "
                        f"Conf: {ml_conf:.0%} | Streak: [{streak_bar}] {st.session_state.confirm_streak}/{confirm_n}"
                    )
                else:
                    status_ph.info(
                        f"🔶 **{label}** — "
                        f"Dist: {dist} cm | Dev: {dev_str} cm | "
                        f"Conf: {ml_conf:.0%}"
                    )

        # ── UPDATE KPIs ───────────────────────────────────────────────────
        ph_count.metric("🕳️ Potholes",  st.session_state.pothole_count)
        ph_bump.metric("🔶 Bumps",      st.session_state.bump_count)
        ph_depth.metric("📏 Last Depth", f"{st.session_state.last_depth} cm")
        ph_length.metric("↔️ Length",   f"{st.session_state.last_length} cm")
        ph_width.metric("↕️ Width",     f"{st.session_state.last_width} cm")
        ph_str.metric("📶 Strength",    st.session_state.last_strength)

        # ── UPDATE CHARTS ─────────────────────────────────────────────────
        chart_header.markdown("### 📡 Live Distance + Baseline (cm)")
        chart_data = pd.DataFrame({
            "Distance (cm)" : list(st.session_state.dist_history),
            "Baseline (cm)" : list(st.session_state.baseline_hist),
        })
        chart_dist.line_chart(chart_data)

        chart_str.line_chart(
            pd.DataFrame({"Signal Strength": list(st.session_state.str_history)})
        )

        # ── DETECTION LOG ─────────────────────────────────────────────────
        if st.session_state.pothole_log:
            log_ph.subheader("📋 Detection Log")
            log_ph.dataframe(
                pd.DataFrame(st.session_state.pothole_log).head(50),
                width="stretch",     # replaces deprecated use_container_width=True
            )

        time.sleep(0.01)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    lidar.close()
    st.session_state.running = False
    st.info("⏹ Monitoring stopped.")
    logger.info("Detection loop stopped.")


# ── UI entry point ────────────────────────────────────────────────────────────
st.title("🕳️ Pothole Detection Dashboard — Single-Point LiDAR")
st.caption(
    "TF02-Pro · Baseline-Deviation Detection · ML-confirmed · Severity-graded"
)

model = load_model()

if model is None:
    st.error(
        "❌ `pothole_model.pkl` not found.\n\n"
        "Run `python model_train.py` to train the model first."
    )
    st.stop()

# ── Baseline info box ─────────────────────────────────────────────────────────
st.info(
    f"**Detection concept:**  "
    f"Sensor points DOWN at road. Default baseline = **{manual_baseline} cm**.  "
    f"Distance **greater** than baseline → pothole (ground farther).  "
    f"Distance **less** than baseline → speed bump (ground closer).  "
    f"Baseline is **auto-calibrated** from the first {calib_n} readings."
)

# Pre-load the user's manual baseline into session state before calibration
if not st.session_state.calibrated:
    st.session_state.baseline_cm = float(manual_baseline)

if not st.session_state.running:
    if st.button("▶ Start Monitoring", type="primary", use_container_width=True):
        st.session_state.running   = True
        st.session_state.calibrated = False
        st.session_state.calib_readings = []
        st.rerun()

if st.session_state.running:
    run_detection(model)