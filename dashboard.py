"""
dashboard.py
=============
Real-time Pothole Detection — TF02-Pro Single-Point LiDAR

WHAT WAS CAUSING "DISTANCE STUCK AT 206 cm"
─────────────────────────────────────────────
  The TF02-Pro emits frames at 100 Hz. The OS serial buffer accumulates
  hundreds of old frames. Previous code never flushed the buffer, so every
  read() call returned stale data from the front of the queue — that's why
  the distance never changed even when a hand was placed on the sensor.

  Fix: reset_input_buffer() before every read so we always get the CURRENT
  frame from the sensor.

DETECTION PIPELINE
──────────────────
  1. Read current frame (buffer-flushed, guaranteed fresh)
  2. Compute deviation = distance - baseline
     • deviation > +POTHOLE_THRESH cm  →  POTHOLE  (ground dropped)
     • deviation < -BUMP_THRESH   cm  →  BUMP     (ground raised)
     • else                           →  FLAT ROAD
  3. For POTHOLE: ML window check + streak gate before logging
  4. Show all dimensions + severity in detection log
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
import logging
from collections import deque

from lidar_driver import TF02Pro, LiDARReadError
from model_train import (
    extract_features,
    WINDOW_SIZE,
    POTHOLE_THRESH,
    BUMP_THRESH,
    BASELINE_CM as DEFAULT_BASELINE,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("dashboard")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="LiDAR Pothole Detector",
    page_icon="🕳️",
)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_LABELS = {0: "🟢 Flat Road", 1: "🟡 Shallow Pothole",
                2: "🔴 Deep Pothole", 3: "🔶 Speed Bump"}
IS_POTHOLE   = {0: False, 1: True, 2: True, 3: False}
IS_BUMP      = {0: False, 1: False, 2: False, 3: True}
FRAME_RATE_HZ = 100

SEVERITY_BANDS = [
    (0,   3,   "—  Noise"),
    (3,   8,   "⚠️  Shallow"),
    (8,  15,   "🔶 Moderate"),
    (15, 999,  "🔴 Deep / Dangerous"),
]


# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ML model …")
def load_model():
    try:
        m = joblib.load("pothole_model.pkl")
        logger.info("pothole_model.pkl loaded")
        return m
    except FileNotFoundError:
        return None


# ── Session-state defaults ────────────────────────────────────────────────────
_defaults = {
    "dist_history"  : deque(maxlen=300),
    "dev_history"   : deque(maxlen=300),
    "str_history"   : deque(maxlen=300),
    "baseline_hist" : deque(maxlen=300),
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
    "calib_readings": [],
    "baseline_cm"   : float(DEFAULT_BASELINE),
    "calibrated"    : False,
    "dist_buf"      : [],
    "str_buf"       : [],
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    lidar_port = st.text_input("Serial Port", value="/dev/ttyUSB0",
                               help="Linux: /dev/ttyUSB0  |  Windows: COM3")
    lidar_baud = st.selectbox("Baud Rate", [115200, 9600], index=0)

    st.markdown("---")
    st.subheader("📏 Baseline")
    manual_baseline = st.number_input(
        "Road distance (cm)", min_value=10, max_value=2000,
        value=int(st.session_state.baseline_cm), step=5,
        help="Height of sensor above flat road. Auto-calibrated on startup.",
    )
    calib_n = st.slider("Calibration samples", 5, 50, 20,
                        help="Flat-road readings to auto-set baseline")
    bypass_calib = st.checkbox(
        "Skip calibration (use manual baseline)",
        value=False,
        help="Enable to skip auto-calibration and use the value above directly",
    )

    st.markdown("---")
    st.subheader("� Detection")
    confirm_n  = st.slider("Confirm streak", 1, 4, 2,
                           help="Consecutive windows that must agree before firing alert")
    pot_thresh = st.number_input(
        "Pothole threshold (cm)", min_value=1.0, max_value=30.0,
        value=float(POTHOLE_THRESH), step=0.5,
        help="Deviation above baseline to count as inside pothole",
    )
    bump_thresh = st.number_input(
        "Bump threshold (cm)", min_value=1.0, max_value=30.0,
        value=float(BUMP_THRESH), step=0.5,
        help="Deviation below baseline to count as speed bump",
    )

    st.markdown("---")
    st.subheader("🚗 Vehicle")
    speed_kmph = st.number_input("Speed (km/h)", 5, 120, 30, step=5)

    st.markdown("---")
    if st.button("🔄 Reset All"):
        for k, v in _defaults.items():
            st.session_state[k] = v if not callable(v) else v()
        st.rerun()


# ── Derived ───────────────────────────────────────────────────────────────────
speed_cm_s       = (speed_kmph * 100_000) / 3600
dist_per_reading = speed_cm_s / FRAME_RATE_HZ


# ── Helpers ───────────────────────────────────────────────────────────────────

def severity_label(depth_cm: float) -> str:
    for lo, hi, label in SEVERITY_BANDS:
        if lo <= depth_cm < hi:
            return label
    return "—"


def compute_dimensions(dist_buf: list, str_buf: list, baseline: float) -> dict:
    arr   = np.array(dist_buf, dtype=float)
    dev   = arr - baseline
    depth_cm    = float(max(dev.max(), 0))
    pts_in_hole = int(np.sum(dev > pot_thresh))
    length_cm   = round(pts_in_hole * dist_per_reading, 1)
    width_cm    = round(length_cm * 0.80, 1)
    avg_str     = float(np.mean(str_buf)) if str_buf else 0.0
    return {
        "depth_cm"    : round(depth_cm, 1),
        "length_cm"   : length_cm,
        "width_cm"    : width_cm,
        "severity"    : severity_label(depth_cm),
        "avg_strength": round(avg_str, 0),
    }


def rule_classify(dist_cm: float, baseline: float) -> int:
    dev = dist_cm - baseline
    if dev > pot_thresh:
        return 1
    if dev < -bump_thresh:
        return 3
    return 0


# ── Detection loop ────────────────────────────────────────────────────────────

def run_detection(model):
    # ── Open LiDAR ───────────────────────────────────────────────────────────
    try:
        lidar = TF02Pro(port=lidar_port, baudrate=lidar_baud)
    except Exception as exc:
        st.error(f"❌ Cannot open `{lidar_port}`: {exc}")
        st.session_state.running = False
        return

    # ── Layout ───────────────────────────────────────────────────────────────
    top_left, top_right = st.columns([5, 1])
    top_left.success(f"✅ LiDAR connected → `{lidar_port}` @ {lidar_baud} baud")
    stop_btn = top_right.button("⏹ Stop", key="stop_btn")

    # KPI row
    k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
    ph_base   = k1.empty()
    ph_dist   = k2.empty()
    ph_dev    = k3.empty()
    ph_count  = k4.empty()
    ph_bump   = k5.empty()
    ph_depth  = k6.empty()
    ph_str    = k7.empty()

    st.markdown("---")
    status_ph   = st.empty()
    calib_ph    = st.empty()

    # Charts
    st.markdown("### 📡 Live LiDAR Readings")
    c1, c2 = st.columns(2)
    chart_dist  = c1.empty()
    chart_dev   = c2.empty()
    chart_str   = st.empty()

    st.markdown("---")
    log_ph = st.empty()

    # ── Local buffers (persist between loop iterations via session_state) ─────
    dist_buf = st.session_state.dist_buf
    str_buf  = st.session_state.str_buf
    consec_err = 0

    # If skipping calibration, mark as already calibrated with manual baseline
    if bypass_calib and not st.session_state.calibrated:
        st.session_state.baseline_cm = float(manual_baseline)
        st.session_state.calibrated  = True
        calib_ph.success(
            f"✅ Using manual baseline: **{st.session_state.baseline_cm:.0f} cm**"
        )

    logger.info("Detection started. Baseline=%.1f cm", st.session_state.baseline_cm)

    while not stop_btn:

        # ── READ CURRENT FRAME ────────────────────────────────────────────────
        try:
            reading    = lidar.read_current()
            consec_err = 0
        except LiDARReadError as exc:
            consec_err += 1
            status_ph.warning(
                f"⚠️ LiDAR read error #{consec_err}: {exc}"
            )
            logger.warning("Read error #%d: %s", consec_err, exc)
            if consec_err > 20:
                st.error(
                    "❌ Too many consecutive errors.\n\n"
                    "**Check:** Is the correct serial port selected? "
                    "Is the LiDAR powered? Try unplugging and re-plugging."
                )
                break
            time.sleep(0.05)
            continue

        dist     = reading["distance_cm"]
        strength = reading["strength"]
        temp     = reading["temperature_c"]

        # ── CALIBRATION PHASE ─────────────────────────────────────────────────
        if not st.session_state.calibrated:
            st.session_state.calib_readings.append(dist)
            done      = len(st.session_state.calib_readings)
            remaining = calib_n - done

            calib_ph.info(
                f"🔵 **Calibrating baseline …** {done}/{calib_n} readings.  "
                f"Keep sensor over **flat road** ({remaining} more needed).  "
                f"Current reading: **{dist} cm** | Strength: {strength}"
            )

            # Still show live data during calibration
            st.session_state.dist_history.append(dist)
            st.session_state.dev_history.append(
                dist - st.session_state.baseline_cm
            )
            st.session_state.str_history.append(strength)
            st.session_state.baseline_hist.append(st.session_state.baseline_cm)

            ph_base.metric("📐 Baseline", f"{st.session_state.baseline_cm:.0f} cm")
            ph_dist.metric("📡 Distance", f"{dist} cm")
            ph_dev.metric("↕️ Deviation",
                          f"{dist - st.session_state.baseline_cm:+.1f} cm")
            ph_count.metric("🕳️ Potholes", st.session_state.pothole_count)
            ph_bump.metric("🔶 Bumps",     st.session_state.bump_count)
            ph_depth.metric("📏 Last Depth",
                            f"{st.session_state.last_depth} cm")
            ph_str.metric("📶 Strength",   f"{strength} | 🌡️ {temp}°C")

            chart_dist.line_chart(pd.DataFrame({
                "Distance (cm)": list(st.session_state.dist_history),
                "Baseline (cm)": list(st.session_state.baseline_hist),
            }))
            chart_dev.line_chart(pd.DataFrame({
                "Deviation (cm)": list(st.session_state.dev_history)
            }))

            if done >= calib_n:
                cal = sorted(st.session_state.calib_readings)
                st.session_state.baseline_cm = float(cal[len(cal) // 2])
                st.session_state.calibrated  = True
                calib_ph.success(
                    f"✅ Baseline locked: **{st.session_state.baseline_cm:.1f} cm**  "
                    f"(median of {done} readings)"
                )
                logger.info("Baseline calibrated: %.1f cm",
                            st.session_state.baseline_cm)

            time.sleep(0.02)
            continue

        # ── ACTIVE DETECTION ──────────────────────────────────────────────────
        baseline = st.session_state.baseline_cm
        dev      = dist - baseline

        # Update history deques
        st.session_state.dist_history.append(dist)
        st.session_state.dev_history.append(dev)
        st.session_state.str_history.append(strength)
        st.session_state.baseline_hist.append(baseline)

        # Sliding window
        dist_buf.append(dist)
        str_buf.append(strength)
        if len(dist_buf) > WINDOW_SIZE:
            dist_buf.pop(0)
            str_buf.pop(0)

        # ─── Quick single-reading rule classification ─────────────────────────
        rule_cls = rule_classify(dist, baseline)

        # ─── ML window classification (when window full) ──────────────────────
        if len(dist_buf) == WINDOW_SIZE and model is not None:
            feats   = extract_features(
                np.array(dist_buf), np.array(str_buf), baseline
            ).reshape(1, -1)
            ml_cls  = int(model.predict(feats)[0])
            ml_prob = model.predict_proba(feats)[0]
            ml_conf = float(ml_prob[ml_cls])
            final_cls = ml_cls if ml_conf >= 0.55 else rule_cls
        else:
            final_cls = rule_cls
            ml_conf   = 0.0

        is_ph   = IS_POTHOLE.get(final_cls, False)
        is_bump = IS_BUMP.get(final_cls, False)

        # ─── Streak accumulator ───────────────────────────────────────────────
        if is_ph or is_bump:
            st.session_state.confirm_streak += 1
        else:
            st.session_state.confirm_streak = 0

        confirmed = st.session_state.confirm_streak >= confirm_n

        # ─── Alert on confirmed detection ─────────────────────────────────────
        if confirmed:
            dims = compute_dimensions(dist_buf, str_buf, baseline)

            if is_ph:
                st.session_state.pothole_count += 1
            elif is_bump:
                st.session_state.bump_count += 1

            # Slide buffer by half (don't wipe — potholes can be multi-window)
            st.session_state.confirm_streak = 0
            half = len(dist_buf) // 2
            dist_buf[:] = dist_buf[half:]
            str_buf[:]  = str_buf[half:]

            st.session_state.last_label    = CLASS_LABELS[final_cls]
            st.session_state.last_depth    = dims["depth_cm"]
            st.session_state.last_length   = dims["length_cm"]
            st.session_state.last_width    = dims["width_cm"]
            st.session_state.last_strength = int(dims["avg_strength"])

            log_entry = {
                "Time"        : time.strftime("%H:%M:%S"),
                "Type"        : CLASS_LABELS[final_cls],
                "Deviation"   : f"{dev:+.1f} cm",
                "Depth (cm)"  : dims["depth_cm"],
                "Length (cm)" : dims["length_cm"],
                "Width (cm)"  : dims["width_cm"],
                "Severity"    : dims["severity"],
                "ML Conf."    : f"{ml_conf:.0%}" if ml_conf > 0 else "rule",
                "Strength"    : int(dims["avg_strength"]),
                "Temp (°C)"   : temp,
                "Baseline"    : f"{baseline:.0f} cm",
            }
            st.session_state.pothole_log.insert(0, log_entry)

            status_ph.error(
                f"🔴 **{CLASS_LABELS[final_cls]} CONFIRMED!**  "
                f"Deviation: **{dev:+.1f} cm** |  "
                f"Depth: **{dims['depth_cm']} cm** |  "
                f"Length: **{dims['length_cm']} cm** |  "
                f"{dims['severity']}"
            )
            logger.info("DETECTED: %s", log_entry)

        else:
            # Live status: no confirmed event
            streak_bar = (
                "█" * st.session_state.confirm_streak
                + "░" * max(0, confirm_n - st.session_state.confirm_streak)
            )
            dev_disp = f"{dev:+.1f}"
            win_info = (f"win {len(dist_buf)}/{WINDOW_SIZE}"
                        if len(dist_buf) < WINDOW_SIZE else "window ✓")

            if final_cls == 0:
                status_ph.success(
                    f"🟢 **Flat Road** — "
                    f"dist: **{dist} cm** | dev: {dev_disp} cm | "
                    f"str: {strength} | temp: {temp}°C | {win_info}"
                )
            elif is_ph:
                status_ph.warning(
                    f"🟡 **Possible {CLASS_LABELS[final_cls]}** — "
                    f"dist: **{dist} cm** | dev: **{dev_disp} cm** | "
                    f"conf: {ml_conf:.0%} | "
                    f"streak [{streak_bar}] {st.session_state.confirm_streak}/{confirm_n}"
                )
            else:
                status_ph.info(
                    f"🔶 **Speed Bump** — "
                    f"dist: **{dist} cm** | dev: **{dev_disp} cm** | "
                    f"conf: {ml_conf:.0%}"
                )

        # ─── KPIs ─────────────────────────────────────────────────────────────
        ph_base.metric("📐 Baseline", f"{baseline:.0f} cm")
        ph_dist.metric("📡 Distance", f"{dist} cm")
        ph_dev.metric(
            "↕️ Deviation", f"{dev:+.1f} cm",
            delta=f"{dev:+.1f}",
            delta_color="inverse",   # red when going up (pothole)
        )
        ph_count.metric("🕳️ Potholes", st.session_state.pothole_count)
        ph_bump.metric("🔶 Bumps",     st.session_state.bump_count)
        ph_depth.metric("📏 Last Depth", f"{st.session_state.last_depth} cm")
        ph_str.metric("📶 Strength",    f"{strength} | {temp}°C")

        # ─── Charts ───────────────────────────────────────────────────────────
        chart_dist.line_chart(pd.DataFrame({
            "Distance (cm)": list(st.session_state.dist_history),
            "Baseline (cm)": list(st.session_state.baseline_hist),
        }))
        chart_dev.line_chart(pd.DataFrame({
            "Deviation (cm)": list(st.session_state.dev_history)
        }))
        chart_str.line_chart(pd.DataFrame({
            "Signal Strength": list(st.session_state.str_history)
        }))

        # ─── Detection log ────────────────────────────────────────────────────
        if st.session_state.pothole_log:
            log_ph.subheader("📋 Detection Log")
            log_ph.dataframe(
                pd.DataFrame(st.session_state.pothole_log).head(50),
                width="stretch",
            )

        time.sleep(0.02)   # ~50 reads/sec max (sensor runs at 100Hz)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    lidar.close()
    st.session_state.running = False
    st.info("⏹ Monitoring stopped.")


# ── Page entry point ──────────────────────────────────────────────────────────
st.title("🕳️ Pothole Detection — Single-Point LiDAR")
st.caption("TF02-Pro · Buffer-Flush Fix · Baseline-Deviation · ML-Confirmed")

model = load_model()
if model is None:
    st.warning(
        "⚠️ `pothole_model.pkl` not found — running in **rule-only mode** "
        "(ML not available). Run `python model_train.py` to enable ML."
    )
    # Don't stop — rule-based detection still works without the model

# Sync manual baseline into session before calibration
if not st.session_state.calibrated:
    st.session_state.baseline_cm = float(manual_baseline)

st.info(
    f"**Detection concept:**  LiDAR points **DOWN** at road.  "
    f"Baseline = **{st.session_state.baseline_cm:.0f} cm** "
    f"({'auto-calibrated from first ' + str(calib_n) + ' readings' if not bypass_calib else 'manual'}).  \n"
    f"📈 Distance **> baseline + {pot_thresh} cm** → **POTHOLE** (ground further away).  \n"
    f"📉 Distance **< baseline - {bump_thresh} cm** → **SPEED BUMP** (ground closer).  \n"
    f"Pothole threshold: **{pot_thresh} cm** | Bump threshold: **{bump_thresh} cm**"
)

if not st.session_state.running:
    col1, col2 = st.columns([2, 3])
    with col1:
        if st.button("▶ Start Monitoring", type="primary"):
            st.session_state.running        = True
            if not bypass_calib:
                st.session_state.calibrated     = False
                st.session_state.calib_readings = []
            st.rerun()

if st.session_state.running:
    run_detection(model)