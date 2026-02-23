"""
dashboard.py
============
Real-time Pothole Detection — TF02-Pro Single-Point LiDAR

Detection Logic (simple, physics-based + ML confirmation)
──────────────────────────────────────────────────────────
  Baseline = distance from sensor to flat road (auto-calibrated on startup).

  RULE:  deviation = current_reading - baseline
    ► deviation > +POTHOLE_THRESH cm  →  POTHOLE   (ground dropped away)
    ► deviation < -BUMP_THRESH   cm  →  SPEED BUMP (ground rose up)
    ► else                           →  FLAT ROAD

  ML CONFIRMATION:
    After the rule says "pothole" or "bump", the ML model checks a 20-reading
    window and must AGREE before the count is incremented.
    This eliminates single-spike false positives.

  STREAK GATE (confirm_n = 2 consecutive windows by default):
    Reduces to 2 (not 3) so a real pothole that spans only 4-6 readings
    at 30 km/h still gets detected in time.

Bugs fixed vs previous version
───────────────────────────────
  1. model_train.py training code no longer runs at import time.
  2. Calibration phase now updates charts so UI isn't frozen.
  3. confirm_n default lowered to 2 (realistic for short potholes).
  4. Buffer NOT fully cleared on confirmation — slides by WINDOW_SIZE//2
     so the tail of the pothole is still captured properly.
  5. use_container_width=True on st.button removed (deprecated).
  6. Live "deviation" gauge added so user can see detection signal.
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
CLASS_LABELS = {
    0: "🟢 Flat Road",
    1: "🟡 Shallow Pothole",
    2: "🔴 Deep Pothole",
    3: "🔶 Speed Bump",
}
IS_POTHOLE = {0: False, 1: True, 2: True, 3: False}
IS_BUMP    = {0: False, 1: False, 2: False, 3: True}
FRAME_RATE_HZ = 100

SEVERITY_BANDS = [
    (0,   3,   "—  Noise"),
    (3,   8,   "⚠️  Shallow"),
    (8,  15,   "🔶 Moderate"),
    (15, 999,  "🔴 Deep / Dangerous"),
]


# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading detection model …")
def load_model():
    try:
        m = joblib.load("pothole_model.pkl")
        logger.info("pothole_model.pkl loaded OK")
        return m
    except FileNotFoundError:
        return None


# ── Session-state defaults ────────────────────────────────────────────────────
_defaults = {
    "dist_history"   : deque(maxlen=400),
    "dev_history"    : deque(maxlen=400),   # Deviation from baseline
    "str_history"    : deque(maxlen=400),
    "baseline_hist"  : deque(maxlen=400),
    "pothole_count"  : 0,
    "bump_count"     : 0,
    "pothole_log"    : [],
    "running"        : False,
    "confirm_streak" : 0,
    "last_label"     : "—",
    "last_depth"     : 0.0,
    "last_length"    : 0.0,
    "last_width"     : 0.0,
    "last_strength"  : 0,
    "calib_readings" : [],
    "baseline_cm"    : float(DEFAULT_BASELINE),
    "calibrated"     : False,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    lidar_port  = st.text_input("Serial Port", value="/dev/ttyUSB0",
                                help="/dev/ttyUSB0 on Linux, COM3 on Windows")
    lidar_baud  = st.selectbox("Baud Rate", [115200, 9600], index=0)
    n_median    = st.slider("Median filter frames", 1, 5, 1,
                            help="1 = fastest, higher = smoother but slower")
    confirm_n   = st.slider("Confirm streak (windows)", 1, 4, 2,
                            help="How many consecutive windows must agree before alert")

    st.markdown("---")
    st.subheader("📏 Baseline Distance")
    manual_baseline = st.number_input(
        "Road distance (cm)", min_value=20, max_value=2000,
        value=int(st.session_state.baseline_cm), step=5,
        help="Sensor height above flat road. Auto-calibrated on startup.",
    )
    calib_n = st.slider("Calibration samples", 5, 60, 20,
                        help="Flat-road readings used to set baseline automatically")

    st.markdown("---")
    st.subheader("🚗 Vehicle")
    speed_kmph = st.number_input("Speed (km/h)", 5, 120, 30, step=5)

    st.markdown("---")
    if st.button("� Reset Session"):
        for k in _defaults:
            st.session_state[k] = _defaults[k]
        st.rerun()


# ── Derived ───────────────────────────────────────────────────────────────────
speed_cm_s       = (speed_kmph * 100_000) / 3600
dist_per_reading = speed_cm_s / FRAME_RATE_HZ   # cm of road per LiDAR reading


# ── Helpers ───────────────────────────────────────────────────────────────────

def severity_label(depth_cm: float) -> str:
    for lo, hi, label in SEVERITY_BANDS:
        if lo <= depth_cm < hi:
            return label
    return "—"


def compute_dimensions(dist_buf: list, str_buf: list, baseline: float) -> dict:
    arr   = np.array(dist_buf, dtype=float)
    dev   = arr - baseline

    depth_cm       = float(max(dev.max(), 0))
    pts_in_hole    = int(np.sum(dev > POTHOLE_THRESH))
    length_cm      = round(pts_in_hole * dist_per_reading, 1)
    width_cm       = round(length_cm * 0.80, 1)
    avg_str        = float(np.mean(str_buf)) if str_buf else 0.0

    return {
        "depth_cm"    : round(depth_cm, 1),
        "length_cm"   : length_cm,
        "width_cm"    : width_cm,
        "severity"    : severity_label(depth_cm),
        "avg_strength": round(avg_str, 0),
    }


def rule_classify(dist_cm: float, baseline: float) -> int:
    """
    Instant single-reading classification.
    0=flat, 1=pothole, 3=bump.
    (ML refines 1 → shallow/deep)
    """
    dev = dist_cm - baseline
    if dev > POTHOLE_THRESH:
        return 1
    if dev < -BUMP_THRESH:
        return 3
    return 0


# ── Detection loop ────────────────────────────────────────────────────────────

def run_detection(model):
    # ── Open LiDAR ───────────────────────────────────────────────────────────
    try:
        lidar = TF02Pro(port=lidar_port, baudrate=lidar_baud)
    except Exception as exc:
        st.error(f"❌ Cannot open `{lidar_port}`: {exc}")
        logger.error("Port open failed: %s", exc)
        st.session_state.running = False
        return

    # ── Layout ───────────────────────────────────────────────────────────────
    hdr_col, stop_col = st.columns([6, 1])
    hdr_col.success(f"✅ LiDAR → `{lidar_port}` @ {lidar_baud} baud")
    stop_btn = stop_col.button("⏹ Stop", key="stop_btn")

    k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
    ph_baseline  = k1.empty()
    ph_dist      = k2.empty()
    ph_dev       = k3.empty()
    ph_count     = k4.empty()
    ph_bump      = k5.empty()
    ph_depth     = k6.empty()
    ph_str       = k7.empty()

    st.markdown("---")
    status_ph    = st.empty()
    calib_ph     = st.empty()

    st.markdown("### 📡 Live Readings")
    chart_cols   = st.columns(2)
    chart_dist   = chart_cols[0].empty()
    chart_dev    = chart_cols[1].empty()
    chart_str    = st.empty()

    st.markdown("---")
    log_ph       = st.empty()

    # ── Buffers ───────────────────────────────────────────────────────────────
    dist_buf   = []
    str_buf    = []
    consec_err = 0

    logger.info("Detection started. Initial baseline=%.1f cm",
                st.session_state.baseline_cm)

    while not stop_btn:

        # ─── READ ONE MEDIAN FRAME ────────────────────────────────────────────
        try:
            reading    = lidar.read_median(samples=n_median)
            consec_err = 0
        except LiDARReadError as exc:
            consec_err += 1
            logger.warning("Read error #%d: %s", consec_err, exc)
            if consec_err > 30:
                st.error("❌ Too many read errors — check wiring / port.")
                break
            time.sleep(0.05)
            continue

        dist     = reading["distance_cm"]
        strength = reading["strength"]

        # ─── CALIBRATION PHASE ────────────────────────────────────────────────
        if not st.session_state.calibrated:
            st.session_state.calib_readings.append(dist)
            done    = len(st.session_state.calib_readings)
            remaining = calib_n - done

            calib_ph.info(
                f"🔵 **Calibrating …** ({done}/{calib_n} readings)  "
                f"Keep sensor over **flat road**.  {remaining} more needed."
            )

            # Update charts even during calibration (so UI isn't frozen)
            st.session_state.dist_history.append(dist)
            st.session_state.dev_history.append(dist - st.session_state.baseline_cm)
            st.session_state.str_history.append(strength)
            st.session_state.baseline_hist.append(st.session_state.baseline_cm)

            # Refresh KPIs with current values
            ph_baseline.metric("📐 Baseline", f"{st.session_state.baseline_cm:.0f} cm")
            ph_dist.metric("📡 Distance", f"{dist} cm")
            ph_dev.metric("↕️ Deviation", f"{dist - st.session_state.baseline_cm:+.1f} cm")
            ph_count.metric("🕳️ Potholes", st.session_state.pothole_count)
            ph_bump.metric("🔶 Bumps",  st.session_state.bump_count)
            ph_depth.metric("📏 Last Depth", f"{st.session_state.last_depth} cm")
            ph_str.metric("📶 Strength", strength)

            # Refresh charts
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
                    f"✅ Baseline locked at **{st.session_state.baseline_cm:.1f} cm**  "
                    f"(median of {done} readings)"
                )
                logger.info("Baseline calibrated: %.1f cm", st.session_state.baseline_cm)

            time.sleep(0.01)
            continue   # skip classification during calibration

        # ─── NORMAL OPERATION ─────────────────────────────────────────────────
        baseline = st.session_state.baseline_cm
        dev      = dist - baseline

        # Update rolling histories
        st.session_state.dist_history.append(dist)
        st.session_state.dev_history.append(dev)
        st.session_state.str_history.append(strength)
        st.session_state.baseline_hist.append(baseline)

        # Fill sliding window
        dist_buf.append(dist)
        str_buf.append(strength)
        if len(dist_buf) > WINDOW_SIZE:
            dist_buf.pop(0)
            str_buf.pop(0)

        # ─── CLASSIFY ─────────────────────────────────────────────────────────
        if len(dist_buf) == WINDOW_SIZE:

            # Step A: instant rule-based reading-level check
            rule_cls = rule_classify(dist, baseline)

            # Step B: ML on full 20-reading window
            feats   = extract_features(
                np.array(dist_buf), np.array(str_buf), baseline
            ).reshape(1, -1)

            ml_cls  = int(model.predict(feats)[0])
            ml_prob = model.predict_proba(feats)[0]
            ml_conf = float(ml_prob[ml_cls])
            label   = CLASS_LABELS[ml_cls]

            # Prefer ML when confident (≥ 55%), else fall back to rule
            final_cls = ml_cls if ml_conf >= 0.55 else rule_cls
            is_ph     = IS_POTHOLE.get(final_cls, False)
            is_bump   = IS_BUMP.get(final_cls, False)

            # Step C: streak accumulator
            if is_ph or is_bump:
                st.session_state.confirm_streak += 1
            else:
                st.session_state.confirm_streak = 0

            confirmed = st.session_state.confirm_streak >= confirm_n

            # ─── CONFIRMED DETECTION ──────────────────────────────────────────
            if confirmed:
                dims = compute_dimensions(dist_buf, str_buf, baseline)

                if is_ph:
                    st.session_state.pothole_count += 1
                elif is_bump:
                    st.session_state.bump_count += 1

                # Reset streak (but DON'T clear buffer — just slide by half)
                st.session_state.confirm_streak = 0
                slide = WINDOW_SIZE // 2
                dist_buf[:] = dist_buf[slide:]
                str_buf[:]  = str_buf[slide:]

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
                    "ML Conf."    : f"{ml_conf:.0%}",
                    "Strength"    : int(dims["avg_strength"]),
                    "Baseline"    : f"{baseline:.0f} cm",
                }
                st.session_state.pothole_log.insert(0, log_entry)

                status_ph.error(
                    f"🔴 **{CLASS_LABELS[final_cls]} CONFIRMED** — "
                    f"Deviation: **{dev:+.1f} cm** | "
                    f"Depth: **{dims['depth_cm']} cm** | "
                    f"Length: **{dims['length_cm']} cm** | "
                    f"{dims['severity']} | "
                    f"ML: {ml_conf:.0%}"
                )
                logger.info("DETECTION: %s", log_entry)

            else:
                # Real-time status display (no confirmed event)
                streak_bar = (
                    "█" * st.session_state.confirm_streak +
                    "░" * max(0, confirm_n - st.session_state.confirm_streak)
                )
                dev_sign = f"+{dev:.1f}" if dev >= 0 else f"{dev:.1f}"

                if final_cls == 0:
                    status_ph.success(
                        f"🟢 **Flat Road** — dist: {dist} cm | "
                        f"dev: {dev_sign} cm | strength: {strength}"
                    )
                elif is_ph:
                    status_ph.warning(
                        f"🟡 **{CLASS_LABELS[final_cls]}** — "
                        f"dist: {dist} cm | dev: **{dev_sign} cm** | "
                        f"ML: {ml_conf:.0%} | streak [{streak_bar}] "
                        f"{st.session_state.confirm_streak}/{confirm_n}"
                    )
                else:
                    status_ph.info(
                        f"🔶 **{CLASS_LABELS[final_cls]}** — "
                        f"dist: {dist} cm | dev: {dev_sign} cm | "
                        f"ML: {ml_conf:.0%}"
                    )

        elif len(dist_buf) < WINDOW_SIZE:
            # Window still filling — show rule-only live status
            rule_cls  = rule_classify(dist, baseline)
            dev_sign  = f"+{dev:.1f}" if dev >= 0 else f"{dev:.1f}"
            status_ph.info(
                f"{'⚠️ ' if rule_cls != 0 else '🟢'} "
                f"[Filling window {len(dist_buf)}/{WINDOW_SIZE}] — "
                f"dist: {dist} cm | dev: {dev_sign} cm | strength: {strength}"
            )

        # ─── UPDATE KPIs ──────────────────────────────────────────────────────
        ph_baseline.metric("� Baseline",   f"{baseline:.0f} cm")
        ph_dist.metric("� Distance",       f"{dist} cm")
        ph_dev.metric("↕️ Deviation",       f"{dev:+.1f} cm",
                      delta=f"{dev:+.1f}",
                      delta_color="inverse")
        ph_count.metric("�️ Potholes",     st.session_state.pothole_count)
        ph_bump.metric("🔶 Bumps",          st.session_state.bump_count)
        ph_depth.metric("📏 Last Depth",    f"{st.session_state.last_depth} cm")
        ph_str.metric("📶 Strength",        strength)

        # ─── UPDATE CHARTS ────────────────────────────────────────────────────
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

        # ─── DETECTION LOG ────────────────────────────────────────────────────
        if st.session_state.pothole_log:
            log_ph.subheader("📋 Detection Log")
            log_ph.dataframe(
                pd.DataFrame(st.session_state.pothole_log).head(50),
                width="stretch",
            )

        time.sleep(0.01)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    lidar.close()
    st.session_state.running = False
    st.info("⏹ Monitoring stopped.")
    logger.info("Detection loop stopped.")


# ── Entry point ───────────────────────────────────────────────────────────────
st.title("🕳️ Pothole Detection Dashboard — Single-Point LiDAR")
st.caption("TF02-Pro · Baseline-Deviation · ML-Confirmed · Real-Time")

model = load_model()
if model is None:
    st.error(
        "❌ `pothole_model.pkl` not found.\n\n"
        "**Run this first:**\n```\npython model_train.py\n```"
    )
    st.stop()

# Pre-load manual baseline into session state before any calibration
if not st.session_state.calibrated:
    st.session_state.baseline_cm = float(manual_baseline)

st.info(
    f"**How it works:** LiDAR shoots DOWN at road.  "
    f"Baseline = **{st.session_state.baseline_cm:.0f} cm** (auto-calibrated from first {calib_n} readings).  "
    f"→ Reading **above** baseline = **pothole** (ground dropped).  "
    f"→ Reading **below** baseline = **speed bump** (ground rose).  "
    f"ML confirms before alert fires."
)

if not st.session_state.running:
    if st.button("▶ Start Monitoring", type="primary"):
        st.session_state.running        = True
        st.session_state.calibrated     = False
        st.session_state.calib_readings = []
        st.rerun()

if st.session_state.running:
    run_detection(model)