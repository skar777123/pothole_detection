"""
dashboard.py
=============
Real-time Pothole Detection — TF02-Pro Single-Point LiDAR
Continuous streaming mode: reads every frame the sensor emits (~100 Hz).

DETECTION PIPELINE
──────────────────
  Every frame from the sensor:
    1. Parse distance, strength, temperature
    2. Compare to baseline:
         deviation = distance - baseline
         > +pot_thresh cm  → POTHOLE  (ground dropped / farther)
         < -bump_thresh cm → BUMP     (ground rose / closer)
         else              → FLAT ROAD
    3. Build a 20-reading sliding window
    4. ML classifier on the window (22 features)
    5. Streak gate: N consecutive positive windows → confirmed alert
    6. Compute depth, length, width, severity → log entry
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
    (0,   3,   "— Noise"),
    (3,   8,   "⚠️ Shallow"),
    (8,  15,   "🔶 Moderate"),
    (15, 999,  "🔴 Deep/Dangerous"),
]


# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ML model …")
def load_model():
    try:
        return joblib.load("pothole_model.pkl")
    except FileNotFoundError:
        return None


# ── Session-state defaults ────────────────────────────────────────────────────
_defaults: dict = {
    "dist_history"  : deque(maxlen=500),
    "dev_history"   : deque(maxlen=500),
    "str_history"   : deque(maxlen=500),
    "baseline_hist" : deque(maxlen=500),
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
    "total_frames"  : 0,     # total frames received from sensor
    "total_errors"  : 0,     # total failed frames
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    lidar_port = st.text_input(
        "Serial Port", value="/dev/ttyUSB0",
        help="Linux: /dev/ttyUSB0  |  Windows: COM3"
    )
    lidar_baud = st.selectbox("Baud Rate", [115200, 9600], index=0)

    st.markdown("---")
    st.subheader("📏 Baseline")
    manual_baseline = st.number_input(
        "Road distance (cm)", min_value=10, max_value=2000,
        value=int(st.session_state.baseline_cm), step=5,
        help="Sensor height above flat road. Auto-calibrated on startup.",
    )
    calib_n = st.slider(
        "Calibration samples", 5, 60, 20,
        help="Readings collected over flat road to set baseline"
    )
    bypass_calib = st.checkbox(
        "Skip calibration (use manual value above)",
        value=False,
    )

    st.markdown("---")
    st.subheader("🔍 Detection Thresholds")
    pot_thresh = st.number_input(
        "Pothole threshold (cm)", 1.0, 30.0,
        value=float(POTHOLE_THRESH), step=0.5,
        help="Deviation ABOVE baseline to consider a reading as inside pothole"
    )
    bump_thresh = st.number_input(
        "Bump threshold (cm)", 1.0, 30.0,
        value=float(BUMP_THRESH), step=0.5,
        help="Deviation BELOW baseline to consider a reading as speed bump"
    )
    confirm_n = st.slider(
        "Confirm streak (windows)", 1, 4, 2,
        help="Consecutive windows that must agree before firing an alert"
    )

    st.markdown("---")
    st.subheader("🚗 Vehicle Speed")
    speed_kmph = st.number_input("Speed (km/h)", 5, 120, 30, step=5)

    st.markdown("---")
    if st.button("🔄 Reset All"):
        for k, v in _defaults.items():
            st.session_state[k] = v
        st.rerun()


# ── Derived values ────────────────────────────────────────────────────────────
speed_cm_s       = (speed_kmph * 100_000) / 3600
dist_per_reading = speed_cm_s / FRAME_RATE_HZ   # cm of road per frame


# ── Helpers ───────────────────────────────────────────────────────────────────

def severity_label(depth_cm: float) -> str:
    for lo, hi, label in SEVERITY_BANDS:
        if lo <= depth_cm < hi:
            return label
    return "—"


def compute_dimensions(dist_buf: list, str_buf: list, baseline: float) -> dict:
    arr          = np.array(dist_buf, dtype=float)
    dev          = arr - baseline
    depth_cm     = float(max(dev.max(), 0))
    pts_in_hole  = int(np.sum(dev > pot_thresh))
    length_cm    = round(pts_in_hole * dist_per_reading, 1)
    width_cm     = round(length_cm * 0.80, 1)
    avg_str      = float(np.mean(str_buf)) if str_buf else 0.0
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
        return 1    # pothole
    if dev < -bump_thresh:
        return 3    # speed bump
    return 0        # flat road


# ── Main detection loop ───────────────────────────────────────────────────────

def run_detection(model):

    # ── Open LiDAR ───────────────────────────────────────────────────────────
    try:
        lidar = TF02Pro(port=lidar_port, baudrate=lidar_baud)
    except Exception as exc:
        st.error(f"❌ Cannot open `{lidar_port}`: {exc}")
        st.session_state.running = False
        return

    # ── UI shell (created once, updated in-place every frame) ────────────────
    hdr_l, hdr_r = st.columns([6, 1])
    hdr_l.success(f"✅ Streaming from `{lidar_port}` @ {lidar_baud} baud")
    stop_btn = hdr_r.button("⏹ Stop", key="stop_btn")

    # KPI row
    kc = st.columns(8)
    ph_base   = kc[0].empty()
    ph_dist   = kc[1].empty()
    ph_dev    = kc[2].empty()
    ph_str    = kc[3].empty()
    ph_temp   = kc[4].empty()
    ph_count  = kc[5].empty()
    ph_bump   = kc[6].empty()
    ph_depth  = kc[7].empty()

    st.markdown("---")

    # Status banner
    status_ph  = st.empty()
    calib_ph   = st.empty()

    # Charts
    st.markdown("### 📡 Continuous LiDAR Stream")
    ch1, ch2 = st.columns(2)
    chart_dist = ch1.empty()
    chart_dev  = ch2.empty()
    chart_str  = st.empty()

    # Frame stats bar
    stats_ph = st.empty()

    st.markdown("---")
    log_ph = st.empty()

    # ── Local shorthand references to session-state buffers ───────────────────
    dist_buf = st.session_state.dist_buf
    str_buf  = st.session_state.str_buf

    # ── Apply bypass calibration if requested ─────────────────────────────────
    if bypass_calib and not st.session_state.calibrated:
        st.session_state.baseline_cm = float(manual_baseline)
        st.session_state.calibrated  = True
        calib_ph.success(
            f"✅ Manual baseline: **{st.session_state.baseline_cm:.0f} cm**"
        )

    logger.info("Streaming started. Baseline=%.1f cm",
                st.session_state.baseline_cm)

    consec_err = 0
    last_chart_update = 0.0   # throttle chart redraws

    # ====================================================================
    # CONTINUOUS LOOP — one iteration = one sensor frame (~10 ms at 100Hz)
    # ====================================================================
    while not stop_btn:

        # ─── READ ONE FRAME from the live stream ──────────────────────────
        try:
            frame = lidar.read_frame()
            consec_err = 0
            st.session_state.total_frames += 1
        except LiDARReadError as exc:
            consec_err += 1
            st.session_state.total_errors += 1
            logger.warning("Frame error #%d: %s", consec_err, exc)
            if consec_err > 30:
                st.error(
                    "❌ Too many consecutive errors.\n\n"
                    "**Check:** correct port? LiDAR powered? Baud rate correct?"
                )
                break
            # Don't sleep long — just retry immediately
            continue

        dist     = frame["distance_cm"]
        strength = frame["strength"]
        temp     = frame["temperature_c"]

        # ─── CALIBRATION PHASE ────────────────────────────────────────────
        if not st.session_state.calibrated:
            st.session_state.calib_readings.append(dist)
            done      = len(st.session_state.calib_readings)
            remaining = calib_n - done

            calib_ph.info(
                f"🔵 **Calibrating** {done}/{calib_n} — "
                f"dist: **{dist} cm** | str: {strength} | temp: {temp}°C | "
                f"{remaining} more readings needed over flat ground."
            )

            st.session_state.dist_history.append(dist)
            st.session_state.dev_history.append(
                dist - st.session_state.baseline_cm
            )
            st.session_state.str_history.append(strength)
            st.session_state.baseline_hist.append(
                st.session_state.baseline_cm
            )

            if done >= calib_n:
                cal = sorted(st.session_state.calib_readings)
                st.session_state.baseline_cm = float(cal[len(cal) // 2])
                st.session_state.calibrated  = True
                calib_ph.success(
                    f"✅ Baseline locked: **{st.session_state.baseline_cm:.1f} cm** "
                    f"(median of {done} readings)"
                )
                logger.info("Baseline: %.1f cm", st.session_state.baseline_cm)

            # Update KPIs during calibration
            ph_base.metric("📐 Baseline",  f"{st.session_state.baseline_cm:.0f} cm")
            ph_dist.metric("📡 Distance",  f"{dist} cm")
            ph_dev.metric("↕️ Deviation",
                          f"{dist - st.session_state.baseline_cm:+.1f} cm")
            ph_str.metric("📶 Strength",   strength)
            ph_temp.metric("🌡️ Temp",      f"{temp}°C")
            ph_count.metric("🕳️ Potholes", st.session_state.pothole_count)
            ph_bump.metric("🔶 Bumps",     st.session_state.bump_count)
            ph_depth.metric("📏 Last Depth",
                            f"{st.session_state.last_depth} cm")
            continue   # skip detection during calibration

        # ─── ACTIVE DETECTION ─────────────────────────────────────────────
        baseline = st.session_state.baseline_cm
        dev      = dist - baseline

        # Update live histories
        st.session_state.dist_history.append(dist)
        st.session_state.dev_history.append(dev)
        st.session_state.str_history.append(strength)
        st.session_state.baseline_hist.append(baseline)

        # Sliding inference window
        dist_buf.append(dist)
        str_buf.append(strength)
        if len(dist_buf) > WINDOW_SIZE:
            dist_buf.pop(0)
            str_buf.pop(0)

        # ─── CLASSIFY ─────────────────────────────────────────────────────
        rule_cls = rule_classify(dist, baseline)

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

        # Streak gate
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

            st.session_state.confirm_streak = 0
            # Slide buffer by half (don't fully clear)
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
                "Dev (cm)"    : f"{dev:+.1f}",
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
                f"🔴 **{CLASS_LABELS[final_cls]} CONFIRMED** — "
                f"Deviation: **{dev:+.1f} cm** | "
                f"Depth: **{dims['depth_cm']} cm** | "
                f"Length: {dims['length_cm']} cm | "
                f"Width: {dims['width_cm']} cm | "
                f"{dims['severity']}"
            )
            logger.info("DETECTED: %s", log_entry)

        else:
            # Live status
            streak_bar = (
                "█" * st.session_state.confirm_streak
                + "░" * max(0, confirm_n - st.session_state.confirm_streak)
            )
            win_pct = f"{len(dist_buf)}/{WINDOW_SIZE}"
            dev_s   = f"{dev:+.1f}"

            if final_cls == 0:
                status_ph.success(
                    f"🟢 **Flat Road** — "
                    f"dist: **{dist} cm** | dev: {dev_s} cm | "
                    f"str: {strength} | temp: {temp}°C | win: {win_pct}"
                )
            elif is_ph:
                status_ph.warning(
                    f"🟡 **{CLASS_LABELS[final_cls]}** — "
                    f"dist: **{dist} cm** | dev: **{dev_s} cm** | "
                    f"conf: {ml_conf:.0%} | streak [{streak_bar}] "
                    f"{st.session_state.confirm_streak}/{confirm_n}"
                )
            else:
                status_ph.info(
                    f"🔶 **Speed Bump** — "
                    f"dist: **{dist} cm** | dev: **{dev_s} cm** | "
                    f"conf: {ml_conf:.0%}"
                )

        # ─── KPIs (every frame) ───────────────────────────────────────────
        ph_base.metric("📐 Baseline",    f"{baseline:.0f} cm")
        ph_dist.metric("📡 Distance",    f"{dist} cm")
        ph_dev.metric("↕️ Deviation",    f"{dev:+.1f} cm",
                      delta=f"{dev:+.1f}", delta_color="inverse")
        ph_str.metric("📶 Strength",     strength)
        ph_temp.metric("🌡️ Temp",        f"{temp}°C")
        ph_count.metric("🕳️ Potholes",  st.session_state.pothole_count)
        ph_bump.metric("🔶 Bumps",       st.session_state.bump_count)
        ph_depth.metric("📏 Last Depth", f"{st.session_state.last_depth} cm")

        # ─── Charts (throttled — Streamlit re-render is ~30ms overhead) ──
        now = time.monotonic()
        if now - last_chart_update >= 0.10:   # update charts at ~10 Hz
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
            last_chart_update = now

        # ─── Frame stats ──────────────────────────────────────────────────
        total = st.session_state.total_frames
        errs  = st.session_state.total_errors
        err_pct = (errs / max(total, 1)) * 100
        stats_ph.caption(
            f"Frames received: **{total}** | "
            f"Errors: **{errs}** ({err_pct:.1f}%) | "
            f"Window: {len(dist_buf)}/{WINDOW_SIZE}"
        )

        # ─── Detection log ────────────────────────────────────────────────
        if st.session_state.pothole_log:
            log_ph.subheader("📋 Detection Log")
            log_ph.dataframe(
                pd.DataFrame(st.session_state.pothole_log).head(50),
                width="stretch",
            )

        # NO time.sleep here — read next frame immediately for max throughput

    # ── Cleanup ───────────────────────────────────────────────────────────────
    lidar.close()
    st.session_state.running = False
    st.info("⏹ Streaming stopped.")
    logger.info("Stopped. Frames=%d Errors=%d",
                st.session_state.total_frames,
                st.session_state.total_errors)


# ── Page entry point ──────────────────────────────────────────────────────────
st.title("🕳️ Pothole Detection — Continuous LiDAR Stream")
st.caption(
    "TF02-Pro · 100 Hz continuous stream · Baseline-deviation · ML-confirmed"
)

model = load_model()
if model is None:
    st.warning(
        "⚠️ `pothole_model.pkl` not found — **rule-only mode**.  "
        "Train the model: `python model_train.py`"
    )

if not st.session_state.calibrated:
    st.session_state.baseline_cm = float(manual_baseline)

st.info(
    f"**How it works:** LiDAR shoots **down** at road.  "
    f"Baseline = **{st.session_state.baseline_cm:.0f} cm**.  \n"
    f"📈 dist − baseline > **+{pot_thresh} cm** → **POTHOLE** (ground dropped).  \n"
    f"📉 dist − baseline < **−{bump_thresh} cm** → **BUMP** (ground rose)."
)

if not st.session_state.running:
    c1, c2 = st.columns([2, 5])
    with c1:
        if st.button("▶ Start Streaming", type="primary"):
            st.session_state.running        = True
            if not bypass_calib:
                st.session_state.calibrated     = False
                st.session_state.calib_readings = []
            st.rerun()

if st.session_state.running:
    run_detection(model)