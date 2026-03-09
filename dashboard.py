"""
dashboard.py
=============
Real-time Pothole Detection — TF02-Pro LiDAR

  Main Streamlit loop (Synchronous):
    • Reads directly from serial sensor inside the main loop
    • Drains buffer automatically on reads
    • Runs detection logic every frame
    • Updates UI at 10 Hz (throttled) to reduce re-render overhead
    • Runs detection logic every frame
    • Updates UI at 10 Hz (throttled) to reduce re-render overhead
    • Detection ALERTS fire immediately (bypass UI throttle)

  Result: distance changes are visible on screen within ~50 ms.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
import logging
import cv2
from collections import deque

from lidar_driver import (
    TF02Pro, LiDARReadError,
    list_ports, FRAME_LEN,
)
from ultrasonic_driver import UltrasonicDriver
from camera_driver import CameraDriver
from model_train import (
    extract_features, WINDOW_SIZE,
    POTHOLE_THRESH, BUMP_THRESH,
    BASELINE_CM as DEFAULT_BASELINE,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("dashboard")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="LiDAR Pothole Detector",
                   page_icon="🕳️")

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_LABELS = {0: "🟢 Flat Road", 1: "🟡 Shallow Pothole",
                2: "🔴 Deep Pothole", 3: "🔶 Speed Bump"}
IS_POTHOLE   = {0: False, 1: True,  2: True,  3: False}
IS_BUMP      = {0: False, 1: False, 2: False, 3: True}
DEEP_THRESH_CM       = 8.0
DETECTION_COOLDOWN_S = 3.0
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
        m = joblib.load("pothole_model.pkl")
        logger.info("RF Model loaded.")
        return m
    except FileNotFoundError:
        return None


# ── Session state ─────────────────────────────────────────────────────────────
BASELINE_WINDOW = 20   # number of readings used to compute rolling baseline

_defaults = {
    "dist_history"      : deque(maxlen=500),
    "us_history"        : deque(maxlen=500),
    "dev_history"       : deque(maxlen=500),
    "str_history"       : deque(maxlen=500),
    "baseline_hist"     : deque(maxlen=500),
    "pothole_count"     : 0,
    "bump_count"        : 0,
    "pothole_log"       : [],
    "running"           : False,
    "confirm_streak"    : 0,
    "last_depth"        : 0.0,
    # Rolling baseline — updated every frame as mean of last 20 readings
    "baseline_cm"       : None,          # None until first 20 readings arrive
    "rolling_baseline_buf": deque(maxlen=BASELINE_WINDOW),
    "calibrated"        : False,         # True once baseline_buf is full
    "recalib_streak"    : 0,             # frames of continuous huge changes to force re-calibration
    "us_dist"           : 0.0,           # Latest ultrasonic reading
    "us_baseline_cm"    : None,
    "us_rolling_buf"    : deque(maxlen=BASELINE_WINDOW),
    "dist_buf"          : [],
    "str_buf"           : [],
    "last_detect_t"     : 0.0,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    available_ports = list_ports()
    lidar_port = st.text_input("LiDAR Port", value="COM3" if "COM" in str(available_ports) else "/dev/ttyUSB0",
                               help="Linux: /dev/ttyUSB0  |  Windows: COM3")
    camera_port = st.text_input("Camera Serial Port", value="COM4" if "COM" in str(available_ports) else "/dev/serial0",
                               help="The port the ESP32-CAM is connected to for serial labels.")
    esp32_cam_url = st.text_input("ESP32-CAM Stream URL", value="http://192.168.1.100:81/stream",
                               help="Paste the IP Stream URL to view the live camera feed.")
    if available_ports:
        st.caption(f"Detected: `{'`, `'.join(available_ports)}`")

    lidar_baud = st.selectbox("Baud Rate", [115200, 9600], index=0)
    send_init  = st.checkbox("Send startup commands", value=True,
                             help="Soft-reset + enable-output + 100Hz on connect")

    st.markdown("---")
    st.subheader("📏 Rolling Baseline")
    _bl = st.session_state.baseline_cm
    _bl_buf_len = len(st.session_state.rolling_baseline_buf)
    if _bl is None:
        st.metric("Live Baseline", f"Warming up … ({_bl_buf_len}/{BASELINE_WINDOW})",
                  help=f"Collecting first {BASELINE_WINDOW} readings to establish baseline.")
    else:
        st.metric("Live Baseline", f"{_bl:.1f} cm",
                  help=f"Rolling mean of last {BASELINE_WINDOW} valid distance readings. "
                       "Updates every frame automatically.")

    st.markdown("---")
    st.subheader("🔍 Detection")
    # TF02-Pro noise at 10 m ≈ ±4 cm → thresholds default just above noise floor
    _NOISE_AT_10M = 4.0
    pot_thresh  = st.number_input("Shallow threshold (cm)", 1.0, 30.0,
                                  value=1.0, step=0.5,
                                  help="Positive deviation to trigger shallow pothole.")
    deep_thresh = st.number_input("Deep threshold (cm)", 1.0, 50.0,
                                  value=float(DEEP_THRESH_CM), step=0.5)
    bump_thresh = st.number_input("Bump threshold (cm)", 1.0, 30.0,
                                  value=1.0, step=0.5,
                                  help="Negative deviation to trigger speed bump.")
    max_plausible = st.number_input("Max Plausible Deviation (cm)", 10.0, 150.0,
                                  value=60.0, step=1.0,
                                  help="Ignore physically impossible deep potholes or tall bumps (e.g. >60cm) caused by sensor glitches.")
    confirm_n   = st.slider("Confirm streak (windows)", 1, 4, 1)
    cooldown_s  = st.number_input("Cooldown (s)", 0.1, 30.0,
                                  value=max(0.1, DETECTION_COOLDOWN_S / 5), step=0.1)

    st.markdown("---")
    st.subheader("🚗 Vehicle Speed")
    speed_kmph = st.number_input("Speed (km/h)", 5, 120, 30, step=5)

    st.markdown("---")
    if st.button("🔄 Reset All (and Stop Sensor)"):
        for k, v in _defaults.items():
            st.session_state[k] = v
        st.session_state.running = False
        st.rerun()


# ── Derived ───────────────────────────────────────────────────────────────────
speed_cm_s       = (speed_kmph * 100_000) / 3600
dist_per_reading = speed_cm_s / 100.0   # cm of road per frame at 100Hz


# ── Helpers ───────────────────────────────────────────────────────────────────

def severity_label(depth_cm):
    for lo, hi, label in SEVERITY_BANDS:
        if lo <= depth_cm < hi:
            return label
    return "—"


def rule_classify(dist_buf, baseline, max_plausible):
    """
    Evaluates the recent history of distances to classify the road state.
    Requires deviations to be present across multiple consecutive frames to 
    reject sudden one-frame noise spikes.
    """
    if len(dist_buf) < 2:
        return 0
        
    # Check the last 2 readings (0.02s at 100Hz) to ensure the spike is sustained (not noise)
    # This allows for extremely fast confirmations of real potholes
    recent_devs = np.array(dist_buf[-2:]) - baseline
    
    # Check for physiologically impossible values (extreme sensor noise tracking)
    # If the reading shows a crater > 40cm deep or a bump > 40cm high, it's noise
    if np.any(np.abs(recent_devs) > max_plausible):
        return 0
    
    # If ANY of the last 3 readings are near zero (flat road), 
    # then the spike was instantaneous noise. Ignore it.
    if np.any((recent_devs >= -bump_thresh) & (recent_devs <= pot_thresh)):
        return 0
        
    # If all recent deviations are deeply negative, it's a solid bump
    if np.all(recent_devs < -bump_thresh):
        return 3
        
    # If all recent deviations are extremely positive, it's a deep pothole
    if np.all(recent_devs > deep_thresh):
        return 2
        
    # If all recent deviations are somewhat positive, it's a shallow pothole
    if np.all(recent_devs > pot_thresh):
        return 1
        
    return 0


def compute_dimensions(dist_buf, str_buf, baseline):
    arr      = np.array(dist_buf, dtype=float)
    dev      = arr - baseline
    
    # Compute max absolute deviation (depth of pothole OR height of bump)
    depth_cm = float(max(abs(dev.min()), dev.max()))
    
    # Count frames in anomaly (both positive and negative)
    in_anomaly  = int(np.sum((dev > pot_thresh) | (dev < -bump_thresh)))
    length   = round(in_anomaly * dist_per_reading, 1)
    
    return {
        "depth_cm"    : round(depth_cm, 1),
        "length_cm"   : length,
        "width_cm"    : round(length * 0.8, 1),
        "severity"    : severity_label(depth_cm),
        "avg_strength": round(float(np.mean(str_buf)) if str_buf else 0, 0),
    }


# ── Diagnostic panel ─────────────────────────────────────────────────────────

def show_diagnostic():
    st.subheader("🔬 Hardware Diagnostic")
    c1, c2 = st.columns(2)

    with c1:
        if st.button("🔍 Raw Byte Test", type="primary"):
            try:
                with st.spinner("Opening port and reading raw bytes …"):
                    with TF02Pro(port=lidar_port, baudrate=lidar_baud,
                                 timeout=0.5, send_init=send_init) as lidar:
                        raw = lidar.diagnostic_raw_dump(90)

                if not raw:
                    st.error(
                        "❌ **0 bytes received** — sensor TX not reaching adapter.\n\n"
                        "**Check:** Sensor TX → Adapter RX wiring. Try baud 9600."
                    )
                elif b'\x59\x59' in raw:
                    st.success(f"✅ **{len(raw)} bytes received** with `59 59` header — sensor OK!")
                    groups = [raw[i:i+9] for i in range(0, len(raw), 9)]
                    lines  = []
                    for i, g in enumerate(groups):
                        h  = g.hex(' ')
                        ok = (len(g) == 9 and g[0] == 0x59 and g[1] == 0x59
                              and (sum(g[:8]) & 0xFF) == g[8])
                        d  = (g[2] | g[3] << 8) if len(g) >= 4 else "?"
                        lines.append(f"[{i:02d}] {h:<27} {'← dist=' + str(d) + ' cm' if ok else ''}")
                    st.code("\n".join(lines))
                else:
                    st.warning(
                        f"⚠️ **{len(raw)} bytes received** but no `59 59` header.\n\n"
                        "Data is arriving but baud rate is wrong. **Try 9600.**"
                    )
            except Exception as exc:
                st.error(f"❌ Port error: {exc}")

    with c2:
        if esp32_cam_url:
            st.markdown(f"**ESP32-CAM Live Feed:**")
            st.markdown(
                f'<img src="{esp32_cam_url}" width="100%" style="border-radius:10px; border:2px solid gray;">',
                unsafe_allow_html=True,
            )
        else:
            st.info("ℹ️ Enter ESP32-CAM Stream URL in the sidebar to view video.")


# ── Main detection loop ───────────────────────────────────────────────────────

def run_detection(model):
    # ── Open LiDAR & Ultrasonic ─────────────────────────────────────────────
    lidar = None
    ultrasonic = None
    try:
        lidar = TF02Pro(port=lidar_port, baudrate=lidar_baud, send_init=send_init)
    except Exception as exc:
        st.error(f"❌ Cannot open `{lidar_port}`: {exc}")
        st.session_state.running = False
        show_diagnostic()
        return
        
    try:
        ultrasonic = UltrasonicDriver(trig_pin=23, echo_pin=24)
        status_ph = st.empty()
        status_ph.success(f"✅ GPIO Ultrasonic Driver initialized.")
    except Exception as exc:
        st.warning(f"⚠️ Ultrasonic GPIO initialization failed: {exc}. Running LiDAR only.")
        
    camera = None
    try:
        camera = CameraDriver(stream_url=esp32_cam_url)
        st.success("📸 Camera AI initialized over Wi-Fi.")
    except Exception as exc:
        st.warning(f"⚠️ Camera AI initialization failed: {exc}. Running without camera label verification.")

    # ── UI layout ─────────────────────────────────────────────────────────────
    hdr_l, hdr_r = st.columns([6, 1])
    hdr_l.success(
        f"✅ **Streaming** `{lidar_port}` @ {lidar_baud} baud  "
        f"| Direct synchronous polling"
    )
    stop_btn = hdr_r.button("⏹ Stop")

    kc = st.columns(10)
    ph_base  = kc[0].empty(); ph_dist = kc[1].empty()
    ph_us    = kc[2].empty(); ph_dev  = kc[3].empty(); ph_cam = kc[4].empty()
    ph_str   = kc[5].empty(); ph_temp = kc[6].empty()
    ph_cnt   = kc[7].empty(); ph_bump = kc[8].empty(); ph_dep = kc[9].empty()

    st.markdown("---")
    
    col_cam, col_status = st.columns([1, 2])
    with col_cam:
        st.markdown("**ESP32-CAM AI Feed**")
        cam_video_ph = st.empty()
        cam_text_ph = st.empty()
        
    with col_status:
        status_ph = st.empty()

    st.markdown("### 📡 Live Stream")
    ca, cb = st.columns(2)
    chart_dist = ca.empty()
    chart_dev  = cb.empty()
    chart_str  = st.empty()
    stats_ph   = st.empty()

    st.markdown("---")
    log_ph = st.empty()

    # Local variables
    dist_buf  = st.session_state.dist_buf
    str_buf   = st.session_state.str_buf

    last_ui_t = 0.0
    UI_INTERVAL = 0.10     # update UI at 10 Hz
    no_data_count = 0
    frames_count = 0
    errors_count = 0
    consec_errors = 0

    logger.info("Detection started (synchronous). Baseline=%.1f cm",
                st.session_state.baseline_cm if st.session_state.baseline_cm else 0.0)

    try:
        # ── Main loop ─────────────────────────────────────────────────────────────
        while not stop_btn:

            # ── GET LATEST ULTRASONIC FRAME (Non-blocking) ──
            if ultrasonic:
                us_val = ultrasonic.read_frame_current()
                if us_val is not None:
                    st.session_state.us_dist = us_val

            # ── GET LATEST LIDAR FRAME directly from serial ──
            try:
                frame = lidar.read_frame_current()
                frames_count += 1
                consec_errors = 0
            except LiDARReadError as exc:
                errors_count += 1
                consec_errors += 1
                no_data_count += 1

                if consec_errors == 3:
                    logger.warning("Soft recovery …")
                    lidar._enable_output()
                elif consec_errors >= 6:
                    logger.warning("Hard reconnect …")
                    lidar.reconnect()
                    consec_errors = 0

                status_ph.warning(
                    f"⏳ Waiting for sensor frames …  "
                    f"Poll #{no_data_count} | Errors: {errors_count} | "
                    f"Frames: {frames_count}  \n\n"
                    f"{'🔴 Sensor silent — check USB connection & power' if consec_errors > 4 else '🟡 Reading error …'}"
                )
                time.sleep(0.05)
                continue

            no_data_count = 0
            dist     = frame["distance_cm"]
            strength = frame["strength"]
            temp     = frame["temperature_c"]
            valid    = frame.get("valid", True)

            # Skip out-of-range readings — show warning, don't affect detection
            if not valid:
                status_ph.warning(
                    f"⚠️ Reading out of range: **{dist} cm** "
                    f"(valid range 1–2200 cm) — skipping frame"
                )
                time.sleep(0.02)
                continue

            # ── FIXED BASELINE LOGIC ──────────────────────────────────────────────────
            # Skip out-of-range readings for calibration
            if not st.session_state.calibrated:
                st.session_state.rolling_baseline_buf.append(dist)
                if ultrasonic and st.session_state.us_dist > 0:
                    st.session_state.us_rolling_buf.append(st.session_state.us_dist)
                    
                if len(st.session_state.rolling_baseline_buf) == BASELINE_WINDOW:
                    st.session_state.baseline_cm  = float(
                        np.mean(st.session_state.rolling_baseline_buf)
                    )
                    if len(st.session_state.us_rolling_buf) > 0:
                        st.session_state.us_baseline_cm = float(np.mean(st.session_state.us_rolling_buf))
                        
                    st.session_state.calibrated = True
                    st.session_state.recalib_streak = 0
            
            # Skip detection until baseline is fully established
            if not st.session_state.calibrated:
                _n = len(st.session_state.rolling_baseline_buf)
                status_ph.info(
                    f"⏳ Establishing locked baseline … ({_n}/{BASELINE_WINDOW} readings) "
                    f"| Current dist: **{dist} cm**"
                )
                time.sleep(0.02)
                continue

            baseline = st.session_state.baseline_cm
            dev      = dist - baseline
            
            # ── RECALIBRATION ON HUGE SHIFT ──────────────────────────────────────────
            # If the road instantly drops or rises significantly and STAYS there
            if abs(dev) > max_plausible:
                st.session_state.recalib_streak += 1
                if st.session_state.recalib_streak >= 50: # 0.5s of continuous huge dev
                    status_ph.warning(f"🔄 Massive shift detected (>40cm). Recalibrating baseline...")
                    st.session_state.calibrated = False
                    st.session_state.rolling_baseline_buf.clear()
                    st.session_state.us_rolling_buf.clear()
                    st.session_state.recalib_streak = 0
                    time.sleep(0.05)
                    continue
            else:
                st.session_state.recalib_streak = 0

            # Update histories
            st.session_state.dist_history.append(dist)
            st.session_state.dev_history.append(dev)
            st.session_state.str_history.append(strength)
            st.session_state.baseline_hist.append(baseline)
            st.session_state.us_history.append(st.session_state.us_dist)

            # Sliding window
            dist_buf.append(dist)
            str_buf.append(strength)
            if len(dist_buf) > WINDOW_SIZE:
                dist_buf.pop(0)
                str_buf.pop(0)

            # ── Classify ──────────────────────────────────────────────────────────
            rule_cls = rule_classify(dist_buf, baseline, max_plausible)
            
            ml_conf = 0.0
            final_cls = rule_cls

            if len(dist_buf) == WINDOW_SIZE and model is not None:
                # Random Forest Inference fallback
                feats   = extract_features(
                    np.array(dist_buf), np.array(str_buf), baseline
                ).reshape(1, -1)
                ml_cls  = int(model.predict(feats)[0])
                rf_conf = float(model.predict_proba(feats)[0][ml_cls])
                if rf_conf >= 0.55:
                    final_cls = ml_cls
                    ml_conf = rf_conf

            # ── OFF-AXIS FUSION OVERRIDE ──
            # The LiDAR beam is tiny (mm). The Ultrasonic is wide (cm).
            # If you steer slightly off, LiDAR sees Flat Road, but Ultrasonic sees the Pothole.
            # We override the Flat Road classification if Ultrasonic shows a massive deviation.
            if final_cls == 0 and ultrasonic and st.session_state.us_baseline_cm is not None and st.session_state.us_dist > 0:
                us_dev = st.session_state.us_dist - st.session_state.us_baseline_cm
                if us_dev > 5.0: # > 5cm deep off-axis hole
                    final_cls = 1
                    ml_conf = 0.0 # Force override status
                elif us_dev < -5.0: # > 5cm tall off-axis bump
                    final_cls = 3
                    ml_conf = 0.0

            is_ph   = IS_POTHOLE.get(final_cls, False)
            is_bump = IS_BUMP.get(final_cls, False)
            cam_label = camera.get_latest_label() if camera else ""

            if is_ph or is_bump:
                # ── SENSOR FUSION VERIFICATION ──
                verified = True
                
                cam_pothole = "pothole" in cam_label.lower()
                cam_bump = "bump" in cam_label.lower()

                us_agrees = False
                us_contradicts = False
                
                if ultrasonic and st.session_state.us_baseline_cm is not None and st.session_state.us_dist > 0:
                    us_dev = st.session_state.us_dist - st.session_state.us_baseline_cm
                    
                    if is_ph:
                        if us_dev >= 2.0: us_agrees = True
                        else: us_contradicts = True
                    elif is_bump:
                        if us_dev <= -2.0: us_agrees = True
                        else: us_contradicts = True
                
                # 1. If Ultrasonic is running and contradicts LiDAR, veto.
                if us_contradicts:
                    verified = False
                    
                # 2. If camera actively disagrees with a small deviation, veto it.
                # However, if camera simply doesn't see anything, we still want to log the anomaly
                # if the LiDAR is confident. So we don't strictly veto *just* because camera is missing.
                elif camera and not (cam_pothole if is_ph else cam_bump):
                    if not us_agrees and abs(dev) < 3.0: # Only veto very tiny deviations
                        # Veto if it's a tiny deviation and there's no backup from US or Camera
                        verified = False
                        
                if verified:
                    st.session_state.confirm_streak += 1
                else:
                    st.session_state.confirm_streak = 0
            else:
                st.session_state.confirm_streak = 0

            # ── Confirmed detection ────────────────────────────────────────────────
            if st.session_state.confirm_streak >= confirm_n:
                now_t   = time.monotonic()
                elapsed = now_t - st.session_state.last_detect_t

                if elapsed < cooldown_s:
                    remaining = cooldown_s - elapsed
                    # In cooldown: show warning but don't log
                    status_ph.warning(
                        f"🟡 **{CLASS_LABELS[final_cls]}** continuing "
                        f"(cooldown ⏳ {remaining:.1f}s) — "
                        f"dev: **{dev:+.1f} cm**"
                    )
                    # Don't reset streak here, otherwise a constant anomaly will NEVER log
                else:
                    # === FIRE ALERT (bypasses UI throttle) ===
                    dims = compute_dimensions(dist_buf, str_buf, baseline)
                    if is_ph:   
                        st.session_state.pothole_count += 1
                        st.session_state.last_depth = dims["depth_cm"]
                    elif is_bump: 
                        st.session_state.bump_count  += 1
                        st.session_state.last_depth = dims["depth_cm"]

                    st.session_state.confirm_streak  = 0
                    st.session_state.last_detect_t   = now_t

                    half = len(dist_buf) // 2
                    dist_buf[:] = dist_buf[half:]
                    str_buf[:]  = str_buf[half:]

                    log_entry = {
                        "Time"       : time.strftime("%H:%M:%S"),
                        "Type"       : CLASS_LABELS[final_cls],
                        "Dev (cm)"   : f"{dev:+.1f}",
                        "Depth (cm)" : dims["depth_cm"],
                        "Length (cm)": dims["length_cm"],
                        "Width (cm)" : dims["width_cm"],
                        "Severity"   : dims["severity"],
                        "US (cm)"    : st.session_state.us_dist,
                        "Cam"        : cam_label if camera else "N/A",
                        "Conf."      : f"{ml_conf:.0%}" if ml_conf else "rule",
                        "Strength"   : int(dims["avg_strength"]),
                        "Baseline"   : f"{baseline:.0f}",
                    }
                    st.session_state.pothole_log.insert(0, log_entry)

                    # Alert fires immediately
                    status_ph.error(
                        f"🔴 **{CLASS_LABELS[final_cls]} CONFIRMED** — "
                        f"dev: **{dev:+.1f} cm** | "
                        f"depth: **{dims['depth_cm']} cm** | "
                        f"{dims['severity']}"
                    )
                    logger.info("DETECTED: %s", log_entry)
            else:
                # Live status (throttled below with UI updates)
                sb    = ("█" * st.session_state.confirm_streak
                         + "░" * max(0, confirm_n - st.session_state.confirm_streak))
                dev_s = f"{dev:+.1f}"
                if final_cls == 0:
                    status_ph.success(
                        f"🟢 Flat Road — dist: **{dist} cm** | "
                        f"dev: {dev_s} cm | str: {strength} | temp: {temp}°C"
                    )
                elif is_ph:
                    status_ph.warning(
                        f"🟡 {CLASS_LABELS[final_cls]} — "
                        f"dist: **{dist} cm** | dev: **{dev_s} cm** | "
                        f"streak [{sb}] {st.session_state.confirm_streak}/{confirm_n}"
                    )
                else:
                    status_ph.info(
                        f"🔶 Speed Bump — dist: **{dist} cm** | dev: **{dev_s} cm** | "
                        f"streak [{sb}] {st.session_state.confirm_streak}/{confirm_n}"
                    )

            # ── UI UPDATE (throttled to 10 Hz) ────────────────────────────────────
            now = time.monotonic()
            if now - last_ui_t >= UI_INTERVAL:
                ph_base.metric("📐 Baseline",    f"{baseline:.0f} cm")
                ph_dist.metric("📡 LiDAR",    f"{dist} cm")
                ph_us.metric("🔊 US Dist",      f"{st.session_state.us_dist} cm")
                ph_dev.metric("↕️ Deviation",   f"{dev:+.1f} cm",
                              delta=f"{dev:+.1f}", delta_color="inverse")
                if camera: 
                    cam_text_ph.markdown(f"**📸 Camera AI:** `{cam_label}`")
                    frame = camera.get_latest_frame()
                    if frame is not None:
                        # OpenCV uses BGR, Streamlit uses RGB
                        cam_video_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
                    else:
                        cam_video_ph.info("⏳ Waiting for video stream...")
                ph_str.metric("📶 Strength",     strength)
                ph_temp.metric("🌡️ Temp",        f"{temp}°C")
                ph_cnt.metric("🕳️ Potholes",    st.session_state.pothole_count)
                ph_bump.metric("🔶 Bumps",       st.session_state.bump_count)
                ph_dep.metric("📏 Last Depth",  f"{st.session_state.last_depth} cm")

                chart_dist.line_chart(pd.DataFrame({
                    "Distance (cm)": list(st.session_state.dist_history),
                    "Baseline (cm)": list(st.session_state.baseline_hist),
                    "Ultrasonic (cm)": list(st.session_state.us_history),
                }))
                chart_dev.line_chart(pd.DataFrame({
                    "Deviation (cm)": list(st.session_state.dev_history),
                }))
                chart_str.line_chart(pd.DataFrame({
                    "Signal Strength": list(st.session_state.str_history),
                }))

                stats_ph.caption(
                    f"Sensor frames: **{frames_count}** | "
                    f"Read errors: **{errors_count}** | "
                    f"Window: {len(dist_buf)}/{WINDOW_SIZE} | "
                    f"Streak: {st.session_state.confirm_streak}/{confirm_n} | "
                    f"Cooldown: {cooldown_s:.0f}s"
                )

                if st.session_state.pothole_log:
                    with log_ph.container():
                        st.subheader("📋 Detection Log")
                        st.dataframe(
                            pd.DataFrame(st.session_state.pothole_log).head(50),
                            width="stretch",
                        )

                last_ui_t = now

            time.sleep(0.02)   # 50 Hz poll rate — very fast, UI still only at 10 Hz

    finally:
        # ── Cleanup ────────────────────────────────────────────────────────
        # Due to Streamlit RerunExceptions triggered by UI interaction, 
        # this block perfectly ensures the serial port is closed.
        try:
            lidar.close()
            if ultrasonic: ultrasonic.close()
            if camera: camera.close()
        except Exception:
            pass
        
    # Only close if user explicitly clicked the Stop button
    if stop_btn:
        st.session_state.running = False
        st.info("⏹ Stopped.")


# ── Entry point ───────────────────────────────────────────────────────────────
st.title("🕳️ Pothole Detection — TF02-Pro LiDAR")
st.caption("Background thread · 100 Hz sensor · 10 Hz UI · Instant distance updates")

model = load_model()
if model is None:
    st.warning("⚠️ `pothole_model.pkl` not found — rule-only mode. "
               "Run `python model_train.py` to enable ML.")

if not st.session_state.running:
    c1, _ = st.columns([2, 5])
    with c1:
        if st.button("▶ Start Monitoring", type="primary"):
            st.session_state.running = True
            st.rerun()

    st.markdown("---")
    show_diagnostic()

if st.session_state.running:
    run_detection(model)