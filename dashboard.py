"""
dashboard.py
=============
Real-time Pothole Detection — TF02-Pro LiDAR
Includes hardware diagnostic panel to debug "sensor not sending data" issues.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
import logging
from collections import deque

from lidar_driver import TF02Pro, LiDARReadError, list_ports
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
                2: "🔴 Deep Pothole",  3: "🔶 Speed Bump"}
IS_POTHOLE   = {0: False, 1: True,  2: True,  3: False}
IS_BUMP      = {0: False, 1: False, 2: False, 3: True}
FRAME_RATE_HZ       = 100
DETECTION_COOLDOWN_S = 3.0   # Seconds between allowed consecutive detections
                               # Prevents repeat-fire while object is held still.
                               # At 30 km/h a vehicle travels 25 m in 3 s — plenty.
DEEP_THRESH_CM      = 8.0    # Deviation above this → Deep Pothole (class 2)
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


# ── Session state ─────────────────────────────────────────────────────────────
_defaults: dict = {
    "dist_history"      : deque(maxlen=500),
    "dev_history"       : deque(maxlen=500),
    "str_history"       : deque(maxlen=500),
    "baseline_hist"     : deque(maxlen=500),
    "pothole_count"     : 0,
    "bump_count"        : 0,
    "pothole_log"       : [],
    "running"           : False,
    "confirm_streak"    : 0,
    "last_label"        : "—",
    "last_depth"        : 0.0,
    "calib_readings"    : [],
    "baseline_cm"       : float(DEFAULT_BASELINE),
    "calibrated"        : False,
    "dist_buf"          : [],
    "str_buf"           : [],
    "total_frames"      : 0,
    "total_errors"      : 0,
    "last_detection_t"  : 0.0,   # monotonic time of last confirmed detection
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    # Auto-detect available ports
    available_ports = list_ports()
    default_port = "/dev/ttyUSB0"

    lidar_port = st.text_input(
        "Serial Port",
        value=default_port,
        help="Linux: /dev/ttyUSB0  |  Windows: COM3, COM4 …",
    )
    if available_ports:
        st.caption(f"Detected ports: `{'`, `'.join(available_ports)}`")
    else:
        st.caption("⚠️ No serial ports detected on this machine")

    lidar_baud = st.selectbox("Baud Rate", [115200, 9600, 19200, 56000], index=0)

    send_init = st.checkbox(
        "Send startup commands to sensor",
        value=True,
        help="Sends soft-reset + enable-output + 100Hz commands on connect. "
             "Fixes 'sensor not sending data' when sensor is in trigger mode.",
    )

    st.markdown("---")
    st.subheader("📏 Baseline")
    manual_baseline = st.number_input(
        "Road distance (cm)", 10, 2000,
        value=int(st.session_state.baseline_cm), step=5,
    )
    calib_n = st.slider("Calibration samples", 5, 60, 20)
    bypass_calib = st.checkbox("Skip calibration (use manual baseline)", value=False)

    st.markdown("---")
    st.subheader("🔍 Detection")
    pot_thresh  = st.number_input("Shallow pothole threshold (cm)", 1.0, 30.0,
                                  value=float(POTHOLE_THRESH), step=0.5,
                                  help="Deviation above baseline = shallow pothole start")
    deep_thresh = st.number_input("Deep pothole threshold (cm)", 1.0, 50.0,
                                  value=float(DEEP_THRESH_CM), step=0.5,
                                  help="Deviation above this = Deep Pothole (class 2)")
    bump_thresh = st.number_input("Bump threshold (cm)", 1.0, 30.0,
                                  value=float(BUMP_THRESH), step=0.5)
    confirm_n   = st.slider("Confirm streak (windows)", 1, 4, 2)
    cooldown_s  = st.number_input(
        "Detection cooldown (s)", 0.5, 30.0,
        value=DETECTION_COOLDOWN_S, step=0.5,
        help="Minimum seconds between consecutive detections. "
             "Prevents repeat-fire when sensor stays over same anomaly."
    )

    st.markdown("---")
    st.subheader("🚗 Vehicle Speed")
    speed_kmph = st.number_input("Speed (km/h)", 5, 120, 30, step=5)

    st.markdown("---")
    if st.button("🔄 Reset All"):
        for k, v in _defaults.items():
            st.session_state[k] = v
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


def compute_dimensions(dist_buf, str_buf, baseline) -> dict:
    arr         = np.array(dist_buf, dtype=float)
    dev         = arr - baseline
    depth_cm    = float(max(dev.max(), 0))
    in_hole     = int(np.sum(dev > pot_thresh))
    length_cm   = round(in_hole * dist_per_reading, 1)
    return {
        "depth_cm"    : round(depth_cm, 1),
        "length_cm"   : length_cm,
        "width_cm"    : round(length_cm * 0.8, 1),
        "severity"    : severity_label(depth_cm),
        "avg_strength": round(float(np.mean(str_buf)) if str_buf else 0, 0),
    }


def rule_classify(dist_cm, baseline) -> int:
    """
    Instant single-reading rule classification.
    Returns:
      0 = Flat road
      1 = Shallow pothole   (+pot_thresh  to +deep_thresh cm deviation)
      2 = Deep pothole      (> +deep_thresh cm deviation)
      3 = Speed bump        (< -bump_thresh cm deviation)
    """
    dev = dist_cm - baseline
    if dev > deep_thresh:  return 2   # deep pothole ← FIX: was always returning 1
    if dev > pot_thresh:   return 1   # shallow pothole
    if dev < -bump_thresh: return 3   # speed bump
    return 0                          # flat road


# ── DIAGNOSTIC PANEL ─────────────────────────────────────────────────────────

def show_diagnostic():
    st.subheader("🔬 Hardware Diagnostic")
    st.markdown("""
Use this panel to verify the LiDAR sensor is **physically sending data** before starting detection.
It opens the port, reads raw bytes (no parsing), and shows you what the sensor is actually transmitting.
""")

    st.markdown("""
**Common causes of "Expected 1 bytes, got 0":**
| Cause | What you see | Fix |
|---|---|---|
| Sensor in **Trigger mode** | 0 bytes in diagnostic | Enable **"Send startup commands"** in sidebar |
| **TX wire not connected** | 0 bytes in diagnostic | Check wiring: Sensor TX → UART Adapter RX |
| **Wrong baud rate** | Garbled bytes (not `59 59`) | Try 9600 or auto-scan |
| **Sensor output disabled** | 0 bytes in diagnostic | Enable **"Send startup commands"** in sidebar |
| Adapter **RX/TX swapped** | 0 bytes in diagnostic | Swap TX↔RX wires |
""")

    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("🔍 Raw Byte Diagnostic", type="primary"):
            try:
                with st.spinner(f"Opening {lidar_port} and reading raw bytes …"):
                    lidar = TF02Pro(port=lidar_port, baudrate=lidar_baud,
                                    timeout=0.5, send_init=send_init)
                    raw = lidar.diagnostic_raw_dump(n_bytes=90)
                    lidar.close()

                if not raw:
                    st.error(
                        f"❌ **Received 0 bytes** from `{lidar_port}` at {lidar_baud} baud.\n\n"
                        "**The sensor IS NOT sending any data.**\n\n"
                        "**Most likely cause:** TX wire from sensor not reaching UART adapter RX.\n\n"
                        "**Check these:**\n"
                        "1. Enable **'Send startup commands'** in the sidebar (toggles sensor output mode)\n"
                        "2. Verify **Sensor TX → Adapter RX** wiring\n"
                        "3. Try baud rate **9600** instead of 115200\n"
                        "4. Try a different USB port or adapter"
                    )
                else:
                    hex_str   = raw.hex(' ')
                    has_hdr   = b'\x59\x59' in raw
                    n_frames  = hex_str.count('59 59')

                    if has_hdr:
                        st.success(
                            f"✅ **Received {len(raw)} bytes** — "
                            f"found `59 59` header **{n_frames} time(s)**. "
                            f"Sensor is streaming correctly!"
                        )
                    else:
                        st.warning(
                            f"⚠️ **Received {len(raw)} bytes** but NO `59 59` header found.\n\n"
                            f"This means the sensor IS connected but baud rate is likely wrong.\n"
                            f"**Try baud rate 9600.**"
                        )

                    st.markdown("**Raw bytes (hex):**")
                    # Show in 9-byte groups (one frame per line)
                    groups = [raw[i:i+9] for i in range(0, len(raw), 9)]
                    lines  = []
                    for i, g in enumerate(groups):
                        h = g.hex(' ')
                        ok = (len(g) == 9 and g[0] == 0x59 and g[1] == 0x59
                              and (sum(g[:8]) & 0xFF) == g[8])
                        dist = (g[2] | g[3] << 8) if len(g) >= 4 else None
                        tag = f"← VALID frame, dist={dist} cm" if ok and dist else ""
                        lines.append(f"[{i:02d}] {h:<27} {tag}")
                    st.code('\n'.join(lines), language="")

            except Exception as exc:
                st.error(f"❌ Could not open port: **{exc}**")
                if available_ports:
                    st.info(f"Available ports: `{'`, `'.join(available_ports)}`")

    with c2:
        if st.button("📡 Test Single Reading"):
            try:
                with st.spinner("Reading one frame …"):
                    lidar = TF02Pro(port=lidar_port, baudrate=lidar_baud,
                                    timeout=0.5, send_init=send_init)
                    r = lidar.read_frame()
                    lidar.close()

                st.success("✅ **Frame received!**")
                st.json({
                    "distance_cm"  : r["distance_cm"],
                    "strength"     : r["strength"],
                    "temperature_c": r["temperature_c"],
                    "valid"        : r["valid"],
                })
            except LiDARReadError as exc:
                st.error(f"❌ Read failed: {exc}")
            except Exception as exc:
                st.error(f"❌ Port error: {exc}")

    with c3:
        if st.button("🔁 Rescan Serial Ports"):
            st.rerun()


# ── Detection loop ────────────────────────────────────────────────────────────

def run_detection(model):
    try:
        lidar = TF02Pro(port=lidar_port, baudrate=lidar_baud,
                        timeout=0.5, send_init=send_init)
    except Exception as exc:
        st.error(
            f"❌ Cannot open `{lidar_port}`: {exc}\n\n"
            f"**Available ports:** `{'`, `'.join(available_ports) or 'none detected'}`\n\n"
            "Use the **Hardware Diagnostic** below to troubleshoot."
        )
        st.session_state.running = False
        show_diagnostic()
        return

    # ── Layout ───────────────────────────────────────────────────────────────
    hdr_l, hdr_r = st.columns([6, 1])
    hdr_l.success(
        f"✅ Streaming from `{lidar_port}` @ {lidar_baud} baud  "
        f"| Init commands: {'✓ sent' if send_init else '✗ skipped'}"
    )
    stop_btn = hdr_r.button("⏹ Stop")

    kc = st.columns(8)
    ph_base  = kc[0].empty(); ph_dist  = kc[1].empty()
    ph_dev   = kc[2].empty(); ph_str   = kc[3].empty()
    ph_temp  = kc[4].empty(); ph_count = kc[5].empty()
    ph_bump  = kc[6].empty(); ph_depth = kc[7].empty()

    st.markdown("---")
    status_ph = st.empty()
    calib_ph  = st.empty()

    st.markdown("### 📡 Live Stream")
    ca, cb = st.columns(2)
    chart_dist = ca.empty()
    chart_dev  = cb.empty()
    chart_str  = st.empty()
    stats_ph   = st.empty()

    st.markdown("---")
    log_ph = st.empty()

    # Local buffer refs
    dist_buf     = st.session_state.dist_buf
    str_buf      = st.session_state.str_buf
    consec_err   = 0
    last_ui_t    = 0.0   # last time ALL Streamlit UI was updated
    UI_HZ        = 10    # target UI refresh rate (Hz)
    UI_INTERVAL  = 1.0 / UI_HZ   # 100 ms between UI updates
    # Detection alerts bypass the throttle and fire immediately

    if bypass_calib and not st.session_state.calibrated:
        st.session_state.baseline_cm = float(manual_baseline)
        st.session_state.calibrated  = True
        calib_ph.success(f"✅ Manual baseline: **{manual_baseline} cm**")

    logger.info("Streaming started. init_commands=%s", send_init)

    # ── Main loop ─────────────────────────────────────────────────────────────
    while not stop_btn:

        # READ FRESHEST FRAME (auto-drains stale OS-buffer backlog)
        try:
            frame = lidar.read_frame_current()
            consec_err = 0
            st.session_state.total_frames += 1
        except LiDARReadError as exc:
            consec_err += 1
            st.session_state.total_errors += 1
            logger.warning("Frame error #%d: %s", consec_err, exc)

            if consec_err == 5:
                status_ph.warning(
                    f"⚠️ **{consec_err} consecutive read errors.**  \n"
                    f"Last error: `{exc}`  \n\n"
                    f"**If this continues:**  \n"
                    f"1. Stop monitoring  \n"
                    f"2. Run the **Hardware Diagnostic** panel below  \n"
                    f"3. Make sure **'Send startup commands'** is enabled in the sidebar"
                )
            elif consec_err >= 30:
                st.error(
                    "❌ **Too many consecutive errors — stopping.**\n\n"
                    "The sensor is not sending data. "
                    "Open the **Hardware Diagnostic** panel to investigate."
                )
                break
            continue

        dist     = frame["distance_cm"]
        strength = frame["strength"]
        temp     = frame["temperature_c"]

        # CALIBRATION
        if not st.session_state.calibrated:
            st.session_state.calib_readings.append(dist)
            done = len(st.session_state.calib_readings)

            calib_ph.info(
                f"🔵 **Calibrating …** {done}/{calib_n} — "
                f"dist: **{dist} cm** | str: {strength} | temp: {temp}°C — "
                f"Keep sensor over flat ground."
            )

            st.session_state.dist_history.append(dist)
            st.session_state.dev_history.append(dist - st.session_state.baseline_cm)
            st.session_state.str_history.append(strength)
            st.session_state.baseline_hist.append(st.session_state.baseline_cm)

            if done >= calib_n:
                cal = sorted(st.session_state.calib_readings)
                st.session_state.baseline_cm = float(cal[len(cal) // 2])
                st.session_state.calibrated  = True
                calib_ph.success(
                    f"✅ Baseline locked: **{st.session_state.baseline_cm:.1f} cm** "
                    f"(median of {done} readings)"
                )

            ph_base.metric("📐 Baseline", f"{st.session_state.baseline_cm:.0f} cm")
            ph_dist.metric("📡 Distance", f"{dist} cm")
            ph_dev.metric("↕️ Deviation",
                          f"{dist - st.session_state.baseline_cm:+.1f} cm")
            ph_str.metric("📶 Strength", strength)
            ph_temp.metric("🌡️ Temp", f"{temp}°C")
            ph_count.metric("🕳️ Potholes", st.session_state.pothole_count)
            ph_bump.metric("🔶 Bumps",     st.session_state.bump_count)
            ph_depth.metric("📏 Last Depth", f"{st.session_state.last_depth} cm")
            continue

        # ACTIVE DETECTION
        baseline = st.session_state.baseline_cm
        dev      = dist - baseline

        st.session_state.dist_history.append(dist)
        st.session_state.dev_history.append(dev)
        st.session_state.str_history.append(strength)
        st.session_state.baseline_hist.append(baseline)

        dist_buf.append(dist)
        str_buf.append(strength)
        if len(dist_buf) > WINDOW_SIZE:
            dist_buf.pop(0)
            str_buf.pop(0)

        rule_cls = rule_classify(dist, baseline)

        if len(dist_buf) == WINDOW_SIZE and model is not None:
            feats   = extract_features(
                np.array(dist_buf), np.array(str_buf), baseline
            ).reshape(1, -1)
            ml_cls  = int(model.predict(feats)[0])
            ml_conf = float(model.predict_proba(feats)[0][ml_cls])
            final_cls = ml_cls if ml_conf >= 0.55 else rule_cls
        else:
            final_cls = rule_cls
            ml_conf   = 0.0

        is_ph   = IS_POTHOLE.get(final_cls, False)
        is_bump = IS_BUMP.get(final_cls, False)

        if is_ph or is_bump:
            st.session_state.confirm_streak += 1
        else:
            st.session_state.confirm_streak = 0

        if st.session_state.confirm_streak >= confirm_n:
            # ── COOLDOWN GATE ───────────────────────────────────────────────────
            # If a detection was just logged recently, skip this one.
            # Prevents repeat-fire when sensor is held still over an anomaly
            # (real potholes last < 0.5 s at driving speeds).
            now_t   = time.monotonic()
            elapsed = now_t - st.session_state.last_detection_t
            in_cooldown = elapsed < cooldown_s

            if in_cooldown:
                # Still in cooldown: show countdown but don’t log again
                remaining = cooldown_s - elapsed
                status_ph.warning(
                    f"🟡 **{CLASS_LABELS[final_cls]}** (cooldown ⏳ {remaining:.1f}s) — "
                    f"dev: **{dev:+.1f} cm** | same anomaly, not re-logged"
                )
                st.session_state.confirm_streak = 0
            else:
                # ── CONFIRMED NEW DETECTION ───────────────────────────
                dims = compute_dimensions(dist_buf, str_buf, baseline)
                if is_ph:
                    st.session_state.pothole_count += 1
                elif is_bump:
                    st.session_state.bump_count += 1

                st.session_state.confirm_streak   = 0
                st.session_state.last_detection_t = now_t
                st.session_state.last_depth       = dims["depth_cm"]
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
                    "Conf."      : f"{ml_conf:.0%}" if ml_conf else "rule",
                    "Strength"   : int(dims["avg_strength"]),
                    "Baseline"   : f"{baseline:.0f}",
                }
                st.session_state.pothole_log.insert(0, log_entry)

                status_ph.error(
                    f"🔴 **{CLASS_LABELS[final_cls]} CONFIRMED** — "
                    f"dev: **{dev:+.1f} cm** | depth: **{dims['depth_cm']} cm** | "
                    f"length: {dims['length_cm']} cm | {dims['severity']} | "
                    f"cooldown: {cooldown_s:.0f}s"
                )
                logger.info("DETECTED: %s", log_entry)
        else:
            sb  = "█" * st.session_state.confirm_streak + "░" * max(0, confirm_n - st.session_state.confirm_streak)
            dev_s = f"{dev:+.1f}"
            if final_cls == 0:
                status_ph.success(
                    f"🟢 Flat Road — dist: **{dist} cm** | dev: {dev_s} cm | "
                    f"str: {strength} | temp: {temp}°C"
                )
            elif is_ph:
                status_ph.warning(
                    f"🟡 {CLASS_LABELS[final_cls]} — dist: **{dist} cm** | "
                    f"dev: **{dev_s} cm** | streak [{sb}] "
                    f"{st.session_state.confirm_streak}/{confirm_n}"
                )
            else:
                status_ph.info(
                    f"🔶 Speed Bump — dist: **{dist} cm** | dev: **{dev_s} cm**"
                )

        # ── THROTTLED UI UPDATE (all Streamlit renders at 10 Hz max) ──────────
        # Detection ALERTS above this block fire immediately (no throttle).
        # KPIs, charts, stats, log are updated at most every 100ms.
        # This keeps the detection loop running fast (≥ 50 Hz) so the OS
        # buffer stays small and distances update almost instantaneously.
        now = time.monotonic()
        queued_bytes = lidar._ser.in_waiting if lidar._ser.is_open else 0
        queued_frames = queued_bytes // FRAME_LEN

        if now - last_ui_t >= UI_INTERVAL:
            # KPIs
            ph_base.metric("📐 Baseline",    f"{baseline:.0f} cm")
            ph_dist.metric("📡 Distance",    f"{dist} cm")
            ph_dev.metric("↕️ Deviation",    f"{dev:+.1f} cm",
                          delta=f"{dev:+.1f}", delta_color="inverse")
            ph_str.metric("📶 Strength",     strength)
            ph_temp.metric("🌡️ Temp",        f"{temp}°C")
            ph_count.metric("🕳️ Potholes",  st.session_state.pothole_count)
            ph_bump.metric("🔶 Bumps",       st.session_state.bump_count)
            ph_depth.metric("📏 Last Depth", f"{st.session_state.last_depth} cm")

            # Charts
            chart_dist.line_chart(pd.DataFrame({
                "Distance (cm)": list(st.session_state.dist_history),
                "Baseline (cm)": list(st.session_state.baseline_hist),
            }))
            chart_dev.line_chart(pd.DataFrame({
                "Deviation (cm)": list(st.session_state.dev_history),
            }))
            chart_str.line_chart(pd.DataFrame({
                "Signal Strength": list(st.session_state.str_history),
            }))

            # Frame stats (shows if buffer is accumulating lag)
            t = st.session_state.total_frames
            e = st.session_state.total_errors
            lag_warn = f" ⚠️ **{queued_frames} frames behind**" if queued_frames > 5 else ""
            stats_ph.caption(
                f"Frames: **{t}** | Errors: **{e}** ({e/max(t,1)*100:.1f}%) | "
                f"Buffer: {queued_bytes} B ({queued_frames} frames){lag_warn} | "
                f"Window: {len(dist_buf)}/{WINDOW_SIZE} | "
                f"Streak: {st.session_state.confirm_streak}/{confirm_n}"
            )

            # Detection log
            if st.session_state.pothole_log:
                log_ph.subheader("📋 Detection Log")
                log_ph.dataframe(
                    pd.DataFrame(st.session_state.pothole_log).head(50),
                    width="stretch",
                )

            last_ui_t = now

    lidar.close()
    st.session_state.running = False
    st.info("⏹ Stopped.")


# ── Page entry point ──────────────────────────────────────────────────────────
st.title("🕳️ Pothole Detection — TF02-Pro LiDAR")
st.caption("Continuous stream · Startup init commands · Hardware diagnostics")

model = load_model()
if model is None:
    st.warning("⚠️ `pothole_model.pkl` not found — rule-only mode. "
               "Run `python model_train.py` to enable ML.")

if not st.session_state.calibrated:
    st.session_state.baseline_cm = float(manual_baseline)

if not st.session_state.running:
    c1, c2 = st.columns([2, 5])
    with c1:
        if st.button("▶ Start Monitoring", type="primary"):
            st.session_state.running        = True
            if not bypass_calib:
                st.session_state.calibrated     = False
                st.session_state.calib_readings = []
            st.rerun()

    st.markdown("---")
    show_diagnostic()

if st.session_state.running:
    run_detection(model)