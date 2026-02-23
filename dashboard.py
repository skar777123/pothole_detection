"""
dashboard.py
============
Real-time Pothole Detection Dashboard using TF02-Pro single-point LiDAR.

Detection pipeline
──────────────────
1. Read raw frame from LiDAR (with checksum + quality gating in driver)
2. Median-filter across N_MEDIAN consecutive valid frames → reduces noise
3. Accumulate a sliding WINDOW_SIZE buffer of confirmed distance readings
4. Extract 26 statistical features from the window
5. Classify with the trained RandomForest pipeline (4 classes)
6. Apply a CONFIDENCE ACCUMULATOR: only trigger alert after
   CONFIRM_FRAMES consecutive positive predictions (avoids single-frame FPs)
7. Compute depth, length, width, severity from the confirmed window
8. Log and display everything in the Streamlit UI
"""

import streamlit as st
import numpy as np
import time
import joblib
import logging
from collections import deque

from lidar_driver import TF02Pro, LiDARReadError
from model_train import extract_features   # reuse feature extractor

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("dashboard")

# ── Config ────────────────────────────────────────────────────────────────────
LIDAR_PORT        = '/dev/ttyUSB0'    # Change to 'COM3' on Windows
LIDAR_BAUD        = 115200

WINDOW_SIZE       = 20               # Readings per inference window
N_MEDIAN          = 3                # Median-filter samples per reading
SLIDE_STEP        = 1                # Overlap: advance 1 reading at a time

# Confidence accumulator: require N consecutive positive frames before alerting
CONFIRM_FRAMES    = 3

VEHICLE_SPEED_KMPH = 30
SENSOR_HEIGHT_CM   = 100            # Update to your actual mounting height (cm)
FRAME_RATE_HZ      = 100            # TF02-Pro native frame rate
SPEED_CM_S         = (VEHICLE_SPEED_KMPH * 1000 * 100) / 3600
DIST_PER_READING   = SPEED_CM_S / FRAME_RATE_HZ   # cm of road per reading

# Pothole severity thresholds
SEVERITY_SHALLOW_MAX_CM = 8         # ≤ 8 cm → shallow
SEVERITY_DEEP_MIN_CM    = 9         # ≥ 9 cm → deep

# Labels returned by the model
CLASS_LABELS = {0: "Flat Road", 1: "Shallow Pothole", 2: "Deep Pothole", 3: "Speed Bump"}
IS_POTHOLE   = {0: False, 1: True, 2: True, 3: False}

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = joblib.load("pothole_model.pkl")
        logger.info("Model loaded successfully.")
        return model
    except FileNotFoundError:
        return None

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="LiDAR Pothole Detection",
    page_icon="🕳️",
)

st.title("🕳️ Real-Time Pothole Detection — Single-Point LiDAR")
st.caption(
    "TF02-Pro · Median-filtered · ML-confirmed · Severity-graded"
)

model = load_model()
if model is None:
    st.error("❌ `pothole_model.pkl` not found. Run `python model_train.py` first.")
    st.stop()

# ── Metric placeholders ───────────────────────────────────────────────────────
kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
ph_count    = kpi_col1.empty()
ph_depth    = kpi_col2.empty()
ph_length   = kpi_col3.empty()
ph_width    = kpi_col4.empty()
ph_strength = kpi_col5.empty()

st.markdown("---")
chart_title   = st.empty()
chart_ph      = st.empty()
strength_ph   = st.empty()
status_ph     = st.empty()
log_ph        = st.empty()

# ── Session state ─────────────────────────────────────────────────────────────
defaults = {
    "dist_history"    : deque(maxlen=300),
    "str_history"     : deque(maxlen=300),
    "pothole_count"   : 0,
    "pothole_log"     : [],      # list of dicts with detection details
    "running"         : False,
    "confirm_streak"  : 0,       # consecutive positive-class frames
    "last_label"      : "—",
    "last_depth"      : 0.0,
    "last_length"     : 0.0,
    "last_width"      : 0.0,
    "last_strength"   : 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Pothole dimension calculator ──────────────────────────────────────────────

def calculate_pothole_dimensions(window: list, strength: list) -> dict:
    """
    Derive depth, length, width, and severity from a confirmed pothole window.

    The LiDAR is mounted pointing DOWN at the road.
    ● Road surface → reading ≈ SENSOR_HEIGHT_CM
    ● Pothole      → reading > SENSOR_HEIGHT_CM (ground is farther away)

    Parameters
    ----------
    window   : list of distance readings (cm) for the detection window
    strength : corresponding signal strength readings

    Returns
    -------
    dict with depth_cm, length_cm, width_cm, severity, avg_strength
    """
    arr   = np.array(window, dtype=float)
    s_arr = np.array(strength, dtype=float)

    # Baseline = estimated road surface (10th percentile of window, robust to outliers)
    baseline_cm = np.percentile(arr, 10)

    # Noise floor = 3 cm above baseline
    noise_margin_cm = 3.0
    dip_threshold   = baseline_cm + noise_margin_cm

    # Depth = deepest reading − baseline
    depth_cm = float(np.max(arr) - baseline_cm)

    # Length = distance travelled while readings stayed above dip_threshold
    points_in_dip = int(np.sum(arr > dip_threshold))
    length_cm     = points_in_dip * DIST_PER_READING

    # Width = geometric estimate (assumes roughly elliptical cross-section)
    # More conservative than a simple 1.1× factor
    width_cm = length_cm * 0.85

    # Severity
    if depth_cm <= 0:
        severity = "None"
    elif depth_cm <= SEVERITY_SHALLOW_MAX_CM:
        severity = "⚠️ Shallow"
    elif depth_cm <= 15:
        severity = "🔶 Moderate"
    else:
        severity = "🔴 Deep / Dangerous"

    avg_strength = float(s_arr.mean()) if len(s_arr) > 0 else 0.0

    return {
        "depth_cm"    : round(depth_cm, 1),
        "length_cm"   : round(length_cm, 1),
        "width_cm"    : round(width_cm, 1),
        "severity"    : severity,
        "avg_strength": round(avg_strength, 0),
    }


# ── Main detection loop ───────────────────────────────────────────────────────

def run_detection():
    """
    Main coroutine called by the Start button.
    Opens the LiDAR, fills a sliding window, runs inference, and updates UI.
    """
    stop_btn = st.button("⏹ Stop Monitoring", key="stop_btn")

    dist_buffer  = []   # Distance readings buffer (raw validated)
    str_buffer   = []   # Corresponding strength readings

    try:
        lidar = TF02Pro(port=LIDAR_PORT, baudrate=LIDAR_BAUD)
    except Exception as exc:
        st.error(f"❌ Cannot open LiDAR on `{LIDAR_PORT}`: {exc}")
        logger.error("LiDAR open failed: %s", exc)
        return

    st.success(f"✅ LiDAR connected on `{LIDAR_PORT}`")
    logger.info("Detection loop started.")

    consec_errors = 0

    while not stop_btn:
        # ── STEP 1: Get a median-confirmed reading ────────────────────────────
        try:
            reading = lidar.read_median(samples=N_MEDIAN)
            consec_errors = 0
        except LiDARReadError as exc:
            consec_errors += 1
            logger.warning("LiDAR read error (#%d): %s", consec_errors, exc)
            if consec_errors > 20:
                st.error("❌ Too many consecutive LiDAR errors. Check wiring.")
                break
            time.sleep(0.05)
            continue

        dist     = reading["distance_cm"]
        strength = reading["strength"]
        temp     = reading["temperature_c"]

        # ── STEP 2: Accumulate into rolling history for chart ─────────────────
        st.session_state.dist_history.append(dist)
        st.session_state.str_history.append(strength)

        # ── STEP 3: Fill the inference window ────────────────────────────────
        dist_buffer.append(dist)
        str_buffer.append(strength)

        # Keep buffer at WINDOW_SIZE
        if len(dist_buffer) > WINDOW_SIZE:
            dist_buffer.pop(0)
            str_buffer.pop(0)

        # ── STEP 4: Run inference when window is full ─────────────────────────
        if len(dist_buffer) == WINDOW_SIZE:
            features = extract_features(
                np.array(dist_buffer),
                np.array(str_buffer)
            ).reshape(1, -1)

            prediction = int(model.predict(features)[0])
            proba      = model.predict_proba(features)[0]
            confidence = float(proba[prediction])
            label      = CLASS_LABELS[prediction]

            # ── STEP 5: Confidence accumulator ───────────────────────────────
            if IS_POTHOLE.get(prediction, False):
                st.session_state.confirm_streak += 1
            else:
                st.session_state.confirm_streak = 0

            confirmed_pothole = (
                st.session_state.confirm_streak >= CONFIRM_FRAMES
            )

            # ── STEP 6: On confirmed pothole, compute & log ───────────────────
            if confirmed_pothole:
                dims = calculate_pothole_dimensions(dist_buffer, str_buffer)
                st.session_state.pothole_count += 1

                # Reset streak so we don't double-count the same hole
                st.session_state.confirm_streak = 0

                # Save last dims for KPIs
                st.session_state.last_label    = label
                st.session_state.last_depth    = dims["depth_cm"]
                st.session_state.last_length   = dims["length_cm"]
                st.session_state.last_width    = dims["width_cm"]
                st.session_state.last_strength = int(dims["avg_strength"])

                log_entry = {
                    "time"      : time.strftime("%H:%M:%S"),
                    "type"      : label,
                    "depth_cm"  : dims["depth_cm"],
                    "length_cm" : dims["length_cm"],
                    "width_cm"  : dims["width_cm"],
                    "severity"  : dims["severity"],
                    "confidence": f"{confidence:.0%}",
                    "strength"  : int(dims["avg_strength"]),
                    "temp_c"    : temp,
                }
                st.session_state.pothole_log.insert(0, log_entry)

                status_ph.error(
                    f"🔴 **{label} Confirmed!** "
                    f"Depth: {dims['depth_cm']} cm | "
                    f"Length: {dims['length_cm']} cm | "
                    f"Severity: {dims['severity']} | "
                    f"Confidence: {confidence:.0%}"
                )
                logger.info("POTHOLE: %s", log_entry)

                # Clear buffer to avoid re-triggering on same hole
                dist_buffer.clear()
                str_buffer.clear()

            else:
                # Show current classification status
                colour = "🟢" if prediction == 0 else "🟡"
                status_ph.info(
                    f"{colour} **{label}** — "
                    f"Dist: {dist} cm | Strength: {strength} | "
                    f"Temp: {temp:.1f}°C | "
                    f"Confidence: {confidence:.0%} | "
                    f"Streak: {st.session_state.confirm_streak}/{CONFIRM_FRAMES}"
                )

        # ── STEP 7: Update KPI metrics ────────────────────────────────────────
        ph_count.metric("🕳️ Potholes Found",    st.session_state.pothole_count)
        ph_depth.metric("📏 Last Depth",        f"{st.session_state.last_depth} cm")
        ph_length.metric("↔️ Last Length",      f"{st.session_state.last_length} cm")
        ph_width.metric("↕️ Est. Width",        f"{st.session_state.last_width} cm")
        ph_strength.metric("📶 Last Strength",  st.session_state.last_strength)

        # ── STEP 8: Update charts ─────────────────────────────────────────────
        chart_title.markdown("### 📡 Live LiDAR Distance (cm)")
        chart_ph.line_chart(list(st.session_state.dist_history))
        strength_ph.line_chart(
            {"Signal Strength": list(st.session_state.str_history)},
        )

        # ── STEP 9: Detection log table ───────────────────────────────────────
        if st.session_state.pothole_log:
            import pandas as pd
            log_ph.subheader("📋 Detection Log")
            log_ph.dataframe(
                pd.DataFrame(st.session_state.pothole_log).head(30),
                use_container_width=True,
            )

        time.sleep(0.01)   # Don't hammer the CPU

    lidar.close()
    st.info("⏹ Monitoring stopped.")
    logger.info("Detection loop stopped.")


# ── Entry point ───────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown(f"""
| Parameter | Value |
|---|---|
| Serial port | `{LIDAR_PORT}` |
| Baud rate | `{LIDAR_BAUD}` |
| Window size | `{WINDOW_SIZE}` readings |
| Median filter | `{N_MEDIAN}` samples |
| Confirm frames | `{CONFIRM_FRAMES}` consecutive |
| Vehicle speed | `{VEHICLE_SPEED_KMPH}` km/h |
| Sensor height | `{SENSOR_HEIGHT_CM}` cm |
""")
st.sidebar.markdown("---")
st.sidebar.markdown("**Model classes**")
for k, v in CLASS_LABELS.items():
    st.sidebar.markdown(f"- `{k}` → {v}")

if not st.session_state.running:
    if st.button("▶ Start Monitoring", type="primary"):
        st.session_state.running = True

if st.session_state.running:
    run_detection()