"""
dashboard_dl.py
===============
Streamlit dashboard for the Deep Learning Pothole Detector.

Run:  streamlit run DL_Model/dashboard_dl.py

Features
────────
  📊  Live simulated or serial sensor feed with animated distance plot
  🧠  Real-time DL model inference  (class + confidence + depth)
  🗺️  Detection log + severity map
  📈  Model statistics  (accuracy, F1, training history)
  ⚙️  Baseline & threshold controls
"""

import os, sys, time, json, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from dl_config import (
    WINDOW_SIZE, N_CLASSES, CLASS_NAMES, BASELINE_CM,
    RT_CONFIDENCE_THRESHOLD, RT_ALERT_DEPTH_CM,
    MODEL_SAVE_PATH, HISTORY_SAVE_PATH, LOGS_DIR,
    ADAPT_MA_WINDOW, ADAPT_HP_ALPHA, ADAPT_MIN_DURATION,
    POTHOLE_THRESH, BUMP_THRESH,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🚦 DL Pothole Detector",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
  html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
  .main { background: #0d1117; }
  .stApp { background: #0d1117; }

  .metric-card {
    background: linear-gradient(135deg, #161b22 0%, #21262d 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 10px;
    box-shadow: 0 4px 16px rgba(0,0,0,.4);
  }
  .metric-card h3 { color: #8b949e; font-size: 13px; margin: 0 0 6px; font-weight: 500; }
  .metric-card .val { color: #e6edf3; font-size: 26px; font-weight: 700; margin: 0; }
  .metric-card .sub { color: #8b949e; font-size: 12px; margin-top: 4px; }

  .alert-box {
    background: linear-gradient(90deg, #3d0000, #6b0000);
    border: 2px solid #f85149;
    border-radius: 10px;
    padding: 14px 20px;
    color: white;
    font-weight: 700;
    font-size: 20px;
    text-align: center;
    animation: pulse 1s infinite;
  }
  @keyframes pulse {
    0%   { box-shadow: 0 0 0 0   rgba(248,81,73,.6); }
    70%  { box-shadow: 0 0 0 12px rgba(248,81,73,0); }
    100% { box-shadow: 0 0 0 0   rgba(248,81,73,0); }
  }

  .cls-flat    { color: #2ea043; font-weight: 700; }
  .cls-shallow { color: #f0e030; font-weight: 700; }
  .cls-deep    { color: #f85149; font-weight: 700; }
  .cls-bump    { color: #58a6ff; font-weight: 700; }

  [data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }
  h1, h2, h3 { color: #e6edf3 !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "readings"   : [],      # raw distance readings
        "strengths"  : [],
        "baselines"  : [],      # adaptive baseline values
        "results"    : [],      # inference result dicts
        "alerts"     : [],      # alert events
        "running"    : False,
        "total"      : 0,
        "detector"   : None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Detector loader ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model …")
def _get_detector(conf_thresh: float, alert_depth: float,
                  ma_window: int, min_dur: int, use_model: bool):
    from adaptive_detector import AdaptiveDetector
    return AdaptiveDetector(
        use_model            = use_model,
        ma_window            = ma_window,
        min_duration         = min_dur,
        confidence_threshold = conf_thresh,
        alert_depth_cm       = alert_depth,
    )


# ── Synthetic stream ───────────────────────────────────────────────────────────
_RNG = np.random.default_rng(42)
_SEGMENT_IDX  = 0
_SEGMENT_POS  = 0
_SEGMENTS = [
    ("flat",    60, 0),
    ("pothole", 12, 8.0),
    ("flat",    40, 0),
    ("pothole", 14, 22.0),
    ("flat",    30, 0),
    ("bump",    10, 15.0),
    ("flat",    35, 0),
    ("pothole", 16, 50.0),
]

def _next_sim_reading(baseline_cm: float) -> tuple[float, float]:
    global _SEGMENT_IDX, _SEGMENT_POS, _RNG
    seg = _SEGMENTS[_SEGMENT_IDX % len(_SEGMENTS)]
    kind, total, depth = seg
    noise = max(1.0, baseline_cm * 0.004)
    d = float(_RNG.normal(baseline_cm, noise))
    if kind == "pothole" and 3 <= _SEGMENT_POS < total - 3:
        d += depth * _RNG.uniform(0.8, 1.2)
    elif kind == "bump" and 2 <= _SEGMENT_POS < total - 2:
        d -= depth * _RNG.uniform(0.8, 1.2)
        d  = max(d, 10)
    _SEGMENT_POS += 1
    if _SEGMENT_POS >= total:
        _SEGMENT_POS = 0
        _SEGMENT_IDX += 1
    s = float(_RNG.uniform(2500, 4000) * max(0.1, 1 - baseline_cm / 1500))
    return d, s


# ── Plotly helpers ─────────────────────────────────────────────────────────────
CLASS_COLORS = {0: "#2ea043", 1: "#f0e030", 2: "#f85149", 3: "#58a6ff"}
DARK_BG  = "#0d1117"
CARD_BG  = "#161b22"
GRID_COL = "#21262d"


def _distance_plot(readings, baselines, results):
    n   = len(readings)
    xs  = list(range(n))
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[3, 2, 1],
        vertical_spacing=0.04,
        subplot_titles=["Distance + Adaptive Baseline",
                        "MA Deviation (cm)", "Depth (cm)"]
    )

    # Distance trace
    fig.add_trace(go.Scatter(x=xs, y=readings, mode="lines",
                             line=dict(color="#58a6ff", width=1.5),
                             name="Distance (cm)"), row=1, col=1)

    # Adaptive baseline
    if baselines:
        fig.add_trace(go.Scatter(x=xs[:len(baselines)], y=baselines,
                                 mode="lines",
                                 line=dict(color="#8b949e", dash="dot", width=1.5),
                                 name="Adaptive Baseline"), row=1, col=1)

    # Colour scatter by class
    if results:
        r_xs = list(range(len(results)))
        r_ys = [r.get("baseline_cm", 0) + r.get("ma_deviation_cm", 0)
                for r in results]
        cs   = [CLASS_COLORS[r["class_id"]] for r in results]
        fig.add_trace(go.Scatter(x=r_xs, y=r_ys, mode="markers",
                                 marker=dict(color=cs, size=8, symbol="diamond"),
                                 name="Prediction"), row=1, col=1)

    # MA Deviation
    if results:
        devs = [r.get("ma_deviation_cm", 0) for r in results]
        dcs  = [CLASS_COLORS[r["class_id"]] for r in results]
        fig.add_trace(go.Scatter(
            x=list(range(len(devs))), y=devs, mode="lines+markers",
            line=dict(color="#d2a8ff", width=1.5),
            marker=dict(color=dcs, size=5),
            name="MA Deviation", fill="tozeroy", fillcolor="rgba(210,168,255,0.12)"),
            row=2, col=1
        )
        fig.add_hline(y=POTHOLE_THRESH,  line_dash="dash",
                      line_color="#f0e030", row=2, col=1)
        fig.add_hline(y=-BUMP_THRESH,    line_dash="dash",
                      line_color="#2ea043", row=2, col=1)

        # Depth bar
        depths = [r["depth_cm"] for r in results]
        d_cs   = [CLASS_COLORS[r["class_id"]] for r in results]
        fig.add_trace(go.Bar(x=list(range(len(depths))), y=depths,
                             marker_color=d_cs, name="Depth (cm)"), row=3, col=1)

    fig.update_layout(
        paper_bgcolor=DARK_BG, plot_bgcolor=CARD_BG,
        font=dict(color="#8b949e"),
        height=520, margin=dict(l=50, r=20, t=40, b=30),
        legend=dict(bgcolor=CARD_BG, bordercolor=GRID_COL, borderwidth=1),
    )
    for ax in ["xaxis", "xaxis2", "xaxis3",
               "yaxis", "yaxis2", "yaxis3"]:
        if hasattr(fig.layout, ax):
            getattr(fig.layout, ax).update(gridcolor=GRID_COL)
    fig.layout.yaxis.title  = "Distance (cm)"
    fig.layout.yaxis2.title = "Deviation (cm)"
    fig.layout.yaxis3.title = "Depth (cm)"
    return fig


def _confidence_fig(result: dict):
    names  = list(result["probs"].keys())
    values = list(result["probs"].values())
    colors = [CLASS_COLORS[i] for i in range(N_CLASSES)]
    fig    = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker_color=colors,
        text=[f"{v*100:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(color="white"),
    ))
    fig.update_layout(
        paper_bgcolor=DARK_BG, plot_bgcolor=CARD_BG,
        font=dict(color="#8b949e"),
        height=200, margin=dict(l=10, r=60, t=10, b=10),
        xaxis=dict(range=[0, 1.1], gridcolor=GRID_COL),
        yaxis=dict(gridcolor=GRID_COL),
        showlegend=False,
    )
    return fig


def _load_training_history():
    if not os.path.exists(HISTORY_SAVE_PATH):
        return None
    with open(HISTORY_SAVE_PATH) as f:
        return json.load(f)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    st.markdown("**Data Source**")
    data_source = st.selectbox("Source", ["Simulation", "Serial LiDAR"],
                               index=0)
    serial_port = None
    serial_baud = 115200
    if data_source == "Serial LiDAR":
        serial_port = st.text_input("Serial port", value="/dev/ttyUSB0",
                                    placeholder="COM3 or /dev/ttyUSB0")
        serial_baud = st.number_input("Baud rate", value=115200, step=9600)

    st.markdown("**Adaptive Baseline**")
    ma_window    = st.slider("MA window (readings)",  10, 60,
                             ADAPT_MA_WINDOW, step=5)
    min_dur      = st.slider("Min duration (readings)", 1, 10,
                             ADAPT_MIN_DURATION, step=1)

    st.markdown("**Model**")
    model_ok     = os.path.exists(MODEL_SAVE_PATH)
    use_model    = st.checkbox("Use LSTM model", value=model_ok)
    conf_thresh  = st.slider("Min confidence",        0.3, 0.95,
                             RT_CONFIDENCE_THRESHOLD,  step=0.05)
    alert_depth  = st.slider("Alert depth (cm)",      1.0, 40.0,
                             RT_ALERT_DEPTH_CM,        step=0.5)
    steps_per_click = st.slider("Readings per batch",  1, 50, 10)
    stream_speed    = st.slider("Stream speed (ms between batches)",
                                50, 2000, 200, step=50,
                                help="Lower → faster updates when streaming")

    st.divider()
    st.markdown("## 🚦 Class Legend")
    for cls_id, name in CLASS_NAMES.items():
        col = CLASS_COLORS[cls_id]
        st.markdown(f'<span style="color:{col};font-weight:700;">'
                    f'●  {name}</span>', unsafe_allow_html=True)

    st.divider()
    if model_ok:
        st.success("✅ Model ready")
    else:
        st.warning("⚠️ Model not trained.\nRun `python train.py` first.\nRule-based mode is active.")


# ── Title ──────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style="background:linear-gradient(90deg,#58a6ff,#f85149);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;
           font-size:2.2rem;font-weight:800;margin-bottom:0;">
  🚦 Real-Time Pothole Detector  —  Deep Learning
</h1>
<p style="color:#8b949e;margin-top:4px;font-size:15px;">
  1D-CNN + BiLSTM + Multi-Head Attention  |  LiDAR time-series classification
</p>
""", unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_live, tab_log, tab_stats = st.tabs(
    ["📡 Live Detection", "📋 Detection Log", "📊 Model Stats"]
)


# ── Serial reader helper ────────────────────────────────────────────────────
def _read_serial_batch(port: str, baud: int, n: int) -> list[tuple[float, float]]:
    """Read up to `n` distance/strength pairs from a TFMini-style serial LiDAR."""
    try:
        import serial
    except ImportError:
        st.error("PySerial is required for serial mode: `pip install pyserial`")
        return []
    try:
        ser = serial.Serial(port, baud, timeout=1)
    except Exception as e:
        st.error(f"Cannot open serial port **{port}**: {e}")
        return []
    pairs: list[tuple[float, float]] = []
    buf = b""
    deadline = time.time() + 2.0          # max 2 s per batch
    try:
        while len(pairs) < n and time.time() < deadline:
            chunk = ser.read(max(1, ser.in_waiting))
            if not chunk:
                continue
            buf += chunk
            while len(buf) >= 9:
                if buf[0] == 0x59 and buf[1] == 0x59:
                    dist = float(buf[3] << 8 | buf[2])
                    strn = float(buf[5] << 8 | buf[4])
                    pairs.append((dist, strn))
                    buf = buf[9:]
                else:
                    buf = buf[1:]
    finally:
        ser.close()
    return pairs


# ╔═══════════════════════════════════════════════════════╗
# ║  TAB 1 — Live Detection                               ║
# ╚═══════════════════════════════════════════════════════╝
with tab_live:
    # ── Control buttons ──────────────────────────────────────────────────
    col_c1, col_c2, col_c3, col_c4 = st.columns([1, 1, 1, 2])
    with col_c1:
        if st.session_state["running"]:
            stop_btn = st.button("⏹ Stop", use_container_width=True,
                                 type="primary")
            if stop_btn:
                st.session_state["running"] = False
                st.rerun()
        else:
            start_btn = st.button("▶ Start", use_container_width=True,
                                  type="primary")
            if start_btn:
                st.session_state["running"] = True
                st.rerun()
    with col_c2:
        step_btn = st.button("⏭ Step", use_container_width=True,
                             disabled=st.session_state["running"])
    with col_c3:
        reset_btn = st.button("🔄 Reset", use_container_width=True)
    with col_c4:
        if st.session_state["running"]:
            st.markdown(
                '<span style="color:#2ea043;font-weight:700;font-size:16px;">'
                '🟢 STREAMING …</span>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<span style="color:#8b949e;font-size:16px;">'
                '⏸ Paused — click <b>▶ Start</b></span>',
                unsafe_allow_html=True)

    # ── Reset handler ────────────────────────────────────────────────────
    if reset_btn:
        st.session_state["readings"]  = []
        st.session_state["strengths"] = []
        st.session_state["baselines"] = []
        st.session_state["results"]   = []
        st.session_state["alerts"]    = []
        st.session_state["total"]     = 0
        st.session_state["running"]   = False
        st.cache_resource.clear()
        st.rerun()

    # ── Process one batch (shared by Step and continuous mode) ───────
    should_process = step_btn or st.session_state["running"]

    if should_process:
        detector = _get_detector(conf_thresh, alert_depth,
                                 ma_window, min_dur,
                                 use_model and model_ok)

        # Get readings from the selected source
        if data_source == "Serial LiDAR" and serial_port:
            pairs = _read_serial_batch(serial_port, serial_baud,
                                       steps_per_click)
        else:
            pairs = [_next_sim_reading(detector.baseline)
                     for _ in range(steps_per_click)]

        for dist, strn in pairs:
            st.session_state["readings"].append(dist)
            st.session_state["strengths"].append(strn)
            st.session_state["baselines"].append(detector.baseline)
            st.session_state["total"] += 1
            result = detector.feed(dist, strn)
            if result:
                st.session_state["results"].append(result)
                if result["alert"]:
                    st.session_state["alerts"].append(result)

        # Keep only the last 500 entries to avoid memory bloat
        for key in ("readings", "strengths", "baselines"):
            st.session_state[key] = st.session_state[key][-500:]
        st.session_state["results"] = st.session_state["results"][-500:]

    # ── KPIs ────────────────────────────────────────────────────────────
    results = st.session_state["results"]
    latest  = results[-1] if results else None

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        class_name = latest["class_name"] if latest else "—"
        cls_id     = latest["class_id"]   if latest else -1
        col        = CLASS_COLORS.get(cls_id, "#8b949e")
        st.markdown(f"""<div class="metric-card">
          <h3>Current Class</h3>
          <p class="val" style="color:{col};">{class_name}</p>
          <p class="sub">{'⚠️ ALERT' if (latest and latest['alert']) else 'OK'}</p>
        </div>""", unsafe_allow_html=True)
    with k2:
        depth = f"{latest['depth_cm']:.1f} cm" if latest else "—"
        st.markdown(f"""<div class="metric-card">
          <h3>Depth</h3>
          <p class="val">{depth}</p>
          <p class="sub">{latest['severity'] if latest else ''}</p>
        </div>""", unsafe_allow_html=True)
    with k3:
        conf = f"{latest['confidence']*100:.1f}%" if latest else "—"
        st.markdown(f"""<div class="metric-card">
          <h3>Confidence</h3>
          <p class="val">{conf}</p>
          <p class="sub">Model certainty</p>
        </div>""", unsafe_allow_html=True)
    with k4:
        n_alerts = len(st.session_state["alerts"])
        st.markdown(f"""<div class="metric-card">
          <h3>Total Alerts</h3>
          <p class="val" style="color:#f85149;">{n_alerts}</p>
          <p class="sub">of {st.session_state['total']} readings</p>
        </div>""", unsafe_allow_html=True)

    # Alert banner
    if latest and latest["alert"]:
        st.markdown(
            f'<div class="alert-box">⚠️  POTHOLE DETECTED! '
            f'Depth: {latest["depth_cm"]:.1f} cm  |  '
            f'{latest["severity"]}  |  '
            f'Conf: {latest["confidence"]*100:.1f}%</div>',
            unsafe_allow_html=True
        )

    # Distance + deviation plot
    st.plotly_chart(
        _distance_plot(
            st.session_state["readings"],
            st.session_state.get("baselines", []),
            st.session_state["results"],
        ),
        use_container_width=True, key="dist_plot"
    )

    # Confidence breakdown
    if latest and latest.get("probs"):
        st.markdown("#### Class Probability Breakdown")
        st.plotly_chart(_confidence_fig(latest),
                        use_container_width=True, key="conf_plot")

    # Extra adaptive info row
    if latest:
        i1, i2, i3, i4 = st.columns(4)
        vel  = latest.get("velocity_cm", 0)
        run  = latest.get("duration_run", 0)
        vp   = latest.get("vel_pattern", False)
        conf_dur = latest.get("duration_confirmed", False)
        i1.metric("Velocity (cm/reading)", f"{vel:+.2f}")
        i2.metric("Duration Run",
                  f"{run} / {min_dur} rdgs",
                  delta="Confirmed!" if conf_dur else None,
                  delta_color="normal" if conf_dur else "off")
        i3.metric("Vel Pattern", "↗↘ Yes" if vp else "—")
        i4.metric("MA Deviation",
                  f"{latest.get('ma_deviation_cm', 0):+.2f} cm")

    # ── Auto-rerun for continuous streaming ───────────────────────────
    if st.session_state["running"]:
        time.sleep(stream_speed / 1000.0)
        st.rerun()


# ╔═══════════════════════════════════════════════════════╗
# ║  TAB 2 — Detection Log                                ║
# ╚═══════════════════════════════════════════════════════╝
with tab_log:
    results = st.session_state["results"]
    if not results:
        st.info("No detections yet. Click ▶ Step in the Live Detection tab.")
    else:
        df = pd.DataFrame([{
            "Class"       : r["class_name"],
            "Conf (%)"    : round(r["confidence"] * 100, 1),
            "Depth (cm)"  : r["depth_cm"],
            "Severity"    : r["severity"],
            "Baseline cm" : r["baseline_cm"],
            "MA Dev (cm)" : round(r.get("ma_deviation_cm", 0), 2),
            "Vel (cm/r)"  : round(r.get("velocity_cm", 0), 2),
            "Dur Run"     : r.get("duration_run", 0),
            "Confirmed"   : "✔" if r.get("duration_confirmed") else "",
            "Alert"       : "⚠️" if r["alert"] else "",
        } for r in results])

        st.dataframe(
            df.style.map(
                lambda v: "color: #f85149" if v == "⚠️" else "",
                subset=["Alert"]
            ),
            use_container_width=True,
            height=400,
        )

        # Severity distribution pie
        sev_counts = df["Severity"].value_counts().reset_index()
        sev_counts.columns = ["Severity", "Count"]
        fig_pie = px.pie(
            sev_counts, values="Count", names="Severity",
            color_discrete_sequence=["#2ea043","#f0e030","#f85149","#58a6ff","#8b949e"],
            hole=0.45,
        )
        fig_pie.update_layout(
            paper_bgcolor=DARK_BG, font=dict(color="#8b949e"),
            margin=dict(l=10, r=10, t=30, b=10),
            title=dict(text="Severity Distribution", font=dict(color="white")),
        )
        st.plotly_chart(fig_pie, use_container_width=True, key="sev_pie")


# ╔═══════════════════════════════════════════════════════╗
# ║  TAB 3 — Model Stats                                  ║
# ╚═══════════════════════════════════════════════════════╝
with tab_stats:
    hist = _load_training_history()
    eval_path = os.path.join(LOGS_DIR, "evaluation_summary.json")

    if hist is None:
        st.warning("No training history found. Run `python train.py` first.")
    else:
        # Summary cards
        test_acc = hist.get("test_accuracy", 0)
        macro_f1 = hist.get("test_macro_f1", 0)
        n_epochs = len(hist.get("accuracy", []))
        train_t  = hist.get("training_seconds", 0)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""<div class="metric-card">
              <h3>Test Accuracy</h3>
              <p class="val" style="color:#2ea043;">{test_acc*100:.2f}%</p>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-card">
              <h3>Macro F1</h3>
              <p class="val" style="color:#58a6ff;">{macro_f1:.4f}</p>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="metric-card">
              <h3>Epochs Trained</h3>
              <p class="val">{n_epochs}</p>
            </div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""<div class="metric-card">
              <h3>Training Time</h3>
              <p class="val">{train_t/60:.1f} min</p>
            </div>""", unsafe_allow_html=True)

        # Training curves
        st.markdown("#### Training Curves")
        epochs   = list(range(1, len(hist.get("accuracy", [])) + 1))
        fig_hist = make_subplots(rows=1, cols=3,
                                 subplot_titles=["Loss", "Accuracy", "Top-2 Accuracy"])

        for i, (key, title) in enumerate([("loss",     "Loss"),
                                           ("accuracy", "Accuracy"),
                                           ("top2_acc", "Top-2 Acc")], 1):
            tr = hist.get(key, [])
            vl = hist.get(f"val_{key}", [])
            e  = list(range(1, len(tr) + 1))
            fig_hist.add_trace(go.Scatter(x=e, y=tr, name="Train",
                                          line=dict(color="#58a6ff")),
                               row=1, col=i)
            fig_hist.add_trace(go.Scatter(x=e, y=vl, name="Val",
                                          line=dict(color="#f78166", dash="dash")),
                               row=1, col=i)

        fig_hist.update_layout(
            paper_bgcolor=DARK_BG, plot_bgcolor=CARD_BG,
            font=dict(color="#8b949e"), height=320,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False,
        )
        for ax in fig_hist.layout:
            if ax.startswith("xaxis") or ax.startswith("yaxis"):
                fig_hist.layout[ax].gridcolor = GRID_COL

        st.plotly_chart(fig_hist, use_container_width=True, key="hist_plot")

        # Load eval images if they exist
        img_cm   = os.path.join(LOGS_DIR, "confusion_matrix_eval.png")
        img_viol = os.path.join(LOGS_DIR, "confidence_violin.png")
        img_grad = os.path.join(LOGS_DIR, "gradcam_samples.png")

        if os.path.exists(img_cm) and os.path.exists(img_viol):
            c1, c2 = st.columns(2)
            with c1:
                st.image(img_cm,   caption="Confusion Matrix", use_container_width=True)
            with c2:
                st.image(img_viol, caption="Confidence by Class", use_container_width=True)

        if os.path.exists(img_grad):
            st.image(img_grad, caption="GradCAM Saliency Maps",
                     use_container_width=True)
