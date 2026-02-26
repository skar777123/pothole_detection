"""
lidar_3d_map.py
===============
Real-time 3D Terrain Mapping — TF02-Pro LiDAR
───────────────────────────────────────────────
This page reads the TF02-Pro sensor in a background thread (shared with the
main dashboard architecture) and accumulates distance samples into a virtual
3-D road surface.

HOW XY COORDINATES ARE DERIVED
  • The sensor is mounted on a moving vehicle, so the Y-axis ("forward")
    advances by  (speed × dt)  for every reading.
  • A gentle sinusoidal lateral sweep is synthesised for the X-axis so the
    map looks like proper scan-stripe coverage (mimicking a rocking mount or
    a side-to-side sweep arm).
  • Z-axis = deviation from the 1000 cm baseline → depressions are potholes,
    protrusions are speed bumps.

The result is a live Plotly 3-D scatter + interpolated surface that updates
every 0.5 seconds (configurable).  Potholes appear as coloured wells;
speed bumps as peaks.
"""

import streamlit as st
import numpy as np
import pandas as pd
import time
import logging
import plotly.graph_objects as go
from scipy.interpolate import griddata
from collections import deque

from lidar_driver import (
    TF02Pro, LiDARReadError, LiDARReaderThread,
    list_ports,
)
from model_train import BASELINE_CM as DEFAULT_BASELINE

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("lidar_3d_map")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="LiDAR 3D Terrain Map",
    page_icon="🗺️",
)

# ── Constants ─────────────────────────────────────────────────────────────────
BASELINE_CM   = 1000.0          # fixed 10 m baseline
MAX_POINTS    = 2000            # max points kept in rolling buffer
SWEEP_AMP_CM  = 20.0            # ±20 cm lateral sweep amplitude
SWEEP_FREQ_HZ = 0.5             # complete a lateral sweep cycle every 2 s
GRID_RES      = 40              # resolution of interpolated surface grid (NxN)

# ── Colour helpers ────────────────────────────────────────────────────────────
COLORSCALE = [
    [0.00, "#0d1b2a"],   # deep blue-black  (deep pothole floor)
    [0.15, "#1b4965"],   # dark blue
    [0.30, "#2ec4b6"],   # teal             (moderate dip)
    [0.50, "#20bf55"],   # green            (flat road  ← zero deviation)
    [0.70, "#f7b731"],   # amber            (small bump)
    [0.85, "#fa4d56"],   # red              (large bump)
    [1.00, "#f4f4f4"],   # white            (very tall feature)
]

# ── Session state defaults ────────────────────────────────────────────────────
_defaults = {
    "pts_x"       : deque(maxlen=MAX_POINTS),
    "pts_y"       : deque(maxlen=MAX_POINTS),
    "pts_z"       : deque(maxlen=MAX_POINTS),
    "pts_label"   : deque(maxlen=MAX_POINTS),  # "flat" | "pothole" | "bump"
    "map_running" : False,
    "y_cursor"    : 0.0,      # virtual forward distance travelled (cm)
    "t_sweep"     : 0.0,      # time reference for sweep sinusoid
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ 3D Map Settings")

    available_ports = list_ports()
    lidar_port = st.text_input(
        "Serial Port", value="/dev/ttyUSB0",
        help="Linux: /dev/ttyUSB0  |  Windows: COM3"
    )
    if available_ports:
        st.caption(f"Detected: `{'`, `'.join(available_ports)}`")

    lidar_baud  = st.selectbox("Baud Rate", [115200, 9600], index=0)
    send_init   = st.checkbox("Send startup commands", value=True)

    st.markdown("---")
    st.subheader("🚗 Vehicle & Sweep")
    speed_kmph  = st.number_input("Speed (km/h)", 5, 120, 30, step=5)
    sweep_amp   = st.slider("Lateral sweep amplitude (cm)",
                            5, 100, int(SWEEP_AMP_CM), step=5,
                            help="Width of the virtual scan stripe")
    sweep_freq  = st.slider("Sweep frequency (Hz)", 0.1, 2.0,
                            SWEEP_FREQ_HZ, step=0.1,
                            help="How fast the sensor sweeps side-to-side")

    st.markdown("---")
    st.subheader("🗺️ Map")
    update_hz   = st.slider("Map refresh rate (Hz)", 0.5, 5.0, 2.0, step=0.5)
    show_surf   = st.checkbox("Show interpolated surface", value=True)
    show_pts    = st.checkbox("Show raw scatter points",   value=True)
    view_z_range = st.slider(
        "Z-axis display range (cm)", 5, 100, 30, step=5,
        help="Half-range around zero for the Z axis (+/- this value)"
    )

    st.markdown("---")
    st.subheader("🔍 Detection thresholds")
    pot_thresh  = st.number_input("Pothole threshold (cm)", 1.0, 30.0, 4.5, step=0.5)
    bump_thresh = st.number_input("Bump threshold (cm)",    1.0, 30.0, 4.5, step=0.5)

    st.markdown("---")
    if st.button("🗑️ Clear Map"):
        for k, v in _defaults.items():
            st.session_state[k] = v
        st.rerun()


# ── Speed → cm / reading at 100 Hz ───────────────────────────────────────────
speed_cm_s   = (speed_kmph * 100_000) / 3600.0
cm_per_frame = speed_cm_s / 100.0          # at 100 Hz sensor rate


# ── Build 3-D figure from buffered points ─────────────────────────────────────

def build_figure(pts_x, pts_y, pts_z, pts_label, z_range, do_surf, do_pts):
    """Return a Plotly Figure with scatter points and optional surface."""
    xs = np.array(pts_x, dtype=float)
    ys = np.array(pts_y, dtype=float)
    zs = np.array(pts_z, dtype=float)
    labels = list(pts_label)

    if len(xs) < 5:
        # Not enough points yet
        fig = go.Figure()
        fig.update_layout(
            title="Collecting data — please wait …",
            template="plotly_dark",
            height=600,
        )
        return fig

    # Colour per-point: green=flat, blue=pothole, orange=bump
    colors = []
    for lb in labels:
        if lb == "pothole":
            colors.append("#2ec4b6")
        elif lb == "bump":
            colors.append("#f7b731")
        else:
            colors.append("#20bf55")

    traces = []

    # ── Interpolated surface ──────────────────────────────────────────────────
    if do_surf and len(xs) >= 10:
        try:
            xi = np.linspace(xs.min(), xs.max(), GRID_RES)
            yi = np.linspace(ys.min(), ys.max(), GRID_RES)
            XI, YI = np.meshgrid(xi, yi)
            ZI = griddata(
                (xs, ys), zs, (XI, YI),
                method="linear", fill_value=0.0
            )
            surf = go.Surface(
                x=XI, y=YI, z=ZI,
                colorscale=COLORSCALE,
                cmin=-z_range, cmax=z_range,
                opacity=0.85,
                showscale=True,
                colorbar=dict(
                    title=dict(text="Deviation (cm)", side="right"),
                    tickfont=dict(color="white"),
                    titlefont=dict(color="white"),
                    len=0.6,
                ),
                name="Terrain Surface",
                hovertemplate=(
                    "X: %{x:.1f} cm<br>"
                    "Y: %{y:.1f} cm<br>"
                    "ΔZ: %{z:.1f} cm<extra></extra>"
                ),
                lighting=dict(
                    ambient=0.6, diffuse=0.8,
                    specular=0.4, roughness=0.5,
                ),
            )
            traces.append(surf)
        except Exception as e:
            logger.warning("Surface interp failed: %s", e)

    # ── Raw scatter points ────────────────────────────────────────────────────
    if do_pts:
        scatter = go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="markers",
            marker=dict(
                size=3,
                color=zs,
                colorscale=COLORSCALE,
                cmin=-z_range, cmax=z_range,
                opacity=0.9,
                showscale=False,
            ),
            name="LiDAR Points",
            hovertemplate=(
                "X: %{x:.1f} cm<br>"
                "Y: %{y:.1f} cm<br>"
                "ΔZ: %{z:.1f} cm<extra></extra>"
            ),
        )
        traces.append(scatter)

    # ── Pothole / bump annotation markers ────────────────────────────────────
    ph_mask   = np.array([lb == "pothole" for lb in labels])
    bump_mask = np.array([lb == "bump"    for lb in labels])

    if ph_mask.any():
        traces.append(go.Scatter3d(
            x=xs[ph_mask], y=ys[ph_mask], z=zs[ph_mask],
            mode="markers",
            marker=dict(size=7, color="#ff4757", symbol="diamond",
                        line=dict(color="white", width=1)),
            name="🕳️ Pothole",
            hovertemplate="Pothole: ΔZ=%{z:.1f} cm<extra></extra>",
        ))

    if bump_mask.any():
        traces.append(go.Scatter3d(
            x=xs[bump_mask], y=ys[bump_mask], z=zs[bump_mask],
            mode="markers",
            marker=dict(size=7, color="#ffa502", symbol="diamond",
                        line=dict(color="white", width=1)),
            name="🔶 Speed Bump",
            hovertemplate="Bump: ΔZ=%{z:.1f} cm<extra></extra>",
        ))

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text="🗺️ LiDAR 3D Terrain Map — Live",
            font=dict(size=20, color="white"),
            x=0.5,
        ),
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        height=680,
        margin=dict(l=0, r=0, t=60, b=0),
        scene=dict(
            xaxis=dict(
                title="X — Lateral (cm)",
                gridcolor="#2a2a3e",
                backgroundcolor="#0d1117",
                showbackground=True,
                range=[-sweep_amp * 1.5, sweep_amp * 1.5],
            ),
            yaxis=dict(
                title="Y — Forward (cm)",
                gridcolor="#2a2a3e",
                backgroundcolor="#0d1117",
                showbackground=True,
            ),
            zaxis=dict(
                title="ΔZ — Depth/Height (cm)",
                gridcolor="#2a2a3e",
                backgroundcolor="#0d1117",
                showbackground=True,
                range=[-z_range, z_range],
            ),
            camera=dict(
                eye=dict(x=1.6, y=-1.6, z=1.2),
            ),
            bgcolor="#0d1117",
        ),
        legend=dict(
            font=dict(color="white"),
            bgcolor="rgba(13,17,23,0.8)",
            bordercolor="#444",
            borderwidth=1,
        ),
        uirevision="map",   # keeps camera angle on refresh
    )
    return fig


# ── Stats panel ───────────────────────────────────────────────────────────────

def render_stats(pts_z, pts_label):
    zs     = np.array(pts_z, dtype=float)
    labels = list(pts_label)
    n_ph   = labels.count("pothole")
    n_bmp  = labels.count("bump")
    n_flat = labels.count("flat")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📡 Points", len(zs))
    c2.metric("🕳️ Pothole pts", n_ph)
    c3.metric("🔶 Bump pts",    n_bmp)
    c4.metric("🟢 Flat pts",    n_flat)
    c5.metric("↕️ Max depth",
              f"{abs(zs.min()):.1f} cm" if len(zs) else "—")


# ── Main map loop ─────────────────────────────────────────────────────────────

def run_map():
    try:
        lidar = TF02Pro(port=lidar_port, baudrate=lidar_baud,
                        send_init=send_init)
    except Exception as exc:
        st.error(f"❌ Cannot open `{lidar_port}`: {exc}")
        st.session_state.map_running = False
        st.info("**Tip:** Check your port in the sidebar and make sure the "
                "sensor is powered on.")
        return

    reader = LiDARReaderThread(lidar, maxlen=10)
    time.sleep(0.3)   # let thread warm up

    # ── UI placeholders ───────────────────────────────────────────────────────
    hdr_l, hdr_r = st.columns([6, 1])
    hdr_l.success(
        f"✅ **Streaming** `{lidar_port}` @ {lidar_baud} baud  "
        f"| Thread: 100 Hz  |  Map refresh: {update_hz:.1f} Hz"
    )
    stop_btn = hdr_r.button("⏹ Stop")

    st.markdown("---")
    stats_ph = st.empty()
    chart_ph = st.empty()
    status_ph = st.empty()

    last_update = 0.0
    update_interval = 1.0 / update_hz

    t0      = time.monotonic()
    y_pos   = st.session_state.y_cursor
    no_data = 0

    while not stop_btn:
        frame = reader.get_latest()

        if frame is None:
            no_data += 1
            status_ph.warning(
                f"⏳ Waiting for sensor frames … poll #{no_data} | "
                f"thread errors: {reader.errors}"
            )
            time.sleep(0.05)
            continue

        no_data = 0
        dist    = frame["distance_cm"]
        valid   = frame.get("valid", True)

        if not valid:
            time.sleep(0.02)
            continue

        # ── Derive 3-D coordinates ────────────────────────────────────────────
        t_now = time.monotonic() - t0
        x_pos = sweep_amp * np.sin(2 * np.pi * sweep_freq * t_now)
        dev   = dist - BASELINE_CM

        if dev > pot_thresh:
            label = "pothole"
        elif dev < -bump_thresh:
            label = "bump"
        else:
            label = "flat"

        st.session_state.pts_x.append(x_pos)
        st.session_state.pts_y.append(y_pos)
        st.session_state.pts_z.append(dev)
        st.session_state.pts_label.append(label)

        y_pos += cm_per_frame
        st.session_state.y_cursor = y_pos

        # ── Refresh map ───────────────────────────────────────────────────────
        now = time.monotonic()
        if now - last_update >= update_interval:
            render_stats(
                st.session_state.pts_z,
                st.session_state.pts_label,
            )
            fig = build_figure(
                st.session_state.pts_x,
                st.session_state.pts_y,
                st.session_state.pts_z,
                st.session_state.pts_label,
                z_range=view_z_range,
                do_surf=show_surf,
                do_pts=show_pts,
            )
            chart_ph.plotly_chart(fig, use_container_width=True)
            status_ph.caption(
                f"Thread frames: **{reader.frames}** | "
                f"Thread errors: **{reader.errors}** | "
                f"Points buffered: **{len(st.session_state.pts_x)}** | "
                f"Y forward: **{y_pos:.0f} cm**"
            )
            last_update = now

        time.sleep(0.002)   # yield — inner loop runs at sensor rate

    # ── Cleanup ───────────────────────────────────────────────────────────────
    reader.stop()
    lidar.close()
    st.session_state.map_running = False
    st.info("⏹ Mapping stopped. Hit **▶ Start Mapping** to resume.")


# ── Demo / offline mode ───────────────────────────────────────────────────────

def run_demo():
    """Generate synthetic LiDAR terrain data for testing without hardware."""
    st.info(
        "🧪 **Demo Mode** — Generating synthetic terrain data. "
        "No sensor required."
    )

    n_pts   = st.slider("Demo point count", 100, 2000, 600, step=100)
    seed    = st.number_input("Random seed", 0, 9999, 42, step=1)

    if st.button("🎲 Generate Demo Map", type="primary"):
        rng = np.random.default_rng(int(seed))

        # Synthesise a road with a few potholes and bumps
        t    = np.linspace(0, 4 * np.pi, n_pts)
        x    = 20 * np.sin(0.5 * t) + rng.normal(0, 0.5, n_pts)
        y    = np.linspace(0, 2000, n_pts) + rng.normal(0, 0.5, n_pts)
        z    = rng.normal(0, 1.5, n_pts)

        # Add 3 potholes
        for cy in [400, 900, 1600]:
            mask = np.abs(y - cy) < 80
            z[mask] += rng.uniform(6, 20, mask.sum())

        # Add 2 bumps
        for cy in [650, 1300]:
            mask = np.abs(y - cy) < 50
            z[mask] -= rng.uniform(4, 10, mask.sum())

        labels = []
        for zi in z:
            if zi > pot_thresh:
                labels.append("pothole")
            elif zi < -bump_thresh:
                labels.append("bump")
            else:
                labels.append("flat")

        # Store in session
        for buf, arr in [
            ("pts_x", x), ("pts_y", y), ("pts_z", z)
        ]:
            st.session_state[buf] = deque(arr.tolist(), maxlen=MAX_POINTS)
        st.session_state.pts_label = deque(labels, maxlen=MAX_POINTS)

        render_stats(st.session_state.pts_z, st.session_state.pts_label)
        fig = build_figure(
            st.session_state.pts_x, st.session_state.pts_y,
            st.session_state.pts_z, st.session_state.pts_label,
            z_range=view_z_range,
            do_surf=show_surf,
            do_pts=show_pts,
        )
        st.plotly_chart(fig, use_container_width=True)


# ── Entry point ───────────────────────────────────────────────────────────────

st.title("🗺️ LiDAR 3D Terrain Mapping")
st.caption(
    "Real-time 3D road surface map from TF02-Pro LiDAR · "
    "Background thread at 100 Hz · Interactive Plotly 3D chart"
)

tab_live, tab_demo = st.tabs(["📡 Live Sensor", "🧪 Demo / Offline"])

with tab_live:
    if not st.session_state.map_running:
        col1, col2, _ = st.columns([2, 2, 5])
        with col1:
            if st.button("▶ Start Mapping", type="primary"):
                st.session_state.map_running = True
                st.rerun()
        with col2:
            # Show last map if available
            if len(st.session_state.pts_x) > 0:
                st.caption(
                    f"📌 {len(st.session_state.pts_x)} points in buffer "
                    f"(from previous run)"
                )

        # Show last map snapshot even when stopped
        if len(st.session_state.pts_x) >= 5:
            st.markdown("### 🗺️ Last Captured Map")
            render_stats(st.session_state.pts_z, st.session_state.pts_label)
            fig = build_figure(
                st.session_state.pts_x, st.session_state.pts_y,
                st.session_state.pts_z, st.session_state.pts_label,
                z_range=view_z_range,
                do_surf=show_surf,
                do_pts=show_pts,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        run_map()

with tab_demo:
    run_demo()
