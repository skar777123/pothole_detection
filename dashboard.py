import streamlit as st
import numpy as np
import time
import joblib
from collections import deque
from lidar_driver import TF02Pro

# --- Config ---
WINDOW_SIZE = 20
VEHICLE_SPEED_KMPH = 30 # Assumption for Length calc
SENSOR_HEIGHT_CM = 100  # Calibration needed
SPEED_CM_S = (VEHICLE_SPEED_KMPH * 100000) / 3600
FRAME_RATE_HZ = 100 # TF02-Pro default
DIST_BETWEEN_READINGS = SPEED_CM_S / FRAME_RATE_HZ

# --- Load Model ---
try:
    model = joblib.load('pothole_model.pkl')
except:
    st.error("Model not found! Run model_train.py first.")
    st.stop()

# --- Setup UI ---
st.set_page_config(layout="wide", page_title="LiDAR Pothole 3D Dashboard")
st.title("🕳️ Pothole 3D Analysis Dashboard")

col1, col2, col3, col4 = st.columns(4)
metric_count = col1.empty()
metric_depth = col2.empty()
metric_length = col3.empty()
metric_width = col4.empty()

chart_placeholder = st.empty()
log_placeholder = st.empty()

# --- State ---
if 'history' not in st.session_state:
    st.session_state.history = deque(maxlen=200)
if 'pothole_count' not in st.session_state:
    st.session_state.pothole_count = 0

# --- Main Loop ---
def run_dashboard():
    lidar = TF02Pro()
    buffer = [] # To hold current window for ML
    
    stop_btn = st.button("Stop Monitoring")
    
    while not stop_btn:
        dist, strength = lidar.read_data()
        
        if dist:
            # Add to rolling history for chart
            st.session_state.history.append(dist)
            buffer.append(dist)
            
            # --- Inference Trigger ---
            if len(buffer) >= WINDOW_SIZE:
                window_np = np.array(buffer).reshape(1, -1)
                prediction = model.predict(window_np)[0]
                
                if prediction == 1: # POTHOLE DETECTED
                    st.session_state.pothole_count += 1
                    
                    # 1. Calculate Depth (Max dist - Baseline)
                    # We assume the 'min' readings in the window are the road surface (baseline)
                    # and the 'max' readings are the pothole bottom.
                    baseline = np.percentile(buffer, 10) 
                    max_depth_read = np.max(buffer)
                    actual_depth = max_depth_read - baseline
                    
                    # 2. Calculate Length
                    # Count how many points are significantly below baseline
                    points_in_hole = np.sum(np.array(buffer) > (baseline + 3)) # 3cm noise margin
                    actual_length = points_in_hole * DIST_BETWEEN_READINGS
                    
                    # 3. Calculate Width (Estimation)
                    # Assumption: Potholes are roughly circular
                    estimated_width = actual_length * 1.1 
                    
                    # Update Metrics
                    metric_count.metric("Total Potholes", st.session_state.pothole_count)
                    metric_depth.metric("Last Depth", f"{actual_depth:.1f} cm")
                    metric_length.metric("Last Length", f"{actual_length:.1f} cm")
                    metric_width.metric("Est. Width", f"{estimated_width:.1f} cm")
                    
                    log_placeholder.warning(f"⚠️ Pothole Detected! Depth: {actual_depth:.1f}cm")
                    
                    buffer = [] # Clear buffer to avoid double counting
                
                else:
                    buffer.pop(0) # Sliding window

            # --- Update Chart ---
            chart_placeholder.line_chart(list(st.session_state.history))
            
        time.sleep(0.01)

if __name__ == "__main__":
    run_dashboard()