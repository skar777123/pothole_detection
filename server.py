"""
server.py
=========
FastAPI WebSocket and REST API server for LiDAR Pothole Detection.
Provides real-time telemetry, connection status, detection alerts,
and hardware diagnostics for the React frontend.
"""

import asyncio
import json
import logging
import random
import threading
import time
from pathlib import Path
from collections import deque
from typing import Dict, Any, List, Optional

import joblib
import numpy as np
import serial.tools.list_ports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from lidar_driver import (
    TF02Pro, LiDARReadError, LiDARReaderThread,
    list_ports, FRAME_LEN,
)
from model_train import (
    extract_features, WINDOW_SIZE,
    POTHOLE_THRESH, BUMP_THRESH,
    BASELINE_CM as DEFAULT_BASELINE,
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("server")

app = FastAPI(title="LiDAR Pothole Detection Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
CLASS_LABELS = {
    0: "Flat Road",
    1: "Shallow Pothole",
    2: "Deep Pothole",
    3: "Speed Bump",
}
IS_POTHOLE = {0: False, 1: True, 2: True, 3: False}
IS_BUMP = {0: False, 1: False, 2: False, 3: True}
DEEP_THRESH_CM = 8.0
DETECTION_COOLDOWN_S = 3.0
BASELINE_WINDOW = 20

SEVERITY_BANDS = [
    (0, 3, "Noise"),
    (3, 8, "Shallow"),
    (8, 15, "Moderate"),
    (15, 999, "Deep / Dangerous"),
]

# Persistent cache directories and files
BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"
DETECTIONS_CACHE_FILE = CACHE_DIR / "detections.json"
SETTINGS_CACHE_FILE = CACHE_DIR / "settings.json"



# ML Model Loader

_last_temp_time = 0
_last_temp_val = None
def get_pi_temperature():
    global _last_temp_time, _last_temp_val
    import time
    if time.time() - _last_temp_time > 5.0:
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                temp = float(f.read().strip()) / 1000.0
                _last_temp_val = round(temp, 1)
        except Exception:
            pass
        _last_temp_time = time.time()
    return _last_temp_val

def load_ml_model():

    try:
        model = joblib.load("pothole_model.pkl")
        logger.info("ML model loaded successfully.")
        return model
    except Exception as exc:
        logger.warning(f"Could not load ML model: {exc}")
        return None

ml_model = load_ml_model()


# Simulated Sensor Reader


# System Manager
class SystemManager:
    def __init__(self):
        self.lock = threading.RLock()

        # Connection state
        self.connected = False
        self.status = "disconnected"  # "disconnected", "connecting", "connected", "error"
        self.status_message = "Sensor not connected"
        self.port = "auto"
        self.baudrate = 115200
        self.send_init = True

        # Driver references
        self.lidar = None
        self.reader = None

        # Settings
        self.speed_kmph = 30.0
        self.pot_thresh = 4.5
        self.deep_thresh = 8.0
        self.bump_thresh = 4.5
        self.confirm_n = 2
        self.cooldown_s = 3.0

        # Processing state
        self.baseline_buf = deque(maxlen=BASELINE_WINDOW)
        self.baseline_cm = None
        self.calibrated = False

        self.dist_buf = []
        self.str_buf = []

        self.confirm_streak = 0
        self.last_detect_t = 0.0
        self.last_depth = 0.0
        self.pothole_count = 0
        self.bump_count = 0

        # Rolling histories for telemetry charts
        self.dist_history = deque(maxlen=100)
        self.dev_history = deque(maxlen=100)
        self.str_history = deque(maxlen=100)
        self.baseline_history = deque(maxlen=100)
        self.timestamps = deque(maxlen=100)

        # Detection logs
        self.detection_log = []

        # Latest processed state
        self.latest_telemetry = {
            "distance_cm": 0.0,
            "baseline_cm": 0.0,
            "deviation_cm": 0.0,
            "strength": 0,
            "temperature_c": 0.0,
            "calibrated": False,
            "warmup_count": 0,
            "warmup_total": BASELINE_WINDOW,
            "class_id": 0,
            "class_name": "Flat Road",
            "confidence": 0.0,
            "streak": 0,
            "streak_target": 2,
            "is_alert": False,
            "alert_message": "",
            "cooldown_remaining": 0.0,
            "pi_temperature": None,
        }

        # Load persistent cache
        self._load_cache()

        # Background processing worker
        self.worker_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.worker_thread.start()

    def _load_cache(self):
        """Loads cached settings and historical detections from disk."""
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            if SETTINGS_CACHE_FILE.exists():
                with open(SETTINGS_CACHE_FILE, "r", encoding="utf-8") as f:
                    s = json.load(f)
                    self.speed_kmph = float(s.get("speed_kmph", self.speed_kmph))
                    self.pot_thresh = float(s.get("pot_thresh", self.pot_thresh))
                    self.deep_thresh = float(s.get("deep_thresh", self.deep_thresh))
                    self.bump_thresh = float(s.get("bump_thresh", self.bump_thresh))
                    self.confirm_n = int(s.get("confirm_n", self.confirm_n))
                    self.cooldown_s = float(s.get("cooldown_s", self.cooldown_s))
                    if "port" in s:
                        self.port = str(s["port"])
                    if "baudrate" in s:
                        self.baudrate = int(s["baudrate"])
                logger.info("Loaded cached settings from %s", SETTINGS_CACHE_FILE)

            if DETECTIONS_CACHE_FILE.exists():
                with open(DETECTIONS_CACHE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.detection_log = data[:100]
                    elif isinstance(data, dict):
                        self.detection_log = data.get("log", [])[:100]
                        self.pothole_count = int(data.get("pothole_count", 0))
                        self.bump_count = int(data.get("bump_count", 0))
                        self.last_depth = float(data.get("last_depth", 0.0))

                    if not self.pothole_count and not self.bump_count and self.detection_log:
                        for item in self.detection_log:
                            t = str(item.get("type", "")).lower()
                            if "pothole" in t:
                                self.pothole_count += 1
                            elif "bump" in t:
                                self.bump_count += 1
                        if self.detection_log:
                            self.last_depth = float(self.detection_log[0].get("depth_cm", 0.0))
                logger.info("Loaded %d cached detections from %s", len(self.detection_log), DETECTIONS_CACHE_FILE)
        except Exception as exc:
            logger.warning("Could not load persistent cache: %s", exc)

    def _save_settings_cache(self):
        """Saves current settings and serial parameters to disk cache."""
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            payload = {
                "speed_kmph": self.speed_kmph,
                "pot_thresh": self.pot_thresh,
                "deep_thresh": self.deep_thresh,
                "bump_thresh": self.bump_thresh,
                "confirm_n": self.confirm_n,
                "cooldown_s": self.cooldown_s,
                "port": self.port,
                "baudrate": self.baudrate,
            }
            tmp = SETTINGS_CACHE_FILE.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            tmp.replace(SETTINGS_CACHE_FILE)
        except Exception as exc:
            logger.warning("Could not save settings cache: %s", exc)

    def _save_detections_cache(self):
        """Saves detection log and anomaly metrics to disk cache."""
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            payload = {
                "pothole_count": self.pothole_count,
                "bump_count": self.bump_count,
                "last_depth": self.last_depth,
                "log": self.detection_log[:100],
            }
            tmp = DETECTIONS_CACHE_FILE.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            tmp.replace(DETECTIONS_CACHE_FILE)
        except Exception as exc:
            logger.warning("Could not save detections cache: %s", exc)

    def get_speed_cm_s(self) -> float:
        return (self.speed_kmph * 100_000) / 3600

    def get_dist_per_reading(self) -> float:
        return self.get_speed_cm_s() / 100.0

    def severity_label(self, depth_cm: float) -> str:
        for lo, hi, label in SEVERITY_BANDS:
            if lo <= depth_cm < hi:
                return label
        return "Noise"

    def rule_classify(self, dist_cm: float, baseline: float) -> int:
        dev = dist_cm - baseline
        if dev > self.deep_thresh:
            return 2
        if dev > self.pot_thresh:
            return 1
        if dev < -self.bump_thresh:
            return 3
        return 0

    def compute_dimensions(self, dist_buf: List[float], str_buf: List[float], baseline: float) -> Dict[str, Any]:
        arr = np.array(dist_buf, dtype=float)
        dev = arr - baseline
        depth_cm = float(max(dev.max(), 0))
        in_hole = int(np.sum(dev > self.pot_thresh))
        dist_per_reading = self.get_dist_per_reading()
        length = round(in_hole * dist_per_reading, 1)
        return {
            "depth_cm": round(depth_cm, 1),
            "length_cm": length,
            "width_cm": round(length * 0.8, 1),
            "severity": self.severity_label(depth_cm),
            "avg_strength": round(float(np.mean(str_buf)) if str_buf else 0, 0),
        }

    def connect(self, port: str, baudrate: int = 115200):
        with self.lock:
            self.disconnect()

            self.port = port
            self.baudrate = baudrate
            self.status = "connecting"
            self.status_message = f"Connecting to {port} at {baudrate} baud"
            try:
                actual_port = self.port
                if actual_port == "auto":
                    from lidar_driver import auto_detect_lidar
                    logger.info("Auto-detecting LiDAR port...")
                    detected_port = auto_detect_lidar(baudrate=self.baudrate, send_init=self.send_init)
                    if not detected_port:
                        raise ValueError("Auto-detect failed: LiDAR not found on any port")
                    actual_port = detected_port
                    self.port = actual_port
                
                self.lidar = TF02Pro(
                    port=actual_port,
                    baudrate=self.baudrate,
                    send_init=self.send_init,
                    timeout=0.20,
                )
                self.reader = LiDARReaderThread(self.lidar, maxlen=5)
                self.connected = True
                self.status = "connected"
                self.status_message = f"Connected to {self.port} at {self.baudrate} baud"
                logger.info(f"Hardware sensor connected on {self.port}")
                self._save_settings_cache()
                return {"success": True, "message": self.status_message}
            except Exception as exc:
                self.status = "error"
                self.status_message = f"Connection failed on {self.port}: {str(exc)}"
                self.connected = False
                self.lidar = None
                self.reader = None
                logger.error(f"Cannot open port {self.port}: {exc}")
                return {"success": False, "message": self.status_message}

    def disconnect(self):
        with self.lock:
            if self.reader is not None:
                try:
                    self.reader.stop()
                except Exception:
                    pass
                self.reader = None

            if self.lidar is not None:
                try:
                    self.lidar.close()
                except Exception:
                    pass
                self.lidar = None

            self.connected = False
            self.status = "disconnected"
            self.status_message = "Sensor disconnected"
            self.confirm_streak = 0
            logger.info("LiDAR disconnected.")
            return {"success": True, "message": "Disconnected"}

    def reset_metrics(self):
        with self.lock:
            self.pothole_count = 0
            self.bump_count = 0
            self.confirm_streak = 0
            self.last_depth = 0.0
            self.dist_history.clear()
            self.dev_history.clear()
            self.str_history.clear()
            self.baseline_history.clear()
            self.timestamps.clear()
            self.detection_log.clear()
            self.baseline_buf.clear()
            self.baseline_cm = None
            self.calibrated = False
            self.dist_buf.clear()
            self.str_buf.clear()
            self._save_detections_cache()
            return {"success": True, "message": "Metrics and histories reset"}

    def _process_loop(self):
        """Continuously pulls frames from reader, runs classification, updates telemetry."""
        while True:
            time.sleep(0.02)  # 50 Hz poll cycle

            if not self.connected or self.reader is None:
                self.latest_telemetry["pi_temperature"] = get_pi_temperature()
                continue

            frame = self.reader.get_latest()
            if frame is None:
                self.latest_telemetry["pi_temperature"] = get_pi_temperature()
                continue

            dist = frame.get("distance_cm", 0.0)
            strength = frame.get("strength", 0)
            temp = frame.get("temperature_c", 0.0)
            valid = frame.get("valid", True)

            if not valid or dist <= 0:
                self.latest_telemetry["pi_temperature"] = get_pi_temperature()
                continue

            # Rolling baseline
            self.baseline_buf.append(dist)
            if len(self.baseline_buf) == BASELINE_WINDOW:
                self.baseline_cm = float(np.mean(self.baseline_buf))
                self.calibrated = True

            if not self.calibrated or self.baseline_cm is None:
                self.latest_telemetry = {
                    "distance_cm": dist,
                    "baseline_cm": dist,
                    "deviation_cm": 0.0,
                    "strength": strength,
                    "temperature_c": temp,
                    "calibrated": False,
                    "warmup_count": len(self.baseline_buf),
                    "warmup_total": BASELINE_WINDOW,
                    "class_id": 0,
                    "class_name": "Calibrating Baseline",
                    "confidence": 0.0,
                    "streak": 0,
                    "streak_target": self.confirm_n,
                    "is_alert": False,
                    "alert_message": f"Establishing baseline: {len(self.baseline_buf)}/{BASELINE_WINDOW} readings",
                    "cooldown_remaining": 0.0,
                    "pi_temperature": get_pi_temperature(),
                }
                continue

            baseline = self.baseline_cm
            dev = dist - baseline

            # Append to history buffers
            now_iso = time.strftime("%H:%M:%S")
            self.dist_history.append(dist)
            self.dev_history.append(round(dev, 2))
            self.str_history.append(strength)
            self.baseline_history.append(round(baseline, 1))
            self.timestamps.append(now_iso)

            # Sliding window for ML
            self.dist_buf.append(dist)
            self.str_buf.append(strength)
            if len(self.dist_buf) > WINDOW_SIZE:
                self.dist_buf.pop(0)
                self.str_buf.pop(0)

            # Classification
            rule_cls = self.rule_classify(dist, baseline)
            ml_conf = 0.0
            final_cls = rule_cls

            if len(self.dist_buf) == WINDOW_SIZE and ml_model is not None:
                try:
                    feats = extract_features(
                        np.array(self.dist_buf), np.array(self.str_buf), baseline
                    ).reshape(1, -1)
                    pred = int(ml_model.predict(feats)[0])
                    proba = float(ml_model.predict_proba(feats)[0][pred])
                    if proba >= 0.55:
                        final_cls = pred
                        ml_conf = proba
                except Exception:
                    final_cls = rule_cls

            is_ph = IS_POTHOLE.get(final_cls, False)
            is_bump = IS_BUMP.get(final_cls, False)

            if is_ph or is_bump:
                self.confirm_streak += 1
            else:
                self.confirm_streak = 0

            is_alert = False
            alert_msg = ""
            now_t = time.monotonic()
            elapsed = now_t - self.last_detect_t
            cooldown_rem = max(0.0, self.cooldown_s - elapsed)

            if self.confirm_streak >= self.confirm_n:
                if cooldown_rem > 0:
                    self.confirm_streak = 0
                    alert_msg = f"{CLASS_LABELS[final_cls]} continuing (Cooldown: {cooldown_rem:.1f}s)"
                else:
                    dims = self.compute_dimensions(self.dist_buf, self.str_buf, baseline)
                    if is_ph:
                        self.pothole_count += 1
                    elif is_bump:
                        self.bump_count += 1

                    self.confirm_streak = 0
                    self.last_detect_t = now_t
                    self.last_depth = dims["depth_cm"]

                    # Half-flush window to prevent re-triggering on same feature
                    half = len(self.dist_buf) // 2
                    self.dist_buf = self.dist_buf[half:]
                    self.str_buf = self.str_buf[half:]

                    log_entry = {
                        "id": int(time.time() * 1000),
                        "time": time.strftime("%H:%M:%S"),
                        "type": CLASS_LABELS[final_cls],
                        "deviation_cm": f"{dev:+.1f}",
                        "depth_cm": dims["depth_cm"],
                        "length_cm": dims["length_cm"],
                        "width_cm": dims["width_cm"],
                        "severity": dims["severity"],
                        "confidence": f"{ml_conf:.0%}" if ml_conf > 0 else "Rule",
                        "strength": int(dims["avg_strength"]),
                        "baseline": f"{baseline:.0f}",
                    }
                    self.detection_log.insert(0, log_entry)
                    if len(self.detection_log) > 100:
                        self.detection_log.pop()
                    self._save_detections_cache()

                    is_alert = True
                    alert_msg = f"{CLASS_LABELS[final_cls]} Confirmed (Depth: {dims['depth_cm']} cm, Severity: {dims['severity']})"
                    logger.info(f"Anomaly detected: {log_entry['type']} depth={dims['depth_cm']}cm")

            # Update latest telemetry snapshot
            self.latest_telemetry = {
                "distance_cm": round(dist, 1),
                "baseline_cm": round(baseline, 1),
                "deviation_cm": round(dev, 1),
                "strength": strength,
                "temperature_c": temp,
                "calibrated": self.calibrated,
                "warmup_count": len(self.baseline_buf),
                "warmup_total": BASELINE_WINDOW,
                "class_id": final_cls,
                "class_name": CLASS_LABELS.get(final_cls, "Unknown"),
                "confidence": round(ml_conf, 2),
                "streak": self.confirm_streak,
                "streak_target": self.confirm_n,
                "is_alert": is_alert,
                "alert_message": alert_msg,
                "cooldown_remaining": round(cooldown_rem, 1),
                "pi_temperature": get_pi_temperature(),
        }


manager = SystemManager()


# Request Models
class ConnectRequest(BaseModel):
    port: str
    baudrate: int = 115200

class SettingsRequest(BaseModel):
    speed_kmph: Optional[float] = None
    pot_thresh: Optional[float] = None
    deep_thresh: Optional[float] = None
    bump_thresh: Optional[float] = None
    confirm_n: Optional[int] = None
    cooldown_s: Optional[float] = None


# REST Endpoints
@app.get("/api/health")
def health():
    return {"status": "ok", "time": time.time()}

@app.get("/api/ports")
def get_available_ports():
    ports = list_ports()
    all_ports = ["auto"] + [p for p in ports if p != "auto"]
    return {"ports": all_ports}

@app.get("/api/status")
def get_status():
    reader_frames = manager.reader.frames if manager.reader else 0
    reader_errors = manager.reader.errors if manager.reader else 0
    return {
        "connected": manager.connected,
        "status": manager.status,
        "status_message": manager.status_message,
        "port": manager.port,
        "baudrate": manager.baudrate,
        "frames_received": reader_frames,
        "errors_count": reader_errors,
        "pothole_count": manager.pothole_count,
        "bump_count": manager.bump_count,
        "last_depth": manager.last_depth,
        "telemetry": manager.latest_telemetry,
        "settings": {
            "speed_kmph": manager.speed_kmph,
            "pot_thresh": manager.pot_thresh,
            "deep_thresh": manager.deep_thresh,
            "bump_thresh": manager.bump_thresh,
            "confirm_n": manager.confirm_n,
            "cooldown_s": manager.cooldown_s,
        },
    }

@app.post("/api/connect")
def api_connect(req: ConnectRequest):
    return manager.connect(port=req.port, baudrate=req.baudrate)

@app.post("/api/disconnect")
def api_disconnect():
    return manager.disconnect()

@app.post("/api/settings")
def update_settings(req: SettingsRequest):
    with manager.lock:
        if req.speed_kmph is not None:
            manager.speed_kmph = max(1.0, req.speed_kmph)
        if req.pot_thresh is not None:
            manager.pot_thresh = max(0.5, req.pot_thresh)
        if req.deep_thresh is not None:
            manager.deep_thresh = max(1.0, req.deep_thresh)
        if req.bump_thresh is not None:
            manager.bump_thresh = max(0.5, req.bump_thresh)
        if req.confirm_n is not None:
            manager.confirm_n = max(1, req.confirm_n)
        if req.cooldown_s is not None:
            manager.cooldown_s = max(0.5, req.cooldown_s)
        manager._save_settings_cache()
    return {"success": True, "message": "Settings updated and cached"}

@app.post("/api/reset")
def reset_counts():
    return manager.reset_metrics()

@app.post("/api/cache/clear")
def clear_cache():
    with manager.lock:
        try:
            if DETECTIONS_CACHE_FILE.exists():
                DETECTIONS_CACHE_FILE.unlink()
            manager.detection_log.clear()
            manager.pothole_count = 0
            manager.bump_count = 0
            manager.last_depth = 0.0
            return {"success": True, "message": "Persistent detection cache cleared"}
        except Exception as exc:
            return {"success": False, "message": f"Error clearing cache: {exc}"}

@app.get("/api/log")
def get_log():
    return {"log": manager.detection_log}

@app.post("/api/diagnostic/raw")
def diagnostic_raw(req: ConnectRequest):
    """Executes a 90-byte raw read on the serial port to inspect hex packets."""
    try:
        lidar = TF02Pro(port=req.port, baudrate=req.baudrate, timeout=0.5, send_init=manager.send_init)
        raw = lidar.diagnostic_raw_dump(90)
        lidar.close()

        if not raw:
            return {
                "success": False,
                "bytes_received": 0,
                "has_header": False,
                "message": "0 bytes received: Sensor TX not reaching adapter. Check wiring and baud rate.",
                "lines": [],
            }

        has_hdr = b"\x59\x59" in raw
        groups = [raw[i:i+9] for i in range(0, len(raw), 9)]
        lines = []
        for i, g in enumerate(groups):
            h = g.hex(" ")
            ok = (len(g) == 9 and g[0] == 0x59 and g[1] == 0x59 and (sum(g[:8]) & 0xFF) == g[8])
            d = (g[2] | g[3] << 8) if len(g) >= 4 else "?"
            status_text = f"<- dist={d} cm" if ok else ""
            lines.append(f"[{i:02d}] {h:<27} {status_text}".strip())

        return {
            "success": has_hdr,
            "bytes_received": len(raw),
            "has_header": has_hdr,
            "message": f"{len(raw)} bytes received with header 59 59: Sensor OK" if has_hdr else f"{len(raw)} bytes received without header 59 59. Try baud 9600.",
            "lines": lines,
        }
    except Exception as exc:
        return {"success": False, "bytes_received": 0, "has_header": False, "message": f"Port error: {str(exc)}", "lines": []}

@app.post("/api/diagnostic/frame")
def diagnostic_frame(req: ConnectRequest):
    """Executes a single frame read."""
    try:
        lidar = TF02Pro(port=req.port, baudrate=req.baudrate, timeout=0.5, send_init=manager.send_init)
        frame = lidar.read_frame()
        lidar.close()
        return {
            "success": True,
            "frame": frame,
            "message": "Frame read successfully",
        }
    except Exception as exc:
        return {"success": False, "frame": None, "message": f"Frame read failed: {str(exc)}"}


# WebSocket Telemetry Stream (10 Hz broadcast)
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected.")
    try:
        while True:
            reader_frames = manager.reader.frames if manager.reader else 0
            reader_errors = manager.reader.errors if manager.reader else 0

            payload = {
                "connected": manager.connected,
                        "status": manager.status,
                "status_message": manager.status_message,
                "port": manager.port,
                "baudrate": manager.baudrate,
                "frames_received": reader_frames,
                "errors_count": reader_errors,
                "pothole_count": manager.pothole_count,
                "bump_count": manager.bump_count,
                "last_depth": manager.last_depth,
                "telemetry": manager.latest_telemetry,
                "history": {
                    "timestamps": list(manager.timestamps),
                    "distance": list(manager.dist_history),
                    "deviation": list(manager.dev_history),
                    "strength": list(manager.str_history),
                    "baseline": list(manager.baseline_history),
                },
                "log_preview": manager.detection_log[:10],
            }

            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(0.10)  # 10 Hz telemetry update rate
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as exc:
        logger.error(f"WebSocket error: {exc}")


import os
from fastapi.staticfiles import StaticFiles

# Serve built React frontend if dist exists
dist_dir = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.exists(dist_dir):
    app.mount("/", StaticFiles(directory=dist_dir, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)

