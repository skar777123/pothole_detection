import os
import sys
import time
import json
import asyncio
import argparse
import threading
import queue
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

# Ensure DL_Model is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adaptive_detector import AdaptiveDetector
from lidar_driver import TF02Pro, LiDARReadError
from dashboard_dl import _simulate_sim_reading, _next_sim_reading

app = FastAPI(title="Real-Time LiDAR Pothole Detector")

# Mount static files
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# Global state
class AppState:
    def __init__(self):
        self.detector = AdaptiveDetector(use_model=True)
        self.clients: list[WebSocket] = []
        self.running = False
        self.total_readings = 0
        self.last_baseline = 1000.0

state = AppState()

# WebSocket Manager
async def broadcast_result(result: dict):
    if not state.clients:
        return
        
    msg = json.dumps(result)
    dead_clients = []
    
    for client in state.clients:
        try:
            await client.send_text(msg)
        except Exception:
            dead_clients.append(client)
            
    for c in dead_clients:
        if c in state.clients:
            state.clients.remove(c)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    state.clients.append(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in state.clients:
            state.clients.remove(websocket)

@app.get("/")
async def get_dashboard():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(), status_code=200)
    return HTMLResponse(content="<h1>Dashboard UI not found in static/index.html</h1>", status_code=404)

@app.get("/api/status")
async def get_status():
    return {
        "running": state.running,
        "total_readings": state.total_readings,
        "clients_connected": len(state.clients),
        "detector_baseline": state.detector.baseline
    }

# Data ingest thread loop
def sensor_loop(port: str, baud: int, simulate: bool, hz: float):
    print(f"Starting sensor loop. Simulate={simulate}, Port={port}")
    state.running = True
    
    if simulate:
        # Simulation loop
        sleep_time = 1.0 / hz
        print(f"Simulating LiDAR at {hz} Hz")
        try:
            # Give server time to start
            time.sleep(2)
            while state.running:
                dist, strength = _next_sim_reading(state.last_baseline)
                state.last_baseline = state.detector.baseline
                
                result = state.detector.feed(dist, strength)
                state.total_readings += 1
                
                if result:
                    # Run in async loop
                    try:
                        loop = asyncio.get_running_loop()
                        asyncio.run_coroutine_threadsafe(broadcast_result(result), loop)
                    except RuntimeError:
                        pass # No event loop
                        
                time.sleep(sleep_time)
        except Exception as e:
            print(f"Simulation error: {e}")
            
    else:
        # Real serial data loop
        print(f"Opening LiDAR on {port} at {baud} baud")
        try:
            lidar = TF02Pro(port=port, baudrate=baud, send_init=True)
            print("LiDAR ready. Waiting for clients...")
            
            while state.running:
                try:
                    r = lidar.read_frame()
                    if not r.get("valid", True):
                        time.sleep(0.01)
                        continue
                        
                    dist = r["distance_cm"]
                    strength = r["strength"]
                    
                    result = state.detector.feed(dist, strength)
                    state.total_readings += 1
                    
                    if result:
                        try:
                            loop = asyncio.get_running_loop()
                            asyncio.run_coroutine_threadsafe(broadcast_result(result), loop)
                        except RuntimeError:
                            pass
                            
                except LiDARReadError as e:
                    print(f"LiDAR Read Error: {e}")
                    time.sleep(0.1)
                
                # Small yield to not peg CPU
                time.sleep(0.001)
        except Exception as e:
            print(f"Fatal serial error: {e}")
        finally:
            if 'lidar' in locals():
                lidar.close()
    
    state.running = False


def main():
    parser = argparse.ArgumentParser(description="Real-Time LiDAR Pothole Server")
    parser.add_argument("--port", type=int, default=8000, help="HTTP Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="HTTP Server host")
    parser.add_argument("--serial-port", type=str, default="COM3", help="Serial port for TF02-Pro")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--simulate", action="store_true", help="Run in simulation mode")
    parser.add_argument("--hz", type=float, default=20.0, help="Simulation rate in Hz")
    
    args = parser.parse_args()
    
    # Start background sensor thread
    sensor_thread = threading.Thread(
        target=sensor_loop, 
        args=(args.serial_port, args.baud, args.simulate, args.hz),
        daemon=True
    )
    sensor_thread.start()
    
    # Run server
    print(f"Starting server at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
