import time
import logging
from collections import deque
from lidar_driver import TF02Pro, LiDARReadError
from ultrasonic_driver import UltrasonicDriver
from camera_driver import CameraDriver

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("sensor_fusion")

# Fusion Logic Constants
BASELINE_WINDOW = 20 # frames to compute rolling baseline
LIDAR_DEVIATION_THRESH = 4.0 # cm
ULTRA_DEVIATION_THRESH = 4.0 # cm

class SensorFusionSystem:
    def __init__(self, lidar_port='/dev/ttyUSB0', camera_port='/dev/serial0'):
        self.lidar = TF02Pro(port=lidar_port)
        self.ultrasonic = UltrasonicDriver(trig_pin=23, echo_pin=24)
        self.camera = CameraDriver(port=camera_port)

        self.lidar_buffer = deque(maxlen=BASELINE_WINDOW)
        self.ultra_buffer = deque(maxlen=BASELINE_WINDOW)

        self.running = True

    def _get_baseline(self, buffer):
        if len(buffer) < BASELINE_WINDOW // 2:
            return None
        valid_vals = [v for v in buffer if v > 0]
        if not valid_vals:
            return None
        return sum(valid_vals) / len(valid_vals)

    def classify_state(self, lidar_dist, ultra_dist, cam_label):
        lidar_baseline = self._get_baseline(self.lidar_buffer)
        ultra_baseline = self._get_baseline(self.ultra_buffer)

        is_pothole_lidar = False
        is_bump_lidar = False
        if lidar_baseline is not None and lidar_dist > 0:
            if lidar_dist > lidar_baseline + LIDAR_DEVIATION_THRESH:
                is_pothole_lidar = True
            elif lidar_dist < lidar_baseline - LIDAR_DEVIATION_THRESH:
                is_bump_lidar = True

        is_pothole_ultra = False
        is_bump_ultra = False
        if ultra_baseline is not None and ultra_dist > 0:
            if ultra_dist > ultra_baseline + ULTRA_DEVIATION_THRESH:
                is_pothole_ultra = True
            elif ultra_dist < ultra_baseline - ULTRA_DEVIATION_THRESH:
                is_bump_ultra = True

        # Camera confirmation
        cam_pothole = "Pothole" in cam_label
        cam_bump = "bump" in cam_label.lower()

        # FUSION LOGIC
        # 1. Very High Confidence: All sensors agree
        if is_pothole_lidar and is_pothole_ultra and cam_pothole:
            return "Pothole detected (High Confidence)"
        if is_bump_lidar and is_bump_ultra and cam_bump:
            return "Speed bump detected (High Confidence)"

        # 2. Distance Consensus: Both distance sensors agree, regardless of camera
        if is_pothole_lidar and is_pothole_ultra:
            return "Pothole detected (Distance Consensus)"
        if is_bump_lidar and is_bump_ultra:
            return "Speed bump detected (Distance Consensus)"

        # 3. Vision + 1 Distance Sensor: Camera agrees with one of the distance readings
        if (is_pothole_lidar or is_pothole_ultra) and cam_pothole:
            return "Pothole detected (Vision + Distance)"
        if (is_bump_lidar or is_bump_ultra) and cam_bump:
            return "Speed bump detected (Vision + Distance)"

        # 4. Fallback logic: If one distance sensor detects severely without vision
        # This increases accuracy when camera misses or lags
        if is_pothole_lidar:
             return "Pothole detected (LiDAR only)"
        if is_pothole_ultra:
             return "Pothole detected (Ultrasonic only)"
            
        if is_bump_lidar:
             return "Speed bump detected (LiDAR only)"
        if is_bump_ultra:
             return "Speed bump detected (Ultrasonic only)"

        return "Normal road"

    def run(self):
        logger.info("Starting Sensor Fusion Loop...")
        try:
            while self.running:
                # 1. Read LiDAR
                try:
                    lidar_data = self.lidar.read_frame_current()
                    lidar_dist = lidar_data.get("distance_cm", -1.0) if lidar_data.get("valid") else -1.0
                except LiDARReadError as e:
                    logger.warning(f"LiDAR Read Error: {e}")
                    lidar_dist = -1.0
                    
                # 2. Read Ultrasonic
                ultra_dist = self.ultrasonic.read_frame_current()
                if ultra_dist is None:
                    ultra_dist = -1.0

                # 3. Read Camera
                cam_label = self.camera.get_latest_label()

                # 4. Maintain Baselines
                if lidar_dist > 0:
                    self.lidar_buffer.append(lidar_dist)
                if ultra_dist > 0:
                    self.ultra_buffer.append(ultra_dist)

                # 5. Classify & Fusion
                final_state = self.classify_state(lidar_dist, ultra_dist, cam_label)
                
                # Terminal output for verification
                logger.info(f"LiDAR: {lidar_dist:6.1f}cm | Ultra: {ultra_dist:6.1f}cm | Cam: {cam_label:15s} => STATE: {final_state}")
                
                # Loop rate roughly 10 Hz (0.1s)
                time.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("Stopping Sensor Fusion...")
        finally:
            self.lidar.close()
            self.ultrasonic.close()
            self.camera.close()

if __name__ == "__main__":
    fusion_system = SensorFusionSystem()
    fusion_system.run()
