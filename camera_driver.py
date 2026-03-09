import cv2
import threading
import time
import logging

logger = logging.getLogger(__name__)

class CameraDriver:
    """
    Driver for reading MJPEG stream from ESP32-CAM over Wi-Fi
    and running local AI (OpenCV/YOLO) on the frames.
    """
    def __init__(self, stream_url="http://192.168.1.100:81/stream", baudrate=None, port=None):
        # We leave baudrate and port in the init args so dashboard.py doesn't crash
        # when it inevitably tries to pass them in. 
        self.stream_url = stream_url
        self.latest_label = "Normal road"
        self.running = True
        self.cap = None

        logger.info("Initializing AI Camera Stream at %s", self.stream_url)
        
        self.thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.thread.start()

    def _poll_loop(self):
        """ Background thread to read video frames and run AI detection """
        self.cap = cv2.VideoCapture(self.stream_url)
        
        # If the stream fails to open, fallback to dummy mode just so the dashboard doesn't hang
        if not self.cap.isOpened():
            logger.warning("Failed to open Camera Stream. Please check the IP address!")
        
        frame_counter = 0
        while self.running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    frame_counter += 1
                    
                    # === RUN AI DETECTION EVERY 10 FRAMES TO SAVE CPU ===
                    if frame_counter % 10 == 0:
                        self.latest_label = self._run_ai_inference(frame)
                        
                else:
                    logger.warning("Camera stream dropped. Retrying...")
                    self.cap.release()
                    time.sleep(2)
                    self.cap = cv2.VideoCapture(self.stream_url)
            else:
                time.sleep(1)
            
            time.sleep(0.01) # Small sleep to prevent maxing out CPU

    def _run_ai_inference(self, frame):
        """
        Placeholder for your actual YOLO/OpenCV Model.
        Right now, it just returns Normal road.
        
        Replace this with actual logic:
        results = my_yolo_model.predict(frame)
        if "pothole" in results: return "Pothole detected"
        ...
        """
        # Example: Fake detection logic. 
        # For now, we just return Normal road to act as the baseline.
        return "Normal road"

    def get_latest_label(self):
        """ Instantly returns the latest classification from the AI """
        return self.latest_label

    def close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        logger.info("Camera Stream closed.")

    def __enter__(self): return self
    def __exit__(self, *_): self.close()
