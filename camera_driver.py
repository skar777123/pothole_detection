import serial
import threading
import time
import logging

logger = logging.getLogger(__name__)

class CameraDriver:
    """
    Driver for reading classification data from ESP32-CAM via UART.
    The ESP32-CAM should be connected to Raspberry Pi's GPIO14 (TX) and GPIO15 (RX).
    """
    def __init__(self, port='/dev/serial0', baudrate=115200, timeout=1.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        
        self.latest_label = "Normal road"
        self.running = True
        
        try:
            self._ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            logger.info("Camera UART Port opened: %s @ %d baud", self.port, self.baudrate)
        except Exception as e:
            logger.warning("Failed to open Camera UART, using MOCK values. Error: %s", e)
            self._ser = None
        
        self.thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.thread.start()

    def _poll_loop(self):
        """ Background thread to read strings from ESP32 UART. """
        while self.running:
            if self._ser and self._ser.is_open:
                try:
                    if self._ser.in_waiting > 0:
                        line = self._ser.readline().decode('utf-8', errors='ignore').strip()
                        if line:
                            label_lower = line.lower()
                            if "pothole" in label_lower:
                                self.latest_label = "Pothole detected"
                            elif "bump" in label_lower:
                                self.latest_label = "Speed bump detected"
                            elif "normal" in label_lower:
                                self.latest_label = "Normal road"
                            else:
                                # Could be raw data, ignore or log
                                pass
                except Exception as e:
                    logger.error("Camera UART read failure: %s", e)
            else:
                # Mock behavior when not connected
                time.sleep(1)
            time.sleep(0.01)

    def get_latest_label(self):
        """ Instantly returns the latest classification from ESP32-CAM. """
        return self.latest_label

    def close(self):
        self.running = False
        if self._ser and self._ser.is_open:
            self._ser.close()
            logger.info("Camera UART closed.")

    def __enter__(self): return self
    def __exit__(self, *_): self.close()
