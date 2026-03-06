import time
import threading
import logging

logger = logging.getLogger(__name__)

# Attempt to load RPi.GPIO. If it fails (e.g. running on Windows), use a mock class.
try:
    import RPi.GPIO as GPIO
    ON_PI = True
except ImportError:
    logger.warning("RPi.GPIO not found. Running with MOCK Ultrasonic sensor.")
    ON_PI = False


class UltrasonicDriver:
    """ 
    Driver for reading a direct HC-SR04 ultrasonic sensor via GPIO.
    Uses a background daemon thread to constantly poll the sensor 
    so the slow `pulseIn` waiting periods don't block the fast LiDAR loop.
    """
    def __init__(self, trig_pin=23, echo_pin=24):
        self.trig_pin = trig_pin
        self.echo_pin = echo_pin
        self.latest_distance = None  # None indicates no valid reading yet
        
        self.running = True
        self.thread = threading.Thread(target=self._poll_loop, daemon=True)
        
        self._setup_pins()
        self.thread.start()

    def _setup_pins(self):
        if not ON_PI:
            logger.info("MOCK: Ultrasonic pins setup for Trigger=%d, Echo=%d", 
                        self.trig_pin, self.echo_pin)
            return

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(self.trig_pin, GPIO.OUT)
        GPIO.setup(self.echo_pin, GPIO.IN)
        GPIO.output(self.trig_pin, False)
        
        logger.info("Ultrasonic GPIO setup complete. Settling sensor...")
        time.sleep(1) # Allow sensor to settle

    def _read_distance_cm(self):
        """ Blocking hardware read of the HC-SR04 pulse. """
        if not ON_PI:
            # Mock return a flat fake distance for Windows testing
            time.sleep(0.05)
            # return ~40cm with slight random noise
            import random
            return 40.0 + random.uniform(-0.5, 0.5)

        # 1. Ensure trigger is low
        GPIO.output(self.trig_pin, False)
        time.sleep(0.002)

        # 2. Fire 10us HIGH pulse
        GPIO.output(self.trig_pin, True)
        time.sleep(0.00001)
        GPIO.output(self.trig_pin, False)

        pulse_start = time.time()
        pulse_end = time.time()
        
        timeout_start = time.time()

        # 3. Wait for echo to go HIGH
        while GPIO.input(self.echo_pin) == 0:
            pulse_start = time.time()
            if pulse_start - timeout_start > 0.1: # 100ms timeout
                return -1.0 # Out of range / error

        # 4. Wait for echo to drop LOW
        while GPIO.input(self.echo_pin) == 1:
            pulse_end = time.time()
            if pulse_end - timeout_start > 0.1:
                return -1.0

        # 5. Math Time
        pulse_duration = pulse_end - pulse_start
        # distance = time * speed of sound (34300 cm/s) / 2
        distance = pulse_duration * 17150
        
        return round(distance, 2)

    def _poll_loop(self):
        """ Background thread constantly updating latest_distance. """
        while self.running:
            try:
                dist = self._read_distance_cm()
                if dist > 0 and dist <= 1000: # Max 10 meters (1000cm)
                    self.latest_distance = dist
                else:
                    self.latest_distance = -1.0 # Error flag
            except Exception as e:
                logger.error("Ultrasonic read failure: %s", e)
                
            # Sleep ~40ms to prevent ultrasonic echoes bouncing off walls overlapping
            time.sleep(0.040)

    def read_frame_current(self):
        """ Instantly returns the latest distance measured by the background thread. """
        return self.latest_distance

    def close(self):
        self.running = False
        if ON_PI:
            GPIO.cleanup((self.trig_pin, self.echo_pin))
            logger.info("Ultrasonic GPIO cleaned up.")

    def __enter__(self): return self
    def __exit__(self, *_): self.close()
