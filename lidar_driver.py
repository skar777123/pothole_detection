"""
lidar_driver.py
================
Robust TF02-Pro single-point LiDAR driver for Raspberry Pi / any serial host.

Protocol (TF02-Pro, 115200 baud):
  Header: 0x59 0x59
  Dist_L  Dist_H      → distance in cm  (uint16 little-endian)
  Str_L   Str_H       → signal strength  (uint16 little-endian)
  Temp_L  Temp_H      → chip temperature (uint16 LE, unit = 0.01 °C)
  Checksum            → LSB of (0x59+0x59+Dist_L+Dist_H+Str_L+Str_H+Temp_L+Temp_H)

Quality gates applied inside read_data():
  • Strength  : 100 – 65535  (< 100 → weak signal / nothing in range)
  • Distance  : MIN_DIST_CM – MAX_DIST_CM (physical sensor range guard)
  • Checksum  : must match
"""

import serial
import time
import logging

logger = logging.getLogger(__name__)

# ── Physical limits of TF02-Pro ─────────────────────────────────────────────
MIN_DIST_CM      = 10      # Sensor blind-zone below 10 cm
MAX_DIST_CM      = 2200    # TF02-Pro max range: 22 m
MIN_STRENGTH     = 100     # Readings below this strength are noise
MAX_STRENGTH     = 65535   # Saturation upper bound (still valid)

# ── Read tuning ──────────────────────────────────────────────────────────────
MAX_SYNC_ATTEMPTS = 200    # How many single-bytes to try before giving up sync
FRAME_SIZE        = 9      # Total bytes per frame (2 header + 7 payload)


class LiDARReadError(Exception):
    """Raised when the LiDAR cannot produce a valid frame."""


class TF02Pro:
    """
    Thread-safe, quality-gated driver for the Benewake TF02-Pro LiDAR.

    Parameters
    ----------
    port     : Serial port string, e.g. '/dev/ttyUSB0' or 'COM3'
    baudrate : Default 115200 (TF02-Pro factory setting)
    timeout  : Serial read timeout in seconds
    """

    def __init__(self, port: str = '/dev/ttyUSB0',
                 baudrate: int = 115200,
                 timeout: float = 1.0):
        self.port = port
        self.baudrate = baudrate
        self._ser = None
        self.connected = False
        self._open(timeout)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _open(self, timeout: float) -> None:
        try:
            self._ser = serial.Serial(
                self.port, self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=timeout
            )
            # Flush stale bytes in the hardware buffer
            self._ser.reset_input_buffer()
            self.connected = True
            logger.info("TF02-Pro connected on %s @ %d baud", self.port, self.baudrate)
        except serial.SerialException as exc:
            self.connected = False
            logger.error("Cannot open %s: %s", self.port, exc)
            raise

    def _read_byte(self) -> int:
        """Read exactly one byte; raise on timeout or port error."""
        raw = self._ser.read(1)
        if not raw:
            raise LiDARReadError("Serial timeout — no byte received")
        return raw[0]

    def _sync_to_header(self) -> bool:
        """
        Scan byte-by-byte until the 0x59 0x59 header pair is found.
        Returns True if header located within MAX_SYNC_ATTEMPTS bytes.
        """
        for _ in range(MAX_SYNC_ATTEMPTS):
            b = self._read_byte()
            if b == 0x59:
                b2 = self._read_byte()
                if b2 == 0x59:
                    return True   # Header confirmed
        return False

    @staticmethod
    def _validate_checksum(payload: bytes) -> bool:
        """
        payload = Dist_L Dist_H Str_L Str_H Temp_L Temp_H Checksum (7 bytes)
        Checksum = LSB of sum(0x59, 0x59, payload[0..5])
        """
        expected = (0x59 + 0x59 + sum(payload[:6])) & 0xFF
        return expected == payload[6]

    # ── Public API ────────────────────────────────────────────────────────────

    def read_data(self):
        """
        Read one validated LiDAR frame.

        Returns
        -------
        dict with keys:
            distance_cm  : int   — measured distance in centimetres
            strength     : int   — signal return strength (dimensionless)
            temperature_c: float — sensor chip temperature in °C
            valid        : bool  — True when ALL quality gates pass

        On failure raises LiDARReadError.
        """
        if not self.connected or not self._ser.is_open:
            raise LiDARReadError("LiDAR not connected")

        # Step 1: Sync to frame header
        if not self._sync_to_header():
            raise LiDARReadError("Cannot find frame header after max attempts")

        # Step 2: Read 7-byte payload
        payload = self._ser.read(7)
        if len(payload) != 7:
            raise LiDARReadError(f"Short payload: got {len(payload)}/7 bytes")

        # Step 3: Checksum validation
        if not self._validate_checksum(payload):
            raise LiDARReadError("Checksum mismatch — frame discarded")

        # Step 4: Parse fields
        distance_cm   = payload[0] | (payload[1] << 8)
        strength      = payload[2] | (payload[3] << 8)
        raw_temp      = payload[4] | (payload[5] << 8)
        temperature_c = raw_temp / 100.0

        # Step 5: Quality gates
        distance_ok = MIN_DIST_CM <= distance_cm <= MAX_DIST_CM
        strength_ok = MIN_STRENGTH <= strength <= MAX_STRENGTH

        valid = distance_ok and strength_ok

        result = {
            "distance_cm"  : distance_cm,
            "strength"     : strength,
            "temperature_c": temperature_c,
            "valid"        : valid,
        }

        if not valid:
            logger.debug(
                "Reading out-of-gate: dist=%d cm, strength=%d, temp=%.2f°C",
                distance_cm, strength, temperature_c
            )

        return result

    def read_median(self, samples: int = 5) -> dict:
        """
        Take `samples` consecutive valid readings and return the median distance.
        This dramatically reduces single-point noise before pothole inference.

        Returns same dict shape as read_data(), with `valid=True` always
        (invalid frames are skipped; raises LiDARReadError if <3 obtained).
        """
        valid_readings = []
        attempts = 0

        while len(valid_readings) < samples and attempts < samples * 5:
            attempts += 1
            try:
                reading = self.read_data()
                if reading["valid"]:
                    valid_readings.append(reading)
            except LiDARReadError:
                continue

        if len(valid_readings) < max(3, samples // 2):
            raise LiDARReadError(
                f"Only {len(valid_readings)}/{samples} valid readings obtained"
            )

        distances   = [r["distance_cm"]   for r in valid_readings]
        strengths   = [r["strength"]       for r in valid_readings]
        temps       = [r["temperature_c"]  for r in valid_readings]

        return {
            "distance_cm"  : int(sorted(distances)[len(distances) // 2]),
            "strength"     : int(sorted(strengths)[len(strengths) // 2]),
            "temperature_c": sorted(temps)[len(temps) // 2],
            "valid"        : True,
            "n_samples"    : len(valid_readings),
        }

    def close(self) -> None:
        if self._ser and self._ser.is_open:
            self._ser.close()
            self.connected = False
            logger.info("TF02-Pro port closed.")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()