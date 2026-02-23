"""
lidar_driver.py
================
TF02-Pro single-point LiDAR driver — with buffer-flush fix.

ROOT CAUSE OF "FROZEN DISTANCE" BUG
─────────────────────────────────────
  The TF02-Pro sends frames at 100 Hz (one every 10 ms).
  If the host reads slower than 100 Hz, the OS serial buffer fills with OLD
  frames. Every read() call returns stale data from the front of the queue,
  so the distance appears frozen at whatever value it was buffered at.

  FIX: Call  _ser.reset_input_buffer()  immediately before reading a new
  frame. This discards all queued bytes and forces the next read to wait
  for the sensor's very next output — i.e. the CURRENT distance.

TF02-Pro 9-byte frame format (115200 baud):
  Byte 0   : 0x59  (header)
  Byte 1   : 0x59  (header)
  Byte 2-3 : Distance cm  (uint16, little-endian)
  Byte 4-5 : Signal flux  (uint16, little-endian)
  Byte 6-7 : Temperature  (uint16, little-endian, unit = 0.01 °C)
  Byte 8   : Checksum = LSB( sum of bytes 0-7 )
"""

import serial
import time
import logging

logger = logging.getLogger(__name__)

# ── Physical sensor limits ────────────────────────────────────────────────────
MIN_DIST_CM   = 5       # TF02-Pro blind zone is ~10 cm; set to 5 for safety
MAX_DIST_CM   = 2200    # Max rated range 22 m
HEADER_BYTE   = 0x59
FRAME_LEN     = 9       # Total bytes per frame


class LiDARReadError(Exception):
    """Raised when a valid frame cannot be obtained."""


class TF02Pro:
    """
    TF02-Pro LiDAR driver with current-frame guarantee.

    Parameters
    ----------
    port     : e.g. '/dev/ttyUSB0' (Linux/Raspberry Pi) or 'COM3' (Windows)
    baudrate : factory default 115200
    timeout  : per-byte read timeout in seconds
    """

    def __init__(self, port: str = '/dev/ttyUSB0',
                 baudrate: int = 115200,
                 timeout: float = 0.5):
        self.port      = port
        self.baudrate  = baudrate
        self._ser      = None
        self.connected = False
        self._open(timeout)

    # ─── Private ─────────────────────────────────────────────────────────────

    def _open(self, timeout: float) -> None:
        try:
            self._ser = serial.Serial(
                self.port,
                self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=timeout,
            )
            # Flush any bytes that arrived while the port was being negotiated
            self._ser.reset_input_buffer()
            time.sleep(0.1)  # Give sensor time to start fresh output
            self._ser.reset_input_buffer()
            self.connected = True
            logger.info("TF02-Pro opened: %s @ %d baud", self.port, self.baudrate)
        except serial.SerialException as exc:
            logger.error("Cannot open %s: %s", self.port, exc)
            raise

    def _read_bytes(self, n: int) -> bytes:
        """Read exactly n bytes; raise on timeout."""
        data = self._ser.read(n)
        if len(data) != n:
            raise LiDARReadError(
                f"Expected {n} bytes, got {len(data)} (serial timeout)"
            )
        return data

    def _read_one_frame_raw(self) -> bytes:
        """
        Synchronise to the next 0x59 0x59 header pair and return the
        complete 9-byte frame (including the two header bytes).

        Scans at most 4 × FRAME_LEN bytes before giving up.
        """
        max_scan = 4 * FRAME_LEN
        for _ in range(max_scan):
            b = self._read_bytes(1)
            if b[0] != HEADER_BYTE:
                continue
            b2 = self._read_bytes(1)
            if b2[0] != HEADER_BYTE:
                continue
            # Found header — read remaining 7 bytes
            rest = self._read_bytes(FRAME_LEN - 2)
            return b + b2 + rest
        raise LiDARReadError("Frame header not found within scan limit")

    @staticmethod
    def _checksum_ok(frame: bytes) -> bool:
        """LSB of sum of first 8 bytes must equal byte 9 (index 8)."""
        return (sum(frame[:8]) & 0xFF) == frame[8]

    # ─── Public API ───────────────────────────────────────────────────────────

    def read_current(self) -> dict:
        """
        Flush the serial buffer, then read ONE fresh frame from the sensor.

        Returns a dict:
            distance_cm   : int   — current distance to target
            strength      : int   — raw signal flux
            temperature_c : float — sensor chip temp in °C
            valid         : bool  — True if distance is in physical range

        Raises LiDARReadError on timeout or bad checksum.
        """
        if not self.connected or not self._ser.is_open:
            raise LiDARReadError("LiDAR port is not open")

        # ── KEY FIX: discard all buffered (stale) bytes ───────────────────────
        self._ser.reset_input_buffer()

        # Wait for sensor to emit a fresh frame.
        # At 100 Hz one frame takes 9 bytes × ~87 µs/byte ≈ 0.8 ms to transmit,
        # but the inter-frame gap means we need ~12 ms for a guaranteed fresh one.
        time.sleep(0.015)

        frame = self._read_one_frame_raw()

        if not self._checksum_ok(frame):
            raise LiDARReadError(
                f"Checksum fail — raw: {frame.hex(' ')}"
            )

        dist_cm  = frame[2] | (frame[3] << 8)
        strength = frame[4] | (frame[5] << 8)
        raw_temp = frame[6] | (frame[7] << 8)
        temp_c   = raw_temp / 100.0

        in_range = MIN_DIST_CM <= dist_cm <= MAX_DIST_CM

        logger.debug(
            "Frame → dist=%d cm  strength=%d  temp=%.1f°C  in_range=%s",
            dist_cm, strength, temp_c, in_range
        )

        if not in_range:
            raise LiDARReadError(
                f"Distance {dist_cm} cm out of valid range "
                f"[{MIN_DIST_CM}–{MAX_DIST_CM} cm]"
            )

        return {
            "distance_cm"  : dist_cm,
            "strength"     : strength,
            "temperature_c": round(temp_c, 1),
            "valid"        : in_range,
        }

    def read_median(self, samples: int = 3) -> dict:
        """
        Take `samples` fresh readings (each with a buffer flush) and
        return the one with the median distance.

        Using median instead of mean rejects single-spike outliers.
        """
        collected = []
        errors    = []

        for _ in range(samples):
            try:
                r = self.read_current()
                collected.append(r)
            except LiDARReadError as exc:
                errors.append(str(exc))

        if not collected:
            raise LiDARReadError(
                f"All {samples} reads failed: {'; '.join(errors[:3])}"
            )

        # Sort by distance and pick the median
        collected.sort(key=lambda r: r["distance_cm"])
        median_r = collected[len(collected) // 2]
        median_r["n_samples"] = len(collected)
        return median_r

    def close(self) -> None:
        if self._ser and self._ser.is_open:
            self._ser.close()
            self.connected = False
            logger.info("TF02-Pro port closed.")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()