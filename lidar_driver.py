"""
lidar_driver.py
================
TF02-Pro single-point LiDAR driver — continuous streaming mode.

STREAMING DESIGN
─────────────────
  The TF02-Pro sends frames at 100 Hz regardless of what the host does.
  The correct read strategy is:

    1. Flush the OS serial buffer ONCE at startup (discard stale backlog).
    2. After that, read frames CONTINUOUSLY from the live byte stream.
       The sensor keeps emitting; we just parse frame-by-frame.
    3. No sleep, no re-flush between reads → latency ≈ one frame (10 ms).

  This replaces the earlier "flush + sleep(15ms) + read" pattern which:
    • Caused artificial 15ms gaps between readings.
    • Discarded perfectly good frames (potential detections lost).
    • Still had stale-data risk when the sleep was too short.

TF02-Pro 9-byte frame format (115200 baud, 100 Hz):
  Byte 0   : 0x59  ─ Header
  Byte 1   : 0x59  ─ Header
  Byte 2   : Dist_L ─┐ Distance  (cm, uint16 little-endian)
  Byte 3   : Dist_H ─┘
  Byte 4   : Flux_L ─┐ Signal flux (uint16 little-endian)
  Byte 5   : Flux_H ─┘
  Byte 6   : Temp_L ─┐ Chip temperature (uint16 LE, unit = 0.01 °C)
  Byte 7   : Temp_H ─┘
  Byte 8   : Checksum = LSB( sum of bytes 0–7 )
"""

import serial
import time
import logging

logger = logging.getLogger(__name__)

# ── Physical sensor limits ────────────────────────────────────────────────────
MIN_DIST_CM  = 5       # Hard minimum (sensor blind zone ~10 cm, 5 for safety)
MAX_DIST_CM  = 2200    # Max rated range 22 m
HEADER_BYTE  = 0x59
FRAME_LEN    = 9       # Total bytes per frame


class LiDARReadError(Exception):
    """Raised when a valid frame cannot be obtained from the stream."""


class TF02Pro:
    """
    Continuous-stream driver for the Benewake TF02-Pro single-point LiDAR.

    Parameters
    ----------
    port     : Serial port  (e.g. '/dev/ttyUSB0'  or  'COM3')
    baudrate : 115200 (factory default)
    timeout  : Per-byte read timeout in seconds. Keep low for fast response.
    """

    def __init__(self, port: str = '/dev/ttyUSB0',
                 baudrate: int = 115200,
                 timeout: float = 0.1):
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
            # ONE-TIME flush only: discard any stale bytes that arrived
            # before we opened the port (hardware buffer backlog).
            self._ser.reset_input_buffer()
            time.sleep(0.05)           # let sensor emit 5 fresh frames
            self._ser.reset_input_buffer()   # flush those too → clean slate

            self.connected = True
            logger.info("TF02-Pro opened: %s @ %d baud (continuous mode)",
                        self.port, self.baudrate)

        except serial.SerialException as exc:
            logger.error("Cannot open %s: %s", self.port, exc)
            raise

    def _read_bytes(self, n: int) -> bytes:
        """Read exactly n bytes from the stream; raise on timeout."""
        data = self._ser.read(n)
        if len(data) != n:
            raise LiDARReadError(
                f"Expected {n} bytes, got {len(data)} — serial timeout"
            )
        return data

    def _sync_and_read_frame(self) -> bytes:
        """
        Scan the byte stream until the 0x59 0x59 header pair is found,
        then read the remaining 7 bytes and return the full 9-byte frame.

        This is the core continuous-stream parser. It is O(1) amortised
        because when the stream is healthy we are already byte-aligned at
        the start of a new frame and the first two bytes ARE the header.

        Scans at most 4 × FRAME_LEN bytes before giving up.
        """
        max_scan = 4 * FRAME_LEN          # ~36 bytes = 4 frames of slop
        for _ in range(max_scan):
            b1 = self._read_bytes(1)
            if b1[0] != HEADER_BYTE:
                continue
            b2 = self._read_bytes(1)
            if b2[0] != HEADER_BYTE:
                continue
            # Header confirmed — read the 7-byte payload
            payload = self._read_bytes(FRAME_LEN - 2)
            return b1 + b2 + payload

        raise LiDARReadError("Frame header not found within scan limit")

    @staticmethod
    def _checksum_ok(frame: bytes) -> bool:
        """Checksum = LSB of sum of first 8 bytes. Must match byte 8."""
        return (sum(frame[:8]) & 0xFF) == frame[8]

    @staticmethod
    def _parse_frame(frame: bytes) -> dict:
        """Extract distance, flux, and temperature from a validated frame."""
        dist_cm  = frame[2] | (frame[3] << 8)
        strength = frame[4] | (frame[5] << 8)
        raw_temp = frame[6] | (frame[7] << 8)
        temp_c   = round(raw_temp / 100.0, 1)
        return {
            "distance_cm"  : dist_cm,
            "strength"     : strength,
            "temperature_c": temp_c,
        }

    # ─── Public API ───────────────────────────────────────────────────────────

    def read_frame(self) -> dict:
        """
        Read the NEXT frame from the continuous serial stream.

        No buffer flushing — this is pure stream reading. Call this in a
        tight loop and you will receive readings at the sensor's native
        rate (up to 100 Hz for TF02-Pro).

        Returns
        -------
        dict:
            distance_cm   : int   — distance to target in centimetres
            strength      : int   — signal flux (higher = stronger return)
            temperature_c : float — sensor chip temperature in °C
            valid         : bool  — True if distance is within physical range

        Raises LiDARReadError on timeout or checksum failure.
        """
        if not self.connected or not self._ser.is_open:
            raise LiDARReadError("LiDAR port is not open")

        frame = self._sync_and_read_frame()

        if not self._checksum_ok(frame):
            raise LiDARReadError(
                f"Checksum mismatch — raw: {frame.hex(' ')}"
            )

        data  = self._parse_frame(frame)
        valid = MIN_DIST_CM <= data["distance_cm"] <= MAX_DIST_CM

        logger.debug(
            "dist=%d cm  str=%d  temp=%.1f°C  valid=%s",
            data["distance_cm"], data["strength"],
            data["temperature_c"], valid
        )

        data["valid"] = valid

        if not valid:
            raise LiDARReadError(
                f"Distance {data['distance_cm']} cm out of range "
                f"[{MIN_DIST_CM}–{MAX_DIST_CM} cm] — "
                f"strength={data['strength']}"
            )

        return data

    def read_median(self, samples: int = 3) -> dict:
        """
        Read `samples` consecutive frames from the live stream and return
        the one with the median distance value.

        Uses the continuous stream — no flushing between samples.
        This gives a noise-robust reading with minimal added latency
        (3 frames at 100 Hz = ~30 ms total).
        """
        collected = []
        errors    = []

        for _ in range(samples * 3):        # allow some retries
            if len(collected) >= samples:
                break
            try:
                r = self.read_frame()
                collected.append(r)
            except LiDARReadError as exc:
                errors.append(str(exc))

        if not collected:
            raise LiDARReadError(
                f"No valid frames in {samples * 3} attempts: "
                + "; ".join(errors[:2])
            )

        collected.sort(key=lambda r: r["distance_cm"])
        med = collected[len(collected) // 2]
        med["n_samples"] = len(collected)
        return med

    def close(self) -> None:
        if self._ser and self._ser.is_open:
            self._ser.close()
            self.connected = False
            logger.info("TF02-Pro port closed.")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()