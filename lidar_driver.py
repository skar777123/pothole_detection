"""
lidar_driver.py
================
Robust TF02-Pro single-point LiDAR driver.

TF02-Pro 9-byte frame protocol (115200 baud):
  Byte 0-1 : 0x59 0x59  (frame header)
  Byte 2   : Dist_L      ─┐ Distance  (cm, uint16 LE)
  Byte 3   : Dist_H      ─┘
  Byte 4   : Str_L       ─┐ Signal strength (uint16 LE)
  Byte 5   : Str_H       ─┘   • 0      = no target / saturated
             (TF02-Pro NOT TF-Luna — Temp bytes are NOT present in TF02Pro)
  Byte 6   : reserved (0x00 on most firmware; treated as Temp_L for compat)
  Byte 7   : reserved (treated as Temp_H)
  Byte 8   : Checksum = LSB of sum(bytes 0..7)

Quality gates (two levels):
  Level 1 – FRAME VALID  : checksum passes + distance in sensor range (10–2200 cm)
  Level 2 – SIGNAL GOOD  : strength >= MIN_STRENGTH_GOOD
                            Readings below this are still returned, just flagged.

WHY we lowered the strength threshold:
  Asphalt at ~180 cm commonly returns strength 20–80 on the TF02-Pro.
  The old threshold of 100 was designed for retro-reflective tape targets.
  We now accept any strength >= 20 as "usable", and log a warning for < 20.
"""

import serial
import time
import logging

logger = logging.getLogger(__name__)

# ── Physical limits of TF02-Pro ────────────────────────────────────────────
MIN_DIST_CM        = 10      # Blind zone below 10 cm
MAX_DIST_CM        = 2200    # Max rated range: 22 m
MIN_STRENGTH_GOOD  = 20      # Reliable signal floor for natural surfaces
                              # (retro targets can be ≥ 200; asphalt ≈ 20–80)

# ── Protocol constants ──────────────────────────────────────────────────────
HEADER_BYTE        = 0x59
FRAME_PAYLOAD_LEN  = 7       # bytes after the two header bytes
MAX_SYNC_ATTEMPTS  = 500     # bytes to scan before giving up frame sync


class LiDARReadError(Exception):
    """Raised when a valid frame cannot be obtained."""


class TF02Pro:
    """
    Quality-gated driver for the Benewake TF02-Pro single-point LiDAR.

    Parameters
    ----------
    port     : Serial port, e.g. '/dev/ttyUSB0' or 'COM3'
    baudrate : Factory default 115200
    timeout  : Per-byte serial read timeout (seconds)
    """

    def __init__(self, port: str = '/dev/ttyUSB0',
                 baudrate: int = 115200,
                 timeout: float = 1.0):
        self.port      = port
        self.baudrate  = baudrate
        self._ser      = None
        self.connected = False
        self._open(timeout)

    # ── Private helpers ─────────────────────────────────────────────────────

    def _open(self, timeout: float) -> None:
        try:
            self._ser = serial.Serial(
                self.port, self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=timeout,
            )
            self._ser.reset_input_buffer()   # Flush stale hardware buffer
            self.connected = True
            logger.info("TF02-Pro connected → port=%s  baud=%d", self.port, self.baudrate)
        except serial.SerialException as exc:
            self.connected = False
            logger.error("Cannot open %s: %s", self.port, exc)
            raise

    def _read_byte(self) -> int:
        b = self._ser.read(1)
        if not b:
            raise LiDARReadError("Serial timeout – no byte received")
        return b[0]

    def _sync_to_header(self) -> bool:
        """
        Scan incoming bytes until the 0x59 0x59 header pair is found.
        Consumes at most MAX_SYNC_ATTEMPTS bytes.
        Returns True when header is located.
        """
        for _ in range(MAX_SYNC_ATTEMPTS):
            b1 = self._read_byte()
            if b1 == HEADER_BYTE:
                b2 = self._read_byte()
                if b2 == HEADER_BYTE:
                    return True   # Both header bytes confirmed
        return False

    @staticmethod
    def _checksum_ok(all_bytes: bytes) -> bool:
        """
        Validate 9-byte frame checksum.
        Checksum = LSB of sum of first 8 bytes.
        `all_bytes` = [0x59, 0x59, payload[0..5], checksum_byte]  (9 bytes total)
        """
        expected = sum(all_bytes[:8]) & 0xFF
        return expected == all_bytes[8]

    # ── Public API ──────────────────────────────────────────────────────────

    def read_data(self) -> dict:
        """
        Read and parse one 9-byte TF02-Pro frame.

        Returns
        -------
        dict:
            distance_cm   (int)   – distance to target
            strength      (int)   – return signal strength (dimensionless)
            signal_good   (bool)  – True if strength ≥ MIN_STRENGTH_GOOD
            frame_valid   (bool)  – True if checksum passes AND distance in range
            raw_bytes     (bytes) – full 9-byte raw frame for debugging

        Raises LiDARReadError on timeout, short frame, or checksum failure.
        """
        if not self.connected or not self._ser.is_open:
            raise LiDARReadError("LiDAR not connected")

        # 1. Find frame header
        if not self._sync_to_header():
            raise LiDARReadError("Frame header not found within sync limit")

        # 2. Read remaining 7 bytes (payload)
        payload = self._ser.read(FRAME_PAYLOAD_LEN)
        if len(payload) != FRAME_PAYLOAD_LEN:
            raise LiDARReadError(
                f"Short payload: got {len(payload)}/{FRAME_PAYLOAD_LEN} bytes"
            )

        # Reconstruct full frame for checksum
        full_frame = bytes([HEADER_BYTE, HEADER_BYTE]) + payload

        # 3. Validate checksum
        if not self._checksum_ok(full_frame):
            raise LiDARReadError(
                f"Checksum mismatch – raw frame: {full_frame.hex(' ')}"
            )

        # 4. Parse distance and strength
        dist_cm  = payload[0] | (payload[1] << 8)
        strength = payload[2] | (payload[3] << 8)

        # 5. Quality flags
        in_range     = MIN_DIST_CM <= dist_cm <= MAX_DIST_CM
        signal_good  = strength >= MIN_STRENGTH_GOOD
        frame_valid  = in_range   # Checksum already passed; just check range

        logger.debug(
            "Frame → dist=%d cm  strength=%d  in_range=%s  signal_good=%s",
            dist_cm, strength, in_range, signal_good
        )

        if not in_range:
            raise LiDARReadError(
                f"Distance {dist_cm} cm out of range [{MIN_DIST_CM}–{MAX_DIST_CM}]"
            )

        if not signal_good:
            # Log at WARNING level so the user can see actual strength values
            logger.warning(
                "Low signal: dist=%d cm  strength=%d (min good=%d). "
                "Check sensor aim / target reflectivity. Reading still used.",
                dist_cm, strength, MIN_STRENGTH_GOOD
            )

        return {
            "distance_cm" : dist_cm,
            "strength"    : strength,
            "signal_good" : signal_good,
            "frame_valid" : frame_valid,
            "raw_bytes"   : full_frame,
        }

    def read_median(self, samples: int = 3) -> dict:
        """
        Collect `samples` consecutive valid frames and return the median reading.

        A frame is accepted if it passes the checksum + distance-range gate
        (even if signal_good is False).  This prevents the case where the
        sensor consistently reads strength < 20 on a particular surface and
        all readings are rejected, causing "Only 0/N valid readings".

        Returns the same dict shape as read_data(), with an added 'n_samples' key.
        """
        collected = []
        attempts  = 0
        max_tries = samples * 10   # Give a generous retry budget

        while len(collected) < samples and attempts < max_tries:
            attempts += 1
            try:
                r = self.read_data()
                collected.append(r)       # Accept any frame that passed checksum
            except LiDARReadError as exc:
                logger.debug("read_data attempt %d failed: %s", attempts, exc)
                continue

        if len(collected) < max(1, samples // 2):
            raise LiDARReadError(
                f"Only {len(collected)}/{samples} frames obtained "
                f"after {attempts} attempts"
            )

        # Median of collected distances and strengths
        distances = sorted(r["distance_cm"] for r in collected)
        strengths = sorted(r["strength"]    for r in collected)
        n         = len(collected)

        return {
            "distance_cm" : distances[n // 2],
            "strength"    : strengths[n // 2],
            "signal_good" : all(r["signal_good"] for r in collected),
            "frame_valid" : True,
            "n_samples"   : n,
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