"""
lidar_driver.py
================
TF02-Pro single-point LiDAR driver — with startup command init + diagnostics.

WHY "Expected 1 bytes, got 0 (serial timeout)"
───────────────────────────────────────────────
  The port opens fine (USB-UART adapter found) but the sensor sends ZERO bytes.
  Possible causes:
    1. Sensor is in TRIGGER mode (only outputs when it receives a trigger cmd).
    2. TX wire from sensor not connected to UART adapter RX pin.
    3. Wrong baud rate (at completely wrong rate → silence, not garbage).
    4. Sensor in output-disabled state (needs "enable output" command).

  Fix applied:
    - On startup, send soft-reset + enable-output + set-100Hz commands.
    - Auto-detect baud rate: try 115200 first, then 9600.
    - Provide a raw_read_bytes() method for diagnostics (no header sync).

TF02-Pro Command Protocol (host → sensor):
  [0x5A][LEN][ID][DATA...][CS]
  CS = LSB of sum of all bytes before CS.

TF02-Pro Output Frame (sensor → host, UART, 100Hz):
  Byte 0-1 : 0x59 0x59 (header)
  Byte 2-3 : Distance cm  (uint16 LE)
  Byte 4-5 : Signal flux  (uint16 LE)
  Byte 6-7 : Temperature  (uint16 LE, unit = 0.01°C)
  Byte 8   : Checksum = LSB( sum of bytes 0-7 )
"""

import serial
import serial.tools.list_ports
import time
import logging

logger = logging.getLogger(__name__)

# ── Physical limits ───────────────────────────────────────────────────────────
MIN_DIST_CM = 5
MAX_DIST_CM = 2200
HEADER_BYTE = 0x59
FRAME_LEN   = 9

# ── Startup commands (host → sensor) ─────────────────────────────────────────
# All follow: 5A [LEN] [ID] [DATA...] [CS]  where CS = LSB(sum of all prev bytes)
CMD_SOFT_RESET      = bytes([0x5A, 0x04, 0x02, 0x60])          # Soft reset
CMD_ENABLE_OUTPUT   = bytes([0x5A, 0x05, 0x07, 0x01, 0x67])    # Enable continuous output
CMD_DISABLE_OUTPUT  = bytes([0x5A, 0x05, 0x07, 0x00, 0x66])    # Disable output
CMD_SET_100HZ       = bytes([0x5A, 0x06, 0x03, 0x64, 0x00, 0xC7])  # Set frame rate 100Hz
CMD_TRIGGER_ONCE    = bytes([0x5A, 0x04, 0x04, 0x62])          # Single measurement (trigger)
CMD_SAVE_SETTINGS   = bytes([0x5A, 0x04, 0x11, 0x6F])          # Save current settings to ROM

BAUD_CANDIDATES = [115200, 9600, 19200, 56000]


class LiDARReadError(Exception):
    """Raised when a valid frame cannot be obtained from the sensor."""


class TF02Pro:
    """
    TF02-Pro LiDAR driver.

    Parameters
    ----------
    port        : Serial port (e.g. '/dev/ttyUSB0' or 'COM3')
    baudrate    : 115200 by default. Pass None to auto-detect.
    timeout     : Per-byte read timeout in seconds
    send_init   : If True, send soft-reset + enable-output commands at startup
    """

    def __init__(self,
                 port: str  = '/dev/ttyUSB0',
                 baudrate: int = 115200,
                 timeout: float = 0.5,
                 send_init: bool = True):
        self.port      = port
        self.baudrate  = baudrate
        self._ser      = None
        self.connected = False
        self._open(timeout, send_init)

    # ─── Private ─────────────────────────────────────────────────────────────

    def _open(self, timeout: float, send_init: bool) -> None:
        try:
            self._ser = serial.Serial(
                self.port,
                self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=timeout,
            )
            logger.info("Opened %s @ %d baud", self.port, self.baudrate)

            # Flush any bytes that arrived before the port was open
            self._ser.reset_input_buffer()

            if send_init:
                self._send_startup_commands()

            # Final flush after init commands + sensor reboot
            self._ser.reset_input_buffer()

            self.connected = True
            logger.info("TF02-Pro ready on %s", self.port)

        except serial.SerialException as exc:
            logger.error("Cannot open %s: %s", self.port, exc)
            raise

    def _send_startup_commands(self) -> None:
        """
        Send the command sequence that ensures the sensor is streaming:
          1. Soft reset      — clear any stuck state
          2. Wait 1 s        — sensor reboot takes ~500ms
          3. Enable output   — in case sensor was in trigger/disabled mode
          4. Set 100Hz rate  — ensure continuous at rated speed
        """
        logger.info("Sending TF02-Pro init commands …")

        try:
            # 1. Soft reset
            self._ser.write(CMD_SOFT_RESET)
            self._ser.flush()
            logger.info("  → Soft reset sent (5A 04 02 60)")

            # 2. Wait for reboot (TF02-Pro takes ~500ms to restart)
            time.sleep(1.0)
            self._ser.reset_input_buffer()

            # 3. Enable continuous output
            self._ser.write(CMD_ENABLE_OUTPUT)
            self._ser.flush()
            logger.info("  → Enable output sent (5A 05 07 01 67)")
            time.sleep(0.1)

            # 4. Set 100 Hz output rate
            self._ser.write(CMD_SET_100HZ)
            self._ser.flush()
            logger.info("  → Set 100Hz sent (5A 06 03 64 00 C7)")
            time.sleep(0.1)

            logger.info("Init commands complete. Waiting for sensor to stream …")

        except serial.SerialException as exc:
            logger.warning("Could not send init commands: %s", exc)

    def _read_raw_bytes(self, n: int) -> bytes:
        """Read exactly n bytes; raise LiDARReadError on timeout."""
        data = self._ser.read(n)
        if len(data) != n:
            raise LiDARReadError(
                f"Expected {n} bytes, got {len(data)} (serial timeout — "
                f"sensor may be in trigger mode or TX wire disconnected)"
            )
        return data

    def _sync_and_read_frame(self) -> bytes:
        """
        Scan the byte stream for the 0x59 0x59 header, then read the
        remaining 7 bytes to form a complete 9-byte frame.
        """
        max_scan = 6 * FRAME_LEN   # scan up to 6 frames of bytes
        for i in range(max_scan):
            b1 = self._read_raw_bytes(1)
            if b1[0] != HEADER_BYTE:
                continue
            b2 = self._read_raw_bytes(1)
            if b2[0] != HEADER_BYTE:
                continue
            payload = self._read_raw_bytes(FRAME_LEN - 2)
            return b1 + b2 + payload

        raise LiDARReadError("Frame header (0x59 0x59) not found in stream")

    @staticmethod
    def _checksum_ok(frame: bytes) -> bool:
        return (sum(frame[:8]) & 0xFF) == frame[8]

    @staticmethod
    def _parse_frame(frame: bytes) -> dict:
        dist_cm  = frame[2] | (frame[3] << 8)
        strength = frame[4] | (frame[5] << 8)
        raw_temp = frame[6] | (frame[7] << 8)
        return {
            "distance_cm"  : dist_cm,
            "strength"     : strength,
            "temperature_c": round(raw_temp / 100.0, 1),
        }

    # ─── Diagnostics ─────────────────────────────────────────────────────────

    def diagnostic_raw_dump(self, n_bytes: int = 90) -> bytes:
        """
        Read n_bytes raw bytes from the serial port WITHOUT any parsing.
        Used to verify the sensor is sending ANYTHING at all.
        Returns the raw bytes received (may be empty if sensor is silent).
        """
        logger.info("Raw diagnostic: reading %d bytes …", n_bytes)
        self._ser.reset_input_buffer()
        time.sleep(0.1)
        data = self._ser.read(n_bytes)
        logger.info("Raw diagnostic: got %d bytes: %s",
                    len(data), data.hex(' ') if data else '(nothing)')
        return data

    def send_trigger(self) -> None:
        """Send a single-measurement trigger (for sensors in trigger mode)."""
        self._ser.write(CMD_TRIGGER_ONCE)
        self._ser.flush()

    def enable_output(self) -> None:
        """Re-enable continuous output (e.g. after a disable)."""
        self._ser.write(CMD_ENABLE_OUTPUT)
        self._ser.flush()
        time.sleep(0.05)

    # ─── Public API ───────────────────────────────────────────────────────────

    def read_frame(self) -> dict:
        """
        Read the next frame from the continuous serial stream.
        No buffer management — use read_frame_current() for real-time response.
        """
        if not self.connected or not self._ser.is_open:
            raise LiDARReadError("LiDAR port is not open")

        frame = self._sync_and_read_frame()

        if not self._checksum_ok(frame):
            raise LiDARReadError(f"Checksum fail  raw={frame.hex(' ')}")

        data  = self._parse_frame(frame)
        valid = MIN_DIST_CM <= data["distance_cm"] <= MAX_DIST_CM
        data["valid"] = valid

        logger.debug("dist=%d cm  str=%d  temp=%.1f°C  valid=%s",
                     data["distance_cm"], data["strength"],
                     data["temperature_c"], valid)

        if not valid:
            raise LiDARReadError(
                f"Distance {data['distance_cm']} cm outside range "
                f"[{MIN_DIST_CM}–{MAX_DIST_CM}]"
            )
        return data

    def read_frame_current(self) -> dict:
        """
        Read the CURRENT (freshest) frame, auto-discarding any stale backlog.

        WHY THIS IS NEEDED
        ──────────────────
        The sensor outputs at 100 Hz (one 9-byte frame every 10 ms).
        Streamlit UI updates take 50–150 ms per iteration, so the OS serial
        buffer accumulates 5–15 stale frames per iteration. After 10 s of
        running, the buffer can hold 150+ old frames, introducing 1.5 s+ lag.

        HOW THIS FIX WORKS
        ──────────────────
        Before reading, check how many bytes are queued in the OS buffer.
        If > FRAME_LEN * 3 bytes (3+ frames worth), the data is stale —
        flush and wait 12 ms for the sensor to emit one fresh frame.
        If buffer is small (fresh), just read normally with no overhead.
        """
        if not self.connected or not self._ser.is_open:
            raise LiDARReadError("LiDAR port is not open")

        # Auto-drain: if more than 3 frames queued, buffer is stale
        stale_threshold = FRAME_LEN * 3   # 27 bytes = 3 frames
        queued = self._ser.in_waiting
        if queued > stale_threshold:
            logger.debug(
                "Draining %d stale bytes (%d frames behind)",
                queued, queued // FRAME_LEN
            )
            self._ser.reset_input_buffer()
            time.sleep(0.012)   # 12ms > one frame period (10ms) at 100Hz

        return self.read_frame()

    def read_median(self, samples: int = 3) -> dict:
        """Read `samples` frames and return the one with the median distance."""
        collected, errors = [], []
        for _ in range(samples * 3):
            if len(collected) >= samples:
                break
            try:
                collected.append(self.read_frame())
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


# ── Utility: list available serial ports ─────────────────────────────────────

def list_ports() -> list[str]:
    """Return all available serial port names on this machine."""
    return [p.device for p in serial.tools.list_ports.comports()]