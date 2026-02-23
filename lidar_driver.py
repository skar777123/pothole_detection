"""
lidar_driver.py — TF02-Pro UART driver. Clean minimal version.

WHAT WENT WRONG PREVIOUSLY
───────────────────────────
We sent CMD_FACTORY_RESET (command 0x10) but on many TF02-Pro firmware
versions, 0x10 is "Set Trigger Mode" not "Factory Reset". We were
accidentally switching the sensor into trigger mode every startup,
then wondering why it stopped streaming.

CORRECT APPROACH
─────────────────
1. Open port
2. Try reading first — sensor may already be streaming
3. Only if silent after 2s: send CMD_ENABLE_OUTPUT
4. On dropout: soft recover (re-enable), then hard recover (reconnect)
5. NEVER send unknown commands
"""

import serial
import serial.tools.list_ports
import threading
import time
import logging
from collections import deque

logger = logging.getLogger(__name__)

MIN_DIST_CM = 1
MAX_DIST_CM = 2200
HEADER_BYTE = 0x59
FRAME_LEN   = 9

# The ONLY safe commands (well-documented for all TF02-Pro firmware):
CMD_ENABLE_OUTPUT = bytes([0x5A, 0x05, 0x07, 0x01, 0x67])
# [0x5A header][0x05 len][0x07 cmd: set-output][0x01 on][CS=0x67]

CMD_SET_100HZ = bytes([0x5A, 0x06, 0x03, 0x64, 0x00, 0xC7])
# [0x5A][0x06][0x03 cmd: set-fps][0x64 0x00 = 100Hz][CS=0xC7]


class LiDARReadError(Exception):
    pass


class TF02Pro:
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200,
                 timeout=0.10, send_init=True):
        self.port      = port
        self.baudrate  = baudrate
        self._timeout  = timeout
        self._ser      = None
        self.connected = False
        self._open_port()
        if send_init:
            self._smart_init()

    # ── Port operations ───────────────────────────────────────────────────────

    def _open_port(self):
        """Open serial port cleanly."""
        if self._ser and self._ser.is_open:
            try: self._ser.close()
            except Exception: pass

        self._ser = serial.Serial(
            self.port, self.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=self._timeout,
        )
        self._ser.reset_input_buffer()
        self.connected = True
        logger.info("Port opened: %s @ %d baud", self.port, self.baudrate)

    def _smart_init(self):
        """
        Try reading first. If sensor is already streaming, do nothing.
        Only send CMD_ENABLE_OUTPUT if sensor is silent for 1 second.
        """
        logger.info("Smart init: checking if sensor already streaming …")
        # Give it 1 second to see if frames arrive on their own
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if self._ser.in_waiting >= FRAME_LEN:
                logger.info("  → Sensor already streaming! No commands needed.")
                self._ser.reset_input_buffer()
                return
            time.sleep(0.05)

        # Nothing arrived — send enable output
        logger.info("  → Sensor silent. Sending enable-output …")
        self._send_enable()

    def _send_enable(self):
        """Send enable-output + set-100Hz. Safe, documented commands only."""
        try:
            self._ser.write(CMD_ENABLE_OUTPUT); self._ser.flush()
            logger.info("  → CMD_ENABLE_OUTPUT sent")
            time.sleep(0.20)
            self._ser.write(CMD_SET_100HZ); self._ser.flush()
            logger.info("  → CMD_SET_100HZ sent")
            time.sleep(0.20)
            self._ser.reset_input_buffer()
        except serial.SerialException as exc:
            logger.warning("Send error: %s", exc)

    # ── Recovery ──────────────────────────────────────────────────────────────

    def _enable_output(self):
        """Soft recovery: re-send enable (called after a few consecutive errors)."""
        try:
            self._ser.write(CMD_ENABLE_OUTPUT); self._ser.flush()
            time.sleep(0.20)
            self._ser.reset_input_buffer()
            logger.info("Soft recovery: enable-output re-sent.")
        except Exception:
            pass

    def reconnect(self):
        """
        Hard recovery: close port, reopen, smart-init.
        Does NOT send any reset commands — just reopen and enable.
        """
        logger.warning("Hard reconnect …")
        try:
            if self._ser and self._ser.is_open:
                self._ser.close()
            time.sleep(0.50)
            self._open_port()
            self._smart_init()
            logger.info("Reconnect done.")
        except Exception as exc:
            logger.error("Reconnect failed: %s", exc)

    # ── Frame reading ─────────────────────────────────────────────────────────

    def _read_bytes(self, n):
        data = self._ser.read(n)
        if len(data) != n:
            raise LiDARReadError(
                f"Expected {n} bytes, got {len(data)} "
                f"(sensor stopped — check 5V power supply)"
            )
        return data

    def _sync_and_read_frame(self):
        """Scan stream for 0x59 0x59 header, read remaining 7 bytes."""
        for _ in range(FRAME_LEN * 8):
            b1 = self._read_bytes(1)
            if b1[0] != HEADER_BYTE: continue
            b2 = self._read_bytes(1)
            if b2[0] != HEADER_BYTE: continue
            return b1 + b2 + self._read_bytes(FRAME_LEN - 2)
        raise LiDARReadError("Sync header 0x59 0x59 not found in stream")

    @staticmethod
    def _checksum_ok(frame):
        return (sum(frame[:8]) & 0xFF) == frame[8]

    @staticmethod
    def _parse(frame):
        return {
            "distance_cm"  : frame[2] | (frame[3] << 8),
            "strength"     : frame[4] | (frame[5] << 8),
            "temperature_c": round((frame[6] | (frame[7] << 8)) / 100.0, 1),
        }

    def read_frame(self, _retry=0):
        """Read one valid frame. Retries checksum failures up to 3x silently."""
        if not self.connected or not self._ser.is_open:
            raise LiDARReadError("Port not open")

        frame = self._sync_and_read_frame()

        if not self._checksum_ok(frame):
            if _retry < 3:
                return self.read_frame(_retry=_retry + 1)
            raise LiDARReadError(f"Checksum fail: {frame.hex(' ')}")

        data = self._parse(frame)
        data["valid"] = (MIN_DIST_CM <= data["distance_cm"] <= MAX_DIST_CM)
        return data

    def read_frame_current(self):
        """Read latest frame — drains stale buffer if grown too large."""
        if not self.connected or not self._ser.is_open:
            raise LiDARReadError("Port not open")
        if self._ser.in_waiting > FRAME_LEN * 3:
            self._ser.reset_input_buffer()
            time.sleep(0.012)
        return self.read_frame()

    def diagnostic_raw_dump(self, n_bytes=90):
        """Read raw bytes for diagnostic purposes."""
        self._ser.reset_input_buffer()
        time.sleep(0.1)
        return self._ser.read(n_bytes)

    def close(self):
        if self._ser and self._ser.is_open:
            self._ser.close()
            self.connected = False
            logger.info("Port closed.")

    def __enter__(self): return self
    def __exit__(self, *_): self.close()


# ── Background reader thread ──────────────────────────────────────────────────

class LiDARReaderThread:
    """
    Reads sensor at ~100Hz in background. Auto-recovers from dropouts.
    Soft recovery (re-enable) after 3 errors.
    Hard recovery (port reconnect) after 6 errors.
    """

    SOFT_AFTER = 3
    HARD_AFTER = 6

    def __init__(self, lidar: TF02Pro, maxlen=5):
        self._lidar   = lidar
        self._buf     = deque(maxlen=maxlen)
        self._lock    = threading.Lock()
        self._running = True
        self.errors   = 0
        self.frames   = 0
        self._consec  = 0
        self._thread  = threading.Thread(
            target=self._loop, daemon=True, name="LiDARReader"
        )
        self._thread.start()

    def _loop(self):
        while self._running:
            try:
                frame = self._lidar.read_frame()
                with self._lock:
                    self._buf.append(frame)
                self.frames += 1
                self._consec = 0
            except LiDARReadError as exc:
                self.errors  += 1
                self._consec += 1
                logger.debug("Error #%d: %s", self._consec, exc)

                if self._consec == self.SOFT_AFTER:
                    logger.warning("Soft recovery …")
                    self._lidar._enable_output()
                elif self._consec >= self.HARD_AFTER:
                    logger.warning("Hard reconnect …")
                    self._lidar.reconnect()
                    self._consec = 0

                time.sleep(0.005)
            except Exception as exc:
                logger.error("Reader: %s", exc)
                time.sleep(0.05)

    def get_latest(self):
        with self._lock:
            return self._buf[-1] if self._buf else None

    def stop(self):
        self._running = False
        self._thread.join(timeout=1.0)


def list_ports():
    return [p.device for p in serial.tools.list_ports.comports()]