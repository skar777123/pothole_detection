"""
lidar_driver.py — TF02-Pro UART driver with auto-reconnect recovery.
"""

import serial
import serial.tools.list_ports
import threading
import time
import logging
from collections import deque

logger = logging.getLogger(__name__)

MIN_DIST_CM  = 1      # allow readings down to 1 cm (was 5 — caused ERR at close range)
MAX_DIST_CM  = 2200
HEADER_BYTE  = 0x59
FRAME_LEN    = 9

CMD_ENABLE_OUTPUT = bytes([0x5A, 0x05, 0x07, 0x01, 0x67])
CMD_SAVE_SETTINGS = bytes([0x5A, 0x04, 0x11, 0x6F])
CMD_SET_100HZ     = bytes([0x5A, 0x06, 0x03, 0x64, 0x00, 0xC7])


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
        self._connect(send_init)

    def _connect(self, send_init=True):
        """Open (or re-open) the serial port and optionally send init commands."""
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
        logger.info("Opened %s @ %d baud", self.port, self.baudrate)

        if send_init:
            self._send_startup_commands()

        self._ser.reset_input_buffer()
        self.connected = True
        logger.info("TF02-Pro ready.")

    def _send_startup_commands(self):
        try:
            self._ser.write(CMD_ENABLE_OUTPUT); self._ser.flush()
            logger.info("  → Enable output sent")
            time.sleep(0.15)
            self._ser.write(CMD_SET_100HZ); self._ser.flush()
            logger.info("  → Set 100Hz sent")
            time.sleep(0.15)
            self._ser.write(CMD_SAVE_SETTINGS); self._ser.flush()
            logger.info("  → Save settings sent")
            time.sleep(0.15)
            self._ser.reset_input_buffer()
        except serial.SerialException as exc:
            logger.warning("Init error: %s", exc)

    def _enable_output(self):
        """Soft recovery: re-send enable command."""
        try:
            self._ser.write(CMD_ENABLE_OUTPUT); self._ser.flush()
            time.sleep(0.20)
            self._ser.reset_input_buffer()
            logger.info("Soft recovery: enable-output re-sent.")
        except Exception:
            pass

    def reconnect(self):
        """Hard recovery: close and reopen the serial port completely."""
        logger.warning("Hard reconnect: closing and reopening %s …", self.port)
        try:
            self._connect(send_init=True)
            logger.info("Reconnect successful.")
        except Exception as exc:
            logger.error("Reconnect failed: %s", exc)

    def _read_bytes(self, n):
        data = self._ser.read(n)
        if len(data) != n:
            raise LiDARReadError(
                f"Expected {n} bytes, got {len(data)} "
                f"(sensor stopped — likely power glitch on VCC)"
            )
        return data

    def _sync_and_read_frame(self):
        for _ in range(FRAME_LEN * 6):
            b1 = self._read_bytes(1)
            if b1[0] != HEADER_BYTE:
                continue
            b2 = self._read_bytes(1)
            if b2[0] != HEADER_BYTE:
                continue
            return b1 + b2 + self._read_bytes(FRAME_LEN - 2)
        raise LiDARReadError("Header 0x59 0x59 not found")

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
        if not self.connected or not self._ser.is_open:
            raise LiDARReadError("Port not open")

        frame = self._sync_and_read_frame()

        if not self._checksum_ok(frame):
            if _retry < 3:
                return self.read_frame(_retry=_retry + 1)
            raise LiDARReadError(f"Checksum fail: {frame.hex(' ')}")

        data = self._parse(frame)
        data["valid"] = (MIN_DIST_CM <= data["distance_cm"] <= MAX_DIST_CM)
        # NOTE: no longer raising error for out-of-range — just sets valid=False
        # so the caller can still see the reading and the stream keeps going
        return data

    def read_frame_current(self):
        if not self.connected or not self._ser.is_open:
            raise LiDARReadError("Port not open")
        queued = self._ser.in_waiting
        if queued > FRAME_LEN * 3:
            self._ser.reset_input_buffer()
            time.sleep(0.012)
        return self.read_frame()

    def diagnostic_raw_dump(self, n_bytes=90):
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


class LiDARReaderThread:
    """
    Background thread that reads sensor at 100Hz continuously.
    Auto-recovers from dropouts: soft recovery first, then hard reconnect.
    """

    SOFT_RECOVER_AFTER = 3    # soft recovery after 3 consecutive errors
    HARD_RECOVER_AFTER = 9    # hard reconnect after 9 consecutive errors (3 soft fails)

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
        logger.info("LiDARReaderThread started.")

    def _loop(self):
        while self._running:
            try:
                frame = self._lidar.read_frame()
                with self._lock:
                    self._buf.append(frame)
                self.frames  += 1
                self._consec  = 0

            except LiDARReadError as exc:
                self.errors  += 1
                self._consec += 1
                logger.debug("Error #%d: %s", self._consec, exc)

                if self._consec == self.SOFT_RECOVER_AFTER:
                    logger.warning("Soft recovery (re-enable output) …")
                    self._lidar._enable_output()

                elif self._consec >= self.HARD_RECOVER_AFTER:
                    logger.warning("Hard recovery (port reconnect) …")
                    self._lidar.reconnect()
                    self._consec = 0

                time.sleep(0.005)

            except Exception as exc:
                logger.error("Unexpected: %s", exc)
                time.sleep(0.05)

    def get_latest(self):
        with self._lock:
            return self._buf[-1] if self._buf else None

    def stop(self):
        self._running = False
        self._thread.join(timeout=1.0)


def list_ports():
    return [p.device for p in serial.tools.list_ports.comports()]