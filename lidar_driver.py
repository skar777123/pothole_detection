"""
lidar_driver.py
================
TF02-Pro single-point LiDAR driver.

ROOT CAUSE OF "WORKS 40 FRAMES THEN STOPS"
────────────────────────────────────────────
  The soft-reset command (0x5A 0x04 0x02 0x60) tells the sensor to reboot
  and reload its SAVED settings. If the sensor's factory-saved config has
  continuous output = OFF (trigger mode), the sensor goes silent after the
  reset finishes. We then send CMD_ENABLE_OUTPUT which temporarily enables
  streaming, but that setting is not saved, so the next internal config
  load reverts it.

  Fix:
    1. Never send soft-reset.
    2. Send CMD_ENABLE_OUTPUT + CMD_SAVE_SETTINGS so the sensor persists
       continuous-output mode across power cycles.
    3. Auto-recover: if N consecutive timeouts occur mid-session, re-send
       CMD_ENABLE_OUTPUT (sensor may have been power-glitched).
    4. Swallow isolated checksum failures and re-sync inline.

TF02-Pro 9-byte UART frame (115200 baud, 100 Hz):
  [0] 0x59  header
  [1] 0x59  header
  [2] Dist_L  ┐ Distance cm  (uint16 LE)
  [3] Dist_H  ┘
  [4] Flux_L  ┐ Signal flux  (uint16 LE)
  [5] Flux_H  ┘
  [6] Temp_L  ┐ Chip temp    (uint16 LE, unit 0.01 °C)
  [7] Temp_H  ┘
  [8] Checksum = LSB( sum bytes 0-7 )
"""

import serial
import serial.tools.list_ports
import threading
import time
import logging
from collections import deque

logger = logging.getLogger(__name__)

# ── Sensor limits ─────────────────────────────────────────────────────────────
MIN_DIST_CM = 5
MAX_DIST_CM = 2200
HEADER_BYTE = 0x59
FRAME_LEN   = 9       # bytes per frame

# ── Host → Sensor commands ────────────────────────────────────────────────────
# CS = LSB( sum of all bytes in the command except CS itself )

# ⚠️  NO SOFT-RESET here on purpose.
#     Soft-reset reloads saved settings which may be "trigger mode".

CMD_ENABLE_OUTPUT = bytes([0x5A, 0x05, 0x07, 0x01, 0x67])
# [0x5A] header | [0x05] len | [0x07] set-output-mode | [0x01] ON
# CS = (0x5A+0x05+0x07+0x01) & 0xFF = 103 = 0x67 ✓

CMD_SAVE_SETTINGS = bytes([0x5A, 0x04, 0x11, 0x6F])
# [0x5A] header | [0x04] len | [0x11] save | CS = 0x6F ✓
# After this the sensor ALWAYS boots in continuous-output mode.

CMD_SET_100HZ = bytes([0x5A, 0x06, 0x03, 0x64, 0x00, 0xC7])
# [0x5A] | [0x06] | [0x03] set-fps | [0x64 0x00] = 100 Hz | CS = 0xC7 ✓

BAUD_CANDIDATES = [115200, 9600]


class LiDARReadError(Exception):
    pass


class TF02Pro:
    """
    TF02-Pro UART driver.

    Parameters
    ----------
    port       : e.g. '/dev/ttyUSB0' or 'COM3'
    baudrate   : 115200 (factory default)
    timeout    : per-byte read timeout in seconds
    send_init  : send enable-output + save-settings commands at startup
    """

    def __init__(self, port: str = '/dev/ttyUSB0',
                 baudrate: int = 115200,
                 timeout: float = 0.5,
                 send_init: bool = True):
        self.port      = port
        self.baudrate  = baudrate
        self._ser      = None
        self.connected = False
        self._open(timeout, send_init)

    # ── Private ───────────────────────────────────────────────────────────────

    def _open(self, timeout: float, send_init: bool) -> None:
        self._ser = serial.Serial(
            self.port, self.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout,
        )
        self._ser.reset_input_buffer()
        logger.info("Opened %s @ %d baud", self.port, self.baudrate)

        if send_init:
            self._send_startup_commands()

        self._ser.reset_input_buffer()
        self.connected = True
        logger.info("TF02-Pro ready on %s", self.port)

    def _send_startup_commands(self) -> None:
        """
        Send enable-output + save-settings so sensor stays in streaming mode.
        NO soft-reset — that would reload saved (possibly trigger-mode) settings.
        """
        logger.info("Sending init commands (NO soft-reset) …")
        try:
            # 1. Enable continuous output
            self._ser.write(CMD_ENABLE_OUTPUT); self._ser.flush()
            logger.info("  → Enable output sent (5A 05 07 01 67)")
            time.sleep(0.15)

            # 2. Set 100 Hz
            self._ser.write(CMD_SET_100HZ); self._ser.flush()
            logger.info("  → Set 100Hz sent (5A 06 03 64 00 C7)")
            time.sleep(0.15)

            # 3. SAVE settings — now this survives power cycles
            self._ser.write(CMD_SAVE_SETTINGS); self._ser.flush()
            logger.info("  → Save settings sent (5A 04 11 6F)")
            time.sleep(0.15)

            self._ser.reset_input_buffer()   # flush ACK responses
            logger.info("  Init complete — continuous output is now SAVED.")
        except serial.SerialException as exc:
            logger.warning("Init command error: %s", exc)

    def _enable_output(self) -> None:
        """Re-send enable-output mid-session (auto-recover)."""
        try:
            self._ser.write(CMD_ENABLE_OUTPUT); self._ser.flush()
            logger.info("Re-sent enable-output (auto-recover).")
            time.sleep(0.15)
            self._ser.reset_input_buffer()
        except Exception:
            pass

    def _read_bytes(self, n: int) -> bytes:
        data = self._ser.read(n)
        if len(data) != n:
            raise LiDARReadError(
                f"Expected {n} bytes, got {len(data)} "
                f"(timeout — check TX wiring or run init)"
            )
        return data

    def _sync_and_read_frame(self) -> bytes:
        """
        Scan stream for 0x59 0x59 header then read next 7 bytes.
        Searches up to FRAME_LEN*6 bytes before giving up.
        """
        for _ in range(FRAME_LEN * 6):
            b1 = self._read_bytes(1)
            if b1[0] != HEADER_BYTE:
                continue
            b2 = self._read_bytes(1)
            if b2[0] != HEADER_BYTE:
                continue
            return b1 + b2 + self._read_bytes(FRAME_LEN - 2)
        raise LiDARReadError("Header 0x59 0x59 not found within search window")

    @staticmethod
    def _checksum_ok(frame: bytes) -> bool:
        return (sum(frame[:8]) & 0xFF) == frame[8]

    @staticmethod
    def _parse(frame: bytes) -> dict:
        return {
            "distance_cm"  : frame[2] | (frame[3] << 8),
            "strength"     : frame[4] | (frame[5] << 8),
            "temperature_c": round((frame[6] | (frame[7] << 8)) / 100.0, 1),
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def read_frame(self, _retry: int = 0) -> dict:
        """
        Read one valid frame from the continuous stream.

        Bad checksum frames (from 1-byte misalignment) are automatically
        retried up to 3 times before raising LiDARReadError.
        """
        if not self.connected or not self._ser.is_open:
            raise LiDARReadError("Port not open")

        frame = self._sync_and_read_frame()

        if not self._checksum_ok(frame):
            if _retry < 3:
                # Misalignment by 1 byte is common — just re-sync silently
                logger.debug("Checksum fail (retry %d): %s", _retry, frame.hex(' '))
                return self.read_frame(_retry=_retry + 1)
            raise LiDARReadError(f"Checksum fail after 3 retries: {frame.hex(' ')}")

        data = self._parse(frame)
        if not (MIN_DIST_CM <= data["distance_cm"] <= MAX_DIST_CM):
            raise LiDARReadError(
                f"Distance {data['distance_cm']} cm outside valid range "
                f"[{MIN_DIST_CM}–{MAX_DIST_CM}]"
            )
        data["valid"] = True
        return data

    def read_frame_current(self) -> dict:
        """
        Read the freshest frame — drains stale OS buffer if grown too large.
        Use this from a slow loop (e.g. every 50ms) to avoid reading stale data.
        """
        if not self.connected or not self._ser.is_open:
            raise LiDARReadError("Port not open")

        queued = self._ser.in_waiting
        if queued > FRAME_LEN * 3:
            logger.debug("Draining %d stale bytes (%d frames)", queued, queued // FRAME_LEN)
            self._ser.reset_input_buffer()
            time.sleep(0.012)   # ~1.2× frame period at 100Hz

        return self.read_frame()

    def diagnostic_raw_dump(self, n_bytes: int = 90) -> bytes:
        """Read raw bytes without parsing (for wiring/baud-rate diagnostics)."""
        self._ser.reset_input_buffer()
        time.sleep(0.1)
        return self._ser.read(n_bytes)

    def close(self) -> None:
        if self._ser and self._ser.is_open:
            self._ser.close()
            self.connected = False
            logger.info("TF02-Pro port closed.")

    def __enter__(self):  return self
    def __exit__(self, *_): self.close()


# ── Background reader thread ──────────────────────────────────────────────────

class LiDARReaderThread:
    """
    Reads TF02-Pro frames continuously in a background daemon thread.

    The thread auto-recovers from consecutive timeouts by re-sending the
    enable-output command (handles sensor power glitches mid-session).

    Usage
    -----
        reader = LiDARReaderThread(lidar)
        frame  = reader.get_latest()    # always the freshest reading
        reader.stop()
    """

    AUTO_RECOVER_AFTER = 10   # re-send enable-output after this many errors

    def __init__(self, lidar: TF02Pro, maxlen: int = 5):
        self._lidar   = lidar
        self._buf     = deque(maxlen=maxlen)
        self._lock    = threading.Lock()
        self._running = True
        self.errors   = 0
        self.frames   = 0
        self._consec  = 0     # consecutive error counter for auto-recover
        self._thread  = threading.Thread(
            target=self._loop, daemon=True, name="LiDARReader"
        )
        self._thread.start()
        logger.info("LiDARReaderThread started (maxlen=%d)", maxlen)

    def _loop(self) -> None:
        while self._running:
            try:
                frame = self._lidar.read_frame()
                with self._lock:
                    self._buf.append(frame)
                self.frames  += 1
                self._consec  = 0   # reset on success

            except LiDARReadError as exc:
                self.errors  += 1
                self._consec += 1
                logger.debug("Reader error #%d: %s", self._consec, exc)

                if self._consec >= self.AUTO_RECOVER_AFTER:
                    logger.warning(
                        "%d consecutive errors — re-sending enable-output …",
                        self._consec,
                    )
                    self._lidar._enable_output()
                    self._consec = 0

                time.sleep(0.005)

            except Exception as exc:
                logger.error("Unexpected reader thread error: %s", exc)
                time.sleep(0.05)

    def get_latest(self) -> dict | None:
        """Return most-recent frame from ring buffer. Thread-safe, zero serial I/O."""
        with self._lock:
            return self._buf[-1] if self._buf else None

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        logger.info("LiDARReaderThread stopped. frames=%d errors=%d",
                    self.frames, self.errors)


# ── Utility ───────────────────────────────────────────────────────────────────

def list_ports() -> list[str]:
    return [p.device for p in serial.tools.list_ports.comports()]