"""
lidar_driver.py
================
TF02-Pro single-point LiDAR driver.

Includes LiDARReaderThread — a background thread that reads sensor frames
continuously at 100 Hz so the main (UI) thread never blocks on serial I/O.

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

# ── Startup commands (host → sensor)  CS = LSB(sum of all preceding bytes) ───
CMD_SOFT_RESET    = bytes([0x5A, 0x04, 0x02, 0x60])
CMD_ENABLE_OUTPUT = bytes([0x5A, 0x05, 0x07, 0x01, 0x67])
CMD_SET_100HZ     = bytes([0x5A, 0x06, 0x03, 0x64, 0x00, 0xC7])

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
    send_init  : send soft-reset + enable-output commands at startup
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
        try:
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

        except serial.SerialException as exc:
            logger.error("Cannot open %s: %s", self.port, exc)
            raise

    def _send_startup_commands(self) -> None:
        logger.info("Sending init commands …")
        try:
            self._ser.write(CMD_SOFT_RESET);   self._ser.flush()
            logger.info("  → Soft reset sent")
            time.sleep(1.0)
            self._ser.reset_input_buffer()

            self._ser.write(CMD_ENABLE_OUTPUT); self._ser.flush()
            logger.info("  → Enable output sent")
            time.sleep(0.1)

            self._ser.write(CMD_SET_100HZ);    self._ser.flush()
            logger.info("  → Set 100Hz sent")
            time.sleep(0.1)

            logger.info("Init complete. Sensor streaming …")
        except serial.SerialException as exc:
            logger.warning("Init command error: %s", exc)

    def _read_bytes(self, n: int) -> bytes:
        data = self._ser.read(n)
        if len(data) != n:
            raise LiDARReadError(
                f"Expected {n} bytes, got {len(data)} "
                f"(timeout — sensor TX disconnected or in trigger mode?)"
            )
        return data

    def _sync_and_read_frame(self) -> bytes:
        """Scan stream for 0x59 0x59 header, then read remaining 7 bytes."""
        for _ in range(FRAME_LEN * 6):
            b1 = self._read_bytes(1)
            if b1[0] != HEADER_BYTE:
                continue
            b2 = self._read_bytes(1)
            if b2[0] != HEADER_BYTE:
                continue
            return b1 + b2 + self._read_bytes(FRAME_LEN - 2)
        raise LiDARReadError("Header 0x59 0x59 not found in stream")

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

    def read_frame(self) -> dict:
        """
        Read one frame from the continuous stream (no buffer management).
        For real-time use, prefer LiDARReaderThread.get_latest() instead.
        """
        if not self.connected or not self._ser.is_open:
            raise LiDARReadError("Port not open")

        frame = self._sync_and_read_frame()
        if not self._checksum_ok(frame):
            raise LiDARReadError(f"Checksum fail: {frame.hex(' ')}")

        data = self._parse(frame)
        if not (MIN_DIST_CM <= data["distance_cm"] <= MAX_DIST_CM):
            raise LiDARReadError(
                f"Distance {data['distance_cm']} cm out of range "
                f"[{MIN_DIST_CM}–{MAX_DIST_CM}]"
            )
        data["valid"] = True
        return data

    def read_frame_current(self) -> dict:
        """
        Read the freshest frame, draining stale buffer if it has grown.
        Useful when calling from a single-threaded loop.
        """
        if not self.connected or not self._ser.is_open:
            raise LiDARReadError("Port not open")

        queued = self._ser.in_waiting
        if queued > FRAME_LEN * 3:   # > 3 frames queued → data is stale
            logger.debug("Draining %d stale bytes", queued)
            self._ser.reset_input_buffer()
            time.sleep(0.012)        # wait ~1.2 frame periods for fresh data

        return self.read_frame()

    def diagnostic_raw_dump(self, n_bytes: int = 90) -> bytes:
        """Read raw bytes without parsing (for wiring/baud diagnostics)."""
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

    WHY THIS EXISTS
    ───────────────
    Streamlit UI updates (metrics, charts) take 50-150 ms each. During that
    time the sensor emits 5-15 frames into the OS serial buffer. When the
    main thread finally calls read_frame() it gets the oldest buffered frame,
    not the current one — creating visible lag.

    With LiDARReaderThread:
    • Background thread reads the sensor at 100 Hz continuously.
    • It always drains the serial buffer instantly.
    • Main (UI) thread calls get_latest() — zero serial I/O, zero lag.
    • Distance updates reflect the CURRENT reading, not what it was 2s ago.

    Usage
    -----
        reader = LiDARReaderThread(lidar, maxlen=5)
        ...
        frame = reader.get_latest()   # always the freshest reading
        ...
        reader.stop()
    """

    def __init__(self, lidar: TF02Pro, maxlen: int = 5):
        self._lidar    = lidar
        self._buf      = deque(maxlen=maxlen)   # ring buffer of recent frames
        self._lock     = threading.Lock()
        self._running  = True
        self.errors    = 0
        self.frames    = 0
        self._thread   = threading.Thread(
            target=self._loop, daemon=True, name="LiDARReader"
        )
        self._thread.start()
        logger.info("LiDARReaderThread started (maxlen=%d)", maxlen)

    def _loop(self) -> None:
        """Background loop: read frame → store in ring buffer, repeat."""
        while self._running:
            try:
                frame = self._lidar.read_frame()   # blocks ~10ms at 100Hz
                with self._lock:
                    self._buf.append(frame)
                self.frames += 1
            except LiDARReadError as exc:
                self.errors += 1
                logger.debug("Reader thread error: %s", exc)
                time.sleep(0.005)   # brief pause before retry
            except Exception as exc:
                logger.error("Unexpected error in reader thread: %s", exc)
                time.sleep(0.05)

    def get_latest(self) -> dict | None:
        """
        Return the most recently received frame, or None if none yet.
        Thread-safe. Zero serial I/O — just returns from the ring buffer.
        """
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