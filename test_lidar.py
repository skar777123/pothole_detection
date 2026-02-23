"""
test_lidar.py
=============
Standalone LiDAR hardware test — NO Streamlit, NO ML.
Prints live distance to terminal. Ctrl+C to stop.

Usage:
    python test_lidar.py
    python test_lidar.py /dev/ttyUSB0 115200
    python test_lidar.py /dev/ttyUSB0 115200 debug
"""

import sys
import time
import logging
from lidar_driver import TF02Pro, LiDARReadError

PORT  = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB0"
BAUD  = int(sys.argv[2])            if len(sys.argv) > 2 else 115200
DEBUG = sys.argv[3].lower() == "debug" if len(sys.argv) > 3 else False

# Enable driver debug logging in debug mode
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.WARNING,
    format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
)

print(f"\n{'='*60}")
print(f"  TF02-Pro Hardware Test")
print(f"  Port: {PORT}  |  Baud: {BAUD}  |  Debug: {DEBUG}")
print(f"{'='*60}")
print("  Move object closer/farther — watch distance change")
print("  Press Ctrl+C to stop\n")

try:
    lidar = TF02Pro(port=PORT, baudrate=BAUD, send_init=True)
    print(f"✅ Connected. Settings saved to sensor (persistent).\n")
except Exception as e:
    print(f"❌ Cannot open {PORT}: {e}")
    sys.exit(1)

ok = 0
err = 0
consecutive_err = 0
prev_dist = None
t0 = time.monotonic()

RECOVER_AFTER = 10   # re-send enable-output after this many consecutive errors

try:
    print(f"{'#':>5}  {'Dist(cm)':>9}  {'Δ':>6}  {'Strength':>9}  {'Temp°C':>7}  {'Hz':>6}")
    print("-" * 60)

    while True:
        try:
            # Use read_frame() directly — checksum retries happen internally
            r    = lidar.read_frame()
            dist = r["distance_cm"]
            ok  += 1
            consecutive_err = 0

            # Compute change vs previous reading
            delta  = ""
            symbol = ""
            if prev_dist is not None:
                d = dist - prev_dist
                if abs(d) >= 1:
                    symbol = "↑" if d > 0 else "↓"
                    delta  = f"{symbol}{abs(d)}"
            prev_dist = dist

            elapsed = time.monotonic() - t0
            hz      = ok / max(elapsed, 0.001)

            print(
                f"{ok:>5}  {dist:>9}  {delta:>6}  "
                f"{r['strength']:>9}  {r['temperature_c']:>7.1f}  {hz:>5.1f}",
                flush=True,
            )

        except LiDARReadError as exc:
            err += 1
            consecutive_err += 1
            print(f"{'ERR':>5}  {str(exc)[:55]}", flush=True)

            if consecutive_err >= RECOVER_AFTER:
                print(
                    f"\n⚠️  {consecutive_err} consecutive errors — "
                    f"re-sending enable-output to sensor …\n",
                    flush=True,
                )
                lidar._enable_output()
                consecutive_err = 0

        # Small sleep so we don't spam terminal — sensor still runs at 100Hz
        # read_frame_current() is NOT used here because we call read_frame()
        # directly and the internal retry handles misalignment cleanly.
        time.sleep(0.05)   # print at ~20 Hz; sensor runs at 100Hz internally

except KeyboardInterrupt:
    elapsed = time.monotonic() - t0
    print(f"\n{'='*60}")
    print(f"  Stopped.")
    print(f"  Frames OK : {ok}")
    print(f"  Errors    : {err}  ({err/(ok+err)*100:.1f}% error rate)")
    print(f"  Duration  : {elapsed:.1f}s")
    print(f"  Avg rate  : {ok/max(elapsed,1):.1f} Hz")
    print(f"{'='*60}\n")
    lidar.close()
