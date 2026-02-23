"""
test_lidar.py
=============
Standalone LiDAR hardware test — NO Streamlit, NO ML.

Run this FIRST to verify the sensor is working correctly.
Press Ctrl+C to stop.

Usage:
    python test_lidar.py
    python test_lidar.py /dev/ttyUSB0 115200
"""

import sys
import time
from lidar_driver import TF02Pro, LiDARReadError

PORT  = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB0"
BAUD  = int(sys.argv[2]) if len(sys.argv) > 2 else 115200

print(f"\n{'='*55}")
print(f"  TF02-Pro LiDAR Hardware Test")
print(f"  Port: {PORT}  |  Baud: {BAUD}")
print(f"{'='*55}")
print("  Move an object closer/farther — distance should change")
print("  Press Ctrl+C to stop\n")

try:
    lidar = TF02Pro(port=PORT, baudrate=BAUD, send_init=True)
    print(f"✅ Connected to {PORT}\n")
except Exception as e:
    print(f"❌ Cannot open {PORT}: {e}")
    sys.exit(1)

ok = 0
err = 0
prev_dist = None
t0 = time.monotonic()

try:
    print(f"{'#':>5}  {'Dist(cm)':>9}  {'Change':>8}  {'Strength':>9}  {'Temp°C':>7}  Status")
    print("-" * 60)
    while True:
        try:
            r    = lidar.read_frame_current()
            dist = r["distance_cm"]
            ok  += 1

            change = ""
            if prev_dist is not None:
                delta = dist - prev_dist
                if abs(delta) >= 1:
                    change = f"{'↑' if delta > 0 else '↓'}{abs(delta):+.0f}"
            prev_dist = dist

            elapsed = time.monotonic() - t0
            hz      = ok / max(elapsed, 0.001)

            print(
                f"{ok:>5}  {dist:>9}  {change:>8}  "
                f"{r['strength']:>9}  {r['temperature_c']:>7.1f}  "
                f"[{hz:.1f} Hz]",
                flush=True,
            )

        except LiDARReadError as e:
            err += 1
            print(f"{'ERR':>5}  {'—':>9}  {'—':>8}  {'—':>9}  {'—':>7}  {e}",
                  flush=True)

        time.sleep(0.05)   # print at ~20Hz (sensor runs at 100Hz internally)

except KeyboardInterrupt:
    elapsed = time.monotonic() - t0
    print(f"\n{'='*55}")
    print(f"  Done.  Frames OK: {ok}  Errors: {err}")
    print(f"  Elapsed: {elapsed:.1f}s  Avg rate: {ok/max(elapsed,1):.1f} Hz")
    print(f"{'='*55}\n")
    lidar.close()
