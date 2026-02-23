"""
test_lidar.py — TF02-Pro terminal test with auto-reconnect.
Usage: python test_lidar.py [port] [baud]
"""
import sys, time, logging
from lidar_driver import TF02Pro, LiDARReadError

PORT  = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB0"
BAUD  = int(sys.argv[2]) if len(sys.argv) > 2 else 115200

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

print(f"\n{'='*60}")
print(f"  TF02-Pro Test  |  Port: {PORT}  |  Baud: {BAUD}")
print(f"  Factory-reset → enable-output → 100Hz (no save)")
print(f"{'='*60}")
print("  Move object — distance should change immediately")
print("  Ctrl+C to stop\n")

lidar = TF02Pro(port=PORT, baudrate=BAUD, send_init=True)
print("✅ Ready.\n")

ok = 0; err = 0; consec = 0; prev = None; t0 = time.monotonic()

try:
    print(f"{'#':>5}  {'Dist':>6}  {'Δ':>6}  {'Strength':>9}  {'Temp':>6}  Hz")
    print("-" * 55)
    while True:
        try:
            r = lidar.read_frame()
            d = r["distance_cm"]; ok += 1; consec = 0
            delta = ""
            if prev is not None and abs(d - prev) >= 1:
                delta = f"{'↑' if d > prev else '↓'}{abs(d-prev)}"
            prev = d
            hz = ok / max(time.monotonic() - t0, 0.001)
            print(f"{ok:>5}  {d:>6}  {delta:>6}  {r['strength']:>9}  "
                  f"{r['temperature_c']:>6.1f}  {hz:.1f}", flush=True)
        except LiDARReadError as exc:
            err += 1; consec += 1
            print(f"  ERR [{consec}]  {str(exc)[:55]}", flush=True)
            if consec == 3:
                print("  → Soft recovery: re-sending enable-output …", flush=True)
                lidar._enable_output()
            elif consec >= 6:
                print("  → Hard recovery: reconnecting port …", flush=True)
                lidar.reconnect()
                consec = 0
        time.sleep(0.05)
except KeyboardInterrupt:
    e = time.monotonic() - t0
    print(f"\n  Frames: {ok}  Errors: {err}  Rate: {ok/max(e,1):.1f}Hz")
    lidar.close()
