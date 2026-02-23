"""
test_lidar.py — TF02-Pro hardware test. Clean version.
Usage: python test_lidar.py [port] [baud]
"""
import sys, time, logging
from lidar_driver import TF02Pro, LiDARReadError

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB0"
BAUD = int(sys.argv[2]) if len(sys.argv) > 2 else 115200

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")

print(f"\n{'='*55}")
print(f"  TF02-Pro  |  {PORT} @ {BAUD}")
print(f"  Smart init: reads first, only enables if silent")
print(f"{'='*55}\n")

lidar = TF02Pro(port=PORT, baudrate=BAUD, send_init=True)
print("✅ Ready. Move object — watch distance change.\n")

ok = 0; err = 0; consec = 0; prev = None; t0 = time.monotonic()

try:
    print(f"{'#':>5}  {'Dist(cm)':>9}  {'Δ':>6}  {'Strength':>9}  Temp   Hz")
    print("-" * 58)
    while True:
        try:
            r = lidar.read_frame()
            d = r["distance_cm"]

            if not r.get("valid", True):
                print(f"  ---  out of range: {d} cm", flush=True)
                time.sleep(0.05); continue

            ok += 1; consec = 0
            delta = ""
            if prev is not None and abs(d - prev) >= 1:
                delta = f"{'↑' if d > prev else '↓'}{abs(d-prev)}"
            prev = d
            hz = ok / max(time.monotonic() - t0, 0.001)
            print(f"{ok:>5}  {d:>9}  {delta:>6}  {r['strength']:>9}"
                  f"  {r['temperature_c']:.1f}  {hz:.1f}", flush=True)

        except LiDARReadError as exc:
            err += 1; consec += 1
            print(f"  ERR[{consec}] {str(exc)[:50]}", flush=True)

            if consec == 3:
                print("  → Soft: re-enabling output …", flush=True)
                lidar._enable_output()
            elif consec >= 6:
                print("  → Hard: reconnecting port …", flush=True)
                lidar.reconnect()
                consec = 0

        time.sleep(0.05)

except KeyboardInterrupt:
    e = time.monotonic() - t0
    print(f"\n  OK:{ok}  ERR:{err}  Err%:{err/(ok+err+1)*100:.1f}  "
          f"Rate:{ok/max(e,1):.1f}Hz  Time:{e:.0f}s")
    lidar.close()
