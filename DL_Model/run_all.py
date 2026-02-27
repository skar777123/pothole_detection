"""
run_all.py
==========
Master launcher for the DL Pothole Detection system.

Steps
─────
  1. Install requirements
  2. Build dataset + train model       (python train.py)
  3. Evaluate model                    (python evaluate.py)
  4. Run real-time demo                (python adaptive_detector.py --demo)
  5. Launch Streamlit dashboard        (streamlit run dashboard_dl.py)

Usage
─────
  # Full pipeline (train → evaluate → demo):
  python run_all.py

  # Skip training (model already exists):
  python run_all.py --skip-train

  # Just launch the dashboard:
  python run_all.py --dashboard-only

  # Train with adaptive 6-feature model:
  python run_all.py --adaptive
"""

import os, sys, argparse, subprocess, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))


def _run(cmd: list[str], desc: str):
    print(f"\n{'='*65}")
    print(f"  ▶  {desc}")
    print(f"{'='*65}\n")
    result = subprocess.run(cmd, cwd=HERE)
    if result.returncode != 0:
        print(f"\n  ✗  [{desc}] exited with code {result.returncode}")
        sys.exit(result.returncode)
    print(f"\n  ✓  [{desc}] done\n")


def main():
    p = argparse.ArgumentParser(description="DL Pothole Detection — Master Launcher")
    p.add_argument("--skip-train",    action="store_true",
                   help="Skip training (use existing model)")
    p.add_argument("--skip-eval",     action="store_true",
                   help="Skip evaluation step")
    p.add_argument("--demo-only",     action="store_true",
                   help="Run only the adaptive detector demo")
    p.add_argument("--dashboard-only",action="store_true",
                   help="Run only the Streamlit dashboard")
    p.add_argument("--adaptive",      action="store_true",
                   help="Train with 6-feature adaptive dataset")
    p.add_argument("--epochs",        type=int, default=None,
                   help="Override training epochs")
    p.add_argument("--port",          default=None,
                   help="Real LiDAR serial port (e.g. /dev/ttyUSB0)")
    args = p.parse_args()

    py = sys.executable

    # ── Dashboard only ────────────────────────────────────────────────────────
    if args.dashboard_only:
        print("\n  Launching Streamlit dashboard …")
        os.execvp("streamlit", ["streamlit", "run",
                                os.path.join(HERE, "dashboard_dl.py")])

    # ── Demo only ─────────────────────────────────────────────────────────────
    if args.demo_only:
        cmd = [py, os.path.join(HERE, "adaptive_detector.py"), "--demo"]
        if args.port:
            cmd += ["--port", args.port]
        _run(cmd, "Adaptive Detector Demo")
        return

    # ── 1. Train ──────────────────────────────────────────────────────────────
    from dl_config import MODEL_SAVE_PATH
    if not args.skip_train:
        train_cmd = [py, os.path.join(HERE, "train.py")]
        if args.adaptive:
            train_cmd.append("--adaptive")
        if args.epochs:
            train_cmd += ["--epochs", str(args.epochs)]
        _run(train_cmd, "Training DL Model")
    else:
        if not os.path.exists(MODEL_SAVE_PATH):
            print(f"  ✗  --skip-train set but no model found at {MODEL_SAVE_PATH}")
            print("     Run without --skip-train to train first.")
            sys.exit(1)
        print(f"\n  ⏩  Skipping training (model found at {MODEL_SAVE_PATH})")

    # ── 2. Evaluate ───────────────────────────────────────────────────────────
    if not args.skip_eval:
        _run([py, os.path.join(HERE, "evaluate.py")], "Model Evaluation")

    # ── 3. Real-time demo ─────────────────────────────────────────────────────
    demo_cmd = [py, os.path.join(HERE, "adaptive_detector.py"), "--demo",
                "--use-model"]
    if args.port:
        demo_cmd = [py, os.path.join(HERE, "adaptive_detector.py"),
                    "--port", args.port, "--use-model"]
    _run(demo_cmd, "Real-Time Adaptive Demo")

    # ── 4. Dashboard ──────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  🚀  All steps complete!")
    print("  To launch the dashboard run:")
    print(f"     streamlit run {os.path.join(HERE, 'dashboard_dl.py')}")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
