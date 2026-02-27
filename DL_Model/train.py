"""
train.py
========
End-to-end training script for the 1D-CNN + BiLSTM pothole detector.

Usage
─────
  python train.py [--epochs N] [--batch B] [--lr LR] [--no-augment]

Steps
─────
  1. Build dataset  (real CSVs + synthetic)
  2. Compute class weights for imbalance handling
  3. Build model
  4. Train with callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)
  5. Evaluate on held-out test set
  6. Save training history, classification report, confusion matrix
  7. Export ONNX (optional – requires tf2onnx)
"""

import os, sys, json, time, argparse, warnings, io
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")
# Force UTF-8 output on Windows so special chars don't crash
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Add parent dir to path so imports work when run from DL_Model/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from dl_config import (
    MODELS_DIR, LOGS_DIR, MODEL_SAVE_PATH, HISTORY_SAVE_PATH,
    REPORT_SAVE_PATH, CONFUSION_SAVE_PATH,
    EPOCHS, BATCH_SIZE, LEARNING_RATE,
    LR_PATIENCE, ES_PATIENCE, RANDOM_SEED,
    CLASS_NAMES, N_CLASSES,
)
from data_pipeline import build_dataset
from dl_model import build_model

# ── CLI args ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train DL Pothole Detector")
    p.add_argument("--epochs",      type=int,   default=EPOCHS)
    p.add_argument("--batch",       type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",          type=float, default=LEARNING_RATE)
    p.add_argument("--no-augment",  action="store_true",
                   help="Skip real-data augmentation (faster, lower accuracy)")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _set_seed(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _compute_class_weights(y_train: np.ndarray) -> dict:
    classes = np.arange(N_CLASSES)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    cw = {int(c): float(w) for c, w in zip(classes, weights)}
    print(f"  Class weights: { {CLASS_NAMES[k]: f'{v:.2f}' for k, v in cw.items()} }")
    return cw


def _plot_history(history, save_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    colors = {"train": "#58a6ff", "val": "#f78166"}

    metrics = [("loss",     "Loss"),
               ("accuracy", "Accuracy"),
               ("top2_acc", "Top-2 Accuracy")]

    for ax, (key, title) in zip(axes, metrics):
        train_vals = history.history.get(key, [])
        val_vals   = history.history.get(f"val_{key}", [])
        epochs     = range(1, len(train_vals) + 1)

        ax.plot(epochs, train_vals, color=colors["train"], lw=2, label="Train")
        ax.plot(epochs, val_vals,   color=colors["val"],   lw=2, label="Val",
                linestyle="--")
        ax.set_title(title, color="white", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch", color="#8b949e")
        ax.legend(facecolor="#21262d", labelcolor="white")

    plt.suptitle("Pothole DL Model — Training History",
                 color="white", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = os.path.join(LOGS_DIR, "training_history.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  History plot saved → {out}")


def _plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: str):
    cm     = confusion_matrix(y_true, y_pred)
    labels = [CLASS_NAMES[i] for i in range(N_CLASSES)]

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(labels, rotation=30, ha="right", color="white", fontsize=11)
    ax.set_yticklabels(labels, color="white", fontsize=11)

    thresh = cm.max() / 2.0
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                    color="white" if cm[i, j] < thresh else "black",
                    fontsize=14, fontweight="bold")

    ax.set_xlabel("Predicted",  color="#8b949e", fontsize=12)
    ax.set_ylabel("True Label", color="#8b949e", fontsize=12)
    ax.set_title("Confusion Matrix — Test Set",
                 color="white", fontsize=14, fontweight="bold", pad=15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Confusion matrix saved → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    _set_seed(RANDOM_SEED)

    print("\n" + "="*65)
    print("  DL MODEL — Pothole / Road Anomaly Detector")
    print("  1D-CNN + BiLSTM + Multi-Head Attention")
    print("="*65)

    gpus = tf.config.list_physical_devices("GPU")
    print(f"\n  TensorFlow {tf.__version__}  |  GPUs available: {len(gpus)}")
    if gpus:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
            print(f"    {g.name}")

    # ── 1. Dataset ────────────────────────────────────────────────────────────
    print("\n[1/5] Building dataset …")
    splits = build_dataset()
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val,   y_val   = splits["X_val"],   splits["y_val"]
    X_test,  y_test  = splits["X_test"],  splits["y_test"]

    # ── 2. Class weights ──────────────────────────────────────────────────────
    print("\n[2/5] Computing class weights …")
    class_weights = _compute_class_weights(y_train)

    # ── 3. Build model ────────────────────────────────────────────────────────
    print("\n[3/5] Building model …")
    model = build_model(learning_rate=args.lr)
    model.summary(line_length=80, print_fn=print)

    # ── 4. Callbacks ──────────────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR,   exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor   = "val_accuracy",
            patience  = ES_PATIENCE,
            mode      = "max",
            restore_best_weights = True,
            verbose   = 1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = 0.4,
            patience = LR_PATIENCE,
            min_lr   = 1e-7,
            verbose  = 1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath         = MODEL_SAVE_PATH,
            monitor          = "val_accuracy",
            save_best_only   = True,
            save_weights_only= False,
            mode             = "max",
            verbose          = 1,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.CSVLogger(
            os.path.join(LOGS_DIR, "epoch_log.csv"), separator=",", append=False
        ),
    ]

    # ── 5. Train ──────────────────────────────────────────────────────────────
    print(f"\n[4/5] Training  (epochs={args.epochs}  batch={args.batch}) …")
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data = (X_val, y_val),
        epochs          = args.epochs,
        batch_size      = args.batch,
        class_weight    = class_weights,
        callbacks       = callbacks,
        shuffle         = True,
        verbose         = 1,
    )
    elapsed = time.time() - t0
    print(f"\n  Training finished in {elapsed/60:.1f} min")

    # Reload best checkpoint
    print(f"  Loading best model from {MODEL_SAVE_PATH} …")
    from dl_model import load_model
    best_model = load_model(MODEL_SAVE_PATH)

    # ── 6. Evaluate ───────────────────────────────────────────────────────────
    print("\n[5/5] Evaluating on held-out test set …")
    test_loss, test_acc, test_top2 = best_model.evaluate(
        X_test, y_test, batch_size=args.batch, verbose=0
    )
    print(f"  Test  loss     : {test_loss:.4f}")
    print(f"  Test  accuracy : {test_acc * 100:.2f}%")
    print(f"  Test  top-2    : {test_top2 * 100:.2f}%")

    y_pred_prob = best_model.predict(X_test, batch_size=args.batch, verbose=0)
    y_pred      = y_pred_prob.argmax(axis=1)

    target_names = [CLASS_NAMES[i] for i in range(N_CLASSES)]
    report = classification_report(y_test, y_pred, target_names=target_names, digits=4)
    print("\n  Classification Report:\n")
    print(report)

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"  Macro F1: {macro_f1:.4f}")

    # Save report
    with open(REPORT_SAVE_PATH, "w") as f:
        f.write(f"Test Accuracy : {test_acc*100:.2f}%\n")
        f.write(f"Test Macro F1 : {macro_f1:.4f}\n")
        f.write(f"Top-2 Accuracy: {test_top2*100:.2f}%\n\n")
        f.write(report)
    print(f"  Report saved → {REPORT_SAVE_PATH}")

    # Save history
    hist_dict = {k: [float(v) for v in vl]
                 for k, vl in history.history.items()}
    hist_dict["training_seconds"] = round(elapsed, 1)
    hist_dict["test_accuracy"]    = round(float(test_acc), 5)
    hist_dict["test_macro_f1"]    = round(float(macro_f1), 5)
    with open(HISTORY_SAVE_PATH, "w") as f:
        json.dump(hist_dict, f, indent=2)
    print(f"  History saved → {HISTORY_SAVE_PATH}")

    # Plots
    _plot_history(history, HISTORY_SAVE_PATH)
    _plot_confusion_matrix(y_test, y_pred, CONFUSION_SAVE_PATH)

    print("\n" + "="*65)
    print(f"  ✅  Model saved  → {MODEL_SAVE_PATH}")
    print(f"  ✅  Final test accuracy : {test_acc*100:.2f}%")
    print(f"  ✅  Final macro F1      : {macro_f1:.4f}")
    print("="*65)


if __name__ == "__main__":
    main()
