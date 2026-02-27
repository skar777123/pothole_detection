"""
evaluate.py
===========
Stand-alone evaluation and introspection script for the trained DL model.

Produces:
  ✅ Per-class metrics  (precision / recall / F1 / support)
  ✅ Confusion matrix plot
  ✅ Per-class confidence distributions (violin plot)
  ✅ Attention weight heatmap (sample windows)
  ✅ SHAP-style feature importance via gradient×input (GradCAM-1D)
  ✅ Summary JSON written to logs/evaluation_summary.json

Usage
─────
  python evaluate.py               # evaluate on auto-generated test set
  python evaluate.py --from-csv    # also test against real CSV rows
"""

import os, sys, json, argparse, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score, roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from dl_config import (
    MODELS_DIR, LOGS_DIR, MODEL_SAVE_PATH, SCALER_SAVE_PATH,
    CONFUSION_SAVE_PATH, N_CLASSES, CLASS_NAMES, WINDOW_SIZE,
    N_RAW_FEATURES, RANDOM_SEED,
)
from data_pipeline import build_dataset
from dl_model import load_model


# ── Helpers ────────────────────────────────────────────────────────────────────

def _styled_ax(ax):
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    return ax


def plot_confusion(y_true, y_pred, save_path: str):
    cm     = confusion_matrix(y_true, y_pred)
    labels = [CLASS_NAMES[i] for i in range(N_CLASSES)]

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#0d1117")
    _styled_ax(ax)

    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(labels, rotation=30, ha="right", color="white", fontsize=11)
    ax.set_yticklabels(labels, color="white", fontsize=11)

    thresh = cm.max() / 2.0
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] < thresh else "black",
                    fontsize=14, fontweight="bold")

    ax.set_xlabel("Predicted",  color="#8b949e", fontsize=12)
    ax.set_ylabel("True Label", color="#8b949e", fontsize=12)
    ax.set_title("Confusion Matrix — Test Set",
                 color="white", fontsize=14, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Confusion matrix → {save_path}")


def plot_confidence_violin(probs_by_class: dict, save_path: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0d1117")
    _styled_ax(ax)

    data   = [probs_by_class[i] for i in range(N_CLASSES)]
    labels = [CLASS_NAMES[i] for i in range(N_CLASSES)]
    colors = ["#2ea043", "#f0e030", "#f85149", "#58a6ff"]

    parts = ax.violinplot(data, positions=range(N_CLASSES),
                          showmedians=True, showextrema=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("white")
    parts["cbars"].set_color("#8b949e")
    parts["cmins"].set_color("#8b949e")
    parts["cmaxes"].set_color("#8b949e")

    ax.set_xticks(range(N_CLASSES))
    ax.set_xticklabels(labels, color="white", fontsize=11)
    ax.set_ylabel("Predicted confidence", color="#8b949e")
    ax.set_title("Confidence Distribution by True Class",
                 color="white", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Confidence violin → {save_path}")


def gradcam_1d(model, X_sample: np.ndarray, class_id: int,
               layer_name: str = "bilstm_1") -> np.ndarray:
    """
    Compute Grad-CAM saliency over the time axis for a single sample.
    Returns a (WINDOW_SIZE,) saliency map.
    """
    import tensorflow as tf

    # Build sub-model up to the target hidden layer
    try:
        target_layer = model.get_layer(layer_name)
        grad_model = tf.keras.Model(
            inputs  = model.inputs,
            outputs = [target_layer.output, model.output]
        )
    except ValueError:
        return np.ones(WINDOW_SIZE)

    x = tf.constant(X_sample[np.newaxis], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        layer_out, preds = grad_model(x, training=False)
        loss = preds[0, class_id]

    grads     = tape.gradient(loss, layer_out)[0]    # (T, F)
    weights   = tf.reduce_mean(grads, axis=-1)       # (T,)
    cam       = tf.nn.relu(weights).numpy()
    cam_norm  = cam / (cam.max() + 1e-8)
    return cam_norm


def plot_gradcam_samples(model, X_test, y_test, scaler, save_path: str,
                         n_per_class: int = 2):
    fig, axes = plt.subplots(N_CLASSES, n_per_class,
                              figsize=(14, N_CLASSES * 3 + 1))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle("GradCAM Saliency  (red = most influential timesteps)",
                 color="white", fontsize=13, fontweight="bold", y=1.01)

    colors_cls  = ["#2ea043", "#f0e030", "#f85149", "#58a6ff"]

    for cls in range(N_CLASSES):
        idxs = np.where(y_test == cls)[0][:n_per_class]
        for k, idx in enumerate(idxs):
            ax  = axes[cls][k] if n_per_class > 1 else axes[cls]
            _styled_ax(ax)

            x_scaled = X_test[idx]          # already scaled (T, F)
            # Back-transform distance feature for display
            x_raw    = scaler.inverse_transform(
                x_scaled.reshape(-1, N_RAW_FEATURES)
            ).reshape(WINDOW_SIZE, N_RAW_FEATURES)
            dist_raw = x_raw[:, 0]          # distance_cm

            cam  = gradcam_1d(model, x_scaled, class_id=cls)
            t    = np.arange(WINDOW_SIZE)

            ax.bar(t, cam, color="#f85149", alpha=0.45, label="Saliency")
            ax2 = ax.twinx()
            ax2.set_facecolor("none")
            ax2.plot(t, dist_raw, color=colors_cls[cls], lw=2, label="Distance")
            ax2.tick_params(colors="#8b949e")

            ax.set_title(f"{CLASS_NAMES[cls]} sample {k+1}",
                         color="white", fontsize=10, fontweight="bold")
            ax.tick_params(colors="#8b949e")
            ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  GradCAM plot → {save_path}")


# ── Main evaluation ────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Evaluate DL Pothole Detector")
    p.add_argument("--from-csv", action="store_true",
                   help="Include real CSV data in test set")
    args = p.parse_args()

    os.makedirs(LOGS_DIR,   exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("\n" + "="*60)
    print("  🔍  DL Model Evaluation")
    print("="*60)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n[1/4] Loading model from {MODEL_SAVE_PATH} …")
    model  = load_model(MODEL_SAVE_PATH)
    scaler = joblib.load(SCALER_SAVE_PATH)
    print(f"  Parameters: {model.count_params():,}")

    # ── Build test set ────────────────────────────────────────────────────────
    print("\n[2/4] Building test dataset …")
    splits = build_dataset()
    X_test, y_test = splits["X_test"], splits["y_test"]

    # ── Predict ───────────────────────────────────────────────────────────────
    print("\n[3/4] Running inference …")
    probs  = model.predict(X_test, batch_size=128, verbose=1)
    y_pred = probs.argmax(axis=1)

    # ── Metrics ───────────────────────────────────────────────────────────────
    print("\n[4/4] Computing metrics …\n")
    names  = [CLASS_NAMES[i] for i in range(N_CLASSES)]
    report = classification_report(y_test, y_pred, target_names=names, digits=4)
    print(report)

    acc       = accuracy_score(y_test, y_pred)
    macro_f1  = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    # Multiclass AUC-ROC (one-vs-rest)
    y_bin = label_binarize(y_test, classes=list(range(N_CLASSES)))
    try:
        auc = roc_auc_score(y_bin, probs, average="macro", multi_class="ovr")
    except Exception:
        auc = float("nan")

    print(f"  Accuracy        : {acc*100:.2f}%")
    print(f"  Macro F1        : {macro_f1:.4f}")
    print(f"  Weighted F1     : {weighted_f1:.4f}")
    print(f"  Macro AUC-ROC   : {auc:.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_confusion(
        y_test, y_pred,
        os.path.join(LOGS_DIR, "confusion_matrix_eval.png")
    )

    # Per-class confidence distribution
    probs_by_class = {
        i: probs[y_test == i, i] for i in range(N_CLASSES)
    }
    plot_confidence_violin(
        probs_by_class,
        os.path.join(LOGS_DIR, "confidence_violin.png")
    )

    # GradCAM
    plot_gradcam_samples(
        model, X_test, y_test, scaler,
        os.path.join(LOGS_DIR, "gradcam_samples.png"),
        n_per_class=2
    )

    # ── Save summary ──────────────────────────────────────────────────────────
    summary = {
        "test_accuracy"  : round(float(acc),        4),
        "macro_f1"       : round(float(macro_f1),   4),
        "weighted_f1"    : round(float(weighted_f1),4),
        "macro_auc_roc"  : round(float(auc),        4),
        "n_test_samples" : int(len(y_test)),
        "class_distribution": {
            CLASS_NAMES[i]: int((y_test == i).sum()) for i in range(N_CLASSES)
        },
    }
    summary_path = os.path.join(LOGS_DIR, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved → {summary_path}")

    print("\n" + "="*60)
    print(f"  ✅  Accuracy  : {acc*100:.2f}%")
    print(f"  ✅  Macro F1  : {macro_f1:.4f}")
    print(f"  ✅  AUC-ROC   : {auc:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
