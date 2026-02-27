"""
dl_model.py
===========
Defines the 1D-CNN + BiLSTM + Multi-Head Attention deep learning model
for pothole / road-anomaly detection from LiDAR time-series windows.

Architecture
────────────
  Input  :  (batch, WINDOW_SIZE, N_RAW_FEATURES)

  ┌─ CNN backbone (3 conv blocks) ─────────────────────────────────────────┐
  │  Conv1D → BatchNorm → GELU → Conv1D → BatchNorm → GELU → MaxPool      │
  │  (× 3 blocks, filters ×2 each block, residual skip where dims match)   │
  └─────────────────────────────────────────────────────────────────────────┘
         ↓
  ┌─ BiLSTM blocks (2 stacked) ────────────────────────────────────────────┐
  │  Bidirectional LSTM(128) → Dropout → Bidirectional LSTM(64) → Dropout  │
  └─────────────────────────────────────────────────────────────────────────┘
         ↓
  ┌─ Multi-Head Self-Attention ─────────────────────────────────────────────┐
  │  MultiHeadAttention(heads=4, key_dim=32) + LayerNorm + Residual         │
  └─────────────────────────────────────────────────────────────────────────┘
         ↓
  ┌─ Aggregation ───────────────────────────────────────────────────────────┐
  │  GlobalAveragePooling1D + GlobalMaxPooling1D  → Concatenate             │
  └─────────────────────────────────────────────────────────────────────────┘
         ↓
  ┌─ Dense head ────────────────────────────────────────────────────────────┐
  │  Dense(128,GELU) → Dropout → Dense(64,GELU) → Dropout → Dense(4,SM)    │
  └─────────────────────────────────────────────────────────────────────────┘

Total trainable params: ~1.5 M  (fast 5-15ms inference on CPU / <1ms GPU)
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # suppress TF info/warning logs

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from dl_config import (
    WINDOW_SIZE, N_RAW_FEATURES, N_CLASSES,
    CNN_FILTERS, CNN_KERNEL, POOL_SIZE, DROPOUT_CNN,
    LSTM_UNITS, DROPOUT_LSTM,
    ATTENTION_HEADS, ATTENTION_KEY_DIM,
    DENSE_UNITS, DROPOUT_DENSE,
    LEARNING_RATE,
)


# ── Custom Layers ──────────────────────────────────────────────────────────────

class SqueezeExcitation(layers.Layer):
    """
    Channel-wise Squeeze-and-Excitation block.
    Recalibrates CNN channel responses by modelling inter-channel dependencies.
    """
    def __init__(self, filters: int, ratio: int = 8, **kw):
        super().__init__(**kw)
        self.gap = layers.GlobalAveragePooling1D()
        self.fc1 = layers.Dense(max(1, filters // ratio), activation="relu")
        self.fc2 = layers.Dense(filters, activation="sigmoid")

    def call(self, x):
        s = self.gap(x)                          # (B, C)
        s = self.fc1(s)
        s = self.fc2(s)                          # (B, C)
        s = tf.reshape(s, (-1, 1, tf.shape(x)[-1]))  # (B, 1, C)
        return x * s


def cnn_block(x, filters: int, kernel_size: int,
              dropout: float, se_ratio: int = 8) -> tf.Tensor:
    """
    One CNN block:
      Conv1D → BN → GELU → Conv1D → BN → SE → GELU → MaxPool
    + optional residual projection if channel dims differ.
    """
    shortcut = x

    x = layers.Conv1D(filters, kernel_size, padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("gelu")(x)

    x = layers.Conv1D(filters, kernel_size, padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = SqueezeExcitation(filters, ratio=se_ratio)(x)
    x = layers.Activation("gelu")(x)

    # Residual projection if shape differs
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding="same")(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Add()([x, shortcut])

    x = layers.MaxPooling1D(POOL_SIZE, padding="same")(x)
    x = layers.Dropout(dropout)(x)
    return x


# ── Model builder ──────────────────────────────────────────────────────────────

def build_model(
    window_size:   int   = WINDOW_SIZE,
    n_features:    int   = N_RAW_FEATURES,
    n_classes:     int   = N_CLASSES,
    learning_rate: float = LEARNING_RATE,
) -> keras.Model:
    """
    Build and compile the 1D-CNN + BiLSTM + Attention model.

    Parameters
    ----------
    window_size   : sequence length (timesteps)
    n_features    : features per timestep
    n_classes     : number of output classes
    learning_rate : initial Adam learning rate

    Returns
    -------
    Compiled keras.Model
    """
    inputs = keras.Input(shape=(window_size, n_features), name="lidar_window")
    x      = inputs

    # ── CNN backbone ──────────────────────────────────────────────────────────
    for i, filt in enumerate(CNN_FILTERS):
        x = cnn_block(x, filters=filt, kernel_size=CNN_KERNEL,
                      dropout=DROPOUT_CNN, se_ratio=8)

    # ── BiLSTM ────────────────────────────────────────────────────────────────
    for i, units in enumerate(LSTM_UNITS):
        return_seq = True   # always return sequences (needed for attention)
        x = layers.Bidirectional(
            layers.LSTM(units, return_sequences=return_seq,
                        dropout=DROPOUT_LSTM * 0.5,
                        recurrent_dropout=0.0,       # keeps it GPU-compatible
                        kernel_regularizer=regularizers.l2(1e-4)),
            name=f"bilstm_{i}"
        )(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(DROPOUT_LSTM)(x)

    # ── Multi-Head Self-Attention ─────────────────────────────────────────────
    attn_out, attn_weights = layers.MultiHeadAttention(
        num_heads = ATTENTION_HEADS,
        key_dim   = ATTENTION_KEY_DIM,
        dropout   = 0.1,
        name      = "mha"
    )(x, x, return_attention_scores=True)
    x = layers.Add()([x, attn_out])            # residual
    x = layers.LayerNormalization()(x)

    # ── Aggregation ───────────────────────────────────────────────────────────
    gap = layers.GlobalAveragePooling1D()(x)
    gmp = layers.GlobalMaxPooling1D()(x)
    x   = layers.Concatenate()([gap, gmp])

    # ── Dense classification head ─────────────────────────────────────────────
    for units in DENSE_UNITS:
        x = layers.Dense(units, kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("gelu")(x)
        x = layers.Dropout(DROPOUT_DENSE)(x)

    outputs = layers.Dense(n_classes, activation="softmax", name="class_probs")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="PotholeDetector_DL")

    # ── Compile ───────────────────────────────────────────────────────────────
    optimizer = keras.optimizers.AdamW(
        learning_rate = learning_rate,
        weight_decay  = 1e-4,
    )
    model.compile(
        optimizer = optimizer,
        loss      = keras.losses.SparseCategoricalCrossentropy(),
        metrics   = [
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2_acc"),
        ],
    )
    return model


def load_model(path: str) -> keras.Model:
    """Load a saved .keras model from disk."""
    return keras.models.load_model(path, custom_objects={"SqueezeExcitation": SqueezeExcitation})


# ── Quick sanity-check ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    m = build_model()
    m.summary(line_length=90)
    dummy = tf.random.normal((4, WINDOW_SIZE, N_RAW_FEATURES))
    out   = m(dummy, training=False)
    print(f"\nForward pass OK  input={dummy.shape}  output={out.shape}")
    print(f"Softmax sum per sample: {tf.reduce_sum(out, axis=-1).numpy()}")
