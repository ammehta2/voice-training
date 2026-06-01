"""
Day 7-8 — Train a DS-CNN (Depthwise Separable CNN) for Navkar detection.

Loads features from data/features/, trains until early stopping fires,
saves best + final checkpoints to models/navkar_v1_<timestamp>/.

Usage:
    python scripts/05_train_model.py

Monitor in another shell:
    tensorboard --logdir logs/
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models

BATCH_SIZE = 64
EPOCHS = 80
INITIAL_LR = 1e-3
DROPOUT = 0.3

# Auto-applies class weights when classes are imbalanced.
# With 117 positives vs 6000 negatives, naive training collapses to "always say no".
# Class weights tell the loss function to penalize misclassified positives more.
USE_CLASS_WEIGHTS = True


def configure_gpu() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU(s): {[g.name for g in gpus]}")
    else:
        print("WARNING: No GPU detected. Training will be slow on CPU.")


def build_ds_cnn(input_shape) -> tf.keras.Model:
    """Depthwise Separable CNN — proven mobile KWS architecture."""
    inputs = layers.Input(shape=input_shape)

    # Aggressive initial downsampling
    x = layers.Conv2D(64, (10, 4), strides=(2, 2), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 4 DS-CNN blocks
    for filters in [64, 64, 64, 64]:
        x = layers.DepthwiseConv2D((3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, (1, 1), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(DROPOUT)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    return models.Model(inputs, outputs, name="navkar_kws")


def plot_history(history, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    pairs = [
        (axes[0, 0], "loss", "Loss"),
        (axes[0, 1], "accuracy", "Accuracy"),
        (axes[1, 0], "precision", "Precision"),
        (axes[1, 1], "recall", "Recall"),
    ]
    for ax, metric, title in pairs:
        if metric in history.history:
            ax.plot(history.history[metric], label="train")
        val_key = f"val_{metric}"
        if val_key in history.history:
            ax.plot(history.history[val_key], label="val")
        ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    plt.savefig(str(out_path))
    print(f"  Training curves: {out_path}")


def main() -> None:
    configure_gpu()

    print("Loading features...")
    X_train = np.load("data/features/X_train.npy")
    y_train = np.load("data/features/y_train.npy")
    X_val = np.load("data/features/X_val.npy")
    y_val = np.load("data/features/y_val.npy")

    print(f"  Train: {X_train.shape} | Val: {X_val.shape}")
    print(f"  Train pos: {int(sum(y_train))}, neg: {int(len(y_train) - sum(y_train))}")

    # Add channel dim: (N, T, F) -> (N, T, F, 1)
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    input_shape = X_train.shape[1:]
    print(f"  Model input shape: {input_shape}")

    model = build_ds_cnn(input_shape)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LR),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path(f"models/navkar_v1_{timestamp}")
    model_dir.mkdir(parents=True, exist_ok=True)

    callbacks_list = [
        callbacks.ModelCheckpoint(
            filepath=str(model_dir / "best.keras"),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.TensorBoard(log_dir=str(Path("logs") / timestamp), histogram_freq=1),
    ]

    # Compute class weights for imbalanced data
    class_weight = None
    if USE_CLASS_WEIGHTS:
        n_pos = int(sum(y_train))
        n_neg = int(len(y_train) - n_pos)
        total = n_pos + n_neg
        # sklearn-style "balanced": n_total / (n_classes * n_samples_in_class)
        w_pos = total / (2 * n_pos) if n_pos > 0 else 1.0
        w_neg = total / (2 * n_neg) if n_neg > 0 else 1.0
        class_weight = {0: w_neg, 1: w_pos}
        print(f"\nClass weights: neg={w_neg:.3f}, pos={w_pos:.3f}  "
              f"(ratio neg:pos = {n_neg}:{n_pos})")

    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks_list,
        class_weight=class_weight,
        verbose=1,
    )

    model.save(str(model_dir / "final.keras"))
    print(f"\nFinal model: {model_dir / 'final.keras'}")
    print(f"Best model:  {model_dir / 'best.keras'}")

    plot_history(history, model_dir / "training_curves.png")

    print("\n=== Final epoch metrics ===")
    for k, v in history.history.items():
        print(f"  {k}: {v[-1]:.4f}")


if __name__ == "__main__":
    main()
