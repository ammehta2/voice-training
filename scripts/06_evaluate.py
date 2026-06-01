"""
Day 9 — Evaluate trained model on the held-out test set.

Produces confusion matrix, ROC curve, and per-threshold precision/recall
table. All artifacts saved next to the model file.

Usage:
    python scripts/06_evaluate.py models/navkar_v1_<TIMESTAMP>/best.keras
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/06_evaluate.py <path/to/model.keras>")
        sys.exit(1)

    model_path = Path(sys.argv[1])
    model_dir = model_path.parent

    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(str(model_path))

    X_test = np.load("data/features/X_test.npy")[..., np.newaxis]
    y_test = np.load("data/features/y_test.npy")

    print(f"Test set: {len(X_test)} samples "
          f"(pos={int(sum(y_test))}, neg={int(len(y_test) - sum(y_test))})")

    print("\nPredicting...")
    y_pred_proba = model.predict(X_test, verbose=1).flatten()

    print("\n=== Performance at different thresholds ===")
    print(f"{'thresh':>7} {'precision':>10} {'recall':>8} {'F1':>6} {'FP':>5} {'FN':>5}")
    for threshold in [0.3, 0.5, 0.7, 0.85, 0.9, 0.95]:
        y_pred = (y_pred_proba >= threshold).astype(int)
        tp = int(((y_pred == 1) & (y_test == 1)).sum())
        fp = int(((y_pred == 1) & (y_test == 0)).sum())
        fn = int(((y_pred == 0) & (y_test == 1)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        print(f"{threshold:>7.2f} {precision:>10.3f} {recall:>8.3f} "
              f"{f1:>6.3f} {fp:>5d} {fn:>5d}")

    # Report at default threshold
    chosen = 0.85
    print(f"\n=== Classification Report (threshold={chosen}) ===")
    y_pred = (y_pred_proba >= chosen).astype(int)
    print(classification_report(y_test, y_pred, target_names=["Negative", "Navkar"]))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Negative", "Navkar"])
    ax.set_yticklabels(["Negative", "Navkar"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    cm_path = model_dir / "confusion_matrix.png"
    plt.savefig(str(cm_path))
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    roc_path = model_dir / "roc_curve.png"
    plt.savefig(str(roc_path))
    plt.close()

    print(f"\nArtifacts saved to {model_dir}")
    print(f"  - {cm_path.name}")
    print(f"  - {roc_path.name}")
    print(f"\nAUC: {roc_auc:.4f}")


if __name__ == "__main__":
    main()
