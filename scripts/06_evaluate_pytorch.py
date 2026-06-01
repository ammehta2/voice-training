"""
Day 9 (PyTorch version) — Evaluate trained PyTorch model on held-out test set.

Generates per-threshold precision/recall/F1 table, confusion matrix, and ROC
curve. Saves artifacts next to the model file.

Usage:
    python scripts/06_evaluate_pytorch.py models/navkar_v1_<TS>/best.pt
"""

import sys
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve

sys.path.insert(0, str(Path(__file__).parent))
train_module = import_module("05_train_model_pytorch")
NavkarKWS = train_module.NavkarKWS


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/06_evaluate_pytorch.py <path/to/best.pt>")
        sys.exit(1)

    model_path = Path(sys.argv[1])
    model_dir = model_path.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model: {model_path}")
    model = NavkarKWS(dropout=0.0).to(device).eval()
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)

    X_test = np.load("data/features/X_test.npy")
    y_test = np.load("data/features/y_test.npy")
    X = torch.from_numpy(X_test[:, np.newaxis, :, :]).float().to(device)
    print(f"Test set: {len(X_test)} samples "
          f"(pos={int(y_test.sum())}, neg={int(len(y_test) - y_test.sum())})")

    with torch.no_grad():
        # Run in batches to avoid OOM
        batch = 64
        logits = []
        for i in range(0, len(X), batch):
            logits.append(model(X[i : i + batch]).cpu())
        logits = torch.cat(logits)
        y_pred_proba = torch.sigmoid(logits).numpy()

    print("\n=== Per-threshold performance ===")
    print(f"{'thresh':>7} {'precision':>10} {'recall':>8} {'F1':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>5}")
    for threshold in [0.3, 0.5, 0.7, 0.85, 0.9, 0.95]:
        y_pred = (y_pred_proba >= threshold).astype(int)
        tp = int(((y_pred == 1) & (y_test == 1)).sum())
        fp = int(((y_pred == 1) & (y_test == 0)).sum())
        fn = int(((y_pred == 0) & (y_test == 1)).sum())
        tn = int(((y_pred == 0) & (y_test == 0)).sum())
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0
        print(f"{threshold:>7.2f} {p:>10.3f} {r:>8.3f} {f1:>6.3f} "
              f"{tp:>4d} {fp:>4d} {fn:>4d} {tn:>5d}")

    chosen = 0.85
    print(f"\n=== Classification report (threshold={chosen}) ===")
    y_pred = (y_pred_proba >= chosen).astype(int)
    print(classification_report(y_test, y_pred, target_names=["Negative", "Navkar"], zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix [rows=actual, cols=predicted]:")
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
    plt.savefig(str(model_dir / "confusion_matrix.png"))
    plt.close()

    # ROC
    if len(np.unique(y_test)) >= 2:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 5))
        plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.savefig(str(model_dir / "roc_curve.png"))
        plt.close()
        print(f"\nROC AUC: {roc_auc:.4f}")

    print(f"\nArtifacts saved to {model_dir}")


if __name__ == "__main__":
    main()
