"""
Day 7-8 (PyTorch version) — Train a DS-CNN for Navkar detection.

Uses PyTorch + CUDA 12.8 (works on RTX 5090 Blackwell).
Saves best.pt + final.pt + training_curves.png to models/navkar_v1_<timestamp>/.

The TFLite export script (08_convert_to_tflite_pytorch.py) handles the
PyTorch -> ONNX -> TFLite conversion afterwards.

Usage:
    python scripts/05_train_model_pytorch.py

Monitor:
    GPU: nvidia-smi -l 1
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ---- Hyperparameters ----
BATCH_SIZE = 64
EPOCHS = 80
INITIAL_LR = 1e-3
DROPOUT = 0.3

# Early stopping & LR schedule
PATIENCE_EARLY_STOP = 15  # epochs without val_auc improvement
PATIENCE_LR_REDUCE = 5    # epochs without val_loss improvement
LR_REDUCE_FACTOR = 0.5
MIN_LR = 1e-6

USE_CLASS_WEIGHTS = True  # auto-handles 5000:117 negative:positive ratio


# ---- Model ----
class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                   padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.depthwise(x)))
        x = F.relu(self.bn2(self.pointwise(x)))
        return x


class NavkarKWS(nn.Module):
    """DS-CNN for Navkar keyword spotting.

    Input: (B, 1, T, F) — channels-first (T=1600 time frames, F=40 mel coefs)
    Output: (B,) — raw logits (apply sigmoid for probability)
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        # Initial aggressive downsampling (replaces TF's (10,4) stride 2 conv)
        self.conv0 = nn.Conv2d(1, 64, kernel_size=(10, 4), stride=(2, 2),
                               padding=(5, 2), bias=False)
        self.bn0 = nn.BatchNorm2d(64)

        # 4 DS-CNN blocks
        self.blocks = nn.Sequential(
            DepthwiseSeparableBlock(64, 64),
            DepthwiseSeparableBlock(64, 64),
            DepthwiseSeparableBlock(64, 64),
            DepthwiseSeparableBlock(64, 64),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn0(self.conv0(x)))
        x = self.blocks(x)
        x = self.global_pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x).squeeze(-1)  # raw logits, shape (B,)


# ---- Training ----
def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5):
    """Returns (accuracy, precision, recall) at given threshold."""
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).long()
    labels = labels.long()

    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()

    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return acc, prec, rec


def compute_auc(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """ROC AUC via trapezoidal integration."""
    from sklearn.metrics import roc_auc_score
    probs = torch.sigmoid(logits).cpu().numpy()
    labels_np = labels.cpu().numpy()
    if len(np.unique(labels_np)) < 2:
        return 0.5
    return float(roc_auc_score(labels_np, probs))


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss = 0.0
    all_logits, all_labels = [], []

    with torch.set_grad_enabled(train):
        for X, y in loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(X)
            loss = criterion(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * X.size(0)
            all_logits.append(logits.detach())
            all_labels.append(y.detach())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    avg_loss = total_loss / len(loader.dataset)
    acc, prec, rec = compute_metrics(logits, labels, threshold=0.5)
    auc = compute_auc(logits, labels)
    return avg_loss, acc, prec, rec, auc


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print("Loading features...")
    X_train = np.load("data/features/X_train.npy")
    y_train = np.load("data/features/y_train.npy")
    X_val = np.load("data/features/X_val.npy")
    y_val = np.load("data/features/y_val.npy")

    # (N, T, F) -> (N, 1, T, F) for channels-first Conv2d
    X_train = X_train[:, np.newaxis, :, :]
    X_val = X_val[:, np.newaxis, :, :]
    print(f"  Train: {X_train.shape} | Val: {X_val.shape}")
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    print(f"  Train pos={n_pos}, neg={n_neg}")

    # DataLoaders
    train_ds = TensorDataset(torch.from_numpy(X_train).float(),
                             torch.from_numpy(y_train).float())
    val_ds = TensorDataset(torch.from_numpy(X_val).float(),
                           torch.from_numpy(y_val).float())
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=(device.type == "cuda"))

    model = NavkarKWS(dropout=DROPOUT).to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Class-weighted BCE: pos_weight scales the positive class loss
    if USE_CLASS_WEIGHTS and n_pos > 0:
        pos_weight = torch.tensor([n_neg / n_pos], device=device)
        print(f"  Using pos_weight={pos_weight.item():.2f}")
    else:
        pos_weight = None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=INITIAL_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_REDUCE_FACTOR,
        patience=PATIENCE_LR_REDUCE, min_lr=MIN_LR
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path(f"models/navkar_v1_{timestamp}")
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to {model_dir}/")

    history = {k: [] for k in
               ["train_loss", "train_acc", "train_prec", "train_rec", "train_auc",
                "val_loss", "val_acc", "val_prec", "val_rec", "val_auc", "lr"]}

    best_val_auc = -1.0
    epochs_since_improve = 0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, tr_prec, tr_rec, tr_auc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc, val_prec, val_rec, val_auc = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False)
        lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["train_prec"].append(tr_prec)
        history["train_rec"].append(tr_rec)
        history["train_auc"].append(tr_auc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_prec"].append(val_prec)
        history["val_rec"].append(val_rec)
        history["val_auc"].append(val_auc)
        history["lr"].append(lr)

        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"loss={tr_loss:.4f}/{val_loss:.4f}  "
              f"acc={tr_acc:.3f}/{val_acc:.3f}  "
              f"P={tr_prec:.3f}/{val_prec:.3f}  "
              f"R={tr_rec:.3f}/{val_rec:.3f}  "
              f"AUC={tr_auc:.3f}/{val_auc:.3f}  "
              f"lr={lr:.1e}")

        scheduler.step(val_loss)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), model_dir / "best.pt")
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= PATIENCE_EARLY_STOP:
                print(f"  Early stopping (no val_auc improvement for "
                      f"{PATIENCE_EARLY_STOP} epochs)")
                break

    torch.save(model.state_dict(), model_dir / "final.pt")
    print(f"\nBest val AUC: {best_val_auc:.4f}")
    print(f"  best.pt:  {model_dir / 'best.pt'}")
    print(f"  final.pt: {model_dir / 'final.pt'}")

    # Save history + plot
    np.savez(model_dir / "history.npz", **{k: np.array(v) for k, v in history.items()})

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    pairs = [
        (axes[0, 0], "loss", "Loss"),
        (axes[0, 1], "acc", "Accuracy"),
        (axes[1, 0], "prec", "Precision"),
        (axes[1, 1], "rec", "Recall"),
    ]
    for ax, key, title in pairs:
        ax.plot(history[f"train_{key}"], label="train")
        ax.plot(history[f"val_{key}"], label="val")
        ax.set_title(title); ax.legend(); ax.set_xlabel("epoch")
    plt.tight_layout()
    plt.savefig(model_dir / "training_curves.png")
    print(f"  training_curves.png: {model_dir / 'training_curves.png'}")


if __name__ == "__main__":
    main()
