"""
Train the opening-phrase ("Namo Arihantaanam") detector.

Reads features from data/opening/features/, saves model to
models/opening_v1_<timestamp>/.

Smaller architecture than 05_train_model_pytorch.py because:
  - Input window is 2.5s instead of 16s
  - Task is detect a 2.5s phrase, not a 12-15s mantra
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

BATCH_SIZE = 64
EPOCHS = 80
INITIAL_LR = 1e-3
DROPOUT = 0.3
PATIENCE_EARLY_STOP = 15
PATIENCE_LR_REDUCE = 5
LR_REDUCE_FACTOR = 0.5
MIN_LR = 1e-6
USE_CLASS_WEIGHTS = True

FEATURES_DIR = Path("data/opening/features")


class DSBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, 3, padding=1, groups=in_c, bias=False)
        self.bn1 = nn.BatchNorm2d(in_c)
        self.pw = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = F.relu(self.bn1(self.dw(x)))
        x = F.relu(self.bn2(self.pw(x)))
        return x


class OpeningDetector(nn.Module):
    """Compact DS-CNN for 2.5s opening-phrase detection.

    Input:  (B, 1, T~251, F=40)  channels-first
    Output: (B,) raw logits
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.conv0 = nn.Conv2d(1, 48, (5, 4), stride=(2, 2), padding=(2, 1), bias=False)
        self.bn0 = nn.BatchNorm2d(48)
        self.blocks = nn.Sequential(
            DSBlock(48, 48),
            DSBlock(48, 48),
            DSBlock(48, 64),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn0(self.conv0(x)))
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        x = self.drop(x)
        return self.fc(x).squeeze(-1)


def compute_metrics(logits, labels, threshold=0.5):
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


def compute_auc(logits, labels):
    from sklearn.metrics import roc_auc_score
    p = torch.sigmoid(logits).cpu().numpy()
    l = labels.cpu().numpy()
    if len(np.unique(l)) < 2:
        return 0.5
    return float(roc_auc_score(l, p))


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss = 0.0
    all_l, all_y = [], []
    with torch.set_grad_enabled(train):
        for X, y in loader:
            X = X.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            logit = model(X)
            loss = criterion(logit, y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * X.size(0)
            all_l.append(logit.detach()); all_y.append(y.detach())
    logits = torch.cat(all_l); labels = torch.cat(all_y)
    avg_loss = total_loss / len(loader.dataset)
    acc, prec, rec = compute_metrics(logits, labels, 0.5)
    auc = compute_auc(logits, labels)
    return avg_loss, acc, prec, rec, auc


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    X_train = np.load(FEATURES_DIR / "X_train.npy")
    y_train = np.load(FEATURES_DIR / "y_train.npy")
    X_val = np.load(FEATURES_DIR / "X_val.npy")
    y_val = np.load(FEATURES_DIR / "y_val.npy")

    X_train = X_train[:, np.newaxis, :, :]
    X_val = X_val[:, np.newaxis, :, :]
    print(f"  Train: {X_train.shape} | Val: {X_val.shape}")
    n_pos = int(y_train.sum()); n_neg = len(y_train) - n_pos
    print(f"  Train pos={n_pos}, neg={n_neg}")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()),
        batch_size=BATCH_SIZE, shuffle=True, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()),
        batch_size=BATCH_SIZE, shuffle=False, pin_memory=(device.type == "cuda"))

    model = OpeningDetector(DROPOUT).to(device)
    print(f"\nModel params: {sum(p.numel() for p in model.parameters()):,}")

    pos_weight = None
    if USE_CLASS_WEIGHTS and n_pos > 0:
        pos_weight = torch.tensor([n_neg / n_pos], device=device)
        print(f"  pos_weight = {pos_weight.item():.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=INITIAL_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_REDUCE_FACTOR,
        patience=PATIENCE_LR_REDUCE, min_lr=MIN_LR)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path(f"models/opening_v1_{timestamp}")
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to {model_dir}/")

    history = {k: [] for k in
               ["train_loss", "train_acc", "train_prec", "train_rec", "train_auc",
                "val_loss", "val_acc", "val_prec", "val_rec", "val_auc", "lr"]}
    best_val_auc = -1.0
    epochs_since_improve = 0

    for epoch in range(1, EPOCHS + 1):
        tr = run_epoch(model, train_loader, criterion, optimizer, device, True)
        va = run_epoch(model, val_loader, criterion, optimizer, device, False)
        lr = optimizer.param_groups[0]["lr"]

        for k, v in zip(["loss", "acc", "prec", "rec", "auc"], tr):
            history[f"train_{k}"].append(v)
        for k, v in zip(["loss", "acc", "prec", "rec", "auc"], va):
            history[f"val_{k}"].append(v)
        history["lr"].append(lr)

        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"loss={tr[0]:.4f}/{va[0]:.4f}  acc={tr[1]:.3f}/{va[1]:.3f}  "
              f"P={tr[2]:.3f}/{va[2]:.3f}  R={tr[3]:.3f}/{va[3]:.3f}  "
              f"AUC={tr[4]:.3f}/{va[4]:.3f}  lr={lr:.1e}")

        scheduler.step(va[0])
        if va[4] > best_val_auc:
            best_val_auc = va[4]
            torch.save(model.state_dict(), model_dir / "best.pt")
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= PATIENCE_EARLY_STOP:
                print(f"  Early stopping")
                break

    torch.save(model.state_dict(), model_dir / "final.pt")
    print(f"\nBest val AUC: {best_val_auc:.4f}")
    print(f"  best.pt:  {model_dir / 'best.pt'}")

    np.savez(model_dir / "history.npz", **{k: np.array(v) for k, v in history.items()})

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, key, title in [
        (axes[0, 0], "loss", "Loss"), (axes[0, 1], "acc", "Accuracy"),
        (axes[1, 0], "prec", "Precision"), (axes[1, 1], "rec", "Recall")]:
        ax.plot(history[f"train_{key}"], label="train")
        ax.plot(history[f"val_{key}"], label="val")
        ax.set_title(title); ax.legend(); ax.set_xlabel("epoch")
    plt.tight_layout()
    plt.savefig(model_dir / "training_curves.png")


if __name__ == "__main__":
    main()
