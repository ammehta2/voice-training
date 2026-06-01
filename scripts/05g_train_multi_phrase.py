"""
Train the v7 multi-phrase model — 2 output heads (closing + pre-closing).

Architecture: same shared DS-CNN backbone as ClosingDetector, with TWO
final Linear(64, 1) heads instead of one. Output shape (B, 2).

Loss: multi-label BCEWithLogitsLoss with per-head pos_weight to handle
class imbalance.

Reads from data/multi/features/ where y has shape (N, 2):
  [1, 0] = closing positive
  [0, 1] = preclosing positive
  [0, 0] = negative
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

FEATURES_DIR = Path("data/multi/features")


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


class MultiPhraseDetector(nn.Module):
    """2-head detector: same backbone, two parallel classification heads.

    Input:  (B, 1, T~251, F=40)  channels-first
    Output: (B, 2)  raw logits  [closing_logit, preclosing_logit]
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
        # Two heads instead of one
        self.fc_closing = nn.Linear(64, 1)
        self.fc_preclosing = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn0(self.conv0(x)))
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        x = self.drop(x)
        closing = self.fc_closing(x)        # (B, 1)
        preclosing = self.fc_preclosing(x)  # (B, 1)
        return torch.cat([closing, preclosing], dim=1)  # (B, 2)


def compute_metrics(logits, labels, threshold=0.5):
    """Per-head precision/recall, plus combined accuracy."""
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).long()
    labels = labels.long()

    metrics = {}
    for i, name in enumerate(["closing", "preclosing"]):
        p = preds[:, i]; l = labels[:, i]
        tp = ((p == 1) & (l == 1)).sum().item()
        fp = ((p == 1) & (l == 0)).sum().item()
        fn = ((p == 0) & (l == 1)).sum().item()
        tn = ((p == 0) & (l == 0)).sum().item()
        metrics[f"{name}_acc"] = (tp + tn) / max(1, tp + tn + fp + fn)
        metrics[f"{name}_prec"] = tp / max(1, tp + fp)
        metrics[f"{name}_rec"] = tp / max(1, tp + fn)
    return metrics


def compute_auc(logits, labels):
    from sklearn.metrics import roc_auc_score
    probs = torch.sigmoid(logits).cpu().numpy()
    lbls = labels.cpu().numpy()
    aucs = {}
    for i, name in enumerate(["closing", "preclosing"]):
        if len(np.unique(lbls[:, i])) < 2:
            aucs[f"{name}_auc"] = 0.5
        else:
            aucs[f"{name}_auc"] = float(roc_auc_score(lbls[:, i], probs[:, i]))
    return aucs


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
    return avg_loss, {**compute_metrics(logits, labels), **compute_auc(logits, labels)}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X_train = np.load(FEATURES_DIR / "X_train.npy")
    y_train = np.load(FEATURES_DIR / "y_train.npy")  # (N, 2)
    X_val = np.load(FEATURES_DIR / "X_val.npy")
    y_val = np.load(FEATURES_DIR / "y_val.npy")

    X_train = X_train[:, np.newaxis, :, :]
    X_val = X_val[:, np.newaxis, :, :]
    print(f"\n  Train: {X_train.shape} | Val: {X_val.shape}")
    print(f"  y_train shape: {y_train.shape}")
    n_close_pos = int(y_train[:, 0].sum())
    n_pre_pos = int(y_train[:, 1].sum())
    n_neg = int(np.sum((y_train[:, 0] == 0) & (y_train[:, 1] == 0)))
    print(f"  Closing positives: {n_close_pos}")
    print(f"  Pre-closing positives: {n_pre_pos}")
    print(f"  Negatives: {n_neg}")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()),
        batch_size=BATCH_SIZE, shuffle=True, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()),
        batch_size=BATCH_SIZE, shuffle=False, pin_memory=(device.type == "cuda"))

    model = MultiPhraseDetector(DROPOUT).to(device)
    print(f"\nModel params: {sum(p.numel() for p in model.parameters()):,}")

    # Per-head pos_weight
    pos_weight = None
    if USE_CLASS_WEIGHTS:
        # Negatives count toward BOTH heads (a negative sample contributes
        # to neg-class counts for both heads). So per-head:
        #   neg_count = n_neg + n_other_head_pos
        # Because for the closing head, preclosing-positive samples are negatives.
        w_close = (n_neg + n_pre_pos) / max(1, n_close_pos)
        w_pre = (n_neg + n_close_pos) / max(1, n_pre_pos)
        pos_weight = torch.tensor([w_close, w_pre], device=device)
        print(f"  pos_weight: closing={w_close:.2f}, preclosing={w_pre:.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=INITIAL_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_REDUCE_FACTOR,
        patience=PATIENCE_LR_REDUCE, min_lr=MIN_LR)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path(f"models/multi_v1_{timestamp}")
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to {model_dir}/")

    history = {}
    best_val_auc = -1.0
    epochs_since_improve = 0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_m = run_epoch(model, train_loader, criterion, optimizer, device, True)
        va_loss, va_m = run_epoch(model, val_loader, criterion, optimizer, device, False)
        lr = optimizer.param_groups[0]["lr"]

        # Average AUC across heads as our "best" metric
        va_avg_auc = (va_m["closing_auc"] + va_m["preclosing_auc"]) / 2

        for k, v in [("loss", tr_loss), ("val_loss", va_loss), ("lr", lr)]:
            history.setdefault(k, []).append(v)
        for k, v in tr_m.items():
            history.setdefault(f"train_{k}", []).append(v)
        for k, v in va_m.items():
            history.setdefault(f"val_{k}", []).append(v)

        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"loss={tr_loss:.4f}/{va_loss:.4f}  "
              f"clos_AUC={tr_m['closing_auc']:.3f}/{va_m['closing_auc']:.3f}  "
              f"pre_AUC={tr_m['preclosing_auc']:.3f}/{va_m['preclosing_auc']:.3f}  "
              f"clos_R={va_m['closing_rec']:.3f} pre_R={va_m['preclosing_rec']:.3f}  "
              f"lr={lr:.1e}")

        scheduler.step(va_loss)
        if va_avg_auc > best_val_auc:
            best_val_auc = va_avg_auc
            torch.save(model.state_dict(), model_dir / "best.pt")
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= PATIENCE_EARLY_STOP:
                print(f"  Early stopping")
                break

    torch.save(model.state_dict(), model_dir / "final.pt")
    print(f"\nBest val avg AUC: {best_val_auc:.4f}")
    print(f"  best.pt: {model_dir / 'best.pt'}")

    np.savez(model_dir / "history.npz",
             **{k: np.array(v) for k, v in history.items()})

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(history["loss"], label="train")
    axes[0, 0].plot(history["val_loss"], label="val")
    axes[0, 0].set_title("Loss"); axes[0, 0].legend()
    axes[0, 1].plot(history["val_closing_auc"], label="closing")
    axes[0, 1].plot(history["val_preclosing_auc"], label="preclosing")
    axes[0, 1].set_title("Val AUC per head"); axes[0, 1].legend()
    axes[1, 0].plot(history["val_closing_prec"], label="closing")
    axes[1, 0].plot(history["val_preclosing_prec"], label="preclosing")
    axes[1, 0].set_title("Val Precision"); axes[1, 0].legend()
    axes[1, 1].plot(history["val_closing_rec"], label="closing")
    axes[1, 1].plot(history["val_preclosing_rec"], label="preclosing")
    axes[1, 1].set_title("Val Recall"); axes[1, 1].legend()
    plt.tight_layout()
    plt.savefig(model_dir / "training_curves.png")


if __name__ == "__main__":
    main()
