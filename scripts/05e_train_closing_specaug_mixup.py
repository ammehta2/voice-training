"""
Train closing-phrase detector v5 — adds SpecAugment + Mixup + temperature scaling.

These are Tier-1 improvements over scripts/05d_train_closing.py:

  SpecAugment (Park et al. 2019)
    Mask random frequency bands and time spans of the MFCC tensor during training.
    Forces the model to generalize across missing acoustic context.
    Standard practice in modern speech models.

  Mixup (Zhang et al. 2017)
    Linearly blend pairs of (sample, label) during training. Smoother decision
    boundaries; particularly effective with our voice-diverse-but-finite data.

  Temperature scaling (Guo et al. 2017)
    Post-hoc calibration: find scalar T such that sigmoid(logits/T) best matches
    the empirical validation distribution. Makes the production threshold mean
    what it says.

Uses the SAME ClosingDetector architecture as 05d (imported), so the only thing
that differs from Model D is the training regimen.

Usage:
    python scripts/05e_train_closing_specaug_mixup.py
"""

import math
import sys
from datetime import datetime
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Import ClosingDetector from 05d to keep architecture identical
sys.path.insert(0, str(Path(__file__).parent))
ClosingDetector = import_module("05d_train_closing").ClosingDetector

BATCH_SIZE = 64
EPOCHS = 80
INITIAL_LR = 1e-3
DROPOUT = 0.3
PATIENCE_EARLY_STOP = 15
PATIENCE_LR_REDUCE = 5
LR_REDUCE_FACTOR = 0.5
MIN_LR = 1e-6
USE_CLASS_WEIGHTS = True

# ---- Tier 1 augmentation settings ----
USE_SPECAUGMENT = True
SPECAUG_FREQ_MASK = 10        # Mask up to 10 of 40 mel coefs
SPECAUG_TIME_MASK = 20        # Mask up to 20 of ~251 time frames
SPECAUG_N_FREQ_MASKS = 2
SPECAUG_N_TIME_MASKS = 2
SPECAUG_PROB = 0.8            # Apply on 80% of batches

USE_MIXUP = True
MIXUP_ALPHA = 0.2             # Beta(alpha, alpha) — small = barely-blended, large = strongly-blended
MIXUP_PROB = 0.5              # Apply on 50% of batches

USE_TEMPERATURE_SCALING = True

FEATURES_DIR = Path("data/closing/features")


# ============================================================================
# Augmentations
# ============================================================================
def spec_augment(x: torch.Tensor) -> torch.Tensor:
    """In-place SpecAugment on a batch of (B, 1, T, F) MFCC tensors.

    Applies independent frequency and time masking on each item in the batch.
    """
    B, _, T, F_dim = x.shape
    for b in range(B):
        # Frequency masks
        for _ in range(SPECAUG_N_FREQ_MASKS):
            width = torch.randint(0, SPECAUG_FREQ_MASK + 1, (1,)).item()
            if width == 0:
                continue
            start = torch.randint(0, max(1, F_dim - width), (1,)).item()
            x[b, :, :, start:start + width] = 0
        # Time masks
        for _ in range(SPECAUG_N_TIME_MASKS):
            width = torch.randint(0, SPECAUG_TIME_MASK + 1, (1,)).item()
            if width == 0:
                continue
            start = torch.randint(0, max(1, T - width), (1,)).item()
            x[b, :, start:start + width, :] = 0
    return x


def mixup_batch(X: torch.Tensor, y: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Mix pairs of samples + labels with random Beta-distributed weights."""
    B = X.size(0)
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(B, device=X.device)
    X_mixed = lam * X + (1 - lam) * X[perm]
    y_mixed = lam * y + (1 - lam) * y[perm]
    return X_mixed, y_mixed


# ============================================================================
# Training / eval
# ============================================================================
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
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Augmentations only during training
            if train:
                if USE_SPECAUGMENT and np.random.rand() < SPECAUG_PROB:
                    X = spec_augment(X.clone())  # clone to avoid mutating cached tensor
                if USE_MIXUP and np.random.rand() < MIXUP_PROB:
                    X, y_mix = mixup_batch(X, y, MIXUP_ALPHA)
                    logit = model(X)
                    loss = criterion(logit, y_mix)
                else:
                    logit = model(X)
                    loss = criterion(logit, y)
            else:
                logit = model(X)
                loss = criterion(logit, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * X.size(0)
            all_l.append(logit.detach())
            # For metrics, use the ORIGINAL y (not the mixup-blended one) so
            # train metrics are interpretable
            all_y.append(y.detach())

    logits = torch.cat(all_l)
    labels = torch.cat(all_y)
    avg_loss = total_loss / len(loader.dataset)
    acc, prec, rec = compute_metrics(logits, labels, 0.5)
    auc = compute_auc(logits, labels)
    return avg_loss, acc, prec, rec, auc


# ============================================================================
# Temperature scaling (post-training)
# ============================================================================
def fit_temperature(model, loader, device, n_iter: int = 200) -> float:
    """Find scalar T that minimizes NLL on val set: sigmoid(logits / T)."""
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device); y = y.to(device)
            all_logits.append(model(X).cpu())
            all_labels.append(y.cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    # Optimize T via LBFGS
    T = nn.Parameter(torch.ones(1) * 1.0)
    bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.LBFGS([T], lr=0.1, max_iter=n_iter)

    def closure():
        optimizer.zero_grad()
        loss = bce(logits / T.clamp(min=1e-3), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(T.detach().item())


# ============================================================================
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"SpecAugment: {USE_SPECAUGMENT} (F<={SPECAUG_FREQ_MASK} T<={SPECAUG_TIME_MASK} "
          f"n_freq={SPECAUG_N_FREQ_MASKS} n_time={SPECAUG_N_TIME_MASKS} p={SPECAUG_PROB})")
    print(f"Mixup:       {USE_MIXUP} (alpha={MIXUP_ALPHA} p={MIXUP_PROB})")
    print(f"Temp scaling: {USE_TEMPERATURE_SCALING}")

    X_train = np.load(FEATURES_DIR / "X_train.npy")
    y_train = np.load(FEATURES_DIR / "y_train.npy")
    X_val = np.load(FEATURES_DIR / "X_val.npy")
    y_val = np.load(FEATURES_DIR / "y_val.npy")

    X_train = X_train[:, np.newaxis, :, :]
    X_val = X_val[:, np.newaxis, :, :]
    print(f"\n  Train: {X_train.shape} | Val: {X_val.shape}")
    n_pos = int(y_train.sum()); n_neg = len(y_train) - n_pos
    print(f"  Train pos={n_pos}, neg={n_neg}")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()),
        batch_size=BATCH_SIZE, shuffle=True, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()),
        batch_size=BATCH_SIZE, shuffle=False, pin_memory=(device.type == "cuda"))

    model = ClosingDetector(DROPOUT).to(device)
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
    model_dir = Path(f"models/closing_v1_{timestamp}")
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

    # ---- Temperature scaling on val set ----
    if USE_TEMPERATURE_SCALING:
        # Reload best model for calibration
        model.load_state_dict(torch.load(model_dir / "best.pt", map_location=device, weights_only=True))
        T = fit_temperature(model, val_loader, device)
        print(f"\nTemperature scaling: T = {T:.4f}")
        print(f"  Save logits divided by T for calibrated probabilities.")
        # Save T alongside the model
        with open(model_dir / "temperature.txt", "w") as f:
            f.write(f"{T:.6f}\n")

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
