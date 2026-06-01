"""
Day 6 (source-aware variant) — extract MFCC features with a SOURCE-AWARE
train/val/test split.

Standard `04_extract_features.py` splits files randomly, which puts augmented
variants of the same source recording across train/val/test → leakage → fake
100% accuracy. This version groups files by source recording stem and ensures
all augmentations of a given source end up in ONE split only.

How "source" is derived from filename:
  real_navkar_01_w00_aug3_light.wav  -> source = "real_navkar_01_w00"
                                                  (window-level grouping)
  real_navkar_03_aug5_medium.wav     -> source = "real_navkar_03"
  ambient_1-100032-A-0_aug0.wav      -> source = "ambient_1-100032-A-0"
  real_neg_05_aug1_medium.wav        -> source = "real_neg_05"

So overlapping windows from the SAME long recording are kept together —
otherwise the model could learn from window 0 and "validate" on window 1
of the same recording (still leakage, just subtler).

Usage:
    python scripts/04b_extract_features_source_split.py
"""

import random
import re
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

# Feature params — match 04_extract_features.py and 07_real_world_test.py
SAMPLE_RATE = 16000
DURATION_SEC = 16
N_SAMPLES = SAMPLE_RATE * DURATION_SEC
N_MFCC = 40
N_FFT = 512
HOP_LENGTH = 160
N_MELS = 40

SEED = 42
TRAIN_FRAC = 0.70  # of source recordings
VAL_FRAC = 0.15
# remainder goes to test


def extract_mfcc(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if len(audio) < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)), mode="constant")
    elif len(audio) > N_SAMPLES:
        start = (len(audio) - N_SAMPLES) // 2
        audio = audio[start : start + N_SAMPLES]
    mfcc = librosa.feature.mfcc(
        y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
        n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
    return mfcc.T.astype(np.float32)


def source_of(filename: str) -> str:
    """Strip augmentation suffix AND window suffix to recover the source recording.

    real_navkar_01_w00_aug3_light.wav -> real_navkar_01  (NOT _w00)
    real_navkar_03_aug5_medium.wav    -> real_navkar_03
    ambient_1-100032-A-0_aug0_light.wav -> ambient_1-100032-A-0

    Windows from the same recording share speaker/environment, so they must
    stay in the same split to avoid leakage.
    """
    stem = Path(filename).stem
    # Strip augmentation suffix first
    stem = re.sub(r"_(aug\d+_(light|medium|heavy)|orig)$", "", stem)
    # Then strip window suffix (_w00, _w01, ...) so all windows of the same
    # source recording group together
    stem = re.sub(r"_w\d+$", "", stem)
    return stem


def group_by_source(directory: Path) -> dict[str, list[Path]]:
    groups = defaultdict(list)
    for f in sorted(directory.glob("*.wav")):
        groups[source_of(f.name)].append(f)
    return dict(groups)


def split_sources(sources: list[str], rng: random.Random):
    """Split source NAMES (not files) into train/val/test."""
    sources = list(sources)
    rng.shuffle(sources)
    n = len(sources)
    n_train = max(1, int(n * TRAIN_FRAC))
    n_val = max(1, int(n * VAL_FRAC))
    # ensure at least 1 in test if possible
    n_test = max(1, n - n_train - n_val) if n >= 3 else 0
    # rebalance if rounding overshot
    while n_train + n_val + n_test > n:
        n_train -= 1
    return sources[:n_train], sources[n_train:n_train + n_val], sources[n_train + n_val:]


def process_split(files: list[Path], label: float):
    X, y = [], []
    for f in tqdm(files, leave=False):
        try:
            X.append(extract_mfcc(str(f)))
            y.append(label)
        except Exception as e:
            print(f"    Failed {f.name}: {e}")
    if not X:
        return np.zeros((0, 0, 0), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def main() -> None:
    rng = random.Random(SEED)

    pos_groups = group_by_source(Path("data/positive_aug"))
    neg_groups = group_by_source(Path("data/negative_aug"))
    print(f"Positive sources: {len(pos_groups)}  ({sum(len(v) for v in pos_groups.values())} files)")
    print(f"Negative sources: {len(neg_groups)}  ({sum(len(v) for v in neg_groups.values())} files)")

    # Split sources INTO train/val/test
    pos_train_src, pos_val_src, pos_test_src = split_sources(list(pos_groups), rng)
    neg_train_src, neg_val_src, neg_test_src = split_sources(list(neg_groups), rng)

    print(f"\nSource split:")
    print(f"  Positives: train={len(pos_train_src)}, val={len(pos_val_src)}, test={len(pos_test_src)}")
    print(f"    train: {pos_train_src}")
    print(f"    val:   {pos_val_src}")
    print(f"    test:  {pos_test_src}")
    print(f"  Negatives: train={len(neg_train_src)}, val={len(neg_val_src)}, test={len(neg_test_src)}")

    # Gather files per split
    def files_for(groups, sources):
        out = []
        for s in sources:
            out.extend(groups[s])
        return out

    print("\nExtracting features per split...")
    print("  Train positives...")
    Xp_tr, yp_tr = process_split(files_for(pos_groups, pos_train_src), 1.0)
    print("  Train negatives...")
    Xn_tr, yn_tr = process_split(files_for(neg_groups, neg_train_src), 0.0)
    print("  Val positives...")
    Xp_v, yp_v = process_split(files_for(pos_groups, pos_val_src), 1.0)
    print("  Val negatives...")
    Xn_v, yn_v = process_split(files_for(neg_groups, neg_val_src), 0.0)
    print("  Test positives...")
    Xp_te, yp_te = process_split(files_for(pos_groups, pos_test_src), 1.0)
    print("  Test negatives...")
    Xn_te, yn_te = process_split(files_for(neg_groups, neg_test_src), 0.0)

    X_train = np.concatenate([Xp_tr, Xn_tr], axis=0) if len(Xp_tr) else Xn_tr
    y_train = np.concatenate([yp_tr, yn_tr], axis=0) if len(yp_tr) else yn_tr
    X_val = np.concatenate([Xp_v, Xn_v], axis=0) if len(Xp_v) else Xn_v
    y_val = np.concatenate([yp_v, yn_v], axis=0) if len(yp_v) else yn_v
    X_test = np.concatenate([Xp_te, Xn_te], axis=0) if len(Xp_te) else Xn_te
    y_test = np.concatenate([yp_te, yn_te], axis=0) if len(yp_te) else yn_te

    # Shuffle each split (so a batch isn't all positives then all negatives)
    for X, y in [(X_train, y_train), (X_val, y_val), (X_test, y_test)]:
        if len(X) > 0:
            idx = np.arange(len(X))
            rng_np = np.random.RandomState(SEED)
            rng_np.shuffle(idx)
            X[:] = X[idx]
            y[:] = y[idx]

    print(f"\nSplit sizes:")
    print(f"  Train: {len(X_train)}  ({int(y_train.sum())} pos / {int(len(y_train) - y_train.sum())} neg)")
    print(f"  Val:   {len(X_val)}  ({int(y_val.sum())} pos / {int(len(y_val) - y_val.sum())} neg)")
    print(f"  Test:  {len(X_test)}  ({int(y_test.sum())} pos / {int(len(y_test) - y_test.sum())} neg)")

    out = Path("data/features")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "X_train.npy", X_train)
    np.save(out / "y_train.npy", y_train)
    np.save(out / "X_val.npy", X_val)
    np.save(out / "y_val.npy", y_val)
    np.save(out / "X_test.npy", X_test)
    np.save(out / "y_test.npy", y_test)
    print(f"\nFeatures saved to {out}")


if __name__ == "__main__":
    main()
