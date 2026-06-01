"""
Feature extraction for the closing-phrase pipeline (2.5s window).

Source-aware split (same as 04b): all augmentations of a source recording
stay in one split. Files of the form `<src>_aug<N>_<sev>.wav` or `<src>_orig.wav`.

Output: data/closing/features/X_{train,val,test}.npy and labels.
"""

import random
import re
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

# 2.5s @ 16kHz → 40000 samples → ~251 MFCC frames @ hop 160
SAMPLE_RATE = 16000
DURATION_SEC = 2.5
N_SAMPLES = int(SAMPLE_RATE * DURATION_SEC)
N_MFCC = 40
N_FFT = 512
HOP_LENGTH = 160
N_MELS = 40

SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

POS_DIR = Path("data/closing/positive_aug")
NEG_DIR = Path("data/closing/negative_aug")
OUT = Path("data/closing/features")


def extract_mfcc(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if len(audio) < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)), mode="constant")
    elif len(audio) > N_SAMPLES:
        # center-crop
        s = (len(audio) - N_SAMPLES) // 2
        audio = audio[s : s + N_SAMPLES]
    mfcc = librosa.feature.mfcc(
        y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
        n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
    )
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
    return mfcc.T.astype(np.float32)


def source_of(filename: str) -> str:
    """Group all augmentations of a source together. Crucially, group far-field
    variants (`_ff01`, `_ff02`, ...) with their close-mic parent so the
    source-aware split keeps them in the same fold (avoids leakage).

    real_navkar_03_closing_aug05_medium.wav       -> real_navkar_03_closing
    real_navkar_03_closing_orig.wav               -> real_navkar_03_closing
    real_navkar_03_ff02_closing_aug05_medium.wav  -> real_navkar_03_closing
    premantra_real_navkar_01_000_aug03_light.wav  -> premantra_real_navkar_01_000
    ambient_1-100032-A-0_03_aug00_light.wav       -> ambient_1-100032-A-0_03
    """
    stem = Path(filename).stem
    # Strip augmentation suffix
    stem = re.sub(r"_(aug\d+_(light|medium|heavy)|orig)$", "", stem)
    # Strip far-field marker so _ff01_closing -> _closing
    stem = re.sub(r"_ff\d+_", "_", stem)
    return stem


def group_by_source(directory: Path) -> dict[str, list[Path]]:
    groups = defaultdict(list)
    for f in sorted(directory.glob("*.wav")):
        groups[source_of(f.name)].append(f)
    return dict(groups)


def split_sources(sources: list[str], rng: random.Random):
    sources = list(sources)
    rng.shuffle(sources)
    n = len(sources)
    n_train = max(1, int(n * TRAIN_FRAC))
    n_val = max(1, int(n * VAL_FRAC))
    n_test = max(1, n - n_train - n_val) if n >= 3 else 0
    while n_train + n_val + n_test > n:
        n_train -= 1
    return sources[:n_train], sources[n_train:n_train + n_val], sources[n_train + n_val:]


def files_for(groups: dict, sources: list[str]) -> list[Path]:
    out = []
    for s in sources:
        out.extend(groups[s])
    return out


def process_files(files: list[Path], label: float):
    X, y = [], []
    for f in tqdm(files, leave=False):
        try:
            X.append(extract_mfcc(str(f)))
            y.append(label)
        except Exception as e:
            print(f"    fail {f.name}: {e}")
    if not X:
        return np.zeros((0, 0, 0), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def main() -> None:
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    pos_groups = group_by_source(POS_DIR)
    neg_groups = group_by_source(NEG_DIR)
    print(f"Positive sources: {len(pos_groups)}  files={sum(len(v) for v in pos_groups.values())}")
    print(f"Negative sources: {len(neg_groups)}  files={sum(len(v) for v in neg_groups.values())}")

    # Group pre-closing negatives by parent recording so they stay together
    def reduce_to_parent(s: str) -> str:
        # premantra_real_navkar_01_023 → real_navkar_01  (matches both _01 and _14a/_14b)
        m = re.match(r"^premantra_(real_navkar_\d+[ab]?)", s)
        if m:
            return f"premantra_{m.group(1)}"
        return s

    bucketed_neg = defaultdict(list)
    for src_key, files in neg_groups.items():
        bucketed_neg[reduce_to_parent(src_key)].extend(files)
    neg_groups = dict(bucketed_neg)
    print(f"  After pre-closing grouping: {len(neg_groups)} negative source groups")

    # Split POSITIVES first (gives us train/val/test source recordings)
    pos_tr_src, pos_v_src, pos_te_src = split_sources(list(pos_groups), rng)

    # For each positive source, find its pre-closing negative key and FORCE it
    # into the same split. This avoids speaker leakage where opening of recording
    # N is in train but pre-closing of N is in test (or vice-versa).
    def closing_to_premantra(s: str) -> str:
        # real_navkar_01_closing → premantra_real_navkar_01
        m = re.match(r"^(real_navkar_\d+[ab]?)_closing", s)
        return f"premantra_{m.group(1)}" if m else None

    tied_train = {closing_to_premantra(s) for s in pos_tr_src} - {None}
    tied_val = {closing_to_premantra(s) for s in pos_v_src} - {None}
    tied_test = {closing_to_premantra(s) for s in pos_te_src} - {None}

    # Split AMBIENT negatives (everything that's NOT pre-closing) randomly
    ambient_keys = [k for k in neg_groups if not k.startswith("premantra_")]
    amb_tr, amb_v, amb_te = split_sources(ambient_keys, rng)

    # Combine: pre-closing tied to positive splits + ambient random split
    neg_tr_src = list(tied_train | set(amb_tr))
    neg_v_src = list(tied_val | set(amb_v))
    neg_te_src = list(tied_test | set(amb_te))
    print(f"  Tied pre-closing: train={len(tied_train)}, val={len(tied_val)}, test={len(tied_test)}")

    print(f"\nPositive source split:")
    print(f"  train: {pos_tr_src}")
    print(f"  val:   {pos_v_src}")
    print(f"  test:  {pos_te_src}")
    print(f"Negative source counts: train={len(neg_tr_src)}, val={len(neg_v_src)}, test={len(neg_te_src)}")

    print("\nExtracting features...")
    Xp_tr, yp_tr = process_files(files_for(pos_groups, pos_tr_src), 1.0)
    Xn_tr, yn_tr = process_files(files_for(neg_groups, neg_tr_src), 0.0)
    Xp_v, yp_v = process_files(files_for(pos_groups, pos_v_src), 1.0)
    Xn_v, yn_v = process_files(files_for(neg_groups, neg_v_src), 0.0)
    Xp_te, yp_te = process_files(files_for(pos_groups, pos_te_src), 1.0)
    Xn_te, yn_te = process_files(files_for(neg_groups, neg_te_src), 0.0)

    def combine_and_shuffle(Xp, yp, Xn, yn):
        X = np.concatenate([Xp, Xn], axis=0) if len(Xp) else Xn
        y = np.concatenate([yp, yn], axis=0) if len(yp) else yn
        if len(X) > 0:
            idx = np.arange(len(X))
            np.random.RandomState(SEED).shuffle(idx)
            X = X[idx]; y = y[idx]
        return X, y

    X_train, y_train = combine_and_shuffle(Xp_tr, yp_tr, Xn_tr, yn_tr)
    X_val, y_val = combine_and_shuffle(Xp_v, yp_v, Xn_v, yn_v)
    X_test, y_test = combine_and_shuffle(Xp_te, yp_te, Xn_te, yn_te)

    print(f"\nSplit sizes:")
    print(f"  Train: {len(X_train)}  ({int(y_train.sum())} pos / {int(len(y_train) - y_train.sum())} neg)")
    print(f"  Val:   {len(X_val)}  ({int(y_val.sum())} pos / {int(len(y_val) - y_val.sum())} neg)")
    print(f"  Test:  {len(X_test)}  ({int(y_test.sum())} pos / {int(len(y_test) - y_test.sum())} neg)")
    print(f"  Feature shape: {X_train[0].shape if len(X_train) else '?'}")

    np.save(OUT / "X_train.npy", X_train)
    np.save(OUT / "y_train.npy", y_train)
    np.save(OUT / "X_val.npy", X_val)
    np.save(OUT / "y_val.npy", y_val)
    np.save(OUT / "X_test.npy", X_test)
    np.save(OUT / "y_test.npy", y_test)
    print(f"Saved to {OUT}/")


if __name__ == "__main__":
    main()
