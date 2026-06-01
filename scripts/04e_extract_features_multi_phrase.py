"""
Feature extraction for v7 multi-phrase model.

Reads from:
  data/multi/closing/positive_aug/      → label [1, 0]
  data/multi/preclosing/positive_aug/   → label [0, 1]
  data/multi/shared/negative_aug/       → label [0, 0]

Source-aware split: all variants (orig + augmentations + far-field versions)
of the same source recording stay in the same fold. Plus mid-mantra
negatives of a given parent recording stay with that recording's positives.

Output: data/multi/features/X_{train,val,test}.npy + y_{...}.npy
        y has shape (N, 2) — multi-label
"""

import random
import re
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

SAMPLE_RATE = 16000
N_SAMPLES = 40000
N_MFCC = 40
N_FFT = 512
HOP_LENGTH = 160
N_MELS = 40
SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

CLOSING_DIR = Path("data/multi/closing/positive_aug")
PRECLOSING_DIR = Path("data/multi/preclosing/positive_aug")
NEG_DIR = Path("data/multi/shared/negative_aug")
OUT = Path("data/multi/features")


def extract_mfcc(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if len(audio) < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)), mode="constant")
    elif len(audio) > N_SAMPLES:
        s = (len(audio) - N_SAMPLES) // 2
        audio = audio[s : s + N_SAMPLES]
    mfcc = librosa.feature.mfcc(
        y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
        n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
    )
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
    return mfcc.T.astype(np.float32)


def source_of(filename: str) -> str:
    """Group all augmentations/far-field variants of one source recording."""
    stem = Path(filename).stem
    stem = re.sub(r"_(aug\d+_(light|medium|heavy|bandwidth_limited)|orig)$", "", stem)
    stem = re.sub(r"_ff\d+_", "_", stem)
    # Strip _closing or _preclosing suffix so the two heads' positive sources
    # of the same recording match (and stay in the same split)
    stem = re.sub(r"_(closing|preclosing)$", "", stem)
    return stem


def reduce_negative_to_parent(s: str) -> str:
    """Group mid-mantra negatives by their parent recording."""
    m = re.match(r"^midmantra_([a-z_]+_[a-z0-9_]+?)(_\d+)?$", s)
    if m:
        return f"midmantra_{m.group(1)}"
    return s


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


def main() -> None:
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    # Group by source RECORDING (each recording has both closing + preclosing positives)
    closing_groups = group_by_source(CLOSING_DIR)
    preclosing_groups = group_by_source(PRECLOSING_DIR)
    neg_groups = group_by_source(NEG_DIR)

    # Union of positive source keys (each recording shows up in both)
    all_positive_sources = sorted(set(closing_groups) | set(preclosing_groups))
    print(f"Source recordings: {len(all_positive_sources)}")
    print(f"  closing groups: {len(closing_groups)}, preclosing groups: {len(preclosing_groups)}")

    # Negative grouping: midmantra by parent, ambient/speech each as own group
    bucketed_neg = defaultdict(list)
    for src_key, files in neg_groups.items():
        bucketed_neg[reduce_negative_to_parent(src_key)].extend(files)
    neg_groups = dict(bucketed_neg)
    print(f"  negative groups: {len(neg_groups)}")

    # Split POSITIVE recordings -- this is the most important split
    pos_train, pos_val, pos_test = split_sources(all_positive_sources, rng)
    print(f"\nPositive recordings: train={len(pos_train)}, val={len(pos_val)}, test={len(pos_test)}")

    # Tie midmantra negatives to the positive split they belong to
    def midmantra_key_for(src: str) -> str:
        return f"midmantra_{src}"

    tied_train = {midmantra_key_for(s) for s in pos_train}
    tied_val = {midmantra_key_for(s) for s in pos_val}
    tied_test = {midmantra_key_for(s) for s in pos_test}

    # Split AMBIENT + SPEECH negatives randomly
    other_neg = [k for k in neg_groups if not k.startswith("midmantra_")]
    amb_train, amb_val, amb_test = split_sources(other_neg, rng)

    neg_train = list(tied_train & set(neg_groups)) + list(amb_train)
    neg_val = list(tied_val & set(neg_groups)) + list(amb_val)
    neg_test = list(tied_test & set(neg_groups)) + list(amb_test)

    print(f"Negative groups: train={len(neg_train)}, val={len(neg_val)}, test={len(neg_test)}")

    # Build (file, label) lists per split
    def build_split(pos_keys, neg_keys):
        out = []
        # closing positives → label [1, 0]
        for k in pos_keys:
            for f in closing_groups.get(k, []):
                out.append((f, np.array([1.0, 0.0], dtype=np.float32)))
        # preclosing positives → label [0, 1]
        for k in pos_keys:
            for f in preclosing_groups.get(k, []):
                out.append((f, np.array([0.0, 1.0], dtype=np.float32)))
        # negatives → label [0, 0]
        for k in neg_keys:
            for f in neg_groups.get(k, []):
                out.append((f, np.array([0.0, 0.0], dtype=np.float32)))
        return out

    train_items = build_split(pos_train, neg_train)
    val_items = build_split(pos_val, neg_val)
    test_items = build_split(pos_test, neg_test)

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_items)}  "
          f"(closing={sum(1 for _,y in train_items if y[0]==1)}, "
          f"preclosing={sum(1 for _,y in train_items if y[1]==1)}, "
          f"neg={sum(1 for _,y in train_items if y[0]==0 and y[1]==0)})")
    print(f"  Val:   {len(val_items)}")
    print(f"  Test:  {len(test_items)}")

    # Extract MFCC + label arrays
    def extract_all(items, name):
        rng2 = np.random.RandomState(SEED)
        idx = list(range(len(items)))
        rng2.shuffle(idx)
        items = [items[i] for i in idx]

        X = np.zeros((len(items), 251, N_MFCC), dtype=np.float32)
        y = np.zeros((len(items), 2), dtype=np.float32)
        for i, (f, lbl) in enumerate(tqdm(items, desc=name)):
            try:
                X[i] = extract_mfcc(str(f))
                y[i] = lbl
            except Exception as e:
                print(f"    fail {f.name}: {e}")
        return X, y

    print("\nExtracting features...")
    X_train, y_train = extract_all(train_items, "train")
    X_val, y_val = extract_all(val_items, "val")
    X_test, y_test = extract_all(test_items, "test")

    np.save(OUT / "X_train.npy", X_train)
    np.save(OUT / "y_train.npy", y_train)
    np.save(OUT / "X_val.npy", X_val)
    np.save(OUT / "y_val.npy", y_val)
    np.save(OUT / "X_test.npy", X_test)
    np.save(OUT / "y_test.npy", y_test)
    print(f"\nSaved to {OUT}/")
    print(f"  y shape: {y_train.shape}  (N, 2 labels per sample)")


if __name__ == "__main__":
    main()
