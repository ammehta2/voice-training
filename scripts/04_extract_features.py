"""
Day 6 — Extract MFCC features from augmented audio.

Outputs train/val/test numpy arrays in data/features/.
Feature parameters (SAMPLE_RATE, N_MFCC, etc.) MUST match the mobile
inference path exactly — they are duplicated as a constants block at the
top of this file so the mobile code can mirror them verbatim.

Usage:
    python scripts/04_extract_features.py
"""

from pathlib import Path

import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ==== Feature parameters — MUST match mobile inference ====
SAMPLE_RATE = 16000
DURATION_SEC = 16
N_SAMPLES = SAMPLE_RATE * DURATION_SEC  # 256,000

N_MFCC = 40
N_FFT = 512
HOP_LENGTH = 160  # 10ms hop @ 16kHz
N_MELS = 40
# ==========================================================


def extract_mfcc(audio_path: str) -> np.ndarray:
    """Load audio, pad/trim to fixed length, return MFCCs as (TIME, FEATURES)."""
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    if len(audio) < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)), mode="constant")
    elif len(audio) > N_SAMPLES:
        start = (len(audio) - N_SAMPLES) // 2
        audio = audio[start : start + N_SAMPLES]

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )

    # Per-utterance mean-variance normalization
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)

    # (n_mfcc, time) -> (time, n_mfcc) for CNN
    return mfcc.T.astype(np.float32)


def process_directory(directory: str, label: float):
    """Extract features from all wav files in a directory."""
    files = sorted(Path(directory).glob("*.wav"))
    print(f"  {directory}: {len(files)} files, label={label}")

    X, y = [], []
    for audio_file in tqdm(files, desc=Path(directory).name, leave=False):
        try:
            features = extract_mfcc(str(audio_file))
            X.append(features)
            y.append(label)
        except Exception as e:
            print(f"    Failed {audio_file.name}: {e}")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def main() -> None:
    output_dir = Path("data/features")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Extracting positive features (label=1.0)...")
    X_pos, y_pos = process_directory("data/positive_aug", label=1.0)

    print("\nExtracting negative features (label=0.0)...")
    X_neg, y_neg = process_directory("data/negative_aug", label=0.0)

    if len(X_pos) == 0 or len(X_neg) == 0:
        print("\nERROR: No features extracted. Did you run scripts 1-3 first?")
        return

    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([y_pos, y_neg], axis=0)
    print(f"\nTotal: {len(X)} samples, shape={X.shape}")
    print(f"  Positives: {int(sum(y))}, Negatives: {int(len(y) - sum(y))}")

    # 70 / 15 / 15 stratified split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    print(f"\n  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "X_val.npy", X_val)
    np.save(output_dir / "y_val.npy", y_val)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_test.npy", y_test)

    print(f"\nFeatures saved to {output_dir}")
    print(f"Feature shape per sample: {X_train[0].shape}")


if __name__ == "__main__":
    main()
