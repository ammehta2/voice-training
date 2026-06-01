"""
Day 10 — Real-world validation on your own recordings.

Drop your recordings into:
  data/real_world_test/positive/   <- you chanting Navkar in different conditions
  data/real_world_test/negative/   <- other sounds, other mantras, speech, etc.

Suggested 10 positives + 10 negatives covering different conditions.

Usage:
    python scripts/07_real_world_test.py models/navkar_v1_<TIMESTAMP>/best.keras
"""

import sys
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf

# Feature parameters — must match scripts/04_extract_features.py exactly
SAMPLE_RATE = 16000
DURATION_SEC = 16
N_SAMPLES = SAMPLE_RATE * DURATION_SEC
N_MFCC = 40
N_FFT = 512
HOP_LENGTH = 160
N_MELS = 40

DETECTION_THRESHOLD = 0.85


def extract_mfcc(audio_path: str) -> np.ndarray:
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
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
    return mfcc.T.astype(np.float32)[np.newaxis, ..., np.newaxis]


def evaluate_directory(model, directory: Path, label_positive: bool) -> tuple[int, int]:
    """Returns (correct, total)."""
    files = sorted(directory.glob("*.wav"))
    if not files:
        print(f"  (no files in {directory})")
        return 0, 0

    correct = 0
    for audio in files:
        features = extract_mfcc(str(audio))
        score = float(model.predict(features, verbose=0)[0][0])
        detected = score >= DETECTION_THRESHOLD

        if label_positive:
            status = "OK  DETECTED" if detected else "MISS"
        else:
            status = "FALSE POS" if detected else "OK  rejected"

        is_correct = detected == label_positive
        correct += int(is_correct)
        marker = " " if is_correct else "*"
        print(f"  {marker} {audio.name:50s} score={score:.3f}  {status}")

    return correct, len(files)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/07_real_world_test.py <path/to/model.keras>")
        sys.exit(1)

    model_path = sys.argv[1]
    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    pos_dir = Path("data/real_world_test/positive")
    neg_dir = Path("data/real_world_test/negative")

    print("\n" + "=" * 70)
    print(f"POSITIVE SAMPLES (should detect; threshold={DETECTION_THRESHOLD})")
    print("=" * 70)
    pos_correct, pos_total = evaluate_directory(model, pos_dir, label_positive=True)

    print("\n" + "=" * 70)
    print(f"NEGATIVE SAMPLES (should NOT detect; threshold={DETECTION_THRESHOLD})")
    print("=" * 70)
    neg_correct, neg_total = evaluate_directory(model, neg_dir, label_positive=False)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if pos_total:
        print(f"  Positive detection rate: {pos_correct}/{pos_total} = {pos_correct/pos_total:.1%}")
    if neg_total:
        print(f"  Negative rejection rate: {neg_correct}/{neg_total} = {neg_correct/neg_total:.1%}")


if __name__ == "__main__":
    main()
