"""
Sliding-window jaap counter — runs the opening-phrase detector over a full
audio file and counts how many "Namo Arihantaanam" occurrences it found.

Usage:
    python scripts/09_count_jaap.py <model.pt> <audio.wav> [--threshold 0.85]

What it does:
  1. Loads the trained opening-phrase detector
  2. Slides a 2.5s window across the audio with HOP_SEC stride
  3. Runs the model on each window -> score 0..1
  4. Marks a "detection" when score crosses threshold (with a small debounce
     window to avoid double-counting the same opening across consecutive windows)
  5. Prints the count + timestamps of each detected start

For ground truth, each of your single-Navkar recordings should produce a count
of exactly 1.

Run it on a longer concatenated recording (multiple chants stitched together)
to verify the counter handles repetitions correctly.
"""

import argparse
import sys
from importlib import import_module
from pathlib import Path

import librosa
import numpy as np
import torch

# Match training-time feature params (04c_extract_features_opening.py)
SAMPLE_RATE = 16000
DURATION_SEC = 2.5
N_SAMPLES = int(SAMPLE_RATE * DURATION_SEC)
N_MFCC = 40
N_FFT = 512
HOP_LENGTH = 160
N_MELS = 40

HOP_SEC = 0.25       # how often to evaluate the model (4 times per second)
DEBOUNCE_SEC = 2.0   # min gap between two counted detections


def extract_mfcc(audio: np.ndarray) -> np.ndarray:
    if len(audio) < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)), mode="constant")
    elif len(audio) > N_SAMPLES:
        audio = audio[:N_SAMPLES]
    mfcc = librosa.feature.mfcc(
        y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
        n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
    )
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
    return mfcc.T.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to opening-phrase .pt model")
    parser.add_argument("audio_path", help="Path to wav file")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--debounce", type=float, default=DEBOUNCE_SEC)
    parser.add_argument("--hop", type=float, default=HOP_SEC)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    OpeningDetector = import_module("05c_train_opening").OpeningDetector
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = OpeningDetector(dropout=0.0).to(device).eval()
    state = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)

    audio, sr = librosa.load(args.audio_path, sr=SAMPLE_RATE, mono=True)
    duration = len(audio) / sr
    print(f"Audio: {args.audio_path} ({duration:.1f}s @ {sr}Hz)")
    print(f"Sliding {DURATION_SEC}s window, hop {args.hop}s, threshold {args.threshold}, debounce {args.debounce}s")
    print()

    # Generate all window MFCC features in a batch (faster than one-by-one)
    hop_samples = int(args.hop * sr)
    window_samples = N_SAMPLES
    starts = list(range(0, max(1, len(audio) - window_samples + 1), hop_samples))

    features = []
    for s in starts:
        chunk = audio[s : s + window_samples]
        features.append(extract_mfcc(chunk))
    X = np.array(features, dtype=np.float32)[:, np.newaxis, :, :]
    X_t = torch.from_numpy(X).to(device)

    with torch.no_grad():
        scores = torch.sigmoid(model(X_t)).cpu().numpy()

    # Count detections with debounce
    count = 0
    last_detection_time = -1e9
    detections = []

    print(f"{'time':>6s}  {'score':>6s}  {'event':<20s}")
    print("-" * 40)
    for s_idx, score in enumerate(scores):
        t = starts[s_idx] / sr
        is_above = score >= args.threshold
        if is_above and (t - last_detection_time) >= args.debounce:
            count += 1
            last_detection_time = t
            detections.append((t, float(score)))
            print(f"{t:6.2f}s  {score:6.3f}  *** DETECTED #{count} ***")
        elif is_above:
            # Above threshold but within debounce window — ignore
            pass
        elif score >= 0.5:
            # Borderline — log for visibility
            print(f"{t:6.2f}s  {score:6.3f}  (borderline)")

    print()
    print(f"=== Result: {count} Navkar(s) detected ===")
    for i, (t, s) in enumerate(detections, start=1):
        print(f"  #{i}  t={t:.2f}s  score={s:.3f}")


if __name__ == "__main__":
    main()
