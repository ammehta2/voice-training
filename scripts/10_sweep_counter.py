"""
Fast threshold/debounce sweep for the counter.

Loads the model ONCE, computes the score curve over each recording ONCE,
then applies different (threshold, debounce) combinations in memory.

Usage:
    python scripts/10_sweep_counter.py models/closing_v1_<TS>/best.pt
"""

import sys
from importlib import import_module
from pathlib import Path

import librosa
import numpy as np
import torch

SAMPLE_RATE = 16000
WINDOW_SEC = 2.5
N_SAMPLES = int(WINDOW_SEC * SAMPLE_RATE)
N_MFCC = 40
N_FFT = 512
HOP_LENGTH = 160
N_MELS = 40
HOP_SEC = 0.25  # 4 inferences/sec


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


def score_file(model, device, audio_path: Path, batch_size: int = 128):
    audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    hop_samples = int(HOP_SEC * SAMPLE_RATE)
    starts = list(range(0, max(1, len(audio) - N_SAMPLES + 1), hop_samples))
    if not starts:
        return np.array([]), np.array([])
    features = np.array([extract_mfcc(audio[s : s + N_SAMPLES]) for s in starts],
                        dtype=np.float32)[:, np.newaxis, :, :]
    X = torch.from_numpy(features).to(device)
    with torch.no_grad():
        scores = []
        for i in range(0, len(X), batch_size):
            scores.append(torch.sigmoid(model(X[i : i + batch_size])).cpu().numpy())
    scores = np.concatenate(scores)
    times = np.array(starts, dtype=np.float64) / SAMPLE_RATE
    return times, scores


def count_with_settings(times, scores, threshold, debounce):
    count = 0
    last_t = -1e9
    for t, s in zip(times, scores):
        if s >= threshold and (t - last_t) >= debounce:
            count += 1
            last_t = t
    return count


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/10_sweep_counter.py <model.pt>")
        sys.exit(1)
    model_path = sys.argv[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sys.path.insert(0, str(Path(__file__).parent))
    # Try ClosingDetector first (closing model), fall back to OpeningDetector
    try:
        Detector = import_module("05d_train_closing").ClosingDetector
    except (ImportError, AttributeError):
        Detector = import_module("05c_train_opening").OpeningDetector

    model = Detector(dropout=0.0).to(device).eval()
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    print(f"Model: {model_path}")

    files = sorted(Path("data/positive_raw_originals").glob("*.wav"))
    expected = {f.stem: 1 for f in files}  # each recording = 1 chant by user's spec

    print(f"\nScoring {len(files)} recordings (loading model once)...")
    cached = {}
    for f in files:
        times, scores = score_file(model, device, f)
        cached[f.stem] = (times, scores)

    print("\n=== Threshold × Debounce sweep ===")
    print(f"{'thresh':>7s} {'debounce':>9s} {'correct':>9s} {'over':>5s} {'miss':>5s} {'mae':>6s}")
    print("-" * 50)

    results = []
    for thresh in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        for debounce in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]:
            correct = over = miss = 0
            mae = 0
            for stem, (t, s) in cached.items():
                c = count_with_settings(t, s, thresh, debounce)
                e = expected[stem]
                mae += abs(c - e)
                if c == e:
                    correct += 1
                elif c > e:
                    over += 1
                else:
                    miss += 1
            results.append((thresh, debounce, correct, over, miss, mae))
            print(f"{thresh:>7.2f} {debounce:>9.1f} {correct:>9d} {over:>5d} {miss:>5d} {mae:>6d}")

    print()
    # Top 3 settings by correct count, breaking ties by lowest MAE
    results.sort(key=lambda r: (-r[2], r[5]))
    print("=== Top 3 settings (by correct count, then min MAE) ===")
    for i, (t, d, c, o, m, mae) in enumerate(results[:3]):
        print(f"  #{i+1}  threshold={t} debounce={d}s  ->  {c}/15 correct, {o} over, {m} miss, MAE={mae}")

    # Per-recording detail at the best settings
    best = results[0]
    bt, bd = best[0], best[1]
    print(f"\n=== Per-recording at threshold={bt}, debounce={bd}s ===")
    for stem, (t, s) in cached.items():
        c = count_with_settings(t, s, bt, bd)
        e = expected[stem]
        status = "OK " if c == e else ("OVR" if c > e else "MIS")
        max_score = float(s.max()) if len(s) else 0
        print(f"  {status} {stem:<25s} detected={c}  expected={e}  max_score={max_score:.3f}")


if __name__ == "__main__":
    main()
