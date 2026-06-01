"""
Verify the raw-audio TFLite counter on all 15 source recordings.
Mirror of scripts/10_sweep_counter.py but uses the bake-in TFLite (FP32)
that takes raw audio instead of pre-computed MFCC features.
"""

import math
import sys
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf

SAMPLE_RATE = 16000
N_SAMPLES = 40000
HOP_SEC = 0.25
THRESHOLD = 0.85
DEBOUNCE_SEC = 5.0


def score_file(interp, in_d, out_d, audio_path: Path):
    audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    hop_samples = int(HOP_SEC * SAMPLE_RATE)
    starts = list(range(0, max(1, len(audio) - N_SAMPLES + 1), hop_samples))
    if not starts:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)), mode="constant")
        starts = [0]

    times = []
    scores = []
    for s in starts:
        chunk = audio[s : s + N_SAMPLES]
        if len(chunk) < N_SAMPLES:
            chunk = np.pad(chunk, (0, N_SAMPLES - len(chunk)), mode="constant")
        chunk_2d = chunk.astype(np.float32).reshape(1, N_SAMPLES)
        interp.set_tensor(in_d["index"], chunk_2d)
        interp.invoke()
        logit = float(interp.get_tensor(out_d["index"]).flatten()[0])
        prob = 1.0 / (1.0 + math.exp(-logit))
        times.append(s / SAMPLE_RATE)
        scores.append(prob)
    return np.array(times), np.array(scores)


def count(times, scores, threshold, debounce):
    c = 0
    last = -1e9
    for t, s in zip(times, scores):
        if s >= threshold and (t - last) >= debounce:
            c += 1
            last = t
    return c


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/12_test_raw_audio_counter.py <path/to/navkar_raw_audio_fp32.tflite>")
        sys.exit(1)
    tflite_path = sys.argv[1]
    print(f"Model: {tflite_path}")

    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]

    files = sorted(Path("data/positive_raw_originals").glob("*.wav"))
    print(f"\n=== Counter results at threshold={THRESHOLD}, debounce={DEBOUNCE_SEC}s ===")
    print(f"{'file':<22s} {'detected':>9s} {'expected':>9s} {'max_score':>10s} status")
    print("-" * 65)
    correct = 0
    for f in files:
        t, s = score_file(interp, in_d, out_d, f)
        c = count(t, s, THRESHOLD, DEBOUNCE_SEC)
        e = 1  # each recording = 1 chant by user's spec
        status = "OK " if c == e else ("OVR" if c > e else "MIS")
        if c == e:
            correct += 1
        print(f"  {f.stem:<22s} {c:>9d} {e:>9d} {float(s.max()):>10.3f}  {status}")
    print("-" * 65)
    print(f"Correct: {correct}/{len(files)}")


if __name__ == "__main__":
    main()
