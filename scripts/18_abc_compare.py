"""
Three-way comparison: Models A vs B vs C on four test sets.

Test sets:
  1. ORIGINALS:   close-mic chants from data/positive_raw_originals/ (no _ff)
                  Expected count per file: 1
  2. FAR-FIELD:   simulated far-field versions (_ff*) of the originals
                  Expected count per file: 1
  3. PARTIALS:    partial chants from data/partial_chants_test/ (cut off before closing)
                  Expected count per file: 0
  4. SPEECH:      held-out Hindi/Gujarati speech from data/speech_negative_test/
                  Expected count per file: 0  (CRITICAL for real-world deployment)

Model naming:
  A = close-mic only (no far-field, no speech)
  B = close-mic + far-field (no speech in negatives)
  C = close-mic + far-field + Hindi/Gujarati speech negatives

Usage:
    python scripts/18_abc_compare.py <model_A.pt> <model_B.pt> <model_C.pt>
"""

import math
import sys
from importlib import import_module
from pathlib import Path

import librosa
import numpy as np
import torch

SAMPLE_RATE = 16000
N_SAMPLES = 40000
HOP_SEC = 0.25
THRESHOLD = 0.85
DEBOUNCE_SEC = 5.0


def extract_mfcc(audio):
    if len(audio) < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)), mode="constant")
    elif len(audio) > N_SAMPLES:
        audio = audio[:N_SAMPLES]
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=40,
                                 n_fft=512, hop_length=160, n_mels=40)
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
    return mfcc.T.astype(np.float32)


def score_file(model, device, path):
    audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    hop = int(HOP_SEC * SAMPLE_RATE)
    starts = list(range(0, max(1, len(audio) - N_SAMPLES + 1), hop)) or [0]
    feats = np.array([extract_mfcc(audio[s:s+N_SAMPLES]) for s in starts])
    X = torch.from_numpy(feats[:, np.newaxis, :, :]).to(device)
    with torch.no_grad():
        scores = torch.sigmoid(model(X)).cpu().numpy()
    return np.array([s / SAMPLE_RATE for s in starts]), scores


def count(times, scores, thr=THRESHOLD, deb=DEBOUNCE_SEC):
    c = 0; last = -1e9
    for t, s in zip(times, scores):
        if s >= thr and t - last >= deb:
            c += 1; last = t
    return c


def evaluate(model, device, files, expected):
    correct = 0
    over = 0
    miss = 0
    for f in files:
        t, s = score_file(model, device, f)
        c = count(t, s)
        if c == expected: correct += 1
        elif c > expected: over += 1
        else: miss += 1
    return {"correct": correct, "over": over, "miss": miss, "total": len(files)}


def main():
    if len(sys.argv) < 4:
        print("Usage: python scripts/18_abc_compare.py <A.pt> <B.pt> <C.pt>")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sys.path.insert(0, str(Path(__file__).parent))
    ClosingDetector = import_module("05d_train_closing").ClosingDetector

    def load(p):
        m = ClosingDetector(dropout=0.0).to(device).eval()
        m.load_state_dict(torch.load(p, map_location=device, weights_only=True))
        return m

    models = {}
    for label, path in zip("ABC", sys.argv[1:4]):
        models[label] = load(path)
        print(f"Model {label}: {path}")

    # Build the four test sets
    originals = sorted([f for f in Path("data/positive_raw_originals").glob("real_navkar_*.wav")
                        if "_ff" not in f.stem])
    farfield = sorted(Path("data/positive_raw_originals").glob("real_navkar_*_ff*.wav"))
    partials = sorted(Path("data/partial_chants_test").glob("*.wav"))
    speech = sorted(Path("data/speech_negative_test").glob("*.wav"))

    sets = [
        ("ORIGINALS (count=1)", originals, 1),
        ("FAR-FIELD (count=1)", farfield, 1),
        ("PARTIALS  (count=0)", partials, 0),
        ("SPEECH    (count=0)", speech, 0),
    ]
    print(f"\nTest sets: {len(originals)} orig, {len(farfield)} ff, {len(partials)} part, {len(speech)} speech")
    print(f"Settings: threshold={THRESHOLD}, debounce={DEBOUNCE_SEC}s")

    # Evaluate everything
    results = {label: {} for label in "ABC"}
    for set_name, files, exp in sets:
        if not files:
            print(f"\n[Skipping {set_name}: no files]")
            continue
        print(f"\nScoring {set_name} ({len(files)} files)...")
        for label in "ABC":
            results[label][set_name] = evaluate(models[label], device, files, exp)

    # Print comparison table
    print("\n" + "=" * 90)
    print(f"{'Test set':<25s} {'A correct':>12s} {'B correct':>12s} {'C correct':>12s} {'C - B':>8s}")
    print("-" * 90)
    grand = {"A": 0, "B": 0, "C": 0}
    grand_total = 0
    for set_name, files, exp in sets:
        if not files:
            continue
        line = f"{set_name:<25s}"
        ra = results["A"][set_name]
        rb = results["B"][set_name]
        rc = results["C"][set_name]
        for label in "ABC":
            r = results[label][set_name]
            line += f" {r['correct']:>4}/{r['total']:<6}"
            grand[label] += r['correct']
        diff = rc['correct'] - rb['correct']
        line += f"   {'+' if diff >= 0 else ''}{diff:>4}"
        grand_total += ra['total']
        print(line)
    print("-" * 90)
    line = f"{'TOTAL':<25s}"
    for label in "ABC":
        line += f" {grand[label]:>4}/{grand_total:<6}"
    diff_total = grand['C'] - grand['B']
    line += f"   {'+' if diff_total >= 0 else ''}{diff_total:>4}"
    print(line)

    # Detailed breakdown per model on speech (the new test)
    print("\n" + "=" * 90)
    print("DETAILED: misses/over-counts per model per set")
    print("-" * 90)
    print(f"{'Test set':<25s} {'A (mis/ovr)':>15s} {'B (mis/ovr)':>15s} {'C (mis/ovr)':>15s}")
    for set_name, files, exp in sets:
        if not files:
            continue
        line = f"{set_name:<25s}"
        for label in "ABC":
            r = results[label][set_name]
            line += f"  {r['miss']:>3}/{r['over']:<3}      "
        print(line)

    # Recommendation
    print("\n" + "=" * 90)
    print("=== Recommendation ===")
    print(f"Best score: ", end="")
    best_label = max("ABC", key=lambda l: grand[l])
    print(f"Model {best_label} ({grand[best_label]}/{grand_total})")
    if grand['C'] > grand['B']:
        print(f"Model C beats B by {grand['C'] - grand['B']} -- speech negatives pay off.")
    elif grand['C'] == grand['B']:
        print(f"Models B and C tie. Prefer C for real-world robustness (it's seen speech).")
    else:
        print(f"Model B beats C. Speech negatives may have hurt rather than helped.")


if __name__ == "__main__":
    main()
