"""
A/B comparison of closing-phrase models on three test sets.

Test sets:
  1. ORIGINALS: data/positive_raw_originals/real_navkar_*.wav (no _ff)
                Expected count per file: 1 (each is one Navkar chant)
  2. PARTIALS:  data/partial_chants_test/*.wav
                Expected count per file: 0 (each is an incomplete chant; closing
                phrase was cut off, so the model should NOT fire)
  3. FARFIELD:  data/positive_raw_originals/real_navkar_*_ff*.wav
                Expected count per file: 1 (each is a far-field version of a chant)

Compares: per-test-set accuracy, false-positive rate on partials, far-field
accuracy. Helps decide whether to ship Model A (close-mic only) or Model B
(close-mic + far-field augmented).

Usage:
    python scripts/15_ab_compare_models.py \\
        models/closing_v1_<ts_A>/best.pt \\
        models/closing_v1_<ts_B>/best.pt
"""

import math
import sys
from importlib import import_module
from pathlib import Path

import librosa
import numpy as np
import torch

SAMPLE_RATE = 16000
N_SAMPLES = 40000  # 2.5s window
HOP_SEC = 0.25
THRESHOLD = 0.85
DEBOUNCE_SEC = 5.0


def extract_mfcc(audio: np.ndarray) -> np.ndarray:
    if len(audio) < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)), mode="constant")
    elif len(audio) > N_SAMPLES:
        audio = audio[:N_SAMPLES]
    mfcc = librosa.feature.mfcc(
        y=audio, sr=SAMPLE_RATE,
        n_mfcc=40, n_fft=512, hop_length=160, n_mels=40,
    )
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
    return mfcc.T.astype(np.float32)


def score_file(model, device, audio_path: Path):
    """Slide a 2.5s window with 250ms hop, return list of (time, score)."""
    audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    hop_samples = int(HOP_SEC * SAMPLE_RATE)
    starts = list(range(0, max(1, len(audio) - N_SAMPLES + 1), hop_samples))
    if not starts:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)), mode="constant")
        starts = [0]

    feats = np.array([extract_mfcc(audio[s : s + N_SAMPLES]) for s in starts])
    X = torch.from_numpy(feats[:, np.newaxis, :, :]).to(device)
    with torch.no_grad():
        scores = torch.sigmoid(model(X)).cpu().numpy()
    times = np.array([s / SAMPLE_RATE for s in starts])
    return times, scores


def count(times, scores, threshold, debounce):
    c = 0
    last = -1e9
    for t, s in zip(times, scores):
        if s >= threshold and (t - last) >= debounce:
            c += 1
            last = t
    return c


def evaluate(model, device, files, expected_count: int):
    """Return per-file dict + summary stats."""
    results = []
    for f in files:
        times, scores = score_file(model, device, f)
        c = count(times, scores, THRESHOLD, DEBOUNCE_SEC)
        max_score = float(scores.max()) if len(scores) > 0 else 0.0
        results.append({
            "file": f.stem,
            "detected": c,
            "expected": expected_count,
            "max_score": max_score,
            "correct": c == expected_count,
        })
    return results


def summarize(name: str, results: list[dict], expected: int) -> dict:
    correct = sum(r["correct"] for r in results)
    total = len(results)
    avg_max = np.mean([r["max_score"] for r in results]) if results else 0
    misses = sum(1 for r in results if r["detected"] < expected)
    extras = sum(1 for r in results if r["detected"] > expected)
    return {
        "name": name,
        "correct": correct,
        "total": total,
        "pct": correct / total * 100 if total else 0,
        "misses": misses,
        "extras": extras,
        "avg_max_score": avg_max,
    }


def print_summary_table(name: str, sa: dict, sb: dict):
    print(f"\n{'=' * 65}")
    print(f"  {name}")
    print(f"{'=' * 65}")
    print(f"  {'metric':<22s} {'Model A':>12s} {'Model B':>12s} {'B - A':>10s}")
    print(f"  {'-' * 60}")
    for key, label in [
        ("correct", "Correct"),
        ("misses", "Misses (count low)"),
        ("extras", "Over-counts"),
        ("avg_max_score", "Avg max score"),
    ]:
        a = sa[key]; b = sb[key]
        if isinstance(a, float):
            diff_str = f"{b - a:+.3f}"
            a_str = f"{a:.3f}"; b_str = f"{b:.3f}"
        else:
            diff_str = f"{b - a:+d}"
            a_str = str(a); b_str = str(b)
        print(f"  {label:<22s} {a_str:>12s} {b_str:>12s} {diff_str:>10s}")
    pct_diff = sb["pct"] - sa["pct"]
    print(f"  {'Pass rate':<22s} {sa['pct']:>11.1f}% {sb['pct']:>11.1f}% {pct_diff:>+9.1f}%")


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python scripts/15_ab_compare_models.py <model_A.pt> <model_B.pt>")
        sys.exit(1)

    model_a_path = sys.argv[1]
    model_b_path = sys.argv[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sys.path.insert(0, str(Path(__file__).parent))
    ClosingDetector = import_module("05d_train_closing").ClosingDetector

    def load(p):
        m = ClosingDetector(dropout=0.0).to(device).eval()
        m.load_state_dict(torch.load(p, map_location=device, weights_only=True))
        return m

    model_a = load(model_a_path)
    model_b = load(model_b_path)
    print(f"Model A: {model_a_path}")
    print(f"Model B: {model_b_path}")

    # Test sets
    originals = sorted([
        f for f in Path("data/positive_raw_originals").glob("real_navkar_*.wav")
        if "_ff" not in f.stem
    ])
    partials = sorted(Path("data/partial_chants_test").glob("*.wav"))
    farfields = sorted([
        f for f in Path("data/positive_raw_originals").glob("real_navkar_*_ff*.wav")
    ])

    print(f"\nTest sets: {len(originals)} originals, {len(partials)} partials, "
          f"{len(farfields)} far-field samples")

    # --- Run each model on each set ---
    print("\nScoring originals...")
    a_orig = evaluate(model_a, device, originals, expected_count=1)
    b_orig = evaluate(model_b, device, originals, expected_count=1)

    print("Scoring partials...")
    a_part = evaluate(model_a, device, partials, expected_count=0)
    b_part = evaluate(model_b, device, partials, expected_count=0)

    print("Scoring far-field samples...")
    a_ff = evaluate(model_a, device, farfields, expected_count=1)
    b_ff = evaluate(model_b, device, farfields, expected_count=1)

    # --- Summaries ---
    print_summary_table("ORIGINALS  (expected count = 1 each)",
                        summarize("A", a_orig, 1), summarize("B", b_orig, 1))
    print_summary_table("PARTIALS  (expected count = 0; hard negatives)",
                        summarize("A", a_part, 0), summarize("B", b_part, 0))
    print_summary_table("FAR-FIELD  (expected count = 1 each)",
                        summarize("A", a_ff, 1), summarize("B", b_ff, 1))

    # --- Failure detail for partials (the most important test) ---
    print("\n\n=== Partial chants: which files triggered false positives? ===")
    for tag, results in [("Model A", a_part), ("Model B", b_part)]:
        fails = [r for r in results if r["detected"] > 0]
        print(f"\n{tag}: {len(fails)} false-positive files")
        for r in fails:
            print(f"  - {r['file']:<35s} detected={r['detected']}  max_score={r['max_score']:.3f}")

    # --- Final recommendation ---
    print("\n\n=== Recommendation ===")
    orig_a = summarize("A", a_orig, 1)
    orig_b = summarize("B", b_orig, 1)
    part_a = summarize("A", a_part, 0)
    part_b = summarize("B", b_part, 0)
    ff_a = summarize("A", a_ff, 1)
    ff_b = summarize("B", b_ff, 1)

    score_a = orig_a["correct"] + part_a["correct"] + ff_a["correct"]
    score_b = orig_b["correct"] + part_b["correct"] + ff_b["correct"]
    total = orig_a["total"] + part_a["total"] + ff_a["total"]
    print(f"  Total correct (out of {total}):  A={score_a}, B={score_b}")
    if score_b > score_a:
        print(f"  -> Model B wins by {score_b - score_a}. Ship it.")
    elif score_a > score_b:
        print(f"  -> Model A wins by {score_a - score_b}. Ship it.")
    else:
        print(f"  -> Tie. Prefer Model B (more robust to far-field by design).")


if __name__ == "__main__":
    main()
