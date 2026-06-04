"""
Critical hard-negative test:
  How often does v7's either-head rule (close>=0.5 OR pre>=0.5) FIRE on
  Hindi/Gujarati/Marathi CONVERSATION audio?

If v7 either-head false-positive rate <= v6, v7 becomes the ship candidate.
If higher, we tighten the rule.
"""
import csv
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 40000
HOP_SAMPLES = 4000

V6_TFLITE = Path("models/closing_v1_20260601_163945/navkar_raw_audio_fp32.tflite")
V7_TFLITE = Path("models/multi_v1_20260601_181227/navkar_multi_phrase_fp32.tflite")
NEG_DIR = Path("data/speech_negative_test")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    v6 = tf.lite.Interpreter(model_path=str(V6_TFLITE)); v6.allocate_tensors()
    v7 = tf.lite.Interpreter(model_path=str(V7_TFLITE)); v7.allocate_tensors()
    in6 = v6.get_input_details()[0]; out6 = v6.get_output_details()[0]
    in7 = v7.get_input_details()[0]; out7 = v7.get_output_details()[0]

    files = sorted(NEG_DIR.glob("*.wav"))
    print(f"Testing {len(files)} speech-negative files against v6 + v7\n")

    n_v6_fp = 0          # v6 score >= 0.50 anywhere
    n_v7_close_fp = 0    # v7 closing alone >= 0.50
    n_v7_either_fp = 0   # v7 closing OR preclosing >= 0.50  (the candidate rule)
    n_v7_both_fp = 0     # v7 BOTH heads >= 0.50 in same window
    n_v7_strong_fp = 0   # v7 closing >= 0.80
    n_v7_either_07_fp = 0  # tightened rule: either head >= 0.70

    worst_v6 = []
    worst_v7_either = []
    worst_v7_pre = []

    for i, f in enumerate(files):
        audio, _ = librosa.load(str(f), sr=SAMPLE_RATE, mono=True)
        if len(audio) < WINDOW_SAMPLES:
            audio = np.pad(audio, (0, WINDOW_SAMPLES - len(audio)))

        starts = list(range(0, max(1, len(audio) - WINDOW_SAMPLES + 1), HOP_SAMPLES))
        if not starts or starts[-1] + WINDOW_SAMPLES < len(audio):
            starts.append(max(0, len(audio) - WINDOW_SAMPLES))

        best_v6 = 0.0
        best_close = 0.0
        best_pre = 0.0
        # For "both same window" rule
        both_same_window_max = 0.0
        # For temporal rule (pre 2-8s before close)
        had_temporal_match = False
        pre_times = []  # (time, score) for pre fires
        close_times = []

        for s in starts:
            win = audio[s:s + WINDOW_SAMPLES].astype(np.float32).reshape(1, -1)
            v6.set_tensor(in6["index"], win); v6.invoke()
            p6 = float(sigmoid(v6.get_tensor(out6["index"])[0, 0]))
            best_v6 = max(best_v6, p6)

            v7.set_tensor(in7["index"], win); v7.invoke()
            logits = v7.get_tensor(out7["index"])[0]
            pc = float(sigmoid(logits[0])); pp = float(sigmoid(logits[1]))
            best_close = max(best_close, pc)
            best_pre = max(best_pre, pp)
            both_same_window_max = max(both_same_window_max, min(pc, pp))
            if pc >= 0.5:
                close_times.append((s / SAMPLE_RATE, pc))
            if pp >= 0.5:
                pre_times.append((s / SAMPLE_RATE, pp))

        for ct, _ in close_times:
            for pt, _ in pre_times:
                if 0 < (ct - pt) <= 8.0:
                    had_temporal_match = True
                    break
            if had_temporal_match:
                break

        if best_v6 >= 0.50: n_v6_fp += 1
        if best_close >= 0.50: n_v7_close_fp += 1
        if best_close >= 0.50 or best_pre >= 0.50: n_v7_either_fp += 1
        if both_same_window_max >= 0.50: n_v7_both_fp += 1
        if best_close >= 0.80: n_v7_strong_fp += 1
        if best_close >= 0.70 or best_pre >= 0.70: n_v7_either_07_fp += 1

        worst_v6.append((best_v6, f.name))
        worst_v7_either.append((max(best_close, best_pre), f.name, best_close, best_pre))
        worst_v7_pre.append((best_pre, f.name))

        if (i + 1) % 25 == 0:
            print(f"  scored {i+1}/{len(files)}")

    n = len(files)
    print()
    print(f"FALSE POSITIVES (count above) out of {n} speech files")
    print(f"  v6 score >= 0.50:                              {n_v6_fp:>3}  ({100*n_v6_fp/n:.1f}%)  <- current ship")
    print(f"  v7 closing alone >= 0.50:                      {n_v7_close_fp:>3}  ({100*n_v7_close_fp/n:.1f}%)")
    print(f"  v7 EITHER head >= 0.50  (candidate rule):      {n_v7_either_fp:>3}  ({100*n_v7_either_fp/n:.1f}%)")
    print(f"  v7 BOTH heads >= 0.50 same window:             {n_v7_both_fp:>3}  ({100*n_v7_both_fp/n:.1f}%)")
    print(f"  v7 closing >= 0.80:                            {n_v7_strong_fp:>3}  ({100*n_v7_strong_fp/n:.1f}%)")
    print(f"  v7 EITHER head >= 0.70  (tightened):           {n_v7_either_07_fp:>3}  ({100*n_v7_either_07_fp/n:.1f}%)")

    print("\nTop 10 worst v6 false-positives (highest scoring speech files):")
    for s, n_ in sorted(worst_v6, reverse=True)[:10]:
        print(f"  {s:.3f}  {n_}")
    print("\nTop 10 worst v7-either false-positives:")
    for s, name, c, p in sorted(worst_v7_either, reverse=True)[:10]:
        print(f"  {s:.3f}  (close={c:.2f}, pre={p:.2f})  {name}")

    # Save
    out_csv = Path("data/real_world_test_v2/speech_negative_scores.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["file", "v6_max", "v7_close_max", "v7_pre_max"])
        # Re-collect
        for (v, name), (_, _, c, p), (pv, _) in zip(
            worst_v6, worst_v7_either, worst_v7_pre
        ):
            w.writerow([name, f"{v:.4f}", f"{c:.4f}", f"{p:.4f}"])
    print(f"\nSaved {out_csv}")


if __name__ == "__main__":
    main()
