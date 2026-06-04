"""
Score the adversarial hard-negative TTS samples against v6 + v7.

For each category, report:
  - How many files trigger v6 >= 0.50
  - How many trigger v7 closing >= 0.50
  - How many trigger v7 preclosing >= 0.50
  - Per-file detail for false positives so we can listen + understand
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
HARD_NEG_ROOT = Path("data/hard_negatives")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def score_file(audio: np.ndarray, v6, v7, in6, out6, in7, out7):
    if len(audio) < WINDOW_SAMPLES:
        audio = np.pad(audio, (0, WINDOW_SAMPLES - len(audio)))
    starts = list(range(0, max(1, len(audio) - WINDOW_SAMPLES + 1), HOP_SAMPLES))
    if not starts or starts[-1] + WINDOW_SAMPLES < len(audio):
        starts.append(max(0, len(audio) - WINDOW_SAMPLES))
    best_v6 = 0.0; best_c = 0.0; best_p = 0.0
    for s in starts:
        win = audio[s:s + WINDOW_SAMPLES].astype(np.float32).reshape(1, -1)
        v6.set_tensor(in6["index"], win); v6.invoke()
        best_v6 = max(best_v6, float(sigmoid(v6.get_tensor(out6["index"])[0, 0])))
        v7.set_tensor(in7["index"], win); v7.invoke()
        logits = v7.get_tensor(out7["index"])[0]
        best_c = max(best_c, float(sigmoid(logits[0])))
        best_p = max(best_p, float(sigmoid(logits[1])))
    return best_v6, best_c, best_p


def main():
    v6 = tf.lite.Interpreter(model_path=str(V6_TFLITE)); v6.allocate_tensors()
    v7 = tf.lite.Interpreter(model_path=str(V7_TFLITE)); v7.allocate_tensors()
    in6 = v6.get_input_details()[0]; out6 = v6.get_output_details()[0]
    in7 = v7.get_input_details()[0]; out7 = v7.get_output_details()[0]

    categories = sorted([d for d in HARD_NEG_ROOT.iterdir() if d.is_dir()])
    print(f"Scoring {len(categories)} categories under {HARD_NEG_ROOT}/\n")

    all_rows = []
    cat_summary = []

    for cat_dir in categories:
        files = sorted(cat_dir.glob("*.wav"))
        if not files:
            continue
        n_v6 = n_v7c = n_v7p = n_v7_either = 0
        rows = []
        for f in files:
            audio, _ = librosa.load(str(f), sr=SAMPLE_RATE, mono=True)
            v6_s, v7c_s, v7p_s = score_file(audio, v6, v7, in6, out6, in7, out7)
            rows.append((f.name, v6_s, v7c_s, v7p_s))
            if v6_s >= 0.5: n_v6 += 1
            if v7c_s >= 0.5: n_v7c += 1
            if v7p_s >= 0.5: n_v7p += 1
            if v7c_s >= 0.5 or v7p_s >= 0.5: n_v7_either += 1
            all_rows.append((cat_dir.name, f.name, v6_s, v7c_s, v7p_s))

        n = len(files)
        cat_summary.append((cat_dir.name, n, n_v6, n_v7c, n_v7p, n_v7_either))
        print(f"== {cat_dir.name} ({n} files) ==")
        for name, v6_s, v7c_s, v7p_s in sorted(rows, key=lambda r: -max(r[1], r[2], r[3])):
            mark_v6 = "FP!" if v6_s >= 0.5 else "   "
            mark_c = "FP!" if v7c_s >= 0.5 else "   "
            mark_p = "FP!" if v7p_s >= 0.5 else "   "
            print(f"  v6={v6_s:.2f}{mark_v6}  close={v7c_s:.2f}{mark_c}  pre={v7p_s:.2f}{mark_p}   {name[:55]}")
        print(f"  --> v6 FP {n_v6}/{n}  |  v7close FP {n_v7c}/{n}  |  v7pre FP {n_v7p}/{n}  |  v7-either FP {n_v7_either}/{n}\n")

    # Overall
    print("="*80)
    print("SUMMARY (% of files in category that produce false positive >= 0.50)")
    print("="*80)
    print(f"{'category':<22} {'n':>4} {'v6':>10} {'v7close':>10} {'v7pre':>10} {'v7either':>10}")
    for cat, n, v6, v7c, v7p, eit in cat_summary:
        print(f"{cat:<22} {n:>4} {f'{v6}/{n}':>10} {f'{v7c}/{n}':>10} {f'{v7p}/{n}':>10} {f'{eit}/{n}':>10}")

    # CSV
    out = Path("data/real_world_test_v2/hard_negative_scores.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category", "file", "v6", "v7_close", "v7_pre"])
        for row in all_rows:
            w.writerow([row[0], row[1], f"{row[2]:.4f}", f"{row[3]:.4f}", f"{row[4]:.4f}"])
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
