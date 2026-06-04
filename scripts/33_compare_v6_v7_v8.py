"""
Comprehensive head-to-head: v6 vs v7 vs v8 on 3 test sets.

Test sets:
  1. Hard negatives (data/hard_negatives/)             ~260 files
  2. Speech negatives (data/speech_negative_test/)      215 files
  3. Real-world app recordings (data/real_world_test_v2/) 11 files

Reports per-set FP rate + per-real-recording detection.
Usage:
    python scripts/33_compare_v6_v7_v8.py --v8 models/multi_v1_<timestamp>/navkar_multi_phrase_fp32.tflite
"""
import argparse
import csv
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 40000
HOP_SAMPLES = 4000
THRESHOLD = 0.5

V6_TFLITE = Path("models/closing_v1_20260601_163945/navkar_raw_audio_fp32.tflite")
V7_TFLITE = Path("models/multi_v1_20260601_181227/navkar_multi_phrase_fp32.tflite")

HARD_NEG_ROOT = Path("data/hard_negatives")
SPEECH_NEG_DIR = Path("data/speech_negative_test")
REAL_TEST_ROOT = Path("data/real_world_test_v2")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def score_one(audio, interp, in_det, out_det, is_multi):
    if len(audio) < WINDOW_SAMPLES:
        audio = np.pad(audio, (0, WINDOW_SAMPLES - len(audio)))
    starts = list(range(0, max(1, len(audio) - WINDOW_SAMPLES + 1), HOP_SAMPLES))
    if not starts or starts[-1] + WINDOW_SAMPLES < len(audio):
        starts.append(max(0, len(audio) - WINDOW_SAMPLES))
    best_a = 0.0; best_b = 0.0
    for s in starts:
        win = audio[s:s + WINDOW_SAMPLES].astype(np.float32).reshape(1, -1)
        interp.set_tensor(in_det["index"], win); interp.invoke()
        out = interp.get_tensor(out_det["index"])[0]
        if is_multi:
            best_a = max(best_a, float(sigmoid(out[0])))
            best_b = max(best_b, float(sigmoid(out[1])))
        else:
            best_a = max(best_a, float(sigmoid(out[0])))
    return best_a, best_b


def make_interp(path):
    interp = tf.lite.Interpreter(model_path=str(path)); interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    out_shape = out_det["shape"]
    is_multi = out_shape[-1] == 2
    return interp, in_det, out_det, is_multi


def evaluate(label, file_iter, v6, v7, v8):
    v6_i, v6_in, v6_out, v6_multi = v6
    v7_i, v7_in, v7_out, v7_multi = v7
    v8_i, v8_in, v8_out, v8_multi = v8

    fp_v6 = 0
    fp_v7_close = 0; fp_v7_pre = 0; fp_v7_either = 0
    fp_v8_close = 0; fp_v8_pre = 0; fp_v8_either = 0
    total = 0
    rows = []

    for f in file_iter:
        audio, _ = librosa.load(str(f), sr=SAMPLE_RATE, mono=True)
        v6_s, _ = score_one(audio, v6_i, v6_in, v6_out, v6_multi)
        v7_c, v7_p = score_one(audio, v7_i, v7_in, v7_out, v7_multi)
        v8_c, v8_p = score_one(audio, v8_i, v8_in, v8_out, v8_multi)
        rows.append((f.name, v6_s, v7_c, v7_p, v8_c, v8_p))
        total += 1
        if v6_s >= THRESHOLD: fp_v6 += 1
        if v7_c >= THRESHOLD: fp_v7_close += 1
        if v7_p >= THRESHOLD: fp_v7_pre += 1
        if v7_c >= THRESHOLD or v7_p >= THRESHOLD: fp_v7_either += 1
        if v8_c >= THRESHOLD: fp_v8_close += 1
        if v8_p >= THRESHOLD: fp_v8_pre += 1
        if v8_c >= THRESHOLD or v8_p >= THRESHOLD: fp_v8_either += 1

    return {
        "label": label, "total": total,
        "v6": fp_v6,
        "v7_close": fp_v7_close, "v7_pre": fp_v7_pre, "v7_either": fp_v7_either,
        "v8_close": fp_v8_close, "v8_pre": fp_v8_pre, "v8_either": fp_v8_either,
        "rows": rows,
    }


def fmt(n, total): return f"{n}/{total}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v8", required=True, help="Path to v8 .tflite")
    args = ap.parse_args()

    v6 = make_interp(V6_TFLITE)
    v7 = make_interp(V7_TFLITE)
    v8 = make_interp(Path(args.v8))
    print(f"v6: {V6_TFLITE}")
    print(f"v7: {V7_TFLITE}")
    print(f"v8: {args.v8}\n")

    results = []

    # 1. Hard negatives (per category)
    print("=" * 80)
    print("HARD NEGATIVES (false positives = bad)")
    print("=" * 80)
    cats = sorted([d for d in HARD_NEG_ROOT.iterdir() if d.is_dir()])
    for cat in cats:
        r = evaluate(f"hard:{cat.name}", cat.glob("*.wav"), v6, v7, v8)
        results.append(r)

    print(f"{'category':<28} {'n':>4} {'v6':>10} {'v7c':>10} {'v7p':>10} {'v7E':>10} {'v8c':>10} {'v8p':>10} {'v8E':>10}")
    for r in results:
        print(f"{r['label']:<28} {r['total']:>4} "
              f"{fmt(r['v6'], r['total']):>10} "
              f"{fmt(r['v7_close'], r['total']):>10} {fmt(r['v7_pre'], r['total']):>10} {fmt(r['v7_either'], r['total']):>10} "
              f"{fmt(r['v8_close'], r['total']):>10} {fmt(r['v8_pre'], r['total']):>10} {fmt(r['v8_either'], r['total']):>10}")
    # Total
    tot = {k: sum(r[k] for r in results) for k in ["total", "v6", "v7_close", "v7_pre", "v7_either", "v8_close", "v8_pre", "v8_either"]}
    print(f"{'TOTAL':<28} {tot['total']:>4} "
          f"{fmt(tot['v6'], tot['total']):>10} "
          f"{fmt(tot['v7_close'], tot['total']):>10} {fmt(tot['v7_pre'], tot['total']):>10} {fmt(tot['v7_either'], tot['total']):>10} "
          f"{fmt(tot['v8_close'], tot['total']):>10} {fmt(tot['v8_pre'], tot['total']):>10} {fmt(tot['v8_either'], tot['total']):>10}")

    # 2. Speech negatives
    print("\n" + "=" * 80)
    print("SPEECH NEGATIVES (Hindi/Gujarati/Marathi conversation; FP = bad)")
    print("=" * 80)
    rsp = evaluate("speech_neg", SPEECH_NEG_DIR.glob("*.wav"), v6, v7, v8)
    print(f"  v6 FP:        {fmt(rsp['v6'], rsp['total'])}")
    print(f"  v7 close FP:  {fmt(rsp['v7_close'], rsp['total'])}")
    print(f"  v7 pre FP:    {fmt(rsp['v7_pre'], rsp['total'])}")
    print(f"  v7 either FP: {fmt(rsp['v7_either'], rsp['total'])}")
    print(f"  v8 close FP:  {fmt(rsp['v8_close'], rsp['total'])}")
    print(f"  v8 pre FP:    {fmt(rsp['v8_pre'], rsp['total'])}")
    print(f"  v8 either FP: {fmt(rsp['v8_either'], rsp['total'])}")

    # 3. Real-world recordings (POSITIVES — we want them to count)
    print("\n" + "=" * 80)
    print("REAL-WORLD APP RECORDINGS (positives — these SHOULD detect)")
    print("=" * 80)
    real_files = []
    for sub in REAL_TEST_ROOT.iterdir():
        if sub.is_dir():
            real_files.extend(sorted(sub.glob("*.wav")))
    rr = evaluate("real_world", iter(real_files), v6, v7, v8)
    print(f"  {'file':<50} {'v6':>5} {'v7c':>5} {'v7p':>5} {'v8c':>5} {'v8p':>5}")
    for name, v6s, v7c, v7p, v8c, v8p in rr["rows"]:
        print(f"  {name[:50]:<50} {v6s:>5.2f} {v7c:>5.2f} {v7p:>5.2f} {v8c:>5.2f} {v8p:>5.2f}")
    print(f"  TOTAL detected (>=0.50): "
          f"v6={rr['v6']}/{rr['total']}  "
          f"v7c={rr['v7_close']}/{rr['total']}  v7E={rr['v7_either']}/{rr['total']}  "
          f"v8c={rr['v8_close']}/{rr['total']}  v8E={rr['v8_either']}/{rr['total']}")

    # CSV
    out_csv = Path("data/real_world_test_v2/v8_comparison.csv")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["set", "file", "v6", "v7_close", "v7_pre", "v8_close", "v8_pre"])
        for r in results + [rsp, rr]:
            for row in r["rows"]:
                w.writerow([r["label"], *row])
    print(f"\nSaved {out_csv}")


if __name__ == "__main__":
    main()
