"""
v8 threshold sweep: find the (close_thresh, pre_thresh) combination that:
  1. Keeps hard-negative FP at 0 (or near 0)
  2. Keeps speech-negative FP at 0 (or near 0)
  3. Maximizes real-world recording detection

Rule under test:
    COUNT if close >= CLOSE_THRESH OR pre >= PRE_THRESH

We sweep CLOSE_THRESH in [0.30, 0.40, 0.50, 0.70] and
       PRE_THRESH   in [0.50, 0.70, 0.85, 0.90, 0.95, 0.99]
and report the Pareto frontier.

We also try the "guarded both" rule:
    COUNT if close >= 0.50 OR (close >= close_low AND pre >= pre_low)
to see if requiring SOMETHING from the closing head reduces FPs without
sacrificing real-world detection too much.
"""
import csv
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 40000
HOP_SAMPLES = 4000

V8 = Path("models/multi_v1_20260604_091812/navkar_multi_phrase_fp32.tflite")
HARD_NEG_ROOT = Path("data/hard_negatives")
SPEECH_NEG = Path("data/speech_negative_test")
REAL_ROOT = Path("data/real_world_test_v2")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def best_scores(audio, interp, in_det, out_det):
    if len(audio) < WINDOW_SAMPLES:
        audio = np.pad(audio, (0, WINDOW_SAMPLES - len(audio)))
    starts = list(range(0, max(1, len(audio) - WINDOW_SAMPLES + 1), HOP_SAMPLES))
    if not starts or starts[-1] + WINDOW_SAMPLES < len(audio):
        starts.append(max(0, len(audio) - WINDOW_SAMPLES))
    best_c, best_p = 0.0, 0.0
    # Also track best joint (max over windows of min(close, pre)) — for "both same window" rule
    best_both = 0.0
    for s in starts:
        win = audio[s:s + WINDOW_SAMPLES].astype(np.float32).reshape(1, -1)
        interp.set_tensor(in_det["index"], win); interp.invoke()
        out = interp.get_tensor(out_det["index"])[0]
        c = float(sigmoid(out[0])); p = float(sigmoid(out[1]))
        best_c = max(best_c, c); best_p = max(best_p, p)
        best_both = max(best_both, min(c, p))
    return best_c, best_p, best_both


def collect_scores(file_iter, interp, in_det, out_det):
    out = []
    for f in file_iter:
        audio, _ = librosa.load(str(f), sr=SAMPLE_RATE, mono=True)
        c, p, b = best_scores(audio, interp, in_det, out_det)
        out.append((f.name, c, p, b))
    return out


def main():
    interp = tf.lite.Interpreter(model_path=str(V8)); interp.allocate_tensors()
    in_det = interp.get_input_details()[0]; out_det = interp.get_output_details()[0]

    print("Collecting v8 scores on all test sets...\n")
    hn = []
    for cat in sorted([d for d in HARD_NEG_ROOT.iterdir() if d.is_dir()]):
        hn += collect_scores(cat.glob("*.wav"), interp, in_det, out_det)
    print(f"  hard negatives:    {len(hn)}")

    sn = collect_scores(SPEECH_NEG.glob("*.wav"), interp, in_det, out_det)
    print(f"  speech negatives:  {len(sn)}")

    real = []
    for sub in REAL_ROOT.iterdir():
        if sub.is_dir():
            real += collect_scores(sub.glob("*.wav"), interp, in_det, out_det)
    print(f"  real recordings:   {len(real)}")
    print()

    def count_or(scores, c_thresh, p_thresh):
        return sum(1 for _, c, p, _ in scores if c >= c_thresh or p >= p_thresh)

    def count_or_guarded(scores, c_thresh, c_low, p_low):
        """close >= c_thresh OR (close >= c_low AND pre >= p_low)"""
        return sum(1 for _, c, p, _ in scores
                   if c >= c_thresh or (c >= c_low and p >= p_low))

    # ----- SWEEP 1: simple OR rule -----
    print("=" * 100)
    print("SWEEP 1: COUNT if close >= C_T  OR  pre >= P_T")
    print("=" * 100)
    print(f"{'close_T':>8} {'pre_T':>6}  {'hard_FP':>8} {'spch_FP':>8} {'real_TP':>8}  {'verdict'}")
    options = []
    for ct in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        for pt in [0.50, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]:
            hn_fp = count_or(hn, ct, pt)
            sn_fp = count_or(sn, ct, pt)
            real_tp = count_or(real, ct, pt)
            verdict = ""
            if hn_fp == 0 and sn_fp == 0:
                verdict = "*** ZERO FP ***"
                options.append((ct, pt, hn_fp, sn_fp, real_tp, "or"))
            print(f"{ct:>8.2f} {pt:>6.2f}  {hn_fp:>3}/{len(hn)} {sn_fp:>3}/{len(sn)} {real_tp:>3}/{len(real)}  {verdict}")

    # ----- SWEEP 2: guarded rule (require some closing activity) -----
    print()
    print("=" * 100)
    print("SWEEP 2: COUNT if close >= 0.50  OR  (close >= C_LOW  AND  pre >= P_LOW)")
    print("=" * 100)
    print(f"{'c_low':>6} {'p_low':>6}  {'hard_FP':>8} {'spch_FP':>8} {'real_TP':>8}  {'verdict'}")
    for cl in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
        for pl in [0.50, 0.70, 0.85, 0.90, 0.95, 0.99]:
            hn_fp = count_or_guarded(hn, 0.50, cl, pl)
            sn_fp = count_or_guarded(sn, 0.50, cl, pl)
            real_tp = count_or_guarded(real, 0.50, cl, pl)
            verdict = ""
            if hn_fp == 0 and sn_fp == 0:
                verdict = "*** ZERO FP ***"
                options.append((cl, pl, hn_fp, sn_fp, real_tp, "guarded"))
            print(f"{cl:>6.2f} {pl:>6.2f}  {hn_fp:>3}/{len(hn)} {sn_fp:>3}/{len(sn)} {real_tp:>3}/{len(real)}  {verdict}")

    # ----- Pareto best -----
    print()
    print("=" * 100)
    print("TOP ZERO-FP RULES BY REAL-WORLD DETECTION")
    print("=" * 100)
    for x, y, h, s, r, kind in sorted(options, key=lambda o: (-o[4], o[0], o[1]))[:15]:
        if kind == "or":
            print(f"  close>={x:.2f} OR pre>={y:.2f}                            -> real {r}/{len(real)} | hard_FP {h} | speech_FP {s}")
        else:
            print(f"  close>=0.50 OR (close>={x:.2f} AND pre>={y:.2f})           -> real {r}/{len(real)} | hard_FP {h} | speech_FP {s}")

    # Also show: detail per real recording at the recommended threshold
    print()
    print("=" * 100)
    print("PER-FILE DETAIL @ recommended thresholds (real captures only)")
    print("=" * 100)
    print(f"  {'file':<52} {'close':>6} {'pre':>6}")
    for name, c, p, b in real:
        print(f"  {name[:52]:<52} {c:>6.2f} {p:>6.2f}")


if __name__ == "__main__":
    main()
