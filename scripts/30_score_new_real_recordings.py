"""
Score the new real-world app recordings against both v6 and v7 TFLite models.

Source files:
  C:/Users/namob/Music/navkar/newapptesting/*.aac   (10 in-app captures)
  C:/Users/namob/Music/navkar/newrecords/*.ogg      (1 fresh WhatsApp recording)

For each file:
  1. Convert to 16 kHz mono WAV via imageio-ffmpeg
  2. Stage at data/real_world_test_v2/<group>/<name>.wav
  3. Slide a 2.5 s window every 250 ms across the whole clip
  4. Run each window through v6 and v7 TFLite, record peak score per file
  5. Compare to the in-app "Would COUNT" verdict (where known)
"""
import shutil
import subprocess
from pathlib import Path

import imageio_ffmpeg
import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 40000   # 2.5 s
HOP_SAMPLES = 4000        # 250 ms

V6_TFLITE = Path("models/closing_v1_20260601_163945/navkar_raw_audio_fp32.tflite")
V7_TFLITE = Path("models/multi_v1_20260601_181227/navkar_multi_phrase_fp32.tflite")

SOURCES = [
    (Path("C:/Users/namob/Music/navkar/newapptesting"), "*.aac", Path("data/real_world_test_v2/newapptesting")),
    (Path("C:/Users/namob/Music/navkar/newrecords"),    "*.ogg", Path("data/real_world_test_v2/newrecords")),
]

# Hand-mapped from the JPG screenshots, by AAC timestamp:
APP_VERDICTS = {
    "WhatsApp Audio 2026-06-02 at 9.44.09 AM": ("MISS",  0.03, 7.8),
    "WhatsApp Audio 2026-06-02 at 9.44.36 AM": ("COUNT", 1.00, 8.4),
    "WhatsApp Audio 2026-06-02 at 9.45.25 AM": ("MISS",  0.01, 13.1),
    "WhatsApp Audio 2026-06-02 at 9.46.07 AM": ("COUNT", 1.00, 8.0),
    "WhatsApp Audio 2026-06-02 at 9.46.44 AM": ("COUNT", 1.00, 8.5),
    "WhatsApp Audio 2026-06-02 at 9.47.33 AM": ("COUNT", 1.00, 7.0),
    "WhatsApp Audio 2026-06-02 at 9.48.01 AM": ("COUNT", 0.99, 7.5),
    "WhatsApp Audio 2026-06-02 at 9.48.17 AM": ("COUNT", 1.00, 7.5),
    "WhatsApp Audio 2026-06-02 at 9.48.37 AM": ("COUNT", 1.00, 7.0),
    "WhatsApp Audio 2026-06-02 at 9.49.04 AM": ("MISS",  0.11, 6.4),
}


def transcode_to_wav(src: Path, dst: Path) -> Path:
    """Decode any audio file to 16 kHz mono WAV via the bundled ffmpeg."""
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ffmpeg, "-y", "-i", str(src), "-ac", "1", "-ar", str(SAMPLE_RATE),
         "-loglevel", "error", str(dst)],
        check=True,
    )
    return dst


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sliding_peaks(audio: np.ndarray, interp_v6, interp_v7):
    """Slide windows, return (best_v6_prob, best_v6_t,
                              best_v7_close_prob, best_v7_close_t,
                              best_v7_pre_prob,   best_v7_pre_t)."""
    if len(audio) < WINDOW_SAMPLES:
        audio = np.pad(audio, (0, WINDOW_SAMPLES - len(audio)))

    in6 = interp_v6.get_input_details()[0]
    out6 = interp_v6.get_output_details()[0]
    in7 = interp_v7.get_input_details()[0]
    out7 = interp_v7.get_output_details()[0]

    best_v6 = (-1.0, 0.0)
    best_v7c = (-1.0, 0.0)
    best_v7p = (-1.0, 0.0)

    starts = list(range(0, max(1, len(audio) - WINDOW_SAMPLES + 1), HOP_SAMPLES))
    if not starts or starts[-1] + WINDOW_SAMPLES < len(audio):
        starts.append(max(0, len(audio) - WINDOW_SAMPLES))

    for s in starts:
        win = audio[s : s + WINDOW_SAMPLES].astype(np.float32).reshape(1, -1)
        t = s / SAMPLE_RATE

        interp_v6.set_tensor(in6["index"], win)
        interp_v6.invoke()
        p6 = float(sigmoid(interp_v6.get_tensor(out6["index"])[0, 0]))
        if p6 > best_v6[0]:
            best_v6 = (p6, t)

        interp_v7.set_tensor(in7["index"], win)
        interp_v7.invoke()
        logits = interp_v7.get_tensor(out7["index"])[0]
        pc = float(sigmoid(logits[0])); pp = float(sigmoid(logits[1]))
        if pc > best_v7c[0]:
            best_v7c = (pc, t)
        if pp > best_v7p[0]:
            best_v7p = (pp, t)

    return best_v6, best_v7c, best_v7p


def main():
    if not V6_TFLITE.exists() or not V7_TFLITE.exists():
        raise SystemExit(f"Missing TFLite. v6={V6_TFLITE.exists()} v7={V7_TFLITE.exists()}")

    v6 = tf.lite.Interpreter(model_path=str(V6_TFLITE)); v6.allocate_tensors()
    v7 = tf.lite.Interpreter(model_path=str(V7_TFLITE)); v7.allocate_tensors()
    print(f"v6: {V6_TFLITE}")
    print(f"v7: {V7_TFLITE}")
    print()

    rows = []
    for src_dir, glob, dst_dir in SOURCES:
        for src in sorted(src_dir.glob(glob)):
            stem = src.stem
            dst = dst_dir / f"{stem}.wav"
            try:
                transcode_to_wav(src, dst)
            except Exception as e:
                print(f"  failed to transcode {src.name}: {e}")
                continue
            audio, _ = librosa.load(str(dst), sr=SAMPLE_RATE, mono=True)
            dur = len(audio) / SAMPLE_RATE
            peak_amp = float(np.abs(audio).max())
            rms = float(np.sqrt((audio.astype(np.float64) ** 2).mean()))

            v6_best, v7c_best, v7p_best = sliding_peaks(audio, v6, v7)
            verdict_app, score_app, len_app = APP_VERDICTS.get(stem, ("?", float("nan"), float("nan")))

            rows.append({
                "name": stem[:60],
                "dur": dur,
                "peak_amp": peak_amp,
                "rms": rms,
                "app_verdict": verdict_app,
                "app_score": score_app,
                "v6_score": v6_best[0],
                "v6_t": v6_best[1],
                "v7_close": v7c_best[0],
                "v7_close_t": v7c_best[1],
                "v7_pre": v7p_best[0],
                "v7_pre_t": v7p_best[1],
            })

    # Print table
    print(f"{'file':<60} {'dur':>5} {'peak':>5} {'app':>6} {'app$':>5} {'v6':>5} {'v6t':>5} {'v7c':>5} {'v7ct':>5} {'v7p':>5} {'v7pt':>5}")
    print("-" * 130)
    n_count_app = 0; n_v6_count = 0; n_v7_close_count = 0; n_v7_either_count = 0
    n_v7_temporal_count = 0  # temporal rule from spec
    for r in rows:
        print(f"{r['name']:<60} {r['dur']:>5.1f} {r['peak_amp']:>5.2f} {r['app_verdict']:>6} {r['app_score']:>5.2f} "
              f"{r['v6_score']:>5.2f} {r['v6_t']:>5.1f} "
              f"{r['v7_close']:>5.2f} {r['v7_close_t']:>5.1f} "
              f"{r['v7_pre']:>5.2f} {r['v7_pre_t']:>5.1f}")
        if r['app_verdict'] == "COUNT":
            n_count_app += 1
        if r['v6_score'] >= 0.5:
            n_v6_count += 1
        if r['v7_close'] >= 0.5:
            n_v7_close_count += 1
        if r['v7_close'] >= 0.5 or r['v7_pre'] >= 0.5:
            n_v7_either_count += 1
        # Temporal rule from V7 spec:
        # closing >= 0.80 alone OR (closing >= 0.50 AND pre fired within 8s BEFORE closing)
        c_strong = r['v7_close'] >= 0.80
        c_weak = r['v7_close'] >= 0.50
        pre_recent = (r['v7_pre'] >= 0.50 and
                      0 <= (r['v7_close_t'] - r['v7_pre_t']) <= 8.0)
        if c_strong or (c_weak and pre_recent):
            n_v7_temporal_count += 1

    print()
    print(f"Files: {len(rows)}")
    print(f"  App (v6 in-app)         COUNT:  {n_count_app}/10")
    print(f"  Python v6 (>=0.50)       COUNT:  {n_v6_count}/{len(rows)}")
    print(f"  Python v7 closing alone  COUNT:  {n_v7_close_count}/{len(rows)}")
    print(f"  Python v7 either head    COUNT:  {n_v7_either_count}/{len(rows)}")
    print(f"  Python v7 temporal rule  COUNT:  {n_v7_temporal_count}/{len(rows)}")

    # CSV
    out_csv = Path("data/real_world_test_v2/scores.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved {out_csv}")


if __name__ == "__main__":
    main()
