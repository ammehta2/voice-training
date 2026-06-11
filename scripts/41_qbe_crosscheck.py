"""
QbE cross-channel sanity check.

The main prototype (40_) separated the user's held-out chants from TTS/FLEURS
negatives perfectly on raw cosine — but enrollment and positives shared a
recording channel, so the separation could be partly channel-similarity.

This script probes that with real recordings from OTHER channels:
  - cross-channel POSITIVES: the original close-mic WhatsApp recordings
    (Navkar*.m4a/ogg) — same phrase, mostly same voice, different channel.
    High scores => matching is content/voice-driven, not channel-driven.
  - real-recording NEGATIVES: Music/navkar/randomthings/* (Birch Tree ambient
    recordings made on a real device) — real-mic channel, no Navkar.
    Low scores => real-channel audio doesn't trivially match.

Templates: same 3 enrollment chants as 40_ (the app captures).
Reference zero-FP thresholds from the 40_ run: full=0.971, close=0.965.
"""
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
import librosa
import numpy as np
import tensorflow as tf

SAMPLE_RATE = 16000
MELSPEC_TFLITE = Path("models/openwakeword/melspectrogram.tflite")
EMBEDDING_TFLITE = Path("models/openwakeword/embedding_model.tflite")
V8_TFLITE = Path("models/multi_v1_20260604_091812/navkar_multi_phrase_fp32.tflite")

EMB_WINDOW = 76
EMB_STRIDE = 8
MS_PER_EMB_STEP = 80
CLOSE_REGION_STEPS = 31

REAL_DIR = Path("data/real_world_test_v2/newapptesting")
MUSIC_DIR = Path("C:/Users/namob/Music/navkar")
RANDOM_DIR = MUSIC_DIR / "randomthings"

ENROLL_NAMES = [
    "WhatsApp Audio 2026-06-02 at 9.44.36 AM.wav",
    "WhatsApp Audio 2026-06-02 at 9.46.07 AM.wav",
    "WhatsApp Audio 2026-06-02 at 9.46.44 AM.wav",
]

ZERO_FP_FULL = 0.971
ZERO_FP_CLOSE = 0.965


class FeatureStack:
    def __init__(self):
        self.mel = tf.lite.Interpreter(model_path=str(MELSPEC_TFLITE))
        self.emb = tf.lite.Interpreter(model_path=str(EMBEDDING_TFLITE))
        self.emb.allocate_tensors()
        self.emb_in = self.emb.get_input_details()[0]
        self.emb_out = self.emb.get_output_details()[0]
        self.mel_in = self.mel.get_input_details()[0]
        self.mel_out = self.mel.get_output_details()[0]

    def melspec(self, audio: np.ndarray) -> np.ndarray:
        x = audio.astype(np.float32).reshape(1, -1)
        self.mel.resize_tensor_input(self.mel_in["index"], x.shape)
        self.mel.allocate_tensors()
        self.mel.set_tensor(self.mel_in["index"], x)
        self.mel.invoke()
        out = self.mel.get_tensor(self.mel_out["index"])
        return np.squeeze(out) / 10.0 + 2.0

    def embed(self, audio: np.ndarray) -> np.ndarray:
        spec = self.melspec(audio)
        if spec.ndim != 2 or spec.shape[0] < EMB_WINDOW:
            return np.zeros((0, 96), dtype=np.float32)
        embs = []
        for start in range(0, spec.shape[0] - EMB_WINDOW + 1, EMB_STRIDE):
            win = spec[start : start + EMB_WINDOW]
            inp = win.reshape(1, EMB_WINDOW, 32, 1).astype(np.float32)
            self.emb.set_tensor(self.emb_in["index"], inp)
            self.emb.invoke()
            embs.append(self.emb.get_tensor(self.emb_out["index"]).reshape(96))
        return np.array(embs, dtype=np.float32)


def l2norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def load_any_audio(path: Path) -> np.ndarray:
    """Load wav/m4a/ogg via librosa, falling back to bundled ffmpeg."""
    try:
        audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
        return audio
    except Exception:
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t:
            tmp = t.name
        try:
            r = subprocess.run(
                [ffmpeg, "-y", "-i", str(path), "-ac", "1", "-ar", str(SAMPLE_RATE),
                 "-loglevel", "error", tmp],
                capture_output=True, timeout=60)
            if r.returncode != 0:
                return np.zeros(0, dtype=np.float32)
            audio, _ = librosa.load(tmp, sr=SAMPLE_RATE, mono=True)
            return audio
        finally:
            try:
                Path(tmp).unlink()
            except OSError:
                pass


def locate_closing_steps(audio, v8):
    interp, in_det, out_det = v8
    n = 40000
    if len(audio) < n:
        audio = np.pad(audio, (0, n - len(audio)))
    best_t, best_score = 0.0, -1.0
    for s in range(0, max(1, len(audio) - n + 1), 4000):
        win = audio[s : s + n].astype(np.float32).reshape(1, -1)
        interp.set_tensor(in_det["index"], win)
        interp.invoke()
        logit = interp.get_tensor(out_det["index"])[0][0]
        p = 1.0 / (1.0 + np.exp(-logit))
        if p > best_score:
            best_score, best_t = p, s / SAMPLE_RATE
    return int(best_t * 1000 / MS_PER_EMB_STEP)


def sliding_max_cosine(seq, template, win_steps):
    if seq.shape[0] == 0:
        return 0.0
    if seq.shape[0] <= win_steps:
        return float(np.dot(l2norm(seq.mean(axis=0)), template))
    cs = np.cumsum(seq, axis=0)
    best = -1.0
    for s in range(seq.shape[0] - win_steps + 1):
        pooled = (cs[s + win_steps - 1] - (cs[s - 1] if s > 0 else 0)) / win_steps
        c = float(np.dot(l2norm(pooled), template))
        if c > best:
            best = c
    return best


def main():
    stack = FeatureStack()
    v8i = tf.lite.Interpreter(model_path=str(V8_TFLITE))
    v8i.allocate_tensors()
    v8 = (v8i, v8i.get_input_details()[0], v8i.get_output_details()[0])

    templates_full, templates_close = [], []
    for name in ENROLL_NAMES:
        audio, _ = librosa.load(str(REAL_DIR / name), sr=SAMPLE_RATE, mono=True)
        seq = stack.embed(audio)
        templates_full.append((l2norm(seq.mean(axis=0)), seq.shape[0]))
        cstart = locate_closing_steps(audio, v8)
        cseq = seq[cstart : cstart + CLOSE_REGION_STEPS]
        if cseq.shape[0] < 5:
            cseq = seq[-CLOSE_REGION_STEPS:]
        templates_close.append((l2norm(cseq.mean(axis=0)), CLOSE_REGION_STEPS))
    print(f"Enrolled {len(templates_full)} templates (same as 40_).\n")

    def score(audio):
        seq = stack.embed(audio)
        sf = max((sliding_max_cosine(seq, t, w) for t, w in templates_full), default=0.0)
        sc = max((sliding_max_cosine(seq, t, w) for t, w in templates_close), default=0.0)
        return sf, sc

    # Cross-channel positives: original close-mic recordings.
    pos_files = sorted(
        f for f in MUSIC_DIR.iterdir()
        if f.is_file() and f.stem.lower().startswith("navkar")
    )
    print(f"CROSS-CHANNEL POSITIVES ({len(pos_files)} files, zero-FP refs: full>{ZERO_FP_FULL}, close>{ZERO_FP_CLOSE}):")
    pos_pass_f = pos_pass_c = 0
    for f in pos_files:
        audio = load_any_audio(f)
        if len(audio) < SAMPLE_RATE:
            print(f"  skip (short/decode fail): {f.name}")
            continue
        sf, sc = score(audio)
        pf = "PASS" if sf > ZERO_FP_FULL else "miss"
        pc = "PASS" if sc > ZERO_FP_CLOSE else "miss"
        if sf > ZERO_FP_FULL: pos_pass_f += 1
        if sc > ZERO_FP_CLOSE: pos_pass_c += 1
        print(f"  full={sf:.3f} {pf}  close={sc:.3f} {pc}   {f.name}")

    # Real-channel negatives: ambient recordings from a real device.
    neg_files = sorted(RANDOM_DIR.iterdir()) if RANDOM_DIR.exists() else []
    print(f"\nREAL-CHANNEL NEGATIVES ({len(neg_files)} files):")
    neg_fp_f = neg_fp_c = 0
    for f in neg_files:
        if not f.is_file():
            continue
        audio = load_any_audio(f)
        if len(audio) < SAMPLE_RATE:
            print(f"  skip (short/decode fail): {f.name}")
            continue
        # Cap at 60 s for speed.
        audio = audio[: SAMPLE_RATE * 60]
        sf, sc = score(audio)
        ff = "FP!" if sf > ZERO_FP_FULL else "ok"
        fc = "FP!" if sc > ZERO_FP_CLOSE else "ok"
        if sf > ZERO_FP_FULL: neg_fp_f += 1
        if sc > ZERO_FP_CLOSE: neg_fp_c += 1
        print(f"  full={sf:.3f} {ff}  close={sc:.3f} {fc}   {f.name}")

    print(f"\nSUMMARY")
    print(f"  cross-channel positives passing: full {pos_pass_f}, close {pos_pass_c} (of those scored)")
    print(f"  real-channel negative FPs:       full {neg_fp_f}, close {neg_fp_c}")


if __name__ == "__main__":
    main()
