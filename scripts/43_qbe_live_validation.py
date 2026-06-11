"""
THE DECISIVE TEST: QbE matching on real live-path captures.

Data (captured via the in-app diagnostics screen, VOICE_RECOGNITION source):
  session1  58.8s continuous chanting  (~5-6 Navkars)
  session2  60.0s continuous chanting  (~5-6 Navkars)
  background 26.8s background voices   (negative)

Experiments:
  A. Audio reality check — RMS / peak / spectral rolloff of the live channel
     (is it bandwidth-limited? how quiet?).
  B. Activity separation — enroll chant templates from session1, score every
     2.5 s window of session2 (should be HIGH: same voice/phrase/channel) vs
     held-out background windows (should be LOW). Dual-cohort scoring:
     competitors = TTS imposter chants + first half of background.
  C. Cycle counting feasibility — closing template anchored at session1's
     v8-close peak (t=39 s); similarity timeline over session2 should show
     periodic peaks (~one per Navkar cycle). Count peaks vs expected.
"""
import json
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf

SAMPLE_RATE = 16000
MELSPEC_TFLITE = Path("models/openwakeword/melspectrogram.tflite")
EMBEDDING_TFLITE = Path("models/openwakeword/embedding_model.tflite")

EMB_WINDOW = 76
EMB_STRIDE = 8
MS_PER_EMB_STEP = 80
WIN_STEPS = 31  # 2.5 s

LIVE_DIR = Path("C:/Users/namob/Music/navkar/drive-download-20260611T134627Z-3-001")
S1 = LIVE_DIR / "navkar_live_2026-06-11_13-42-37.wav"
S2 = LIVE_DIR / "navkar_live_2026-06-11_13-44-01.wav"
BG = LIVE_DIR / "navkar_live_2026-06-11_13-44-41-backgroundvoices.wav"
HARD_NEG_ROOT = Path("data/hard_negatives")

IMPOSTER_FILES = [
    "mangalam_jain/sarvam_mangalam_bhagwan_anushka_p85.wav",
    "mangalam_jain/sarvam_jaine_dharmo_karun_p85.wav",
    "hindu_namo/sarvam_om_namo_narayanaya_vidya_p114.wav",
    "hindu_namo/sarvam_om_namah_shivaya_anushka_p85.wav",
    "truncated/sarvam_opening_only_manisha_p85.wav",
    "other_jain/sarvam_chattari_mangalam_manisha_p130.wav",
    "buddhist_namo/sarvam_namo_tassa_manisha_p85.wav",
    "spoken_about/sarvam_about_mangalam_vidya_p114.wav",
]

# v8-close anchor in session1 (from the live scores JSON): closing at ~39.0 s.
S1_CLOSING_ANCHOR_SEC = 39.0


class FeatureStack:
    def __init__(self):
        self.mel = tf.lite.Interpreter(model_path=str(MELSPEC_TFLITE))
        self.emb = tf.lite.Interpreter(model_path=str(EMBEDDING_TFLITE))
        self.emb.allocate_tensors()
        self.emb_in = self.emb.get_input_details()[0]
        self.emb_out = self.emb.get_output_details()[0]
        self.mel_in = self.mel.get_input_details()[0]
        self.mel_out = self.mel.get_output_details()[0]

    def melspec(self, audio):
        x = audio.astype(np.float32).reshape(1, -1)
        self.mel.resize_tensor_input(self.mel_in["index"], x.shape)
        self.mel.allocate_tensors()
        self.mel.set_tensor(self.mel_in["index"], x)
        self.mel.invoke()
        return np.squeeze(self.mel.get_tensor(self.mel_out["index"])) / 10.0 + 2.0

    def embed(self, audio):
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


def window_means(seq, win_steps=WIN_STEPS):
    if seq.shape[0] == 0:
        return np.zeros((0, 96), dtype=np.float32)
    if seq.shape[0] <= win_steps:
        return l2norm(seq.mean(axis=0)).reshape(1, 96)
    cs = np.cumsum(seq, axis=0)
    out = []
    for s in range(seq.shape[0] - win_steps + 1):
        pooled = (cs[s + win_steps - 1] - (cs[s - 1] if s > 0 else 0)) / win_steps
        out.append(l2norm(pooled))
    return np.array(out, dtype=np.float32)


def audio_stats(path):
    audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    rms = float(np.sqrt((audio.astype(np.float64) ** 2).mean()))
    peak = float(np.abs(audio).max())
    # Spectral rolloff (95% energy) — the bandwidth question.
    S = np.abs(librosa.stft(audio, n_fft=1024))
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=SAMPLE_RATE, roll_percent=0.95)
    return audio, rms, peak, float(np.median(rolloff))


def main():
    stack = FeatureStack()

    # ---- A. Audio reality check ----
    print("=" * 70)
    print("A. LIVE CHANNEL AUDIO STATS")
    print("=" * 70)
    audios = {}
    for name, path in [("session1", S1), ("session2", S2), ("background", BG)]:
        audio, rms, peak, roll = audio_stats(path)
        audios[name] = audio
        print(f"  {name:<11} dur={len(audio)/SAMPLE_RATE:5.1f}s  rms={rms:.4f}  peak={peak:.3f}  rolloff95={roll:6.0f} Hz")

    # ---- Embeddings ----
    print("\nEmbedding sessions...")
    seq1 = stack.embed(audios["session1"])
    seq2 = stack.embed(audios["session2"])
    seqbg = stack.embed(audios["background"])
    w1 = window_means(seq1)
    w2 = window_means(seq2)
    wbg = window_means(seqbg)
    print(f"  session1: {w1.shape[0]} windows | session2: {w2.shape[0]} | background: {wbg.shape[0]}")

    # ---- B. Activity separation (dual-cohort) ----
    # Templates: every 8th window of session1 (~every 0.64 s of chant).
    templates = w1[::8]
    # Background split: first half = cohort, second half = held-out negatives.
    half = wbg.shape[0] // 2
    bg_cohort, bg_test = wbg[:half], wbg[half:]
    # Imposter windows from TTS chants.
    imp = []
    for rel in IMPOSTER_FILES:
        audio, _ = librosa.load(str(HARD_NEG_ROOT / rel), sr=SAMPLE_RATE, mono=True)
        imp.append(window_means(stack.embed(audio)))
    imposters = np.concatenate(imp, axis=0)
    competitors = np.concatenate([bg_cohort, imposters], axis=0)
    print(f"  templates={templates.shape[0]}  competitors={competitors.shape[0]} (bg {bg_cohort.shape[0]} + imp {imposters.shape[0]})")

    def delta_scores(wins):
        t = (wins @ templates.T).max(axis=1)
        c = (wins @ competitors.T).max(axis=1)
        return t - c

    d2 = delta_scores(w2)
    dbg = delta_scores(bg_test)

    print("\n" + "=" * 70)
    print("B. ACTIVITY SEPARATION (per 2.5 s window, dual-cohort delta)")
    print("=" * 70)
    print(f"  session2 (chant):    p10={np.percentile(d2,10):+.3f} med={np.median(d2):+.3f} p90={np.percentile(d2,90):+.3f}")
    print(f"  background (test):   min={dbg.min():+.3f} med={np.median(dbg):+.3f} max={dbg.max():+.3f}")
    for thr in [0.0, 0.005, 0.01, 0.02]:
        frac_chant = float((d2 > thr).mean())
        n_bg = int((dbg > thr).sum())
        print(f"  thr {thr:+.3f}: chant windows above = {frac_chant*100:5.1f}%   background FP windows = {n_bg}/{len(dbg)}")

    # ---- C. Cycle counting via closing template ----
    print("\n" + "=" * 70)
    print("C. CYCLE COUNTING (closing template from session1 @ t=39 s)")
    print("=" * 70)
    anchor_step = int(S1_CLOSING_ANCHOR_SEC * 1000 / MS_PER_EMB_STEP)
    closing_template = l2norm(seq1[anchor_step : anchor_step + WIN_STEPS].mean(axis=0)).reshape(1, 96)

    def closing_delta(wins):
        t = (wins @ closing_template.T).max(axis=1)
        c = (wins @ competitors.T).max(axis=1)
        return t - c

    c2 = closing_delta(w2)
    cbg = closing_delta(bg_test)
    c1 = closing_delta(w1)  # self-session sanity (anchor excluded implicitly fine)

    # Peak picking: local maxima above threshold with 5 s min separation.
    def count_peaks(scores, thr, min_sep_steps=int(5000 / MS_PER_EMB_STEP)):
        peaks = []
        last = -min_sep_steps
        for i in range(len(scores)):
            if scores[i] > thr and i - last >= min_sep_steps:
                # local max in +-6 steps
                lo, hi = max(0, i - 6), min(len(scores), i + 7)
                if scores[i] >= scores[lo:hi].max() - 1e-9:
                    peaks.append(i)
                    last = i
        return peaks

    print(f"  closing-delta session2: med={np.median(c2):+.3f} max={c2.max():+.3f}")
    print(f"  closing-delta background(test): max={cbg.max():+.3f}")
    for thr in [0.0, 0.005, 0.01]:
        p1 = count_peaks(c1, thr)
        p2 = count_peaks(c2, thr)
        pbg = count_peaks(cbg, thr)
        t2 = [round(i * MS_PER_EMB_STEP / 1000, 1) for i in p2]
        print(f"  thr {thr:+.3f}: session1 peaks={len(p1)}  session2 peaks={len(p2)} at {t2}  bg peaks={len(pbg)}")
    print("\n  (expected: ~5-6 peaks per session if one per Navkar cycle; 0 for bg)")


if __name__ == "__main__":
    main()
