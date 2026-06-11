"""
QbE with DUAL-COHORT normalization — the production scoring design candidate.

Findings so far:
  40_: raw cosine ranks all 8 app-channel positives above all TTS/FLEURS
       negatives, but margin is thin (+0.005) and speech-cohort normalization
       made it WORSE (conversation is the wrong cohort for chant imposters).
  41_: matching is phrase-driven (13/14 cross-channel positives pass) but
       real-channel ambience FPs at TTS-calibrated thresholds (2/8 Birch).

Design under test here — per-window open-set verification score:
    score(win) = cos(template, win) - max( cos(bg_centroid, win),
                                           max_i cos(imposter_i, win) )
  where
    template   = enrolled user chant (mean-pooled embedding)
    bg_centroid= background audio from a real device (channel competitor)
    imposters  = TTS chants of OTHER prayers (content competitors:
                 Mangalam Bhagwan, Om Namo Narayanaya, truncated Navkar)

  file score = max over windows, max over enrollment templates.

Eval:
  positives: 8 held-out app chants + 14 cross-channel close-mic chants
  negatives: 260 hard negatives (minus imposters) + 165 speech (minus 40_'s
             cohort) + 5 held-out Birch ambient files
  cohorts (excluded from eval): 3 Birch files (bg), 12 TTS files (imposters)
"""
import subprocess
import tempfile
from pathlib import Path

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

REAL_DIR = Path("data/real_world_test_v2/newapptesting")
NEWREC_DIR = Path("data/real_world_test_v2/newrecords")
MUSIC_DIR = Path("C:/Users/namob/Music/navkar")
RANDOM_DIR = MUSIC_DIR / "randomthings"
SPEECH_NEG_DIR = Path("data/speech_negative_test")
HARD_NEG_ROOT = Path("data/hard_negatives")

ENROLL_NAMES = [
    "WhatsApp Audio 2026-06-02 at 9.44.36 AM.wav",
    "WhatsApp Audio 2026-06-02 at 9.46.07 AM.wav",
    "WhatsApp Audio 2026-06-02 at 9.46.44 AM.wav",
]

# Background cohort: 3 Birch ambient files (held out of eval).
BG_NAMES = ["Birch Tree Way 4.m4a", "Birch Tree Way 5.m4a", "Birch Tree Way 8.m4a"]

# Imposter cohort: 12 TTS chants spanning the confusable categories.
IMPOSTER_FILES = [
    "mangalam_jain/sarvam_mangalam_bhagwan_anushka_p85.wav",
    "mangalam_jain/sarvam_mangalam_bhagwan_karun_p85.wav",
    "mangalam_jain/sarvam_jaine_dharmo_karun_p85.wav",
    "mangalam_jain/sarvam_jaine_dharmo_anushka_p85.wav",
    "hindu_namo/sarvam_om_namo_narayanaya_vidya_p114.wav",
    "hindu_namo/sarvam_om_namo_narayanaya_anushka_p100.wav",
    "hindu_namo/sarvam_om_namah_shivaya_anushka_p85.wav",
    "truncated/sarvam_opening_only_manisha_p85.wav",
    "truncated/sarvam_namo_arihantanam_loop_arya_p130.wav",
    "other_jain/sarvam_chattari_mangalam_manisha_p130.wav",
    "buddhist_namo/sarvam_namo_tassa_manisha_p85.wav",
    "spoken_about/sarvam_about_mangalam_vidya_p114.wav",
]

WIN_STEPS = 31  # 2.5 s matching window everywhere (uniform, production-like)


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


def load_any_audio(path: Path) -> np.ndarray:
    try:
        audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
        return audio
    except Exception:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t:
            tmp = t.name
        try:
            r = subprocess.run(
                [ffmpeg, "-y", "-i", str(path), "-ac", "1", "-ar", str(SAMPLE_RATE),
                 "-loglevel", "error", tmp], capture_output=True, timeout=60)
            if r.returncode != 0:
                return np.zeros(0, dtype=np.float32)
            audio, _ = librosa.load(tmp, sr=SAMPLE_RATE, mono=True)
            return audio
        finally:
            try:
                Path(tmp).unlink()
            except OSError:
                pass


def window_means(seq: np.ndarray, win_steps: int) -> np.ndarray:
    """All L2-normalized mean-pooled sliding windows: (W, 96)."""
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


def main():
    stack = FeatureStack()

    # ---- Templates (uniform 2.5 s windows from enrollment chants) ----
    # Take ALL 2.5 s windows of each enrollment chant whose v8-free proxy is
    # simply: every window. Template per chant = the window set; we match a
    # test window against the BEST template window (max-max). This avoids
    # needing the v8 head to locate the closing — production keeps it simple.
    print("Enrolling...")
    template_wins = []  # list of (W_i, 96) arrays
    for name in ENROLL_NAMES:
        audio = load_any_audio(REAL_DIR / name)
        seq = stack.embed(audio)
        tw = window_means(seq, WIN_STEPS)
        template_wins.append(tw)
        print(f"  {name[:48]}: {tw.shape[0]} template windows")
    templates = np.concatenate(template_wins, axis=0)  # (T, 96)

    # ---- Cohorts ----
    print("Building background cohort (3 Birch files)...")
    bg_wins = []
    for name in BG_NAMES:
        audio = load_any_audio(RANDOM_DIR / name)[: SAMPLE_RATE * 60]
        bg_wins.append(window_means(stack.embed(audio), WIN_STEPS))
    bg = np.concatenate(bg_wins, axis=0)  # (B, 96)

    print("Building imposter cohort (12 TTS chants)...")
    imp_wins = []
    for rel in IMPOSTER_FILES:
        audio = load_any_audio(HARD_NEG_ROOT / rel)
        imp_wins.append(window_means(stack.embed(audio), WIN_STEPS))
    imposters = np.concatenate(imp_wins, axis=0)  # (I, 96)
    print(f"  templates={templates.shape[0]} bg={bg.shape[0]} imposters={imposters.shape[0]} windows")

    def score_file(path: Path) -> float:
        """max over test windows of [max-template cos - max-competitor cos]."""
        audio = load_any_audio(path)
        if len(audio) < SAMPLE_RATE:
            return float("nan")
        audio = audio[: SAMPLE_RATE * 60]
        wins = window_means(stack.embed(audio), WIN_STEPS)  # (W, 96)
        if wins.shape[0] == 0:
            return float("nan")
        t_cos = wins @ templates.T            # (W, T)
        c_cos = np.concatenate([wins @ bg.T, wins @ imposters.T], axis=1)  # (W, B+I)
        delta = t_cos.max(axis=1) - c_cos.max(axis=1)  # (W,)
        return float(delta.max())

    # ---- Eval sets ----
    imposter_set = {Path(r).name for r in IMPOSTER_FILES}
    bg_set = set(BG_NAMES)

    pos_app = [f for f in sorted(REAL_DIR.glob("*.wav")) if f.name not in ENROLL_NAMES]
    pos_app += sorted(NEWREC_DIR.glob("*.wav"))
    pos_cross = sorted(
        f for f in MUSIC_DIR.iterdir()
        if f.is_file() and f.stem.lower().startswith("navkar")
    )
    all_speech = sorted(SPEECH_NEG_DIR.glob("*.wav"))
    speech_eval = [f for i, f in enumerate(all_speech) if f not in all_speech[::4][:50]]
    hard_eval = []
    for cat in sorted(d for d in HARD_NEG_ROOT.iterdir() if d.is_dir()):
        hard_eval += [f for f in sorted(cat.glob("*.wav")) if f.name not in imposter_set]
    birch_eval = [f for f in sorted(RANDOM_DIR.iterdir())
                  if f.is_file() and f.name not in bg_set]

    def run(files, label):
        rows = []
        for i, f in enumerate(files):
            s = score_file(f)
            if not np.isnan(s):
                rows.append((f.name, s))
            if (i + 1) % 50 == 0:
                print(f"  {label}: {i + 1}/{len(files)}")
        return rows

    print(f"\nScoring {len(pos_app)} app positives...")
    r_pos_app = run(pos_app, "pos_app")
    for n, s in r_pos_app:
        print(f"  {s:+.3f}  {n[:55]}")
    print(f"\nScoring {len(pos_cross)} cross-channel positives...")
    r_pos_cross = run(pos_cross, "pos_cross")
    for n, s in r_pos_cross:
        print(f"  {s:+.3f}  {n[:55]}")
    print(f"\nScoring negatives: {len(speech_eval)} speech, {len(hard_eval)} hard, {len(birch_eval)} birch...")
    r_speech = run(speech_eval, "speech")
    r_hard = run(hard_eval, "hard")
    r_birch = run(birch_eval, "birch")
    for n, s in r_birch:
        print(f"  birch {s:+.3f}  {n}")

    # ---- Verdict ----
    pos_all = [s for _, s in r_pos_app + r_pos_cross]
    neg_all = [s for _, s in r_speech + r_hard + r_birch]
    max_neg = max(neg_all)
    n_zero_fp = sum(1 for s in pos_all if s > max_neg)
    print(f"\n{'=' * 70}\nDUAL-COHORT VERDICT\n{'=' * 70}")
    print(f"  positives ({len(pos_all)}): min={min(pos_all):+.3f} mean={np.mean(pos_all):+.3f}")
    print(f"  negatives ({len(neg_all)}): mean={np.mean(neg_all):+.3f} max={max_neg:+.3f}")
    print(f"  ZERO-FP catch: {n_zero_fp}/{len(pos_all)}  (margin {min(pos_all) - max_neg:+.3f})")
    worst = sorted(r_speech + r_hard + r_birch, key=lambda r: -r[1])[:10]
    print("  worst negatives:")
    for n, s in worst:
        print(f"    {s:+.3f}  {n[:55]}")
    # Sensible operating point: also show catch rate at a small positive margin.
    for thr_off in [0.0, 0.01, 0.02]:
        thr = max_neg + thr_off
        print(f"  catch at threshold {thr:+.3f}: "
              f"{sum(1 for s in pos_all if s > thr)}/{len(pos_all)}")


if __name__ == "__main__":
    main()
