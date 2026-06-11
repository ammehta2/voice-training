"""
Phase 1 prototype: query-by-example (QbE) chant detection using the
openWakeWord feature stack (melspectrogram + frozen Google speech embedding).

Hypothesis under test: enrolling 3 of the user's own chants and matching by
embedding cosine similarity separates their Navkar from (a) other speech and
(b) the adversarial Mangalam/Namo hard negatives — WITHOUT any task-specific
training. If true, the mobile rebuild ports this stack instead of iterating
on the from-scratch DS-CNN.

Pipeline (mirrors openWakeWord):
  16 kHz audio → melspectrogram.tflite → (frames, 32) melspec
              → /10 + 2 transform
              → embedding_model.tflite over 76-frame windows, stride 8
              → (T, 96) embedding sequence  (one vector per 80 ms)

Two template variants per enrollment chant:
  A. full   — mean-pooled embedding of the whole chant
  B. close  — mean-pooled embedding of the 2.5 s closing region,
              located by the v8 closing head (production design: v8 locates,
              embedding matches)

Scoring a test file: slide a window of the template's length across its
embedding sequence (stride 1 = 80 ms), mean-pool, cosine vs template;
file score = max over windows, max over the 3 enrollment templates.

Eval:
  positives  = the user's remaining 8 real recordings (NOT enrolled)
  negatives  = 215 speech (FLEURS hi/gu/mr) + 260 adversarial hard negatives

Honest caveat: enrollment + positives share a capture channel; negatives are
TTS/FLEURS (different channels). Clean separation here is necessary but not
sufficient — the decisive test is on Phase 0 live captures. This prototype
rules the approach in or out cheaply.
"""
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf

SAMPLE_RATE = 16000
MELSPEC_TFLITE = Path("models/openwakeword/melspectrogram.tflite")
EMBEDDING_TFLITE = Path("models/openwakeword/embedding_model.tflite")
V8_TFLITE = Path("models/multi_v1_20260604_091812/navkar_multi_phrase_fp32.tflite")

EMB_WINDOW = 76      # melspec frames per embedding (~0.76 s context)
EMB_STRIDE = 8       # melspec frames between embeddings (80 ms)
MS_PER_EMB_STEP = 80
CLOSE_REGION_STEPS = 31  # 2.5 s / 80 ms

REAL_DIR = Path("data/real_world_test_v2/newapptesting")
NEWREC_DIR = Path("data/real_world_test_v2/newrecords")
SPEECH_NEG_DIR = Path("data/speech_negative_test")
HARD_NEG_ROOT = Path("data/hard_negatives")

# Enrollment: the 3 cleanest chants (app verdict COUNT at 1.00).
ENROLL_NAMES = [
    "WhatsApp Audio 2026-06-02 at 9.44.36 AM.wav",
    "WhatsApp Audio 2026-06-02 at 9.46.07 AM.wav",
    "WhatsApp Audio 2026-06-02 at 9.46.44 AM.wav",
]


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
        """audio float32 [-1,1] → (frames, 32) melspec with oWW transform."""
        x = audio.astype(np.float32).reshape(1, -1)
        self.mel.resize_tensor_input(self.mel_in["index"], x.shape)
        self.mel.allocate_tensors()
        self.mel.set_tensor(self.mel_in["index"], x)
        self.mel.invoke()
        out = self.mel.get_tensor(self.mel_out["index"])  # (1, 1, frames, 32)
        return np.squeeze(out) / 10.0 + 2.0

    def embed(self, audio: np.ndarray) -> np.ndarray:
        """audio → (T, 96) embedding sequence, one vector per 80 ms."""
        spec = self.melspec(audio)
        if spec.ndim != 2 or spec.shape[0] < EMB_WINDOW:
            return np.zeros((0, 96), dtype=np.float32)
        embs = []
        for start in range(0, spec.shape[0] - EMB_WINDOW + 1, EMB_STRIDE):
            win = spec[start : start + EMB_WINDOW]
            inp = win.reshape(1, EMB_WINDOW, 32, 1).astype(np.float32)
            self.emb.set_tensor(self.emb_in["index"], inp)
            self.emb.invoke()
            vec = self.emb.get_tensor(self.emb_out["index"]).reshape(96)
            embs.append(vec)
        return np.array(embs, dtype=np.float32)


def l2norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def locate_closing_steps(audio: np.ndarray, v8) -> int:
    """Best 2.5 s window start (in embedding steps) per the v8 closing head."""
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


def sliding_max_cosine(seq: np.ndarray, template: np.ndarray, win_steps: int) -> float:
    """Max cosine between template and mean-pooled sliding windows of seq."""
    if seq.shape[0] == 0:
        return 0.0
    if seq.shape[0] <= win_steps:
        return float(np.dot(l2norm(seq.mean(axis=0)), template))
    # Cumulative sum for O(1) window means.
    cs = np.cumsum(seq, axis=0)
    best = -1.0
    for s in range(seq.shape[0] - win_steps + 1):
        pooled = (cs[s + win_steps - 1] - (cs[s - 1] if s > 0 else 0)) / win_steps
        c = float(np.dot(l2norm(pooled), template))
        if c > best:
            best = c
    return best


def sliding_best_window(seq: np.ndarray, template: np.ndarray, win_steps: int):
    """Return (best_cosine, best_window_mean_vec) over sliding windows."""
    if seq.shape[0] == 0:
        return 0.0, None
    if seq.shape[0] <= win_steps:
        pooled = l2norm(seq.mean(axis=0))
        return float(np.dot(pooled, template)), pooled
    cs = np.cumsum(seq, axis=0)
    best, best_vec = -1.0, None
    for s in range(seq.shape[0] - win_steps + 1):
        pooled = (cs[s + win_steps - 1] - (cs[s - 1] if s > 0 else 0)) / win_steps
        pooled = l2norm(pooled)
        c = float(np.dot(pooled, template))
        if c > best:
            best, best_vec = c, pooled
    return best, best_vec


def main():
    stack = FeatureStack()
    v8i = tf.lite.Interpreter(model_path=str(V8_TFLITE))
    v8i.allocate_tensors()
    v8 = (v8i, v8i.get_input_details()[0], v8i.get_output_details()[0])

    # ---- Embed everything ONCE, cache sequences in memory ----
    def load_seq(path: Path) -> np.ndarray:
        audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
        return stack.embed(audio)

    # ---- Enrollment ----
    print("Enrolling on 3 chants...")
    templates_full = []   # (template_vec, win_steps)
    templates_close = []
    for name in ENROLL_NAMES:
        audio, _ = librosa.load(str(REAL_DIR / name), sr=SAMPLE_RATE, mono=True)
        seq = stack.embed(audio)
        if seq.shape[0] == 0:
            print(f"  SKIP {name} (too short)")
            continue
        templates_full.append((l2norm(seq.mean(axis=0)), seq.shape[0]))
        cstart = locate_closing_steps(audio, v8)
        cseq = seq[cstart : cstart + CLOSE_REGION_STEPS]
        if cseq.shape[0] < 5:
            cseq = seq[-CLOSE_REGION_STEPS:]
        templates_close.append((l2norm(cseq.mean(axis=0)), CLOSE_REGION_STEPS))
        print(f"  {name[:50]}: {seq.shape[0]} steps, closing@step {cstart}")

    # ---- Cohort: generic-speech centroid for score normalization ----
    # Standard QbE trick: normalized score = cos(template, win) - cos(cohort, win).
    # Subtracting the "how much does this window look like generic speech" term
    # expands the thin raw-cosine margins. Cohort files are EXCLUDED from eval.
    all_speech = sorted(SPEECH_NEG_DIR.glob("*.wav"))
    cohort_files = all_speech[::4][:50]   # every 4th file, 50 total
    cohort_set = {f.name for f in cohort_files}
    print(f"\nBuilding cohort centroid from {len(cohort_files)} speech files...")
    cohort_vecs = []
    for f in cohort_files:
        seq = load_seq(f)
        if seq.shape[0] > 0:
            cohort_vecs.append(l2norm(seq.mean(axis=0)))
    cohort = l2norm(np.mean(cohort_vecs, axis=0))

    def score_file(path: Path):
        seq = load_seq(path)
        # Raw cosines + cohort-normalized deltas for both template variants.
        sf, sf_vec = max(
            (sliding_best_window(seq, t, w) for t, w in templates_full),
            key=lambda r: r[0], default=(0.0, None))
        sc, sc_vec = max(
            (sliding_best_window(seq, t, w) for t, w in templates_close),
            key=lambda r: r[0], default=(0.0, None))
        df = sf - float(np.dot(sf_vec, cohort)) if sf_vec is not None else 0.0
        dc = sc - float(np.dot(sc_vec, cohort)) if sc_vec is not None else 0.0
        return sf, sc, df, dc

    # ---- Positives (held out) ----
    positives = [f for f in sorted(REAL_DIR.glob("*.wav")) if f.name not in ENROLL_NAMES]
    positives += sorted(NEWREC_DIR.glob("*.wav"))
    print(f"\nScoring {len(positives)} held-out positives...")
    pos_rows = []
    for f in positives:
        sf, sc, df, dc = score_file(f)
        pos_rows.append((f.name, sf, sc, df, dc))
        print(f"  full={sf:.3f} close={sc:.3f} dfull={df:+.3f} dclose={dc:+.3f}  {f.name[:50]}")

    # ---- Negatives (cohort files excluded) ----
    neg_files = [f for f in all_speech if f.name not in cohort_set]
    hard_files = []
    for cat in sorted(d for d in HARD_NEG_ROOT.iterdir() if d.is_dir()):
        hard_files += sorted(cat.glob("*.wav"))
    print(f"\nScoring {len(neg_files)} speech + {len(hard_files)} hard negatives...")
    neg_rows = []
    for i, f in enumerate(neg_files + hard_files):
        sf, sc, df, dc = score_file(f)
        kind = "speech" if i < len(neg_files) else f.parent.name
        neg_rows.append((f.name, kind, sf, sc, df, dc))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(neg_files) + len(hard_files)}")

    # ---- Analysis ----
    VARIANTS = [
        ("FULL-CHANT raw cosine", 1, 2),
        ("CLOSING-REGION raw cosine", 2, 3),
        ("FULL-CHANT cohort-normalized", 3, 4),
        ("CLOSING-REGION cohort-normalized", 4, 5),
    ]
    for variant, pidx, nidx in VARIANTS:
        pos_scores = [r[pidx] for r in pos_rows]
        neg_scores = [r[nidx] for r in neg_rows]
        worst_negs = sorted(neg_rows, key=lambda r: -r[nidx])[:8]
        max_neg = max(neg_scores)
        n_pass = sum(1 for s in pos_scores if s > max_neg)
        margin = min(pos_scores) - max_neg
        print(f"\n{'=' * 70}\nVARIANT {variant}\n{'=' * 70}")
        print(f"  positives: min={min(pos_scores):.3f} mean={np.mean(pos_scores):.3f} max={max(pos_scores):.3f}")
        print(f"  negatives: mean={np.mean(neg_scores):.3f} max={max_neg:.3f}")
        print(f"  ZERO-FP threshold = {max_neg:.3f} -> catches {n_pass}/{len(pos_scores)} positives"
              f"  (margin {margin:+.3f})")
        print("  worst negatives:")
        for row in worst_negs:
            print(f"    {row[nidx]:.3f}  [{row[1]}]  {row[0][:50]}")

    # CSV
    import csv
    out = Path("data/real_world_test_v2/qbe_prototype_scores.csv")
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "kind", "full_cos", "close_cos", "full_delta", "close_delta"])
        for name, sf, sc, df, dc in pos_rows:
            w.writerow([name, "positive", f"{sf:.4f}", f"{sc:.4f}", f"{df:.4f}", f"{dc:.4f}"])
        for name, kind, sf, sc, df, dc in neg_rows:
            w.writerow([name, kind, f"{sf:.4f}", f"{sc:.4f}", f"{df:.4f}", f"{dc:.4f}"])
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
