"""
LiteRT vs tf.lite framing + numeric-equivalence check for the qbekws crash.

The two runtimes CANNOT be imported into one Python process (segfault — same
duplicate-runtime conflict as Android), so this script runs per-runtime:

  python scripts/45_litert_framing_check.py --runtime tflite
  python scripts/45_litert_framing_check.py --runtime litert
  python scripts/45_litert_framing_check.py --compare

tflite/litert passes print the samples->frames mapping, test embedding-input
resizability, and dump embeddings of 10 s of live session1 to an .npz.
The litert pass additionally re-runs the separation check (templates from
session1 vs session2 + background) using the SHIPPED imposter JSON, since
that asset was computed under tf.lite and must stay valid on LiteRT devices.
--compare loads both .npz files and reports per-step cosine equivalence.
"""
import argparse
import json
from pathlib import Path

import numpy as np

MELSPEC = "models/openwakeword/melspectrogram.tflite"
EMBEDDING = "models/openwakeword/embedding_model.tflite"
SAMPLE_RATE = 16000
LIVE = Path("data/live_captures")
S1 = LIVE / "navkar_live_2026-06-11_13-42-37.wav"
S2 = LIVE / "navkar_live_2026-06-11_13-44-01.wav"
BG = LIVE / "navkar_live_2026-06-11_13-44-41-backgroundvoices.wav"
IMPOSTERS_JSON = Path(
    "C:/Amit/projects/namobuddy-platform/apps/mobile/assets/models/qbe_imposters.json"
)
TMP = Path("data/live_captures/_runtime_check")

EMB_WINDOW = 76
EMB_STRIDE = 8
WIN_STEPS = 31


def get_interpreter_cls(runtime: str):
    if runtime == "litert":
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter
    import tensorflow as tf
    return tf.lite.Interpreter


def load_wav_16k_mono(path: Path) -> np.ndarray:
    """Stdlib WAV decode — librosa segfaults when imported next to litert."""
    import wave
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SAMPLE_RATE and w.getnchannels() == 1
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def l2(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def window_means(seq):
    cs = np.cumsum(seq, axis=0)
    out = []
    for s in range(seq.shape[0] - WIN_STEPS + 1):
        pooled = (cs[s + WIN_STEPS - 1] - (cs[s - 1] if s > 0 else 0)) / WIN_STEPS
        out.append(l2(pooled))
    return np.array(out, dtype=np.float32)


class Stack:
    """Device-style pipeline replica: the mel input is resized ONCE to the
    smallest N yielding exactly 76 frames under the active runtime, then each
    embedding step feeds a sliding window with a 1280-sample (80 ms) stride —
    identical to QbeKwsModule.kt. (Also avoids the LiteRT segfault seen when
    resizing the mel input to whole-session lengths.)"""

    def __init__(self, cls):
        self.mel = cls(model_path=MELSPEC)
        self.mel_in = self.mel.get_input_details()[0]
        self.mel_out = self.mel.get_output_details()[0]
        # Probe for the runtime's 76-frame input size (tf.lite: 12480, LiteRT: 12640).
        self.mel_samples = None
        for n in [12480, 12640, 12800]:
            self.mel.resize_tensor_input(self.mel_in["index"], [1, n])
            self.mel.allocate_tensors()
            self.mel.set_tensor(self.mel_in["index"], np.zeros((1, n), np.float32))
            self.mel.invoke()
            if self.mel.get_tensor(self.mel_out["index"]).shape[2] == EMB_WINDOW:
                self.mel_samples = n
                break
        if self.mel_samples is None:
            raise RuntimeError("no input size yields 76 mel frames")
        print(f"  (mel input for 76 frames under this runtime: {self.mel_samples})")

        self.emb = cls(model_path=EMBEDDING)
        self.emb.allocate_tensors()
        self.emb_in = self.emb.get_input_details()[0]
        self.emb_out = self.emb.get_output_details()[0]

    def embed(self, audio):
        step = 1280  # 80 ms
        n = self.mel_samples
        embs = []
        for start in range(0, len(audio) - n + 1, step):
            x = audio[start : start + n].astype(np.float32).reshape(1, -1)
            self.mel.set_tensor(self.mel_in["index"], x)
            self.mel.invoke()
            spec = np.squeeze(self.mel.get_tensor(self.mel_out["index"])) / 10.0 + 2.0
            win = spec.reshape(1, EMB_WINDOW, 32, 1).astype(np.float32)
            self.emb.set_tensor(self.emb_in["index"], win)
            self.emb.invoke()
            embs.append(self.emb.get_tensor(self.emb_out["index"]).reshape(96))
        return np.array(embs, dtype=np.float32)


def run_runtime(runtime: str):
    cls = get_interpreter_cls(runtime)
    TMP.mkdir(parents=True, exist_ok=True)

    print(f"RUNTIME: {runtime}")
    print("samples -> mel frames:")
    for n in [12320, 12480, 12640, 12800]:
        it = cls(model_path=MELSPEC)
        d = it.get_input_details()[0]
        it.resize_tensor_input(d["index"], [1, n])
        it.allocate_tensors()
        it.set_tensor(d["index"], np.zeros((1, n), np.float32))
        it.invoke()
        frames = it.get_tensor(it.get_output_details()[0]["index"]).shape[2]
        print(f"  {n} -> {frames}")

    # NOTE: do NOT attempt embedding.resize_tensor_input([1,75,32,1]) — it
    # SEGFAULTS the process under LiteRT (hard native crash, not a Python
    # error). The embedding model requires exactly 76 mel frames; the only
    # viable fix is feeding the melspec model enough samples to produce 76.

    stack = Stack(cls)
    audio = load_wav_16k_mono(S1)
    e = stack.embed(audio[: SAMPLE_RATE * 10])
    np.savez(TMP / f"emb_{runtime}.npz", e=e)
    print(f"dumped {e.shape} embeddings -> emb_{runtime}.npz")

    if runtime == "litert":
        print("\nSEPARATION RE-CHECK UNDER LITERT (shipped imposters JSON):")
        a1 = load_wav_16k_mono(S1)
        a2 = load_wav_16k_mono(S2)
        abg = load_wav_16k_mono(BG)
        w1 = window_means(stack.embed(a1))
        w2 = window_means(stack.embed(a2))
        wbg = window_means(stack.embed(abg))
        templates = w1[::8]
        half = wbg.shape[0] // 2
        bg_cohort, bg_test = wbg[:half], wbg[half:]
        imposters = np.array(json.loads(IMPOSTERS_JSON.read_text())["vectors"], dtype=np.float32)
        competitors = np.concatenate([bg_cohort, imposters], axis=0)
        d2 = (w2 @ templates.T).max(axis=1) - (w2 @ competitors.T).max(axis=1)
        dbg = (bg_test @ templates.T).max(axis=1) - (bg_test @ competitors.T).max(axis=1)
        print(f"  session2 chant: p10={np.percentile(d2,10):+.3f} med={np.median(d2):+.3f}")
        print(f"  background test: max={dbg.max():+.3f}")
        print(f"  thr +0.005: chant above = {(d2 > 0.005).mean()*100:.1f}%  bg FP = {(dbg > 0.005).sum()}/{len(dbg)}")


def compare():
    a = np.load(TMP / "emb_tflite.npz")["e"]
    b = np.load(TMP / "emb_litert.npz")["e"]
    n = min(len(a), len(b))
    cos = [float(np.dot(l2(a[i]), l2(b[i]))) for i in range(n)]
    print(f"steps: tflite={len(a)} litert={len(b)}")
    print(f"per-step cosine(tflite, litert): min={min(cos):.6f} mean={np.mean(cos):.6f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", choices=["tflite", "litert"])
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    if args.compare:
        compare()
    else:
        run_runtime(args.runtime)
