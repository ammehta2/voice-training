"""
Bake MFCC + peak normalization INTO the model.

Why: the mobile app was struggling to reproduce librosa's MFCC in JavaScript
(meyda's mel filterbank diverges, log scaling differs, DCT normalization differs).
Each of those mismatches shifts the model input distribution and breaks the
trained 0.85 threshold. The clean fix is to do the MFCC inside the TFLite graph
using ops that are bit-identical on both sides.

We also add per-utterance peak normalization so the model is robust to the
amplitude difference between training audio (loud) and phone-mic audio (20-50x
quieter). No retraining needed — the math matches librosa exactly.

This script:
  1. Loads a trained closing-phrase ClosingDetector .pt checkpoint.
  2. Wraps it in:
       PeakNormalize -> Conv1d STFT -> Slaney mel filterbank -> 10*log10 ->
       orthonormal DCT-II -> per-utterance mean/std norm -> trained KWS model
  3. Verifies the wrapped output matches the original two-stage pipeline
     (librosa MFCC -> PyTorch model) within float precision.
  4. Exports to ONNX, then to TFLite via onnx2tf.
  5. Verifies the TFLite model produces matching scores on real audio.

Mobile app gets a single TFLite file. Input: raw float32 audio in [-1, 1],
shape [1, 40000]. Output: single raw logit (apply sigmoid for probability).

Usage:
    python scripts/11_export_with_mfcc_head.py models/closing_v1_<TS>/best.pt
"""

import math
import sys
import subprocess
from importlib import import_module
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Constants (must match training; see scripts/04c_extract_features_opening.py) ----
SAMPLE_RATE = 16000
WINDOW_SEC = 2.5
N_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC)  # 40000
N_FFT = 512
HOP_LENGTH = 160
N_MELS = 40
N_MFCC = 40
N_FRAMES = 251
FMIN = 0.0
FMAX = SAMPLE_RATE / 2  # 8000 -- matches librosa default fmax=sr/2


# ---- MFCC head implemented as ONNX-exportable PyTorch ops ----
class LibrosaCompatibleMFCC(nn.Module):
    """Computes MFCC features that match librosa.feature.mfcc(...) bit-for-bit
    (within float32 precision). Uses:

      - STFT implemented as Conv1d with Hann-windowed sine/cosine basis (most
        ONNX/TFLite-friendly — torch.stft has spotty TFLite support).
      - Slaney-normalized mel filterbank, identical numerical content to
        `librosa.filters.mel(..., norm='slaney')`.
      - 10 * log10 clipped at 1e-10 (matches librosa.power_to_db's amin=1e-10).
      - Orthonormal DCT-II (matches scipy.fftpack.dct(type=2, norm='ortho')).
      - Per-utterance mean/std normalization (matches training).

    Input:  (batch, n_samples) raw audio in [-1, 1]
    Output: (batch, n_frames, n_mfcc)
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        n_mels: int = N_MELS,
        n_mfcc: int = N_MFCC,
        fmin: float = FMIN,
        fmax: float = FMAX,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        # ---- STFT as Conv2d (most onnx2tf-friendly) ----
        # We frame the audio using a Conv2d with kernel (n_fft, 1) and stride
        # (hop_length, 1). Each output channel is one DFT bin (Hann-windowed).
        # Using Conv2d on a (B, 1, N+pad, 1) tensor avoids the onnx2tf Conv1d bug.
        window = torch.hann_window(n_fft, periodic=True)  # librosa default
        n = torch.arange(n_fft).float()
        k = torch.arange(n_fft // 2 + 1).float().unsqueeze(-1)  # (F, 1)
        omega = 2.0 * math.pi * k * n / n_fft                    # (F, n_fft)
        cos_basis = torch.cos(omega) * window.unsqueeze(0)       # (F, n_fft)
        sin_basis = -torch.sin(omega) * window.unsqueeze(0)
        # Conv2d weight: (out_channels=F, in_channels=1, kH=n_fft, kW=1)
        self.register_buffer(
            "cos_kernel", cos_basis.unsqueeze(1).unsqueeze(-1).float().contiguous()
        )
        self.register_buffer(
            "sin_kernel", sin_basis.unsqueeze(1).unsqueeze(-1).float().contiguous()
        )

        # ---- Mel filterbank (Slaney, exact librosa values) ----
        mel_fb = librosa.filters.mel(
            sr=sample_rate, n_fft=n_fft, n_mels=n_mels,
            fmin=fmin, fmax=fmax, norm="slaney",
        )  # (n_mels, n_fft//2 + 1)
        self.register_buffer("mel_fb", torch.from_numpy(mel_fb).float())

        # ---- Orthonormal DCT-II matrix ----
        M = n_mels
        K = n_mfcc
        n_idx = torch.arange(M).float()
        k_idx = torch.arange(K).float().unsqueeze(-1)
        dct = torch.cos(math.pi * k_idx * (2.0 * n_idx + 1.0) / (2.0 * M))  # (K, M)
        norms = torch.full((K, 1), math.sqrt(2.0 / M))
        norms[0, 0] = math.sqrt(1.0 / M)
        dct = norms * dct
        self.register_buffer("dct", dct.float())

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: (B, N_SAMPLES)
        # Pad for center=True STFT: reflect-pad n_fft//2 on each side
        pad = self.n_fft // 2
        audio_padded = F.pad(audio.unsqueeze(1), (pad, pad), mode="reflect")
        # (B, 1, N_SAMPLES + n_fft)

        # Reshape to 4D for Conv2d: (B, 1, length, 1)
        x = audio_padded.unsqueeze(-1)  # (B, 1, N+pad, 1)

        # Conv2d with kernel (n_fft, 1) and stride (hop_length, 1)
        # Output: (B, F, T, 1)
        real = F.conv2d(x, self.cos_kernel, stride=(self.hop_length, 1))
        imag = F.conv2d(x, self.sin_kernel, stride=(self.hop_length, 1))
        # Drop the trailing 1 dim -> (B, F, T)
        real = real.squeeze(-1)
        imag = imag.squeeze(-1)
        power = real * real + imag * imag  # (B, F, T)

        # Mel: (B, T, n_mels)
        power_TF = power.transpose(1, 2)  # (B, T, F)
        mel = torch.matmul(power_TF, self.mel_fb.t())  # (B, T, n_mels)

        # Log compression: 10 * log10(max(mel, 1e-10))
        log_mel = 10.0 * torch.log10(torch.clamp(mel, min=1e-10))  # (B, T, n_mels)

        # DCT-II: (B, T, n_mfcc) = log_mel @ dct.T
        # dct shape: (n_mfcc, n_mels), so dct.T: (n_mels, n_mfcc)
        mfcc = torch.matmul(log_mel, self.dct.t())  # (B, T, n_mfcc)

        # Per-utterance mean/std normalization (over T x C, per batch)
        mean = mfcc.mean(dim=(1, 2), keepdim=True)
        # Use population std (matches numpy/librosa default), unbiased=False
        std = mfcc.std(dim=(1, 2), keepdim=True, unbiased=False) + 1e-8
        mfcc = (mfcc - mean) / std

        return mfcc


# ---- Full wrapped model: audio -> peak-normalize -> MFCC head -> KWS backbone -> logit ----
class FullModel(nn.Module):
    def __init__(self, kws_backbone: nn.Module):
        super().__init__()
        self.mfcc_head = LibrosaCompatibleMFCC()
        self.kws = kws_backbone

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # Peak-normalize per utterance to ~0.95 amplitude.
        # This is the single biggest robustness win — handles phone-mic audio that's
        # 20-50x quieter than training data without needing wider amplitude augmentation.
        peak = torch.amax(torch.abs(audio), dim=-1, keepdim=True)
        peak = torch.clamp(peak, min=1e-6)
        audio = audio / peak * 0.95

        mfcc = self.mfcc_head(audio)            # (B, T, F)
        mfcc = mfcc.unsqueeze(1)                # (B, 1, T, F) — channels-first for Conv2d
        return self.kws(mfcc)                   # (B,) logits


def verify_against_librosa(full_model: FullModel, audio_path: Path, kws_backbone: nn.Module):
    """Sanity check: run wrapped pipeline vs the original librosa-based pipeline.
    Scores should match within float precision."""
    print(f"\n=== Verifying against librosa pipeline on {audio_path.name} ===")
    audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    # Use last 2.5s (matches what the model was trained on)
    if len(audio) < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)), mode="constant")
    else:
        audio = audio[-N_SAMPLES:]

    # --- Original pipeline (librosa MFCC -> KWS) ---
    # Note: peak normalize FIRST (matches what FullModel does inside)
    peak = float(np.max(np.abs(audio))) or 1e-6
    audio_norm = audio / peak * 0.95

    mfcc = librosa.feature.mfcc(
        y=audio_norm, sr=SAMPLE_RATE,
        n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
    )
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
    mfcc_T = mfcc.T  # (n_frames, n_mfcc)
    pt_in = torch.from_numpy(mfcc_T[np.newaxis, np.newaxis, :, :]).float()
    with torch.no_grad():
        orig_logit = float(kws_backbone(pt_in).item())

    # --- Wrapped pipeline (audio in -> logit out) ---
    audio_t = torch.from_numpy(audio[np.newaxis, :]).float()  # (1, N_SAMPLES)
    with torch.no_grad():
        wrapped_logit = float(full_model(audio_t).item())

    orig_prob = 1.0 / (1.0 + math.exp(-orig_logit))
    wrap_prob = 1.0 / (1.0 + math.exp(-wrapped_logit))
    diff = abs(orig_prob - wrap_prob)
    print(f"  original (librosa MFCC):     logit={orig_logit:+.4f}  prob={orig_prob:.6f}")
    print(f"  wrapped  (MFCC inside model): logit={wrapped_logit:+.4f}  prob={wrap_prob:.6f}")
    print(f"  probability diff: {diff:.6f}")
    if diff < 0.01:
        print("  OK  wrapped model matches librosa within tolerance")
    else:
        print("  WARN  wrapped model DIVERGES from librosa — investigate before exporting")
    return diff


def export_onnx_tflite(full_model: FullModel, model_dir: Path):
    print(f"\n=== Exporting to ONNX -> TFLite ===")
    full_model.eval()

    # ONNX — fixed batch=1 (mobile only ever runs one window at a time)
    dummy = torch.randn(1, N_SAMPLES)
    onnx_path = model_dir / "navkar_raw_audio.onnx"
    torch.onnx.export(
        full_model, dummy, str(onnx_path),
        input_names=["audio"], output_names=["logit"],
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"  ONNX: {onnx_path}  ({onnx_path.stat().st_size / 1024:.1f} KB)")

    # TFLite via onnx2tf
    print(f"  Running onnx2tf...")
    result = subprocess.run(
        [sys.executable, "-m", "onnx2tf",
         "-i", str(onnx_path),
         "-o", str(model_dir),
         "-nuo"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("  onnx2tf stderr:")
        print(result.stderr[-3000:])
        raise RuntimeError("onnx2tf conversion failed")

    # Rename the produced tflite files with our naming convention
    renames = []
    for src_name, dst_name in [
        ("navkar_raw_audio_float32.tflite", "navkar_raw_audio_fp32.tflite"),
        ("navkar_raw_audio_float16.tflite", "navkar_raw_audio_fp16.tflite"),
    ]:
        src = model_dir / src_name
        dst = model_dir / dst_name
        if src.exists():
            src.replace(dst)
            renames.append(dst)

    print(f"\n=== TFLite files produced ===")
    for tflite in sorted(model_dir.glob("navkar_raw_audio_*.tflite")):
        print(f"  {tflite.name:42s} {tflite.stat().st_size / 1024:6.1f} KB")
    return renames


def verify_tflite(model_dir: Path, audio_path: Path, full_model: FullModel):
    """Run the TFLite raw-audio model vs the PyTorch full model on real audio."""
    print(f"\n=== Verifying TFLite vs PyTorch on {audio_path.name} ===")
    import tensorflow as tf

    audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    # Slide 2.5s window with 250ms hop, take max score
    hop_samples = SAMPLE_RATE // 4  # 4 inferences/sec
    starts = list(range(0, max(1, len(audio) - N_SAMPLES + 1), hop_samples))
    if not starts:
        # Audio shorter than window — pad and use one window
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)), mode="constant")
        starts = [0]

    fp16_path = model_dir / "navkar_raw_audio_fp16.tflite"
    if not fp16_path.exists():
        # Try the unrenamed version
        fp16_path = next(model_dir.glob("*float16*.tflite"), None)
        if fp16_path is None:
            print("  No fp16 TFLite found, skipping TFLite verification")
            return

    interp = tf.lite.Interpreter(model_path=str(fp16_path))
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    print(f"  TFLite input shape: {in_d['shape']} dtype: {in_d['dtype'].__name__}")
    print(f"  TFLite output shape: {out_d['shape']} dtype: {out_d['dtype'].__name__}")
    # Handle dynamic batch axis: resize to (1, N_SAMPLES) before allocating
    interp.resize_tensor_input(in_d['index'], [1, N_SAMPLES], strict=False)
    interp.allocate_tensors()
    # Re-fetch input details (index may have changed)
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]

    pt_max = -1e9
    tf_max = -1e9
    for s in starts:
        chunk = audio[s : s + N_SAMPLES]
        if len(chunk) < N_SAMPLES:
            chunk = np.pad(chunk, (0, N_SAMPLES - len(chunk)), mode="constant")
        chunk_2d = chunk.astype(np.float32).reshape(1, N_SAMPLES)

        # PyTorch
        with torch.no_grad():
            pt_logit = float(full_model(torch.from_numpy(chunk_2d)).item())
        pt_prob = 1.0 / (1.0 + math.exp(-pt_logit))

        # TFLite
        interp.set_tensor(in_d["index"], chunk_2d)
        interp.invoke()
        tf_logit = float(interp.get_tensor(out_d["index"]).flatten()[0])
        tf_prob = 1.0 / (1.0 + math.exp(-tf_logit))

        if pt_prob > pt_max: pt_max = pt_prob
        if tf_prob > tf_max: tf_max = tf_prob

    print(f"  PyTorch max prob over file:  {pt_max:.6f}")
    print(f"  TFLite  max prob over file:  {tf_max:.6f}")
    print(f"  diff: {abs(pt_max - tf_max):.6f}")
    if abs(pt_max - tf_max) < 0.02:
        print("  OK  TFLite matches PyTorch")
    else:
        print("  WARN  TFLite diverges — investigate")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/11_export_with_mfcc_head.py <path/to/best.pt>")
        sys.exit(1)

    pt_path = Path(sys.argv[1])
    if not pt_path.exists():
        print(f"Model not found: {pt_path}")
        sys.exit(1)
    model_dir = pt_path.parent
    print(f"Source PyTorch model: {pt_path}")
    print(f"Output dir: {model_dir}")

    # Load trained KWS backbone
    sys.path.insert(0, str(Path("scripts").resolve()))
    ClosingDetector = import_module("05d_train_closing").ClosingDetector
    kws_backbone = ClosingDetector(dropout=0.0).eval()
    kws_backbone.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
    for p in kws_backbone.parameters():
        p.requires_grad = False
    print(f"Loaded KWS backbone with {sum(p.numel() for p in kws_backbone.parameters()):,} params")

    full_model = FullModel(kws_backbone).eval()
    print(f"Full model params (incl. MFCC head buffers): "
          f"{sum(p.numel() for p in full_model.parameters()):,} trainable, "
          f"{sum(b.numel() for b in full_model.buffers()):,} in buffers")

    # ---- Verify wrapped model matches librosa pipeline on a real recording ----
    test_audio = Path("data/positive_raw_originals/real_navkar_07.wav")
    if test_audio.exists():
        verify_against_librosa(full_model, test_audio, kws_backbone)

    # ---- Export ----
    export_onnx_tflite(full_model, model_dir)

    # ---- Verify TFLite ----
    if test_audio.exists():
        verify_tflite(model_dir, test_audio, full_model)

    print(f"\n=== Done ===")
    print(f"Ship: {model_dir / 'navkar_raw_audio_fp16.tflite'}")
    print(f"Mobile input: float32 array of shape (1, {N_SAMPLES}), values in [-1, 1]")
    print(f"Mobile output: single float32 logit. Apply sigmoid for probability.")


if __name__ == "__main__":
    main()
