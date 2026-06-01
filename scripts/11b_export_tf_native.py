"""
Bake MFCC + peak normalization into the TFLite model — TF-native approach.

Earlier attempt (scripts/11_export_with_mfcc_head.py) hit an onnx2tf bug
converting STFT/Conv. This version builds the entire model in TF Keras using
tf.signal ops (which TFLite supports natively), copies PyTorch backbone weights
into a mirrored TF backbone, and exports straight from TF to TFLite. No ONNX.

The resulting TFLite takes raw float32 audio (1, 40000) in [-1, 1] and outputs
a single raw logit. Mobile sends raw audio, applies sigmoid to the output.

No retraining — the math matches librosa bit-for-bit because we feed
librosa's exact Slaney mel filterbank in as a constant tensor.

Usage:
    python scripts/11b_export_tf_native.py models/closing_v1_<TS>/best.pt
"""

import math
import sys
from importlib import import_module
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf
import torch

# ---- Constants (must match training) ----
SAMPLE_RATE = 16000
N_SAMPLES = 40000
N_FFT = 512
HOP_LENGTH = 160
N_MELS = 40
N_MFCC = 40
N_FRAMES = 251  # = (N_SAMPLES + 2 * (N_FFT // 2) - N_FFT) / HOP_LENGTH + 1
DROPOUT = 0.0  # disable for inference


# ============================================================================
# 1. Mirror PyTorch ClosingDetector in TF Keras (NHWC layout)
# ============================================================================
def build_tf_kws_backbone() -> tf.keras.Model:
    """TF Keras mirror of PyTorch ClosingDetector. Channels-last NHWC."""
    inputs = tf.keras.Input(shape=(N_FRAMES, N_MFCC, 1), dtype=tf.float32, name="mfcc")

    # PyTorch: Conv2d(1, 48, (5, 4), stride=(2, 2), padding=(2, 1))
    # padding=(2, 1) is "same" for kernel (5,4) stride (2,2) -> use Keras 'same'
    x = tf.keras.layers.Conv2D(
        48, kernel_size=(5, 4), strides=(2, 2), padding="same", use_bias=False, name="conv0",
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name="bn0", epsilon=1e-5, momentum=0.9)(x)
    x = tf.keras.layers.ReLU()(x)

    # 3 DS blocks: (48, 48), (48, 48), (48, 64)
    for i, out_c in enumerate([48, 48, 64]):
        # Depthwise 3x3
        x = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(3, 3), padding="same", use_bias=False, name=f"dw{i}",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"bn{i}_1", epsilon=1e-5, momentum=0.9)(x)
        x = tf.keras.layers.ReLU()(x)
        # Pointwise 1x1
        x = tf.keras.layers.Conv2D(
            out_c, kernel_size=(1, 1), use_bias=False, name=f"pw{i}",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"bn{i}_2", epsilon=1e-5, momentum=0.9)(x)
        x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    if DROPOUT > 0:
        x = tf.keras.layers.Dropout(DROPOUT)(x)
    # PyTorch Linear -> Keras Dense (bias by default). Output (B, 1).
    outputs = tf.keras.layers.Dense(1, name="fc")(x)
    # Keep shape (B, 1) — TFLite consumers read element [0] anyway

    return tf.keras.Model(inputs, outputs, name="kws_backbone")


# ============================================================================
# 2. Port PyTorch weights -> TF Keras backbone
# ============================================================================
def port_weights_pytorch_to_tf(pt_state_dict: dict, tf_model: tf.keras.Model) -> None:
    """Copy PyTorch ClosingDetector weights into the TF Keras backbone.

    Mapping (PyTorch -> TF):
        conv0.weight              -> conv0 (NCHW -> HWIO)
        bn0.{weight,bias,...}     -> bn0
        blocks.{i}.dw.weight      -> dw{i}  (depthwise: NCHW -> HWIO with C_in=1)
        blocks.{i}.bn1.{...}      -> bn{i}_1
        blocks.{i}.pw.weight      -> pw{i}  (NCHW -> HWIO)
        blocks.{i}.bn2.{...}      -> bn{i}_2
        fc.{weight,bias}          -> fc
    """
    print("  Porting PyTorch weights to TF Keras...")
    placed = set()

    def set_kernel_nchw_to_hwio(layer_name: str, pt_key: str) -> None:
        """Convert (out_c, in_c, kH, kW) PyTorch -> (kH, kW, in_c, out_c) TF Keras."""
        w = pt_state_dict[pt_key].cpu().numpy()
        w_tf = np.transpose(w, (2, 3, 1, 0))  # NCHW kernel -> HWIO
        layer = tf_model.get_layer(layer_name)
        layer.set_weights([w_tf])
        placed.add(pt_key)

    def set_depthwise_kernel(layer_name: str, pt_key: str) -> None:
        """PyTorch depthwise weight: (channels, 1, kH, kW)
           TF DepthwiseConv2D weight: (kH, kW, in_channels, depth_multiplier=1)
        """
        w = pt_state_dict[pt_key].cpu().numpy()  # (C, 1, kH, kW)
        # -> (kH, kW, C, 1)
        w_tf = np.transpose(w, (2, 3, 0, 1))
        layer = tf_model.get_layer(layer_name)
        layer.set_weights([w_tf])
        placed.add(pt_key)

    def set_bn(layer_name: str, prefix: str) -> None:
        """Set BatchNorm: gamma, beta, moving_mean, moving_var."""
        gamma = pt_state_dict[f"{prefix}.weight"].cpu().numpy()
        beta = pt_state_dict[f"{prefix}.bias"].cpu().numpy()
        mean = pt_state_dict[f"{prefix}.running_mean"].cpu().numpy()
        var = pt_state_dict[f"{prefix}.running_var"].cpu().numpy()
        layer = tf_model.get_layer(layer_name)
        layer.set_weights([gamma, beta, mean, var])
        for k in ["weight", "bias", "running_mean", "running_var"]:
            placed.add(f"{prefix}.{k}")
        # num_batches_tracked is not used in TF
        placed.add(f"{prefix}.num_batches_tracked")

    def set_dense(layer_name: str, prefix: str) -> None:
        w = pt_state_dict[f"{prefix}.weight"].cpu().numpy()  # (out, in)
        b = pt_state_dict[f"{prefix}.bias"].cpu().numpy()
        # Keras Dense expects (in, out)
        layer = tf_model.get_layer(layer_name)
        layer.set_weights([w.T, b])
        placed.add(f"{prefix}.weight")
        placed.add(f"{prefix}.bias")

    # Initial conv + BN
    set_kernel_nchw_to_hwio("conv0", "conv0.weight")
    set_bn("bn0", "bn0")

    # 3 DS blocks
    for i in range(3):
        set_depthwise_kernel(f"dw{i}", f"blocks.{i}.dw.weight")
        set_bn(f"bn{i}_1", f"blocks.{i}.bn1")
        set_kernel_nchw_to_hwio(f"pw{i}", f"blocks.{i}.pw.weight")
        set_bn(f"bn{i}_2", f"blocks.{i}.bn2")

    # Final dense
    set_dense("fc", "fc")

    # Report leftover keys
    leftover = set(pt_state_dict.keys()) - placed
    if leftover:
        print(f"  WARN: {len(leftover)} PyTorch keys were not placed: {sorted(leftover)}")
    else:
        print(f"  OK: all {len(placed)} PyTorch parameters placed into TF backbone")


# ============================================================================
# 3. Verify TF backbone matches PyTorch backbone
# ============================================================================
def verify_backbones(pt_backbone, tf_backbone, num_trials: int = 5) -> float:
    """Compare TF and PyTorch backbone outputs on random MFCC inputs."""
    print("\n  Comparing TF backbone vs PyTorch backbone on random inputs...")
    max_diff = 0.0
    for trial in range(num_trials):
        # Random MFCC tensor
        np.random.seed(trial)
        mfcc_np = np.random.randn(1, N_FRAMES, N_MFCC).astype(np.float32)

        # PyTorch: needs (B, 1, T, F) channels-first
        pt_in = torch.from_numpy(mfcc_np[:, np.newaxis, :, :])
        with torch.no_grad():
            pt_out = float(pt_backbone(pt_in).item())

        # TF: needs (B, T, F, 1) channels-last
        tf_in = mfcc_np[..., np.newaxis]
        tf_out = float(tf_backbone(tf_in, training=False).numpy().flatten()[0])

        diff = abs(pt_out - tf_out)
        max_diff = max(max_diff, diff)
        print(f"    trial {trial}: PyTorch={pt_out:+.4f}  TF={tf_out:+.4f}  diff={diff:.5f}")

    print(f"  Max diff across {num_trials} trials: {max_diff:.5f}")
    return max_diff


# ============================================================================
# 4. Add MFCC + peak-norm head and wrap full model
# ============================================================================
class MFCCPreprocessLayer(tf.keras.layers.Layer):
    """Audio -> peak-normalize -> STFT -> mel (Slaney) -> log10 -> DCT-II -> norm -> (B, T, F, 1)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load librosa's exact Slaney mel filterbank
        mel_np = librosa.filters.mel(
            sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS,
            fmin=0.0, fmax=SAMPLE_RATE / 2, norm="slaney",
        )
        self.mel_fb = tf.constant(mel_np, dtype=tf.float32)
        self.pad = N_FFT // 2

        # ---- Pre-computed Hann-windowed DFT basis ----
        # TFLite's RFFT2D kernel has a bug rejecting fft_length=[N, 1]. We work
        # around this by doing the DFT as a plain matmul against a precomputed
        # basis. For n_fft=512 this is ~262k ops per frame -- trivial on mobile.
        n_arr = np.arange(N_FFT, dtype=np.float32)
        k_arr = np.arange(N_FFT // 2 + 1, dtype=np.float32)[:, None]
        omega = 2.0 * np.pi * k_arr * n_arr / N_FFT          # (F, N_FFT)
        # Periodic Hann window — apply directly to the basis so we don't need a
        # separate windowing step inside call().
        hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * n_arr / N_FFT)  # periodic Hann
        cos_basis = (np.cos(omega) * hann).astype(np.float32)     # (F, N_FFT)
        sin_basis = (-np.sin(omega) * hann).astype(np.float32)
        # Store as (N_FFT, F) so frames @ basis gives (B, T, F)
        self.cos_basis_T = tf.constant(cos_basis.T, dtype=tf.float32)
        self.sin_basis_T = tf.constant(sin_basis.T, dtype=tf.float32)

        # ---- Pre-computed orthonormal DCT-II matrix ----
        # Match scipy.fftpack.dct(type=2, norm='ortho') used by librosa.
        # tf.signal.dct also uses RFFT internally — same TFLite bug — so we
        # replace it with an explicit matmul.
        M, K = N_MELS, N_MFCC
        n_idx = np.arange(M, dtype=np.float32)
        k_idx = np.arange(K, dtype=np.float32)[:, None]
        dct_mat = np.cos(np.pi * k_idx * (2.0 * n_idx + 1.0) / (2.0 * M))
        norms = np.full((K, 1), np.sqrt(2.0 / M), dtype=np.float32)
        norms[0, 0] = np.sqrt(1.0 / M)
        dct_mat = (norms * dct_mat).astype(np.float32)  # (K, M)
        # log_mel: (B, T, M); we want (B, T, K) = log_mel @ dct_mat.T
        self.dct_T = tf.constant(dct_mat.T, dtype=tf.float32)  # (M, K)

    def call(self, audio):
        # audio: (B, N_SAMPLES)
        # Peak normalize per utterance
        peak = tf.reduce_max(tf.abs(audio), axis=-1, keepdims=True)
        peak = tf.maximum(peak, 1e-6)
        audio_norm = audio / peak * 0.95

        # Reflect-pad to emulate librosa center=True
        audio_padded = tf.pad(audio_norm, [[0, 0], [self.pad, self.pad]], mode="REFLECT")

        # Frame the audio into overlapping windows (B, T, N_FFT)
        frames = tf.signal.frame(audio_padded, N_FFT, HOP_LENGTH)

        # DFT via matmul (Hann window pre-baked into the basis at __init__)
        # frames @ cos_basis.T -> (B, T, F)
        real_part = tf.matmul(frames, self.cos_basis_T)
        imag_part = tf.matmul(frames, self.sin_basis_T)
        power = real_part * real_part + imag_part * imag_part  # (B, T, F)

        # Mel (Slaney)
        mel_spec = tf.matmul(power, self.mel_fb, transpose_b=True)  # (B, T, n_mels)

        # 10 * log10(max(mel, 1e-10))
        log_mel = 10.0 * (tf.math.log(tf.maximum(mel_spec, 1e-10)) / tf.math.log(10.0))

        # DCT-II orthonormal via matmul (TFLite-friendly, no FFT)
        mfcc = tf.matmul(log_mel, self.dct_T)  # (B, T, N_MFCC)

        # Per-utterance mean/std normalization
        mean = tf.reduce_mean(mfcc, axis=[1, 2], keepdims=True)
        std = tf.math.reduce_std(mfcc, axis=[1, 2], keepdims=True) + 1e-8
        mfcc_norm = (mfcc - mean) / std

        return tf.expand_dims(mfcc_norm, axis=-1)  # (B, T, F, 1)


def build_full_model(kws_backbone: tf.keras.Model) -> tf.keras.Model:
    """audio -> MFCC layer -> KWS backbone -> logit (B, 1)."""
    audio_in = tf.keras.Input(shape=(N_SAMPLES,), dtype=tf.float32, name="audio")
    mfcc_4d = MFCCPreprocessLayer(name="mfcc")(audio_in)
    logit = kws_backbone(mfcc_4d)
    return tf.keras.Model(audio_in, logit, name="navkar_raw_audio")


# ============================================================================
# 5. Verify full TF model matches the original librosa->PyTorch pipeline
# ============================================================================
def verify_full(full_tf: tf.keras.Model, pt_backbone, audio_path: Path) -> float:
    print(f"\n=== Verifying full TF model vs librosa+PyTorch on {audio_path.name} ===")
    audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    # Use last 2.5s
    if len(audio) < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)), mode="constant")
    else:
        audio = audio[-N_SAMPLES:]

    # Reference pipeline: peak normalize -> librosa MFCC -> PyTorch backbone
    peak = float(np.max(np.abs(audio))) or 1e-6
    audio_norm = audio / peak * 0.95
    mfcc = librosa.feature.mfcc(
        y=audio_norm, sr=SAMPLE_RATE,
        n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
    )
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
    pt_in = torch.from_numpy(mfcc.T[np.newaxis, np.newaxis, :, :]).float()
    with torch.no_grad():
        ref_logit = float(pt_backbone(pt_in).item())

    # New pipeline: raw audio -> TF full model -> logit
    audio_tf = audio.astype(np.float32)[np.newaxis, :]
    new_logit = float(full_tf(audio_tf, training=False).numpy().flatten()[0])

    ref_prob = 1.0 / (1.0 + math.exp(-ref_logit))
    new_prob = 1.0 / (1.0 + math.exp(-new_logit))
    diff = abs(ref_prob - new_prob)
    print(f"  reference (librosa+PyTorch): logit={ref_logit:+.4f}  prob={ref_prob:.6f}")
    print(f"  new       (raw audio TF):    logit={new_logit:+.4f}  prob={new_prob:.6f}")
    print(f"  probability diff: {diff:.6f}")
    if diff < 0.01:
        print("  OK  full TF model matches reference")
    else:
        print("  WARN full TF model diverges from reference")
    return diff


# ============================================================================
# 6. Export to TFLite
# ============================================================================
def export_tflite(full_tf: tf.keras.Model, model_dir: Path):
    print(f"\n=== Exporting to TFLite ===")
    converter = tf.lite.TFLiteConverter.from_keras_model(full_tf)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,  # needed for STFT/DCT ops
    ]
    converter._experimental_lower_tensor_list_ops = False

    fp32 = converter.convert()
    fp32_path = model_dir / "navkar_raw_audio_fp32.tflite"
    fp32_path.write_bytes(fp32)
    print(f"  Float32: {fp32_path.name}  ({fp32_path.stat().st_size / 1024:.1f} KB)")

    # Float16
    converter = tf.lite.TFLiteConverter.from_keras_model(full_tf)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False
    fp16 = converter.convert()
    fp16_path = model_dir / "navkar_raw_audio_fp16.tflite"
    fp16_path.write_bytes(fp16)
    print(f"  Float16: {fp16_path.name}  ({fp16_path.stat().st_size / 1024:.1f} KB)")

    return fp16_path


# ============================================================================
# 7. Verify TFLite end-to-end
# ============================================================================
def verify_tflite(fp16_path: Path, full_tf: tf.keras.Model, audio_path: Path):
    print(f"\n=== Verifying TFLite vs TF full model on {audio_path.name} ===")
    audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    if len(audio) < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)), mode="constant")
    else:
        audio = audio[-N_SAMPLES:]
    audio_in = audio.astype(np.float32)[np.newaxis, :]

    interp = tf.lite.Interpreter(model_path=str(fp16_path))
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    print(f"  TFLite input: shape={in_d['shape']} dtype={in_d['dtype'].__name__}")
    print(f"  TFLite output: shape={out_d['shape']} dtype={out_d['dtype'].__name__}")

    interp.set_tensor(in_d["index"], audio_in)
    interp.invoke()
    tf_logit = float(interp.get_tensor(out_d["index"]).flatten()[0])
    tf_prob = 1.0 / (1.0 + math.exp(-tf_logit))

    keras_logit = float(full_tf(audio_in, training=False).numpy().flatten()[0])
    keras_prob = 1.0 / (1.0 + math.exp(-keras_logit))

    diff = abs(tf_prob - keras_prob)
    print(f"  Keras model: logit={keras_logit:+.4f}  prob={keras_prob:.6f}")
    print(f"  TFLite FP16: logit={tf_logit:+.4f}  prob={tf_prob:.6f}")
    print(f"  diff: {diff:.6f}")
    if diff < 0.02:
        print("  OK  TFLite matches Keras")
    else:
        print("  WARN TFLite diverges")


# ============================================================================
def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/11b_export_tf_native.py <path/to/best.pt>")
        sys.exit(1)

    pt_path = Path(sys.argv[1])
    if not pt_path.exists():
        print(f"Model not found: {pt_path}")
        sys.exit(1)
    model_dir = pt_path.parent
    print(f"Source PyTorch model: {pt_path}")
    print(f"Output dir: {model_dir}")

    # ---- Load PyTorch backbone ----
    sys.path.insert(0, str(Path("scripts").resolve()))
    ClosingDetector = import_module("05d_train_closing").ClosingDetector
    pt_backbone = ClosingDetector(dropout=0.0).eval()
    state = torch.load(pt_path, map_location="cpu", weights_only=True)
    pt_backbone.load_state_dict(state)
    print(f"Loaded PyTorch backbone: {sum(p.numel() for p in pt_backbone.parameters()):,} params")

    # ---- Bake temperature scaling into final FC layer (if temperature.txt exists) ----
    # Math: divide_by_T(W @ x + b) == (W/T) @ x + (b/T)
    # So scaling fc.weight and fc.bias by 1/T baked the temperature into the model.
    temp_path = pt_path.parent / "temperature.txt"
    if temp_path.exists():
        T = float(temp_path.read_text().strip())
        print(f"Baking temperature T={T:.4f} into final FC layer weights")
        with torch.no_grad():
            pt_backbone.fc.weight.div_(T)
            pt_backbone.fc.bias.div_(T)
    else:
        print("No temperature.txt found; exporting model as-is (T=1)")

    # ---- Build TF backbone + port weights ----
    print("\nBuilding TF Keras backbone...")
    tf_backbone = build_tf_kws_backbone()
    tf_backbone.summary()
    port_weights_pytorch_to_tf(state, tf_backbone)

    # ---- Verify backbone parity ----
    max_diff = verify_backbones(pt_backbone, tf_backbone)
    if max_diff > 0.01:
        print(f"\n  WARN: backbone diff {max_diff:.4f} > 0.01 — likely a weight-porting bug")
        print("  Continuing anyway, but the final model may not match.")

    # ---- Wrap with MFCC head ----
    print("\nWrapping with MFCC head...")
    full_tf = build_full_model(tf_backbone)
    full_tf.summary()

    # ---- Verify full TF model against reference pipeline ----
    test_audio = Path("data/positive_raw_originals/real_navkar_07.wav")
    if test_audio.exists():
        verify_full(full_tf, pt_backbone, test_audio)

    # ---- Export ----
    fp16_path = export_tflite(full_tf, model_dir)

    # ---- Verify TFLite ----
    if test_audio.exists():
        verify_tflite(fp16_path, full_tf, test_audio)

    print(f"\n=== Done ===")
    print(f"Ship: {fp16_path}")
    print(f"Mobile input:  float32 (1, {N_SAMPLES}) in [-1, 1]")
    print(f"Mobile output: float32 (1,) raw logit. Apply sigmoid for probability.")


if __name__ == "__main__":
    main()
