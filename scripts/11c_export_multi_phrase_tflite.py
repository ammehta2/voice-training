"""
Export the v7 multi-phrase MultiPhraseDetector to a raw-audio TFLite model.

Same approach as 11b_export_tf_native.py:
  - Build TF Keras backbone mirroring the PyTorch architecture
  - Port weights over
  - Wrap with the MFCC + peak normalization head
  - Export to TFLite

Only difference: TWO output heads instead of one. Output shape (1, 2).

Mobile inference:
  out = sigmoid(model(audio))  # shape (2,)
  closing_prob = out[0]
  preclosing_prob = out[1]
"""

import math
import sys
from importlib import import_module
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf
import torch

SAMPLE_RATE = 16000
N_SAMPLES = 40000
N_FFT = 512
HOP_LENGTH = 160
N_MELS = 40
N_MFCC = 40
N_FRAMES = 251
DROPOUT = 0.0


def build_tf_kws_backbone() -> tf.keras.Model:
    """TF Keras mirror of MultiPhraseDetector. Two output heads."""
    inputs = tf.keras.Input(shape=(N_FRAMES, N_MFCC, 1), dtype=tf.float32, name="mfcc")

    x = tf.keras.layers.Conv2D(
        48, kernel_size=(5, 4), strides=(2, 2), padding="same", use_bias=False, name="conv0",
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name="bn0", epsilon=1e-5, momentum=0.9)(x)
    x = tf.keras.layers.ReLU()(x)

    for i, out_c in enumerate([48, 48, 64]):
        x = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(3, 3), padding="same", use_bias=False, name=f"dw{i}",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"bn{i}_1", epsilon=1e-5, momentum=0.9)(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(
            out_c, kernel_size=(1, 1), use_bias=False, name=f"pw{i}",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"bn{i}_2", epsilon=1e-5, momentum=0.9)(x)
        x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Two heads, concatenated to shape (B, 2)
    closing = tf.keras.layers.Dense(1, name="fc_closing")(x)
    preclosing = tf.keras.layers.Dense(1, name="fc_preclosing")(x)
    outputs = tf.keras.layers.Concatenate(axis=-1)([closing, preclosing])

    return tf.keras.Model(inputs, outputs, name="multi_phrase_backbone")


def port_weights_pytorch_to_tf(pt_state_dict: dict, tf_model: tf.keras.Model) -> None:
    print("  Porting PyTorch weights to TF Keras...")
    placed = set()

    def set_kernel(layer_name: str, pt_key: str) -> None:
        w = pt_state_dict[pt_key].cpu().numpy()
        w_tf = np.transpose(w, (2, 3, 1, 0))
        tf_model.get_layer(layer_name).set_weights([w_tf])
        placed.add(pt_key)

    def set_dw(layer_name: str, pt_key: str) -> None:
        w = pt_state_dict[pt_key].cpu().numpy()
        w_tf = np.transpose(w, (2, 3, 0, 1))
        tf_model.get_layer(layer_name).set_weights([w_tf])
        placed.add(pt_key)

    def set_bn(layer_name: str, prefix: str) -> None:
        tf_model.get_layer(layer_name).set_weights([
            pt_state_dict[f"{prefix}.weight"].cpu().numpy(),
            pt_state_dict[f"{prefix}.bias"].cpu().numpy(),
            pt_state_dict[f"{prefix}.running_mean"].cpu().numpy(),
            pt_state_dict[f"{prefix}.running_var"].cpu().numpy(),
        ])
        for k in ["weight", "bias", "running_mean", "running_var"]:
            placed.add(f"{prefix}.{k}")
        placed.add(f"{prefix}.num_batches_tracked")

    def set_dense(layer_name: str, prefix: str) -> None:
        w = pt_state_dict[f"{prefix}.weight"].cpu().numpy()
        b = pt_state_dict[f"{prefix}.bias"].cpu().numpy()
        tf_model.get_layer(layer_name).set_weights([w.T, b])
        placed.add(f"{prefix}.weight"); placed.add(f"{prefix}.bias")

    set_kernel("conv0", "conv0.weight")
    set_bn("bn0", "bn0")
    for i in range(3):
        set_dw(f"dw{i}", f"blocks.{i}.dw.weight")
        set_bn(f"bn{i}_1", f"blocks.{i}.bn1")
        set_kernel(f"pw{i}", f"blocks.{i}.pw.weight")
        set_bn(f"bn{i}_2", f"blocks.{i}.bn2")
    set_dense("fc_closing", "fc_closing")
    set_dense("fc_preclosing", "fc_preclosing")

    leftover = set(pt_state_dict.keys()) - placed
    if leftover:
        print(f"  WARN: leftover keys: {sorted(leftover)}")
    else:
        print(f"  OK: all {len(placed)} keys placed")


def verify_backbones(pt_backbone, tf_backbone, n_trials=5) -> float:
    max_diff = 0.0
    for trial in range(n_trials):
        np.random.seed(trial)
        mfcc_np = np.random.randn(1, N_FRAMES, N_MFCC).astype(np.float32)
        pt_in = torch.from_numpy(mfcc_np[:, np.newaxis, :, :])
        with torch.no_grad():
            pt_out = pt_backbone(pt_in).numpy()[0]
        tf_in = mfcc_np[..., np.newaxis]
        tf_out = tf_backbone(tf_in, training=False).numpy()[0]
        diff = float(np.abs(pt_out - tf_out).max())
        max_diff = max(max_diff, diff)
        print(f"    trial {trial}: PyTorch={pt_out}  TF={tf_out}  diff={diff:.5f}")
    print(f"  Max diff: {max_diff:.5f}")
    return max_diff


class MFCCPreprocessLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        mel_np = librosa.filters.mel(
            sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS,
            fmin=0.0, fmax=SAMPLE_RATE / 2, norm="slaney",
        )
        self.mel_fb = tf.constant(mel_np, dtype=tf.float32)
        self.pad = N_FFT // 2

        n_arr = np.arange(N_FFT, dtype=np.float32)
        k_arr = np.arange(N_FFT // 2 + 1, dtype=np.float32)[:, None]
        omega = 2.0 * np.pi * k_arr * n_arr / N_FFT
        hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * n_arr / N_FFT)
        cos_basis = (np.cos(omega) * hann).astype(np.float32)
        sin_basis = (-np.sin(omega) * hann).astype(np.float32)
        self.cos_basis_T = tf.constant(cos_basis.T, dtype=tf.float32)
        self.sin_basis_T = tf.constant(sin_basis.T, dtype=tf.float32)

        M, K = N_MELS, N_MFCC
        n_idx = np.arange(M, dtype=np.float32)
        k_idx = np.arange(K, dtype=np.float32)[:, None]
        dct_mat = np.cos(np.pi * k_idx * (2.0 * n_idx + 1.0) / (2.0 * M))
        norms = np.full((K, 1), np.sqrt(2.0 / M), dtype=np.float32)
        norms[0, 0] = np.sqrt(1.0 / M)
        dct_mat = (norms * dct_mat).astype(np.float32)
        self.dct_T = tf.constant(dct_mat.T, dtype=tf.float32)

    def call(self, audio):
        peak = tf.reduce_max(tf.abs(audio), axis=-1, keepdims=True)
        peak = tf.maximum(peak, 1e-6)
        audio_norm = audio / peak * 0.95
        audio_padded = tf.pad(audio_norm, [[0, 0], [self.pad, self.pad]], mode="REFLECT")
        frames = tf.signal.frame(audio_padded, N_FFT, HOP_LENGTH)
        real_part = tf.matmul(frames, self.cos_basis_T)
        imag_part = tf.matmul(frames, self.sin_basis_T)
        power = real_part * real_part + imag_part * imag_part
        mel_spec = tf.matmul(power, self.mel_fb, transpose_b=True)
        log_mel = 10.0 * (tf.math.log(tf.maximum(mel_spec, 1e-10)) / tf.math.log(10.0))
        mfcc = tf.matmul(log_mel, self.dct_T)
        mean = tf.reduce_mean(mfcc, axis=[1, 2], keepdims=True)
        std = tf.math.reduce_std(mfcc, axis=[1, 2], keepdims=True) + 1e-8
        return tf.expand_dims((mfcc - mean) / std, axis=-1)


def build_full_model(kws_backbone: tf.keras.Model) -> tf.keras.Model:
    audio_in = tf.keras.Input(shape=(N_SAMPLES,), dtype=tf.float32, name="audio")
    mfcc_4d = MFCCPreprocessLayer(name="mfcc")(audio_in)
    logits = kws_backbone(mfcc_4d)  # (B, 2)
    return tf.keras.Model(audio_in, logits, name="navkar_multi_phrase")


def verify_full(full_tf, pt_backbone, audio_path: Path):
    print(f"\n=== Verifying full TF model on {audio_path.name} ===")
    audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    if len(audio) < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)), mode="constant")
    else:
        audio = audio[-N_SAMPLES:]

    # Reference: librosa MFCC -> pytorch backbone
    peak = float(np.max(np.abs(audio))) or 1e-6
    audio_norm = audio / peak * 0.95
    mfcc = librosa.feature.mfcc(
        y=audio_norm, sr=SAMPLE_RATE,
        n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
    )
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
    pt_in = torch.from_numpy(mfcc.T[np.newaxis, np.newaxis, :, :]).float()
    with torch.no_grad():
        ref_logits = pt_backbone(pt_in).numpy()[0]

    audio_tf = audio.astype(np.float32)[np.newaxis, :]
    new_logits = full_tf(audio_tf, training=False).numpy()[0]

    print(f"  reference: closing={ref_logits[0]:+.4f}  preclosing={ref_logits[1]:+.4f}")
    print(f"  new:       closing={new_logits[0]:+.4f}  preclosing={new_logits[1]:+.4f}")
    diff = float(np.abs(ref_logits - new_logits).max())
    print(f"  max diff: {diff:.6f}")
    return diff


def export_tflite(full_tf: tf.keras.Model, model_dir: Path):
    print(f"\n=== Exporting to TFLite ===")
    converter = tf.lite.TFLiteConverter.from_keras_model(full_tf)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter._experimental_lower_tensor_list_ops = False
    fp32 = converter.convert()
    fp32_path = model_dir / "navkar_multi_phrase_fp32.tflite"
    fp32_path.write_bytes(fp32)
    print(f"  Float32: {fp32_path.name}  ({fp32_path.stat().st_size / 1024:.1f} KB)")

    converter = tf.lite.TFLiteConverter.from_keras_model(full_tf)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter._experimental_lower_tensor_list_ops = False
    fp16 = converter.convert()
    fp16_path = model_dir / "navkar_multi_phrase_fp16.tflite"
    fp16_path.write_bytes(fp16)
    print(f"  Float16: {fp16_path.name}  ({fp16_path.stat().st_size / 1024:.1f} KB)")
    return fp16_path, fp32_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/11c_export_multi_phrase_tflite.py <path/to/best.pt>")
        sys.exit(1)

    pt_path = Path(sys.argv[1])
    model_dir = pt_path.parent
    print(f"Source: {pt_path}")
    print(f"Output: {model_dir}")

    sys.path.insert(0, str(Path("scripts").resolve()))
    MultiPhraseDetector = import_module("05g_train_multi_phrase").MultiPhraseDetector
    pt_backbone = MultiPhraseDetector(dropout=0.0).eval()
    state = torch.load(pt_path, map_location="cpu", weights_only=True)
    pt_backbone.load_state_dict(state)
    print(f"Loaded PyTorch backbone")

    print("\nBuilding TF Keras backbone...")
    tf_backbone = build_tf_kws_backbone()
    port_weights_pytorch_to_tf(state, tf_backbone)
    verify_backbones(pt_backbone, tf_backbone)

    print("\nWrapping with MFCC head...")
    full_tf = build_full_model(tf_backbone)

    test_audio = Path("data/positive_raw_originals/real_navkar_07.wav")
    if test_audio.exists():
        verify_full(full_tf, pt_backbone, test_audio)

    fp16_path, fp32_path = export_tflite(full_tf, model_dir)

    print(f"\n=== Done ===")
    print(f"Ship: {fp32_path}")
    print(f"Mobile input:  float32 (1, 40000)")
    print(f"Mobile output: float32 (1, 2)  -- [closing_logit, preclosing_logit]")
    print(f"Apply sigmoid to each output element separately.")


if __name__ == "__main__":
    main()
