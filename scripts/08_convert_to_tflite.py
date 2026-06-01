"""
Day 11 — Convert trained Keras model to TFLite for mobile deployment.

Produces three variants:
  - navkar_fp32.tflite   (baseline)
  - navkar_fp16.tflite   (recommended for v1 — best size/accuracy balance)
  - navkar_int8.tflite   (smallest, slight accuracy hit)

Uses a subset of the training data for INT8 calibration.

Usage:
    python scripts/08_convert_to_tflite.py models/navkar_v1_<TIMESTAMP>/best.keras
"""

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/08_convert_to_tflite.py <path/to/model.keras>")
        sys.exit(1)

    model_path = Path(sys.argv[1])
    model_dir = model_path.parent

    print(f"Loading: {model_path}")
    model = tf.keras.models.load_model(str(model_path))

    # ---- Float32 baseline ----
    print("\nConverting -> Float32...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_fp32 = converter.convert()
    fp32_path = model_dir / "navkar_fp32.tflite"
    fp32_path.write_bytes(tflite_fp32)
    print(f"  {fp32_path.name}: {fp32_path.stat().st_size / 1024:.1f} KB")

    # ---- Float16 (recommended) ----
    print("\nConverting -> Float16...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_fp16 = converter.convert()
    fp16_path = model_dir / "navkar_fp16.tflite"
    fp16_path.write_bytes(tflite_fp16)
    print(f"  {fp16_path.name}: {fp16_path.stat().st_size / 1024:.1f} KB")

    # ---- INT8 (smallest) ----
    print("\nConverting -> INT8 (with calibration)...")
    X_train = np.load("data/features/X_train.npy")[..., np.newaxis]
    representative_data = X_train[:300]

    def representative_dataset():
        for sample in representative_data:
            yield [sample[np.newaxis, ...].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_int8 = converter.convert()
    int8_path = model_dir / "navkar_int8.tflite"
    int8_path.write_bytes(tflite_int8)
    print(f"  {int8_path.name}: {int8_path.stat().st_size / 1024:.1f} KB")

    print("\n=== Conversion summary ===")
    print(f"  Float32: {fp32_path.stat().st_size / 1024:7.1f} KB  {fp32_path}")
    print(f"  Float16: {fp16_path.stat().st_size / 1024:7.1f} KB  {fp16_path}")
    print(f"  INT8:    {int8_path.stat().st_size / 1024:7.1f} KB  {int8_path}")
    print("\nRecommendation: ship navkar_fp16.tflite for v1.")
    print("Use INT8 only if app bundle size is critical.")


if __name__ == "__main__":
    main()
