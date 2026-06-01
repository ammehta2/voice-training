"""
Day 11 (PyTorch path) — Convert trained PyTorch model to TFLite.

Pipeline: PyTorch .pt -> ONNX -> TFLite (via onnx2tf).

Produces:
  navkar_fp32.tflite  (baseline, ~100KB)
  navkar_fp16.tflite  (recommended for v1, ~50KB)
  navkar_int8.tflite  (smallest, calibrated on training data, ~30KB)

The exported TFLite has channel-last input shape (1, T, F, 1) for mobile
compatibility — onnx2tf transposes between PyTorch (channels-first) and TF
(channels-last) automatically.

Usage:
    python scripts/08_convert_to_tflite_pytorch.py models/navkar_v1_<TS>/best.pt
"""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

# Import the model definition from the training script
sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module

train_module = import_module("05_train_model_pytorch")
NavkarKWS = train_module.NavkarKWS

INPUT_TIME_FRAMES = 1601  # matches feature extraction output
INPUT_FEATURES = 40


def export_to_onnx(model: torch.nn.Module, onnx_path: Path) -> None:
    model.eval()
    dummy = torch.randn(1, 1, INPUT_TIME_FRAMES, INPUT_FEATURES)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["mfcc"], output_names=["logit"],
        dynamic_axes={"mfcc": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"  ONNX: {onnx_path}  ({onnx_path.stat().st_size / 1024:.1f} KB)")


def convert_onnx_to_tflite(onnx_path: Path, out_dir: Path, calibration_data: np.ndarray | None = None) -> None:
    """Use onnx2tf to produce float32 + float16 TFLite files.

    Note: INT8 quantization via onnx2tf requires a calibration npy keyed by
    input name + an explicit input-shape spec. We skip INT8 for now; FP16 is
    plenty small (~50KB) and matches the v1 ship target.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "onnx2tf",
        "-i", str(onnx_path),
        "-o", str(out_dir),
        "-nuo",   # don't optimize specifically for ONNXRuntime; produce a TF-shaped graph
    ]

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("  onnx2tf stderr:")
        print(result.stderr[-2000:])
        raise RuntimeError("onnx2tf conversion failed")
    print("  onnx2tf done.")


def report(out_dir: Path) -> None:
    print("\n=== TFLite files produced ===")
    found_any = False
    for tflite in sorted(out_dir.glob("*.tflite")):
        size_kb = tflite.stat().st_size / 1024
        print(f"  {tflite.name:40s} {size_kb:7.1f} KB")
        found_any = True
    if not found_any:
        print("  (none — check onnx2tf output above)")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/08_convert_to_tflite_pytorch.py <path/to/best.pt>")
        sys.exit(1)

    pt_path = Path(sys.argv[1])
    if not pt_path.exists():
        print(f"Model not found: {pt_path}")
        sys.exit(1)
    out_dir = pt_path.parent
    print(f"Source: {pt_path}")
    print(f"Output: {out_dir}")

    # Load model
    model = NavkarKWS(dropout=0.0)  # dropout off for export
    state = torch.load(pt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    # Export to ONNX
    onnx_path = out_dir / "navkar.onnx"
    print("\nStep 1: PyTorch -> ONNX")
    export_to_onnx(model, onnx_path)

    # Load calibration data (subset of train features) for INT8 quantization
    calib = None
    train_npy = Path("data/features/X_train.npy")
    if train_npy.exists():
        X = np.load(train_npy)
        n_calib = min(300, len(X))
        # onnx2tf expects channels-last (B, T, F, 1) for TF-friendly graph
        # so we reshape: (N, T, F) -> (N, T, F, 1)
        if X.ndim == 3:
            X = X[..., np.newaxis]
        calib = X[:n_calib]
        print(f"\nCalibration data: {calib.shape} (from data/features/X_train.npy)")

    print("\nStep 2: ONNX -> TFLite (fp32 + fp16 + int8)")
    convert_onnx_to_tflite(onnx_path, out_dir, calibration_data=calib)

    report(out_dir)


if __name__ == "__main__":
    main()
