"""
Augment the multi-phrase (v7) training data.

For each positive directory (closing, pre-closing) generate N augmented
variants per source. For the shared negative directory, generate fewer
augmentations since we already have ~24K source negatives.

Same audiomentations pipelines as 03d_augment_closing.py (including the
v6 bandwidth_limited severity that proved critical for mobile audio).
"""

from pathlib import Path

import librosa
import soundfile as sf
from audiomentations import (
    AddGaussianNoise, Compose, Gain, LowPassFilter, PitchShift, TimeStretch,
)
try:
    from audiomentations import RoomSimulator
    HAS_ROOM = True
except ImportError:
    HAS_ROOM = False
from tqdm import tqdm

SAMPLE_RATE = 16000
POSITIVE_AUGMENTATIONS = 8       # 2985 x 9 = ~27K per positive class
NEGATIVE_AUGMENTATIONS = 1       # 24K x 2 = ~48K negatives total

DIRS = [
    (Path("data/multi/closing/positive_raw"),     Path("data/multi/closing/positive_aug"),     POSITIVE_AUGMENTATIONS),
    (Path("data/multi/preclosing/positive_raw"),  Path("data/multi/preclosing/positive_aug"),  POSITIVE_AUGMENTATIONS),
    (Path("data/multi/shared/negative_raw"),      Path("data/multi/shared/negative_aug"),      NEGATIVE_AUGMENTATIONS),
]


def pipeline_for(severity: str):
    if severity == "light":
        return Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.005, p=0.5),
            PitchShift(min_semitones=-1, max_semitones=1, p=0.3),
            Gain(min_gain_db=-6, max_gain_db=6, p=0.5),
        ])
    if severity == "medium":
        return Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.7),
            PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
            TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5),
            Gain(min_gain_db=-12, max_gain_db=6, p=0.7),
            LowPassFilter(min_cutoff_freq=3000, max_cutoff_freq=7500, p=0.3),
        ])
    if severity == "heavy":
        ops = [
            AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.03, p=0.8),
            PitchShift(min_semitones=-3, max_semitones=3, p=0.6),
            TimeStretch(min_rate=0.85, max_rate=1.15, p=0.6),
            Gain(min_gain_db=-20, max_gain_db=10, p=0.8),
            LowPassFilter(min_cutoff_freq=2000, max_cutoff_freq=7500, p=0.5),
        ]
        if HAS_ROOM:
            ops.append(RoomSimulator(min_target_rt60=0.15, max_target_rt60=0.8, p=0.5))
        return Compose(ops)
    if severity == "bandwidth_limited":
        # The v6 win — simulates mobile pipeline that low-passes audio to ~500Hz
        ops = [
            LowPassFilter(min_cutoff_freq=400, max_cutoff_freq=2000, p=1.0),
            AddGaussianNoise(min_amplitude=0.002, max_amplitude=0.02, p=0.8),
            Gain(min_gain_db=-25, max_gain_db=5, p=0.9),
            PitchShift(min_semitones=-2, max_semitones=2, p=0.4),
        ]
        return Compose(ops)
    raise ValueError(f"Unknown severity: {severity}")


def augment_directory(input_dir: Path, output_dir: Path, num_augmentations: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.glob("*.wav"))
    if not files:
        print(f"  No files in {input_dir}")
        return

    print(f"  {input_dir.name}: {len(files)} sources x {num_augmentations} aug + orig = "
          f"{len(files) * (num_augmentations + 1)} outputs")
    severities = ["light", "medium", "heavy", "bandwidth_limited"]

    for f in tqdm(files, desc=input_dir.name):
        try:
            audio, sr = librosa.load(str(f), sr=SAMPLE_RATE, mono=True)
        except Exception as e:
            print(f"  skip {f.name}: {e}")
            continue

        orig_out = output_dir / f"{f.stem}_orig.wav"
        if not orig_out.exists():
            sf.write(str(orig_out), audio, sr)

        for i in range(num_augmentations):
            severity = severities[i % len(severities)]
            aug_out = output_dir / f"{f.stem}_aug{i:02d}_{severity}.wav"
            if aug_out.exists():
                continue
            try:
                aug = pipeline_for(severity)(samples=audio, sample_rate=sr)
                sf.write(str(aug_out), aug, sr)
            except Exception:
                pass  # heavy aug occasionally fails on short clips


def main() -> None:
    for in_dir, out_dir, n in DIRS:
        print(f"\nAugmenting {in_dir.name} ({n} per source)...")
        augment_directory(in_dir, out_dir, n)

    print("\nFinal counts:")
    for in_dir, out_dir, _ in DIRS:
        n = len(list(out_dir.glob("*.wav")))
        print(f"  {out_dir}: {n}")


if __name__ == "__main__":
    main()
