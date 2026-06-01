"""
Augmentation for the opening-phrase pipeline.

Same approach as 03_augment.py but:
  - operates on data/opening/{positive_raw,negative_raw}
  - heavier on positives (only 5 sources, so we need 20x augmentation)
  - lighter on negatives (already 4125 source files)

Usage:
    python scripts/03c_augment_opening.py
"""

from pathlib import Path

import librosa
import soundfile as sf
from audiomentations import (
    AddGaussianNoise, Compose, Gain, LowPassFilter, PitchShift, TimeStretch,
)
# Optional: RoomSimulator — sometimes flaky on very short clips
try:
    from audiomentations import RoomSimulator
    HAS_ROOM = True
except ImportError:
    HAS_ROOM = False

from tqdm import tqdm

SAMPLE_RATE = 16000
POSITIVE_AUGMENTATIONS = 20    # 5 sources × 21 (incl. orig) = 105 positives
NEGATIVE_AUGMENTATIONS = 1     # 4125 × 2 (incl. orig) = ~8200 negatives

POS_IN = Path("data/opening/positive_raw")
POS_OUT = Path("data/opening/positive_aug")
NEG_IN = Path("data/opening/negative_raw")
NEG_OUT = Path("data/opening/negative_aug")


def pipeline_for(severity: str):
    """Augmentation pipelines tuned for short (2.5s) clips."""
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
    raise ValueError(f"Unknown severity: {severity}")


def augment_directory(input_dir: Path, output_dir: Path, num_augmentations: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.glob("*.wav"))
    if not files:
        print(f"  No files in {input_dir}")
        return

    print(f"  {len(files)} sources × {num_augmentations} aug + orig = "
          f"{len(files) * (num_augmentations + 1)} outputs")
    severities = ["light", "medium", "heavy"]

    for f in tqdm(files, desc=input_dir.name):
        try:
            audio, sr = librosa.load(str(f), sr=SAMPLE_RATE, mono=True)
        except Exception as e:
            print(f"  skip {f.name}: {e}")
            continue

        # Original
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
            except Exception as e:
                pass  # heavy aug sometimes fails on tiny clips; skip silently


def main() -> None:
    print(f"Augmenting positives ({POSITIVE_AUGMENTATIONS} per source)...")
    augment_directory(POS_IN, POS_OUT, POSITIVE_AUGMENTATIONS)
    print(f"\nAugmenting negatives ({NEGATIVE_AUGMENTATIONS} per source)...")
    augment_directory(NEG_IN, NEG_OUT, NEGATIVE_AUGMENTATIONS)

    pos = len(list(POS_OUT.glob("*.wav")))
    neg = len(list(NEG_OUT.glob("*.wav")))
    print(f"\nDone. positives={pos}, negatives={neg}, total={pos + neg}")


if __name__ == "__main__":
    main()
