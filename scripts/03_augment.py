"""
Day 4-5 — Apply audio augmentations to expand the dataset.

Positives: 3 augmentations per file + original = 4x
Negatives: 2 augmentations per file + original = 3x

Critical for model robustness against real-world conditions.

Usage:
    python scripts/03_augment.py
"""

from pathlib import Path

import librosa
import soundfile as sf
from audiomentations import (
    AddGaussianNoise,
    Compose,
    Gain,
    LowPassFilter,
    PitchShift,
    RoomSimulator,
    TimeStretch,
)
from tqdm import tqdm

SAMPLE_RATE = 16000

# Augmentation counts per source file.
# With very few real positives, bump POSITIVE high to expand the training set.
# Negatives are plentiful (ESC-50 = 2000), so a modest multiplier is enough.
POSITIVE_AUGMENTATIONS = 12  # 9 positives x (12+1) = 117 augmented positives
NEGATIVE_AUGMENTATIONS = 2   # 2000 negatives x (2+1) = 6000 augmented negatives


def get_pipeline(severity: str):
    """Returns an audiomentations Compose for the given severity level."""
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
        return Compose([
            AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.03, p=0.8),
            PitchShift(min_semitones=-3, max_semitones=3, p=0.6),
            TimeStretch(min_rate=0.85, max_rate=1.15, p=0.6),
            Gain(min_gain_db=-20, max_gain_db=10, p=0.8),
            LowPassFilter(min_cutoff_freq=2000, max_cutoff_freq=7500, p=0.5),
            RoomSimulator(min_target_rt60=0.15, max_target_rt60=0.8, p=0.5),
        ])

    raise ValueError(f"Unknown severity: {severity}")


def augment_directory(input_dir: str, output_dir: str, num_augmentations: int = 3) -> None:
    """Apply N augmentations per file and write originals + augmented to output."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(input_path.glob("*.wav"))
    if not audio_files:
        print(f"  No files found in {input_dir}")
        return

    print(f"Augmenting {len(audio_files)} files x {num_augmentations} + original "
          f"= {len(audio_files) * (num_augmentations + 1)} outputs")

    severities = ["light", "medium", "heavy"]

    for audio_file in tqdm(audio_files, desc=input_path.name):
        try:
            audio, sr = librosa.load(str(audio_file), sr=SAMPLE_RATE, mono=True)
        except Exception as e:
            print(f"  Skipping {audio_file.name}: {e}")
            continue

        # Copy original
        orig_out = output_path / f"{audio_file.stem}_orig.wav"
        if not orig_out.exists():
            sf.write(str(orig_out), audio, sr)

        # Generate augmentations
        for i in range(num_augmentations):
            severity = severities[i % len(severities)]
            aug_out = output_path / f"{audio_file.stem}_aug{i}_{severity}.wav"
            if aug_out.exists():
                continue

            try:
                pipeline = get_pipeline(severity)
                augmented = pipeline(samples=audio, sample_rate=sr)
                sf.write(str(aug_out), augmented, sr)
            except Exception as e:
                print(f"  Aug failed for {audio_file.name}: {e}")


def main() -> None:
    print(f"Augmenting positives ({POSITIVE_AUGMENTATIONS} per source)...")
    augment_directory("data/positive_raw", "data/positive_aug",
                      num_augmentations=POSITIVE_AUGMENTATIONS)

    print(f"\nAugmenting negatives ({NEGATIVE_AUGMENTATIONS} per source)...")
    augment_directory("data/negative_raw", "data/negative_aug",
                      num_augmentations=NEGATIVE_AUGMENTATIONS)

    pos_total = len(list(Path("data/positive_aug").glob("*.wav")))
    neg_total = len(list(Path("data/negative_aug").glob("*.wav")))
    print(f"\nDone. positives={pos_total}, negatives={neg_total}, total={pos_total + neg_total}")


if __name__ == "__main__":
    main()
