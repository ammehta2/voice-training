"""
Augment the adversarial hard-negative corpus and FOLD it directly into the
v7 multi-phrase training data (data/multi/shared/negative_aug/).

Source: data/hard_negatives/<category>/*.wav  (260 raw TTS samples)
Output: data/multi/shared/negative_aug/hard_neg_<category>_<name>_<aug>.wav

Augmentation: same 4 severities as v6/v7 (light/medium/heavy/bandwidth_limited)
plus the original. 8 augmentations per source = ~2080 augmented hard negatives.

These get labeled [0, 0] (negative for both heads) by 04e_extract_features.
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
N_AUGMENTATIONS = 8  # 2 of each severity per source

SRC_ROOT = Path("data/hard_negatives")
DST_DIR = Path("data/multi/shared/negative_aug")


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
        return Compose([
            LowPassFilter(min_cutoff_freq=400, max_cutoff_freq=2000, p=1.0),
            AddGaussianNoise(min_amplitude=0.002, max_amplitude=0.02, p=0.8),
            Gain(min_gain_db=-25, max_gain_db=5, p=0.9),
            PitchShift(min_semitones=-2, max_semitones=2, p=0.4),
        ])
    raise ValueError(f"Unknown severity: {severity}")


def main() -> None:
    DST_DIR.mkdir(parents=True, exist_ok=True)
    severities = ["light", "medium", "heavy", "bandwidth_limited"]

    total_orig = 0
    total_aug = 0
    total_skipped = 0

    categories = sorted([d for d in SRC_ROOT.iterdir() if d.is_dir()])
    for cat_dir in categories:
        files = sorted(cat_dir.glob("*.wav"))
        if not files:
            continue
        print(f"\n{cat_dir.name}: {len(files)} sources -> {len(files) * (N_AUGMENTATIONS + 1)} outputs")

        for f in tqdm(files, desc=cat_dir.name):
            try:
                audio, sr = librosa.load(str(f), sr=SAMPLE_RATE, mono=True)
            except Exception as e:
                print(f"  skip {f.name}: {e}")
                continue

            prefix = f"hard_neg_{cat_dir.name}_{f.stem}"

            # Original
            orig_out = DST_DIR / f"{prefix}_orig.wav"
            if not orig_out.exists():
                sf.write(str(orig_out), audio, sr)
                total_orig += 1
            else:
                total_skipped += 1

            # Augmentations
            for i in range(N_AUGMENTATIONS):
                severity = severities[i % len(severities)]
                aug_out = DST_DIR / f"{prefix}_aug{i:02d}_{severity}.wav"
                if aug_out.exists():
                    total_skipped += 1
                    continue
                try:
                    aug = pipeline_for(severity)(samples=audio, sample_rate=sr)
                    sf.write(str(aug_out), aug, sr)
                    total_aug += 1
                except Exception:
                    pass

    print()
    print(f"Originals written:  {total_orig}")
    print(f"Augmentations:      {total_aug}")
    print(f"Skipped (existed):  {total_skipped}")

    # Final count
    hard_neg_count = len(list(DST_DIR.glob("hard_neg_*.wav")))
    total_neg_count = len(list(DST_DIR.glob("*.wav")))
    print(f"\nIn {DST_DIR}/:")
    print(f"  hard_neg_*.wav:     {hard_neg_count}")
    print(f"  TOTAL negatives:    {total_neg_count}")


if __name__ == "__main__":
    main()
