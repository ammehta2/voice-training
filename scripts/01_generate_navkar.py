"""
Day 1-2 — Generate synthetic Navkar Mantra chants using Coqui XTTS-v2.

Reads reference voices from voice_samples/, produces ~1500 audio files
in data/positive_raw/. Resume-friendly (skips files that already exist).

Usage:
    python scripts/01_generate_navkar.py
"""

import random
from pathlib import Path

import torch
from TTS.api import TTS
from tqdm import tqdm

NAVKAR_TEXT = """
Namo Arihantaanam.
Namo Siddhaanam.
Namo Aayariyaanam.
Namo Uvajjhaayaanam.
Namo Loe Savva Saahoonam.
Eso Pancha Namokkaaro, Savva Paavappanaasano.
Mangalaanam cha Savvesim, Padhamam Havai Mangalam.
"""

NAVKAR_VARIANTS = [
    NAVKAR_TEXT,
    NAVKAR_TEXT.replace("Aayariyaanam", "Ayariyaanam"),
    NAVKAR_TEXT.replace("Saahoonam", "Sahunam"),
    NAVKAR_TEXT.replace(".", ","),
    NAVKAR_TEXT.replace(".", "..."),
]

OUTPUT_DIR = Path("data/positive_raw")
VOICE_SAMPLES_DIR = Path("voice_samples")
TARGET_SAMPLES = 1500


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading XTTS-v2 model on {device}...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    voice_files = sorted(VOICE_SAMPLES_DIR.glob("*.wav"))
    if len(voice_files) < 5:
        print(f"WARNING: Only {len(voice_files)} voice samples in {VOICE_SAMPLES_DIR}/")
        print("Add at least 10 voice samples (5-15s each) before continuing.")
        print("See voice_samples/README.md for sourcing options.")
        return

    print(f"Found {len(voice_files)} reference voices")
    samples_per_voice = TARGET_SAMPLES // len(voice_files)
    print(f"Generating {samples_per_voice} samples per voice -> ~{samples_per_voice * len(voice_files)} total")

    generated = 0
    skipped = 0

    for voice_file in tqdm(voice_files, desc="Voices"):
        voice_name = voice_file.stem

        for i in range(samples_per_voice):
            output_path = OUTPUT_DIR / f"{voice_name}_{i:04d}.wav"
            if output_path.exists():
                skipped += 1
                continue

            text = random.choice(NAVKAR_VARIANTS)
            speed = random.uniform(0.85, 1.15)

            try:
                tts.tts_to_file(
                    text=text,
                    speaker_wav=str(voice_file),
                    language="hi",  # Hindi closest to Prakrit pronunciation
                    file_path=str(output_path),
                    speed=speed,
                )
                generated += 1
            except Exception as e:
                print(f"  Failed {output_path.name}: {e}")

    print(f"\nDone. Generated {generated} new samples ({skipped} already existed).")
    print(f"Total files in {OUTPUT_DIR}: {len(list(OUTPUT_DIR.glob('*.wav')))}")


if __name__ == "__main__":
    main()
