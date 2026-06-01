"""
Day 3 — Gather and process negative training samples (non-Navkar audio).

Pulls from:
  1. Mozilla Common Voice (Hindi) — conversational speech
  2. ESC-50 — environmental sounds (auto-downloads)
  3. TTS-generated other Jain mantras (Logassa, Chattari, Uvasaggaharam)

All output is normalized to 16kHz mono WAV in data/negative_raw/.

Usage:
    python scripts/02_gather_negatives.py
"""

import random
import subprocess
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

NEGATIVE_DIR = Path("data/negative_raw")
SOURCES_DIR = Path("data/sources")
VOICE_SAMPLES_DIR = Path("voice_samples")

# Other Jain mantras — should NOT trigger Navkar detector
OTHER_MANTRAS = {
    "logassa": (
        "Loassa Ujjoa-gare, Dhammatithayare Jine. Arihante kittaissam, Chauvisam pi Kevali. "
        "Usabhamajiyam cha Vande, Sambhavamabhinandanam cha Sumaim cha. Paumappaham Supasam, "
        "Jinam cha Chandappaham Vande."
    ),
    "chattari": (
        "Chattari Mangalam, Arihanta Mangalam, Siddha Mangalam, Sahu Mangalam, "
        "Kevali Pannatto Dhammo Mangalam. Chattari Logguttama, Arihanta Loguttama."
    ),
    "uvasaggaharam": (
        "Uvasaggaharam Pasam, Pasam Vandami Kamma-ghana-mukkam. "
        "Visahara Visa Ninnasam, Mangala Kallana Avasam."
    ),
}


def run_ffmpeg(input_path: Path, output_path: Path) -> bool:
    """Convert audio to 16kHz mono WAV. Returns True on success."""
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(input_path),
                "-ar", "16000", "-ac", "1",
                str(output_path),
            ],
            capture_output=True,
            timeout=30,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  ffmpeg failed for {input_path.name}: {e}")
        return False


def process_common_voice(count: int = 500) -> int:
    """Convert Common Voice clips to negatives.

    Expects extracted Common Voice corpus at:
        data/sources/common_voice/cv-corpus-*/hi/clips/*.mp3
    """
    pattern_roots = list(SOURCES_DIR.glob("common_voice/cv-corpus-*/hi/clips"))
    if not pattern_roots:
        print(f"  Common Voice not found under {SOURCES_DIR / 'common_voice'}/")
        print("  Download Hindi subset from https://commonvoice.mozilla.org/en/datasets")
        return 0

    clips = []
    for root in pattern_roots:
        clips.extend(root.glob("*.mp3"))

    if not clips:
        print("  Common Voice directory exists but contains no .mp3 files")
        return 0

    print(f"  Found {len(clips)} Common Voice clips, sampling {count}")
    random.shuffle(clips)

    converted = 0
    for i, clip in enumerate(tqdm(clips[: count * 2], desc="  Common Voice", leave=False)):
        if converted >= count:
            break
        out_path = NEGATIVE_DIR / f"speech_{i:04d}.wav"
        if out_path.exists():
            converted += 1
            continue
        if run_ffmpeg(clip, out_path):
            converted += 1

    return converted


def download_esc50() -> Path | None:
    """Download and extract ESC-50 dataset."""
    extract_path = SOURCES_DIR / "esc50"
    if (extract_path / "ESC-50-master").exists():
        print("  ESC-50 already downloaded")
        return extract_path / "ESC-50-master"

    extract_path.mkdir(parents=True, exist_ok=True)
    zip_path = SOURCES_DIR / "esc50.zip"

    if not zip_path.exists():
        url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
        print(f"  Downloading ESC-50 from {url}")
        try:
            response = requests.get(url, stream=True, timeout=60)
            total = int(response.headers.get("content-length", 0))
            with open(zip_path, "wb") as f:
                with tqdm(total=total, unit="B", unit_scale=True, leave=False) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
        except Exception as e:
            print(f"  Download failed: {e}")
            return None

    print("  Extracting ESC-50...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_path)

    return extract_path / "ESC-50-master"


def process_esc50(count: int = 600) -> int:
    """Convert ESC-50 environmental sounds to negatives."""
    esc50_root = download_esc50()
    if esc50_root is None:
        return 0

    audio_dir = esc50_root / "audio"
    clips = list(audio_dir.glob("*.wav"))
    if not clips:
        print(f"  No audio files found in {audio_dir}")
        return 0

    random.shuffle(clips)
    print(f"  Processing {min(count, len(clips))} ESC-50 clips")

    converted = 0
    for i, clip in enumerate(tqdm(clips[:count], desc="  ESC-50", leave=False)):
        out_path = NEGATIVE_DIR / f"ambient_{i:04d}.wav"
        if out_path.exists():
            converted += 1
            continue
        if run_ffmpeg(clip, out_path):
            converted += 1

    return converted


def generate_other_mantras() -> int:
    """Use TTS to generate other Jain mantras as negatives."""
    try:
        import torch
        from TTS.api import TTS
    except ImportError:
        print("  TTS not installed; skipping other-mantra generation")
        return 0

    voice_samples = sorted(VOICE_SAMPLES_DIR.glob("*.wav"))
    if not voice_samples:
        print(f"  No voice samples in {VOICE_SAMPLES_DIR}/; skipping")
        return 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading XTTS-v2 on {device}...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    voices = voice_samples[:5]
    samples_per_combo = 30
    total_target = len(OTHER_MANTRAS) * len(voices) * samples_per_combo
    print(f"  Generating up to {total_target} other-mantra negatives")

    generated = 0
    for mantra_name, text in OTHER_MANTRAS.items():
        for voice in tqdm(voices, desc=f"  {mantra_name}", leave=False):
            for i in range(samples_per_combo):
                out_path = NEGATIVE_DIR / f"mantra_{mantra_name}_{voice.stem}_{i:03d}.wav"
                if out_path.exists():
                    continue
                try:
                    tts.tts_to_file(
                        text=text,
                        speaker_wav=str(voice),
                        language="hi",
                        file_path=str(out_path),
                    )
                    generated += 1
                except Exception as e:
                    print(f"    Failed: {e}")

    return generated


def main() -> None:
    NEGATIVE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCES_DIR.mkdir(parents=True, exist_ok=True)

    print("\nStep 1: Common Voice speech samples")
    cv_count = process_common_voice(count=500)
    print(f"  -> {cv_count} samples")

    print("\nStep 2: ESC-50 environmental sounds")
    esc_count = process_esc50(count=600)
    print(f"  -> {esc_count} samples")

    print("\nStep 3: Other Jain mantras (TTS)")
    mantra_count = generate_other_mantras()
    print(f"  -> {mantra_count} samples")

    total = len(list(NEGATIVE_DIR.glob("*.wav")))
    print(f"\nTotal negatives in {NEGATIVE_DIR}: {total}")


if __name__ == "__main__":
    main()
