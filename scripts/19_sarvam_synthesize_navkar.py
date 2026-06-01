"""
Generate synthetic Navkar Mantra recordings using Sarvam AI's Bulbul TTS.

Why: 26 real recordings limits voice diversity. The model misses recordings
with voice characteristics unlike the training set (recordings 04, 05, 08, 20
consistently fail). Sarvam Bulbul provides 12 native Indian voices and high
audio quality.

Critical constraints (per user feedback):
  - NOT singing -- spoken recitation cadence only
  - Fast jaap pace: 8-14 seconds per Navkar
  - All 12 Sarvam voices for diversity
  - Multiple pace variants

API:    https://api.sarvam.ai/text-to-speech
Auth:   header `api-subscription-key`
Model:  bulbul:v2 (current latest)
Cost:   ~$0.005-0.01 per request; 500 samples ~= $3-5

Output: data/positive_raw_originals/sarvam_navkar_<voice>_<pace>_<i>.wav
        (will be picked up by 00d_prep_closing_phrase.py for closing extraction)

Usage:
    # Set API key first:
    set SARVAM_API_KEY=sk_xxxxx                            (PowerShell: $env:SARVAM_API_KEY=...)
    # Generate 10 test samples (verify NOT singing):
    python scripts/19_sarvam_synthesize_navkar.py --test
    # If quality good, generate full set:
    python scripts/19_sarvam_synthesize_navkar.py --n-per-voice 40
"""

import argparse
import base64
import io
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import requests
import soundfile as sf
from tqdm import tqdm

SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
SAMPLE_RATE = 16000
OUT_DIR = Path("data/positive_raw_originals")

# Navkar Mantra in Devanagari (Hindi-readable transliteration of Prakrit).
# IMPORTANT: removed all punctuation (। and ॥) so the TTS produces continuous
# chanting without inter-line pauses. Real jaap flows without breaks between
# "Namo Arihantaanam / Namo Siddhaanam / ..." -- TTS that inserts commas/full
# stops sounds wrong.
NAVKAR_TEXT = (
    "नमो अरिहंताणं नमो सिद्धाणं नमो आयरियाणं "
    "नमो उवज्झायाणं नमो लोए सव्व साहूणं "
    "एसो पंच नमोक्कारो सव्व पावप्पणासणो "
    "मंगलाणं च सव्वेसिं पढमं हवइ मंगलं"
)

# Sarvam Bulbul v2 voices (7 official voices as of late 2025).
# Source: API error message when an unsupported speaker is sent.
# Mix of male and female; sufficient diversity for our use case.
VOICES = [
    "anushka", "manisha", "vidya", "arya",   # female-sounding
    "abhilash", "karun", "hitesh",            # male-sounding
]

# Pace variants -- targets user's 8-14s jaap range.
# Empirical observation: pace 1.2 ~= 9s, 1.4 ~= 8s, 1.0 ~= 11s, 0.8 ~= 14s.
PACE_VARIANTS = [0.85, 1.0, 1.15, 1.3]

# Min/max acceptable duration for a generated chant
MIN_DURATION_S = 7.0
MAX_DURATION_S = 16.0


def get_api_key() -> str:
    key = os.environ.get("SARVAM_API_KEY")
    if not key:
        print("ERROR: SARVAM_API_KEY environment variable not set")
        print("  Set it via:")
        print("    set SARVAM_API_KEY=<your_key>          (cmd.exe)")
        print("    $env:SARVAM_API_KEY = '<your_key>'    (PowerShell)")
        print("    export SARVAM_API_KEY=<your_key>      (bash)")
        sys.exit(1)
    return key


def synthesize(text: str, voice: str, pace: float, api_key: str, language: str = "hi-IN") -> bytes | None:
    """Call Sarvam Bulbul. Returns raw 22050 Hz WAV bytes, or None on failure."""
    payload = {
        "inputs": [text],
        "target_language_code": language,
        "speaker": voice,
        "pitch": 0.0,          # No melodic variation -- speech, not singing
        "pace": pace,
        "loudness": 1.0,
        "speech_sample_rate": 22050,
        # Disable Sarvam's text preprocessing -- their default may add commas/pauses
        # that we explicitly want to avoid for continuous chanting.
        "enable_preprocessing": False,
        "model": "bulbul:v2",
    }
    headers = {
        "api-subscription-key": api_key,
        "Content-Type": "application/json",
    }
    try:
        r = requests.post(SARVAM_TTS_URL, json=payload, headers=headers, timeout=60)
    except Exception as e:
        print(f"  Network error: {e}")
        return None

    if r.status_code != 200:
        body = r.text[:300] if r.text else ""
        print(f"  HTTP {r.status_code}: {body}")
        return None

    data = r.json()
    # Sarvam returns {"audios": ["<base64-encoded-wav>", ...]}
    audios = data.get("audios") or []
    if not audios:
        print(f"  No audio in response: {data}")
        return None
    return base64.b64decode(audios[0])


def to_16khz_mono(wav_bytes: bytes) -> np.ndarray | None:
    """Decode WAV bytes and resample to 16kHz mono via ffmpeg."""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ti:
        ti.write(wav_bytes)
        ip = ti.name
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as to:
        op = to.name
    try:
        r = subprocess.run(
            [ffmpeg, "-y", "-i", ip, "-ar", str(SAMPLE_RATE), "-ac", "1",
             "-c:a", "pcm_s16le", op],
            capture_output=True, timeout=30,
        )
        if r.returncode != 0:
            return None
        audio, _ = sf.read(op)
        return audio.astype(np.float32)
    finally:
        try: Path(ip).unlink()
        except: pass
        try: Path(op).unlink()
        except: pass


def peak_normalize(audio: np.ndarray, target: float = 0.95) -> np.ndarray:
    p = float(np.max(np.abs(audio)))
    return (audio / p * target).astype(np.float32) if p > 1e-6 else audio.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Generate just one sample per voice (12 samples) for quality check.")
    parser.add_argument("--n-per-voice", type=int, default=40,
                        help="Samples to generate per voice (when not in test mode). "
                             "Total = n_per_voice * 12 voices.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retry-on-bad-duration", type=int, default=2,
                        help="Re-generate if duration outside [%d, %d]s (random pace each attempt)" %
                             (int(MIN_DURATION_S), int(MAX_DURATION_S)))
    args = parser.parse_args()

    api_key = get_api_key()
    random.seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n_per_voice = 1 if args.test else args.n_per_voice
    total_target = n_per_voice * len(VOICES)
    print(f"Generating {n_per_voice} samples per voice x {len(VOICES)} voices = {total_target} total")
    print(f"Pace variants: {PACE_VARIANTS}")
    print(f"Output: {OUT_DIR}/")
    print()

    generated = 0
    failed = 0
    skipped = 0
    bad_duration = 0
    bad_quality = 0
    total_cost_estimate = 0.0  # rough

    pbar = tqdm(total=total_target, desc="TTS")
    for voice in VOICES:
        for i in range(n_per_voice):
            # Try up to retry_on_bad_duration attempts to hit duration window
            success = False
            for attempt in range(args.retry_on_bad_duration + 1):
                pace = random.choice(PACE_VARIANTS)
                pace_int = int(pace * 10)
                out_path = OUT_DIR / f"sarvam_navkar_{voice}_{pace_int}_{i:02d}.wav"

                if out_path.exists():
                    skipped += 1
                    success = True
                    break

                wav_bytes = synthesize(NAVKAR_TEXT, voice, pace, api_key)
                total_cost_estimate += 0.008  # rough estimate

                if wav_bytes is None:
                    failed += 1
                    break

                audio = to_16khz_mono(wav_bytes)
                if audio is None or len(audio) < SAMPLE_RATE:
                    bad_quality += 1
                    continue

                dur = len(audio) / SAMPLE_RATE
                if dur < MIN_DURATION_S or dur > MAX_DURATION_S:
                    bad_duration += 1
                    continue  # try again with different pace

                # Peak normalize and save
                audio = peak_normalize(audio)
                sf.write(str(out_path), audio, SAMPLE_RATE, subtype="PCM_16")
                generated += 1
                success = True
                break

            pbar.update(1)
            pbar.set_postfix(gen=generated, fail=failed, bad_dur=bad_duration)

            # Brief sleep to avoid rate limits
            if not args.test:
                time.sleep(0.1)
    pbar.close()

    print()
    print(f"Generated: {generated}, Skipped (already existed): {skipped}, Failed: {failed}")
    print(f"Bad duration retries: {bad_duration}, Bad quality: {bad_quality}")
    print(f"Rough cost estimate: ~${total_cost_estimate:.2f}")
    sarvam_files = sorted(OUT_DIR.glob("sarvam_navkar_*.wav"))
    print(f"\nTotal Sarvam files in {OUT_DIR}: {len(sarvam_files)}")

    # Quality summary
    if generated > 0:
        print(f"\nQuality check on generated samples:")
        durs = []
        for f in sarvam_files[-min(20, generated):]:
            audio, _ = sf.read(str(f))
            durs.append(len(audio) / SAMPLE_RATE)
        print(f"  durations: mean={np.mean(durs):.1f}s, min={min(durs):.1f}s, max={max(durs):.1f}s")
        if args.test:
            print(f"\n*** TEST MODE: Listen to a sample BEFORE running the full generation ***")
            print(f"*** Verify it sounds like RECITATION not SINGING               ***")
            print(f"*** Sample to check: {sarvam_files[-1] if sarvam_files else 'N/A'} ***")


if __name__ == "__main__":
    main()
