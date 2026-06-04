"""
Adversarial hard-negative TTS generator.

Generates 6 categories of "looks-like-Navkar-but-isn't" phrases using
Sarvam Bulbul TTS. These probe whether the model is overfitting on:
  - The word "Namo" + rhythm (closing-head shortcut)
  - The word "Mangalam" + rhythm (preclosing-head shortcut)
  - Repetition of Navkar's opening lines without the closing

Categories:
  1. mangalam_jain  - "Mangalam Bhagwan Veero..." (the killer test;
                      every Jain household recites this; uses "Mangalam"
                      pattern that overlaps with Navkar's closing)
  2. buddhist_namo  - "Namo Tassa Bhagavato Arahato Samma Sambuddhassa"
  3. hindu_namo     - "Om Namo Bhagavate Vasudevaya" / "Om Namah Shivaya"
                      / "Om Namo Narayanaya"
  4. truncated      - Navkar opening lines WITHOUT the closing (the
                      counter must NOT count partial chants)
  5. other_jain     - Logassa, Chattari Mangalam (Jain prayers that
                      share vocabulary but aren't Navkar)
  6. spoken_about   - Hindi commentary mentioning "Navkar Mantra"
                      (someone saying "I will now recite the Navkar..."
                      should NOT trigger a count)

Output: data/hard_negatives/<category>/sarvam_<phrase_id>_<voice>_<pace>.wav
"""

import argparse
import base64
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
OUT_ROOT = Path("data/hard_negatives")

VOICES = ["anushka", "manisha", "vidya", "arya", "abhilash", "karun", "hitesh"]
PACE_VARIANTS = [0.85, 1.0, 1.15, 1.3]

MIN_DURATION_S = 5.0
MAX_DURATION_S = 20.0

# ============================================================
# Phrase corpus
# Each (id, devanagari_text). 2-3 variants per category for diversity.
# All written for ~5-15s recitation at neutral pace.
# Punctuation stripped: TTS produces continuous flow without pauses.
# ============================================================

PHRASES = {
    # "Mangalam X" — overlap with Navkar's closing word
    "mangalam_jain": [
        ("mangalam_bhagwan", "मंगलं भगवान वीरो मंगलं गौतम गणधरो मंगलं स्थूलिभद्राद्या जैने धर्मो अस्तु मंगलं"),
        ("mangalam_bhagwan_short", "मंगलं भगवान वीरो मंगलं गौतम गणधरो मंगलं स्थूलिभद्राद्या"),
        ("jaine_dharmo", "जैने धर्मो अस्तु मंगलं जैने धर्मो अस्तु मंगलं जैने धर्मो अस्तु मंगलं"),
    ],
    # Buddhist Pali — different "Namo" context
    "buddhist_namo": [
        ("namo_tassa", "नमो तस्स भगवतो अरहतो सम्मा सम्बुद्धस्स नमो तस्स भगवतो अरहतो सम्मा सम्बुद्धस्स"),
        ("buddham_saranam", "बुद्धं शरणं गच्छामि धम्मं शरणं गच्छामि संघं शरणं गच्छामि"),
    ],
    # Hindu mantras with Namo
    "hindu_namo": [
        ("om_namo_bhagavate", "ॐ नमो भगवते वासुदेवाय ॐ नमो भगवते वासुदेवाय ॐ नमो भगवते वासुदेवाय"),
        ("om_namah_shivaya", "ॐ नमः शिवाय ॐ नमः शिवाय ॐ नमः शिवाय ॐ नमः शिवाय ॐ नमः शिवाय"),
        ("om_namo_narayanaya", "ॐ नमो नारायणाय ॐ नमो नारायणाय ॐ नमो नारायणाय ॐ नमो नारायणाय"),
    ],
    # Truncated Navkar — opening lines, NO closing. Must NOT count!
    "truncated": [
        # First 5 lines only — no "Esopanch... Padhamam Havai Mangalam"
        ("opening_only", "नमो अरिहंताणं नमो सिद्धाणं नमो आयरियाणं नमो उवज्झायाणं नमो लोए सव्व साहूणं"),
        # Just first 2 lines repeated
        ("first_two_loop", "नमो अरिहंताणं नमो सिद्धाणं नमो अरिहंताणं नमो सिद्धाणं नमो अरिहंताणं नमो सिद्धाणं"),
        # First line alone, repeated
        ("namo_arihantanam_loop", "नमो अरिहंताणं नमो अरिहंताणं नमो अरिहंताणं नमो अरिहंताणं नमो अरिहंताणं"),
    ],
    # Other Jain devotional prayers (same vocabulary domain, different prayer)
    "other_jain": [
        ("chattari_mangalam", "चत्तारि मंगलं अरिहंता मंगलं सिद्धा मंगलं साहू मंगलं केवलिपन्नत्तो धम्मो मंगलं"),
        ("logassa_short", "लोगस्स उज्जोयगरे धम्मतित्थयरे जिणे अरिहंते कित्तइस्सं चउवीसं पि केवली"),
        ("uvasaggaharam", "उवसग्गहरं पासं पासं वंदामि कम्मघण मुक्कं विसहर विस निन्नासं मंगल कल्लाण आवासं"),
    ],
    # Spoken-about-Navkar — Hindi commentary mentioning the mantra
    "spoken_about": [
        ("intro_chant", "आज मैं नवकार मंत्र का जाप करूँगा नवकार मंत्र जैन धर्म का सबसे महत्वपूर्ण मंत्र है"),
        ("explain_navkar", "नवकार मंत्र में पाँच पंक्तियाँ होती हैं और अंत में चूलिका होती है जिसमें मंगलम शब्द आता है"),
        ("about_mangalam", "मंगलम शब्द का अर्थ होता है शुभ और कल्याणकारी नवकार मंत्र का अंत मंगलम से होता है"),
    ],
}


def get_api_key() -> str:
    key = os.environ.get("SARVAM_API_KEY")
    if not key:
        print("ERROR: SARVAM_API_KEY environment variable not set")
        sys.exit(1)
    return key


def synthesize(text: str, voice: str, pace: float, api_key: str) -> bytes | None:
    payload = {
        "inputs": [text],
        "target_language_code": "hi-IN",
        "speaker": voice,
        "pitch": 0.0,
        "pace": pace,
        "loudness": 1.0,
        "speech_sample_rate": 22050,
        "enable_preprocessing": False,
        "model": "bulbul:v2",
    }
    headers = {"api-subscription-key": api_key, "Content-Type": "application/json"}
    try:
        r = requests.post(SARVAM_TTS_URL, json=payload, headers=headers, timeout=60)
    except Exception as e:
        return None
    if r.status_code != 200:
        return None
    data = r.json()
    audios = data.get("audios") or []
    if not audios:
        return None
    return base64.b64decode(audios[0])


def to_16khz_mono(wav_bytes: bytes) -> np.ndarray | None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ti:
        ti.write(wav_bytes); ip = ti.name
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as to_:
        op = to_.name
    try:
        r = subprocess.run(
            [ffmpeg, "-y", "-i", ip, "-ar", str(SAMPLE_RATE), "-ac", "1",
             "-c:a", "pcm_s16le", op],
            capture_output=True, timeout=30,
        )
        if r.returncode != 0: return None
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
                        help="Generate one sample per phrase (single voice) for sanity check")
    parser.add_argument("--voices-per-phrase", type=int, default=3,
                        help="How many voices to use per phrase (max 7)")
    parser.add_argument("--paces-per-voice", type=int, default=2,
                        help="How many paces per voice (max 4)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    api_key = get_api_key()
    random.seed(args.seed)

    if args.test:
        voices_per_phrase = 1
        paces_per_voice = 1
    else:
        voices_per_phrase = min(args.voices_per_phrase, len(VOICES))
        paces_per_voice = min(args.paces_per_voice, len(PACE_VARIANTS))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    n_phrases = sum(len(p) for p in PHRASES.values())
    total_target = n_phrases * voices_per_phrase * paces_per_voice
    print(f"Phrases: {n_phrases} across {len(PHRASES)} categories")
    print(f"Voices per phrase: {voices_per_phrase}  |  Paces per voice: {paces_per_voice}")
    print(f"Target samples: {total_target}")
    print()

    generated = 0; failed = 0; skipped = 0; bad_dur = 0

    pbar = tqdm(total=total_target, desc="TTS")
    for category, phrase_list in PHRASES.items():
        cat_dir = OUT_ROOT / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        for phrase_id, text in phrase_list:
            voices = random.sample(VOICES, voices_per_phrase)
            for voice in voices:
                paces = random.sample(PACE_VARIANTS, paces_per_voice)
                for pace in paces:
                    pace_int = int(pace * 100)
                    out_path = cat_dir / f"sarvam_{phrase_id}_{voice}_p{pace_int}.wav"
                    if out_path.exists():
                        skipped += 1; pbar.update(1); continue

                    wav_bytes = synthesize(text, voice, pace, api_key)
                    if wav_bytes is None:
                        failed += 1; pbar.update(1); continue

                    audio = to_16khz_mono(wav_bytes)
                    if audio is None or len(audio) < SAMPLE_RATE:
                        failed += 1; pbar.update(1); continue

                    dur = len(audio) / SAMPLE_RATE
                    if dur < MIN_DURATION_S or dur > MAX_DURATION_S:
                        bad_dur += 1
                        # Still save it -- duration anomaly is OK for negatives
                    audio = peak_normalize(audio)
                    sf.write(str(out_path), audio, SAMPLE_RATE, subtype="PCM_16")
                    generated += 1
                    pbar.update(1)
                    pbar.set_postfix(gen=generated, fail=failed, bad_dur=bad_dur)
                    time.sleep(0.1)
    pbar.close()

    print()
    print(f"Generated: {generated} | Skipped: {skipped} | Failed: {failed} | Bad-duration: {bad_dur}")
    for category in PHRASES:
        n = len(list((OUT_ROOT / category).glob("*.wav")))
        print(f"  {category:<20} {n:>4} files")


if __name__ == "__main__":
    main()
