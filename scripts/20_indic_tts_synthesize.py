"""
Generate synthetic Navkar recordings using Meta MMS-TTS (Hindi/Gujarati/Marathi).

Why: complements Sarvam (script 19). MMS-TTS is fully public (no auth),
fast on GPU, and covers multiple Indian languages -- each language uses a
slightly different voice/timbre, adding diversity orthogonal to Sarvam's
voice roster.

Models (all public on HuggingFace):
  - facebook/mms-tts-hin (Hindi)
  - facebook/mms-tts-guj (Gujarati)
  - facebook/mms-tts-mar (Marathi)

Diversity via:
  - 3 languages × 1 voice each = 3 base voices
  - VITS noise_scale parameter (0.5-0.8) for variation within each voice
  - VITS noise_scale_w (0.7-1.0) for duration variance
  - speaking_rate variation built into the model

NOT singing -- MMS-TTS is naturally spoken-cadence. Good for our use case.

Output: data/positive_raw_originals/mmstts_navkar_<lang>_<i>.wav

Usage:
    python scripts/20_indic_tts_synthesize.py --test           # 3 samples (one per language)
    python scripts/20_indic_tts_synthesize.py --n-per-lang 40  # full generation (120 samples)
"""

import argparse
import random
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import soundfile as sf
import torch
from transformers import AutoTokenizer, VitsModel
from tqdm import tqdm

OUT_DIR = Path("data/positive_raw_originals")
TARGET_SR = 16000
MIN_DURATION_S = 6.0
MAX_DURATION_S = 18.0

# Navkar in Devanagari script (continuous, no punctuation -- matches script 19)
NAVKAR_TEXT_DEVANAGARI = (
    "नमो अरिहंताणं नमो सिद्धाणं नमो आयरियाणं "
    "नमो उवज्झायाणं नमो लोए सव्व साहूणं "
    "एसो पंच नमोक्कारो सव्व पावप्पणासणो "
    "मंगलाणं च सव्वेसिं पढमं हवइ मंगलं"
)

# Gujarati MMS-TTS expects Gujarati script. The Devanagari->Gujarati mapping
# is direct (same Indic phonemes, different glyphs). Quick conversion:
DEVANAGARI_TO_GUJARATI = str.maketrans({
    "अ": "અ", "आ": "આ", "इ": "ઇ", "ई": "ઈ", "उ": "ઉ", "ऊ": "ઊ",
    "ए": "એ", "ऐ": "ઐ", "ओ": "ઓ", "औ": "ઔ",
    "क": "ક", "ख": "ખ", "ग": "ગ", "घ": "ઘ", "ङ": "ઙ",
    "च": "ચ", "छ": "છ", "ज": "જ", "झ": "ઝ", "ञ": "ઞ",
    "ट": "ટ", "ठ": "ઠ", "ड": "ડ", "ढ": "ઢ", "ण": "ણ",
    "त": "ત", "थ": "થ", "द": "દ", "ध": "ધ", "न": "ન",
    "प": "પ", "फ": "ફ", "ब": "બ", "भ": "ભ", "म": "મ",
    "य": "ય", "र": "ર", "ल": "લ", "व": "વ",
    "श": "શ", "ष": "ષ", "स": "સ", "ह": "હ",
    "ा": "ા", "ि": "િ", "ी": "ી", "ु": "ુ", "ू": "ૂ",
    "े": "ે", "ै": "ૈ", "ो": "ો", "ौ": "ૌ",
    "्": "્", "ं": "ં", "ः": "ઃ", "ँ": "ઁ",
})

LANGUAGES = [
    # (lang_code, model_id, text, label)
    ("hin", "facebook/mms-tts-hin", NAVKAR_TEXT_DEVANAGARI, "Hindi"),
    ("guj", "facebook/mms-tts-guj", NAVKAR_TEXT_DEVANAGARI.translate(DEVANAGARI_TO_GUJARATI), "Gujarati"),
    ("mar", "facebook/mms-tts-mar", NAVKAR_TEXT_DEVANAGARI, "Marathi"),
]


def peak_normalize(audio: np.ndarray, target: float = 0.95) -> np.ndarray:
    p = float(np.max(np.abs(audio)))
    return (audio / p * target).astype(np.float32) if p > 1e-6 else audio.astype(np.float32)


def resample_to_16khz(audio: np.ndarray, src_sr: int) -> np.ndarray:
    if src_sr == TARGET_SR:
        return audio
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ti:
        sf.write(ti.name, audio, src_sr, subtype="PCM_16")
        ip = ti.name
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as to:
        op = to.name
    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", ip, "-ar", str(TARGET_SR), "-ac", "1",
             "-c:a", "pcm_s16le", op],
            capture_output=True, timeout=30,
        )
        audio_16k, _ = sf.read(op)
        return audio_16k.astype(np.float32)
    finally:
        try: Path(ip).unlink()
        except: pass
        try: Path(op).unlink()
        except: pass


def synthesize(model, tokenizer, text: str, device: str,
               noise_scale: float = 0.667, noise_scale_w: float = 0.8, length_scale: float = 1.0) -> np.ndarray:
    """Run MMS-TTS inference with VITS sampling parameters for variation."""
    model.noise_scale = noise_scale       # acoustic variation (0.0-1.0)
    model.noise_scale_w = noise_scale_w   # phoneme duration variation (0.0-1.0)
    model.speaking_rate = 1.0 / length_scale  # >1.0 = faster

    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**inputs).waveform
    return output.cpu().numpy().squeeze().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="3 samples (one per language) for QA")
    parser.add_argument("--n-per-lang", type=int, default=40,
                        help="Samples per language (3 langs total)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    n_per = 1 if args.test else args.n_per_lang
    total_target = len(LANGUAGES) * n_per
    print(f"Generating {n_per} samples x {len(LANGUAGES)} languages = {total_target}")
    print()

    generated = 0
    failed = 0
    bad_dur = 0
    skipped = 0

    for lang_code, model_id, text, label in LANGUAGES:
        print(f"\nLoading {model_id} ({label})...")
        try:
            model = VitsModel.from_pretrained(model_id).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            src_sr = model.config.sampling_rate
            print(f"  Loaded; sampling_rate={src_sr}")
        except Exception as e:
            print(f"  Failed to load: {e}")
            continue

        pbar = tqdm(total=n_per, desc=f"  {label}")
        for i in range(n_per):
            out_path = OUT_DIR / f"mmstts_navkar_{lang_code}_{i:02d}.wav"
            if out_path.exists():
                skipped += 1
                pbar.update(1)
                continue

            # Vary VITS parameters per-sample for diversity
            noise_scale = random.uniform(0.55, 0.85)
            noise_scale_w = random.uniform(0.70, 1.00)
            length_scale = random.uniform(0.85, 1.15)  # ~equivalent to pace variation

            try:
                audio = synthesize(model, tokenizer, text, device,
                                   noise_scale, noise_scale_w, length_scale)
                audio = resample_to_16khz(audio, src_sr)

                dur = len(audio) / TARGET_SR
                if dur < MIN_DURATION_S or dur > MAX_DURATION_S:
                    bad_dur += 1
                    out_path = OUT_DIR / f"mmstts_navkar_{lang_code}_{i:02d}_dur{int(dur)}.wav"

                audio = peak_normalize(audio)
                sf.write(str(out_path), audio, TARGET_SR, subtype="PCM_16")
                generated += 1
            except Exception as e:
                print(f"\n  Failed {lang_code}#{i}: {type(e).__name__}: {str(e)[:80]}")
                failed += 1

            pbar.update(1)
            pbar.set_postfix(gen=generated, fail=failed, dur_oob=bad_dur)
        pbar.close()

        # Free GPU memory between languages
        del model, tokenizer
        torch.cuda.empty_cache()

    print()
    print(f"Generated: {generated} ({bad_dur} outside ideal duration)")
    print(f"Failed: {failed}, Skipped: {skipped}")
    mms_files = sorted(OUT_DIR.glob("mmstts_navkar_*.wav"))
    print(f"Total MMS-TTS files: {len(mms_files)}")

    if args.test and generated > 0:
        print(f"\n*** TEST MODE — listen to verify quality before full run ***")
        for f in mms_files[-min(3, generated):]:
            audio, _ = sf.read(str(f))
            print(f"  {f.name}: {len(audio)/TARGET_SR:.1f}s")


if __name__ == "__main__":
    main()
