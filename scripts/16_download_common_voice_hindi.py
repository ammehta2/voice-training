"""
Download Hindi speech samples from Google FLEURS (via HuggingFace), slice them
into 2.5s windows, and add them as negatives for the closing-phrase model.

Why: the model has never seen conversational Hindi/Gujarati. The first time
a user has a phone call near the running counter, it will probably fire
because the model's understanding of "negative" is "ESC-50 environmental
sounds" -- not "people talking."

Originally planned to use Mozilla Common Voice but it requires custom dataset
scripts that were removed in datasets >= 4.0. FLEURS is a Google/HuggingFace
multilingual speech dataset (102 languages, 12 hours each) that's publicly
available and well-supported. It serves the same purpose: real conversational
speech in Hindi/other Indian languages.

We stream the dataset (avoid full download), use cast_column(decode=False) to
get raw WAV bytes (avoid the torchcodec Windows DLL issue), then process
via ffmpeg directly.

Usage:
    python scripts/16_download_common_voice_hindi.py --n-clips 500
    python scripts/16_download_common_voice_hindi.py --n-clips 200 --lang gu_in
"""

import argparse
import io
import random
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from tqdm import tqdm

SAMPLE_RATE = 16000
WINDOW_SEC = 2.5
N_SAMPLES = int(WINDOW_SEC * SAMPLE_RATE)
OUT_DIR = Path("data/closing/negative_raw")


def peak_normalize(audio: np.ndarray, target: float = 0.95) -> np.ndarray:
    p = float(np.max(np.abs(audio)))
    return (audio / p * target).astype(np.float32) if p > 1e-6 else audio.astype(np.float32)


def decode_audio_bytes(wav_bytes: bytes) -> np.ndarray | None:
    """Decode arbitrary audio bytes to 16kHz mono float32 via ffmpeg."""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    in_path = None
    out_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp_in:
            tmp_in.write(wav_bytes)
            in_path = tmp_in.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
            out_path = tmp_out.name
        r = subprocess.run(
            [ffmpeg, "-y", "-i", in_path, "-ar", str(SAMPLE_RATE), "-ac", "1",
             "-c:a", "pcm_s16le", out_path],
            capture_output=True, timeout=30,
        )
        if r.returncode != 0:
            return None
        audio, _ = sf.read(out_path)
        return audio.astype(np.float32)
    except Exception:
        return None
    finally:
        if in_path:
            try: Path(in_path).unlink()
            except: pass
        if out_path:
            try: Path(out_path).unlink()
            except: pass


def slice_to_windows(audio: np.ndarray) -> list[np.ndarray]:
    """Non-overlapping 2.5s windows. Pad if shorter than one window."""
    windows = []
    for start in range(0, len(audio) - N_SAMPLES + 1, N_SAMPLES):
        windows.append(audio[start : start + N_SAMPLES])
    if not windows and len(audio) > 0:
        padded = np.pad(audio, (0, max(0, N_SAMPLES - len(audio))), mode="constant")
        windows.append(padded[:N_SAMPLES])
    return windows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-clips", type=int, default=500,
                        help="Number of FLEURS clips to pull (each yields 1-3 windows of 2.5s)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lang", default="hi_in",
                        help="FLEURS language config (hi_in=Hindi, gu_in=Gujarati, "
                             "mr_in=Marathi, sd_in=Sindhi, ta_in=Tamil, te_in=Telugu, etc.)")
    parser.add_argument("--dataset", default="google/fleurs",
                        help="HuggingFace dataset path")
    parser.add_argument("--prefix", default=None,
                        help="Output filename prefix (default: speech_<lang>)")
    args = parser.parse_args()

    random.seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix or f"speech_{args.lang.split('_')[0]}"
    print(f"Dataset: {args.dataset} ({args.lang})")
    print(f"Target: {args.n_clips} clips -> {WINDOW_SEC}s windows -> {OUT_DIR}/")
    print(f"Output prefix: {prefix}_")
    print()

    # Stream + raw bytes (avoid torchcodec Windows DLL issue)
    ds = load_dataset(args.dataset, args.lang, split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    total_clips = 0
    total_windows = 0
    failed = 0

    pbar = tqdm(total=args.n_clips, desc=f"Clips ({args.lang})")
    for item in ds:
        if total_clips >= args.n_clips:
            break

        audio = item.get("audio", {})
        if not isinstance(audio, dict):
            failed += 1
            continue
        wav_bytes = audio.get("bytes")
        if not wav_bytes:
            failed += 1
            continue

        decoded = decode_audio_bytes(wav_bytes)
        if decoded is None or len(decoded) == 0:
            failed += 1
            continue

        # Skip super-quiet clips
        if np.max(np.abs(decoded)) < 0.01:
            failed += 1
            continue

        windows = slice_to_windows(decoded)
        for w_idx, window in enumerate(windows):
            window = peak_normalize(window)
            out_name = f"{prefix}_{total_clips:04d}_{w_idx:02d}.wav"
            out_path = OUT_DIR / out_name
            if out_path.exists():
                continue
            sf.write(str(out_path), window, SAMPLE_RATE, subtype="PCM_16")
            total_windows += 1

        total_clips += 1
        pbar.update(1)
        pbar.set_postfix(windows=total_windows, failed=failed)

    pbar.close()
    print()
    print(f"Done. Pulled {total_clips} clips -> {total_windows} negative windows ({failed} failed/skipped)")
    print(f"Total negatives in {OUT_DIR}: {len(list(OUT_DIR.glob('*.wav')))}")


if __name__ == "__main__":
    main()
