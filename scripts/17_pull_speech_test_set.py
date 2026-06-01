"""
Pull a HELD-OUT speech test set (separate from training negatives) to
evaluate whether trained models correctly reject Hindi/Gujarati conversation.

Uses FLEURS test/validation split (different from train split used by script 16)
so we know the speech clips here were never seen by any model.

Output: data/speech_negative_test/*.wav -- 2.5s windows of conversational
        Hindi and Gujarati. Models should produce ZERO detections on these.

Usage:
    python scripts/17_pull_speech_test_set.py
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from tqdm import tqdm

SAMPLE_RATE = 16000
N_SAMPLES = 40000  # 2.5s
OUT_DIR = Path("data/speech_negative_test")


def decode_bytes(wav_bytes: bytes):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as ti:
        ti.write(wav_bytes); ip = ti.name
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as to:
        op = to.name
    try:
        r = subprocess.run([ffmpeg, "-y", "-i", ip, "-ar", str(SAMPLE_RATE),
                           "-ac", "1", "-c:a", "pcm_s16le", op],
                          capture_output=True, timeout=30)
        if r.returncode != 0:
            return None
        audio, _ = sf.read(op)
        return audio.astype(np.float32)
    finally:
        try: Path(ip).unlink()
        except: pass
        try: Path(op).unlink()
        except: pass


def slice_windows(audio, n_samples=N_SAMPLES):
    out = []
    for s in range(0, len(audio) - n_samples + 1, n_samples):
        out.append(audio[s:s+n_samples])
    if not out and len(audio) > 0:
        padded = np.pad(audio, (0, max(0, n_samples - len(audio))), mode="constant")
        out.append(padded[:n_samples])
    return out


def peak_norm(audio, target=0.95):
    p = float(np.max(np.abs(audio)))
    return (audio / p * target).astype(np.float32) if p > 1e-6 else audio.astype(np.float32)


def pull_lang(lang: str, n_clips: int, split: str = "test"):
    """Use FLEURS test split so these clips were never in training."""
    print(f"  {lang}: pulling {n_clips} clips from {split} split")
    ds = load_dataset("google/fleurs", lang, split=split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    n = 0
    windows_made = 0
    pbar = tqdm(total=n_clips, desc=f"  {lang}", leave=False)
    for item in ds:
        if n >= n_clips:
            break
        wav = item.get("audio", {}).get("bytes")
        if not wav:
            continue
        audio = decode_bytes(wav)
        if audio is None or len(audio) < 1000 or np.max(np.abs(audio)) < 0.01:
            continue
        for w_idx, w in enumerate(slice_windows(audio)):
            out_path = OUT_DIR / f"speech_{lang}_{n:04d}_{w_idx:02d}.wav"
            if out_path.exists():
                continue
            sf.write(str(out_path), peak_norm(w), SAMPLE_RATE, subtype="PCM_16")
            windows_made += 1
        n += 1
        pbar.update(1)
    pbar.close()
    print(f"  {lang}: {n} clips -> {windows_made} windows")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-hindi", type=int, default=30)
    parser.add_argument("--n-gujarati", type=int, default=20)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Pulling HELD-OUT Hindi+Gujarati speech to {OUT_DIR}/")
    pull_lang("hi_in", args.n_hindi)
    pull_lang("gu_in", args.n_gujarati)
    total = len(list(OUT_DIR.glob("*.wav")))
    print(f"\nTotal test windows: {total}")
    print("Each should produce count=0 from the trained models.")


if __name__ == "__main__":
    main()
