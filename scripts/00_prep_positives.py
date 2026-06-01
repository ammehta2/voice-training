"""
Pre-processing for the limited real-data scenario.

For each file in data/positive_raw/:
  1. Peak-normalize to amplitude 0.95 (fixes inter-recording loudness mismatch).
  2. If duration > 25s, split into overlapping 16s windows (12s hop -> 4s overlap).
  3. Otherwise leave as one file.

Originals are preserved in data/positive_raw_originals/ and replaced in place.

Usage:
    python scripts/00_prep_positives.py
"""

import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

POSITIVE_DIR = Path("data/positive_raw")
BACKUP_DIR = Path("data/positive_raw_originals")
SAMPLE_RATE = 16000
WINDOW_SEC = 16
HOP_SEC = 12  # 4s overlap between windows
SPLIT_THRESHOLD_SEC = 25  # only split if duration > this


def peak_normalize(audio: np.ndarray, target: float = 0.95) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak < 1e-6:
        return audio
    return (audio / peak) * target


def main() -> None:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(POSITIVE_DIR.glob("*.wav"))
    if not files:
        print(f"No files in {POSITIVE_DIR}")
        return

    # Back up originals once (only if backup is empty)
    if not list(BACKUP_DIR.glob("*.wav")):
        for f in files:
            shutil.copy2(str(f), str(BACKUP_DIR / f.name))
        print(f"Backed up {len(files)} originals to {BACKUP_DIR}/")

    total_out = 0
    print(f"\n{'file':<25s} {'dur':>5s}  {'orig_pk':>8s}  -> outputs")
    print("-" * 70)
    for f in files:
        audio, sr = sf.read(str(f))
        assert sr == SAMPLE_RATE, f"{f.name} sr={sr} != {SAMPLE_RATE}"
        duration = len(audio) / sr
        orig_peak = float(np.max(np.abs(audio)))

        # Peak-normalize
        audio = peak_normalize(audio)

        stem = f.stem
        f.unlink()  # remove original from positive_raw (kept in backup)

        if duration <= SPLIT_THRESHOLD_SEC:
            out = POSITIVE_DIR / f"{stem}.wav"
            sf.write(str(out), audio, sr, subtype="PCM_16")
            total_out += 1
            print(f"{f.name:<25s} {duration:5.1f}s {orig_peak:>8.3f}  -> 1 window (kept as-is)")
        else:
            # Split into overlapping windows
            samples_per_window = sr * WINDOW_SEC
            hop_samples = sr * HOP_SEC
            n_windows = (len(audio) - samples_per_window) // hop_samples + 1
            n_windows = max(1, n_windows)

            windows_made = 0
            for w in range(n_windows):
                start = w * hop_samples
                end = start + samples_per_window
                if end > len(audio):
                    break
                window = audio[start:end]
                out = POSITIVE_DIR / f"{stem}_w{w:02d}.wav"
                sf.write(str(out), window, sr, subtype="PCM_16")
                windows_made += 1
                total_out += 1
            print(f"{f.name:<25s} {duration:5.1f}s {orig_peak:>8.3f}  -> {windows_made} windows")

    print("-" * 70)
    print(f"Total output files in {POSITIVE_DIR}: {total_out}")
    print(f"Originals preserved in {BACKUP_DIR}")


if __name__ == "__main__":
    main()
