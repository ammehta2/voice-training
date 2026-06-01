"""
Extract partial-chant versions of each recording for hard-negative testing.

For each original recording in data/positive_raw_originals/, create three
"partial" versions that contain Navkar audio but DO NOT include the closing
phrase ("Padhamam Havai Mangalam"):

  - 40% of the duration (just the opening lines, far from the end)
  - 70% of the duration (most of the mantra, cuts off mid line-7 "Mangalaanam cha...")
  - 90% of the duration (cuts off mid-closing "Padhamam Havai...")

These go in data/partial_chants_test/ (separate from training data).

Purpose: a correctly-trained closing-phrase detector should produce ZERO
detections on any of these partial files. If it fires on a 90% partial, the
model has learned a shortcut (probably "Mangalaanam in line 7") rather than
specifically the closing phrase.

This is hard-negative validation, the gold standard for confirming a KWS
model is detecting the right thing.

Usage:
    python scripts/14_extract_partial_chants.py
"""

from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
SOURCE_DIR = Path("data/positive_raw_originals")
OUT_DIR = Path("data/partial_chants_test")

# Fraction of the original duration to keep
PARTIAL_FRACTIONS = [0.40, 0.70, 0.90]

# Onset/endpoint detection (matches 00d_prep_closing_phrase.py)
SILENCE_THRESHOLD = 0.02
SILENCE_WIN_MS = 50


def find_endpoint(audio: np.ndarray, sr: int) -> int:
    """Last non-silent sample index."""
    win = int(sr * SILENCE_WIN_MS / 1000)
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-6:
        return len(audio)
    normalized = audio / peak
    for i in range(len(normalized) - win, 0, -win):
        rms = np.sqrt(np.mean(normalized[i : i + win] ** 2))
        if rms > SILENCE_THRESHOLD:
            return i + win
    return len(audio)


def find_onset(audio: np.ndarray, sr: int) -> int:
    win = int(sr * SILENCE_WIN_MS / 1000)
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-6:
        return 0
    normalized = audio / peak
    for i in range(0, len(normalized) - win, win):
        rms = np.sqrt(np.mean(normalized[i : i + win] ** 2))
        if rms > SILENCE_THRESHOLD:
            return i
    return 0


def peak_normalize(audio: np.ndarray, target: float = 0.95) -> np.ndarray:
    p = float(np.max(np.abs(audio)))
    return (audio / p * target).astype(np.float32) if p > 1e-6 else audio.astype(np.float32)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Only process originals (not _ff or other variants)
    sources = sorted([
        f for f in SOURCE_DIR.glob("real_navkar_*.wav")
        if "_ff" not in f.stem
    ])
    print(f"Source recordings: {len(sources)}")
    print(f"Generating {len(PARTIAL_FRACTIONS)} partial versions per recording")
    print(f"Output: {OUT_DIR}/")
    print()

    print(f"{'source':<22s} {'onset':>6s}  {'endpoint':>9s}  "
          + "  ".join(f"{int(f*100)}%_dur" for f in PARTIAL_FRACTIONS))
    print("-" * 75)

    total_made = 0
    for src in sources:
        audio, sr = sf.read(str(src))
        assert sr == SAMPLE_RATE

        onset = find_onset(audio, sr)
        endpoint = find_endpoint(audio, sr)
        speech = audio[onset:endpoint]
        speech_dur = len(speech) / sr

        partial_durs = []
        for frac in PARTIAL_FRACTIONS:
            partial_len = int(len(speech) * frac)
            partial = peak_normalize(speech[:partial_len])
            partial_dur = len(partial) / sr
            partial_durs.append(partial_dur)

            out_path = OUT_DIR / f"{src.stem}_partial{int(frac*100):02d}.wav"
            sf.write(str(out_path), partial, sr, subtype="PCM_16")
            total_made += 1

        print(f"{src.stem:<22s} {onset/sr:>5.2f}s {endpoint/sr:>8.2f}s   "
              + "    ".join(f"{d:>4.1f}s" for d in partial_durs))

    print(f"\nGenerated {total_made} partial chants in {OUT_DIR}/")
    print(f"Test methodology: run each through the trained model and verify count == 0.")


if __name__ == "__main__":
    main()
