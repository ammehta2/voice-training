"""
Pivot prep — extract opening-phrase ("Namo Arihantaanam") training data.

For each real Navkar recording in data/positive_raw_originals/:
  - Detect onset (first significant audio)
  - POSITIVE: onset -> onset + OPENING_SEC ("Namo Arihantaanam")
  - MID-MANTRA NEGATIVES: rest of recording, sliced into OPENING_SEC windows
    with hop NEGATIVE_HOP_SEC, all peak-normalized

The mid-mantra negatives are critical — they teach the model that "Namo
Siddhaanam", "Loe Savva Saahoonam", etc. should NOT fire the counter.

Also carries over the ambient-style negatives (ESC-50 + Birch Tree) by
trimming them to OPENING_SEC windows.

Output:
    data/opening/positive_raw/     (5 opening-phrase samples)
    data/opening/negative_raw/     (mid-mantra + ambient negatives)

Usage:
    python scripts/00c_prep_opening_phrase.py
"""

from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
OPENING_SEC = 2.5             # length of the opening phrase
NEGATIVE_HOP_SEC = 1.0        # hop when slicing mid-mantra into negatives
SILENCE_THRESHOLD = 0.02      # RMS threshold for onset detection (after normalization)
SILENCE_WIN_MS = 50           # window size for RMS-based onset detection

POSITIVE_SOURCE = Path("data/positive_raw_originals")  # 5 ORIGINAL recordings
AMBIENT_SOURCE_DIR = Path("data/negative_raw")          # ESC-50 + Birch Tree (already 16kHz mono)

OUT_POS = Path("data/opening/positive_raw")
OUT_NEG = Path("data/opening/negative_raw")


def find_onset(audio: np.ndarray, sr: int) -> int:
    """Return sample index of the first non-silent region."""
    win = int(sr * SILENCE_WIN_MS / 1000)
    peak = np.max(np.abs(audio))
    if peak < 1e-6:
        return 0
    normalized = audio / peak
    # Sliding RMS
    for i in range(0, len(normalized) - win, win):
        rms = np.sqrt(np.mean(normalized[i : i + win] ** 2))
        if rms > SILENCE_THRESHOLD:
            return i
    return 0


def peak_normalize(audio: np.ndarray, target: float = 0.95) -> np.ndarray:
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-6:
        return audio
    return (audio / peak) * target


def extract_opening(source_path: Path, out_pos_dir: Path, out_neg_dir: Path) -> tuple[int, int]:
    """Extract one opening positive + N mid-mantra negatives from a recording.
    Returns (n_positive, n_mid_negative).
    """
    audio, sr = sf.read(str(source_path))
    assert sr == SAMPLE_RATE, f"{source_path.name} sr={sr}"

    onset = find_onset(audio, sr)
    opening_samples = int(OPENING_SEC * sr)
    opening = audio[onset : onset + opening_samples]
    # Pad if shorter than 2.5s
    if len(opening) < opening_samples:
        opening = np.pad(opening, (0, opening_samples - len(opening)), mode="constant")
    opening = peak_normalize(opening)

    pos_path = out_pos_dir / f"{source_path.stem}_opening.wav"
    sf.write(str(pos_path), opening, sr, subtype="PCM_16")

    # Mid-mantra negatives: from onset + OPENING_SEC to end, sliced
    mid_start = onset + opening_samples
    rest = audio[mid_start:]
    if len(rest) < opening_samples:
        return 1, 0  # nothing left to slice

    hop_samples = int(NEGATIVE_HOP_SEC * sr)
    n_mid = 0
    window_idx = 0
    while True:
        start = window_idx * hop_samples
        end = start + opening_samples
        if end > len(rest):
            break
        window = peak_normalize(rest[start:end].copy())
        neg_path = out_neg_dir / f"midmantra_{source_path.stem}_{window_idx:03d}.wav"
        sf.write(str(neg_path), window, sr, subtype="PCM_16")
        n_mid += 1
        window_idx += 1

    return 1, n_mid


def slice_ambient_to_windows(audio: np.ndarray, sr: int) -> list[np.ndarray]:
    """Cut a long ambient clip into OPENING_SEC chunks (no hop, just sequential)."""
    win = int(OPENING_SEC * sr)
    chunks = []
    for start in range(0, len(audio) - win + 1, win):
        chunks.append(audio[start : start + win])
    if not chunks and len(audio) > 0:
        # If shorter than OPENING_SEC, pad to OPENING_SEC
        padded = np.pad(audio, (0, max(0, win - len(audio))), mode="constant")
        chunks.append(padded[:win])
    return chunks


def main() -> None:
    OUT_POS.mkdir(parents=True, exist_ok=True)
    OUT_NEG.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: opening phrase positives + mid-mantra negatives ----
    sources = sorted(POSITIVE_SOURCE.glob("*.wav"))
    if not sources:
        print(f"No source recordings in {POSITIVE_SOURCE}")
        return

    print(f"=== Stage 1: opening phrase extraction ===")
    print(f"{'file':<30s} {'dur':>5s}  {'onset':>6s}  {'opening':>8s}  {'mid_neg':>8s}")
    print("-" * 65)
    total_pos = total_mid_neg = 0
    for src in sources:
        audio, sr = sf.read(str(src))
        onset = find_onset(audio, sr)
        n_pos, n_mid = extract_opening(src, OUT_POS, OUT_NEG)
        total_pos += n_pos
        total_mid_neg += n_mid
        print(f"{src.name:<30s} {len(audio)/sr:5.1f}s {onset/sr:>6.2f}s "
              f"{'1 pos':>8s} {n_mid:>5d}")

    print(f"Total: {total_pos} opening positives, {total_mid_neg} mid-mantra negatives")

    # ---- Step 2: ambient negatives sliced to OPENING_SEC ----
    ambient_files = sorted(AMBIENT_SOURCE_DIR.glob("*.wav"))
    print(f"\n=== Stage 2: slice ambient ({len(ambient_files)} source files) to {OPENING_SEC}s ===")

    total_ambient = 0
    for src in ambient_files:
        audio, sr = sf.read(str(src))
        if sr != SAMPLE_RATE:
            continue
        chunks = slice_ambient_to_windows(audio, sr)
        for j, chunk in enumerate(chunks):
            out_path = OUT_NEG / f"ambient_{src.stem}_{j:02d}.wav"
            if out_path.exists():
                total_ambient += 1
                continue
            sf.write(str(out_path), chunk, sr, subtype="PCM_16")
            total_ambient += 1

    pos_count = len(list(OUT_POS.glob("*.wav")))
    neg_count = len(list(OUT_NEG.glob("*.wav")))
    print(f"\nFinal dataset for opening-phrase model:")
    print(f"  data/opening/positive_raw/: {pos_count} files ({OPENING_SEC}s each)")
    print(f"  data/opening/negative_raw/: {neg_count} files ({OPENING_SEC}s each)")
    print(f"    - mid-mantra (phonetically similar): {total_mid_neg}")
    print(f"    - ambient (ESC-50 + Birch Tree):    {neg_count - total_mid_neg}")


if __name__ == "__main__":
    main()
