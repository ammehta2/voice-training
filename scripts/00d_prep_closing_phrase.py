"""
Prep for the closing-phrase ("Padhamam Havai Mangalam") detector.

For each real Navkar recording in data/positive_raw_originals/:
  - Detect end-of-audio (trim trailing silence)
  - POSITIVE: last CLOSING_SEC seconds before that endpoint
  - PRE-CLOSING NEGATIVES: rest of recording (everything before the closing),
    sliced into CLOSING_SEC windows with NEGATIVE_HOP_SEC hop. These teach
    the model that "Namo Arihantaanam", "Namo Siddhaanam", etc. should NOT
    fire the counter — only "Padhamam Havai Mangalam" should.

Output:
    data/closing/positive_raw/     (one closing per recording)
    data/closing/negative_raw/     (pre-closing mid-mantra + ambient)

Why closing-phrase has a better shot than opening-phrase:
  - "Padhamam Havai Mangalam" shares no phonemes with the 5 "Namo X" lines
  - "Mangalam" is acoustically very distinctive
  - One occurrence per chant by structure

Usage:
    python scripts/00d_prep_closing_phrase.py
"""

from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
CLOSING_SEC = 2.5
NEGATIVE_HOP_SEC = 1.0
SILENCE_THRESHOLD = 0.02
SILENCE_WIN_MS = 50

POSITIVE_SOURCE = Path("data/positive_raw_originals")
AMBIENT_SOURCE_DIR = Path("data/negative_raw")

OUT_POS = Path("data/closing/positive_raw")
OUT_NEG = Path("data/closing/negative_raw")


def find_endpoint(audio: np.ndarray, sr: int) -> int:
    """Return sample index of the LAST non-silent moment (trailing silence trimmed)."""
    win = int(sr * SILENCE_WIN_MS / 1000)
    peak = np.max(np.abs(audio))
    if peak < 1e-6:
        return len(audio)
    normalized = audio / peak
    # Sliding RMS from the end
    for i in range(len(normalized) - win, 0, -win):
        rms = np.sqrt(np.mean(normalized[i : i + win] ** 2))
        if rms > SILENCE_THRESHOLD:
            return i + win
    return len(audio)


def find_onset(audio: np.ndarray, sr: int) -> int:
    """Return sample index of the FIRST non-silent moment."""
    win = int(sr * SILENCE_WIN_MS / 1000)
    peak = np.max(np.abs(audio))
    if peak < 1e-6:
        return 0
    normalized = audio / peak
    for i in range(0, len(normalized) - win, win):
        rms = np.sqrt(np.mean(normalized[i : i + win] ** 2))
        if rms > SILENCE_THRESHOLD:
            return i
    return 0


def peak_normalize(audio: np.ndarray, target: float = 0.95) -> np.ndarray:
    peak = float(np.max(np.abs(audio)))
    return (audio / peak * target) if peak > 1e-6 else audio


def extract_closing(source_path: Path, out_pos_dir: Path, out_neg_dir: Path) -> tuple[int, int]:
    """Extract one closing positive + N pre-closing negatives.
    Returns (n_positive, n_pre_negative).
    """
    audio, sr = sf.read(str(source_path))
    assert sr == SAMPLE_RATE, f"{source_path.name} sr={sr}"

    onset = find_onset(audio, sr)
    endpoint = find_endpoint(audio, sr)
    closing_samples = int(CLOSING_SEC * sr)

    # POSITIVE: last CLOSING_SEC ending at endpoint
    pos_start = max(onset, endpoint - closing_samples)
    closing = audio[pos_start:endpoint]
    if len(closing) < closing_samples:
        # Pad at the BEGINNING (silence before closing)
        closing = np.pad(closing, (closing_samples - len(closing), 0), mode="constant")
    elif len(closing) > closing_samples:
        closing = closing[-closing_samples:]
    closing = peak_normalize(closing)

    pos_path = out_pos_dir / f"{source_path.stem}_closing.wav"
    sf.write(str(pos_path), closing, sr, subtype="PCM_16")

    # NEGATIVES: everything between onset and pos_start, sliced into windows
    pre = audio[onset:pos_start]
    if len(pre) < closing_samples:
        return 1, 0

    hop_samples = int(NEGATIVE_HOP_SEC * sr)
    n_pre = 0
    window_idx = 0
    while True:
        start = window_idx * hop_samples
        end = start + closing_samples
        if end > len(pre):
            break
        window = peak_normalize(pre[start:end].copy())
        neg_path = out_neg_dir / f"premantra_{source_path.stem}_{window_idx:03d}.wav"
        sf.write(str(neg_path), window, sr, subtype="PCM_16")
        n_pre += 1
        window_idx += 1
    return 1, n_pre


def slice_ambient_to_windows(audio: np.ndarray, sr: int) -> list[np.ndarray]:
    win = int(CLOSING_SEC * sr)
    chunks = []
    for start in range(0, len(audio) - win + 1, win):
        chunks.append(audio[start : start + win])
    if not chunks and len(audio) > 0:
        padded = np.pad(audio, (0, max(0, win - len(audio))), mode="constant")
        chunks.append(padded[:win])
    return chunks


def main() -> None:
    OUT_POS.mkdir(parents=True, exist_ok=True)
    OUT_NEG.mkdir(parents=True, exist_ok=True)

    sources = sorted(POSITIVE_SOURCE.glob("*.wav"))
    if not sources:
        print(f"No source recordings in {POSITIVE_SOURCE}")
        return

    print(f"=== Stage 1: closing phrase extraction ===")
    print(f"{'file':<28s} {'dur':>5s}  {'onset':>6s}  {'endpoint':>9s}  {'pos':>4s} {'pre_neg':>7s}")
    print("-" * 75)
    total_pos = total_pre_neg = 0
    for src in sources:
        audio, sr = sf.read(str(src))
        onset = find_onset(audio, sr)
        endpoint = find_endpoint(audio, sr)
        n_pos, n_pre = extract_closing(src, OUT_POS, OUT_NEG)
        total_pos += n_pos
        total_pre_neg += n_pre
        print(f"{src.name:<28s} {len(audio)/sr:5.1f}s {onset/sr:>6.2f}s "
              f"{endpoint/sr:>8.2f}s  {n_pos:>4d} {n_pre:>7d}")
    print(f"Total: {total_pos} closing positives, {total_pre_neg} pre-closing negatives")

    # Stage 2: ambient negatives sliced to CLOSING_SEC windows
    ambient_files = sorted(AMBIENT_SOURCE_DIR.glob("*.wav"))
    print(f"\n=== Stage 2: slice ambient ({len(ambient_files)} source files) to {CLOSING_SEC}s ===")
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
    print(f"\nFinal dataset for closing-phrase model:")
    print(f"  data/closing/positive_raw/: {pos_count} files ({CLOSING_SEC}s each)")
    print(f"  data/closing/negative_raw/: {neg_count} files ({CLOSING_SEC}s each)")
    print(f"    - pre-closing (phonetically similar): {total_pre_neg}")
    print(f"    - ambient (ESC-50 + Birch Tree):     {neg_count - total_pre_neg}")


if __name__ == "__main__":
    main()
