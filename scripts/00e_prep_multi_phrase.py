"""
Prep for the multi-phrase (v7) model.

The Navkar mantra ends:
    Mangalaanam cha Savvesim, Padhamam Havai Mangalam

The CLOSING is "Padhamam Havai Mangalam" (current model's target).
The PRE-CLOSING is "Mangalaanam cha Savvesim", which immediately precedes it.

By detecting BOTH and requiring temporal sequence (pre → closing within 2-3s)
in mobile inference, we gain confidence on quiet/ambiguous chants where the
closing phrase alone scores only ~0.4.

Each source recording gets:
  - 1 closing positive  (last 2.5s before trailing silence)
  - 1 pre-closing positive  (2.5s window immediately before the closing window)

Output:
    data/multi/closing/positive_raw/      one window per recording, last 2.5s
    data/multi/preclosing/positive_raw/   one window per recording, prev 2.5s
    data/multi/shared/negative_raw/       shared negatives (mid-mantra + ambient + speech)

Mid-mantra negatives skip the LAST 5s (which would overlap with our two
positive windows) — so we don't accidentally include the closing or
pre-closing as a negative.

Usage:
    python scripts/00e_prep_multi_phrase.py
"""

from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
WINDOW_SEC = 2.5
N_SAMPLES = int(WINDOW_SEC * SAMPLE_RATE)
NEGATIVE_HOP_SEC = 1.0
SILENCE_THRESHOLD = 0.02
SILENCE_WIN_MS = 50

POSITIVE_SOURCE = Path("data/positive_raw_originals")
AMBIENT_SOURCE_DIR = Path("data/negative_raw")  # ESC-50 + Birch Tree

OUT_CLOSING = Path("data/multi/closing/positive_raw")
OUT_PRECLOSING = Path("data/multi/preclosing/positive_raw")
OUT_NEGATIVES = Path("data/multi/shared/negative_raw")


def find_endpoint(audio: np.ndarray, sr: int) -> int:
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


def extract_positives(source_path: Path) -> tuple[int, int]:
    """Extract closing + pre-closing positives. Returns (n_closing, n_preclosing)."""
    audio, sr = sf.read(str(source_path))
    assert sr == SAMPLE_RATE

    onset = find_onset(audio, sr)
    endpoint = find_endpoint(audio, sr)

    n_close = 0
    n_pre = 0

    # --- Closing window: last N_SAMPLES ending at endpoint ---
    close_start = max(onset, endpoint - N_SAMPLES)
    closing = audio[close_start:endpoint]
    if len(closing) < N_SAMPLES:
        closing = np.pad(closing, (N_SAMPLES - len(closing), 0), mode="constant")
    elif len(closing) > N_SAMPLES:
        closing = closing[-N_SAMPLES:]
    closing = peak_normalize(closing)
    sf.write(str(OUT_CLOSING / f"{source_path.stem}_closing.wav"), closing, sr, subtype="PCM_16")
    n_close = 1

    # --- Pre-closing window: 2.5s ending immediately before closing ---
    pre_end = close_start
    pre_start = pre_end - N_SAMPLES
    if pre_start >= onset:
        pre = audio[pre_start:pre_end]
        if len(pre) == N_SAMPLES:
            pre = peak_normalize(pre)
            sf.write(str(OUT_PRECLOSING / f"{source_path.stem}_preclosing.wav"),
                     pre, sr, subtype="PCM_16")
            n_pre = 1
    # else: recording too short for both windows (rare; would be <5s of speech)

    return n_close, n_pre


def extract_mid_mantra_negatives(source_path: Path) -> int:
    """Slice everything BEFORE the pre-closing window into 2.5s windows as
    mid-mantra negatives. These teach the model: 'this is Navkar voice but
    NOT the closing or pre-closing.'"""
    audio, sr = sf.read(str(source_path))
    onset = find_onset(audio, sr)
    endpoint = find_endpoint(audio, sr)
    # Pre-closing ends at endpoint - N_SAMPLES. We want negatives before that.
    pre_end = endpoint - N_SAMPLES
    if pre_end - onset < N_SAMPLES:
        return 0

    mid_audio = audio[onset:pre_end]
    hop_samples = int(NEGATIVE_HOP_SEC * sr)
    n = 0
    w = 0
    while True:
        start = w * hop_samples
        end = start + N_SAMPLES
        if end > len(mid_audio):
            break
        window = peak_normalize(mid_audio[start:end].copy())
        out = OUT_NEGATIVES / f"midmantra_{source_path.stem}_{w:03d}.wav"
        sf.write(str(out), window, sr, subtype="PCM_16")
        n += 1
        w += 1
    return n


def slice_ambient_to_windows(audio: np.ndarray, sr: int) -> list[np.ndarray]:
    win = N_SAMPLES
    chunks = []
    for start in range(0, len(audio) - win + 1, win):
        chunks.append(audio[start : start + win])
    if not chunks and len(audio) > 0:
        padded = np.pad(audio, (0, max(0, win - len(audio))), mode="constant")
        chunks.append(padded[:win])
    return chunks


def main() -> None:
    OUT_CLOSING.mkdir(parents=True, exist_ok=True)
    OUT_PRECLOSING.mkdir(parents=True, exist_ok=True)
    OUT_NEGATIVES.mkdir(parents=True, exist_ok=True)

    sources = sorted([
        f for f in POSITIVE_SOURCE.glob("*.wav")
        if "_partial" not in f.stem
    ])
    print(f"Sources: {len(sources)} positive source recordings")
    print()

    print("=== Stage 1: extract closing + pre-closing positives ===")
    total_close = 0
    total_pre = 0
    too_short = 0
    for src in sources:
        n_close, n_pre = extract_positives(src)
        total_close += n_close
        total_pre += n_pre
        if n_pre == 0:
            too_short += 1
    print(f"  Closing positives:     {total_close}")
    print(f"  Pre-closing positives: {total_pre}  ({too_short} too-short recordings)")

    print("\n=== Stage 2: extract mid-mantra negatives ===")
    total_mid = 0
    for src in sources:
        total_mid += extract_mid_mantra_negatives(src)
    print(f"  Mid-mantra negatives: {total_mid}")

    print("\n=== Stage 3: slice ambient files into 2.5s windows ===")
    ambient_files = sorted(AMBIENT_SOURCE_DIR.glob("*.wav"))
    total_ambient = 0
    for src in ambient_files:
        audio, sr = sf.read(str(src))
        if sr != SAMPLE_RATE:
            continue
        chunks = slice_ambient_to_windows(audio, sr)
        for j, chunk in enumerate(chunks):
            out = OUT_NEGATIVES / f"ambient_{src.stem}_{j:02d}.wav"
            if out.exists():
                total_ambient += 1
                continue
            sf.write(str(out), chunk, sr, subtype="PCM_16")
            total_ambient += 1
    print(f"  Ambient negatives: {total_ambient}")

    print(f"\nFinal counts:")
    print(f"  Closing positives:     {len(list(OUT_CLOSING.glob('*.wav')))}")
    print(f"  Pre-closing positives: {len(list(OUT_PRECLOSING.glob('*.wav')))}")
    print(f"  Negatives (shared):    {len(list(OUT_NEGATIVES.glob('*.wav')))}")


if __name__ == "__main__":
    main()
