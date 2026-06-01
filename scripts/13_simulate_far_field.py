"""
Generate far-field versions of close-mic recordings using room impulse
response (RIR) simulation with pyroomacoustics.

Why: training data was recorded close to the phone mic, but at inference time
the phone often sits 1-3m away on a table while the user chants. The audio
characteristics differ substantially (more reverb, lower DRR, more attenuation
of high frequencies). Without far-field training data, the model may detect
"close-mic chanting acoustics" rather than "Navkar closing phrase".

Standard fix in production KWS (Google Home, Alexa, etc.): augment training
data by convolving each close-mic recording with simulated RIRs spanning
realistic room geometries and mic-source distances. This is called
"multi-condition training" (MCT).

For each source recording in data/positive_raw_originals/, this generates
N_FARFIELD_PER_RECORDING far-field versions with different room/distance
combinations, saved to data/positive_raw_originals/ alongside the originals
(naming: real_navkar_07_ff01.wav, _ff02.wav, etc.).

Usage:
    python scripts/13_simulate_far_field.py [--n-per-recording 3] [--noise-db -45]
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pyroomacoustics as pra
import soundfile as sf
from tqdm import tqdm

SAMPLE_RATE = 16000
SOURCE_DIR = Path("data/positive_raw_originals")

# Room configurations: (dimensions in meters, absorption, description)
# Absorption: lower = more reverb. Realistic interior values 0.3-0.7.
ROOM_CONFIGS = [
    # (W,    L,    H,   absorption_avg, name)
    ((4.0, 5.0, 3.0), 0.50, "bedroom_normal"),
    ((4.0, 5.0, 3.0), 0.35, "bedroom_reverb"),     # less furniture
    ((5.0, 7.0, 3.0), 0.45, "living_normal"),
    ((5.0, 7.0, 3.0), 0.30, "living_reverb"),
    ((6.0, 8.0, 3.5), 0.40, "large_room"),
    ((3.5, 4.0, 2.7), 0.55, "small_room"),
    ((8.0, 10.0, 4.0), 0.30, "hall_reverb"),
]

# Source-mic distance categories (mic stays roughly centered, source moves)
# (distance_m, source_height_m, description)
DISTANCE_CONFIGS = [
    (0.5, 1.2, "close_seated"),   # Phone on lap or held; chanter sitting
    (1.0, 1.2, "1m_seated"),       # Phone on table, chanter sitting at table
    (1.5, 1.6, "1.5m_standing"),   # Phone on table, chanter standing nearby
    (2.0, 1.6, "2m_standing"),     # Phone on shelf, chanter across room
    (3.0, 1.6, "3m_standing"),     # Phone in corner, chanter on the other side
]


def generate_rir(room_config, distance_config, mic_height: float = 1.0) -> np.ndarray:
    """Generate one room impulse response.

    Returns a 1D numpy array (the RIR for the mic, given the source position).
    """
    (W, L, H), abs_coef, _ = room_config
    distance, source_height, _ = distance_config

    # Place mic near center, slightly off-center for asymmetry
    mic_pos = np.array([W / 2, L / 2 - 0.5, mic_height])

    # Place source at given distance, random angle around mic
    angle = random.uniform(0, 2 * np.pi)
    source_pos = mic_pos + np.array([
        distance * np.cos(angle),
        distance * np.sin(angle),
        (source_height - mic_height),
    ])

    # Clamp source into the room (with 0.3m margin from walls)
    source_pos[0] = np.clip(source_pos[0], 0.3, W - 0.3)
    source_pos[1] = np.clip(source_pos[1], 0.3, L - 0.3)
    source_pos[2] = np.clip(source_pos[2], 0.3, H - 0.3)

    # Build room. Use Sabine's formula to set materials matching desired RT60.
    # pra uses absorption coefficient; higher = more absorption = shorter RT60.
    e_absorption, max_order = pra.inverse_sabine(
        rt60=_rt60_from_absorption(abs_coef, W, L, H),
        room_dim=[W, L, H],
    )
    room = pra.ShoeBox(
        [W, L, H], fs=SAMPLE_RATE,
        materials=pra.Material(e_absorption),
        max_order=min(max_order, 12),  # cap for speed
    )
    room.add_source(source_pos.tolist())
    room.add_microphone(mic_pos.tolist())
    room.compute_rir()

    rir = room.rir[0][0]
    return rir


def _rt60_from_absorption(abs_coef: float, W: float, L: float, H: float) -> float:
    """Rough mapping from average absorption to RT60 via Sabine's formula."""
    # T60 = 0.161 * V / (S * a)  where V is volume, S is surface, a is mean absorption
    V = W * L * H
    S = 2 * (W * L + W * H + L * H)
    rt60 = 0.161 * V / (S * abs_coef)
    return max(0.15, min(rt60, 1.5))  # clamp to realistic range


def convolve_with_rir(audio: np.ndarray, rir: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Convolve audio with RIR, peak-normalize result."""
    convolved = np.convolve(audio, rir, mode="full")
    # Trim leading delay caused by RIR's direct-path offset (use up to original length)
    convolved = convolved[: len(audio) + len(rir) - 1]
    # Trim to roughly the original length plus a small reverb tail
    tail = min(len(rir), int(SAMPLE_RATE * 0.5))  # max 500ms tail
    convolved = convolved[: len(audio) + tail]

    # Peak normalize
    peak = float(np.max(np.abs(convolved)))
    if peak > 1e-6:
        convolved = convolved / peak * target_peak
    return convolved.astype(np.float32)


def add_background_noise(audio: np.ndarray, noise_db: float = -45.0) -> np.ndarray:
    """Add quiet white noise to simulate room noise floor.

    noise_db is the noise RMS in dBFS (e.g., -45 dBFS is barely audible).
    """
    if noise_db is None or noise_db < -100:
        return audio
    noise_amp = 10 ** (noise_db / 20.0)
    noise = np.random.randn(len(audio)).astype(np.float32) * noise_amp
    return audio + noise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-recording", type=int, default=3,
                        help="How many far-field versions to generate per source recording")
    parser.add_argument("--noise-db", type=float, default=-45.0,
                        help="Background noise level in dBFS (e.g., -45). Use -120 to disable.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Source files = only the ORIGINAL close-mic recordings (don't recurse into _ff variants)
    sources = sorted([
        f for f in SOURCE_DIR.glob("real_navkar_*.wav")
        if "_ff" not in f.stem and "_partial" not in f.stem
    ])
    print(f"Found {len(sources)} close-mic source recordings")
    print(f"Generating {args.n_per_recording} far-field versions per source -> "
          f"{len(sources) * args.n_per_recording} new files")
    print(f"Background noise: {args.noise_db} dBFS")
    print()

    total_generated = 0
    skipped = 0
    for src in tqdm(sources, desc="Sources"):
        audio, sr = sf.read(str(src))
        assert sr == SAMPLE_RATE
        for i in range(args.n_per_recording):
            out_path = SOURCE_DIR / f"{src.stem}_ff{i+1:02d}.wav"
            if out_path.exists():
                skipped += 1
                continue

            # Random room and distance for variety
            room_cfg = random.choice(ROOM_CONFIGS)
            dist_cfg = random.choice(DISTANCE_CONFIGS)

            try:
                rir = generate_rir(room_cfg, dist_cfg)
                far_audio = convolve_with_rir(audio, rir)
                far_audio = add_background_noise(far_audio, args.noise_db)
                # Re-clip after noise addition
                far_audio = np.clip(far_audio, -1.0, 1.0)
                sf.write(str(out_path), far_audio, sr, subtype="PCM_16")
                total_generated += 1
            except Exception as e:
                print(f"  Failed {out_path.name}: {e}")

    print(f"\nGenerated {total_generated} far-field samples ({skipped} already existed)")
    total_chants = len(list(SOURCE_DIR.glob("real_navkar_*.wav")))
    print(f"Total source chants in {SOURCE_DIR}: {total_chants}")


if __name__ == "__main__":
    main()
