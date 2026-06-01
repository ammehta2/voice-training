"""
Far-field augmentation v2 — pushes harder than scripts/13_simulate_far_field.py.

Motivation: user reported real-world scores peaking at 0.65 (well below v3's
0.88 threshold), meaning the model is genuinely uncertain on audio further
than our previous simulation covered. v1 simulated 0.5-3m at RT60 0.3-0.7s.
v2 pushes to 4-6m, RT60 up to 1.2s, louder background noise, and adds an
"occlusion" mode that low-passes the signal to simulate phone-in-pocket or
behind-an-object scenarios.

Per-source-recording: generates 6 far-field versions (vs 3 in v1) so the
combined dataset (originals + v1 far-field + v2 far-field) has much better
coverage of the realistic deployment manifold.

Naming: outputs `_ff2_NN.wav` to distinguish from v1's `_ff_NN.wav`. Both
co-exist; both will be picked up by 00d_prep_closing_phrase.py.

Usage:
    python scripts/13b_simulate_far_field_v2.py
    python scripts/13b_simulate_far_field_v2.py --n-per-recording 6 --noise-db -35
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pyroomacoustics as pra
import soundfile as sf
from scipy import signal
from tqdm import tqdm

SAMPLE_RATE = 16000
SOURCE_DIR = Path("data/positive_raw_originals")

# More aggressive room configs (bigger + more reverberant)
ROOM_CONFIGS = [
    # (W,    L,    H,    absorption_avg, name)
    ((4.0,  5.0,  3.0), 0.50, "bedroom_normal"),
    ((4.0,  5.0,  3.0), 0.30, "bedroom_reverb"),
    ((5.0,  7.0,  3.0), 0.45, "living_normal"),
    ((5.0,  7.0,  3.0), 0.28, "living_reverb"),
    ((6.0,  8.0,  3.5), 0.35, "large_room"),
    ((3.5,  4.0,  2.7), 0.55, "small_room"),
    ((8.0,  10.0, 4.0), 0.25, "hall_reverb"),
    # NEW v2 rooms
    ((10.0, 12.0, 4.0), 0.20, "very_reverberant_hall"),  # gymnasium / temple
    ((6.0,  4.0,  3.0), 0.40, "kitchen_with_hard_surfaces"),
    ((3.0,  3.5,  2.5), 0.65, "absorbed_studio"),  # carpet + curtains, low reverb
]

# Pushed-out distance configs (up to 6m)
DISTANCE_CONFIGS = [
    # (distance_m, source_height_m, name)
    (0.5, 1.2, "close_seated"),
    (1.0, 1.2, "1m_seated"),
    (1.5, 1.6, "1.5m_standing"),
    (2.0, 1.6, "2m_standing"),
    (3.0, 1.6, "3m_standing"),
    # NEW v2 distances
    (4.0, 1.6, "4m_across_room"),
    (5.0, 1.6, "5m_far_corner"),
    (6.0, 1.6, "6m_very_far"),
]

# Occlusion presets — applied AFTER RIR convolution to simulate
# phone-in-pocket / behind-object / muffled scenarios. None = no occlusion.
OCCLUSION_PRESETS = [
    None,                                # 30% of samples: no occlusion
    None,
    None,
    {"lowpass_hz": 5000, "attenuation_db": -3},   # mild: phone face-down
    {"lowpass_hz": 3500, "attenuation_db": -6},   # moderate: phone in shirt pocket
    {"lowpass_hz": 2500, "attenuation_db": -9},   # heavy: phone in pant pocket / behind cushion
]


def _rt60_from_absorption(abs_coef: float, W: float, L: float, H: float) -> float:
    V = W * L * H
    S = 2 * (W * L + W * H + L * H)
    rt60 = 0.161 * V / (S * abs_coef)
    return max(0.15, min(rt60, 2.0))


def generate_rir(room_config, distance_config, mic_height: float = 1.0) -> np.ndarray:
    (W, L, H), abs_coef, _ = room_config
    distance, source_height, _ = distance_config

    mic_pos = np.array([W / 2, L / 2 - 0.5, mic_height])

    angle = random.uniform(0, 2 * np.pi)
    source_pos = mic_pos + np.array([
        distance * np.cos(angle),
        distance * np.sin(angle),
        (source_height - mic_height),
    ])
    source_pos[0] = np.clip(source_pos[0], 0.3, W - 0.3)
    source_pos[1] = np.clip(source_pos[1], 0.3, L - 0.3)
    source_pos[2] = np.clip(source_pos[2], 0.3, H - 0.3)

    e_abs, max_order = pra.inverse_sabine(
        rt60=_rt60_from_absorption(abs_coef, W, L, H),
        room_dim=[W, L, H],
    )
    room = pra.ShoeBox(
        [W, L, H], fs=SAMPLE_RATE,
        materials=pra.Material(e_abs),
        max_order=min(max_order, 15),  # higher than v1 for more accurate big rooms
    )
    room.add_source(source_pos.tolist())
    room.add_microphone(mic_pos.tolist())
    room.compute_rir()
    return room.rir[0][0]


def convolve_with_rir(audio: np.ndarray, rir: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    convolved = np.convolve(audio, rir, mode="full")
    tail = min(len(rir), int(SAMPLE_RATE * 0.8))  # allow longer reverb tail than v1
    convolved = convolved[: len(audio) + tail]
    peak = float(np.max(np.abs(convolved)))
    if peak > 1e-6:
        convolved = convolved / peak * target_peak
    return convolved.astype(np.float32)


def apply_occlusion(audio: np.ndarray, preset: dict | None) -> np.ndarray:
    """Simulate phone-in-pocket / behind-object via low-pass + attenuation."""
    if preset is None:
        return audio
    cutoff = preset["lowpass_hz"]
    atten_db = preset["attenuation_db"]
    # Butterworth low-pass
    sos = signal.butter(4, cutoff / (SAMPLE_RATE / 2), btype="low", output="sos")
    filtered = signal.sosfiltfilt(sos, audio).astype(np.float32)
    # Apply attenuation
    gain = 10 ** (atten_db / 20.0)
    return (filtered * gain).astype(np.float32)


def add_background_noise(audio: np.ndarray, noise_db: float) -> np.ndarray:
    if noise_db is None or noise_db < -100:
        return audio
    noise_amp = 10 ** (noise_db / 20.0)
    noise = np.random.randn(len(audio)).astype(np.float32) * noise_amp
    return audio + noise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-recording", type=int, default=6,
                        help="Far-field versions per source recording (v1 used 3)")
    parser.add_argument("--noise-db", type=float, default=-35.0,
                        help="Noise floor in dBFS (v1 used -45; v2 default -35 is louder)")
    parser.add_argument("--seed", type=int, default=43,
                        help="Use different seed than v1 (which was 42)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Sources = ALL positive source recordings that aren't already-augmented variants.
    # Includes: real recordings, Sarvam TTS, MMS-TTS synthetics.
    # Excludes: existing _ff (v1) and _ff2 (v2) far-field variants, partials.
    sources = sorted([
        f for f in SOURCE_DIR.glob("*.wav")
        if (("real_navkar_" in f.stem or "sarvam_navkar_" in f.stem or "mmstts_navkar_" in f.stem)
            and "_ff" not in f.stem
            and "_partial" not in f.stem)
    ])
    print(f"Sources: {len(sources)} close-mic recordings (real + Sarvam + MMS-TTS)")
    print(f"Generating {args.n_per_recording} v2 far-field versions per source -> "
          f"{len(sources) * args.n_per_recording} new files")
    print(f"Distances: {[d[0] for d in DISTANCE_CONFIGS]} m")
    print(f"Rooms: {[r[2] for r in ROOM_CONFIGS]}")
    print(f"Noise: {args.noise_db} dBFS")
    print(f"Occlusion presets: {len(OCCLUSION_PRESETS)} (3 no-op, 3 active)")
    print()

    generated = 0
    skipped = 0
    for src in tqdm(sources, desc="Sources"):
        audio, sr = sf.read(str(src))
        assert sr == SAMPLE_RATE, f"{src.name}: sr={sr}"

        for i in range(args.n_per_recording):
            out_path = SOURCE_DIR / f"{src.stem}_ff2_{i+1:02d}.wav"
            if out_path.exists():
                skipped += 1
                continue

            room_cfg = random.choice(ROOM_CONFIGS)
            dist_cfg = random.choice(DISTANCE_CONFIGS)
            occlusion = random.choice(OCCLUSION_PRESETS)

            try:
                rir = generate_rir(room_cfg, dist_cfg)
                far_audio = convolve_with_rir(audio, rir)
                far_audio = apply_occlusion(far_audio, occlusion)
                far_audio = add_background_noise(far_audio, args.noise_db)
                far_audio = np.clip(far_audio, -1.0, 1.0)
                sf.write(str(out_path), far_audio, sr, subtype="PCM_16")
                generated += 1
            except Exception as e:
                print(f"  Failed {out_path.name}: {e}")

    print(f"\nGenerated {generated} v2 far-field samples ({skipped} already existed)")
    total = len(list(SOURCE_DIR.glob("real_navkar_*.wav")))
    print(f"Total chants in {SOURCE_DIR}: {total}")


if __name__ == "__main__":
    main()
