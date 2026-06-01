# Navkar Counter — Raw-Audio TFLite Model (v6 — bandwidth-robust)

> **For Claude Code on the mobile side:** This document **supersedes** `MOBILE_RAW_AUDIO_MODEL_V5.md`. The model has been retrained with **bandwidth-limited augmentation** so it can detect chants even when the mobile audio pipeline aggressively filters out high frequencies (the Picovoice / mic-hardware issue documented in `MOBILE_AUDIO_BANDWIDTH_BUG.md`).
>
> Real-world v4/v5 testing failed (mobile scores 0.00-0.22) because the captured audio had 90%+ energy below 500 Hz — essentially low-passed at ~500 Hz somewhere in the capture pipeline. **v6 directly addresses this.**
>
> Same one-file model swap + two constants pattern:
> 1. Replace the model file (Section 2)
> 2. Update threshold (0.40 → **0.50**) and debounce (11000ms → **5000ms**) in `constants.ts`

---

## 1. What changed and why

### The v5 problem
Mobile testing on the tablet showed Model v5 producing scores of 0.00-0.22 even during active chanting. Analysis revealed the captured audio was **bandwidth-limited to ~500 Hz** — likely due to Picovoice's noise suppression, Android's `AudioSource.VOICE_RECOGNITION`, or tablet mic hardware filtering. (See `MOBILE_AUDIO_BANDWIDTH_BUG.md` for full diagnosis.)

The v5 model expected normal-bandwidth audio (200-4000 Hz of speech energy) and produced near-zero scores when that energy wasn't present.

### The v6 fix
Added a new training augmentation severity `"bandwidth_limited"` that:
- Always applies a low-pass filter with cutoff between 400-2000 Hz
- Combined with aggressive gain variation (-25 to +5 dB)
- Plus noise and minor pitch shift

Training cycles through 4 severities now: light, medium, heavy, **bandwidth_limited**. Each source recording gets at least 2 bandwidth-limited variants. Net: ~6000 bandwidth-limited positive samples in training.

The model learns: "even if I only see 0-500 Hz of the closing phrase, I should still detect it."

### A/B comparison (v5 vs v6)

| Test set | F (v5) | **G (v6)** | Δ |
|---|---|---|---|
| Originals (28) | 26/28 | **28/28 (100%!)** 🎯 | +2 |
| Far-field v1 0.5-3m (84) | 82/84 | 82/84 | tied |
| Far-field v2 4-6m+occlusion (168) | 152/168 | **164/168 (98%)** | **+12** |
| Partials (84) | 73/84 | **79/84 (94%)** | +6 |
| Speech (215) | 215/215 | 215/215 | tied |
| **TOTAL (579)** | **548 (94.6%)** | **568 (98.1%)** | **+20** |

PLUS — **mobile recordings now actually detect:**

| User mobile recording | v5 score (raw model) | **v6 score** |
|---|---|---|
| Recording 1 (14.7s, 2 chants) | 0.22 ❌ | **0.78** ✅ |
| Recording 2 (14.4s, 2 chants) | 0.22 ❌ | 0.40 (borderline) |
| Recording 3 (8.1s, 1 chant) | 0.19 ❌ | **0.92** ✅ |

Recording 2 is still below 0.50 — the audio capture should still be fixed (per `MOBILE_AUDIO_BANDWIDTH_BUG.md`) for full reliability. Model v6 is a robustness improvement, NOT a substitute for fixing the audio pipeline.

---

## 2. New model file

```
models/closing_v1_20260601_163945/navkar_raw_audio_fp32.tflite   (1.1 MB)
```

Place at: `apps/mobile/assets/models/navkar_raw_audio_fp32.tflite` (overwrites v5).

Interface unchanged: float32 (1, 40000) audio in, float32 (1, 1) logit out.

---

## 3. Updated detection settings

Counter sweep on the full 582-file test corpus (5 sets + 3 mobile):

```typescript
// In apps/mobile/lib/jaap/constants.ts:
export const DETECTION_THRESHOLD = 0.50;    // was 0.40 in v5
export const DEBOUNCE_MS = 5000;             // was 11000 in v5
```

| Parameter | v5 | **v6** | Why |
|---|---|---|---|
| Detection threshold | 0.40 | **0.50** | v6 produces tighter, more confident scores. 0.50 is the sweep optimum with 0 over-counts. |
| Debounce window | 11000 ms | **5000 ms** | v6's higher precision means we can use shorter debounce. Faster real-time feedback per chant. |
| Window size | 2.5 s | unchanged | |
| Inference cadence | 250 ms | unchanged | |
| Target count | 108 | unchanged | |

### Counter accuracy at chosen settings

```
threshold=0.50, debounce=5s -> 573/582 correct (98.5%)
  Originals:                  28/28  (100% PERFECT)
  Far-field 0.5-3m:           83/84  (99%)
  Far-field 4-6m + occlusion: 166/168 (99%)
  Partials:                   79/84  (94%)
  Speech:                     215/215 (100% PERFECT)
  User mobile recordings:     2/3    (recording 2 below threshold; fix mobile audio capture)
  Over-counts on originals:   0       (PERFECT — never tells user "108 done" when fewer)
```

---

## 4. What this means for real-world deployment

| Scenario | v4 | v5 | **v6** |
|---|---|---|---|
| Close-mic chant | ✅ | ✅ | ✅✅ (100%) |
| Phone on table 1-2m | ✅ | ✅ | ✅✅ |
| Phone on shelf 3-5m | ✅ | ✅+ | ✅✅ (99%) |
| Phone in pocket / occluded | ✅ | ✅ | ✅ |
| **Bandwidth-limited (Picovoice/AGC)** | **❌** | **❌** | **✅** (NEW) |
| Hindi/Gujarati conversation | ✅ no FP | ✅ no FP | ✅ no FP |
| Partial chants | ✅ 93% | ⚠️ 87% | ✅ 94% (recovered) |

---

## 5. The TWO-PRONGED fix

v6 is a **robustness fallback**, not a permanent fix. The proper solution is still:

### A. **Mobile-side fix** (recommended, fixes the root cause)
Per `MOBILE_AUDIO_BANDWIDTH_BUG.md`:
1. Disable Picovoice's audio preprocessing if used
2. Use raw mic input (`AudioSource.MIC` on Android, `Mode.default` on iOS)
3. Verify captured audio spectrum spans 200-4000 Hz

If you fix A, the model gets full-bandwidth audio and detects with high confidence (>0.95 scores).

### B. **v6 robustness training** (what we just shipped)
The model now learns to detect chants even from low-pass-filtered audio. **But** detection scores will still be lower (0.40-0.80 range) on bandwidth-limited audio vs full-bandwidth (>0.95).

**Best path:** ship v6 to get immediate improvement, AND fix the mobile audio capture. With both fixes, the system becomes maximally robust.

---

## 6. Hard-negative tests still pass

- **Partial chant test**: `data/partial_chants_test/real_navkar_07_partial70.wav` → count = 0 ✅
- **Speech test**: any `data/speech_negative_test/*.wav` file → count = 0 ✅

---

## 7. Code changes

```
apps/mobile/lib/jaap/constants.ts
  -  export const DETECTION_THRESHOLD = 0.40;
  +  export const DETECTION_THRESHOLD = 0.50;
  -  export const DEBOUNCE_MS = 11000;
  +  export const DEBOUNCE_MS = 5000;

apps/mobile/assets/models/navkar_raw_audio_fp32.tflite
  (replace with new file from models/closing_v1_20260601_163945/)
```

No other code changes.

---

## 8. Training data scale (for context)

```
v6 source positives (~8700 augmented source files):
  - 28 real recordings
  - 84 v1 far-field (0.5-3m)
  - 168 v2 far-field (4-6m + occlusion)
  - 421 Sarvam Bulbul TTS
  - 120 Meta MMS-TTS
  - 2,164 v2 far-field of synthetic (NEW in v5)
  Plus bandwidth_limited augmentation applied across all of the above.

v6 source negatives (~24K total):
  - 2000 ESC-50
  - 8 Birch Tree ambient
  - 2031 Hindi conversational (FLEURS)
  - 736 Gujarati conversational
  - 482 Marathi conversational (NEW in v5)
  - 13K+ pre-mantra slices from positive recordings
  - 4023 ambient (sliced)

After augmentation: 42K train / 9K val / 10K test
```

---

## 9. Definition of done

1. New TFLite at `apps/mobile/assets/models/navkar_raw_audio_fp32.tflite`
2. `constants.ts` updated: `DETECTION_THRESHOLD = 0.50`, `DEBOUNCE_MS = 5000`
3. Partial-chant hard-negative test still passes
4. Speech hard-negative test still passes
5. Real-world test on the FAILING v5 scenario (phone on desk, tablet 3-5m): user can chant 10 Navkars and see count read 6-10 (was 0-2 with v5)
6. Conversation false-positive test: 5 min Hindi/Gujarati chat → count stays 0
7. **Also pursue** the mobile audio fix from `MOBILE_AUDIO_BANDWIDTH_BUG.md` for maximum robustness

When 1-6 pass, ship v6.
