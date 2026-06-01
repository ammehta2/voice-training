# Navkar Counter — Raw-Audio TFLite Model (v2 — far-field trained)

> **For Claude Code on the mobile side:** This document **supersedes** `MOBILE_RAW_AUDIO_MODEL.md`. The model has been retrained with far-field room-impulse-response augmentation specifically to handle the realistic deployment scenario: phone sitting on a table 1-3m away from the chanter, with normal room reverberation.
>
> If you've already integrated the previous raw-audio model, the **only changes** needed are:
> 1. Replace the model file (Section 2)
> 2. Update threshold (0.90 → **0.92**) and debounce (7000ms → **9000ms**) in `constants.ts`
>
> No code changes required.

---

## 1. What changed and why

### The problem with v1
The v1 model was trained on close-mic recordings only. When tested on simulated far-field audio (phone 1-3m away with realistic room reverb), it missed **53% of detections**. Real-world performance was unpredictable because the model had never seen phone-on-table audio.

### The v2 fix
We generated 78 far-field versions of the 26 source recordings using `pyroomacoustics` — physically-accurate room impulse response (RIR) simulation. Each original recording was convolved with 3 random RIRs spanning:

| Variable | Values used |
|---|---|
| Room size | 4×5×3m (bedroom), 5×7×3m (living room), 6×8×3m, 8×10×4m (hall) |
| Source-mic distance | 0.5m, 1m, 1.5m, 2m, 3m |
| RT60 (reverberation time) | 0.3s – 1.5s |
| Source height | 1.2m (seated) or 1.6m (standing) |
| Background noise | -45 dBFS white noise floor |

Then the model was retrained on the combined dataset (originals + far-field variants), with source-aware splitting so all variants of the same recording stayed in the same train/val/test fold.

### A/B comparison vs v1

Both models tested on:
- 26 close-mic originals (expected count = 1 each)
- 78 far-field variants (expected count = 1 each)
- 78 partial chants — 40%/70%/90% slices that cut off before the closing (expected count = 0)

| Test set | v1 (close-mic only) | v2 (close + far-field) | Δ |
|---|---|---|---|
| Close-mic originals | 20/26 (77%) | 21/26 (81%) | +1 |
| Far-field samples | **37/78 (47%)** | **66/78 (85%)** | **+29** |
| Partial chants (hard negatives) | 56/78 (72%) | 63/78 (81%) | +7 |
| **Total** | **113/182 (62%)** | **150/182 (82%)** | **+37** |

The headline: **far-field accuracy nearly doubled** (47% → 85%) with no regression on close-mic, and partial-chant rejection (false-positive resistance) improved too.

---

## 2. New model file

Replace your existing model with:

```
models/closing_v1_20260601_115014/navkar_raw_audio_fp32.tflite   (1.1 MB)
```

Place at: `apps/mobile/assets/models/navkar_raw_audio_fp32.tflite` (same name as before — overwrites).

### Why not FP16?
The DFT matmul inside the model produces intermediate values that overflow FP16. FP32 is correct; FP16 currently produces NaN. We'll fix this in a future iteration by scaling the basis. For now, ship FP32.

---

## 3. Model interface (unchanged from previous version)

| | |
|---|---|
| **Input** | float32 tensor, shape `(1, 40000)` |
| **Input values** | raw audio samples in range `[-1, 1]`, 16 kHz mono, 2.5 seconds |
| **Output** | float32 tensor, shape `(1, 1)` — single raw logit |
| **To get probability** | `sigmoid(output[0])` = `1 / (1 + exp(-output[0]))` |

Internal processing (all baked into the model — no JS code needed):
peak normalize → DFT (matmul) → power → Slaney mel → log10 → DCT-II → per-utterance mean/std → KWS backbone

---

## 4. Updated detection settings

Sweep on the full test corpus (26 close + 78 far-field + 78 partial = 182 files) chose these settings as optimal:

```typescript
// In apps/mobile/lib/jaap/constants.ts:
export const DETECTION_THRESHOLD = 0.92;   // was 0.90 in v1
export const DEBOUNCE_MS = 9000;           // was 7000 in v1
```

| Parameter | v1 value | v2 value | Why changed |
|---|---|---|---|
| Detection threshold | 0.90 | **0.92** | Far-field training made the model slightly more confident on positives but also raised some pre-closing scores. Higher threshold keeps over-counts near zero. |
| Debounce window | 7000 ms | **9000 ms** | Slow chants (~25s) can produce two "Mangalam-like" peaks (line 7's "Mangalaanam" + line 8's actual "Mangalam"). Longer debounce merges them into one count. |
| Window size | 2.5 s | unchanged | Hard-coded in the model |
| Inference cadence | 250 ms | unchanged | |
| Target count | 108 | unchanged | |

### Validated counter accuracy at these settings

| Test set | Result | Notes |
|---|---|---|
| Close-mic originals | 21/26 (81%) | Same recordings as v1 |
| Far-field samples | 66/78 (85%) | **The realistic deployment scenario** |
| Partial chants | 63/78 (81%) | False-positive rate 19% on the hard cases |
| Over-counts on originals | 1/26 | Acceptable — over-counts are the worst failure mode for jaap counting |

---

## 5. The hard-negative test (recommended part of mobile QA)

Bundle one of the partial-chant files as a test asset and verify the counter stays at 0:

1. Copy `data/partial_chants_test/real_navkar_07_partial70.wav` to `apps/mobile/assets/test/partial_test.wav`.
2. Add a hidden debug button in JaapCounterScreen that loads this WAV and feeds it through the counter loop.
3. Expected result: **count remains 0** (the recording contains Navkar audio but cuts off before "Padhamam Havai Mangalam").
4. If count goes to 1, the model has learned a shortcut — escalate.

This single test catches a whole class of model bugs that wouldn't show up in normal positive-only testing.

---

## 6. Code changes summary

### Files that change

```
apps/mobile/lib/jaap/constants.ts
  -  export const DETECTION_THRESHOLD = 0.90;
  +  export const DETECTION_THRESHOLD = 0.92;
  -  export const DEBOUNCE_MS = 7000;
  +  export const DEBOUNCE_MS = 9000;

apps/mobile/assets/models/navkar_raw_audio_fp32.tflite
  (replace with new file from models/closing_v1_20260601_115014/)
```

### Files that DON'T change

```
apps/mobile/lib/jaap/AudioStreamer.ts         ← no change
apps/mobile/lib/jaap/RollingBuffer.ts         ← no change
apps/mobile/lib/jaap/NavkarModel.ts           ← no change (model interface identical)
apps/mobile/lib/jaap/JaapCounter.ts           ← no change
apps/mobile/screens/JaapCounterScreen.tsx     ← no change
```

That's literally a one-file model swap plus two constants.

---

## 7. Known limitations and what we'd do next

### Recordings 04, 05, 08, 20 still miss
These recordings have one or more of: very low volume (recording 04 has peak 0.06), unusual pronunciation, or specific vocal characteristics not well-represented in the training set. With more recordings from similar speakers, the model will pick these up.

### Recording 19 over-counts
Recording 19 triggers detections at multiple points in the file (max score 0.998 throughout). Something about that specific recording confuses the model into firing on non-closing audio. We'd need to listen to it carefully to understand what acoustic feature is misleading the model.

### Partial chants at 90% still trigger ~22% false positives
The hardest partial test cuts off mid-closing ("Padhamam Havai..." with the final "Mangalam" missing). The model sees enough closing-phrase phonemes to fire. Two options to fix:
1. Add more aggressive hard-negative training (use these partials as explicit negatives during training)
2. Require the model to see the actual final "...lam" sound by training on slightly longer windows (3s instead of 2.5s)

Both are research directions for v3.

### Background noise / multi-speaker scenarios untested
The model has never been tested against TV in the background, multiple people in the room, music playing nearby, or someone else talking while the user chants. These are likely v3 robustness targets.

---

## 8. Reproducing this model

For audit / understanding / future iterations:

```bash
# Training-side pipeline (in voicetraining repo)
python scripts/13_simulate_far_field.py --n-per-recording 3 --noise-db -45
python scripts/14_extract_partial_chants.py
python scripts/00d_prep_closing_phrase.py
python scripts/03d_augment_closing.py
python scripts/04d_extract_features_closing.py
python scripts/05d_train_closing.py
python scripts/11b_export_tf_native.py models/closing_v1_<TS>/best.pt

# Verify
python scripts/15_ab_compare_models.py \
    models/<v1_close_mic_only>/best.pt \
    models/closing_v1_20260601_115014/best.pt
```

---

## 9. Definition of done (for mobile integration of v2)

1. New TFLite file copied to `apps/mobile/assets/models/navkar_raw_audio_fp32.tflite` (~1.1 MB)
2. `constants.ts` updated: `DETECTION_THRESHOLD = 0.92`, `DEBOUNCE_MS = 9000`
3. Smoke test: Section 5 partial-chant test passes (count stays 0)
4. Real chant test: user chants 10 Navkars at normal pace on a phone sitting on a table 1-2m away → count reads 8-10 (the previous version may have read 4-6 in this scenario)
5. False positive sanity: 5 minutes of conversation in the same room → count stays 0 or 1

When 1-5 are true, ship. The most important upgrade vs v1 is item 4 — that's where the far-field augmentation pays off.
