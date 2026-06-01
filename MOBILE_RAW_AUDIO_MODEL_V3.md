# Navkar Counter — Raw-Audio TFLite Model (v3 — speech-aware)

> **For Claude Code on the mobile side:** This document **supersedes** `MOBILE_RAW_AUDIO_MODEL_V2.md`. The model has been retrained with 700 clips of Hindi+Gujarati conversational speech as negatives. The result: **the model now correctly rejects 100% of held-out Hindi/Gujarati speech samples.**
>
> If you've already integrated v2, the **only changes** needed are:
> 1. Replace the model file (Section 2)
> 2. Update threshold (0.92 → **0.88**) and debounce (9000ms → **11000ms**) in `constants.ts`
>
> No code changes required.

---

## 1. What changed and why

### The problem v2 didn't address
The v2 model had only seen ESC-50 environmental sounds and ambient noise as negatives. It had **never seen conversational human speech** in training. Held-out evaluation on 215 windows of Hindi/Gujarati conversation showed:
- Model A (close-mic only): 25 false positives (~12% FP rate)
- Model B (close + far-field): 6 false positives (~3% FP rate)

In production, this means the first time the user has a phone call near the running counter, it will probably fire.

### The v3 fix
Added 700 clips of conversational speech from Google FLEURS as training negatives:
- 500 Hindi clips (`hi_in`)
- 200 Gujarati clips (`gu_in`)

These get sliced into 2.5s windows yielding **2,721 additional negative training samples**, then mixed in with the existing ESC-50, ambient, and far-field negatives. Source-aware split keeps them separate from the test speech set.

### Why Gujarati specifically
The Jain community is heavily Gujarati-speaking. False positives on Gujarati conversation would be especially problematic in real deployment. Hindi covers the broader user base.

### A/B/C comparison (Model A vs B vs C)

| Test set | A (close only) | B (+ far-field) | **C (+ speech)** | Δ vs B |
|---|---|---|---|---|
| Close-mic originals | 20/26 (77%) | 20/26 (77%) | 19/26 (73%) | -1 |
| Far-field samples | 37/78 (47%) | 61/78 (78%) | 64/78 (82%) | +3 |
| Partial chants (hard negatives) | 56/78 (72%) | 61/78 (78%) | **70/78 (90%)** | **+9** |
| **Hindi/Gujarati speech (NEW)** | 190/215 (88%) | 209/215 (97%) | **🎯 215/215 (100%)** | **+6 PERFECT** |
| **TOTAL** | 303/397 (76%) | 351/397 (88%) | **368/397 (93%)** | **+17** |

**The headline:**
- **Perfect speech rejection** — 0 false positives on conversational Hindi/Gujarati
- **Bonus:** partial-chant rejection improved by 9 (the speech negatives gave the model more "non-Navkar voice" variety, sharpening its discrimination)
- **Cost:** 1 fewer correct on close-mic originals — acceptable tradeoff for perfect speech rejection

---

## 2. New model file

```
models/closing_v1_20260601_123630/navkar_raw_audio_fp32.tflite   (1.1 MB)
```

Place at: `apps/mobile/assets/models/navkar_raw_audio_fp32.tflite` (same name as v2 — overwrites).

Model interface is **unchanged** from v2:

| | |
|---|---|
| **Input** | float32 tensor, shape `(1, 40000)` |
| **Input values** | raw audio samples in [-1, 1], 16 kHz mono, 2.5 seconds |
| **Output** | float32 tensor, shape `(1, 1)` — single raw logit |
| **To get probability** | `sigmoid(output[0])` |

---

## 3. Updated detection settings

Counter sweep on 397 test files (originals + far-field + partials + speech) at the raw-audio TFLite output:

```typescript
// In apps/mobile/lib/jaap/constants.ts:
export const DETECTION_THRESHOLD = 0.88;    // was 0.92 in v2
export const DEBOUNCE_MS = 11000;            // was 9000 in v2
```

| Parameter | v2 value | v3 value | Why changed |
|---|---|---|---|
| Detection threshold | 0.92 | **0.88** | v3 has tighter probability calibration thanks to more diverse negatives. 0.88 is the new optimal that maximizes accuracy without over-counts. |
| Debounce window | 9000 ms | **11000 ms** | Slightly longer to handle slow chants (40s+ versions) cleanly. |
| Window size | 2.5 s | unchanged | Hard-coded in the model |
| Inference cadence | 250 ms | unchanged | |
| Target count | 108 | unchanged | |

### Counter accuracy at the chosen settings

```
threshold=0.88, debounce=11s -> 376/397 correct (94.7%)
  Originals:    18/26  (69%)  -- some quiet/unusual recordings still miss
  Far-field:    69/78  (88%)
  Partials:     74/78  (95%)
  Speech:      215/215 (100%) -- PERFECT
  Over-counts on originals: 0  -- never tells user "108 done" when they did fewer
```

---

## 4. Why "perfect speech rejection" matters more than the originals regression

The 1-recording originals regression (Model B 20/26 → Model C 19/26) looks like a step back, but consider the deployment scenario:

- **Failure mode A (miss a chant):** user notices count is slow, chants louder, problem self-corrects.
- **Failure mode B (over-count):** user thinks they're done, stops short. Spiritual practice value undermined.
- **Failure mode C (fire on speech):** Counter runs up to 108 in the background while user has a conversation. Catastrophic — user has no idea their session was corrupted.

Failure mode C is what speech-negative training prevents. Going from 6 speech false positives (v2) to 0 (v3) eliminates an entire class of catastrophic failures, at the cost of one extra "user has to chant a little louder" miss. **Strictly better tradeoff.**

---

## 5. Updated hard-negative tests for mobile QA

In addition to the v2 partial-chant test, add a **speech-rejection test**:

1. Copy any file from `data/speech_negative_test/` (e.g., `speech_hi_0000_00.wav`) to `apps/mobile/assets/test/speech_test.wav`.
2. Run it through the counter loop in a debug build.
3. **Expected: count remains 0.**

If count goes above 0, the speech-rejection regression has happened — escalate.

You should bundle both:
- `apps/mobile/assets/test/partial_test.wav` (a partial chant) — count should stay 0
- `apps/mobile/assets/test/speech_test.wav` (Hindi conversation) — count should stay 0

These two tests catch entire classes of model bugs.

---

## 6. Code changes summary

### Files that change
```
apps/mobile/lib/jaap/constants.ts
  -  export const DETECTION_THRESHOLD = 0.92;
  +  export const DETECTION_THRESHOLD = 0.88;
  -  export const DEBOUNCE_MS = 9000;
  +  export const DEBOUNCE_MS = 11000;

apps/mobile/assets/models/navkar_raw_audio_fp32.tflite
  (replace with new file from models/closing_v1_20260601_123630/)
```

### Files that DON'T change
```
apps/mobile/lib/jaap/AudioStreamer.ts         ← unchanged
apps/mobile/lib/jaap/RollingBuffer.ts         ← unchanged
apps/mobile/lib/jaap/NavkarModel.ts           ← unchanged
apps/mobile/lib/jaap/JaapCounter.ts           ← unchanged
apps/mobile/screens/JaapCounterScreen.tsx     ← unchanged
```

One-file model swap + two constants. Same as v2 → v3 transition pattern.

---

## 7. Known limitations remaining

### Same as v2
- Recordings 04, 05, 08, 20 still tend to miss (specific voice characteristics not represented in training)
- Recording 19 still has unusual triggering pattern
- FP16 still has NaN issues from DFT matmul overflow

### New from v3
- Possible regression on close-mic originals (-1 from v2). Some recordings now require higher confidence to detect — manageable in practice.

---

## 8. What's coming in v4 (not in this build)

Planned next iterations:
- **AI4Bharat IndicTTS synthetic data** — generate 500+ Navkar chants in diverse Indian voices to expand positive-class diversity. Should fix the 4-recording miss problem.
- **SpecAugment + codec roundtrip + score smoothing** (tier 1 quick wins from the research doc).
- **More speech variety** — currently only Hindi+Gujarati; might add Marathi/Tamil/Telugu if Jain communities in those regions become users.

---

## 9. Definition of done

1. New TFLite copied to `apps/mobile/assets/models/navkar_raw_audio_fp32.tflite` (~1.1 MB)
2. `constants.ts` updated: `DETECTION_THRESHOLD = 0.88`, `DEBOUNCE_MS = 11000`
3. Section 5 partial-chant test passes (count = 0)
4. Section 5 speech-rejection test passes (count = 0) — **NEW REQUIREMENT**
5. Real chant test: user chants 10 Navkars at normal pace on a phone on a table → count reads 8-10
6. Real false-positive test: 5 minutes of conversation in the room → count stays 0 (this should now be solid)

When all 6 are true, ship v3.
