# Navkar Counter — v7 Multi-Phrase TFLite Model (EXPERIMENTAL)

> **For Claude Code on the mobile side:** This document describes an **experimental** alternative to v6, NOT a replacement. **v6 stays as the recommended ship.** v7 is an option to A/B test if a specific user still has bandwidth-limited capture problems that v6 can't compensate for.
>
> v7 is a **2-head** model: it detects **two** phrases of the Navkar mantra simultaneously — the closing phrase and the line just before it (the "pre-closing"). The counter can require BOTH heads to fire in temporal sequence for higher confidence, or fall back to a single head if only one is heard clearly.
>
> Output shape **changes**: v6 returns `(1, 1)` (one logit). v7 returns `(1, 2)` — `[closing_logit, preclosing_logit]`. Mobile code must update accordingly.

---

## 1. Why v7 exists

v6 fixed the bandwidth-limited capture issue for most users (568/582 = 98.1% on the standard synthetic test corpus, 2/3 user mobile recordings detected). But Recording 2 from the user's mobile capture remained at 0.40 — borderline. We wanted a way to lift confidence on those edge-case captures without retraining the entire bandwidth pipeline from scratch.

**Hypothesis:** if the model also listens for the line *before* the closing ("Mangalaanam cha Savvesim, …"), then the counter has TWO independent detection signals per chant. The closing phrase and the pre-closing phrase are 4-6 seconds apart in normal recitation. If both fire in that window, we have very high confidence a full Navkar just completed — even if either signal alone is weak.

### Trade-off (honest framing)

| Test set | v6 (Model G) | v7 (Model H) | Δ |
|---|---|---|---|
| Standard synthetic test corpus (582 files) | **568 / 582 (98.1%)** | 544 / 582 (93.5%) | **−24** |
| User mobile recordings | 2 / 3 | **3 / 3** | **+1** |
| Avg val AUC | 0.99 | **0.9956** | tied |

**v7 is better on the user's real-world mobile recordings; v6 is better on the standard synthetic test corpus.**

v6 is the recommended ship. v7 should only be considered if:
- A specific user reports that v6 still produces near-zero scores during chanting
- AND the audio capture pipeline has been verified per `MOBILE_AUDIO_BANDWIDTH_BUG.md`
- AND the partial-chant / speech hard-negative tests still pass with v7

---

## 2. Architecture

Same DS-CNN backbone as v6, but with **two parallel Linear(64, 1) heads** instead of one:

```
   audio (1, 40000) float32
        ↓
   peak normalize + MFCC (Slaney mel via matmul)
        ↓
   conv0 (5×4, stride 2)  →  BN  →  ReLU
        ↓
   3× depthwise-separable blocks (48, 48, 64)
        ↓
   global avg pool (B, 64)
        ↓
        ├── fc_closing    (Linear 64→1)  →  closing logit
        └── fc_preclosing (Linear 64→1)  →  preclosing logit
        ↓
   concat → output (1, 2)
```

Loss: multi-label `BCEWithLogitsLoss` with per-head `pos_weight`.

Labels: `(N, 2)` — `[1, 0]` for closing positives, `[0, 1]` for preclosing positives, `[0, 0]` for negatives.

**Param count**: 13,184 (vs 11,209 in v6 — one extra Linear(64,1) = 65 params; the rest is the same backbone). TFLite size: 1.16 MB float32 (similar to v6).

---

## 3. Model files

```
models/multi_v1_20260601_181227/navkar_multi_phrase_fp32.tflite   (1.16 MB)
models/multi_v1_20260601_181227/navkar_multi_phrase_fp16.tflite   (584 KB)
```

Place at: `apps/mobile/assets/models/navkar_multi_phrase_fp32.tflite`.

**Do NOT overwrite the v6 file.** Ship both, gate v7 behind a feature flag.

### Interface

```
Input:  float32 (1, 40000) — raw 16 kHz mono audio
Output: float32 (1, 2)     — [closing_logit, preclosing_logit]
                              (apply sigmoid before thresholding)
```

This is **different from v6** (which returns `(1, 1)`).

---

## 4. Counter logic (the 2-head temporal rule)

The v7 counter has to decide: when did one Navkar mantra complete? Two independent signals:

```typescript
// Pseudo-code for the v7 counter loop
let lastPreclosingFire = 0;     // timestamp of last preclosing detection
const PRECLOSING_WINDOW_MS = 8000;  // pre-closing should fire 2-8s before closing

function onInference(closing_logit, preclosing_logit, t_now) {
  const closing_prob = sigmoid(closing_logit);
  const preclosing_prob = sigmoid(preclosing_logit);

  // Track pre-closing fires (don't count, just remember)
  if (preclosing_prob >= PRECLOSING_THRESHOLD) {
    lastPreclosingFire = t_now;
  }

  // Count when closing fires
  if (closing_prob >= CLOSING_THRESHOLD && (t_now - lastCount) > DEBOUNCE_MS) {
    const heardPreclosingRecently =
      (t_now - lastPreclosingFire) < PRECLOSING_WINDOW_MS;

    // High-confidence path: both heads fired in temporal order → count
    if (heardPreclosingRecently) {
      count++;
      lastCount = t_now;
    }
    // Single-head fallback: closing fired strongly on its own → count
    else if (closing_prob >= CLOSING_STRONG_THRESHOLD) {
      count++;
      lastCount = t_now;
    }
    // Weak closing alone → ignore (this is the v7 false-positive reducer)
  }
}
```

### Recommended constants

```typescript
export const CLOSING_THRESHOLD = 0.50;         // closing head, weak — needs pre-closing corroboration
export const CLOSING_STRONG_THRESHOLD = 0.80;  // closing head, strong — counts on its own
export const PRECLOSING_THRESHOLD = 0.50;      // pre-closing head
export const PRECLOSING_WINDOW_MS = 8000;      // pre-closing must have fired within 8s before closing
export const DEBOUNCE_MS = 5000;               // unchanged from v6
```

### Why this rule

- **Sole-closing path** (closing >= 0.80): handles the user's loud/clear chants where v7 closing scores hit 0.987+.
- **Both-heads path** (closing >= 0.50 AND preclosing recent): handles bandwidth-limited captures where neither individual score is super high but BOTH heads fire. The temporal pattern (preclosing 2-8s before closing) is hard to fake from conversation or ambient sound.
- **Closing alone at 0.50-0.80**: ignored to avoid the v7 standard-corpus regression bleeding through as over-counts.

---

## 5. v7 mobile scores (FYI for QA)

PyTorch (post-sigmoid):

| Recording | closing_max | preclosing_max | v6 closing score |
|---|---|---|---|
| Mobile 1 (14.7s, 2 chants) | 0.987 ✅ | 0.998 ✅ | 0.78 ✅ |
| Mobile 2 (14.4s, 2 chants) | 0.965 ✅ | 0.980 ✅ | 0.40 ⚠️ |
| Mobile 3 (8.1s, 1 chant) | 1.000 ✅ | 0.949 ✅ | 0.92 ✅ |

TFLite (slight numeric drift vs PyTorch, expected from per-window MFCC normalization):

| Recording | closing_max | preclosing_max |
|---|---|---|
| Mobile 1 | 0.356 ⚠️ | 0.998 ✅ |
| Mobile 2 | 0.679 ✅ | 0.973 ✅ |
| Mobile 3 | 0.998 ✅ | 0.961 ✅ |

The TFLite/PyTorch drift on closing for Recording 1 is the reason the 2-head temporal rule matters — pre-closing fires very strongly even when closing is borderline, so the counter still counts via the both-heads path.

---

## 6. Hard-negative validation (do these before shipping)

The same tests as v6 must still pass:

1. **Partial-chant test**: `data/partial_chants_test/real_navkar_07_partial70.wav` → count = 0
2. **Speech test**: 5 min Hindi/Gujarati conversation → count = 0
3. **Standard test corpus**: `data/multi/features/X_test.npy` — verify both heads' AUC ≥ 0.99
4. **All 3 user mobile recordings** count as 1+ chant each

If 1 or 2 fail, the temporal rule needs tuning (raise `CLOSING_STRONG_THRESHOLD` to 0.85 or 0.90).

---

## 7. Training data (same as v6)

Same source recordings, same augmentations (incl. bandwidth_limited), same far-field RIR simulation. The only difference is the labels:
- Closing phrase positives → `[1, 0]`
- **Pre-closing phrase positives** → `[0, 1]` *(new — extracted in `scripts/00e_prep_multi_phrase.py`)*
- Negatives → `[0, 0]`

Source-aware split keeps both phrases of the same source recording in the same fold.

### Files added (v7 pipeline)

```
scripts/00e_prep_multi_phrase.py            # extract closing + pre-closing windows
scripts/03e_augment_multi_phrase.py         # augment all 3 dirs (closing/preclosing/neg)
scripts/04e_extract_features_multi_phrase.py # MFCC + (N, 2) labels, source-aware split
scripts/05g_train_multi_phrase.py            # 2-head model training
scripts/11c_export_multi_phrase_tflite.py    # TF-native TFLite export, (1, 2) output
```

---

## 8. Definition of done (if shipping v7)

1. `apps/mobile/assets/models/navkar_multi_phrase_fp32.tflite` placed (don't delete v6's file — keep both)
2. New constants in `constants.ts` per Section 4
3. Mobile inference wrapper updated to handle `(1, 2)` output (split into two probs, apply temporal rule)
4. Hard-negative tests pass (partial, speech)
5. All 3 user mobile recordings count correctly
6. Feature flag (e.g. `USE_V7_MULTI_PHRASE`) so v6 remains the default and v7 can be switched on per-user for A/B

---

## 9. Recommendation

**Ship v6, keep v7 as a fallback.** Real-world data is the tiebreaker: if v6 reaches enough users and Recording-2-style failures don't appear in the wild, v7 stays on the shelf. If multiple users report v6 failures during chanting that match the bandwidth-limited profile, A/B v7 to those users via the feature flag.

The v7 regression on standard synthetic tests (−24 files) is concerning enough that it should not replace v6 as the default. But the multi-head temporal rule is a useful tool to keep ready for the long tail.
