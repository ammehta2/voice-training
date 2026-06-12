# Query-by-Example (QbE) Detector — Prototype Findings

> Context: after v6/v7/v8, the auto-counter shipped **disabled** — the
> from-scratch DS-CNN that looked excellent on files scored too low on the
> live mic path (`VOICE_RECOGNITION` + HAL AGC), and the preclosing head fired
> on ambient noise. Root-cause audit: every training/validation sample came
> from a different capture channel than production, and a 13K-param scratch
> CNN cannot learn channel invariance from ~30 recordings.
>
> New approach under test: **speaker/phrase-dependent matching** — enroll the
> user's own chants on their own device, detect by embedding similarity.
> Feature stack: openWakeWord's frozen pipeline (melspectrogram.tflite →
> Google speech_embedding.tflite, Apache-2.0, both in `models/openwakeword/`).
> No task-specific training anywhere.

## Method

- Embedding: 16 kHz audio → melspec → (/10 + 2) → embedding over 76-frame
  windows, stride 8 → one 96-dim vector per 80 ms.
- Enrollment: 3 of the user's app-captured chants (9:44:36, 9:46:07, 9:46:44).
- Matching: 2.5 s sliding windows (31 embedding steps), mean-pooled, cosine.

## Experiments (scripts 40/41/42)

### 40 — raw cosine vs TTS/FLEURS negatives (app-channel positives)
| | held-out positives (8) | negatives (425) | zero-FP catch |
|---|---|---|---|
| full-chant template | min 0.976 | max 0.971 | **8/8** (margin +0.005) |
| closing-region template | min 0.970 | max 0.965 | **8/8** (margin +0.004) |

All 8 rank above all 425 — including the 13.1 s slow chant and the
near-silent capture that every CNN version missed. Speech-cohort
normalization made things WORSE (conversation is the wrong cohort; it boosts
anything chant-like, e.g. Om Namo Narayanaya).

### 41 — cross-channel sanity check
- Old close-mic recordings (different channel, 14 files): **13/14 pass** the
  TTS-calibrated threshold → matching is phrase/voice-driven, not channel-driven.
  (Even a different voice chanting Navkar passes — phrase-level matching.)
- Real-device ambient recordings (Birch Tree, 8 files): **2/8 false-positive**
  at that threshold → real-channel audio scores systematically higher than TTS
  negatives; thresholds calibrated on synthetic negatives do NOT transfer.

### 42 — dual-cohort normalization (production design candidate)
Per-window score = `cos(user templates) − max(cos(background), cos(imposter chants))`
- background cohort: 3 real-device ambient files (held out of eval)
- imposter cohort: 12 TTS chants (Mangalam Bhagwan, Om Namo Narayanaya,
  truncated Navkar, Chattari Mangalam, Namo Tassa, spoken-about)

| | value |
|---|---|
| positives (22, both channels) | min +0.011, mean +0.028 — all positive |
| negatives (418) | mean −0.023, max +0.015 |
| zero-FP catch | **19/22 (86%)**, margin −0.004 |
| TTS hard negatives after normalization | best one only +0.009 (was the FP class) |

Remaining top negatives are the real-channel ambient files (+0.013…+0.015) —
expected, because the background cohort here is from the *old* channel, not
the device/session being scored. In production the background sample is
recorded during calibration on the SAME device, which should match better.

## Verdict

The QbE approach is **validated to the limit of available data**:
1. Perfect ranking within-channel with no training (AUC 1.0).
2. Phrase-driven (cross-channel positives pass), not channel-driven.
3. Dual-cohort normalization collapses the chant-imposter FP class and yields
   86% zero-FP catch across two channels — with three known upgrades unused:
   per-device background enrollment, k-of-n temporal smoothing (production
   counts on sustained matches + debounce, not single-window max over a file),
   and per-user threshold from enrollment statistics.

## Decisive test — PASSED (script 43, live captures 2026-06-11)

User captured via the in-app diagnostics screen (real VOICE_RECOGNITION
path): two ~60 s continuous-chanting sessions + one 27 s background-voices
session.

### A. The live channel, finally measured
| | rms | peak | rolloff95 |
|---|---|---|---|
| session1 (chant) | 0.0029 | 0.028 | 5625 Hz |
| session2 (chant) | 0.0034 | 0.031 | 5531 Hz |
| background | 0.0025 | 0.037 | 6969 Hz |

- The channel is NOT band-limited (full ~5.5 kHz speech bandwidth) — it is
  just VERY quiet (peak ~0.03). The v6 "bandwidth_limited" hypothesis was
  wrong for this device; it was a gain problem.
- **`RMS_GATE = 0.006` in the live engine is above the real chant RMS
  (~0.003) — the noise gate was silently blocking the model on real chants.**
  Major contributor to the "missing a lot / unusable" end state.
- v8 close head live: fired >=0.5 only ~2x per ~9-chant session (confirms
  "scores too low"). Pre head fires on background voices (max 0.72) —
  confirms why it had to be disabled.

### B. QbE activity separation on the live channel: PERFECT
Templates = every 8th window of session1; competitors = first half of
background + 8 TTS imposter chants; test = session2 vs held-out background.

- session2 chant windows: p10 +0.018, median +0.026
- background windows: max -0.012
- **Any threshold in (-0.012, +0.018) → 100% of chant windows pass, 0/148
  background FPs.** Recommended operating point: +0.005.

### C. Cycle counting on the live channel: WORKS
Closing template = 2.5 s of session1 anchored at its v8-close peak (t=39 s).
Peak-picking on the delta timeline (min 5 s separation):

- session2: **9 peaks** at consistent ~5-7.5 s gaps (user chants ~6.5 s per
  Navkar at jaap pace → ~9 chants per 60 s — matches the requested "~10")
- session1 (self): 9-10 peaks
- background: **0 peaks** at every threshold tested

## Production port plan (mobile)

```
Native module (Kotlin, like the existing rawmic module — react-native-fast-tflite
cannot resize the melspec model's dynamic input):
  melspectrogram.tflite (streamed in fixed chunks, resizeInput once)
  embedding_model.tflite (1,76,32,1) -> 96-dim vector per 80 ms
  ring buffers; per 250 ms hop: window mean over last 31 steps ->
  delta = max cos(templates) - max(cos(background), cos(imposters)) ->
  peak detection with 5 s min separation -> count event

Enrollment (rework calibration screen):
  3 discrete chants + ~20 s background on the user's device
  templates = chant window embeddings; bg cohort = background windows
  imposter embeddings precomputed offline from TTS, shipped as an asset

Threshold: fixed +0.005 initially; later derived per user from enrollment stats.
```

The from-scratch DS-CNN line (v1-v8) is retired as the primary detector.
v8 optionally stays as a closing-locator during enrollment only.

## Runtime addendum — LiteRT framing (script 45, 2026-06-11)

The mobile app ships LiteRT 1.4.0 (forced: react-native-fast-tflite v3 uses
LiteRT, and two TFLite runtimes can't coexist in one APK). On-device crash
"[1,1,75,32] vs [1,1,76,32]" traced to a wrong constant, NOT runtime
divergence. Verified by direct probe under BOTH runtimes (Python
ai-edge-litert 2.1.5 + tf.lite):

- samples -> mel frames is IDENTICAL on both: 12480 -> 75, **12640 -> 76**.
  The original 12480 constant was bad arithmetic, never probed directly.
- The embedding model requires EXACTLY 76 frames. resize_tensor_input to
  [1,75,32,1] **segfaults LiteRT natively** — dynamic frame counts are not
  an option.
- Numeric equivalence: per-step embedding cosine(tf.lite, LiteRT) =
  1.000000 over 116 steps of live capture.
- Separation re-check fully under LiteRT with the shipped imposter JSON:
  chant p10 +0.018 / med +0.026, background max -0.012, 100% chant above
  +0.005, 0/148 background FPs — identical to the validated numbers. No
  asset regeneration needed; LiteRT 1.4.0 alignment is safe.

Mobile fix: MEL_INPUT_SAMPLES = 12640 + a load-time probe that fails loudly
if a future runtime shifts framing (namobuddy commit 2de1d2a).
