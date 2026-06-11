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

## Decisive test (pending)

Phase 0 live captures (namobuddy mobile commit `54f742c`: Counter Settings →
Diagnostics → "Record what the counter hears") through the REAL
`VOICE_RECOGNITION` path: 2-3 chanting sessions + 1 background/talking
session. Re-run script 42 with enrollment + background from those captures.
If separation holds → port to mobile:

```
melspectrogram.tflite + embedding_model.tflite (run via react-native-fast-tflite)
enrollment = calibration screen (3 chants + 20 s background)
score = dual-cohort cosine, threshold from enrollment stats
counter = k-of-n window matches + 5 s debounce
```

v8 stays as an optional corroborating signal; the from-scratch DS-CNN line
(v1-v8) is retired as the primary detector.
