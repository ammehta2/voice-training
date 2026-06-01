# Navkar Mantra Jaap Counter — Mobile App Integration Spec

> **For Claude Code:** This document is a complete, self-contained specification for integrating an on-device Navkar Mantra counter into a React Native (Expo) mobile app. The audio ML model has already been trained and exported; your job is to wire it up. Read this whole document before starting, then implement section-by-section. Pay special attention to **Section 6** (parameter matching) and **Section 10** (validation) — those are the two failure modes.

---

## 0. Context

The user is building a Jain spiritual-practice app called Namobuddy. One feature: automatically count how many times the user has recited the **Navkar Mantra** during a jaap session (traditional target: 108 recitations).

A small TFLite keyword-spotting model has been trained that detects the **closing phrase** of the Navkar Mantra — "Padhamam Havai Mangalam" — once per recitation. The mobile app runs the mic continuously, feeds 2.5-second audio windows into the model 4 times per second, and counts each high-confidence detection (with a debounce window to prevent double-counting a single closing phrase).

**Why closing phrase, not whole mantra:** The mantra is ~12–15 seconds long. A whole-mantra detector can only say "is there a chant somewhere in this window" — not "did a chant just complete," which is what we need for counting. The closing phrase ("...Mangalam") is acoustically distinctive (unique consonants `m`, `ng`, `l`) and occurs exactly once per recitation by structure.

**Why this design works:** Offline testing on 15 real recordings achieves **14/15 correct counts (0 over-counts, 1 miss)**. Over-counts are catastrophic for spiritual practice (telling the user "108 done" when they actually did 95). Under-counts are graceful (user just keeps chanting). Settings tuned to threshold=0.85, debounce=5000ms.

---

## 1. Architecture

```
┌──────────────────────────────────────────────────────┐
│  User taps "Start Jaap"                              │
└─────────────────────┬────────────────────────────────┘
                      ▼
┌──────────────────────────────────────────────────────┐
│  Phone microphone → 16 kHz mono PCM stream           │
│  (continuous, emitted in ~250 ms chunks)             │
└─────────────────────┬────────────────────────────────┘
                      ▼
┌──────────────────────────────────────────────────────┐
│  2.5-second rolling audio buffer                     │
│  (FIFO: oldest 250 ms drops, newest 250 ms appends)  │
└─────────────────────┬────────────────────────────────┘
                      ▼
┌──────────────────────────────────────────────────────┐
│  Every 250 ms: compute MFCC features                 │
│  Output shape: (251 frames, 40 coefficients) float32 │
└─────────────────────┬────────────────────────────────┘
                      ▼
┌──────────────────────────────────────────────────────┐
│  TFLite inference (navkar_float16.tflite, 29 KB)     │
│  Input: (1, 251, 40, 1) → Output: 1 logit            │
│  score = sigmoid(logit)  ∈ [0.0, 1.0]                │
└─────────────────────┬────────────────────────────────┘
                      ▼
┌──────────────────────────────────────────────────────┐
│  if score > 0.85 AND now - lastDetection > 5000 ms:  │
│    count += 1                                        │
│    haptic + tick sound                               │
│    update UI                                         │
│  if count == 108: completion celebration             │
└──────────────────────────────────────────────────────┘
```

---

## 2. Fixed Parameters (DO NOT CHANGE)

These match the training pipeline exactly. If you deviate, the model won't work.

| Parameter | Value | Used by |
|---|---|---|
| Sample rate | **16,000 Hz** | Mic, MFCC |
| Channels | **1 (mono)** | Mic |
| Bit depth | **16-bit signed PCM** | Mic capture |
| Audio buffer length | **2.5 s = 40,000 samples** | Rolling buffer |
| MFCC FFT size | **512** | MFCC |
| MFCC hop length | **160 samples (10 ms)** | MFCC |
| MFCC coefficients (N_MFCC) | **40** | MFCC, model input |
| Mel filter banks (N_MELS) | **40** | MFCC |
| MFCC time frames | **251** | model input shape |
| Model input shape | **(1, 251, 40, 1) float32** | TFLite |
| Model output shape | **(1,) float32 raw logit** | TFLite |
| Inference cadence | **every 250 ms** | Counter loop |
| Detection threshold | **0.85** (probability after sigmoid) | Counter |
| Debounce window | **5,000 ms** | Counter |
| Default target | **108** (configurable per session) | Counter |

---

## 3. Model File

The trained model file is:

```
<this-spec-was-shipped-from>/models/closing_v1_<timestamp>/navkar_float16.tflite
```

If you don't have it, ask the user. File size: ~29 KB. Format: TFLite FP16.

**Place it at:** `apps/mobile/assets/models/navkar_float16.tflite`

If `apps/mobile/assets/models/` doesn't exist, create it.

---

## 4. Dependencies to Install

Working directory: `apps/mobile/` (Expo project, dev client required — not Expo Go).

```bash
# TFLite inference
npx expo install react-native-fast-tflite

# Live audio streaming (raw PCM, 16 kHz mono)
npx expo install @dr.pogodin/react-native-audio
# If install fails or it's unmaintained on your RN version, fall back to:
# npx expo install react-native-live-audio-stream

# MFCC computation in JavaScript
npm install meyda

# Haptic feedback when count increments
npx expo install expo-haptics

# Optional: short tick sound on each count (recommended for UX)
npx expo install expo-av

# Base64 decoding helper (audio chunks arrive base64-encoded)
npm install js-base64
```

**After installing**, run `npx expo prebuild --clean` to regenerate native projects. Then build with EAS or locally — Expo Go won't load native modules.

---

## 5. Permissions

### iOS — `apps/mobile/app.json`

```json
{
  "expo": {
    "ios": {
      "infoPlist": {
        "NSMicrophoneUsageDescription": "Namobuddy uses the microphone to count your Navkar Mantra recitations automatically.",
        "UIBackgroundModes": ["audio"]
      }
    }
  }
}
```

The `UIBackgroundModes: ["audio"]` lets the counter keep running when the user locks the screen during a long jaap session. Skip it if you only want foreground-only counting.

### Android — `apps/mobile/app.json`

```json
{
  "expo": {
    "android": {
      "permissions": [
        "android.permission.RECORD_AUDIO",
        "android.permission.FOREGROUND_SERVICE",
        "android.permission.FOREGROUND_SERVICE_MICROPHONE"
      ]
    }
  }
}
```

### Bundle the model

```json
{
  "expo": {
    "assetBundlePatterns": [
      "assets/**/*",
      "assets/models/*.tflite"
    ]
  }
}
```

Re-run `npx expo prebuild --clean` after editing `app.json`.

---

## 6. ⚠️ CRITICAL RISK: MFCC parameter matching

The model was trained on MFCC features extracted with **librosa** in Python. The mobile app will compute MFCC with **meyda.js**. These libraries use slightly different conventions (mel filter bank formulas, DCT normalization, pre-emphasis defaults). If the JS MFCC values diverge from librosa's, the threshold of 0.85 won't transfer — the counter will either never trigger or trigger constantly.

**Mitigation strategy:**
1. Implement Meyda with the exact parameters in Section 2.
2. **Run Section 10's validation step before declaring the mobile counter "done."** If it fails, the proper fix is to bake the MFCC computation into the TFLite model itself (Section 13.A). Do not ship without validation.

**Meyda configuration that approximately matches librosa:**

```typescript
import Meyda from 'meyda';

Meyda.bufferSize = 512;                     // n_fft
Meyda.sampleRate = 16000;
Meyda.numberOfMFCCCoefficients = 40;        // n_mfcc
Meyda.melBands = 40;                        // n_mels
Meyda.windowingFunction = 'hann';
// Meyda does NOT have explicit hop_length — you slide the window manually.
```

After extracting MFCC frames, **apply per-utterance normalization**: subtract the mean of all values in the matrix and divide by std + 1e-8. This matches `mfcc = (mfcc - mean(mfcc)) / (std(mfcc) + 1e-8)` in the training code.

---

## 7. Files to Implement

Create exactly these files. Implementations follow in Sections 8–11.

```
apps/mobile/
├── app.json                                 ← updated in Section 5
├── assets/
│   └── models/
│       └── navkar_float16.tflite            ← copied in Section 3
├── lib/
│   └── jaap/
│       ├── constants.ts                     ← Section 8
│       ├── AudioStreamer.ts                 ← Section 9
│       ├── MFCCExtractor.ts                 ← Section 10
│       ├── NavkarModel.ts                   ← Section 11
│       └── JaapCounter.ts                   ← Section 12
└── screens/
    └── JaapCounterScreen.tsx                ← Section 13
```

Then wire `JaapCounterScreen` into the app's existing navigation stack (project-specific — find the nav config and add the route).

---

## 8. `lib/jaap/constants.ts`

```typescript
// All hard-coded numbers from the training pipeline live here.
// CHANGING THESE WILL BREAK THE MODEL.

export const SAMPLE_RATE = 16000;
export const CHANNELS = 1;
export const BITS_PER_SAMPLE = 16;

export const WINDOW_SEC = 2.5;
export const N_SAMPLES = SAMPLE_RATE * WINDOW_SEC;   // 40000

export const N_MFCC = 40;
export const N_FFT = 512;
export const HOP_LENGTH = 160;                        // 10 ms
export const N_MELS = 40;
export const N_FRAMES = 251;                          // matches librosa output for 40000 samples

export const CHUNK_MS = 250;                          // audio emit cadence
export const INFERENCE_HOP_MS = 250;                  // run model every 250 ms

export const DETECTION_THRESHOLD = 0.85;
export const DEBOUNCE_MS = 5000;
export const DEFAULT_TARGET = 108;
```

---

## 9. `lib/jaap/AudioStreamer.ts`

Captures the mic and emits 250 ms chunks of `Float32Array` (values in [-1, 1]).

```typescript
import { PermissionsAndroid, Platform } from 'react-native';
import LiveAudioStream from '@dr.pogodin/react-native-audio';
import { Base64 } from 'js-base64';
import {
  SAMPLE_RATE, CHANNELS, BITS_PER_SAMPLE, CHUNK_MS,
} from './constants';

export type AudioChunkHandler = (samples: Float32Array) => void;

const BUFFER_SIZE_BYTES =
  (SAMPLE_RATE * CHANNELS * (BITS_PER_SAMPLE / 8)) * (CHUNK_MS / 1000);

export class AudioStreamer {
  private handler: AudioChunkHandler | null = null;
  private listener: any = null;
  private running = false;

  async requestPermission(): Promise<boolean> {
    if (Platform.OS === 'android') {
      const granted = await PermissionsAndroid.request(
        PermissionsAndroid.PERMISSIONS.RECORD_AUDIO,
        {
          title: 'Microphone permission',
          message: 'Namobuddy needs the microphone to count Navkars.',
          buttonPositive: 'Allow',
        },
      );
      return granted === PermissionsAndroid.RESULTS.GRANTED;
    }
    return true; // iOS prompts automatically on first use
  }

  async start(handler: AudioChunkHandler): Promise<void> {
    if (this.running) return;
    const granted = await this.requestPermission();
    if (!granted) throw new Error('Microphone permission denied');

    this.handler = handler;

    LiveAudioStream.init({
      sampleRate: SAMPLE_RATE,
      channels: CHANNELS,
      bitsPerSample: BITS_PER_SAMPLE,
      audioSource: 6,             // VOICE_RECOGNITION on Android
      bufferSize: BUFFER_SIZE_BYTES,
    });

    this.listener = LiveAudioStream.on('data', (b64: string) => {
      this.handler?.(decodeInt16PcmToFloat32(b64));
    });

    LiveAudioStream.start();
    this.running = true;
  }

  stop(): void {
    if (!this.running) return;
    try { LiveAudioStream.stop(); } catch { /* ignore */ }
    this.listener?.remove();
    this.listener = null;
    this.handler = null;
    this.running = false;
  }

  isRunning(): boolean { return this.running; }
}

function decodeInt16PcmToFloat32(b64: string): Float32Array {
  const bytes = Base64.toUint8Array(b64);
  const int16 = new Int16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) {
    float32[i] = int16[i] / 32768;
  }
  return float32;
}
```

---

## 10. `lib/jaap/MFCCExtractor.ts` + rolling buffer

Maintains the 2.5 s buffer and computes MFCC features that match the training pipeline.

```typescript
import Meyda from 'meyda';
import {
  N_SAMPLES, N_MFCC, N_FFT, HOP_LENGTH, N_MELS, N_FRAMES, SAMPLE_RATE,
} from './constants';

// Configure Meyda once
Meyda.bufferSize = N_FFT;
Meyda.sampleRate = SAMPLE_RATE;
Meyda.numberOfMFCCCoefficients = N_MFCC;
Meyda.melBands = N_MELS;
Meyda.windowingFunction = 'hann';

export class MFCCExtractor {
  /**
   * Compute the (N_FRAMES * N_MFCC) MFCC matrix for a fixed-length audio buffer.
   * Input: Float32Array of length N_SAMPLES (40000).
   * Output: Float32Array of length N_FRAMES * N_MFCC laid out row-major
   *         as [frame][coef] — exactly what the TFLite model expects after
   *         reshape to (1, N_FRAMES, N_MFCC, 1).
   * Throws if `audio.length !== N_SAMPLES`.
   */
  extract(audio: Float32Array): Float32Array {
    if (audio.length !== N_SAMPLES) {
      throw new Error(`MFCCExtractor expected ${N_SAMPLES} samples, got ${audio.length}`);
    }

    const out = new Float32Array(N_FRAMES * N_MFCC);

    // Slide an N_FFT window across `audio` with HOP_LENGTH stride.
    // Frames whose window would exceed the buffer are left as zeros.
    for (let frame = 0; frame < N_FRAMES; frame++) {
      const start = frame * HOP_LENGTH;
      const end = start + N_FFT;
      if (end > audio.length) break;

      const slice = audio.subarray(start, end);
      const features = Meyda.extract(['mfcc'], slice) as { mfcc?: number[] } | null;
      if (!features?.mfcc) continue;

      const base = frame * N_MFCC;
      for (let c = 0; c < N_MFCC; c++) {
        out[base + c] = features.mfcc[c];
      }
    }

    // Per-utterance mean/std normalization (matches Python training)
    let sum = 0;
    for (let i = 0; i < out.length; i++) sum += out[i];
    const mean = sum / out.length;

    let sqsum = 0;
    for (let i = 0; i < out.length; i++) {
      const d = out[i] - mean;
      sqsum += d * d;
    }
    const std = Math.sqrt(sqsum / out.length) + 1e-8;

    for (let i = 0; i < out.length; i++) {
      out[i] = (out[i] - mean) / std;
    }

    return out;
  }
}

/**
 * Fixed-capacity rolling audio buffer (size N_SAMPLES).
 * Newest samples sit at the end; oldest at the start.
 * Not ready until at least N_SAMPLES of audio have been appended.
 */
export class RollingBuffer {
  private buffer = new Float32Array(N_SAMPLES);
  private filled = 0;

  append(chunk: Float32Array): void {
    if (chunk.length === 0) return;

    if (chunk.length >= N_SAMPLES) {
      // Bigger chunk than buffer — keep only the last N_SAMPLES samples
      this.buffer.set(chunk.subarray(chunk.length - N_SAMPLES));
      this.filled = N_SAMPLES;
      return;
    }

    // Shift existing samples left, then place new samples at the end
    this.buffer.copyWithin(0, chunk.length);
    this.buffer.set(chunk, N_SAMPLES - chunk.length);
    this.filled = Math.min(N_SAMPLES, this.filled + chunk.length);
  }

  isReady(): boolean { return this.filled >= N_SAMPLES; }

  /** Returns the underlying buffer. Treat as read-only. */
  read(): Float32Array { return this.buffer; }

  reset(): void {
    this.buffer.fill(0);
    this.filled = 0;
  }
}
```

---

## 11. `lib/jaap/NavkarModel.ts`

Thin wrapper around `react-native-fast-tflite`.

```typescript
import { loadTensorflowModel, TensorflowModel } from 'react-native-fast-tflite';
import { N_FRAMES, N_MFCC } from './constants';

export class NavkarModel {
  private model: TensorflowModel | null = null;
  private loadingPromise: Promise<void> | null = null;

  async load(): Promise<void> {
    if (this.model) return;
    if (this.loadingPromise) return this.loadingPromise;

    this.loadingPromise = (async () => {
      this.model = await loadTensorflowModel(
        require('../../assets/models/navkar_float16.tflite'),
      );
      // Useful sanity log on first load — remove or downgrade in production
      console.log('[NavkarModel] inputs:', this.model.inputs);
      console.log('[NavkarModel] outputs:', this.model.outputs);
    })();

    await this.loadingPromise;
  }

  isReady(): boolean { return this.model !== null; }

  /**
   * Run inference on a flat MFCC vector (row-major [frame][coef]).
   * Returns sigmoid probability in [0, 1].
   */
  async predict(mfcc: Float32Array): Promise<number> {
    if (!this.model) throw new Error('Model not loaded — call load() first');
    if (mfcc.length !== N_FRAMES * N_MFCC) {
      throw new Error(`MFCC length ${mfcc.length}, expected ${N_FRAMES * N_MFCC}`);
    }
    const output = await this.model.run([mfcc]);
    const logit = (output[0] as Float32Array)[0];
    return 1 / (1 + Math.exp(-logit));
  }
}
```

---

## 12. `lib/jaap/JaapCounter.ts`

The counting state machine. Pure logic — no UI, no model — easy to unit-test.

```typescript
import * as Haptics from 'expo-haptics';
import { DEBOUNCE_MS, DEFAULT_TARGET, DETECTION_THRESHOLD } from './constants';

export interface JaapCounterConfig {
  threshold?: number;       // default 0.85
  debounceMs?: number;      // default 5000
  target?: number;          // default 108
  onCountChange?: (count: number, target: number) => void;
  onComplete?: () => void;
  onScore?: (score: number) => void;  // for debug score-bar UI
}

export class JaapCounter {
  private threshold: number;
  private debounceMs: number;
  private target: number;
  private onCountChange?: (count: number, target: number) => void;
  private onComplete?: () => void;
  private onScore?: (score: number) => void;

  private count = 0;
  private lastDetectionMs = 0;
  private completed = false;

  constructor(cfg: JaapCounterConfig = {}) {
    this.threshold = cfg.threshold ?? DETECTION_THRESHOLD;
    this.debounceMs = cfg.debounceMs ?? DEBOUNCE_MS;
    this.target = cfg.target ?? DEFAULT_TARGET;
    this.onCountChange = cfg.onCountChange;
    this.onComplete = cfg.onComplete;
    this.onScore = cfg.onScore;
  }

  /** Call on every model inference. */
  feedScore(score: number): void {
    this.onScore?.(score);
    if (this.completed) return;
    if (score < this.threshold) return;

    const now = Date.now();
    if (now - this.lastDetectionMs < this.debounceMs) return;

    this.count += 1;
    this.lastDetectionMs = now;
    this.onCountChange?.(this.count, this.target);

    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium).catch(() => {});

    if (this.count >= this.target) {
      this.completed = true;
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success).catch(() => {});
      this.onComplete?.();
    }
  }

  reset(): void {
    this.count = 0;
    this.lastDetectionMs = 0;
    this.completed = false;
  }

  getCount(): number { return this.count; }
  getTarget(): number { return this.target; }
  isComplete(): boolean { return this.completed; }
}
```

---

## 13. `screens/JaapCounterScreen.tsx`

Wires everything together. Adjust styles to match the rest of the app's design system.

```tsx
import React, { useEffect, useRef, useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Alert } from 'react-native';
import { AudioStreamer } from '../lib/jaap/AudioStreamer';
import { MFCCExtractor, RollingBuffer } from '../lib/jaap/MFCCExtractor';
import { NavkarModel } from '../lib/jaap/NavkarModel';
import { JaapCounter } from '../lib/jaap/JaapCounter';
import { INFERENCE_HOP_MS, DEFAULT_TARGET } from '../lib/jaap/constants';

export function JaapCounterScreen() {
  const [count, setCount] = useState(0);
  const [score, setScore] = useState(0);
  const [running, setRunning] = useState(false);
  const [modelReady, setModelReady] = useState(false);

  const streamer = useRef(new AudioStreamer()).current;
  const buffer = useRef(new RollingBuffer()).current;
  const mfcc = useRef(new MFCCExtractor()).current;
  const model = useRef(new NavkarModel()).current;
  const counter = useRef(
    new JaapCounter({
      target: DEFAULT_TARGET,
      onCountChange: (c) => setCount(c),
      onScore: (s) => setScore(s),
      onComplete: () =>
        Alert.alert('Jaap Complete', 'You have completed 108 Navkars 🙏'),
    }),
  ).current;

  const lastInferMs = useRef(0);
  const inferring = useRef(false);

  useEffect(() => {
    model.load().then(() => setModelReady(true)).catch((e) => {
      console.error('Model load failed', e);
      Alert.alert('Model error', 'Could not load Navkar model.');
    });
  }, []);

  // Stop the mic if the screen is unmounted
  useEffect(() => () => streamer.stop(), []);

  const handleChunk = async (chunk: Float32Array) => {
    buffer.append(chunk);
    if (!buffer.isReady() || !modelReady) return;

    const now = Date.now();
    if (now - lastInferMs.current < INFERENCE_HOP_MS) return;
    if (inferring.current) return;
    lastInferMs.current = now;
    inferring.current = true;

    try {
      const features = mfcc.extract(buffer.read());
      const s = await model.predict(features);
      counter.feedScore(s);
    } catch (e) {
      console.error('Inference error', e);
    } finally {
      inferring.current = false;
    }
  };

  const start = async () => {
    if (running || !modelReady) return;
    counter.reset();
    setCount(0);
    buffer.reset();
    try {
      await streamer.start(handleChunk);
      setRunning(true);
    } catch (e: any) {
      Alert.alert('Cannot start', e?.message ?? 'Unknown error');
    }
  };

  const stop = () => {
    streamer.stop();
    setRunning(false);
  };

  return (
    <View style={styles.container}>
      <Text style={styles.count}>{count}</Text>
      <Text style={styles.target}>/ {counter.getTarget()}</Text>

      {/* Debug score bar — remove or hide behind a flag in production */}
      <View style={styles.scoreBar}>
        <View style={[styles.scoreFill, { width: `${Math.round(score * 100)}%` }]} />
        <View style={[styles.thresholdLine, { left: '85%' }]} />
      </View>
      <Text style={styles.scoreText}>score: {score.toFixed(2)}</Text>

      <TouchableOpacity
        style={[styles.button, !modelReady && styles.buttonDisabled]}
        onPress={running ? stop : start}
        disabled={!modelReady}
      >
        <Text style={styles.buttonText}>
          {!modelReady ? 'Loading…' : running ? 'Stop' : 'Start Jaap'}
        </Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 24 },
  count: { fontSize: 96, fontWeight: '700' },
  target: { fontSize: 24, opacity: 0.6, marginBottom: 24 },
  scoreBar: {
    width: '100%', height: 12, backgroundColor: '#eee', borderRadius: 6,
    marginVertical: 16, position: 'relative', overflow: 'hidden',
  },
  scoreFill: { height: '100%', backgroundColor: '#4caf50' },
  thresholdLine: { position: 'absolute', top: 0, bottom: 0, width: 2, backgroundColor: 'red' },
  scoreText: { fontSize: 14, opacity: 0.7, marginBottom: 24 },
  button: {
    paddingHorizontal: 32, paddingVertical: 16, backgroundColor: '#1976d2',
    borderRadius: 8, marginTop: 16,
  },
  buttonDisabled: { backgroundColor: '#999' },
  buttonText: { color: '#fff', fontSize: 18, fontWeight: '600' },
});
```

---

## 14. Validation Step (MUST DO BEFORE CALLING IT DONE)

The single biggest risk is that JavaScript MFCC values don't match Python librosa MFCC values. If they diverge, the threshold of 0.85 means nothing. Validate before shipping:

### A. Bundle a known test recording

Copy one short Navkar recording into `apps/mobile/assets/test/test_navkar.wav` — ideally one that the Python pipeline scores at ~0.99 confidence.

### B. Get the expected score from Python

The user can run, in the voicetraining repo:

```bash
python scripts/09_count_jaap.py models/closing_v1_<TS>/best.pt data/positive_raw_originals/real_navkar_07.wav
```

…and tell you the max score that appeared (should be ≥ 0.99 for recording 07).

### C. Run the same file through the mobile pipeline

Add a temporary debug button to `JaapCounterScreen` that:
1. Loads the bundled WAV via `Asset.fromModule(require('../assets/test/test_navkar.wav')).downloadAsync()` and decodes it to a `Float32Array` at 16 kHz mono. (Use `expo-asset` + a small WAV decoder, or any Node-style WAV parser bundled for RN.)
2. Slides a 2.5-s window across it every 250 ms.
3. Runs each window through `MFCCExtractor.extract()` + `NavkarModel.predict()`.
4. Logs the max score.

### D. Compare

- **Max score within 0.05 of the Python value** → MFCC matches well enough. Proceed.
- **Max score consistently 0.10+ lower than Python** → MFCC values are off. Try toggling `Meyda.preEmphasis = 0` or `0.97`. Try logging a single frame's 40 MFCC coefficients alongside Python's and see if values are linearly related (different scale = fixable; sign-flipped or wildly different = not fixable). If you can't get within 0.10, **escalate to the user** — the proper fix is to retrain with MFCC baked into the model (Section 16.A).

Do not skip this. The whole counter relies on the threshold staying meaningful.

---

## 15. Testing Checklist

Before declaring v1 done, test on at least one physical iPhone and one physical Android device:

- [ ] First app launch prompts for microphone permission
- [ ] Model loads within 1 second of opening JaapCounterScreen
- [ ] Console shows `[NavkarModel] inputs: [{shape: [1, 251, 40, 1]}]` (or equivalent)
- [ ] Inference logs show <50 ms per call after warmup
- [ ] **Section 14 validation passed**
- [ ] Real chant: chant 10 Navkars slowly (~15 s each) → count reads 9, 10, or 11
- [ ] Real chant: chant 10 Navkars quickly (~8 s each) → count reads 9, 10, or 11
- [ ] False-positive: have a 5-minute Hindi/Gujarati conversation in the same room → count stays 0–1
- [ ] False-positive: play instrumental music for 5 minutes → count stays 0
- [ ] False-positive: play recorded Logassa or Chattari Mangalam chant → count stays 0
- [ ] Reaching count of 108 triggers the completion alert + success haptic
- [ ] Stop button immediately halts mic capture (mic indicator disappears)
- [ ] Backgrounding the app: counter continues if `UIBackgroundModes` is set; otherwise pauses gracefully
- [ ] 30 minutes of counting drains <5% battery on a typical device
- [ ] No memory leaks: counter screen can be opened/closed 20 times without crash

If real-chant accuracy is off by more than 1 in 10, do not ship — see Section 16.

---

## 16. Known Issues & v2 Improvements

### A. Bake MFCC into the TFLite model (HIGHEST PRIORITY for v2)

The cleanest fix to the MFCC-matching risk is to put the MFCC computation inside the model. The mobile app then sends raw audio (`Float32Array` of length 40000) directly to TFLite. No `meyda` in the JS path at all.

To do this, the training repo needs to:
1. Wrap `ClosingDetector` in a module that runs `torchaudio.transforms.MFCC` on raw audio before the conv layers.
2. Re-export to ONNX → TFLite.
3. Mobile updates `NavkarModel.predict()` to accept `Float32Array[40000]` instead of MFCC features. Remove `MFCCExtractor` entirely.

Talk to the user about this if the Section 14 validation gives borderline results.

### B. INT8 quantization

The current FP16 model is 29 KB. Properly calibrated INT8 could drop it to ~10–12 KB and run 2–3× faster on mobile CPUs. The training repo's `08d_convert_closing_to_tflite.py` already attempts INT8 but the calibration data spec needs fixing (`-qcind` flag + per-input npy path).

### C. Score smoothing

If real-world testing reveals occasional double-counts (score dips and rises during a single closing phrase), apply an exponential moving average to the score before thresholding:

```typescript
private smoothed = 0;
private alpha = 0.6;

feedScore(rawScore: number) {
  this.smoothed = this.alpha * rawScore + (1 - this.alpha) * this.smoothed;
  // ...threshold against this.smoothed instead of rawScore
}
```

### D. Background mode

For long meditation sessions where the user locks the screen, ensure the audio session category is set to allow background recording:

- iOS: `UIBackgroundModes: ["audio"]` in Info.plist (done in Section 5), plus configure audio session on start.
- Android: `FOREGROUND_SERVICE_MICROPHONE` (done in Section 5) — start a foreground service while counting. Use `expo-task-manager` or a custom native module.

### E. Known model limitations

- The model was trained on 15 recordings of one core voice family. It may miss certain voice timbres entirely. The user is collecting more recordings — a v2 model with 30+ source voices is planned.
- The model has not been tested against Hindi/Gujarati conversational speech as a negative class. There may be false positives on certain syllable patterns. If Section 15's false-positive test reveals issues, escalate.

---

## 17. Definition of Done

This task is complete when:

1. All files in Section 7 exist and pass TypeScript / lint.
2. `npx expo prebuild --clean && npx expo run:ios` (and `run:android`) builds and runs without native errors.
3. Section 14 validation has been run and the mobile max score is within 0.05 of the Python max score on the same test recording.
4. Section 15 checklist items all pass on at least one physical device.
5. The user can chant 108 Navkars in a real environment and the count reaches 108 ± 2.

If item 3 fails, document the discrepancy in the project and ask the user before proceeding — likely needs the v2 baked-MFCC approach.

If item 5 fails (count is off by >3 in 108), do not ship. Tune `DETECTION_THRESHOLD` and `DEBOUNCE_MS` in `constants.ts`, retest, and if no setting works, escalate.

---

## 18. Source of Truth

Training pipeline & model artifacts: `<voicetraining-repo>` (ask the user for path if needed). Specifically:

- Model: `models/closing_v1_<timestamp>/navkar_float16.tflite`
- Reference Python inference: `scripts/09_count_jaap.py`
- MFCC parameters: `scripts/04c_extract_features_opening.py` (constants near top)
- Architecture: `scripts/05c_train_opening.py` (`OpeningDetector` class; the closing model uses the same architecture renamed `ClosingDetector` in `05d_train_closing.py`)

When in doubt about a parameter, check the Python source.
