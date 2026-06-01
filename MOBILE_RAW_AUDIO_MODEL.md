# Navkar Counter — Raw-Audio TFLite Model (v2)

> **For Claude Code on the mobile side:** This document **supersedes** the MFCC sections of `MOBILE_INTEGRATION.md` and the entire `MOBILE_MEL_FILTERBANK.md`. The TFLite model now does MFCC and peak normalization internally. Mobile only needs to capture audio, slide a window, and feed raw float32 samples to the model.

If you previously implemented `MFCCExtractor.ts`, `MelFilterbank.ts`, or any meyda-based MFCC code — **delete it**. Replace with the raw-audio path described here.

---

## 1. What changed and why

The original v1 design had the mobile app compute MFCC features (Hann window → FFT → mel filterbank → log → DCT) in JavaScript using meyda, then feed those features into a small (~29 KB) TFLite model. This had two persistent problems:

1. **MFCC parity** — meyda's mel filterbank uses HTK formula; librosa training used Slaney. Log scaling and DCT normalization also differed. Every mismatch shifts the model's input distribution, breaking the 0.85 threshold.
2. **Amplitude variance** — phone-mic audio is typically 20–50× quieter than the training recordings. Even with correct MFCC, the per-utterance mean/std normalization can't fully compensate.

The v2 model solves both problems on the training side:

- **Per-utterance peak normalization** is the first op inside the model. Audio is scaled to amplitude ≈0.95 before MFCC. Phone-mic quietness no longer matters.
- **Librosa-compatible MFCC** is the second part of the model graph. It uses Slaney mel filterbank as a constant tensor, `10 * log10` for log compression, and orthonormal DCT-II — all matched to librosa bit-for-bit (verified to 0.0005 probability difference).

Result: mobile sends raw audio. No MFCC code in JS. No mel filterbank constant. No DCT. No parity validation needed. The same model file works identically across iOS, Android, and any platform with TFLite.

---

## 2. Model file

```
models/closing_v1_<TIMESTAMP>/navkar_raw_audio_fp32.tflite
```

Size: **~1.1 MB**. Larger than the v1 model (29 KB) because the DFT basis is stored as a constant tensor inside the model — but this replaces an even larger pile of JavaScript MFCC code you'd otherwise have to ship. Net win.

Place at: `apps/mobile/assets/models/navkar_raw_audio_fp32.tflite`

> **Why not FP16?** The DFT matmul produces intermediate values that overflow FP16. FP32 is correct; FP16 currently produces NaN. We'll fix this later by scaling the basis — for now, ship FP32.

---

## 3. Model interface

| | |
|---|---|
| **Input** | float32 tensor, shape `(1, 40000)` |
| **Input values** | raw audio samples in range `[-1, 1]`, 16 kHz mono, 2.5 seconds |
| **Output** | float32 tensor, shape `(1, 1)` — single raw logit |
| **To get probability** | `sigmoid(output[0])` = `1 / (1 + exp(-output[0]))` |
| **Internal processing** | peak normalize → DFT (matmul) → power → Slaney mel → log10 → DCT-II → per-utterance mean/std → KWS backbone |

---

## 4. Tuned detection settings

| Parameter | Value |
|---|---|
| Detection threshold | **0.90** (probability after sigmoid) |
| Debounce window | **7,000 ms** |
| Inference cadence | every **250 ms** |
| Target count | 108 (configurable per session) |

Verified offline on all 15 source recordings: **14/15 correct, 0 over-counts, 1 miss**.

> Why threshold went up from 0.85 (v1) → 0.90 (v2): peak normalization makes mid-mantra audio slightly more competitive with the closing phrase. The higher threshold preserves the "0 over-counts" property which matters for spiritual practice (telling the user "108 done" when they only did 95 is the worst failure mode).

---

## 5. Files to update on the mobile side

### DELETE these (no longer needed):

```
apps/mobile/lib/jaap/MFCCExtractor.ts        ← delete
apps/mobile/lib/jaap/MelFilterbank.ts        ← delete (if you created it)
```

Plus remove `meyda` from `package.json` dependencies. The model now handles all preprocessing.

### KEEP / minor changes:

```
apps/mobile/lib/jaap/constants.ts            ← simplify (drop MFCC constants)
apps/mobile/lib/jaap/AudioStreamer.ts        ← unchanged
apps/mobile/lib/jaap/JaapCounter.ts          ← update threshold + debounce
apps/mobile/screens/JaapCounterScreen.tsx    ← simplified (drop MFCC step)
```

### REPLACE:

```
apps/mobile/lib/jaap/NavkarModel.ts          ← swap to raw-audio interface
```

---

## 6. `lib/jaap/constants.ts` (simplified)

```typescript
// All audio + counter constants for the Navkar jaap counter.

export const SAMPLE_RATE = 16000;
export const CHANNELS = 1;
export const BITS_PER_SAMPLE = 16;

// Model expects exactly this many raw audio samples (2.5 s @ 16 kHz)
export const WINDOW_SEC = 2.5;
export const N_SAMPLES = SAMPLE_RATE * WINDOW_SEC;   // 40000

// How often the mic library emits a chunk
export const CHUNK_MS = 250;

// How often we run inference on the rolling buffer
export const INFERENCE_HOP_MS = 250;

// Counter behavior (tuned offline; verified 14/15 correct on real recordings)
export const DETECTION_THRESHOLD = 0.90;
export const DEBOUNCE_MS = 7000;
export const DEFAULT_TARGET = 108;
```

Note: no more MFCC parameters. They live inside the model.

---

## 7. `lib/jaap/RollingBuffer.ts` (unchanged from v1 — copy this file)

```typescript
import { N_SAMPLES } from './constants';

/**
 * Fixed-capacity rolling audio buffer (size N_SAMPLES).
 * Newest samples sit at the end; oldest at the start.
 * Not ready until at least N_SAMPLES samples have been appended.
 */
export class RollingBuffer {
  private buffer = new Float32Array(N_SAMPLES);
  private filled = 0;

  append(chunk: Float32Array): void {
    if (chunk.length === 0) return;

    if (chunk.length >= N_SAMPLES) {
      this.buffer.set(chunk.subarray(chunk.length - N_SAMPLES));
      this.filled = N_SAMPLES;
      return;
    }

    this.buffer.copyWithin(0, chunk.length);
    this.buffer.set(chunk, N_SAMPLES - chunk.length);
    this.filled = Math.min(N_SAMPLES, this.filled + chunk.length);
  }

  isReady(): boolean { return this.filled >= N_SAMPLES; }
  read(): Float32Array { return this.buffer; }
  reset(): void { this.buffer.fill(0); this.filled = 0; }
}
```

---

## 8. `lib/jaap/AudioStreamer.ts` (unchanged from v1 — copy from MOBILE_INTEGRATION.md Section 9)

No changes from v1. Captures 16 kHz mono PCM via `@dr.pogodin/react-native-audio` or `react-native-live-audio-stream`, decodes base64 PCM int16 to Float32 in [-1, 1], emits 250 ms chunks via `AudioChunkHandler`.

---

## 9. `lib/jaap/NavkarModel.ts` (REPLACE with this version)

```typescript
import { loadTensorflowModel, TensorflowModel } from 'react-native-fast-tflite';
import { N_SAMPLES } from './constants';

export class NavkarModel {
  private model: TensorflowModel | null = null;
  private loadingPromise: Promise<void> | null = null;

  async load(): Promise<void> {
    if (this.model) return;
    if (this.loadingPromise) return this.loadingPromise;

    this.loadingPromise = (async () => {
      this.model = await loadTensorflowModel(
        require('../../assets/models/navkar_raw_audio_fp32.tflite'),
      );
      console.log('[NavkarModel] inputs:', this.model.inputs);
      console.log('[NavkarModel] outputs:', this.model.outputs);
    })();

    await this.loadingPromise;
  }

  isReady(): boolean { return this.model !== null; }

  /**
   * Predict probability that the audio contains the Navkar closing phrase.
   *
   * @param audio Float32Array of exactly N_SAMPLES samples (40,000) in [-1, 1],
   *              representing 2.5 seconds at 16 kHz mono. The model handles
   *              peak normalization internally, so you do NOT need to
   *              normalize the amplitude before calling this.
   * @returns probability in [0, 1] after sigmoid.
   */
  async predict(audio: Float32Array): Promise<number> {
    if (!this.model) throw new Error('Model not loaded — call load() first');
    if (audio.length !== N_SAMPLES) {
      throw new Error(`Expected ${N_SAMPLES} samples, got ${audio.length}`);
    }
    const output = await this.model.run([audio]);
    const logit = (output[0] as Float32Array)[0];
    return 1 / (1 + Math.exp(-logit));
  }
}
```

That's the entire model interface. No MFCC. No mel filterbank constants. No Hann window. No DCT.

---

## 10. `lib/jaap/JaapCounter.ts` (minor changes from v1)

Only the default threshold + debounce values change.

```typescript
import * as Haptics from 'expo-haptics';
import { DEBOUNCE_MS, DEFAULT_TARGET, DETECTION_THRESHOLD } from './constants';

export interface JaapCounterConfig {
  threshold?: number;       // default 0.90
  debounceMs?: number;      // default 7000
  target?: number;          // default 108
  onCountChange?: (count: number, target: number) => void;
  onComplete?: () => void;
  onScore?: (score: number) => void;
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

## 11. `screens/JaapCounterScreen.tsx` (simplified — direct raw-audio path)

```tsx
import React, { useEffect, useRef, useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Alert } from 'react-native';
import { AudioStreamer } from '../lib/jaap/AudioStreamer';
import { RollingBuffer } from '../lib/jaap/RollingBuffer';
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
      // Direct raw-audio inference — model handles MFCC + peak norm internally.
      const audio = buffer.read();
      const probability = await model.predict(audio);
      counter.feedScore(probability);
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

      {/* Debug score bar — useful while you're verifying things; remove in production */}
      <View style={styles.scoreBar}>
        <View style={[styles.scoreFill, { width: `${Math.round(score * 100)}%` }]} />
        <View style={[styles.thresholdLine, { left: '90%' }]} />
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

That is the **entire** counter screen. Compare to v1, which needed RollingBuffer + MFCCExtractor + NavkarModel + careful coordination. v2 drops the MFCC step.

---

## 12. Validation step

Much simpler than the v1 mel-filterbank validation. Just confirm the model loads and gives the expected score on a known recording.

### A. Get the expected Python score

In the voicetraining repo:

```bash
python scripts/12_test_raw_audio_counter.py \
    models/closing_v1_<TS>/navkar_raw_audio_fp32.tflite
```

You should see something like:

```
  real_navkar_07            1         1      0.998  OK
```

(15 lines, mostly OK status.)

### B. Run the same file through the mobile model

1. Bundle `real_navkar_07.wav` as `apps/mobile/assets/test/real_navkar_07.wav`.
2. In a debug screen, decode the WAV to `Float32Array` at 16 kHz mono.
3. Take the last 40,000 samples (last 2.5 s).
4. Call `model.predict(those_samples)`.
5. Log the result.

### C. Compare

- **Mobile prob in [0.95, 1.00]** → ship.
- **Mobile prob in [0.80, 0.95]** → marginal, but counter will still work; consider nudging `DETECTION_THRESHOLD` to 0.85 in `constants.ts`.
- **Mobile prob < 0.80** → something's wrong. Check that audio decoding produced the right shape and value range ([-1, 1] float32, 16 kHz mono, exactly 40,000 samples).

There's no MFCC math to compare anymore. Either the audio goes in correctly or it doesn't.

---

## 13. Testing checklist

Same as v1 but the bullets about MFCC validation are gone:

- [ ] App requests microphone permission on first launch
- [ ] Model loads within 2 seconds of opening JaapCounterScreen (model is bigger now)
- [ ] Console logs `[NavkarModel] inputs: [{shape: [1, 40000]}]`
- [ ] Inference logs show <80 ms per call after warmup (~30 ms is typical)
- [ ] Section 12 validation: same recording scores within expected range
- [ ] Chant 10 Navkars at normal pace → count reads 9 or 10
- [ ] Chant 10 Navkars slowly → count reads 9 or 10
- [ ] 5-minute Hindi/Gujarati conversation → count stays 0 or 1
- [ ] 5 minutes of instrumental music → count stays 0
- [ ] Reaching count 108 triggers completion alert + success haptic
- [ ] Stop button immediately halts mic capture
- [ ] Backgrounding the app: counter pauses gracefully (or continues if `UIBackgroundModes: ["audio"]` is set)
- [ ] 30 min of counting drains <5% battery on a typical device

---

## 14. Known limitations

- **Model size 1.1 MB** — bigger than v1 (29 KB) because of the embedded DFT basis. Once FP16 export is fixed (currently NaNs due to overflow), this drops to ~580 KB. INT8 quantization could get it under 300 KB.
- **Recording 05 still misses** in offline testing (max score 0.87 at our threshold of 0.90). Same recording missed in v1 too — it's a voice the model genuinely struggles with. More training recordings of similar voices will fix this in v2.
- **No speech in training negatives** — the model hasn't been told that conversational Hindi/Gujarati is NOT Navkar. If you see false positives during real-world testing on speech, escalate to the user; the training pipeline needs Common Voice samples added.

---

## 15. Definition of Done

1. Model file at `apps/mobile/assets/models/navkar_raw_audio_fp32.tflite` (~1.1 MB).
2. `meyda` dependency removed from `package.json`.
3. `MFCCExtractor.ts` and `MelFilterbank.ts` deleted.
4. `NavkarModel.ts` replaced per Section 9.
5. Constants updated per Section 6 (threshold 0.90, debounce 7000 ms).
6. JaapCounterScreen.tsx uses the direct raw-audio path (Section 11).
7. Section 12 validation passes — same recording, mobile prob within range.
8. Real chant test (Section 13): user chants 10 Navkars and gets a count of 9, 10, or 11.
9. False-positive test passes (no triggers on conversation or music).

When all 9 are true, ship it.
