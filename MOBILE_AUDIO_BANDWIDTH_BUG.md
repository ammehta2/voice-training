# Mobile Audio Capture is Bandwidth-Limited (Diagnosis + Fix)

> **For the mobile-side agent:** Real-world testing of Model v4 / v5 on the tablet showed scores of 0.00–0.22 (well below the 0.50 threshold) even when the user was actively chanting Navkar. After analyzing the captured WAV files, **the root cause is bandwidth-limiting somewhere in the mobile audio capture pipeline.** This document explains the problem and the steps to verify/fix.

---

## The evidence

Three test recordings were captured by the mobile app (saved to disk at the same point the audio would be fed to the TFLite model):

```
navkar_2026-06-01T19-50-40-603Z.wav   14.7s   peak=0.092  RMS=0.013   mobile score=0.00
navkar_2026-06-01T19-52-06-575Z.wav   14.4s   peak=0.091  RMS=0.012   mobile score=0.22
navkar_2026-06-01T19-53-08-218Z.wav    8.1s   peak=0.112  RMS=0.015   mobile score=0.00
```

All three are 16 kHz mono PCM16 — **format is correct.**

### Frequency content analysis (all three recordings)

```
0-200 Hz:    50% of energy   ← dominant rumble band
200-500 Hz:  41%
500-1000 Hz: 6%               ← already a steep drop here
1000-2000 Hz: 1%
2000-4000 Hz: 0.3%
4000-8000 Hz: 0%              ← essentially zero
```

**Normal human speech has significant energy 200–4000 Hz**, especially for consonants:
- "Padhamam" — broadband 'p', mid 'd', high 'h'
- "Havai" — high 'h', mid 'v'
- "Mangalam" — nasal 'm/ng' (mid), bright 'l' (mid-high)

The user's recording has been low-passed to essentially **<500 Hz**. The closing phrase loses most of the consonant information the model needs.

### Controlled experiment (proves bandwidth is the cause, not amplitude)

We took a recording the model normally detects with 99.8% confidence and applied progressive transforms:

| Transform | Model F score |
|---|---|
| Original (peak 0.95, full spectrum) | **0.998 ✅** |
| Reduce amplitude to peak 0.09 (= user's level) | 0.993 ✅ |
| Add Gaussian noise | 0.999 ✅ |
| **Low-pass at 800 Hz** (alone, full amplitude) | **0.349 ❌** |
| **Low-pass 600 Hz + amplitude 0.09 (matches user)** | **0.130 ❌** |

The model is **robust to amplitude variation** (peak normalization is built into the TFLite model itself). It is **NOT robust to losing the 1–4 kHz band**.

Score 0.13 in this synthetic test matches the user's actual mobile readings (0.0–0.22). **Confirmed**: bandwidth limitation, not amplitude, is the failure.

---

## Where is the bandwidth limit coming from?

Most likely culprits, in order of probability:

### 1. Picovoice processing (90% likely)

If the audio capture pipeline routes through any Picovoice product before being saved/fed to the TFLite model, that's almost certainly the cause:

- **Picovoice Koala** (noise suppression) — explicitly low-passes to remove "noisy" high frequencies
- **Picovoice Cobra** (VAD) — typically applies pre-emphasis filters
- **Picovoice Eagle** (speaker recognition) — may do bandwidth limiting
- **Picovoice Porcupine** (wake-word) — accepts only specific sample rates and may downsample

**Fix**: capture raw PCM directly from the mic. Do NOT pass through any Picovoice processing stage before feeding to the TFLite model.

### 2. Android AudioSource.VOICE_RECOGNITION (10% likely if Android)

On Android, `MediaRecorder.AudioSource.VOICE_RECOGNITION` enables an OS-level "speech enhancement" pipeline that aggressively low-passes audio above 4 kHz on many devices. Even if Picovoice isn't involved, this OS setting can cause the issue.

**Fix on Android**: use `MediaRecorder.AudioSource.MIC` (raw mic input) or `AudioSource.UNPROCESSED` (Android 7+).

### 3. iOS AVAudioSession voice processing (less likely on tablet)

Less common but possible. iOS has `setMode(AVAudioSession.Mode.voiceChat)` which enables echo cancellation + noise suppression. Use `Mode.default` for our use case.

### 4. Hardware microphone

Some cheap tablet/laptop microphones have built-in low-pass filtering at 4–8 kHz. **Test**: try a different device.

---

## How to verify the fix

After making the change, capture a test recording and check the spectrum.

### Python snippet to analyze any WAV file's bandwidth

```python
import soundfile as sf
import numpy as np
from scipy.fft import rfft, rfftfreq

audio, sr = sf.read('test.wav')
audio = audio[len(audio)//4:3*len(audio)//4]  # middle half (the chant part)
spectrum = np.abs(rfft(audio))
freqs = rfftfreq(len(audio), 1/sr)
total = np.sum(spectrum**2)

for lo, hi in [(0,200), (200,500), (500,1000), (1000,2000), (2000,4000), (4000,8000)]:
    mask = (freqs >= lo) & (freqs < hi)
    pct = 100 * np.sum(spectrum[mask]**2) / total
    print(f'{lo:>5}-{hi:<5} Hz: {pct:5.1f}%')
```

### Good spectrum (will detect)

```
0-200    Hz:  10-20%
200-500  Hz:  20-25%
500-1000 Hz:  20-25%
1000-2000 Hz: 15-25%
2000-4000 Hz: 5-15%
4000-8000 Hz: 1-5%
```

### Bad spectrum (current user recording)

```
0-200    Hz:  50% ← way too much bass
200-500  Hz:  41%
500-1000 Hz:  6%
1000-2000 Hz: 1%
2000-4000 Hz: 0.3%
4000-8000 Hz: 0%  ← should not be zero
```

### Acceptable spectrum (some loss but probably workable)

```
0-200    Hz:  20-30%
200-500  Hz:  30-40%
500-1000 Hz:  15-20%
1000-2000 Hz: 10-15%
2000-4000 Hz: 3-8%
4000-8000 Hz: 0.5-2%
```

---

## Action items

1. **First** — bypass any Picovoice processing in the audio capture path. Test with raw mic PCM.
2. **Verify** — capture a test WAV and run the spectrum analyzer above. Confirm energy spans the full 200-4000 Hz range.
3. **If still bandwidth-limited after removing Picovoice** — try `AudioSource.MIC` on Android or `Mode.default` on iOS.
4. **If still bandwidth-limited** — try a different physical device (the mic itself may be limiting).

Once the captured audio has normal bandwidth, the model should score **0.5-1.0** on actual chants instead of 0.0-0.22.

---

## Backup plan: training-side fix

While the mobile pipeline is being debugged, we're training **Model G** with aggressive low-pass augmentation so the model can detect chants even when the audio bandwidth is limited. This will be the v6 release if needed.

Model G adds a new "bandwidth_limited" augmentation severity to the training pipeline:
- Always applies low-pass filter with cutoff 400-2000 Hz
- Combined with gain reduction (-25 to +5 dB)
- Some pitch shift and noise

This should make the model robust even if the mobile pipeline can't be fixed to capture full-bandwidth audio. **But** the proper fix is still to remove the bandwidth limiting on the mobile side, because:

1. Other parts of the audio pipeline (false-positive rejection, partial-chant rejection) will degrade if we train on band-limited audio
2. Future models will be more constrained
3. The user experience benefits from being able to hear the actual chanting

We'll have Model G to test against the user's existing recordings within ~30-45 minutes.
