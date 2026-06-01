# Voice Samples for TTS Reference

Place **10-20 reference voice samples** in this directory. They're used by `scripts/01_generate_navkar.py` to clone voices and synthesize varied Navkar Mantra recordings.

## Required Format

- **Format:** WAV (mp3 also works but slower)
- **Sample rate:** 16kHz (will be resampled if not)
- **Channels:** Mono
- **Duration:** 5-15 seconds each
- **Content:** Natural speech (not chanting)
- **Filename:** `voice_01.wav`, `voice_02.wav`, ..., `voice_20.wav`

## Why 10-20 voices

The synthetic data inherits the voice diversity of these samples. Fewer voices = the model overfits to a narrow voice space and fails on real users.

**Aim for variety:**
- Male + female
- Young + middle-aged + elderly
- Different accents (Gujarati, Hindi, Marathi, etc.)
- Different speaking styles (calm, energetic)

## Sourcing Options

### Option A: Mozilla Common Voice (recommended)

Free, large, varied Indian-language voices.

1. Go to https://commonvoice.mozilla.org/en/datasets
2. Download the **Hindi** subset (also grab Gujarati if available)
3. Extract — you'll get a `.tsv` index and `.mp3` clips
4. Pick 20 varied clips (different `client_id` values = different speakers)
5. Convert to WAV 16kHz mono:

```powershell
# Example with ffmpeg
ffmpeg -i source.mp3 -ar 16000 -ac 1 voice_01.wav
```

### Option B: Family/Community Recordings

1. Ask 10+ family/community members for a 10-second voice clip
2. They can read anything natural — a recipe, a news headline, etc.
3. Record on phone (Voice Memos / Google Recorder)
4. Convert to WAV 16kHz mono using ffmpeg (see above)

### Option C: Quick-Start (lower quality)

If you just want to test the pipeline first:
1. Record yourself in 5-6 styles: normal, fast, slow, soft, loud, formal
2. Save as `voice_01.wav` through `voice_06.wav`
3. Limited diversity — the model won't generalize well, but pipeline runs

## After Adding Samples

Verify with:

```powershell
# Should show your wav files
ls voice_samples/*.wav

# Quick sanity check on one file
python -c "import librosa; y, sr = librosa.load('voice_samples/voice_01.wav', sr=None); print(f'sr={sr}, duration={len(y)/sr:.1f}s, mono={y.ndim==1}')"
```

Then proceed to Day 1: `python scripts/01_generate_navkar.py`
