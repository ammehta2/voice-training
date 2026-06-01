# Navkar KWS Training

Train a keyword-spotting model that detects the Navkar Mantra. Output is a `~50KB .tflite` file ready for the mobile app.

**Architecture:** Depthwise Separable CNN (DS-CNN)
**Input:** 16s audio @ 16kHz → MFCC (1600 frames × 40 mel bins)
**Output:** Single probability score
**Target:** ≥95% precision, ≥95% recall

## Day 0 — Environment Setup (1 hour)

### 1. Verify GPU

```powershell
nvidia-smi
# Should show RTX 5090 with ~32GB VRAM
```

### 2. Create Python virtual environment (Python 3.11)

```powershell
# From the project root
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip
```

### 3. Install dependencies

```powershell
# TensorFlow + CUDA
pip install -r requirements.txt

# PyTorch separately (uses different index)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Verify TF sees the GPU

```powershell
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
# Expected: GPUs: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

## Project Layout

```
voicetraining/
├── data/
│   ├── positive_raw/        # Synthetic Navkar chants (Day 1-2)
│   ├── positive_aug/        # Augmented positives (Day 4-5)
│   ├── negative_raw/        # Non-Navkar sounds (Day 3)
│   ├── negative_aug/        # Augmented negatives (Day 4-5)
│   ├── features/            # MFCC numpy arrays (Day 6)
│   ├── sources/             # Downloaded datasets (Common Voice, ESC-50)
│   └── real_world_test/     # Your own recordings (Day 10)
├── voice_samples/           # Reference voices for TTS (you provide)
├── models/                  # Trained models
├── scripts/                 # 8 numbered scripts — run in order
├── logs/                    # TensorBoard logs, plots
└── venv/                    # Python virtual env (created at Day 0)
```

## Day-by-Day

| Day  | Script                       | Output                          | Wall time   |
|------|------------------------------|---------------------------------|-------------|
| 0    | —                            | venv + packages                 | 1h          |
| 1-2  | `01_generate_navkar.py`      | ~1500 .wav in `data/positive_raw/` | 4-8h GPU |
| 3    | `02_gather_negatives.py`     | ~2500 .wav in `data/negative_raw/` | 4h work  |
| 4-5  | `03_augment.py`              | ~13,500 augmented samples       | 4-8h CPU    |
| 6    | `04_extract_features.py`     | `data/features/X_*.npy`         | 1-2h CPU    |
| 7-8  | `05_train_model.py`          | `models/navkar_v1_*/best.keras` | 2-6h GPU    |
| 9    | `06_evaluate.py <model>`     | confusion matrix, ROC, metrics  | 1h          |
| 10   | `07_real_world_test.py <model>` | per-file scores              | 2-3h        |
| 11   | `08_convert_to_tflite.py <model>` | `navkar_fp16.tflite` (~50KB) | 30min |
| 12   | mobile integration           | model running on phone          | 2h          |

Each script is self-contained. Run from the project root with `venv` activated.

## What YOU Need to Provide

### Voice samples (before Day 1)
Place **10-20 reference voice samples** in `voice_samples/`:
- 5-15 second clips, varied voices (gender, age, accent)
- WAV format, 16kHz mono preferred
- Filename: `voice_01.wav`, `voice_02.wav`, ...

**Options:**
- (A) Record family/community members talking naturally
- (B) Download Mozilla Common Voice (Hindi subset) and pick 20 varied clips
- (C) Quick-start: 5-6 recordings of yourself in different styles (works but limited)

See `voice_samples/README.md` for detailed sourcing instructions.

### Common Voice corpus (before Day 3)
Download Hindi subset from https://commonvoice.mozilla.org/en/datasets

Extract to `data/sources/common_voice/cv-corpus-XX.X-XXXX-XX-XX/hi/clips/` — the script will pull 500 random clips.

### Real-world recordings (before Day 10)
- 10 recordings of you chanting Navkar in different conditions → `data/real_world_test/positive/`
- 10 recordings of non-Navkar sounds → `data/real_world_test/negative/`

## Acceptance Criteria (v1 ship gate)

- ✅ Test set accuracy ≥ 95%
- ✅ Test set precision ≥ 95%
- ✅ Test set recall ≥ 90%
- ✅ Model size ≤ 100KB
- ✅ Mobile inference ≤ 100ms per chunk
- ✅ Real-world detection ≥ 90% in quiet conditions
- ✅ False-positive rate ≤ 1 per 30 min of normal household noise

If you don't hit these — iterate. Most failures are data issues, not model issues. Add more diverse training data.

## Common Failure Modes

| Symptom                           | Fix                                                          |
|-----------------------------------|--------------------------------------------------------------|
| All positives missed              | Threshold too high. Try 0.70                                 |
| Some false positives              | Add 500 more negatives in the failing category               |
| Whispered chanting missed         | Add quiet/whispered training data                            |
| Fast chanting missed              | Lower `time_stretch.min_rate` in augmentation                |
| Train acc 99% / val acc 70%       | Overfitting — add more data variety                          |
| Train acc < 80%                   | Data or architecture issue — review feature extraction       |
| Sim accuracy good / real bad      | Sim-to-real gap — collect real recordings, fine-tune         |
| Mobile inference too slow         | Use INT8 instead of FP16; consider pruning                   |

## Notes

- **TensorBoard:** while training, run `tensorboard --logdir logs/` in another shell
- **Resume-friendly:** `01_generate_navkar.py` skips files that already exist
- **GPU usage:** `nvidia-smi -l 1` in another shell to monitor GPU during training
- **Disk:** budget ~50GB for raw data + features + augmentations
