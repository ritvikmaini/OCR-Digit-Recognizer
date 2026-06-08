# MNIST Digit Recognizer — Improvement Design

**Date:** 2026-06-08
**Repo:** https://github.com/ritvikmaini/OCR-Digit-Recognizer
**Challenge:** [Kaggle Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) (MNIST classification)

## Goal

Improve the existing single-CNN solution (~97.7% val accuracy, with bugs) into a
polished, published-quality solution reaching ~99.6–99.7% via a small CNN ensemble
with data augmentation. Target environment: local Mac CPU. Deliverable: a cleaned,
documented notebook + README + requirements.

## Current State & Problems

The existing `Kaggle_Digit_Recognizer.ipynb` has:

1. **Double normalization bug** — `X` is divided by 255, then `X_train`/`X_test`
   are divided by 255 *again*, crushing pixel values into `[0, 0.004]`. This
   handicaps the model.
2. **No-op shuffle** — `train.sample(frac=1)` runs *after* `X`/`y` were already
   extracted, so it has no effect on training.
3. **No data augmentation**, plain CNN (no BatchNorm), single model, 15 epochs.
4. **No data shipped** — notebook reads `data/train.csv`/`data/test.csv` which are
   absent, so the repo cannot run end-to-end out of the box.

## Design

### 1. Bug fixes
- Single, correct `/255.0` normalization applied once.
- Remove the no-op shuffle; use a stratified train/val split.
- Centralize reshape/normalize/one-hot in one preprocessing step.

### 2. Data handling (runnable by anyone)
- If `data/train.csv` and `data/test.csv` exist (real Kaggle data), use them and
  produce a leaderboard-compatible `submission.csv`.
- Otherwise **fall back to `keras.datasets.mnist`** so the notebook runs with zero
  setup (no Kaggle credentials needed).
- README documents how to fetch the real Kaggle data via the Kaggle API.

### 3. Model — compact-but-strong CNN (×N for ensemble)
Per model:
- Block 1: `[Conv(32,3×3)→BN→ReLU] ×2 → Conv(32,5×5,stride2,same)→BN→ReLU → Dropout(0.4)`
- Block 2: `[Conv(64,3×3)→BN→ReLU] ×2 → Conv(64,5×5,stride2,same)→BN→ReLU → Dropout(0.4)`
- Head: `Flatten → Dense(128)→BN→ReLU→Dropout(0.4) → Dense(10, softmax)`

Stride-2 convolutions replace pooling for learnable downsampling. BatchNorm +
Dropout are the main accuracy jump over the current plain CNN. Each model targets
~99.4–99.5%.

### 4. Training
- Optimizer: Adam; loss: categorical crossentropy.
- **Data augmentation** (`ImageDataGenerator`): rotation ±10°, zoom 0.1,
  width/height shift 0.1. No flips (digits aren't symmetric).
- `ReduceLROnPlateau` on val loss; batch size 128.
- Train N models with different random seeds.
- **Ensemble**: average softmax probabilities across models, then argmax.

### 5. Run modes
- `RUN_MODE = "quick"`: 1 model, few epochs — verifies the full pipeline in minutes
  on CPU.
- `RUN_MODE = "full"`: 3-model ensemble, ~25–35 epochs each — the publishable result.
  Documented as ~1–3 hours on Mac CPU.

### 6. Reporting & polish
- Training/validation accuracy & loss curves.
- Validation accuracy of each model and of the ensemble.
- Confusion matrix on the validation set.
- A grid of misclassified examples.
- Generate `submission.csv` from the ensemble.

### 7. Deliverables
- Cleaned, documented `Kaggle_Digit_Recognizer.ipynb`.
- `README.md`: problem, approach, architecture, results table, how to run, how to
  get the Kaggle data.
- `requirements.txt`.

## Expected Results

| Solution                         | Val accuracy |
|----------------------------------|--------------|
| Original (buggy, plain CNN)       | ~97.7%       |
| Single improved CNN (BN + aug)    | ~99.4–99.5%  |
| 3-CNN ensemble                    | ~99.6–99.7%  |

## Out of Scope
- GPU-specific tuning, very large (10–15 model) ensembles.
- Test-time augmentation (could be a future enhancement; noted in README).
- Unrelated refactors beyond what serves this goal.
