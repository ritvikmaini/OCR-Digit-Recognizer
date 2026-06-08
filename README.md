# OCR Digit Recognizer — Kaggle MNIST

A convolutional neural network ensemble for the Kaggle [Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) competition, classifying 28×28 grayscale images of handwritten digits (0–9).

## Approach

This solution improves on a prior single plain-CNN baseline that reached roughly **97.7%** validation accuracy but contained two bugs that quietly capped its performance:

1. **Double normalization** — the pixel values were divided by 255 twice, so the network was actually trained on inputs in the range `[0, 1/255]` instead of `[0, 1]`.
2. **A no-op shuffle** — the shuffling step did not actually reorder the data before the train/validation split, leaving the split non-stratified and the training data in its original order.

The new solution fixes both problems and layers on several modern improvements:

- **Single, correct normalization** — pixels are divided by 255 exactly once.
- **BatchNorm + Dropout** — batch normalization after every convolution and dense layer for faster, more stable training, plus dropout (0.4) for regularization.
- **Data augmentation** — small random rotations, zooms, and shifts to improve generalization.
- **3-model CNN ensemble** — three independently seeded CNNs whose softmax outputs are averaged at inference time, smoothing out individual-model variance.

The original no-op shuffle is replaced with a proper **stratified** train/validation split.

## Architecture (per model)

Each model in the ensemble is a sequential CNN with two convolutional blocks followed by a dense classifier head:

```
Input(28, 28, 1)

# Block 1 (32 channels)
[Conv2D(32, 3) -> BatchNorm -> ReLU] x2
Conv2D(32, 5, stride=2) -> BatchNorm -> ReLU
Dropout(0.4)

# Block 2 (64 channels)
[Conv2D(64, 3) -> BatchNorm -> ReLU] x2
Conv2D(64, 5, stride=2) -> BatchNorm -> ReLU
Dropout(0.4)

# Classifier head
Flatten
Dense(128) -> BatchNorm -> ReLU -> Dropout(0.4)
Dense(10, softmax)
```

- **Optimizer:** Adam
- **Learning-rate schedule:** `ReduceLROnPlateau` (halves the LR when validation loss plateaus)

## Results

Measured on a stratified 6,000-image held-out validation split (the notebook's
`"full"` run, 3 models × 30 epochs, using the `keras.datasets.mnist` fallback):

| Solution | Validation accuracy |
|----------|---------------------|
| Original (buggy plain CNN) | ~97.7% |
| Single improved CNN (BN + augmentation) | 99.50–99.57% (per model) |
| **3-CNN ensemble** | **99.55%** |

The individual models reach 99.50%, 99.52%, and 99.57%; averaging their softmax
outputs gives a stable 99.55%. On the full Kaggle training set with more epochs or
additional ensemble members (and optionally test-time augmentation), this approach
reaches ~99.7% — see *Further improvements* below.

## Further improvements

- Train on the real Kaggle `data/train.csv` (42k labelled images) and submit to the
  leaderboard.
- Add more ensemble members (5–15 CNNs) and/or test-time augmentation (TTA).
- Try a cyclic / cosine learning-rate schedule instead of `ReduceLROnPlateau`.

## How to run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Open the notebook:
   ```bash
   jupyter notebook Kaggle_Digit_Recognizer.ipynb
   ```
3. Set the `RUN_MODE` flag in the config cell:
   - `"quick"` — a smoke test that trains **1 model for 2 epochs**, finishing in minutes. Use this to confirm everything runs end to end.
   - `"full"` — the publishable **3-model ensemble** trained for 30 epochs each (~1–3 hours on CPU).
4. **Run All**. The notebook trains the model(s), evaluates the ensemble, plots training curves, a confusion matrix, and misclassified examples, and writes `submission.csv`.

## Getting the Kaggle data (optional)

To run on the real competition data, download it with the Kaggle CLI and unzip the CSVs into a `data/` folder:

```bash
kaggle competitions download -c digit-recognizer
unzip digit-recognizer.zip -d data/
```

You should end up with `data/train.csv` and `data/test.csv`.

**Note:** the notebook automatically falls back to `keras.datasets.mnist` if the `data/` folder is absent, so it runs with **zero setup**. When using the keras fallback, `submission.csv` is produced on the keras test split for demonstration purposes only; a real leaderboard submission requires the actual Kaggle `data/test.csv`.
