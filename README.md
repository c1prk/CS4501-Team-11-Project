# Detecting Deepfake Spatial Asymmetry using SIFT Features and CNNs

CS 4501: Computer Vision — Ryan Kim (qec4gc) & Siwon Park (qra4bh)

We compare classical computer vision and deep learning for deepfake detection using the FaceForensics++ dataset. The classical track uses Harris Corner + SIFT bilateral asymmetry and a Bag-of-Visual-Words SVM. The deep learning track fine-tunes a ResNet-18 CNN.

## Results

| Method | Test Accuracy |
|---|---|
| Majority baseline | 66.5% |
| Spatial asymmetry + SVM | 51.0% |
| BoVW + SVM (improved) | 66.9% |
| ResNet-18 + TTA | **84.5%** |

## Setup

```bash
pip install -r requirements.txt
```

You'll also need the FaceForensics++ dataset downloaded into `data/raw/`. Request access at [github.com/ondyari/FaceForensics](https://github.com/ondyari/FaceForensics).

Expected structure:
```
data/raw/
  manipulated_sequences/
    Deepfakes/c23/videos/
    Face2Face/c23/videos/
    FaceSwap/c23/videos/
    FaceShifter/c23/videos/
    NeuralTextures/c23/videos/
    DeepFakeDetection/c23/videos/
  original_sequences/
    actors/c23/videos/
    youtube/c23/videos/
```

## Usage

**Step 1 — Preprocess videos into face frames:**
```bash
python src/preprocess.py
```
This extracts 1 frame/sec, detects and crops faces (224×224), and creates train/val/test splits under `data/splits/`.

**Step 2a — Run the classical pipeline (BoVW + SVM):**
```bash
python -m src.classifier
```

**Step 2b — Run the asymmetry pipeline:**
```bash
python src/sift_pipeline/tests/batch_process.py
python src/sift_pipeline/tests/train_svm.py
```

**Step 3 — Train the CNN:**
```bash
python -m src.cnn_pipeline.train --backbone resnet18 --epochs 20
```
Checkpoints save to `data/weights/`.

**Step 4 — Evaluate the CNN:**
```bash
python -m src.evaluate --tta
```
Prints accuracy, AUC, per-class metrics, and per-manipulation breakdown. Saves confusion matrix and ROC curve to `docs/figures/`.

## Project Structure

```
src/
  preprocess.py              # video → face frames + splits
  classifier.py              # BoVW + spatial pyramid + SVM
  dataset.py                 # PyTorch dataset loader
  evaluate.py                # CNN evaluation + figures
  sift_pipeline/
    sift_extractor.py        # Harris + SIFT asymmetry score
    tests/
      batch_process.py       # batch asymmetry feature extraction
      classical_classifier.py
      train_svm.py
  cnn_pipeline/
    model.py                 # ResNet-18 with custom head
    train.py                 # progressive fine-tuning training loop
data/
  splits/                    # train.txt, val.txt, test.txt
  processed/                 # extracted face frames (generated)
  weights/                   # model checkpoints (generated)
docs/
  figures/                   # evaluation plots (generated)
```