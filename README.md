# Hot Dog / Not Hot Dog — Binary Image Classifier

A binary image classifier inspired by the Silicon Valley TV show, built using transfer
learning on EfficientNetV2B0. Trained on the Food-101 dataset with balanced sampling,
data augmentation, and two-phase fine-tuning to classify images as either hot dog or
not hot dog.

## Dataset

- Source: Food-101 via TensorFlow Datasets (tfds.load("food101"))
- Total images: 101,000 across 101 food categories
- Hot dog class: label 55 (750 training / 250 validation images)
- Task: Binary classification — hot dog (1) vs not hot dog (0)
- Class balance: 50/50 balanced sampling using tf.data.Dataset.sample_from_datasets

Since hot dogs represent only 1 of 101 classes (~1% of data), balanced sampling is
applied at each training step to prevent the model from predicting "not hot dog" for
everything.

## Pipeline Overview

1. Load Food-101 dataset via TensorFlow Datasets
2. Convert 101-class labels to binary (1 = hot dog, 0 = everything else)
3. Separate hot dog and not hot dog streams and apply 50/50 balanced sampling
4. Resize images to 224x224 and apply EfficientNetV2 preprocessing
5. Apply data augmentation during training
6. Phase 1 — train classification head only (8 epochs, frozen backbone)
7. Phase 2 — unfreeze last 60 layers and fine-tune (12 epochs, lower LR)
8. Track best model by validation AUC with ModelCheckpoint
9. Visualise training curves and sample predictions

## Model Architecture

```
Input: (224, 224, 3)
        |
Data Augmentation (flip, rotation, zoom, contrast, translation)
        |
EfficientNetV2 preprocessing (built-in normalisation)
        |
EfficientNetV2B0 (pretrained on ImageNet — frozen in Phase 1)
        |
GlobalAveragePooling2D
        |
Dropout (0.3)
        |
Dense (1 unit, Sigmoid)
        |
Output: probability of hot dog
```

## Training Configuration

| Setting | Phase 1 (Head) | Phase 2 (Fine-tune) |
|---|---|---|
| Epochs | 8 | 12 (early stopped at 11) |
| Learning rate | 1e-3 | 1e-5 |
| Optimiser | Adam | Adam |
| Loss | Binary Cross-Entropy | Binary Cross-Entropy |
| Steps per epoch | 300 | 300 |
| Batch size | 32 | 32 |

Callbacks: ModelCheckpoint (best val AUC), EarlyStopping (patience=4), ReduceLROnPlateau

## Tech Stack

| Library | Purpose |
|---|---|
| TensorFlow / Keras | Model building, training, and callbacks |
| TensorFlow Datasets | Food-101 dataset loading |
| EfficientNetV2B0 | ImageNet pretrained backbone |
| NumPy / matplotlib | Visualisation and sample prediction plots |
