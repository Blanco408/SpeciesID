# SpeciesID Model Architecture

## Overview

SpeciesID uses a **MobileNetV3-Small** backbone fine-tuned for 3-class marine species classification. The PyTorch model is converted to **Core ML** (`.mlpackage`) for on-device inference on iOS via the Vision framework.

## 2026 Upgrade Notes

The current codebase now supports a scalable class map and multi-species detection workflow:

- **Dynamic class mapping in training/eval/export**:
  - Class IDs are inferred from split CSVs (`class_label`) instead of hardcoded constants.
  - Checkpoints save `class_to_idx` and `idx_to_class`.
  - Core ML conversion uses checkpoint class labels automatically.

- **Multi-species on-device inference (offline)**:
  - iOS now performs tiled multi-crop classification over one photo.
  - Predictions are merged with IoU-based suppression to return multiple species detections with approximate bounding boxes.
  - No network dependency is required for classification.

- **Local/offline data schema**:
  - Observations now store multi-identification payloads (`identificationsJSON`) while remaining backward compatible with legacy single-label records.
  - Sync/export/history consume the new multi-identification format.

```
┌─────────────────────────────────────────────────────────┐
│                    TRAINING (PyTorch)                    │
│                                                         │
│  Images ─► Augmentation ─► MobileNetV3-Small ─► Logits │
│  224x224   (color jitter,   (ImageNet pretrained)  ↓    │
│             flip, rotate,                     Softmax   │
│             crop, affine)                        ↓      │
│                                          3 class probs  │
│                                  [brittlestar,          │
│                                   sea_cucumber,         │
│                                   seahare]              │
└─────────────────────┬───────────────────────────────────┘
                      │ convert_to_coreml.py
                      ▼
┌─────────────────────────────────────────────────────────┐
│                 INFERENCE (iOS / Core ML)                │
│                                                         │
│  UIImage ─► VNCoreMLRequest ─► SpeciesClassifier ─►     │
│             (center crop       (.mlpackage)        ↓    │
│              to 224x224)    scale=1/255 + ImageNet ClassificationResult │
│                             normalization baked in       │
└─────────────────────────────────────────────────────────┘
```

## Model: MobileNetV3-Small

| Property | Value |
|---|---|
| Architecture | MobileNetV3-Small |
| Pretrained weights | ImageNet-1K (torchvision) |
| Input size | 224 x 224 x 3 (RGB) |
| Output classes | 3 |
| PyTorch checkpoint | 18 MB (`best_model.pth`) |
| Core ML package | 3 MB (`SpeciesClassifier.mlpackage`) |
| Core ML format | mlprogram (iOS 17+) |
| Compute units | CPU + Neural Engine |

### Network Structure

MobileNetV3-Small uses inverted residual blocks with squeeze-and-excitation (SE) attention. Only the final classifier layer is modified:

```
MobileNetV3-Small
├── features (frozen first few epochs, then fine-tuned)
│   ├── Conv2d(3, 16) + BatchNorm + Hardswish
│   ├── InvertedResidual blocks x11
│   │   └── Each: expand ─► depthwise conv ─► SE ─► project
│   └── Conv2d(96, 576) + BatchNorm + Hardswish
├── avgpool (AdaptiveAvgPool2d → 1x1)
└── classifier (MODIFIED)
    ├── Linear(576, 1024)
    ├── Hardswish
    ├── Dropout(p=0.2)
    └── Linear(1024, 3)  ◄── replaced from 1000 (ImageNet) → 3
```

### Output Classes

| Index | Class ID | Display Name | Scientific Name |
|---|---|---|---|
| 0 | `brittlestar` | Brittle Star | Ophiuroidea |
| 1 | `sea_cucumber` | Sea Cucumber | Holothuroidea |
| 2 | `seahare` | California Seahare | Aplysia californica |

## Dataset

| Split | Samples |
|---|---|
| Train | 6,297 |
| Validation | 1,352 |
| Test | 1,344 |
| **Total** | **8,993** |

Images per class: ~2,997 each (balanced).
Source: iNaturalist observation photos.

## Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| LR schedule | Cosine annealing |
| Warmup epochs | 2 |
| Max epochs | 30 |
| Early stopping | 7 epochs patience |
| Batch size | 32 |
| Loss | CrossEntropyLoss (class-weighted) |
| Mixed precision | CUDA: yes, MPS/CPU: no |

### Data Augmentation (Training)

```
RandomResizedCrop(224, scale=0.7-1.0)
RandomHorizontalFlip
RandomVerticalFlip
ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
RandomRotation(20°)
RandomAffine(translate=10%)
Normalize(ImageNet mean/std)
```

### Validation/Test Transform

```
Resize(256)
CenterCrop(224)
Normalize(ImageNet mean/std)
```

## Core ML Conversion

The conversion wraps the model with a `NormalizedModel` layer so that Core ML only needs to handle pixel scaling:

```
Core ML pipeline:
  1. Input: RGB image (any size)
  2. Core ML resizes to 224x224
  3. Scale pixels: x * (1/255) → [0, 1]
  4. NormalizedModel applies: (x - ImageNet_mean) / ImageNet_std
  5. MobileNetV3-Small forward pass
  6. Output: classLabel (string) + classProbabilities (dict)
```

Key conversion settings:
- `convert_to="mlprogram"` (newer, faster format)
- `minimum_deployment_target=iOS17`
- `ClassifierConfig` for native label output
- `ImageType` input with `scale=1/255`, `color_layout=RGB`

## iOS Inference Pipeline

```swift
// SpeciesClassifierService.swift
1. Load:    SpeciesClassifier(configuration: .cpuAndNeuralEngine)
            → VNCoreMLModel
2. Classify: VNCoreMLRequest with .centerCrop
             → VNImageRequestHandler(cgImage)
             → [VNClassificationObservation]
3. Result:  Top prediction + top-2 alternatives with confidence scores
```

## File Map

```
ml/
├── training/
│   ├── model.py          # MobileNetV3-Small creation + loading
│   ├── dataset.py        # Dataset class, transforms, class mappings
│   ├── train.py          # Training loop (AdamW, cosine LR, early stopping)
│   └── evaluate.py       # Test-set eval (confusion matrix, precision/recall/F1)
├── scripts/
│   ├── convert_to_coreml.py    # PyTorch → Core ML conversion
│   ├── download_images.py      # iNaturalist image scraper
│   ├── download_sea_cucumbers.py
│   ├── split_dataset.py        # Train/val/test split generation
│   └── validate_dataset.py     # Dataset integrity checks
├── models/
│   ├── best_model.pth          # Trained PyTorch checkpoint (18 MB)
│   └── SpeciesClassifier.mlpackage/  # (generated, also copied to Xcode)
└── data/
    ├── images/{class}/*.jpg    # ~8,993 images
    └── splits/*.csv            # Train/val/test CSVs

SpeciesID/SpeciesID/
├── SpeciesClassifier.mlpackage/        # Core ML model (3 MB)
├── SpeciesClassifierService.swift      # Inference service (Vision + CoreML)
├── ClassificationResultView.swift      # Result display UI
└── CameraView.swift                    # Camera capture + classification trigger
```
