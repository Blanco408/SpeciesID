# SpeciesID Model Architecture Comparison Report

**Date:** April 14, 2026
**Dataset:** 20 Pacific coast marine species
**Training samples:** 3,500 | **Validation:** 750 | **Test:** 752
**Data source:** iNaturalist research-grade observations

---

## Executive Summary

Two model architectures were trained and evaluated for on-device marine species classification: **MobileNetV3-Large** and **EfficientNet-B0**. Both use ImageNet-pretrained weights with a custom 20-class classifier head.

**EfficientNet-B0 is the recommended model.** It outperforms MobileNetV3-Large by 4 percentage points on top-1 accuracy (87.0% vs 83.0%) and 4 points on macro F1 (0.870 vs 0.830), while using fewer parameters (4.0M vs 4.2M). The Core ML model is 7.8 MB — well under the 50 MB budget.

---

## Overall Performance

| Metric | MobileNetV3-Large | EfficientNet-B0 | Delta |
|--------|-------------------|-----------------|-------|
| **Top-1 Accuracy** | 83.0% | **87.0%** | +4.0% |
| **Top-2 Accuracy** | 89.5% | **93.1%** | +3.6% |
| **Top-3 Accuracy** | 93.8% | **95.9%** | +2.1% |
| **Top-5 Accuracy** | 96.7% | **98.4%** | +1.7% |
| **Macro F1** | 0.830 | **0.870** | +0.040 |
| **Weighted F1** | 0.830 | **0.870** | +0.040 |

### Key Observations
- EfficientNet-B0 meets the 85% top-1 accuracy target; MobileNetV3-Large does not.
- Top-3 accuracy of 95.9% means the correct species is almost always in the top 3 suggestions.
- Top-5 accuracy of 98.4% indicates strong coverage across the full prediction set.

---

## Model Specifications

| Attribute | MobileNetV3-Large | EfficientNet-B0 |
|-----------|-------------------|-----------------|
| Parameters | 4.2M | 4.0M |
| Core ML Size | ~11 MB (est.) | **7.8 MB** |
| Training Time | 1,076 sec (18 min) | 1,296 sec (22 min) |
| Best Epoch | 15/15 | 14/15 |
| Optimizer | AdamW (lr=1e-4) | AdamW (lr=1e-4) |
| Scheduler | Cosine Annealing | Cosine Annealing |
| Warmup | 2 epochs | 2 epochs |
| Label Smoothing | 0.05 | 0.05 |
| Early Stopping | Patience=5 (macro F1) | Patience=5 (macro F1) |

---

## Per-Class Performance (EfficientNet-B0)

### Top Performers (F1 > 0.90)
| Species | Precision | Recall | F1 |
|---------|-----------|--------|----|
| Spanish Shawl Nudibranch | 1.000 | 1.000 | **1.000** |
| Giant Green Anemone | 0.974 | 0.925 | 0.949 |
| Sea Lemon Nudibranch | 0.905 | 0.974 | 0.938 |
| Red Sea Urchin | 0.973 | 0.900 | 0.935 |
| Seahare | 0.944 | 0.919 | 0.932 |
| Ochre Sea Star | 0.921 | 0.921 | 0.921 |
| Owl Limpet | 0.919 | 0.919 | 0.919 |
| Acorn Barnacle | 0.897 | 0.921 | 0.909 |

### Needs Improvement (F1 < 0.85)
| Species | Precision | Recall | F1 | Primary Confusion |
|---------|-----------|--------|----|-------------------|
| East Pacific Red Octopus | 0.781 | 0.676 | **0.725** | Red Abalone, Red Rock Crab |
| Blueband Hermit Crab | 0.903 | 0.718 | 0.800 | Kelp Crab, Acorn Barnacle |
| Brittlestar | 0.848 | 0.757 | 0.800 | Aggregating Anemone |
| Red Rock Crab | 0.733 | 0.892 | 0.805 | Kelp Crab, Red Abalone |
| Aggregating Anemone | 0.816 | 0.816 | 0.816 | Brittlestar, Red Rock Crab |
| Sea Cucumber | 0.833 | 0.811 | 0.822 | Sea Lemon Nudibranch |
| Red Abalone | 0.761 | 0.921 | 0.833 | Owl Limpet |
| California Mussel | 0.720 | 1.000 | 0.837 | (false positives from others) |

### Analysis
- **Best classified:** Spanish Shawl Nudibranch (100% — highly distinctive purple/orange coloring)
- **Most confused:** East Pacific Red Octopus (F1=0.725) — confused with similarly-colored species in rocky habitats
- **California Mussel** has perfect recall (never missed) but low precision (other species misclassified as mussel) — likely because mussel beds are a common background in many photos
- **Crustaceans** as a group are harder to distinguish from each other (crabs, hermit crabs)

---

## Architecture Comparison: Per-Class Deltas

Species where EfficientNet-B0 significantly outperforms MobileNetV3-Large:

| Species | MNV3-L F1 | EffB0 F1 | Improvement |
|---------|-----------|----------|-------------|
| East Pacific Red Octopus | 0.600 | 0.725 | **+0.125** |
| Giant Green Anemone | 0.854 | 0.949 | **+0.095** |
| Acorn Barnacle | 0.818 | 0.909 | **+0.091** |
| Red Abalone | 0.767 | 0.833 | **+0.066** |
| Bat Star | 0.842 | 0.892 | **+0.050** |
| Spanish Shawl Nudibranch | 0.951 | 1.000 | **+0.049** |

Species where MobileNetV3-Large performed better:
| Species | MNV3-L F1 | EffB0 F1 | Delta |
|---------|-----------|----------|-------|
| Brittlestar | 0.870 | 0.800 | -0.070 |
| Blueband Hermit Crab | 0.831 | 0.800 | -0.031 |

EfficientNet-B0 wins on **18 of 20 species**.

---

## Deployment

The EfficientNet-B0 model has been converted to Core ML and deployed:
- **Format:** mlpackage (mlprogram, iOS 17+)
- **Size:** 7.8 MB
- **Input:** 224x224 RGB image
- **Output:** 20-class probabilities + class label
- **Validation:** 100% prediction match between PyTorch and Core ML
- **Location:** `SpeciesID/SpeciesID/SpeciesClassifier.mlpackage`

---

## Recommendations for Further Improvement

1. **More training data for weak classes:** East Pacific Red Octopus, Brittlestar, and Blueband Hermit Crab would benefit from 500+ images each (currently ~250).

2. **Expand to 50-75 species:** The pipeline is ready. A 66-species config has been prepared at `ml/config/marine_species_names_75.json`. Run `validate_species_config.py` to verify data availability, then download and retrain.

3. **EfficientNet-B1:** With the expanded dataset (more data per class), EfficientNet-B1 (6.5M params, ~15MB Core ML) may yield further gains. The training pipeline supports it via `--architecture efficientnet_b1`.

4. **Confidence calibration:** Consider temperature scaling on the validation set to improve confidence score reliability.

5. **Hard negative mining:** For confused species pairs (octopus/abalone, crab species), adding targeted training examples of the confusing pairs could help.

---

## Training Artifacts

| File | Description |
|------|-------------|
| `ml/models/effb0_20sp/best_model.pth` | EfficientNet-B0 checkpoint (best) |
| `ml/models/effb0_20sp/experiment_log.json` | Training hyperparameters and results |
| `ml/models/mnv3l_20sp/best_model.pth` | MobileNetV3-Large checkpoint (comparison) |
| `ml/models/mnv3l_20sp/experiment_log.json` | Training hyperparameters and results |
| `ml/outputs/effb0_20sp/test_metrics.json` | Full test set metrics (JSON) |
| `ml/outputs/effb0_20sp/confusion_matrix.png` | Confusion matrix visualization |
| `ml/outputs/effb0_20sp/f1_per_class.png` | Per-class F1 bar chart |
| `ml/outputs/mnv3l_20sp/test_metrics.json` | MobileNetV3-Large test metrics |
| `ml/outputs/mnv3l_20sp/confusion_matrix.png` | MobileNetV3-Large confusion matrix |
| `ml/outputs/mnv3l_20sp/f1_per_class.png` | MobileNetV3-Large F1 bar chart |
