"""
MobileNetV3-Small model for species classification.

MobileNetV3-Small is chosen for its excellent mobile performance:
- ~7-12MB Core ML model (well under 100MB limit)
- Fast inference on Neural Engine (~50-200ms)
- Good accuracy with transfer learning from ImageNet
"""

import torch
import torch.nn as nn
from torchvision import models

from .dataset import NUM_CLASSES


def create_model(pretrained: bool = True) -> nn.Module:
    """Create MobileNetV3-Small with custom classifier head for species classification."""
    if pretrained:
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    else:
        weights = None

    model = models.mobilenet_v3_small(weights=weights)

    # Replace classifier for our number of classes
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)

    return model


def load_trained_model(checkpoint_path: str, device: str = "cpu") -> nn.Module:
    """Load a trained model from checkpoint."""
    model = create_model(pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model
