"""
Model architectures for species classification.

Supported architectures:
- MobileNetV3-Small: ~7-12MB Core ML model, fast inference on Neural Engine
- MobileNetV3-Large: ~22MB, better accuracy, still mobile-friendly
- EfficientNet-B0: ~20MB, strong accuracy/size tradeoff
- EfficientNet-B1: ~30MB, higher accuracy at moderate size cost
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from .dataset import DEFAULT_CLASS_TO_IDX

SUPPORTED_ARCHITECTURES = [
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "efficientnet_b1",
]


def create_model(
    num_classes: int,
    pretrained: bool = True,
    architecture: str = "mobilenet_v3_small",
) -> nn.Module:
    """Create a classification model with custom classifier head.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to load ImageNet pretrained weights.
        architecture: One of 'mobilenet_v3_small', 'mobilenet_v3_large',
                      'efficientnet_b0', 'efficientnet_b1'.
    """
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"Unsupported architecture '{architecture}'. "
            f"Choose from: {SUPPORTED_ARCHITECTURES}"
        )

    if architecture == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
    elif architecture == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
    elif architecture == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
    elif architecture == "efficientnet_b1":
        weights = models.EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b1(weights=weights)

    # Replace classifier for our number of classes
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    return model


def _normalize_class_map(raw_map) -> dict[str, int]:
    """Normalize class map key/value types from checkpoint payloads."""
    if not raw_map:
        return {}
    return {str(k): int(v) for k, v in raw_map.items()}


def _infer_num_classes_from_state_dict(state_dict) -> int:
    """Infer class count from classifier weight shape when metadata is missing."""
    if not isinstance(state_dict, dict):
        return len(DEFAULT_CLASS_TO_IDX)

    # MobileNetV3 classifier head (torchvision): classifier.3.weight
    classifier_weight = state_dict.get("classifier.3.weight")
    if hasattr(classifier_weight, "shape") and len(classifier_weight.shape) >= 1:
        return int(classifier_weight.shape[0])

    # EfficientNet classifier head (torchvision): classifier.1.weight
    classifier_weight = state_dict.get("classifier.1.weight")
    if hasattr(classifier_weight, "shape") and len(classifier_weight.shape) >= 1:
        return int(classifier_weight.shape[0])

    # Fallback: try to find a classifier weight tensor.
    for key, tensor in state_dict.items():
        if key.endswith(".weight") and "classifier" in key and hasattr(tensor, "shape") and len(tensor.shape) >= 1:
            return int(tensor.shape[0])

    return len(DEFAULT_CLASS_TO_IDX)


def load_trained_model(
    checkpoint_path: str,
    device: str = "cpu",
    return_class_mapping: bool = False,
    architecture: str | None = None,
):
    """Load a trained model from checkpoint and optionally return class mapping.

    Args:
        checkpoint_path: Path to the .pth checkpoint file.
        device: Device to load the model onto.
        return_class_mapping: If True, return (model, class_to_idx) tuple.
        architecture: If provided, override the architecture stored in the
                      checkpoint. Otherwise read from checkpoint, falling back
                      to 'mobilenet_v3_small'.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Determine architecture: explicit override > checkpoint > default
    if architecture is None:
        architecture = checkpoint.get("architecture", "mobilenet_v3_small")

    class_to_idx = _normalize_class_map(checkpoint.get("class_to_idx"))
    if not class_to_idx and checkpoint.get("idx_to_class"):
        class_to_idx = {
            str(v): int(k)
            for k, v in checkpoint["idx_to_class"].items()
        }
    if not class_to_idx:
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        inferred_num_classes = _infer_num_classes_from_state_dict(state_dict)
        if inferred_num_classes == len(DEFAULT_CLASS_TO_IDX):
            class_to_idx = dict(DEFAULT_CLASS_TO_IDX)
        else:
            class_to_idx = {f"class_{i:03d}": i for i in range(inferred_num_classes)}

    model = create_model(
        num_classes=len(class_to_idx),
        pretrained=False,
        architecture=architecture,
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    if return_class_mapping:
        return model, class_to_idx
    return model
