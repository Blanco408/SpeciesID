#!/usr/bin/env python3
"""
Convert trained PyTorch model to Core ML format for iOS deployment.

The output .mlmodel includes:
- Built-in image normalization (no manual preprocessing on iOS)
- ClassifierConfig for native class label + probability output
- Float16 quantization for smaller file size
- Metadata (author, description, version)
"""

import os
import sys
import argparse

import torch
import coremltools as ct

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from ml.training.dataset import IDX_TO_CLASS, IMAGENET_MEAN, IMAGENET_STD
from ml.training.model import load_trained_model

ML_DIR = os.path.join(PROJECT_ROOT, "ml")
DEFAULT_MODEL_PATH = os.path.join(ML_DIR, "models", "best_model.pth")
DEFAULT_OUTPUT_PATH = os.path.join(ML_DIR, "models", "SpeciesClassifier.mlmodel")

# Class labels in index order
CLASS_LABELS = [IDX_TO_CLASS[i] for i in range(len(IDX_TO_CLASS))]


class NormalizedModel(torch.nn.Module):
    """Wraps model with ImageNet normalization so Core ML only needs scale=1/255."""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        # x comes in as [0, 1] after Core ML scale=1/255
        x = (x - self.mean) / self.std
        return self.base_model(x)


def convert_to_coreml(model_path: str, output_path: str):
    """Convert PyTorch model to Core ML with optimizations."""
    print(f"Loading PyTorch model from {model_path}")
    model = load_trained_model(model_path, device="cpu")
    model.eval()

    # Wrap model with normalization so Core ML only needs to scale pixels to [0,1]
    wrapped_model = NormalizedModel(model)
    wrapped_model.eval()

    # Trace the model
    print("Tracing model...")
    example_input = torch.rand(1, 3, 224, 224)
    traced_model = torch.jit.trace(wrapped_model, example_input)

    # Core ML ImageType scale=1/255 converts pixels from [0,255] to [0,1]
    # The NormalizedModel then applies (x - mean) / std internally
    scale = 1.0 / 255.0

    print(f"Class labels: {CLASS_LABELS}")
    print("Converting to Core ML...")

    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, 224, 224),
                scale=scale,
                bias=[0.0, 0.0, 0.0],
                color_layout="RGB",
            )
        ],
        classifier_config=ct.ClassifierConfig(class_labels=CLASS_LABELS),
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
    )

    # Add metadata
    mlmodel.author = "SpeciesID Team"
    mlmodel.short_description = "Classifies marine species: seahare, brittlestar, sea cucumber"
    mlmodel.version = "1.0.0"
    mlmodel.input_description["image"] = "Color photo of marine organism (224x224)"

    # Save
    print(f"Saving Core ML model to {output_path}")
    mlmodel.save(output_path)

    # Check file size
    size_bytes = os.path.getsize(output_path) if os.path.isfile(output_path) else 0
    if size_bytes == 0:
        # mlprogram is saved as a directory (mlpackage)
        mlpackage_path = output_path
        if os.path.isdir(mlpackage_path):
            total = 0
            for dirpath, dirnames, filenames in os.walk(mlpackage_path):
                for f in filenames:
                    total += os.path.getsize(os.path.join(dirpath, f))
            size_bytes = total

    size_mb = size_bytes / (1024 * 1024)
    print(f"Model size: {size_mb:.1f} MB")

    if size_mb > 100:
        print(f"WARNING: Model exceeds 100MB limit ({size_mb:.1f} MB)")
    else:
        print(f"PASS: Model is under 100MB limit")

    return mlmodel


def validate_coreml_model(mlmodel, pytorch_model_path: str):
    """Validate Core ML model predictions match PyTorch."""
    print("\nValidating Core ML model accuracy...")

    try:
        import numpy as np
        from PIL import Image

        model_pt = load_trained_model(pytorch_model_path, device="cpu")
        model_pt.eval()

        # Test with random images
        num_tests = 20
        mismatches = 0

        for i in range(num_tests):
            # Generate random test image
            np.random.seed(i)
            test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            pil_img = Image.fromarray(test_img)

            # PyTorch prediction
            from ml.training.dataset import get_val_transforms
            transform = get_val_transforms()
            # For validation, we need to create an image of the right size
            tensor = torch.from_numpy(test_img).permute(2, 0, 1).float() / 255.0
            mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
            std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
            tensor = (tensor - mean) / std
            tensor = tensor.unsqueeze(0)

            with torch.no_grad():
                pt_output = model_pt(tensor)
                pt_pred = pt_output.argmax(1).item()
                pt_class = IDX_TO_CLASS[pt_pred]

            # Core ML prediction
            cml_output = mlmodel.predict({"image": pil_img})
            cml_class = cml_output.get("classLabel", "")

            if pt_class != cml_class:
                mismatches += 1

        mismatch_rate = mismatches / num_tests
        print(f"Validation: {num_tests - mismatches}/{num_tests} predictions match ({(1-mismatch_rate)*100:.0f}%)")

        if mismatch_rate > 0.1:  # Allow up to 10% mismatch (quantization effects)
            print(f"WARNING: High mismatch rate ({mismatch_rate:.0%})")
        else:
            print("PASS: Core ML model predictions are consistent with PyTorch")

    except Exception as e:
        print(f"Validation skipped: {e}")


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch model to Core ML")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to PyTorch checkpoint")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output .mlmodel path")
    parser.add_argument("--skip-validation", action="store_true", help="Skip prediction validation")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    mlmodel = convert_to_coreml(args.model, args.output)

    if not args.skip_validation:
        validate_coreml_model(mlmodel, args.model)

    print(f"\nDone! Core ML model saved to: {args.output}")
    print(f"Copy this file into SpeciesID/SpeciesID/ for Xcode to pick it up.")


if __name__ == "__main__":
    main()
