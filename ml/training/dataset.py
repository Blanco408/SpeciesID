"""PyTorch dataset and class-map helpers for species classification."""

from __future__ import annotations

import csv
import os
from typing import Dict, Iterable, Optional

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# Backward-compatible fallback when loading older checkpoints with no class map.
DEFAULT_CLASS_TO_IDX: Dict[str, int] = {
    "brittlestar": 0,
    "sea_cucumber": 1,
    "seahare": 2,
}


def build_class_to_idx_from_csv(csv_paths: Iterable[str]) -> Dict[str, int]:
    """Build deterministic class->index mapping from one or more split CSV files."""
    labels = set()
    for path in csv_paths:
        if not path or not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = (row.get("class_label") or "").strip()
                if label:
                    labels.add(label)
    if not labels:
        return dict(DEFAULT_CLASS_TO_IDX)
    return {label: idx for idx, label in enumerate(sorted(labels))}


def get_idx_to_class(class_to_idx: Dict[str, int]) -> Dict[int, str]:
    """Invert class map."""
    return {idx: label for label, idx in class_to_idx.items()}


def get_train_transforms():
    """Augmentation pipeline for training.

    Aggressive color jitter for underwater photo variation (blue/green cast,
    turbidity, lighting). Rotation and flips for varied viewing angles.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.12), ratio=(0.3, 3.0), value="random"),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms():
    """Standard validation/test transforms."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class SpeciesDataset(Dataset):
    """Dataset for species classification from split CSV files."""

    def __init__(
        self,
        split_csv: str,
        class_to_idx: Optional[Dict[str, int]] = None,
        transform=None,
    ):
        """
        Args:
            split_csv: Path to split CSV (train.csv, val.csv, or test.csv)
            class_to_idx: Optional explicit class mapping. If omitted, inferred from
                this split CSV.
            transform: torchvision transforms to apply
        """
        self.transform = transform
        self.samples = []
        self.class_to_idx = class_to_idx or build_class_to_idx_from_csv([split_csv])
        self.idx_to_class = get_idx_to_class(self.class_to_idx)

        with open(split_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = row["image_path"]
                class_label = row["class_label"]
                if os.path.exists(img_path) and class_label in self.class_to_idx:
                    self.samples.append((img_path, self.class_to_idx[class_label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label
