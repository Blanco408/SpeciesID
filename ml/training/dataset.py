"""
PyTorch Dataset for species classification training.
"""

import csv
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Class label to index mapping
CLASS_TO_IDX = {
    "brittlestar": 0,
    "sea_cucumber": 1,
    "seahare": 2,
}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
NUM_CLASSES = len(CLASS_TO_IDX)


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

    def __init__(self, split_csv: str, transform=None):
        """
        Args:
            split_csv: Path to split CSV (train.csv, val.csv, or test.csv)
            transform: torchvision transforms to apply
        """
        self.transform = transform
        self.samples = []

        with open(split_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = row["image_path"]
                class_label = row["class_label"]
                if os.path.exists(img_path) and class_label in CLASS_TO_IDX:
                    self.samples.append((img_path, CLASS_TO_IDX[class_label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label
