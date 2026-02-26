#!/usr/bin/env python3
"""
Train MobileNetV3-Small for species classification.

Usage:
    python -m ml.training.train
    python -m ml.training.train --epochs 30 --batch-size 32
"""

import os
import sys
import argparse
import time
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from ml.training.dataset import (
    SpeciesDataset, get_train_transforms, get_val_transforms,
    IDX_TO_CLASS, NUM_CLASSES,
)
from ml.training.model import create_model

ML_DIR = os.path.join(PROJECT_ROOT, "ml")
DEFAULT_SPLITS_DIR = os.path.join(ML_DIR, "data", "splits")
DEFAULT_OUTPUT_DIR = os.path.join(ML_DIR, "models")
DEFAULT_PLOTS_DIR = os.path.join(ML_DIR, "outputs")


def compute_class_weights(dataset: SpeciesDataset) -> torch.Tensor:
    """Compute inverse frequency class weights for imbalanced datasets."""
    counts = defaultdict(int)
    for _, label in dataset.samples:
        counts[label] += 1

    total = sum(counts.values())
    weights = []
    for i in range(NUM_CLASSES):
        count = counts.get(i, 1)
        weights.append(total / (NUM_CLASSES * count))

    return torch.FloatTensor(weights)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """Train for one epoch. Returns average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate model. Returns loss, accuracy, and per-class metrics."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        for pred, label in zip(predicted, labels):
            class_total[label.item()] += 1
            if pred == label:
                class_correct[label.item()] += 1

    avg_loss = running_loss / total
    accuracy = correct / total

    per_class = {}
    for idx in range(NUM_CLASSES):
        class_name = IDX_TO_CLASS[idx]
        total_c = class_total.get(idx, 0)
        correct_c = class_correct.get(idx, 0)
        per_class[class_name] = correct_c / total_c if total_c > 0 else 0.0

    return avg_loss, accuracy, per_class


def save_training_curves(train_losses, val_losses, train_accs, val_accs, output_path):
    """Save training curves plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        epochs = range(1, len(train_losses) + 1)

        ax1.plot(epochs, train_losses, "b-", label="Train")
        ax1.plot(epochs, val_losses, "r-", label="Validation")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss")
        ax1.legend()

        ax2.plot(epochs, train_accs, "b-", label="Train")
        ax2.plot(epochs, val_accs, "r-", label="Validation")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Accuracy")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Training curves saved to {output_path}")
    except ImportError:
        print("matplotlib not available, skipping training curves plot")


def main():
    parser = argparse.ArgumentParser(description="Train species classifier")
    parser.add_argument("--splits-dir", default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--plots-dir", default=DEFAULT_PLOTS_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon GPU
        print("Using Apple Silicon MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Datasets
    train_csv = os.path.join(args.splits_dir, "train.csv")
    val_csv = os.path.join(args.splits_dir, "val.csv")

    if not os.path.exists(train_csv):
        print(f"ERROR: Train split not found at {train_csv}")
        print("Run split_dataset.py first!")
        sys.exit(1)

    print("Loading datasets...")
    train_dataset = SpeciesDataset(train_csv, transform=get_train_transforms())
    val_dataset = SpeciesDataset(val_csv, transform=get_val_transforms())

    print(f"Train: {len(train_dataset)} samples")
    print(f"Val:   {len(val_dataset)} samples")

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("ERROR: Empty dataset!")
        sys.exit(1)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Model
    print(f"\nCreating MobileNetV3-Small (pretrained=True, classes={NUM_CLASSES})")
    model = create_model(pretrained=True)
    model = model.to(device)

    # Class weights for imbalanced data
    class_weights = compute_class_weights(train_dataset).to(device)
    print(f"Class weights: {class_weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)

    # Mixed precision for CUDA
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print(f"\nTraining for up to {args.epochs} epochs (patience={args.patience})")
    print(f"{'='*70}")

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        # Warmup: lower LR for first few epochs
        if epoch <= args.warmup_epochs:
            warmup_lr = args.lr * epoch / args.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_acc, per_class = validate(model, val_loader, criterion, device)

        if epoch > args.warmup_epochs:
            scheduler.step()

        elapsed = time.time() - start_time
        lr = optimizer.param_groups[0]["lr"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Print epoch results
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {lr:.6f} | {elapsed:.1f}s")

        # Per-class accuracy
        per_class_str = " | ".join(f"{name}: {acc:.3f}" for name, acc in per_class.items())
        print(f"         Per-class: {per_class_str}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "class_to_idx": {v: k for k, v in IDX_TO_CLASS.items()},
            }
            best_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            print(f"         * Best model saved! (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

        # Save periodic checkpoint
        if epoch % 5 == 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
            }, ckpt_path)

    print(f"\n{'='*70}")
    print(f"Training complete! Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best model saved to: {os.path.join(args.output_dir, 'best_model.pth')}")

    # Save training curves
    curves_path = os.path.join(args.plots_dir, "training_curves.png")
    save_training_curves(train_losses, val_losses, train_accs, val_accs, curves_path)


if __name__ == "__main__":
    main()
