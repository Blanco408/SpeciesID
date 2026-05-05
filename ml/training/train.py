#!/usr/bin/env python3
"""
Train a species classification model.

Usage:
    python -m ml.training.train
    python -m ml.training.train --epochs 30 --batch-size 32
    python -m ml.training.train --architecture efficientnet_b0 --experiment-name effnet_run1
"""

import csv
import itertools
import os
import sys
import argparse
import time
import json
from collections import defaultdict
from datetime import datetime, timezone

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from ml.training.dataset import (
    SpeciesDataset,
    build_class_to_idx_from_csv,
    get_idx_to_class,
    get_train_transforms,
    get_val_transforms,
)
from ml.training.model import (
    create_model,
    expand_classifier_head,
    load_trained_model,
    SUPPORTED_ARCHITECTURES,
)
from ml.training.metrics import compute_metrics, compute_macro_f1


class OutlierExposureDataset(Dataset):
    """Yields just images (no labels) from a path-only manifest CSV.

    Used for the Outlier Exposure auxiliary loss: every batch from this dataset
    pushes the species classifier toward a uniform distribution over classes,
    which generalizes to "treat unfamiliar inputs as low-confidence" at test
    time. Images here MUST be disjoint from the `nothing`-class rows in the
    main split CSVs to avoid the OE signal collapsing into rote memorization.
    """

    def __init__(self, csv_path: str, transform):
        self.transform = transform
        self.paths: list[str] = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                p = (row.get("image_path") or "").strip()
                if p and os.path.exists(p):
                    self.paths.append(p)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def _build_init_from_model(
    checkpoint_path: str,
    architecture: str,
    new_class_to_idx: dict,
) -> nn.Module:
    """Load an existing checkpoint and grow the head to match new_class_to_idx.

    Assumes new_class_to_idx is the alphabetically-sorted superset of the
    checkpoint's class map (i.e. every old class is still present). Computes
    the indices of newly-added classes (e.g. `nothing`) and seeds them via
    expand_classifier_head.
    """
    print(f"Initializing from checkpoint: {checkpoint_path}")
    old_model, old_class_to_idx = load_trained_model(
        checkpoint_path,
        device="cpu",
        return_class_mapping=True,
        architecture=architecture,
    )

    missing = [name for name in old_class_to_idx if name not in new_class_to_idx]
    if missing:
        raise ValueError(
            f"--init-from checkpoint contains classes not present in the new "
            f"split: {missing}. Refusing to drop weights silently."
        )

    new_indices = sorted(
        idx for name, idx in new_class_to_idx.items() if name not in old_class_to_idx
    )
    added_names = sorted(name for name in new_class_to_idx if name not in old_class_to_idx)
    print(f"  carrying over {len(old_class_to_idx)} classes from checkpoint")
    print(f"  adding {len(added_names)} new classes: {added_names}")
    print(f"  new class indices: {new_indices}")

    expand_classifier_head(
        old_model,
        new_num_classes=len(new_class_to_idx),
        new_class_indices=new_indices,
    )
    return old_model

ML_DIR = os.path.join(PROJECT_ROOT, "ml")
DEFAULT_SPLITS_DIR = os.path.join(ML_DIR, "data", "splits")
DEFAULT_OUTPUT_DIR = os.path.join(ML_DIR, "models")
DEFAULT_PLOTS_DIR = os.path.join(ML_DIR, "outputs")


def compute_class_weights(dataset: SpeciesDataset, num_classes: int) -> torch.Tensor:
    """Compute inverse frequency class weights for imbalanced datasets."""
    counts = defaultdict(int)
    for _, label in dataset.samples:
        counts[label] += 1

    total = sum(counts.values())
    weights = []
    for i in range(num_classes):
        count = counts.get(i, 1)
        weights.append(total / (num_classes * count))

    return torch.FloatTensor(weights)


def _oe_loss(logits: torch.Tensor) -> torch.Tensor:
    """Outlier Exposure loss: pushes the softmax toward uniform.

    Equivalent (up to an additive constant) to KL(softmax(logits) || U). Lower
    is better when the input is OOD. Mean is over both batch and class dims so
    the magnitude is comparable to per-element CE.
    """
    return -F.log_softmax(logits, dim=1).mean()


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler=None,
    oe_loader=None,
    oe_weight: float = 0.0,
):
    """Train for one epoch. Returns average loss and accuracy.

    If oe_loader is provided, each step also draws a batch from it and adds
    `oe_weight * _oe_loss(model(oe_batch))` to the loss. The OE loader cycles
    independently so it can be smaller than the main loader.
    """
    model.train()
    running_loss = 0.0
    running_ce = 0.0
    running_oe = 0.0
    correct = 0
    total = 0

    oe_iter = itertools.cycle(oe_loader) if (oe_loader is not None and oe_weight > 0) else None

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                ce_loss = criterion(outputs, labels)
                if oe_iter is not None:
                    oe_images = next(oe_iter).to(device, non_blocking=True)
                    oe_logits = model(oe_images)
                    oe_loss = _oe_loss(oe_logits)
                    loss = ce_loss + oe_weight * oe_loss
                else:
                    oe_loss = torch.tensor(0.0, device=device)
                    loss = ce_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            ce_loss = criterion(outputs, labels)
            if oe_iter is not None:
                oe_images = next(oe_iter).to(device, non_blocking=True)
                oe_logits = model(oe_images)
                oe_loss = _oe_loss(oe_logits)
                loss = ce_loss + oe_weight * oe_loss
            else:
                oe_loss = torch.tensor(0.0, device=device)
                loss = ce_loss
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_ce += ce_loss.item() * images.size(0)
        running_oe += oe_loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    avg_ce = running_ce / total
    avg_oe = running_oe / total
    return avg_loss, accuracy, avg_ce, avg_oe


@torch.no_grad()
def validate(model, loader, criterion, device, idx_to_class: dict[int, str], num_classes: int):
    """Validate model. Returns loss, accuracy, per-class accuracy, and macro-F1."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    all_preds = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        for pred, label in zip(predicted, labels):
            class_total[label.item()] += 1
            if pred == label:
                class_correct[label.item()] += 1

    avg_loss = running_loss / total
    accuracy = correct / total

    per_class = {}
    for idx in range(num_classes):
        class_name = idx_to_class[idx]
        total_c = class_total.get(idx, 0)
        correct_c = class_correct.get(idx, 0)
        per_class[class_name] = correct_c / total_c if total_c > 0 else 0.0

    # Compute macro-F1 via shared metrics module
    per_class_metrics = compute_metrics(all_preds, all_labels, idx_to_class, num_classes)
    macro_f1 = compute_macro_f1(per_class_metrics)

    return avg_loss, accuracy, per_class, macro_f1


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
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--architecture",
        default="mobilenet_v3_small",
        choices=SUPPORTED_ARCHITECTURES,
        help="Model architecture to train",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Experiment name; if provided, outputs go to ml/models/{experiment_name}/",
    )
    parser.add_argument(
        "--init-from",
        default=None,
        help="Path to a checkpoint to fine-tune from. The classifier head is "
             "expanded to match the current split's class set; new classes "
             "(e.g. 'nothing') get a mean-of-existing prior.",
    )
    parser.add_argument(
        "--oe-csv",
        default=None,
        help="Path to an Outlier Exposure manifest (image_path column). Adds "
             "an auxiliary loss that pushes OOD inputs toward uniform output.",
    )
    parser.add_argument(
        "--oe-weight",
        type=float,
        default=0.5,
        help="Weight on the Outlier Exposure auxiliary loss",
    )
    args = parser.parse_args()

    # Sensible defaults when fine-tuning: lower LR, fewer epochs, tighter patience.
    if args.init_from:
        if "--lr" not in sys.argv:
            args.lr = 2e-5
        if "--epochs" not in sys.argv:
            args.epochs = 10
        if "--patience" not in sys.argv:
            args.patience = 4

    # Override output dir if experiment name is provided
    if args.experiment_name:
        args.output_dir = os.path.join(ML_DIR, "models", args.experiment_name)

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
    class_to_idx = build_class_to_idx_from_csv([train_csv, val_csv])
    idx_to_class = get_idx_to_class(class_to_idx)
    num_classes = len(class_to_idx)

    train_dataset = SpeciesDataset(
        train_csv,
        class_to_idx=class_to_idx,
        transform=get_train_transforms(),
    )
    val_dataset = SpeciesDataset(
        val_csv,
        class_to_idx=class_to_idx,
        transform=get_val_transforms(),
    )

    print(f"Train: {len(train_dataset)} samples")
    print(f"Val:   {len(val_dataset)} samples")
    print(f"Classes: {num_classes}")
    for class_name in sorted(class_to_idx.keys()):
        print(f"  - {class_name} -> {class_to_idx[class_name]}")

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

    # Optional Outlier Exposure dataloader
    oe_loader = None
    if args.oe_csv and args.oe_weight > 0:
        if not os.path.exists(args.oe_csv):
            print(f"ERROR: --oe-csv not found: {args.oe_csv}")
            sys.exit(1)
        oe_dataset = OutlierExposureDataset(args.oe_csv, transform=get_train_transforms())
        if len(oe_dataset) == 0:
            print(f"WARNING: OE dataset at {args.oe_csv} is empty; disabling OE")
        else:
            print(f"OE pool: {len(oe_dataset)} images (weight={args.oe_weight})")
            oe_loader = DataLoader(
                oe_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=max(1, args.num_workers // 2),
                pin_memory=True,
                drop_last=True,
            )

    # Model
    if args.init_from:
        model = _build_init_from_model(
            checkpoint_path=args.init_from,
            architecture=args.architecture,
            new_class_to_idx=class_to_idx,
        )
    else:
        print(f"\nCreating {args.architecture} (pretrained=True, classes={num_classes})")
        model = create_model(
            num_classes=num_classes, pretrained=True, architecture=args.architecture
        )
    model = model.to(device)

    # Class weights for imbalanced data
    class_weights = compute_class_weights(train_dataset, num_classes=num_classes).to(device)
    print(f"Class weights: {class_weights.tolist()}")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)

    # Mixed precision for CUDA
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # Training loop
    best_val_acc = 0.0
    best_macro_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    training_start_time = time.time()

    print(f"\nTraining for up to {args.epochs} epochs (patience={args.patience})")
    print(f"{'='*70}")

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        # Warmup: lower LR for first few epochs
        if epoch <= args.warmup_epochs:
            warmup_lr = args.lr * epoch / args.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr

        train_loss, train_acc, train_ce, train_oe = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            oe_loader=oe_loader, oe_weight=args.oe_weight,
        )
        val_loss, val_acc, per_class, macro_f1 = validate(
            model,
            val_loader,
            criterion,
            device,
            idx_to_class=idx_to_class,
            num_classes=num_classes,
        )

        if epoch > args.warmup_epochs:
            scheduler.step()

        elapsed = time.time() - start_time
        lr = optimizer.param_groups[0]["lr"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Print epoch results
        oe_part = f" (CE: {train_ce:.4f} OE: {train_oe:.4f})" if oe_loader is not None else ""
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f}{oe_part} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {macro_f1:.4f} | "
              f"LR: {lr:.6f} | {elapsed:.1f}s")

        # Per-class accuracy
        per_class_str = " | ".join(f"{name}: {acc:.3f}" for name, acc in per_class.items())
        print(f"         Per-class: {per_class_str}")

        # Save best model (use macro-F1 as criterion for better imbalanced-class handling)
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "macro_f1": macro_f1,
                "class_to_idx": class_to_idx,
                "idx_to_class": idx_to_class,
                "architecture": args.architecture,
            }
            best_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            print(f"         * Best model saved! (macro_f1={macro_f1:.4f}, val_acc={val_acc:.4f})")
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
                "architecture": args.architecture,
            }, ckpt_path)

    print(f"\n{'='*70}")
    print(f"Training complete! Best macro-F1: {best_macro_f1:.4f} (val_acc: {best_val_acc:.4f})")
    print(f"Best model saved to: {os.path.join(args.output_dir, 'best_model.pth')}")

    class_map_path = os.path.join(args.output_dir, "class_to_idx.json")
    with open(class_map_path, "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2, sort_keys=True)
    print(f"Class map saved to: {class_map_path}")

    # Save training curves
    curves_path = os.path.join(args.plots_dir, "training_curves.png")
    save_training_curves(train_losses, val_losses, train_accs, val_accs, curves_path)

    # Write experiment log
    training_time_seconds = time.time() - training_start_time
    experiment_log = {
        "architecture": args.architecture,
        "num_classes": num_classes,
        "epochs_trained": epoch,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_macro_f1": best_macro_f1,
        "hyperparameters": {
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "patience": args.patience,
            "warmup_epochs": args.warmup_epochs,
            "init_from": args.init_from,
            "oe_csv": args.oe_csv,
            "oe_weight": args.oe_weight if oe_loader is not None else 0.0,
            "oe_pool_size": (len(oe_loader.dataset) if oe_loader is not None else 0),
        },
        "training_time_seconds": round(training_time_seconds, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    log_path = os.path.join(args.output_dir, "experiment_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(experiment_log, f, indent=2)
    print(f"Experiment log saved to: {log_path}")


if __name__ == "__main__":
    main()
