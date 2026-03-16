#!/usr/bin/env python3
"""
Evaluate trained model on test set.

Produces confusion matrix, per-class precision/recall/F1,
and top-1/top-2 accuracy metrics.
"""

import os
import sys
import argparse
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from ml.training.dataset import (
    SpeciesDataset,
    get_idx_to_class,
    get_val_transforms,
)
from ml.training.model import load_trained_model

ML_DIR = os.path.join(PROJECT_ROOT, "ml")
DEFAULT_MODEL_PATH = os.path.join(ML_DIR, "models", "best_model.pth")
DEFAULT_TEST_CSV = os.path.join(ML_DIR, "data", "splits", "test.csv")
DEFAULT_OUTPUT_DIR = os.path.join(ML_DIR, "outputs")


@torch.no_grad()
def evaluate(model, loader, device):
    """Run full evaluation on test set."""
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)

        all_probs.append(probs.cpu())
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.tolist())

    all_probs = torch.cat(all_probs, dim=0)
    return all_preds, all_labels, all_probs


def compute_metrics(preds, labels, idx_to_class: dict[int, str], num_classes: int):
    """Compute precision, recall, F1 per class."""
    metrics = {}

    for idx in range(num_classes):
        class_name = idx_to_class[idx]
        tp = sum(1 for p, l in zip(preds, labels) if p == idx and l == idx)
        fp = sum(1 for p, l in zip(preds, labels) if p == idx and l != idx)
        fn = sum(1 for p, l in zip(preds, labels) if p != idx and l == idx)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(1 for l in labels if l == idx),
        }

    return metrics


def compute_top_k_accuracy(probs, labels, k=2):
    """Compute top-k accuracy."""
    correct = 0
    for prob, label in zip(probs, labels):
        top_k = prob.topk(k).indices.tolist()
        if label in top_k:
            correct += 1
    return correct / len(labels)


def build_confusion_matrix(preds, labels, num_classes: int):
    """Build confusion matrix as 2D list."""
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for pred, label in zip(preds, labels):
        matrix[label][pred] += 1
    return matrix


def save_confusion_matrix(matrix, output_path, idx_to_class: dict[int, str], num_classes: int):
    """Save confusion matrix as image."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        class_names = [idx_to_class[i] for i in range(num_classes)]
        mat = np.array(matrix)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(mat, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set(
            xticks=range(num_classes),
            yticks=range(num_classes),
            xticklabels=class_names,
            yticklabels=class_names,
            ylabel="True Label",
            xlabel="Predicted Label",
            title="Confusion Matrix",
        )

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        # Add text annotations
        thresh = mat.max() / 2.0
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, format(mat[i, j], "d"),
                        ha="center", va="center",
                        color="white" if mat[i, j] > thresh else "black")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Confusion matrix saved to {output_path}")
    except ImportError:
        print("matplotlib not available, printing confusion matrix as text:")
        class_names = [idx_to_class[i] for i in range(num_classes)]
        header = "          " + "  ".join(f"{n:>12}" for n in class_names)
        print(header)
        for i, row in enumerate(matrix):
            row_str = "  ".join(f"{v:>12d}" for v in row)
            print(f"{class_names[i]:>10}  {row_str}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate species classifier on test set")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to trained model checkpoint")
    parser.add_argument("--test-csv", default=DEFAULT_TEST_CSV, help="Path to test split CSV")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for plots")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading model from {args.model}")
    model, class_to_idx = load_trained_model(
        args.model,
        device=str(device),
        return_class_mapping=True,
    )
    model = model.to(device)
    idx_to_class = get_idx_to_class(class_to_idx)
    num_classes = len(class_to_idx)
    print(f"Classes: {num_classes}")

    # Load test data
    print(f"Loading test set from {args.test_csv}")
    test_dataset = SpeciesDataset(
        args.test_csv,
        class_to_idx=class_to_idx,
        transform=get_val_transforms(),
    )
    print(f"Test samples: {len(test_dataset)}")

    if len(test_dataset) == 0:
        print("ERROR: Empty test set!")
        sys.exit(1)

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    # Evaluate
    print("\nRunning evaluation...")
    preds, labels, probs = evaluate(model, test_loader, device)

    # Overall accuracy
    top1_acc = sum(1 for p, l in zip(preds, labels) if p == l) / len(labels)
    top2_acc = compute_top_k_accuracy(probs, labels, k=2)

    print(f"\n{'='*50}")
    print(f"Results:")
    print(f"  Top-1 Accuracy: {top1_acc:.4f} ({top1_acc*100:.1f}%)")
    print(f"  Top-2 Accuracy: {top2_acc:.4f} ({top2_acc*100:.1f}%)")

    # Per-class metrics
    metrics = compute_metrics(preds, labels, idx_to_class=idx_to_class, num_classes=num_classes)
    print(f"\nPer-class metrics:")
    print(f"  {'Class':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print(f"  {'-'*60}")
    for class_name, m in metrics.items():
        print(f"  {class_name:<15} {m['precision']:<12.4f} {m['recall']:<12.4f} "
              f"{m['f1']:<12.4f} {m['support']:<10d}")

    # Confusion matrix
    matrix = build_confusion_matrix(preds, labels, num_classes=num_classes)
    print(f"\nConfusion Matrix:")
    class_names = [idx_to_class[i] for i in range(num_classes)]
    header = "              " + "  ".join(f"{n:>12}" for n in class_names)
    print(header)
    for i, row in enumerate(matrix):
        row_str = "  ".join(f"{v:>12d}" for v in row)
        print(f"  {class_names[i]:>12}  {row_str}")

    # Save confusion matrix plot
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    save_confusion_matrix(
        matrix,
        cm_path,
        idx_to_class=idx_to_class,
        num_classes=num_classes,
    )

    # Summary verdict
    print(f"\n{'='*50}")
    if top1_acc >= 0.85:
        print(f"PASS: Top-1 accuracy {top1_acc:.1%} meets the 85% target")
    else:
        print(f"BELOW TARGET: Top-1 accuracy {top1_acc:.1%} (target: 85%)")
        print("Consider: more training data, more epochs, or a larger model (MobileNetV3-Large)")

    if top2_acc >= 0.95:
        print(f"PASS: Top-2 accuracy {top2_acc:.1%} meets the 95% target")
    else:
        print(f"BELOW TARGET: Top-2 accuracy {top2_acc:.1%} (target: 95%)")


if __name__ == "__main__":
    main()
