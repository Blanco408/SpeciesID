#!/usr/bin/env python3
"""
Evaluate trained model on test set.

Produces confusion matrix, per-class precision/recall/F1,
top-1/top-2/top-3/top-5 accuracy, macro/weighted F1, and saves
all metrics to test_metrics.json.
"""

import os
import sys
import json
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
from ml.training.metrics import (
    compute_metrics,
    compute_top_k_accuracy,
    build_confusion_matrix,
    compute_macro_f1,
    compute_weighted_f1,
    compute_auroc,
    compute_fpr_at_tpr,
    energy_score,
)

NOTHING_CLASS_NAME = "nothing"

ML_DIR = os.path.join(PROJECT_ROOT, "ml")
DEFAULT_MODEL_PATH = os.path.join(ML_DIR, "models", "best_model.pth")
DEFAULT_TEST_CSV = os.path.join(ML_DIR, "data", "splits", "test.csv")
DEFAULT_OUTPUT_DIR = os.path.join(ML_DIR, "outputs")


@torch.no_grad()
def evaluate(model, loader, device, return_logits: bool = False):
    """Run full evaluation on test set.

    If return_logits=True, also returns the raw logits tensor (needed for
    energy-score OOD detection). Existing callers that unpack 3 values keep
    working.
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    all_logits = []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)

        all_probs.append(probs.cpu())
        if return_logits:
            all_logits.append(outputs.cpu())
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.tolist())

    all_probs = torch.cat(all_probs, dim=0)
    if return_logits:
        return all_preds, all_labels, all_probs, torch.cat(all_logits, dim=0)
    return all_preds, all_labels, all_probs


def compute_ood_section(
    *,
    probs: torch.Tensor,
    logits: torch.Tensor,
    labels: list[int],
    class_to_idx: dict[str, int],
) -> dict:
    """Compute the in-scope-vs-nothing block: binary metrics + AUROC + FPR@95TPR.

    Treats every label != nothing_idx as "in-scope" (positive). Three detectors
    are scored: max-softmax over species classes only, energy score, and the
    explicit `nothing` class probability (low = positive).

    Returns an empty dict if the model has no `nothing` class.
    """
    if NOTHING_CLASS_NAME not in class_to_idx:
        return {}

    nothing_idx = class_to_idx[NOTHING_CLASS_NAME]
    species_idx = [i for i in range(probs.shape[1]) if i != nothing_idx]

    # Binary labels: 1 = in-scope species, 0 = nothing
    binary_labels = [0 if l == nothing_idx else 1 for l in labels]

    # Detector 1: max softmax probability over species classes only.
    species_probs = probs[:, species_idx]
    max_species_prob = species_probs.max(dim=1).values.tolist()

    # Detector 2: energy score using ALL logits (closer to 0 = more in-distribution).
    # Negate so that higher = more confident in-scope.
    energies = energy_score(logits, temperature=1.0)
    neg_energy = [-e for e in energies]

    # Detector 3: 1 - P(nothing). Higher = more confident in-scope.
    nothing_probs = probs[:, nothing_idx].tolist()
    one_minus_nothing = [1.0 - p for p in nothing_probs]

    # Confusion matrix on the binary task using argmax-based prediction
    # (predict in-scope unless argmax == nothing_idx).
    pred_in_scope = [(int(p.argmax().item()) != nothing_idx) for p in probs]
    tp = sum(1 for p, l in zip(pred_in_scope, binary_labels) if p and l)
    fp = sum(1 for p, l in zip(pred_in_scope, binary_labels) if p and not l)
    fn = sum(1 for p, l in zip(pred_in_scope, binary_labels) if not p and l)
    tn = sum(1 for p, l in zip(pred_in_scope, binary_labels) if not p and not l)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # OOS rejection rate

    detectors = {
        "max_species_softmax": {
            "auroc": compute_auroc(max_species_prob, binary_labels),
            "fpr_at_95_tpr": compute_fpr_at_tpr(max_species_prob, binary_labels, tpr=0.95),
        },
        "neg_energy": {
            "auroc": compute_auroc(neg_energy, binary_labels),
            "fpr_at_95_tpr": compute_fpr_at_tpr(neg_energy, binary_labels, tpr=0.95),
        },
        "one_minus_nothing_prob": {
            "auroc": compute_auroc(one_minus_nothing, binary_labels),
            "fpr_at_95_tpr": compute_fpr_at_tpr(one_minus_nothing, binary_labels, tpr=0.95),
        },
    }

    return {
        "binary_in_scope_vs_nothing": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "specificity_oos_rejection": round(specificity, 4),
            "support_in_scope": tp + fn,
            "support_nothing": tn + fp,
        },
        "ood_detectors": {
            name: {k: round(v, 4) for k, v in metrics.items()}
            for name, metrics in detectors.items()
        },
    }


def save_confusion_matrix(matrix, output_path, idx_to_class: dict[int, str], num_classes: int):
    """Save confusion matrix as image."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        class_names = [idx_to_class[i] for i in range(num_classes)]
        mat = np.array(matrix)

        fig_w = max(12, num_classes * 0.3)
        fig_h = max(10, num_classes * 0.25)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
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

        tick_fontsize = 6 if num_classes > 25 else 10
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=tick_fontsize)
        plt.setp(ax.get_yticklabels(), fontsize=tick_fontsize)

        # Add text annotations (skip for large class counts to avoid clutter)
        if num_classes <= 25:
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


def save_f1_bar_chart(per_class_metrics, output_path):
    """Save horizontal bar chart of per-class F1 scores."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = list(per_class_metrics.keys())
        f1_scores = [per_class_metrics[n]["f1"] for n in names]

        # Sort by F1 ascending so worst classes are at bottom visually
        sorted_pairs = sorted(zip(names, f1_scores), key=lambda x: x[1])
        names = [p[0] for p in sorted_pairs]
        f1_scores = [p[1] for p in sorted_pairs]

        fig_h = max(4, len(names) * 0.35)
        fig, ax = plt.subplots(figsize=(10, fig_h))
        bars = ax.barh(names, f1_scores, color="steelblue")
        ax.set_xlabel("F1 Score")
        ax.set_title("Per-Class F1 Score")
        ax.set_xlim(0, 1.0)

        # Add value labels
        for bar, score in zip(bars, f1_scores):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{score:.3f}", va="center", fontsize=8)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"F1 bar chart saved to {output_path}")
    except ImportError:
        print("matplotlib not available, skipping F1 bar chart")


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

    # Evaluate (collect logits too so we can compute energy-based OOD scores)
    print("\nRunning evaluation...")
    preds, labels, probs, logits = evaluate(model, test_loader, device, return_logits=True)

    # Overall accuracy
    top1_acc = sum(1 for p, l in zip(preds, labels) if p == l) / len(labels)
    top2_acc = compute_top_k_accuracy(probs, labels, k=2)
    top3_acc = compute_top_k_accuracy(probs, labels, k=min(3, num_classes))
    top5_acc = compute_top_k_accuracy(probs, labels, k=min(5, num_classes))

    print(f"\n{'='*50}")
    print(f"Results:")
    print(f"  Top-1 Accuracy: {top1_acc:.4f} ({top1_acc*100:.1f}%)")
    print(f"  Top-2 Accuracy: {top2_acc:.4f} ({top2_acc*100:.1f}%)")
    print(f"  Top-3 Accuracy: {top3_acc:.4f} ({top3_acc*100:.1f}%)")
    print(f"  Top-5 Accuracy: {top5_acc:.4f} ({top5_acc*100:.1f}%)")

    # Per-class metrics
    metrics = compute_metrics(preds, labels, idx_to_class=idx_to_class, num_classes=num_classes)
    macro_f1 = compute_macro_f1(metrics)
    weighted_f1 = compute_weighted_f1(metrics)

    print(f"\n  Macro F1:    {macro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")

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

    # Save per-class F1 bar chart
    save_f1_bar_chart(metrics, os.path.join(args.output_dir, "f1_per_class.png"))

    # OOD section: only emits if the model has a `nothing` class.
    ood_section = compute_ood_section(
        probs=probs, logits=logits, labels=labels, class_to_idx=class_to_idx
    )
    if ood_section:
        print(f"\n{'='*50}")
        print("In-scope vs nothing:")
        b = ood_section["binary_in_scope_vs_nothing"]
        print(f"  Precision: {b['precision']:.4f}  Recall: {b['recall']:.4f}  F1: {b['f1']:.4f}")
        print(f"  Specificity (OOS rejection): {b['specificity_oos_rejection']:.4f}")
        print(f"  Support: {b['support_in_scope']} in-scope, {b['support_nothing']} nothing")
        print("OOD detectors:")
        for name, m in ood_section["ood_detectors"].items():
            print(f"  {name:>26}  AUROC={m['auroc']:.4f}  FPR@95TPR={m['fpr_at_95_tpr']:.4f}")

    # Save all metrics to JSON
    metrics_json = {
        "top1_accuracy": round(top1_acc, 4),
        "top2_accuracy": round(top2_acc, 4),
        "top3_accuracy": round(top3_acc, 4),
        "top5_accuracy": round(top5_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": metrics,
    }
    if ood_section:
        metrics_json["ood"] = ood_section
    metrics_path = os.path.join(args.output_dir, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

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
