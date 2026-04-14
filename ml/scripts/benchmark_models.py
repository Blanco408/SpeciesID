#!/usr/bin/env python3
"""
Benchmark and compare multiple trained models.

Evaluates accuracy, parameter count, model size, and inference latency
for each checkpoint, producing a comparison table.

Usage:
    python -m ml.scripts.benchmark_models \
        --checkpoints ml/models/exp_mnv3l/best_model.pth ml/models/exp_effb0/best_model.pth \
        --test-csv ml/data/splits/test.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from ml.training.dataset import SpeciesDataset, get_idx_to_class, get_val_transforms
from ml.training.model import load_trained_model
from ml.training.metrics import (
    compute_metrics,
    compute_top_k_accuracy,
    compute_macro_f1,
    compute_weighted_f1,
)
from ml.training.evaluate import evaluate

ML_DIR = os.path.join(PROJECT_ROOT, "ml")
DEFAULT_TEST_CSV = os.path.join(ML_DIR, "data", "splits", "test.csv")


def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_latency(
    model: torch.nn.Module,
    device: torch.device,
    num_runs: int = 100,
    input_size: tuple = (1, 3, 224, 224),
) -> float:
    """Measure average inference latency in milliseconds on CPU."""
    model.eval()
    # Always benchmark on CPU for fair comparison
    model_cpu = model.to("cpu")
    dummy_input = torch.randn(*input_size)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model_cpu(dummy_input)

    # Timed runs
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            model_cpu(dummy_input)
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / num_runs) * 1000
    model.to(device)
    return avg_ms


def estimate_checkpoint_size(checkpoint_path: str) -> float:
    """Get checkpoint file size in MB."""
    return os.path.getsize(checkpoint_path) / (1024 * 1024)


def benchmark_model(
    checkpoint_path: str,
    test_loader: DataLoader,
    class_to_idx: dict,
    device: torch.device,
) -> dict:
    """Benchmark a single model checkpoint."""
    idx_to_class = get_idx_to_class(class_to_idx)
    num_classes = len(class_to_idx)

    # Load model
    model, loaded_class_to_idx = load_trained_model(
        checkpoint_path,
        device=str(device),
        return_class_mapping=True,
    )
    model = model.to(device)

    # Read architecture from checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    architecture = checkpoint.get("architecture", "mobilenet_v3_small")
    best_epoch = checkpoint.get("epoch", "?")

    # Parameter count
    params = count_parameters(model)

    # Checkpoint size
    ckpt_size_mb = estimate_checkpoint_size(checkpoint_path)

    # Inference latency
    latency_ms = measure_inference_latency(model, device)

    # Evaluate on test set
    preds, labels, probs = evaluate(model, test_loader, device)

    top1_acc = sum(1 for p, l in zip(preds, labels) if p == l) / len(labels)
    top2_acc = compute_top_k_accuracy(probs, labels, k=2)
    top3_acc = compute_top_k_accuracy(probs, labels, k=3)
    top5_acc = compute_top_k_accuracy(probs, labels, k=min(5, num_classes))

    per_class = compute_metrics(preds, labels, idx_to_class, num_classes)
    macro_f1 = compute_macro_f1(per_class)
    weighted_f1 = compute_weighted_f1(per_class)

    return {
        "checkpoint": checkpoint_path,
        "architecture": architecture,
        "best_epoch": best_epoch,
        "params": params,
        "params_m": params / 1e6,
        "ckpt_size_mb": ckpt_size_mb,
        "latency_ms": latency_ms,
        "top1_acc": top1_acc,
        "top2_acc": top2_acc,
        "top3_acc": top3_acc,
        "top5_acc": top5_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "num_classes": num_classes,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark and compare trained models")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Paths to model checkpoints to compare")
    parser.add_argument("--test-csv", default=DEFAULT_TEST_CSV,
                        help="Path to test split CSV")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", default=None,
                        help="Optional JSON output path for results")
    args = parser.parse_args()

    if not os.path.exists(args.test_csv):
        print(f"ERROR: Test CSV not found at {args.test_csv}")
        sys.exit(1)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load test data using first checkpoint's class mapping
    first_checkpoint = torch.load(args.checkpoints[0], map_location="cpu", weights_only=True)
    class_to_idx = first_checkpoint.get("class_to_idx", {})
    if not class_to_idx and first_checkpoint.get("idx_to_class"):
        class_to_idx = {str(v): int(k) for k, v in first_checkpoint["idx_to_class"].items()}

    test_dataset = SpeciesDataset(
        args.test_csv,
        class_to_idx=class_to_idx,
        transform=get_val_transforms(),
    )
    print(f"Test samples: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Benchmark each model
    results = []
    for ckpt_path in args.checkpoints:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {ckpt_path}")
        print(f"{'='*60}")

        result = benchmark_model(ckpt_path, test_loader, class_to_idx, device)
        results.append(result)

        print(f"  Architecture: {result['architecture']}")
        print(f"  Parameters: {result['params_m']:.1f}M")
        print(f"  Checkpoint size: {result['ckpt_size_mb']:.1f} MB")
        print(f"  Inference latency: {result['latency_ms']:.1f} ms (CPU)")
        print(f"  Top-1 Accuracy: {result['top1_acc']:.1%}")
        print(f"  Top-3 Accuracy: {result['top3_acc']:.1%}")
        print(f"  Macro F1: {result['macro_f1']:.4f}")

    # Print comparison table
    print(f"\n{'='*100}")
    print("COMPARISON TABLE")
    print(f"{'='*100}")

    header = (
        f"{'Architecture':<25} {'Params':<10} {'Ckpt MB':<10} "
        f"{'Latency':<12} {'Top-1':<10} {'Top-3':<10} {'Top-5':<10} "
        f"{'Macro F1':<10} {'W. F1':<10}"
    )
    print(header)
    print("-" * 100)

    for r in results:
        row = (
            f"{r['architecture']:<25} {r['params_m']:<10.1f} {r['ckpt_size_mb']:<10.1f} "
            f"{r['latency_ms']:<12.1f} {r['top1_acc']:<10.1%} {r['top3_acc']:<10.1%} "
            f"{r['top5_acc']:<10.1%} {r['macro_f1']:<10.4f} {r['weighted_f1']:<10.4f}"
        )
        print(row)

    # Recommendation
    best = max(results, key=lambda r: r["macro_f1"])
    print(f"\nRECOMMENDATION: {best['architecture']} (macro F1: {best['macro_f1']:.4f})")

    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
