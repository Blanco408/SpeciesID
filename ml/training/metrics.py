#!/usr/bin/env python3
"""
Shared metric functions for training and evaluation.

Used by both evaluate.py and train.py to ensure consistent metric computation.
"""


def compute_metrics(preds, labels, idx_to_class: dict[int, str], num_classes: int):
    """Per-class precision, recall, F1, support."""
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
    """Top-k accuracy."""
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


def compute_macro_f1(per_class_metrics):
    """Average F1 across all classes (unweighted)."""
    f1_scores = [m["f1"] for m in per_class_metrics.values()]
    if not f1_scores:
        return 0.0
    return sum(f1_scores) / len(f1_scores)


def compute_weighted_f1(per_class_metrics):
    """Support-weighted F1 across all classes."""
    total_support = sum(m["support"] for m in per_class_metrics.values())
    if total_support == 0:
        return 0.0
    weighted_sum = sum(m["f1"] * m["support"] for m in per_class_metrics.values())
    return weighted_sum / total_support
