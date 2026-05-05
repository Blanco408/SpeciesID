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


def compute_auroc(scores, labels):
    """Area under ROC curve for a binary detector.

    Args:
        scores: iterable of floats. Higher = more confident the example is positive.
        labels: iterable of 0/1 (or False/True). Same length as scores.

    Returns:
        AUROC in [0, 1]. Returns 0.5 if either class is missing.

    Implementation note: this is the "probability that a random positive scores
    higher than a random negative" definition, computed via rank statistics so
    we don't need scipy/sklearn at runtime.
    """
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    n_pos = sum(1 for _, l in pairs if l)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Average rank of positives, handling ties.
    # ranks are 1-indexed.
    rank_sum = 0.0
    i = 0
    n = len(pairs)
    while i < n:
        j = i
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        # Tied group is pairs[i:j]; assign each member the average rank.
        avg_rank = (i + 1 + j) / 2.0  # average of (i+1) ... j inclusive
        for k in range(i, j):
            if pairs[k][1]:
                rank_sum += avg_rank
        i = j

    # Mann-Whitney U formulation.
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def compute_fpr_at_tpr(scores, labels, tpr: float = 0.95):
    """False positive rate at the threshold that achieves the requested TPR.

    Standard OOD-detection metric (lower is better). `scores` should be the
    detector's "this is positive (in-distribution)" score.

    Args:
        scores: iterable of floats.
        labels: iterable of 0/1; 1 = positive (in-distribution).
        tpr: target true-positive rate (default 0.95).

    Returns:
        FPR in [0, 1]. Returns 1.0 if there are no negatives or the target TPR
        is unreachable.
    """
    pos_scores = sorted((s for s, l in zip(scores, labels) if l), reverse=False)
    neg_scores = sorted((s for s, l in zip(scores, labels) if not l), reverse=False)
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return 1.0

    # Threshold: the smallest score such that >= tpr fraction of positives are >= threshold.
    # Equivalent to picking the (1 - tpr) quantile of positive scores.
    target_excluded = int(n_pos * (1.0 - tpr))
    if target_excluded >= n_pos:
        target_excluded = n_pos - 1
    threshold = pos_scores[target_excluded]

    # FPR = fraction of negatives with score >= threshold.
    # Use binary search rather than scanning to keep this tractable on big runs.
    import bisect
    idx = bisect.bisect_left(neg_scores, threshold)
    fp = n_neg - idx
    return fp / n_neg


def energy_score(logits, temperature: float = 1.0):
    """Free-energy OOD score: -T * logsumexp(logits / T).

    Lower energy = more in-distribution. Multiply by -1 to use as an
    "in-distribution confidence" score (higher = more confident).

    Accepts either a torch.Tensor of shape [N, K] or a list of lists.
    Returns a list of floats.
    """
    try:
        import torch
        if not isinstance(logits, torch.Tensor):
            logits = torch.tensor(logits, dtype=torch.float32)
        scaled = logits / max(temperature, 1e-6)
        e = -temperature * torch.logsumexp(scaled, dim=1)
        return e.tolist()
    except ImportError:
        # Pure-Python fallback for tooling that doesn't depend on torch.
        import math
        out = []
        for row in logits:
            scaled = [v / max(temperature, 1e-6) for v in row]
            m = max(scaled)
            lse = m + math.log(sum(math.exp(s - m) for s in scaled))
            out.append(-temperature * lse)
        return out
