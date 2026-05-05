#!/usr/bin/env python3
"""
Standalone OOD-evaluation harness.

Inputs:
  --model            checkpoint produced by ml/training/train.py
  --in-scope-csv     CSV with columns at least image_path,class_label
                     (typically ml/data/splits/test.csv after the nothing-class
                      retraining)
  --ood-manifest     One of:
                       - ml/data/negatives_manifest.csv (build_negative_dataset.py)
                       - ml/data/fresh_testset/out_of_scope_manifest.csv
                         (build_fresh_testset.py)
                     Either format works; we look for an `image_path` column
                     and fall back to `local_path`.

Outputs (under ml/outputs/<experiment>/):
  ood_report.json          all numbers, including per-detector AUROC/FPR and
                           the threshold-sweep table
  thresholds.json          the iOS-bundled file with the chosen operating
                           point (max min(TPR,TNR) s.t. TPR >= --target-tpr)
  ood_threshold_scatter.png   TPR vs TNR for each combo
  ood_per_class_oos.png       per-OOS-source rejection rate

The threshold sweep simulates the production decision logic exactly:
  predict_in_scope = (argmax_class != "nothing")
                     AND (top_softmax >= minimumDetectionConfidence)
                     AND (top_softmax - second_softmax >= minimumTopMargin)
                     AND (entropy / log(K) <= maxEntropyRatio)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.training.dataset import get_val_transforms, get_idx_to_class
from ml.training.model import load_trained_model
from ml.training.metrics import (
    compute_auroc,
    compute_fpr_at_tpr,
    energy_score,
)

NOTHING_CLASS_NAME = "nothing"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "best_model.pth"
DEFAULT_IN_SCOPE_CSV = PROJECT_ROOT / "ml" / "data" / "splits" / "test.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "ml" / "outputs" / "ood_eval"


@dataclass
class ImageItem:
    path: str
    is_in_scope: bool
    source_label: str  # class label for in-scope, source_taxon/class_label for OOS


class _ImageListDataset(Dataset):
    def __init__(self, items: list[ImageItem], transform):
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        img = Image.open(item.path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, idx


def _load_in_scope_items(csv_path: Path) -> list[ImageItem]:
    items: list[ImageItem] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = (row.get("class_label") or "").strip()
            if label == NOTHING_CLASS_NAME or not label:
                continue  # in-scope CSV may include nothing rows; skip them here
            path = (row.get("image_path") or row.get("local_path") or "").strip()
            if path and os.path.exists(path):
                items.append(ImageItem(path=path, is_in_scope=True, source_label=label))
    return items


def _load_ood_items(csv_path: Path) -> list[ImageItem]:
    items: list[ImageItem] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = (row.get("image_path") or row.get("local_path") or "").strip()
            if not path or not os.path.exists(path):
                continue
            source = (
                row.get("source_taxon")
                or row.get("class_label")
                or row.get("subset")
                or "ood"
            ).strip()
            items.append(ImageItem(path=path, is_in_scope=False, source_label=source))
    return items


@torch.no_grad()
def _run_model(model, items: list[ImageItem], device, batch_size: int, num_workers: int):
    """Forward pass on a list of items. Returns (logits, probs, item_order)."""
    dataset = _ImageListDataset(items, transform=get_val_transforms())
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    logits_chunks: list[torch.Tensor] = []
    probs_chunks: list[torch.Tensor] = []
    order: list[int] = []
    for images, idxs in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        logits_chunks.append(logits.cpu())
        probs_chunks.append(probs.cpu())
        order.extend(idxs.tolist())
    return torch.cat(logits_chunks, dim=0), torch.cat(probs_chunks, dim=0), order


def _entropy_ratio(probs_row: torch.Tensor) -> float:
    """Normalized entropy: entropy(p) / log(K). 0 = certain, 1 = uniform."""
    p = probs_row.clamp_min(1e-12)
    h = -(p * p.log()).sum().item()
    k = p.numel()
    if k <= 1:
        return 0.0
    return h / math.log(k)


def _threshold_sweep(
    *,
    in_scope_logits: torch.Tensor,
    in_scope_probs: torch.Tensor,
    ood_logits: torch.Tensor,
    ood_probs: torch.Tensor,
    nothing_idx: Optional[int],
    confidence_grid: list[float],
    margin_grid: list[float],
    entropy_grid: list[float],
    target_tpr: float,
):
    """Walk every (conf, margin, entropy) combo. Return list of dicts."""

    def _accept_fraction(probs: torch.Tensor, conf: float, margin: float, entropy: float) -> float:
        accepted = 0
        for row in probs:
            top2 = torch.topk(row, k=min(2, row.numel())).indices.tolist()
            top_idx = top2[0]
            if nothing_idx is not None and top_idx == nothing_idx:
                continue  # explicit reject by argmax
            top_p = row[top_idx].item()
            second_p = row[top2[1]].item() if len(top2) > 1 else 0.0
            if top_p < conf:
                continue
            if (top_p - second_p) < margin:
                continue
            if _entropy_ratio(row) > entropy:
                continue
            accepted += 1
        return accepted / max(1, probs.shape[0])

    rows = []
    for conf in confidence_grid:
        for margin in margin_grid:
            for entropy in entropy_grid:
                tpr = _accept_fraction(in_scope_probs, conf, margin, entropy)
                # TNR: fraction of OOD examples that we did NOT accept
                ood_accept = _accept_fraction(ood_probs, conf, margin, entropy)
                tnr = 1.0 - ood_accept
                rows.append(
                    {
                        "minimumDetectionConfidence": round(conf, 3),
                        "minimumTopMargin": round(margin, 3),
                        "maxEntropyRatio": round(entropy, 3),
                        "tpr_in_scope": round(tpr, 4),
                        "tnr_oos": round(tnr, 4),
                        "min_tpr_tnr": round(min(tpr, tnr), 4),
                    }
                )

    # Pick the combo maximizing min(TPR, TNR) subject to TPR >= target_tpr.
    feasible = [r for r in rows if r["tpr_in_scope"] >= target_tpr]
    if feasible:
        chosen = max(feasible, key=lambda r: (r["min_tpr_tnr"], r["tnr_oos"]))
        chosen_reason = f"max min(TPR,TNR) s.t. TPR>={target_tpr}"
    else:
        # Fall back to the absolute best min(TPR,TNR) and warn.
        chosen = max(rows, key=lambda r: r["min_tpr_tnr"])
        chosen_reason = f"NO combo met TPR>={target_tpr}; picked global best min(TPR,TNR)"

    return rows, chosen, chosen_reason


def _energy_threshold(
    *,
    in_scope_logits: torch.Tensor,
    ood_logits: torch.Tensor,
    target_tpr: float,
):
    """Pick an energy threshold (lower-energy = in-scope) hitting target TPR.

    Returns the threshold E such that fraction of in-scope examples with
    energy <= E is >= target_tpr. We report the resulting OOS TNR there.
    """
    in_e = sorted(energy_score(in_scope_logits))
    ood_e = energy_score(ood_logits)
    n_in = len(in_e)
    if n_in == 0:
        return None, 0.0
    keep_idx = max(0, min(n_in - 1, int(target_tpr * n_in) - 1))
    threshold = in_e[keep_idx]
    tnr = sum(1 for e in ood_e if e > threshold) / max(1, len(ood_e))
    return threshold, tnr


def _per_class_oos_rejection(
    *,
    items: list[ImageItem],
    probs: torch.Tensor,
    nothing_idx: Optional[int],
    chosen: dict,
):
    """Group OOD items by source_label and compute rejection rate at chosen op-point."""
    by_source: dict[str, list[int]] = {}
    for i, item in enumerate(items):
        by_source.setdefault(item.source_label, []).append(i)

    out: dict[str, dict] = {}
    conf = chosen["minimumDetectionConfidence"]
    margin = chosen["minimumTopMargin"]
    entropy_max = chosen["maxEntropyRatio"]

    for source, idxs in by_source.items():
        rejected = 0
        for i in idxs:
            row = probs[i]
            top2 = torch.topk(row, k=min(2, row.numel())).indices.tolist()
            top_idx = top2[0]
            top_p = row[top_idx].item()
            second_p = row[top2[1]].item() if len(top2) > 1 else 0.0
            if nothing_idx is not None and top_idx == nothing_idx:
                rejected += 1
                continue
            if top_p < conf or (top_p - second_p) < margin or _entropy_ratio(row) > entropy_max:
                rejected += 1
        out[source] = {
            "n": len(idxs),
            "rejected": rejected,
            "rejection_rate": round(rejected / max(1, len(idxs)), 4),
        }
    return out


def _save_threshold_scatter(rows: list[dict], chosen: dict, output_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping scatter plot")
        return

    xs = [r["tpr_in_scope"] for r in rows]
    ys = [r["tnr_oos"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(xs, ys, alpha=0.5, s=15, label="threshold combos")
    ax.scatter(
        [chosen["tpr_in_scope"]], [chosen["tnr_oos"]],
        color="red", s=80, label="chosen op-point", zorder=10,
    )
    ax.set_xlabel("In-scope TPR (recall)")
    ax.set_ylabel("OOS TNR (specificity)")
    ax.set_title("Threshold sweep")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _save_per_class_oos_plot(per_source: dict, output_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    items = sorted(per_source.items(), key=lambda kv: kv[1]["rejection_rate"])
    names = [k for k, _ in items]
    rates = [v["rejection_rate"] for _, v in items]
    fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.3)))
    ax.barh(names, rates, color="steelblue")
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Rejection rate (1 = always rejected)")
    ax.set_title("Per-OOS-source rejection at chosen op-point")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OOD evaluation + threshold calibration")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--in-scope-csv", type=Path, default=DEFAULT_IN_SCOPE_CSV)
    parser.add_argument("--ood-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--target-tpr", type=float, default=0.95)
    parser.add_argument(
        "--confidence-grid",
        type=str,
        default="0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80",
    )
    parser.add_argument(
        "--margin-grid",
        type=str,
        default="0.05,0.10,0.15,0.20",
    )
    parser.add_argument(
        "--entropy-grid",
        type=str,
        default="0.50,0.60,0.70,0.80",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    print(f"Loading model from {args.model}")
    model, class_to_idx = load_trained_model(
        str(args.model), device=str(device), return_class_mapping=True
    )
    model = model.to(device).eval()
    idx_to_class = get_idx_to_class(class_to_idx)
    nothing_idx = class_to_idx.get(NOTHING_CLASS_NAME)
    if nothing_idx is None:
        print(
            "WARN: model has no 'nothing' class. The threshold sweep will still "
            "calibrate confidence/margin/entropy but the explicit-abstain detector "
            "won't be available.",
            file=sys.stderr,
        )

    in_items = _load_in_scope_items(args.in_scope_csv)
    ood_items = _load_ood_items(args.ood_manifest)
    print(f"In-scope images: {len(in_items)}")
    print(f"OOD images:      {len(ood_items)}")
    if not in_items or not ood_items:
        print("ERROR: need both in-scope and OOD items.")
        sys.exit(1)

    print("Forward pass on in-scope set...")
    in_logits, in_probs, _ = _run_model(model, in_items, device, args.batch_size, args.num_workers)
    print("Forward pass on OOD set...")
    ood_logits, ood_probs, ood_order = _run_model(model, ood_items, device, args.batch_size, args.num_workers)
    # Reorder ood_items so positions in ood_probs match items
    ood_items_ordered = [ood_items[i] for i in ood_order]

    # Detector AUROC / FPR@95TPR
    binary_labels = [1] * in_logits.shape[0] + [0] * ood_logits.shape[0]

    # Detector A: max softmax over species classes only
    species_idx = [i for i in range(in_probs.shape[1]) if i != nothing_idx]
    in_max_species = in_probs[:, species_idx].max(dim=1).values.tolist()
    ood_max_species = ood_probs[:, species_idx].max(dim=1).values.tolist()

    # Detector B: -energy (higher = in-scope)
    in_neg_e = [-e for e in energy_score(in_logits)]
    ood_neg_e = [-e for e in energy_score(ood_logits)]

    # Detector C: 1 - p(nothing); only meaningful when nothing class exists.
    if nothing_idx is not None:
        in_one_minus_nothing = (1.0 - in_probs[:, nothing_idx]).tolist()
        ood_one_minus_nothing = (1.0 - ood_probs[:, nothing_idx]).tolist()
    else:
        in_one_minus_nothing = ood_one_minus_nothing = []

    detectors = {
        "max_species_softmax": {
            "auroc": round(compute_auroc(in_max_species + ood_max_species, binary_labels), 4),
            "fpr_at_95_tpr": round(
                compute_fpr_at_tpr(in_max_species + ood_max_species, binary_labels, tpr=args.target_tpr), 4
            ),
        },
        "neg_energy": {
            "auroc": round(compute_auroc(in_neg_e + ood_neg_e, binary_labels), 4),
            "fpr_at_95_tpr": round(
                compute_fpr_at_tpr(in_neg_e + ood_neg_e, binary_labels, tpr=args.target_tpr), 4
            ),
        },
    }
    if nothing_idx is not None:
        detectors["one_minus_nothing_prob"] = {
            "auroc": round(compute_auroc(in_one_minus_nothing + ood_one_minus_nothing, binary_labels), 4),
            "fpr_at_95_tpr": round(
                compute_fpr_at_tpr(
                    in_one_minus_nothing + ood_one_minus_nothing, binary_labels, tpr=args.target_tpr
                ),
                4,
            ),
        }

    # Threshold sweep on the iOS-style decision logic
    confidence_grid = [float(x) for x in args.confidence_grid.split(",") if x.strip()]
    margin_grid = [float(x) for x in args.margin_grid.split(",") if x.strip()]
    entropy_grid = [float(x) for x in args.entropy_grid.split(",") if x.strip()]
    sweep_rows, chosen, chosen_reason = _threshold_sweep(
        in_scope_logits=in_logits,
        in_scope_probs=in_probs,
        ood_logits=ood_logits,
        ood_probs=ood_probs,
        nothing_idx=nothing_idx,
        confidence_grid=confidence_grid,
        margin_grid=margin_grid,
        entropy_grid=entropy_grid,
        target_tpr=args.target_tpr,
    )

    # Energy threshold (separate scalar)
    energy_thresh, energy_tnr = _energy_threshold(
        in_scope_logits=in_logits,
        ood_logits=ood_logits,
        target_tpr=args.target_tpr,
    )

    # Per-OOS-source rejection at chosen op-point
    per_source = _per_class_oos_rejection(
        items=ood_items_ordered,
        probs=ood_probs,
        nothing_idx=nothing_idx,
        chosen=chosen,
    )

    # Persist results
    report = {
        "model_path": str(args.model),
        "in_scope_csv": str(args.in_scope_csv),
        "ood_manifest": str(args.ood_manifest),
        "n_in_scope": len(in_items),
        "n_ood": len(ood_items),
        "target_tpr": args.target_tpr,
        "ood_detectors": detectors,
        "energy_threshold": {
            "value": (round(energy_thresh, 4) if energy_thresh is not None else None),
            "tnr_at_target_tpr": round(energy_tnr, 4),
        },
        "chosen_op_point": chosen,
        "chosen_reason": chosen_reason,
        "threshold_sweep": sweep_rows,
        "per_oos_source_rejection": per_source,
    }
    report_path = args.output_dir / "ood_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport: {report_path}")

    # iOS-bundled thresholds
    thresholds_payload = {
        "version": "1.0",
        "minimumDetectionConfidence": chosen["minimumDetectionConfidence"],
        "minimumTopMargin": chosen["minimumTopMargin"],
        "maxEntropyRatio": chosen["maxEntropyRatio"],
        "energyThreshold": (round(energy_thresh, 4) if energy_thresh is not None else None),
        "nothingClassId": NOTHING_CLASS_NAME,
        "calibration": {
            "target_tpr": args.target_tpr,
            "achieved_tpr": chosen["tpr_in_scope"],
            "achieved_tnr": chosen["tnr_oos"],
            "n_in_scope": len(in_items),
            "n_ood": len(ood_items),
        },
    }
    thresholds_path = args.output_dir / "thresholds.json"
    with thresholds_path.open("w", encoding="utf-8") as f:
        json.dump(thresholds_payload, f, indent=2)
    print(f"Thresholds: {thresholds_path}")

    # Plots
    _save_threshold_scatter(sweep_rows, chosen, args.output_dir / "ood_threshold_scatter.png")
    _save_per_class_oos_plot(per_source, args.output_dir / "ood_per_class_oos.png")

    # Console summary
    print("\n=== Summary ===")
    print(f"Detectors:")
    for name, m in detectors.items():
        print(f"  {name:>26}  AUROC={m['auroc']:.4f}  FPR@{int(args.target_tpr*100)}TPR={m['fpr_at_95_tpr']:.4f}")
    print(f"\nChosen op-point ({chosen_reason}):")
    print(f"  conf >= {chosen['minimumDetectionConfidence']}")
    print(f"  margin >= {chosen['minimumTopMargin']}")
    print(f"  entropy/log(K) <= {chosen['maxEntropyRatio']}")
    print(f"  -> in-scope TPR = {chosen['tpr_in_scope']}, OOS TNR = {chosen['tnr_oos']}")
    if energy_thresh is not None:
        print(f"  Energy threshold = {energy_thresh:.4f} (OOS TNR at target TPR: {energy_tnr:.4f})")


if __name__ == "__main__":
    main()
