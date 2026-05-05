#!/usr/bin/env python3
"""
Build the 'nothing' / out-of-scope dataset used to teach the model to abstain.

Downloads three negative subsets via the iNaturalist fetcher used elsewhere in
the project, then emits two manifests:

  ml/data/negatives_manifest.csv  - rows that will be split into train/val/test
                                    as the `nothing` class.
  ml/data/negatives_oe.csv        - disjoint hold-out used as the Outlier
                                    Exposure auxiliary signal during training.

Subsets:
  oos          - marine species the classifier doesn't support
                 (config: ml/config/out_of_scope_marine_species_10.json)
  background   - marine substrate / kelp / algae habitat shots without a
                 focal animal (config: ml/config/negatives_backgrounds.json)
  non_marine   - terrestrial life and household-style scenes
                 (config: ml/config/negatives_non_marine.json)

Reuses download_taxa_dataset.{resolve_class_configs, fetch_observations,
download_and_save_image}.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from download_taxa_dataset import (
    download_and_save_image,
    fetch_observations,
    resolve_class_configs,
)

ML_DIR = PROJECT_ROOT / "ml"
DEFAULT_OOS_CONFIG = ML_DIR / "config" / "out_of_scope_marine_species_10.json"
DEFAULT_BG_CONFIG = ML_DIR / "config" / "negatives_backgrounds.json"
DEFAULT_NM_CONFIG = ML_DIR / "config" / "negatives_non_marine.json"
DEFAULT_OUTPUT_DIR = ML_DIR / "data" / "negatives"
DEFAULT_TRAIN_MANIFEST = ML_DIR / "data" / "negatives_manifest.csv"
DEFAULT_OE_MANIFEST = ML_DIR / "data" / "negatives_oe.csv"


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _download_subset(
    *,
    subset: str,
    config_path: Path,
    output_dir: Path,
    workers: int,
    per_class_cap: int,
) -> list[dict[str, str]]:
    """Resolve taxa, fetch observations, download images. Returns image rows."""
    if not config_path.exists():
        print(f"  WARN: config missing for subset '{subset}': {config_path}")
        return []

    config = _load_config(config_path)
    quality_grade = config.get("quality_grade", "research")
    require_geotag = bool(config.get("require_geotag", False))

    print(f"\n[{subset}] resolving classes from {config_path.name}")
    resolved = resolve_class_configs(
        class_configs=config.get("classes", []),
        default_max_images=per_class_cap,
        max_images_cap=per_class_cap,
    )

    download_tasks: list[dict[str, str]] = []
    existing_rows: list[dict[str, str]] = []

    for cls in resolved:
        print(f"[{subset}] fetching {cls.label} (taxon={cls.taxon_id}, n<={cls.max_images})")
        try:
            obs_rows = fetch_observations(
                taxon_id=cls.taxon_id,
                max_images=cls.max_images,
                quality_grade=quality_grade,
                require_geotag=require_geotag,
            )
        except Exception as exc:
            print(f"  ! fetch failed for {cls.label}: {exc}")
            continue

        subset_dir = output_dir / subset / cls.label
        subset_dir.mkdir(parents=True, exist_ok=True)

        for obs in obs_rows:
            obs_id = obs["observation_id"]
            local_path = subset_dir / f"{obs_id}.jpg"
            row = {
                "image_path": str(local_path),
                "subset": subset,
                "source_taxon": cls.label,
                "observation_id": obs_id,
                "image_url": obs["image_url"],
            }
            if local_path.exists():
                existing_rows.append(row)
            else:
                download_tasks.append(row)

    if download_tasks:
        print(f"[{subset}] downloading {len(download_tasks)} new images...")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    download_and_save_image, row["image_url"], row["image_path"]
                ): row
                for row in download_tasks
            }
            with tqdm(total=len(futures), desc=f"download[{subset}]") as pbar:
                for future in as_completed(futures):
                    row = futures[future]
                    if future.result():
                        existing_rows.append(row)
                    pbar.update(1)

    # Final sanity filter: only keep rows whose file actually exists
    final = [r for r in existing_rows if Path(r["image_path"]).exists()]
    print(f"[{subset}] {len(final)} images on disk")
    return final


def _split_train_oe(
    rows: list[dict[str, str]], oe_fraction: float, seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Stratified split per (subset, source_taxon) into train and OE pools."""
    rng = random.Random(seed)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["subset"], row["source_taxon"])
        grouped.setdefault(key, []).append(row)

    train_rows: list[dict[str, str]] = []
    oe_rows: list[dict[str, str]] = []

    for _, group in grouped.items():
        rng.shuffle(group)
        oe_count = max(1, int(len(group) * oe_fraction)) if len(group) > 4 else 0
        oe_rows.extend(group[:oe_count])
        train_rows.extend(group[oe_count:])

    rng.shuffle(train_rows)
    rng.shuffle(oe_rows)
    return train_rows, oe_rows


def _write_manifest(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_path", "subset", "source_taxon", "observation_id"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image_path": row["image_path"],
                    "subset": row["subset"],
                    "source_taxon": row["source_taxon"],
                    "observation_id": row.get("observation_id", ""),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download negative/abstain training data")
    parser.add_argument("--out-of-scope-config", type=Path, default=DEFAULT_OOS_CONFIG)
    parser.add_argument("--backgrounds-config", type=Path, default=DEFAULT_BG_CONFIG)
    parser.add_argument("--non-marine-config", type=Path, default=DEFAULT_NM_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--oe-manifest", type=Path, default=DEFAULT_OE_MANIFEST)
    parser.add_argument(
        "--per-class-cap",
        type=int,
        default=200,
        help="Max images per taxon (across all subsets)",
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--oe-fraction",
        type=float,
        default=0.15,
        help="Fraction of negatives reserved for the Outlier Exposure pool",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-oos", action="store_true")
    parser.add_argument("--skip-backgrounds", action="store_true")
    parser.add_argument("--skip-non-marine", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, str]] = []
    if not args.skip_oos:
        all_rows.extend(
            _download_subset(
                subset="oos",
                config_path=args.out_of_scope_config,
                output_dir=args.output_dir,
                workers=args.workers,
                per_class_cap=args.per_class_cap,
            )
        )
    if not args.skip_backgrounds:
        all_rows.extend(
            _download_subset(
                subset="background",
                config_path=args.backgrounds_config,
                output_dir=args.output_dir,
                workers=args.workers,
                per_class_cap=args.per_class_cap,
            )
        )
    if not args.skip_non_marine:
        all_rows.extend(
            _download_subset(
                subset="non_marine",
                config_path=args.non_marine_config,
                output_dir=args.output_dir,
                workers=args.workers,
                per_class_cap=args.per_class_cap,
            )
        )

    if not all_rows:
        print("\nERROR: no negative images downloaded.", file=sys.stderr)
        sys.exit(1)

    train_rows, oe_rows = _split_train_oe(all_rows, args.oe_fraction, args.seed)
    _write_manifest(train_rows, args.train_manifest)
    _write_manifest(oe_rows, args.oe_manifest)

    print("\nDone.")
    print(f"  Total negatives downloaded: {len(all_rows)}")
    print(f"  Train manifest ({args.train_manifest}): {len(train_rows)} rows")
    print(f"  OE manifest    ({args.oe_manifest}): {len(oe_rows)} rows")

    # Per-subset breakdown
    by_subset: dict[str, int] = {}
    for r in train_rows:
        by_subset[r["subset"]] = by_subset.get(r["subset"], 0) + 1
    print("  Train per-subset:")
    for k, v in sorted(by_subset.items()):
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
