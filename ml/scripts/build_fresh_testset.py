#!/usr/bin/env python3
"""
Build a fresh, holdout test set from iNaturalist.

What this script creates:
1) in_scope: labeled single-species images for model-supported classes
2) out_of_scope: species the model does not support (for false-positive testing)
3) multi_species: synthetic 2-4 species collages made from in_scope images

All in_scope and out_of_scope downloads exclude observation IDs present in the
training manifest to avoid train/test leakage.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from PIL import Image
from tqdm import tqdm

from download_taxa_dataset import (
    OBSERVATIONS_API,
    download_and_save_image,
    request_json,
    resolve_class_configs,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
ML_DIR = PROJECT_ROOT / "ml"

DEFAULT_IN_SCOPE_CONFIG = ML_DIR / "config" / "marine_species_names_20.resolved.json"
DEFAULT_OUT_SCOPE_CONFIG = ML_DIR / "config" / "out_of_scope_marine_species_10.json"
DEFAULT_TRAIN_MANIFEST = ML_DIR / "data" / "manifest.csv"
DEFAULT_OUTPUT_DIR = ML_DIR / "data" / "fresh_testset"

DEFAULT_PER_CLASS = 5
DEFAULT_OUT_SCOPE_PER_CLASS = 3
DEFAULT_MULTI_SPECIES_COUNT = 30
DEFAULT_WORKERS = 16


@dataclass
class ObservationCandidate:
    subset: str
    class_label: str
    taxon_id: int
    taxon_name: str
    observation_id: str
    user_id: str
    image_url: str


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_training_observation_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            obs_id = str(row.get("observation_id", "")).strip()
            if obs_id:
                ids.add(obs_id)
    return ids


def fetch_candidates_for_taxon(
    *,
    taxon_id: int,
    taxon_name: str,
    class_label: str,
    subset: str,
    target_count: int,
    excluded_observation_ids: set[str],
    seen_ids: set[str],
    quality_grade: str,
    require_geotag: bool,
    created_after: Optional[str],
    max_pages: int = 25,
    per_page: int = 200,
) -> list[ObservationCandidate]:
    rows: list[ObservationCandidate] = []
    page = 1

    while len(rows) < target_count and page <= max_pages:
        params: dict[str, Any] = {
            "taxon_id": taxon_id,
            "quality_grade": quality_grade,
            "photos": "true",
            "order_by": "id",
            "order": "desc",
            "page": page,
            "per_page": per_page,
        }
        if created_after:
            params["created_d1"] = created_after

        payload = request_json(OBSERVATIONS_API, params=params, timeout=30)
        results = payload.get("results", [])
        if not results:
            break

        for obs in results:
            obs_id = str(obs.get("id", "")).strip()
            if not obs_id:
                continue
            if obs_id in excluded_observation_ids or obs_id in seen_ids:
                continue

            photos = obs.get("photos") or []
            if not photos:
                continue

            location = obs.get("location")
            if require_geotag and not location:
                continue

            photo_url = str(photos[0].get("url", "")).strip()
            if not photo_url:
                continue

            row = ObservationCandidate(
                subset=subset,
                class_label=class_label,
                taxon_id=taxon_id,
                taxon_name=taxon_name,
                observation_id=obs_id,
                user_id=str((obs.get("user") or {}).get("id", "")).strip(),
                image_url=photo_url,
            )
            rows.append(row)
            seen_ids.add(obs_id)

            if len(rows) >= target_count:
                break

        page += 1
        time.sleep(1.0)

    return rows


def write_manifest(rows: Iterable[dict[str, str]], path: Path) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "subset",
                "class_label",
                "taxon_id",
                "taxon_name",
                "observation_id",
                "user_id",
                "local_path",
                "image_url",
                "source_observation_url",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def build_multi_species_collages(
    *,
    in_scope_rows: list[dict[str, str]],
    output_dir: Path,
    count: int,
    seed: int,
    min_species: int = 2,
    max_species: int = 4,
) -> list[dict[str, str]]:
    rng = random.Random(seed)

    by_class: dict[str, list[Path]] = {}
    for row in in_scope_rows:
        class_label = row["class_label"]
        local_path = Path(row["local_path"])
        if local_path.exists():
            by_class.setdefault(class_label, []).append(local_path)

    valid_classes = [k for k, v in by_class.items() if v]
    if len(valid_classes) < min_species:
        return []

    out_dir = output_dir / "multi_species"
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata: list[dict[str, str]] = []

    for idx in range(count):
        species_count = rng.randint(min_species, min(max_species, len(valid_classes)))
        chosen_classes = rng.sample(valid_classes, species_count)

        # Build a 2x2 collage with light random jitter for variety.
        canvas = Image.new("RGB", (1024, 1024), color=(245, 247, 250))
        cells = [
            (0, 0, 512, 512),
            (512, 0, 1024, 512),
            (0, 512, 512, 1024),
            (512, 512, 1024, 1024),
        ]
        rng.shuffle(cells)

        source_paths: list[str] = []
        source_classes: list[str] = []

        for class_label, cell in zip(chosen_classes, cells):
            src_path = rng.choice(by_class[class_label])
            source_paths.append(str(src_path))
            source_classes.append(class_label)

            with Image.open(src_path).convert("RGB") as img:
                # Fill the cell while keeping center crop.
                img = img.resize((560, 560), Image.Resampling.LANCZOS)
                x0, y0, x1, y1 = cell
                pad_x = rng.randint(-20, 20)
                pad_y = rng.randint(-20, 20)
                target_x = x0 + pad_x
                target_y = y0 + pad_y
                canvas.paste(img, (target_x, target_y))

        filename = f"synthetic_multi_{idx + 1:03d}.jpg"
        out_path = out_dir / filename
        canvas.save(out_path, "JPEG", quality=92)

        metadata.append(
            {
                "subset": "multi_species",
                "file_name": filename,
                "local_path": str(out_path),
                "species_labels": ";".join(source_classes),
                "source_paths": ";".join(source_paths),
            }
        )

    return metadata


def download_rows(
    rows: list[ObservationCandidate],
    output_dir: Path,
    workers: int,
) -> list[dict[str, str]]:
    manifest_rows: list[dict[str, str]] = []
    tasks: list[tuple[ObservationCandidate, Path]] = []

    for row in rows:
        subset_dir = output_dir / row.subset / row.class_label
        subset_dir.mkdir(parents=True, exist_ok=True)
        local_path = subset_dir / f"{row.observation_id}.jpg"
        if local_path.exists():
            manifest_rows.append(
                {
                    "subset": row.subset,
                    "class_label": row.class_label,
                    "taxon_id": str(row.taxon_id),
                    "taxon_name": row.taxon_name,
                    "observation_id": row.observation_id,
                    "user_id": row.user_id,
                    "local_path": str(local_path),
                    "image_url": row.image_url,
                    "source_observation_url": f"https://www.inaturalist.org/observations/{row.observation_id}",
                }
            )
            continue
        tasks.append((row, local_path))

    if tasks:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(download_and_save_image, row.image_url, str(local_path)): (row, local_path)
                for row, local_path in tasks
            }
            with tqdm(total=len(futures), desc="Downloading test images") as pbar:
                for future in as_completed(futures):
                    row, local_path = futures[future]
                    if future.result():
                        manifest_rows.append(
                            {
                                "subset": row.subset,
                                "class_label": row.class_label,
                                "taxon_id": str(row.taxon_id),
                                "taxon_name": row.taxon_name,
                                "observation_id": row.observation_id,
                                "user_id": row.user_id,
                                "local_path": str(local_path),
                                "image_url": row.image_url,
                                "source_observation_url": f"https://www.inaturalist.org/observations/{row.observation_id}",
                            }
                        )
                    pbar.update(1)

    return sorted(
        manifest_rows,
        key=lambda x: (x["subset"], x["class_label"], x["observation_id"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a fresh holdout test set from iNaturalist")
    parser.add_argument("--in-scope-config", type=Path, default=DEFAULT_IN_SCOPE_CONFIG)
    parser.add_argument("--out-of-scope-config", type=Path, default=DEFAULT_OUT_SCOPE_CONFIG)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--in-scope-per-class", type=int, default=DEFAULT_PER_CLASS)
    parser.add_argument("--out-of-scope-per-class", type=int, default=DEFAULT_OUT_SCOPE_PER_CLASS)
    parser.add_argument("--multi-species-count", type=int, default=DEFAULT_MULTI_SPECIES_COUNT)
    parser.add_argument("--skip-out-of-scope", action="store_true")
    parser.add_argument("--skip-multi-species", action="store_true")
    parser.add_argument(
        "--created-after",
        default=None,
        help="Optional ISO date filter (YYYY-MM-DD) for observations recency.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    if not args.in_scope_config.exists():
        raise FileNotFoundError(f"In-scope config missing: {args.in_scope_config}")

    in_scope_cfg = load_config(args.in_scope_config)
    quality_grade = in_scope_cfg.get("quality_grade", "research")
    require_geotag = bool(in_scope_cfg.get("require_geotag", True))

    in_scope_classes = resolve_class_configs(
        class_configs=in_scope_cfg.get("classes", []),
        default_max_images=args.in_scope_per_class,
        max_images_cap=args.in_scope_per_class,
    )
    if not in_scope_classes:
        raise ValueError("No classes found in in-scope config")

    excluded_ids = load_training_observation_ids(args.train_manifest)
    print(f"Loaded {len(excluded_ids)} training observation IDs to exclude.")

    seen_ids: set[str] = set(excluded_ids)
    planned_rows: list[ObservationCandidate] = []

    print(f"\nCollecting in-scope images ({args.in_scope_per_class} per class)...")
    for cls in in_scope_classes:
        rows = fetch_candidates_for_taxon(
            taxon_id=cls.taxon_id,
            taxon_name=cls.taxon_name,
            class_label=cls.label,
            subset="in_scope",
            target_count=args.in_scope_per_class,
            excluded_observation_ids=excluded_ids,
            seen_ids=seen_ids,
            quality_grade=quality_grade,
            require_geotag=require_geotag,
            created_after=args.created_after,
        )
        planned_rows.extend(rows)
        print(f"  {cls.label}: {len(rows)}")

    if not args.skip_out_of_scope:
        if not args.out_of_scope_config.exists():
            print(f"WARNING: Out-of-scope config missing: {args.out_of_scope_config}", file=sys.stderr)
        else:
            out_cfg = load_config(args.out_of_scope_config)
            out_classes = resolve_class_configs(
                class_configs=out_cfg.get("classes", []),
                default_max_images=args.out_of_scope_per_class,
                max_images_cap=args.out_of_scope_per_class,
            )
            out_quality_grade = out_cfg.get("quality_grade", quality_grade)
            out_require_geotag = bool(out_cfg.get("require_geotag", require_geotag))

            print(f"\nCollecting out-of-scope images ({args.out_of_scope_per_class} per class)...")
            for cls in out_classes:
                rows = fetch_candidates_for_taxon(
                    taxon_id=cls.taxon_id,
                    taxon_name=cls.taxon_name,
                    class_label=cls.label,
                    subset="out_of_scope",
                    target_count=args.out_of_scope_per_class,
                    excluded_observation_ids=excluded_ids,
                    seen_ids=seen_ids,
                    quality_grade=out_quality_grade,
                    require_geotag=out_require_geotag,
                    created_after=args.created_after,
                )
                planned_rows.extend(rows)
                print(f"  {cls.label}: {len(rows)}")

    print(f"\nDownloading {len(planned_rows)} fresh test images...")
    manifest_rows = download_rows(planned_rows, output_dir=args.output_dir, workers=args.workers)

    in_scope_rows = [r for r in manifest_rows if r["subset"] == "in_scope"]
    out_scope_rows = [r for r in manifest_rows if r["subset"] == "out_of_scope"]

    write_manifest(in_scope_rows, args.output_dir / "in_scope_manifest.csv")
    if out_scope_rows:
        write_manifest(out_scope_rows, args.output_dir / "out_of_scope_manifest.csv")

    multi_rows: list[dict[str, str]] = []
    if not args.skip_multi_species and args.multi_species_count > 0 and in_scope_rows:
        print(f"\nBuilding {args.multi_species_count} synthetic multi-species collages...")
        multi_rows = build_multi_species_collages(
            in_scope_rows=in_scope_rows,
            output_dir=args.output_dir,
            count=args.multi_species_count,
            seed=args.seed,
        )
        if multi_rows:
            with (args.output_dir / "multi_species_manifest.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["subset", "file_name", "local_path", "species_labels", "source_paths"],
                )
                writer.writeheader()
                writer.writerows(multi_rows)

    print("\nDone.")
    print(f"  Output dir: {args.output_dir}")
    print(f"  In-scope images: {len(in_scope_rows)}")
    print(f"  Out-of-scope images: {len(out_scope_rows)}")
    print(f"  Multi-species collages: {len(multi_rows)}")
    print("  Manifests:")
    print(f"    - {args.output_dir / 'in_scope_manifest.csv'}")
    if out_scope_rows:
        print(f"    - {args.output_dir / 'out_of_scope_manifest.csv'}")
    if multi_rows:
        print(f"    - {args.output_dir / 'multi_species_manifest.csv'}")


if __name__ == "__main__":
    main()
