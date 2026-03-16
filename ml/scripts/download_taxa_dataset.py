#!/usr/bin/env python3
"""
Download multi-species training data from iNaturalist API.

Supports two class config modes:
1. Explicit taxon ID:
   {"label": "seahare", "taxon_id": 47768, "max_images": 3000}
2. Taxon query resolution:
   {"label": "bat_star", "taxon_query": "Patiria miniata", "preferred_rank": "species"}
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
from io import BytesIO
from typing import Any, Optional

import requests
from PIL import Image
from tqdm import tqdm


OBSERVATIONS_API = "https://api.inaturalist.org/v1/observations"
TAXA_AUTOCOMPLETE_API = "https://api.inaturalist.org/v1/taxa/autocomplete"
DEFAULT_PER_PAGE = 200
TARGET_SIZE = 256

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
ML_DIR = os.path.join(PROJECT_ROOT, "ml")
DEFAULT_OUTPUT_DIR = os.path.join(ML_DIR, "data", "images")
DEFAULT_MANIFEST = os.path.join(ML_DIR, "data", "manifest.csv")
DEFAULT_FAILURES = os.path.join(ML_DIR, "data", "download_failures.log")


@dataclass
class ResolvedClassConfig:
    label: str
    taxon_id: int
    taxon_name: str
    taxon_rank: str
    max_images: int


def request_json(url: str, params: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def choose_best_taxon_result(
    results: list[dict[str, Any]],
    preferred_rank: Optional[str],
    iconic_taxon_name: Optional[str],
) -> Optional[dict[str, Any]]:
    if not results:
        return None

    ranked: list[tuple[int, dict[str, Any]]] = []
    preferred_rank = (preferred_rank or "").lower().strip()
    iconic_taxon_name = (iconic_taxon_name or "").lower().strip()

    for result in results:
        score = 0
        rank = (result.get("rank") or "").lower()
        iconic = ((result.get("iconic_taxon_name") or "")).lower()
        preferred_common = ((result.get("preferred_common_name") or "")).lower()
        name = ((result.get("name") or "")).lower()

        if preferred_rank and rank == preferred_rank:
            score += 5
        if iconic_taxon_name and iconic == iconic_taxon_name:
            score += 3
        if preferred_rank and rank in {"species", "subspecies", "variety"}:
            score += 1
        if "california" in preferred_common or "california" in name:
            score += 1

        ranked.append((score, result))

    ranked.sort(key=lambda t: t[0], reverse=True)
    return ranked[0][1]


def resolve_taxon(
    taxon_query: str,
    preferred_rank: Optional[str] = None,
    iconic_taxon_name: Optional[str] = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "q": taxon_query,
        "per_page": 30,
        "is_active": "true",
    }
    if preferred_rank:
        params["rank"] = preferred_rank

    payload = request_json(TAXA_AUTOCOMPLETE_API, params=params, timeout=30)
    results = payload.get("results", [])
    choice = choose_best_taxon_result(results, preferred_rank, iconic_taxon_name)
    if not choice:
        raise ValueError(f"No matching taxon found for query '{taxon_query}'")
    return choice


def resolve_class_configs(
    class_configs: list[dict[str, Any]],
    default_max_images: int,
    max_images_cap: Optional[int] = None,
) -> list[ResolvedClassConfig]:
    resolved: list[ResolvedClassConfig] = []

    for class_cfg in class_configs:
        label = class_cfg["label"]
        max_images = int(class_cfg.get("max_images", default_max_images))
        if max_images_cap is not None:
            max_images = min(max_images, int(max_images_cap))

        if class_cfg.get("taxon_id") is not None:
            taxon_id = int(class_cfg["taxon_id"])
            taxon_name = class_cfg.get("taxon_name", str(taxon_id))
            taxon_rank = class_cfg.get("taxon_rank", "unknown")
            resolved.append(
                ResolvedClassConfig(
                    label=label,
                    taxon_id=taxon_id,
                    taxon_name=taxon_name,
                    taxon_rank=taxon_rank,
                    max_images=max_images,
                )
            )
            continue

        taxon_query = class_cfg.get("taxon_query") or class_cfg.get("scientific_name") or label
        preferred_rank = class_cfg.get("preferred_rank")
        iconic_taxon_name = class_cfg.get("iconic_taxon_name")

        taxon = resolve_taxon(
            taxon_query=taxon_query,
            preferred_rank=preferred_rank,
            iconic_taxon_name=iconic_taxon_name,
        )

        resolved.append(
            ResolvedClassConfig(
                label=label,
                taxon_id=int(taxon["id"]),
                taxon_name=taxon.get("name", taxon_query),
                taxon_rank=taxon.get("rank", "unknown"),
                max_images=max_images,
            )
        )

        time.sleep(0.35)  # be polite on taxonomy endpoint

    return resolved


def fetch_observations(
    taxon_id: int,
    max_images: int,
    quality_grade: str = "research",
    require_geotag: bool = True,
) -> list[dict[str, str]]:
    """Fetch observations with photos for a taxon."""
    rows: list[dict[str, str]] = []
    page = 1

    while len(rows) < max_images:
        params = {
            "taxon_id": taxon_id,
            "quality_grade": quality_grade,
            "photos": "true",
            "per_page": DEFAULT_PER_PAGE,
            "page": page,
            "order_by": "id",
            "order": "desc",
        }
        payload = request_json(OBSERVATIONS_API, params=params, timeout=30)
        results = payload.get("results", [])
        if not results:
            break

        for obs in results:
            photos = obs.get("photos") or []
            if not photos:
                continue

            location = obs.get("location")
            if require_geotag and not location:
                continue

            photo_url = photos[0].get("url", "")
            if not photo_url:
                continue

            rows.append(
                {
                    "observation_id": str(obs.get("id", "")),
                    "user_id": str((obs.get("user") or {}).get("id", "")),
                    "scientific_name": (obs.get("taxon") or {}).get("name", ""),
                    "image_url": photo_url,
                }
            )
            if len(rows) >= max_images:
                break

        page += 1
        time.sleep(1.0)  # rate-limit friendly

    return rows


def download_and_save_image(url: str, save_path: str) -> bool:
    """Download, resize, and save image."""
    try:
        if "/square." in url:
            url = url.replace("/square.", "/medium.")
        elif "/small." in url:
            url = url.replace("/small.", "/medium.")

        response = requests.get(url, timeout=20)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")

        w, h = image.size
        scale = TARGET_SIZE / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)

        left = (new_w - TARGET_SIZE) // 2
        top = (new_h - TARGET_SIZE) // 2
        image = image.crop((left, top, left + TARGET_SIZE, top + TARGET_SIZE))

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        image.save(save_path, "JPEG", quality=90)
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Download training images from iNaturalist taxa")
    parser.add_argument("--config", required=True, help="Path to JSON config containing class list")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--failures-log", default=DEFAULT_FAILURES)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--default-max-images", type=int, default=3000)
    parser.add_argument("--max-images-cap", type=int, default=None, help="Hard cap applied to every class' max_images")
    parser.add_argument("--min-images-per-class", type=int, default=120)
    parser.add_argument("--write-resolved-config", default=None, help="Optional output path for resolved taxon IDs JSON")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    class_configs = config.get("classes", [])
    if not class_configs:
        raise ValueError("Config must include a non-empty 'classes' list")

    quality_grade = config.get("quality_grade", "research")
    require_geotag = bool(config.get("require_geotag", True))

    random.seed(args.seed)

    print(f"Resolving {len(class_configs)} classes to iNaturalist taxa...")
    resolved_classes = resolve_class_configs(
        class_configs=class_configs,
        default_max_images=args.default_max_images,
        max_images_cap=args.max_images_cap,
    )

    print("\nResolved classes:")
    for item in resolved_classes:
        print(f"  - {item.label}: taxon_id={item.taxon_id} ({item.taxon_name}, rank={item.taxon_rank})")

    if args.write_resolved_config:
        resolved_payload = {
            "quality_grade": quality_grade,
            "require_geotag": require_geotag,
            "classes": [
                {
                    "label": item.label,
                    "taxon_id": item.taxon_id,
                    "taxon_name": item.taxon_name,
                    "taxon_rank": item.taxon_rank,
                    "max_images": item.max_images,
                }
                for item in resolved_classes
            ],
        }
        os.makedirs(os.path.dirname(args.write_resolved_config), exist_ok=True)
        with open(args.write_resolved_config, "w", encoding="utf-8") as f:
            json.dump(resolved_payload, f, indent=2)
        print(f"\nResolved config written to: {args.write_resolved_config}")

    manifest_rows: list[dict[str, str]] = []
    download_tasks: list[dict[str, str]] = []
    skipped_classes: list[str] = []

    print(f"\nOutput dir: {args.output_dir}")
    print(f"Manifest: {args.manifest}")
    print(f"Workers: {args.workers}")

    for class_item in resolved_classes:
        print(
            f"\nFetching {class_item.label} "
            f"(taxon_id={class_item.taxon_id}, max_images={class_item.max_images})..."
        )
        rows = fetch_observations(
            taxon_id=class_item.taxon_id,
            max_images=class_item.max_images,
            quality_grade=quality_grade,
            require_geotag=require_geotag,
        )
        print(f"  fetched {len(rows)} candidate observations")

        if len(rows) < args.min_images_per_class:
            print(
                f"  skipping {class_item.label}: {len(rows)} < min-images-per-class {args.min_images_per_class}"
            )
            skipped_classes.append(class_item.label)
            continue

        if len(rows) > class_item.max_images:
            rows = random.sample(rows, class_item.max_images)

        for row in rows:
            filename = f"{row['observation_id']}.jpg"
            local_path = os.path.join(args.output_dir, class_item.label, filename)
            row["class_label"] = class_item.label
            row["local_path"] = local_path

            if os.path.exists(local_path):
                manifest_rows.append(
                    {
                        "observation_id": row["observation_id"],
                        "local_path": local_path,
                        "class_label": class_item.label,
                        "user_id": row["user_id"],
                        "scientific_name": row["scientific_name"],
                    }
                )
            else:
                download_tasks.append(row)

    print(f"\nDownloading {len(download_tasks)} images...")
    success = 0
    failed = 0
    os.makedirs(os.path.dirname(args.failures_log), exist_ok=True)
    if os.path.exists(args.failures_log):
        os.remove(args.failures_log)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_and_save_image, row["image_url"], row["local_path"]): row
            for row in download_tasks
        }

        with tqdm(total=len(futures), desc="Downloading") as pbar:
            for future in as_completed(futures):
                row = futures[future]
                if future.result():
                    success += 1
                    manifest_rows.append(
                        {
                            "observation_id": row["observation_id"],
                            "local_path": row["local_path"],
                            "class_label": row["class_label"],
                            "user_id": row["user_id"],
                            "scientific_name": row["scientific_name"],
                        }
                    )
                else:
                    failed += 1
                    with open(args.failures_log, "a", encoding="utf-8") as f:
                        f.write(
                            f"{row['class_label']},{row['observation_id']},{row['image_url']}\n"
                        )
                pbar.update(1)

    os.makedirs(os.path.dirname(args.manifest), exist_ok=True)
    with open(args.manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["observation_id", "local_path", "class_label", "user_id", "scientific_name"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    class_counts: dict[str, int] = {}
    for row in manifest_rows:
        class_counts[row["class_label"]] = class_counts.get(row["class_label"], 0) + 1

    print("\nDone.")
    print(f"  Manifest rows: {len(manifest_rows)}")
    print(f"  Downloaded: {success}")
    print(f"  Failed: {failed}")
    if skipped_classes:
        print(f"  Skipped classes: {', '.join(skipped_classes)}")
    print("  Final class counts:")
    for label in sorted(class_counts.keys()):
        print(f"    {label}: {class_counts[label]}")

    if not manifest_rows:
        print("\nERROR: No dataset rows were produced.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
