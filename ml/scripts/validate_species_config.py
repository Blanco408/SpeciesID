#!/usr/bin/env python3
"""
Validate species config by checking iNaturalist observation counts.

Queries the iNaturalist API for each species in the config to verify
sufficient research-grade observations exist for training data.

Usage:
    python -m ml.scripts.validate_species_config --config ml/config/marine_species_names_75.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from ml.scripts.download_taxa_dataset import (
    resolve_class_configs,
    request_json,
    OBSERVATIONS_API,
)


def check_observation_count(
    taxon_id: int,
    quality_grade: str = "research",
    require_geotag: bool = True,
) -> int:
    """Get total observation count for a taxon from iNaturalist."""
    params: dict[str, Any] = {
        "taxon_id": taxon_id,
        "quality_grade": quality_grade,
        "photos": "true",
        "per_page": 1,
    }
    payload = request_json(OBSERVATIONS_API, params=params, timeout=30)
    return payload.get("total_results", 0)


def main():
    parser = argparse.ArgumentParser(description="Validate species config data availability")
    parser.add_argument("--config", required=True, help="Path to species config JSON")
    parser.add_argument("--min-observations", type=int, default=200,
                        help="Minimum observations required (default: 200)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    class_configs = config.get("classes", [])
    if not class_configs:
        print("ERROR: Config must include a non-empty 'classes' list")
        sys.exit(1)

    quality_grade = config.get("quality_grade", "research")
    require_geotag = bool(config.get("require_geotag", True))

    print(f"Resolving {len(class_configs)} classes...")
    resolved = resolve_class_configs(
        class_configs=class_configs,
        default_max_images=500,
    )

    print(f"\nChecking observation counts (quality={quality_grade}, geotag={require_geotag})...")
    print(f"{'Label':<35} {'Taxon ID':<12} {'Taxon Name':<40} {'Observations':<15} {'Status'}")
    print("-" * 120)

    pass_count = 0
    warn_count = 0
    skip_count = 0

    for item in resolved:
        try:
            count = check_observation_count(
                taxon_id=item.taxon_id,
                quality_grade=quality_grade,
                require_geotag=require_geotag,
            )
        except Exception as e:
            count = -1
            print(f"  ERROR querying {item.label}: {e}")

        if count >= args.min_observations:
            status = "PASS"
            pass_count += 1
        elif count >= args.min_observations // 2:
            status = "WARN"
            warn_count += 1
        else:
            status = "SKIP"
            skip_count += 1

        print(f"{item.label:<35} {item.taxon_id:<12} {item.taxon_name:<40} {count:<15} {status}")
        time.sleep(0.5)

    print(f"\n{'='*60}")
    print(f"Results: {pass_count} PASS, {warn_count} WARN, {skip_count} SKIP")
    print(f"Total classes checked: {len(resolved)}")

    if skip_count > 0:
        print(f"\nWARNING: {skip_count} species have fewer than {args.min_observations // 2} "
              f"observations and should be removed from the config.")

    if warn_count > 0:
        print(f"\nNOTE: {warn_count} species have between {args.min_observations // 2} and "
              f"{args.min_observations} observations. Consider whether they have enough data.")


if __name__ == "__main__":
    main()
