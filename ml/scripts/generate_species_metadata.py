#!/usr/bin/env python3
"""Generate species_metadata.json for the iOS app bundle.

Merges:
  1. A resolved training config (label + taxon_name per species)
  2. A hand-curated display-info JSON (displayName, scientificName, iconName, category)

Species present in the training config but absent from the display-info file
receive auto-generated display names (title-cased label) and sensible defaults.

Usage:
    python generate_species_metadata.py \
        --config ml/config/marine_species_names_20.resolved.json \
        --display-info ml/config/species_display_info.json \
        --output SpeciesID/SpeciesID/species_metadata.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


def auto_display_name(label: str) -> str:
    """Convert a snake_case label to Title Case display name."""
    return label.replace("_", " ").title()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate species_metadata.json for the iOS bundle."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=pathlib.Path,
        help="Path to the resolved training config JSON (e.g. marine_species_names_20.resolved.json)",
    )
    parser.add_argument(
        "--display-info",
        required=True,
        type=pathlib.Path,
        help="Path to species_display_info.json",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("SpeciesID/SpeciesID/species_metadata.json"),
        help="Output path for the generated metadata JSON (default: SpeciesID/SpeciesID/species_metadata.json)",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load inputs
    # ------------------------------------------------------------------
    if not args.config.exists():
        print(f"Error: training config not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    if not args.display_info.exists():
        print(f"Error: display info not found: {args.display_info}", file=sys.stderr)
        sys.exit(1)

    with open(args.config, "r") as f:
        training_config = json.load(f)

    with open(args.display_info, "r") as f:
        display_info = json.load(f)

    # ------------------------------------------------------------------
    # Build species list
    # ------------------------------------------------------------------
    classes = training_config.get("classes", [])
    if not classes:
        print("Error: training config contains no classes", file=sys.stderr)
        sys.exit(1)

    species_list: list[dict] = []
    for cls in classes:
        label = cls["label"]
        taxon_name = cls.get("taxon_name", "")

        info = display_info.get(label, {})

        entry = {
            "id": label,
            "displayName": info.get("displayName", auto_display_name(label)),
            "scientificName": info.get("scientificName", taxon_name),
            "iconName": info.get("iconName", "fish.fill"),
        }

        for optional_field in ("category", "referenceImageUrl", "description"):
            value = info.get(optional_field)
            if value:
                entry[optional_field] = value

        species_list.append(entry)

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    output_data = {
        "version": "2.0",
        "species": species_list,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Generated {args.output} with {len(species_list)} species")


if __name__ == "__main__":
    main()
