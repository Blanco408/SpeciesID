#!/usr/bin/env python3
"""
Validate downloaded images: remove corrupt, tiny, and duplicate images.
"""

import os
import sys
import argparse
import hashlib
from collections import Counter

from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DEFAULT_IMAGE_DIR = os.path.join(PROJECT_ROOT, "ml", "data", "images")

MIN_SIZE = 50  # Minimum image dimension


def compute_dhash(img: Image.Image, hash_size: int = 8) -> str:
    """Compute difference hash for near-duplicate detection."""
    img = img.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    pixels = list(img.getdata())
    bits = []
    for row in range(hash_size):
        for col in range(hash_size):
            idx = row * (hash_size + 1) + col
            bits.append(1 if pixels[idx] < pixels[idx + 1] else 0)
    return "".join(str(b) for b in bits)


def validate_class(class_dir: str, class_name: str) -> dict:
    """Validate all images in a class directory."""
    stats = {
        "total": 0,
        "valid": 0,
        "corrupt": 0,
        "too_small": 0,
        "duplicates": 0,
        "removed": [],
    }

    files = [f for f in os.listdir(class_dir) if f.endswith(".jpg")]
    stats["total"] = len(files)

    seen_hashes = {}
    valid_files = []

    for filename in tqdm(files, desc=f"Validating {class_name}"):
        filepath = os.path.join(class_dir, filename)

        try:
            img = Image.open(filepath)
            img.verify()  # Verify it's a valid image
            img = Image.open(filepath)  # Re-open after verify
            w, h = img.size

            # Check minimum size
            if w < MIN_SIZE or h < MIN_SIZE:
                stats["too_small"] += 1
                stats["removed"].append(filepath)
                os.remove(filepath)
                continue

            # Check for near-duplicates
            dhash = compute_dhash(img)
            if dhash in seen_hashes:
                stats["duplicates"] += 1
                stats["removed"].append(filepath)
                os.remove(filepath)
                continue

            seen_hashes[dhash] = filename
            stats["valid"] += 1
            valid_files.append(filename)

        except Exception:
            stats["corrupt"] += 1
            stats["removed"].append(filepath)
            try:
                os.remove(filepath)
            except OSError:
                pass

    return stats


def main():
    parser = argparse.ArgumentParser(description="Validate downloaded training images")
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR, help="Directory containing class subdirectories")
    args = parser.parse_args()

    print(f"Validating images in: {args.image_dir}")
    print()

    classes = sorted([d for d in os.listdir(args.image_dir)
                      if os.path.isdir(os.path.join(args.image_dir, d))])

    if not classes:
        print("ERROR: No class directories found!")
        sys.exit(1)

    total_stats = Counter()

    for class_name in classes:
        class_dir = os.path.join(args.image_dir, class_name)
        stats = validate_class(class_dir, class_name)

        print(f"\n{class_name}:")
        print(f"  Total:      {stats['total']}")
        print(f"  Valid:      {stats['valid']}")
        print(f"  Corrupt:    {stats['corrupt']}")
        print(f"  Too small:  {stats['too_small']}")
        print(f"  Duplicates: {stats['duplicates']}")

        for key in ["total", "valid", "corrupt", "too_small", "duplicates"]:
            total_stats[key] += stats[key]

    print(f"\n{'='*50}")
    print(f"Overall:")
    print(f"  Total:      {total_stats['total']}")
    print(f"  Valid:      {total_stats['valid']}")
    print(f"  Removed:    {total_stats['total'] - total_stats['valid']}")

    # Check minimum viable dataset
    for class_name in classes:
        class_dir = os.path.join(args.image_dir, class_name)
        count = len([f for f in os.listdir(class_dir) if f.endswith(".jpg")])
        if count < 500:
            print(f"\n  WARNING: {class_name} has only {count} images (minimum 500 recommended)")


if __name__ == "__main__":
    main()
