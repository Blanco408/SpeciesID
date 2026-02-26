#!/usr/bin/env python3
"""
Split dataset into train/validation/test sets.

Uses stratified splitting by user_id to prevent data leakage
(same photographer's images won't appear in both train and test).
"""

import csv
import os
import sys
import argparse
import random
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
ML_DIR = os.path.join(PROJECT_ROOT, "ml")
DEFAULT_IMAGE_DIR = os.path.join(ML_DIR, "data", "images")
DEFAULT_MANIFEST = os.path.join(ML_DIR, "data", "manifest.csv")
DEFAULT_SPLITS_DIR = os.path.join(ML_DIR, "data", "splits")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42


def load_manifest(manifest_path: str, image_dir: str) -> list:
    """Load manifest and verify images exist."""
    entries = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Check the image actually exists (might have been removed by validation)
            img_path = os.path.join(image_dir, row["class_label"], f"{row['observation_id']}.jpg")
            if os.path.exists(img_path):
                entries.append({
                    "observation_id": row["observation_id"],
                    "image_path": img_path,
                    "class_label": row["class_label"],
                    "user_id": row.get("user_id", "unknown"),
                })
    return entries


def split_by_user(entries: list) -> tuple:
    """
    Split entries by user_id to prevent data leakage.
    Groups all images from the same user into the same split.
    """
    random.seed(SEED)

    # Group entries by (class, user_id)
    class_user_groups = defaultdict(lambda: defaultdict(list))
    for entry in entries:
        class_user_groups[entry["class_label"]][entry["user_id"]].append(entry)

    train, val, test = [], [], []

    for class_name, user_groups in class_user_groups.items():
        # Shuffle users
        users = list(user_groups.keys())
        random.shuffle(users)

        # Calculate target counts
        total = sum(len(user_groups[u]) for u in users)
        train_target = int(total * TRAIN_RATIO)
        val_target = int(total * VAL_RATIO)

        # Assign users to splits
        current_train, current_val, current_test = 0, 0, 0
        for user in users:
            user_entries = user_groups[user]
            count = len(user_entries)

            if current_train < train_target:
                train.extend(user_entries)
                current_train += count
            elif current_val < val_target:
                val.extend(user_entries)
                current_val += count
            else:
                test.extend(user_entries)
                current_test += count

    # Shuffle within each split
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def write_split(entries: list, filepath: str):
    """Write split file."""
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["observation_id", "image_path", "class_label"])
        writer.writeheader()
        writer.writerows([{
            "observation_id": e["observation_id"],
            "image_path": e["image_path"],
            "class_label": e["class_label"],
        } for e in entries])


def main():
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="Manifest CSV from download step")
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR, help="Image directory")
    parser.add_argument("--output-dir", default=DEFAULT_SPLITS_DIR, help="Output directory for split CSVs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading manifest: {args.manifest}")
    entries = load_manifest(args.manifest, args.image_dir)
    print(f"Total valid entries: {len(entries)}")

    if len(entries) == 0:
        print("ERROR: No valid entries found!")
        sys.exit(1)

    # Show class distribution
    class_counts = defaultdict(int)
    for e in entries:
        class_counts[e["class_label"]] += 1
    print("\nClass distribution:")
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls}: {count}")

    # Split
    print(f"\nSplitting {TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%} by user...")
    train, val, test = split_by_user(entries)

    # Write splits
    train_path = os.path.join(args.output_dir, "train.csv")
    val_path = os.path.join(args.output_dir, "val.csv")
    test_path = os.path.join(args.output_dir, "test.csv")

    write_split(train, train_path)
    write_split(val, val_path)
    write_split(test, test_path)

    # Print summary
    print(f"\nSplit results:")
    print(f"  Train: {len(train)} ({len(train)/len(entries):.1%})")
    print(f"  Val:   {len(val)} ({len(val)/len(entries):.1%})")
    print(f"  Test:  {len(test)} ({len(test)/len(entries):.1%})")

    # Per-class per-split breakdown
    for split_name, split_data in [("Train", train), ("Val", val), ("Test", test)]:
        counts = defaultdict(int)
        for e in split_data:
            counts[e["class_label"]] += 1
        print(f"\n  {split_name}:")
        for cls, count in sorted(counts.items()):
            print(f"    {cls}: {count}")

    print(f"\nSplit files written to: {args.output_dir}")


if __name__ == "__main__":
    main()
