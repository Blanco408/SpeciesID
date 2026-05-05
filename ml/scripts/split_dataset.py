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


def load_negatives_manifest(manifest_path: str) -> list:
    """Load the negatives manifest produced by build_negative_dataset.py.

    Each surviving row becomes a class_label='nothing' training entry. We use
    `source_taxon` as a pseudo user_id so the user-aware split still avoids
    leakage between negative families across train/val/test.
    """
    entries = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = row.get("image_path", "").strip()
            if not img_path or not os.path.exists(img_path):
                continue
            source_taxon = row.get("source_taxon") or "unknown"
            subset = row.get("subset") or "negative"
            obs_id = row.get("observation_id") or os.path.splitext(os.path.basename(img_path))[0]
            entries.append({
                "observation_id": obs_id,
                "image_path": img_path,
                "class_label": "nothing",
                # `subset:taxon` keeps splits coherent: e.g. all kelp_canopy negatives
                # land in the same fold rather than being scattered.
                "user_id": f"neg:{subset}:{source_taxon}",
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
        test_target = total - train_target - val_target

        # If the class has too few unique users, fall back to per-image split so all
        # classes remain represented in val/test.
        if len(users) < 4:
            class_entries = []
            for user in users:
                class_entries.extend(user_groups[user])
            random.shuffle(class_entries)

            val_count = max(1, int(total * VAL_RATIO))
            test_count = max(1, int(total * TEST_RATIO))
            train_count = max(1, total - val_count - test_count)
            # Keep exact total after rounding.
            overflow = train_count + val_count + test_count - total
            if overflow > 0:
                train_count -= overflow

            train.extend(class_entries[:train_count])
            val.extend(class_entries[train_count:train_count + val_count])
            test.extend(class_entries[train_count + val_count:train_count + val_count + test_count])
            continue

        # Assign one user to val and one to test first, to guarantee class coverage.
        val_seed_user = users.pop()
        test_seed_user = users.pop()
        val.extend(user_groups[val_seed_user])
        test.extend(user_groups[test_seed_user])
        current_train = 0
        current_val = len(user_groups[val_seed_user])
        current_test = len(user_groups[test_seed_user])

        # Greedily assign remaining users to the split most under target.
        for user in users:
            user_entries = user_groups[user]
            count = len(user_entries)

            deficits = {
                "train": train_target - current_train,
                "val": val_target - current_val,
                "test": test_target - current_test,
            }
            destination = max(deficits.keys(), key=lambda k: deficits[k])

            if destination == "train":
                train.extend(user_entries)
                current_train += count
            elif destination == "val":
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
    parser.add_argument(
        "--negatives-manifest",
        default=None,
        help="Optional manifest CSV from build_negative_dataset.py; rows are added "
             "to every split as the 'nothing' class.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading manifest: {args.manifest}")
    entries = load_manifest(args.manifest, args.image_dir)
    print(f"Total valid entries: {len(entries)}")

    if args.negatives_manifest:
        print(f"Loading negatives manifest: {args.negatives_manifest}")
        negative_entries = load_negatives_manifest(args.negatives_manifest)
        print(f"Negative entries (class_label='nothing'): {len(negative_entries)}")
        entries.extend(negative_entries)

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
