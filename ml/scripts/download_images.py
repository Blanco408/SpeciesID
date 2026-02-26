#!/usr/bin/env python3
"""
Download training images from iNaturalist observation CSV files.

Reads image URLs from the observation CSVs and downloads them into
class-specific directories for model training.
"""

import csv
import os
import sys
import argparse
import hashlib
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import requests
from PIL import Image
from tqdm import tqdm

# Default paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ML_DIR = os.path.join(PROJECT_ROOT, "ml")
DEFAULT_OUTPUT_DIR = os.path.join(ML_DIR, "data", "images")

# Class configuration
CLASS_CONFIG = {
    "seahare": {
        "csv_subdir": "observations-seahares.csv",
        "csv_filename": "observations-683953.csv",
        "filter_genus": "Aplysia",  # Only keep Aplysia genus
        "max_images": 8000,
    },
    "brittlestar": {
        "csv_subdir": "observations-brittlestar.csv",
        "csv_filename": "observations-683947.csv",
        "filter_genus": None,  # Keep all (order-level class)
        "max_images": 8000,
    },
    "sea_cucumber": {
        "csv_subdir": "observations-seacucumber.csv",
        "csv_filename": "observations-683951.csv",
        "filter_genus": None,  # Keep all Holothuroidea
        "max_images": 8000,
    },
}

TARGET_SIZE = 256  # Resize to 256x256, will random crop to 224 during training


def download_and_save_image(url: str, save_path: str, timeout: int = 15) -> bool:
    """Download an image, resize to TARGET_SIZE, and save as JPEG."""
    try:
        # Try medium size first, fall back to original URL
        if "/square." in url:
            url = url.replace("/square.", "/medium.")
        elif "/small." in url:
            url = url.replace("/small.", "/medium.")

        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()

        img = Image.open(BytesIO(resp.content))
        img = img.convert("RGB")

        # Resize maintaining aspect ratio, then center crop
        w, h = img.size
        scale = TARGET_SIZE / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

        # Center crop to TARGET_SIZE x TARGET_SIZE
        left = (new_w - TARGET_SIZE) // 2
        top = (new_h - TARGET_SIZE) // 2
        img = img.crop((left, top, left + TARGET_SIZE, top + TARGET_SIZE))

        img.save(save_path, "JPEG", quality=90)
        return True

    except Exception:
        return False


def load_observations(csv_path: str, class_name: str, config: dict) -> list:
    """Load observations from a CSV file, applying genus filter if specified."""
    observations = []

    if not os.path.exists(csv_path):
        print(f"  WARNING: CSV not found: {csv_path}")
        return observations

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_url = row.get("image_url", "").strip()
            if not image_url:
                continue

            # Apply genus filter if specified
            if config.get("filter_genus"):
                sci_name = row.get("scientific_name", "")
                if not sci_name.startswith(config["filter_genus"]):
                    continue

            obs_id = row.get("id", "")
            if not obs_id:
                continue

            observations.append({
                "id": obs_id,
                "image_url": image_url,
                "class": class_name,
                "scientific_name": row.get("scientific_name", ""),
                "user_id": row.get("user_id", ""),
            })

    return observations


def main():
    parser = argparse.ArgumentParser(description="Download training images from iNaturalist CSVs")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Directory containing observation CSVs")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for images")
    parser.add_argument("--workers", type=int, default=15, help="Number of download threads")
    parser.add_argument("--max-per-class", type=int, default=5000, help="Max images per class")
    args = parser.parse_args()

    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Workers: {args.workers}")
    print(f"Max per class: {args.max_per_class}")
    print()

    # Create output directories
    for class_name in CLASS_CONFIG:
        os.makedirs(os.path.join(args.output_dir, class_name), exist_ok=True)

    # Load all observations
    all_tasks = []
    for class_name, config in CLASS_CONFIG.items():
        csv_path = os.path.join(args.data_dir, config["csv_subdir"], config["csv_filename"])
        print(f"Loading {class_name} from {csv_path}...")
        observations = load_observations(csv_path, class_name, config)
        print(f"  Found {len(observations)} valid observations")

        if len(observations) == 0:
            print(f"  WARNING: No observations for {class_name}! Run download_sea_cucumbers.py first.")
            continue

        # Subsample if needed
        max_images = min(args.max_per_class, config["max_images"])
        if len(observations) > max_images:
            random.seed(42)
            observations = random.sample(observations, max_images)
            print(f"  Subsampled to {len(observations)}")

        all_tasks.extend(observations)

    if not all_tasks:
        print("ERROR: No observations to download!")
        sys.exit(1)

    print(f"\nTotal images to download: {len(all_tasks)}")

    # Download images
    success_count = 0
    fail_count = 0
    fail_log_path = os.path.join(ML_DIR, "data", "download_failures.log")
    manifest_path = os.path.join(ML_DIR, "data", "manifest.csv")

    manifest_rows = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for obs in all_tasks:
            save_path = os.path.join(args.output_dir, obs["class"], f"{obs['id']}.jpg")
            # Skip already downloaded
            if os.path.exists(save_path):
                manifest_rows.append({
                    "observation_id": obs["id"],
                    "local_path": save_path,
                    "class_label": obs["class"],
                    "user_id": obs["user_id"],
                    "scientific_name": obs["scientific_name"],
                })
                success_count += 1
                continue

            future = executor.submit(download_and_save_image, obs["image_url"], save_path)
            futures[future] = obs

        # Process results with progress bar
        remaining = len(futures)
        if remaining > 0:
            print(f"Downloading {remaining} new images ({success_count} already cached)...")
            with tqdm(total=remaining, desc="Downloading") as pbar:
                for future in as_completed(futures):
                    obs = futures[future]
                    save_path = os.path.join(args.output_dir, obs["class"], f"{obs['id']}.jpg")

                    if future.result():
                        success_count += 1
                        manifest_rows.append({
                            "observation_id": obs["id"],
                            "local_path": save_path,
                            "class_label": obs["class"],
                            "user_id": obs["user_id"],
                            "scientific_name": obs["scientific_name"],
                        })
                    else:
                        fail_count += 1
                        with open(fail_log_path, "a") as flog:
                            flog.write(f"{obs['id']},{obs['image_url']},{obs['class']}\n")

                    pbar.update(1)
        else:
            print(f"All {success_count} images already cached!")

    # Write manifest
    print(f"\nWriting manifest to {manifest_path}")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["observation_id", "local_path", "class_label", "user_id", "scientific_name"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    # Summary
    print(f"\n{'='*50}")
    print(f"Download complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed:  {fail_count}")

    # Per-class counts
    for class_name in CLASS_CONFIG:
        class_dir = os.path.join(args.output_dir, class_name)
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir) if f.endswith(".jpg")])
            print(f"  {class_name}: {count} images")


if __name__ == "__main__":
    main()
