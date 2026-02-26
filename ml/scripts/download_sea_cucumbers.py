#!/usr/bin/env python3
"""
Fetch sea cucumber observations from the iNaturalist API.

The original iNaturalist export used the wrong taxon filter, resulting in
an empty CSV. This script fetches research-grade sea cucumber observations
(Holothuroidea, taxon_id=47720) with photos via the iNaturalist API v1.
"""

import csv
import os
import sys
import time
import argparse
import requests

API_BASE = "https://api.inaturalist.org/v1/observations"

# Match the column format of the existing iNaturalist CSV exports
CSV_COLUMNS = [
    "id", "uuid", "observed_on_string", "observed_on", "time_observed_at",
    "time_zone", "user_id", "user_login", "user_name", "created_at",
    "updated_at", "quality_grade", "license", "url", "image_url", "sound_url",
    "tag_list", "description", "num_identification_agreements",
    "num_identification_disagreements", "captive_cultivated",
    "oauth_application_id", "place_guess", "latitude", "longitude",
    "positional_accuracy", "private_place_guess", "private_latitude",
    "private_longitude", "public_positional_accuracy", "geoprivacy",
    "taxon_geoprivacy", "coordinates_obscured", "positioning_method",
    "positioning_device", "species_guess", "scientific_name", "common_name",
    "iconic_taxon_name", "taxon_id",
]


def fetch_page(page: int, per_page: int = 200) -> dict:
    """Fetch a single page of sea cucumber observations."""
    params = {
        "taxon_id": 47720,  # Holothuroidea (sea cucumbers)
        "quality_grade": "research",
        "photos": "true",
        "per_page": per_page,
        "page": page,
        "order": "desc",
        "order_by": "id",
    }
    resp = requests.get(API_BASE, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def observation_to_row(obs: dict) -> dict:
    """Convert an iNaturalist API observation to our CSV row format."""
    taxon = obs.get("taxon") or {}
    photos = obs.get("photos") or []
    image_url = photos[0]["url"].replace("square", "medium") if photos else ""
    location = obs.get("location")
    lat, lon = ("", "")
    if location:
        parts = location.split(",")
        if len(parts) == 2:
            lat, lon = parts[0].strip(), parts[1].strip()

    return {
        "id": obs.get("id", ""),
        "uuid": obs.get("uuid", ""),
        "observed_on_string": obs.get("observed_on_string", ""),
        "observed_on": obs.get("observed_on_details", {}).get("date", "") if obs.get("observed_on_details") else "",
        "time_observed_at": obs.get("time_observed_at", ""),
        "time_zone": obs.get("observed_time_zone", ""),
        "user_id": obs.get("user", {}).get("id", ""),
        "user_login": obs.get("user", {}).get("login", ""),
        "user_name": obs.get("user", {}).get("name", ""),
        "created_at": obs.get("created_at", ""),
        "updated_at": obs.get("updated_at", ""),
        "quality_grade": obs.get("quality_grade", ""),
        "license": obs.get("license_code", ""),
        "url": f"http://www.inaturalist.org/observations/{obs.get('id', '')}",
        "image_url": image_url,
        "sound_url": "",
        "tag_list": "",
        "description": obs.get("description", "") or "",
        "num_identification_agreements": obs.get("identifications_most_agree", ""),
        "num_identification_disagreements": obs.get("identifications_most_disagree", ""),
        "captive_cultivated": str(obs.get("captive", False)).lower(),
        "oauth_application_id": "",
        "place_guess": obs.get("place_guess", "") or "",
        "latitude": lat,
        "longitude": lon,
        "positional_accuracy": obs.get("positional_accuracy", ""),
        "private_place_guess": "",
        "private_latitude": "",
        "private_longitude": "",
        "public_positional_accuracy": obs.get("public_positional_accuracy", ""),
        "geoprivacy": obs.get("geoprivacy", "") or "",
        "taxon_geoprivacy": obs.get("taxon_geoprivacy", "") or "",
        "coordinates_obscured": str(obs.get("obscured", False)).lower(),
        "positioning_method": "",
        "positioning_device": "",
        "species_guess": obs.get("species_guess", "") or "",
        "scientific_name": taxon.get("name", ""),
        "common_name": taxon.get("preferred_common_name", ""),
        "iconic_taxon_name": taxon.get("iconic_taxon_name", ""),
        "taxon_id": taxon.get("id", ""),
    }


def main():
    parser = argparse.ArgumentParser(description="Download sea cucumber observations from iNaturalist")
    parser.add_argument("--output", default=None, help="Output CSV path")
    parser.add_argument("--max-observations", type=int, default=8000, help="Max observations to fetch")
    args = parser.parse_args()

    # Default output path
    if args.output is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        data_dir = os.path.join(project_root, "data", "observations-seacucumber.csv")
        os.makedirs(data_dir, exist_ok=True)
        args.output = os.path.join(data_dir, "observations-683951.csv")

    print(f"Fetching sea cucumber observations from iNaturalist API...")
    print(f"Output: {args.output}")
    print(f"Target: {args.max_observations} observations")

    # First, check total available
    initial = fetch_page(1, per_page=1)
    total_available = initial.get("total_results", 0)
    print(f"Total available: {total_available} research-grade observations with photos")

    target = min(args.max_observations, total_available)
    rows = []
    page = 1
    per_page = 200

    while len(rows) < target:
        try:
            data = fetch_page(page, per_page)
            results = data.get("results", [])
            if not results:
                print(f"No more results at page {page}")
                break

            for obs in results:
                row = observation_to_row(obs)
                if row["image_url"] and row["latitude"]:  # require photo and location
                    rows.append(row)
                    if len(rows) >= target:
                        break

            print(f"  Page {page}: fetched {len(results)} obs, total collected: {len(rows)}/{target}")
            page += 1

            # Respect rate limits (1 req/sec)
            time.sleep(1.0)

        except requests.exceptions.RequestException as e:
            print(f"  Error on page {page}: {e}")
            time.sleep(5.0)
            continue

    # Write CSV
    print(f"\nWriting {len(rows)} observations to {args.output}")
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done! {len(rows)} sea cucumber observations saved.")


if __name__ == "__main__":
    main()
