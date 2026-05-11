# SpeciesID

iOS app for identifying Pacific coast marine species in the field. Classification runs entirely on-device, so it works without a signal — designed for tide pools, beaches, and dive sites where reception is unreliable. Observations sync to Firebase when the device is back online.

## Features

- **On-device species classification** using a Core ML model (EfficientNet-B0, ~7.8 MB) covering 20 Pacific coast intertidal species
- **Multi-species detection** in a single photo via tiled multi-crop inference with IoU-based suppression and approximate bounding boxes
- **Offline-first observations** — capture, classify, and store entries with no network, then sync automatically when reconnected
- **Geotagged records** with map view of past observations
- **CSV / JSON export** of observation history (with or without photo attachments)
- **Firebase auth** via Sign in with Apple, Google, or email/password

## Supported species

20 Pacific coast species across echinoderms, anemones, molluscs, crustaceans, cephalopods, and nudibranchs:

Brittle Star · Sea Cucumber · California Seahare · Bat Star · Ochre Sea Star · Purple Sea Urchin · Red Sea Urchin · Giant Green Anemone · Aggregating Anemone · California Mussel · Red Abalone · Owl Limpet · Gooseneck Barnacle · Acorn Barnacle · Kelp Crab · Red Rock Crab · Blueband Hermit Crab · East Pacific Red Octopus · Sea Lemon Nudibranch · Spanish Shawl Nudibranch

The classifier hits **87.0% top-1** and **95.9% top-3** accuracy on the held-out test set. See `MODEL_COMPARISON_REPORT.md` for full per-class metrics and the MobileNetV3-Large baseline comparison.

## Architecture

The project has three parts:

- **`SpeciesID/`** — the SwiftUI iOS app (iOS 17+). Camera capture, classification, observation history, map view, export, settings.
- **`Backend/`** — Swift data models and Firestore repositories shared by the app. Defines the `User`, `Observation`, `Photo`, `Species`, and `Region` schemas, plus the repository layer that talks to Firestore.
- **`ml/`** — the Python training pipeline. iNaturalist scrapers, train/val/test split tooling, the PyTorch training loop, evaluation scripts, and the Core ML converter that produces `SpeciesClassifier.mlpackage`.

See `MODEL_ARCHITECTURE.md` for the full ML pipeline, data augmentation pipeline, and Core ML conversion details.

## iOS setup

1. Clone the repo.

2. Get the Firebase config file:
   - Open the **SpeciesID** project at https://console.firebase.google.com (ask the team for access if you need it)
   - Project Settings → "Your apps" → iOS app → download `GoogleService-Info.plist`

3. Open `SpeciesID/SpeciesID.xcodeproj` in Xcode.

4. Drag `GoogleService-Info.plist` into the `SpeciesID` folder alongside `ContentView.swift`. Check **Copy items if needed**.

5. Select the project in the sidebar → **Signing & Capabilities** → pick your team.

6. Build with `Cmd + B`. Run on a physical device for camera testing (the simulator can't access the camera).

The Core ML model (`SpeciesClassifier.mlpackage`) is already committed to the Xcode project, so no separate model download is needed to run the app.

## ML pipeline setup

The training pipeline is only needed if you want to retrain the model or add species.

```bash
cd ml
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Typical workflow:

```bash
# 1. Download images from iNaturalist (research-grade observations only)
python scripts/download_taxa_dataset.py --config config/marine_species_names_20.json

# 2. Validate the dataset
python scripts/validate_dataset.py

# 3. Generate train/val/test splits
python scripts/split_dataset.py

# 4. Train
python -m training.train --architecture efficientnet_b0

# 5. Evaluate on the test set
python -m training.evaluate

# 6. Convert the best checkpoint to Core ML
python scripts/convert_to_coreml.py
```

The converted `.mlpackage` is copied into `SpeciesID/SpeciesID/SpeciesClassifier.mlpackage`, replacing the shipped model.

### Expanding to more species

A 66-species config is prepared at `ml/config/marine_species_names_75.json`. Run `scripts/validate_species_config.py` first to confirm enough research-grade iNaturalist data exists for each taxon, then download and retrain.

## Project layout

```
SpeciesID/
├── README.md
├── MODEL_ARCHITECTURE.md           # ML pipeline + Core ML conversion details
├── MODEL_COMPARISON_REPORT.md      # EfficientNet-B0 vs MobileNetV3-L benchmark
├── firebase.json                   # Firebase project config
├── firestore.rules                 # Firestore security rules
├── firestore.indexes.json          # Firestore composite indexes
├── storage.rules                   # Firebase Storage security rules
├── Backend/
│   ├── Models/                     # User, Observation, Photo, Species, Region
│   └── Repositories/               # UserRepository, ObservationRepository, Authentication
├── SpeciesID/                      # Xcode project
│   └── SpeciesID/
│       ├── SpeciesIDApp.swift              # App entry point
│       ├── MainTabView.swift               # Tab navigation root
│       ├── HomeView.swift                  # Dashboard
│       ├── CameraView.swift                # Capture + on-device classification
│       ├── ClassificationResultView.swift  # Prediction display
│       ├── SpeciesClassifierService.swift  # Vision + Core ML wrapper
│       ├── SpeciesClassifier.mlpackage     # Shipped Core ML model
│       ├── ObservationStore.swift          # Local persistence
│       ├── ObservationHistoryView.swift    # List of past observations
│       ├── ObservationDetailView.swift
│       ├── ObservationMapView.swift        # Map of geotagged observations
│       ├── ExportService.swift             # CSV / JSON export
│       ├── ExportView.swift
│       ├── SupportedSpeciesView.swift      # Species reference UI
│       ├── SyncService.swift               # Firestore sync
│       ├── AuthenticationManager.swift     # Apple / Google / email auth
│       ├── LoginView.swift
│       ├── EmailLogin.swift
│       ├── SettingsView.swift
│       ├── ImageStore.swift                # On-disk photo cache
│       ├── species_metadata.json           # Display names, scientific names
│       └── classifier_thresholds.json      # Per-class confidence thresholds
└── ml/
    ├── requirements.txt
    ├── config/                     # Species lists (20, 75), negatives, OOD eval sets
    ├── training/
    │   ├── model.py                # EfficientNet-B0 / MobileNetV3 head swap
    │   ├── dataset.py
    │   ├── train.py                # AdamW + cosine schedule + early stopping
    │   ├── evaluate.py             # Confusion matrix, precision/recall/F1
    │   └── metrics.py
    └── scripts/
        ├── download_taxa_dataset.py
        ├── download_images.py
        ├── split_dataset.py
        ├── validate_dataset.py
        ├── validate_species_config.py
        ├── build_negative_dataset.py
        ├── build_fresh_testset.py
        ├── convert_to_coreml.py
        ├── benchmark_models.py
        ├── evaluate_ood.py             # Out-of-distribution evaluation
        ├── smoke_test_ood.py
        └── generate_species_metadata.py
```

## Firestore data model

**`users/{userId}`** — `email`, `display_name`, `date_created`, `last_login`, `downloaded_regions`, `preferences`

**`observations/{observationId}`** — `user_id`, `timestamp`, `coordinates`, `region_name`, `identifications` (multi-species payload), `notes`, `sync_status`

**`observations/{observationId}/photos/{photoId}`** — `storage_url`, `thumbnail_url`, `local_path`, `upload_status`

**`species/{speciesId}`** — read-only species reference data

**`regions/{regionId}`** — read-only info about regional models

Observations now store a multi-identification payload (`identificationsJSON`) to support multiple detections per photo, while remaining backward-compatible with legacy single-label records.

## Deploying Firebase rule changes

After editing `firestore.rules` or `storage.rules`:

```bash
firebase deploy --only firestore:rules
firebase deploy --only storage
```

## Requirements

- **iOS:** 17.0+
- **Xcode:** 15+
- **ML pipeline:** Python 3.10+, PyTorch 2.0+, coremltools 7.0+ (CUDA optional; MPS supported on Apple Silicon)
