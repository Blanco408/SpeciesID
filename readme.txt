# SpeciesID

iOS app for identifying marine species in the field, works offline.

## What I Set Up (Henry)

I got the backend database working with Firebase:

- **Firestore** - our database, has offline sync built in
- **Firebase Storage** - where photos and ML models will go
- **Firebase Auth** - ready for login stuff (not implemented yet, just configured)
- **Security rules** - so users can only see their own data

Also created the Xcode project with Firebase SDK already added.

---

## Setup (for the team)

1. Pull the repo

2. Get the Firebase config file:
   - Go to https://console.firebase.google.com
   - Open **SpeciesID** project (DM me your gmail if you don't have access)
   - Gear icon → Project Settings → scroll down to "Your apps" → iOS app
   - Download `GoogleService-Info.plist`

3. Open `SpeciesID/SpeciesID.xcodeproj` in Xcode

4. Drag the `GoogleService-Info.plist` you downloaded into the SpeciesID folder in Xcode (the one with ContentView.swift). Check "Copy items if needed"

5. Click the project at the top of the sidebar → Signing & Capabilities → pick your Apple ID under Team

6. `Cmd + B` to build, should work with no errors

---

## Project Structure
```
SepeciesID/
├── Backend/
│   ├── Models/                  # data structures
│   │   ├── User.swift
│   │   ├── Observation.swift
│   │   ├── Photo.swift
│   │   ├── Species.swift
│   │   └── Region.swift
│   └── Repositories/            # database operations
│       ├── UserRepository.swift
│       └── ObservationRepository.swift
├── SpeciesID/                   # xcode project
├── firestore.rules              # security rules
├── firestore.indexes.json
└── storage.rules
```

---

## Database Collections

**users/{userId}**
- email, display_name, date_created, last_login, downloaded_regions, preferences

**observations/{observationId}**
- user_id, timestamp, coordinates, region_name, identifications, notes, sync_status

**observations/{observationId}/photos/{photoId}**
- storage_url, thumbnail_url, local_path, upload_status

**species/{speciesId}** - read only, the species database

**regions/{regionId}** - read only, info about regional ML models

---

## Using the Repositories

After someone implements auth, you'd create a user profile like this:
```swift
let userRepo = UserRepository()
let user = try await userRepo.createUser(
    userId: "firebase-auth-uid", 
    email: "test@test.com",
    displayName: "Test User"
)
```

Creating an observation:
```swift
let obsRepo = ObservationRepository()
let obs = try await obsRepo.createOfflineObservation(
    userId: currentUserId,
    coordinates: GeoCoordinates(latitude: 32.88, longitude: -117.23),
    identifications: [...],
    regionName: "Southern California",
    notes: "Found in tide pool"
)
```

Check the repository files for all the available methods.

---

## What Still Needs to Be Done

| Task | Status |
|------|--------|
| Database/Backend | ✅ done (Henry) |
| Auth (login/signup UI + Firebase Auth) | not started |
| Camera/photo capture | not started |
| ML model stuff | not started |
| Observations list UI | not started |
| Export to CSV | not started |

---

## Deploying rule changes

If you edit firestore.rules or storage.rules:
```bash
firebase deploy --only firestore:rules
firebase deploy --only storage
```