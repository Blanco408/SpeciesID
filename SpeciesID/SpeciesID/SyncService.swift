import Combine
import Foundation
import UIKit
import FirebaseFirestore
import FirebaseStorage

@MainActor
final class SyncService: ObservableObject {
    @Published var syncState: SyncState = .idle
    @Published var pendingCount: Int = 0

    enum SyncState: Equatable {
        case idle, syncing, completed
        case failed(String)
    }

    private let db = Firestore.firestore()
    private let retryQueueKey = "syncRetryQueue"

    // MARK: - Retry Queue

    private var retryQueue: [String] {
        get { UserDefaults.standard.stringArray(forKey: retryQueueKey) ?? [] }
        set { UserDefaults.standard.set(newValue, forKey: retryQueueKey) }
    }

    private func addToRetryQueue(_ id: String) {
        var queue = retryQueue
        if !queue.contains(id) {
            queue.append(id)
            retryQueue = queue
        }
    }

    private func removeFromRetryQueue(_ id: String) {
        retryQueue = retryQueue.filter { $0 != id }
    }

    // MARK: - Orchestrator

    func sync(userId: String) async {
        syncState = .syncing
        do {
            try await syncObservations(userId: userId)
            syncState = .completed
        } catch {
            syncState = .failed(error.localizedDescription)
        }
    }

    // MARK: - Observation Sync

    private func syncObservations(userId: String) async throws {
        var toSync = ObservationStore.shared.getUnsyncedObservations()

        // Also retry previously failed IDs
        let alreadyIncluded = Set(toSync.compactMap { $0.id?.uuidString })
        for idString in retryQueue where !alreadyIncluded.contains(idString) {
            if let uuid = UUID(uuidString: idString),
               let obs = ObservationStore.shared.getObservation(id: uuid) {
                toSync.append(obs)
            }
        }

        pendingCount = toSync.count

        for saved in toSync {
            guard let localId = saved.id?.uuidString else { continue }
            do {
                let docRef = db.collection("observations").document()

                let localIdentifications = ObservationStore.shared.identifications(for: saved)

                var identifications: [[String: Any]] = localIdentifications.map { local in
                    var payload: [String: Any] = [
                        "id": local.id,
                        "species_id": local.speciesId,
                        "display_name": local.displayName,
                        "confidence_score": local.confidenceScore,
                        "model_version": local.modelVersion,
                        "is_user_verified": local.isUserVerified,
                        "alternative_species": local.alternativeSpecies.map {
                            [
                                "species_id": $0.speciesId,
                                "display_name": $0.displayName,
                                "confidence_score": $0.confidenceScore,
                            ]
                        },
                    ]
                    if let box = local.boundingBox {
                        payload["bounding_box"] = [
                            "x": box.x,
                            "y": box.y,
                            "width": box.width,
                            "height": box.height,
                        ]
                    }
                    return payload
                }

                // Backward compatibility for legacy records.
                if identifications.isEmpty, let speciesId = saved.speciesId {
                    identifications.append([
                        "id": UUID().uuidString,
                        "species_id": speciesId,
                        "display_name": speciesId,
                        "confidence_score": saved.confidence,
                        "model_version": "legacy-single-label",
                        "is_user_verified": false,
                    ])
                }

                let data: [String: Any] = [
                    "user_id": userId,
                    "coordinates": [
                        "latitude": saved.latitude,
                        "longitude": saved.longitude
                    ],
                    "identifications": identifications,
                    "notes": saved.notes as Any,
                    "sync_status": "pending",
                    "timestamp": FieldValue.serverTimestamp(),
                    "last_modified": FieldValue.serverTimestamp()
                ]

                try await docRef.setData(data)
                let observationId = docRef.documentID

                if let imagePath = saved.imagePath {
                    try await uploadPhoto(imagePath: imagePath, observationId: observationId, userId: userId)
                }

                ObservationStore.shared.markAsSynced(saved)
                removeFromRetryQueue(localId)
                pendingCount -= 1
            } catch {
                addToRetryQueue(localId)
                print("SyncService: failed to sync observation \(localId): \(error)")
            }
        }
    }

    // MARK: - Photo Upload

    private func uploadPhoto(imagePath: String, observationId: String, userId: String) async throws {
        guard let image = ImageStore.shared.loadImage(filename: imagePath),
              let data = image.jpegData(compressionQuality: 0.8) else {
            throw SyncError.imageLoadFailed
        }

        let photoId = UUID().uuidString
        let storagePath = "users/\(userId)/observations/\(observationId)/\(photoId).jpg"
        let storageRef = Storage.storage().reference().child(storagePath)

        let metadata = StorageMetadata()
        metadata.contentType = "image/jpeg"
        _ = try await storageRef.putDataAsync(data, metadata: metadata)
        let downloadURL = try await storageRef.downloadURL()

        let photoData: [String: Any] = [
            "observation_id": observationId,
            "storage_url": downloadURL.absoluteString,
            "local_path": imagePath,
            "upload_status": "uploaded",
            "sort_order": 0,
            "captured_at": FieldValue.serverTimestamp()
        ]

        try await db.collection("observations")
            .document(observationId)
            .collection("photos")
            .document(photoId)
            .setData(photoData)
    }

    // MARK: - Preference Sync

    func syncPreferences(userId: String, autoSavePhotos: Bool, confidenceThreshold: Double) async throws {
        try await db.collection("users").document(userId).updateData([
            "preferences": [
                "auto_save_photos": autoSavePhotos,
                "high_contrast_mode": false,
                "large_text_mode": false,
                "confidence_threshold": confidenceThreshold
            ]
        ])
    }
}

// MARK: - Errors

enum SyncError: LocalizedError {
    case imageLoadFailed

    var errorDescription: String? {
        switch self {
        case .imageLoadFailed:
            return "Failed to load image for upload"
        }
    }
}
