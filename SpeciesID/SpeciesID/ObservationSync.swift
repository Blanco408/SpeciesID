import Foundation

/// Syncs local observations to Firebase
/// Uses the existing ObservationRepository from Backend
class ObservationSync {
    static let shared = ObservationSync()

    private let repository = ObservationRepository()

    /// Syncs all unsynced local observations to Firebase
    func syncPendingObservations(userId: String) async {
        let unsynced = ObservationStore.shared.getUnsyncedObservations()

        for local in unsynced {
            do {
                // Convert local observation to Firebase model
                let coords = GeoCoordinates(
                    latitude: local.latitude,
                    longitude: local.longitude
                )

                // Build identification if we have species data
                var identifications: [SpeciesIdentification] = []
                if let speciesId = local.speciesId {
                    identifications.append(SpeciesIdentification(
                        speciesId: speciesId,
                        commonName: nil,
                        scientificName: nil,
                        confidence: local.confidence,
                        source: .userIdentified,
                        isUserVerified: false
                    ))
                }

                // Upload to Firebase
                _ = try await repository.createOfflineObservation(
                    userId: userId,
                    coordinates: coords,
                    identifications: identifications,
                    regionName: nil,
                    notes: local.notes
                )

                // Mark as synced locally
                ObservationStore.shared.markAsSynced(local)
                print("Synced observation: \(local.id?.uuidString ?? "unknown")")

            } catch {
                print("Failed to sync observation: \(error)")
            }
        }
    }

    /// Call this when app comes online or on app launch
    func syncIfNeeded(userId: String) {
        Task {
            await syncPendingObservations(userId: userId)
        }
    }
}
