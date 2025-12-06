import Foundation
import FirebaseFirestore
import FirebaseFirestoreSwift

/// Repository for Observation data operations
/// Implements offline-first architecture with sync status tracking
actor ObservationRepository {
    
    private let db = Firestore.firestore()
    private let collectionName = "observations"
    
    private var observationsCollection: CollectionReference {
        db.collection(collectionName)
    }
    
    // MARK: - Initialization
    
    init() {
        // Enable offline persistence (enabled by default on iOS, but explicit is good)
        let settings = FirestoreSettings()
        settings.cacheSettings = PersistentCacheSettings(sizeBytes: 100 * 1024 * 1024 as NSNumber) // 100MB cache
        db.settings = settings
    }
    
    // MARK: - Create
    
    /// Creates a new observation
    func createObservation(_ observation: Observation) async throws -> Observation {
        var newObservation = observation
        newObservation.syncStatus = .pending
        newObservation.lastModified = Date()
        
        let documentRef = observationsCollection.document()
        try documentRef.setData(from: newObservation)
        
        newObservation.id = documentRef.documentID
        return newObservation
    }
    
    /// Creates an observation from local/offline data
    func createOfflineObservation(
        userId: String,
        coordinates: GeoCoordinates,
        identifications: [SpeciesIdentification],
        regionName: String?,
        notes: String?
    ) async throws -> Observation {
        let observation = Observation(
            userId: userId,
            coordinates: coordinates,
            regionName: regionName,
            identifications: identifications,
            notes: notes,
            conditions: nil,
            syncStatus: .pending,
            lastModified: Date()
        )
        
        return try await createObservation(observation)
    }
    
    // MARK: - Read
    
    /// Fetches a single observation by ID
    func getObservation(observationId: String) async throws -> Observation? {
        let document = try await observationsCollection.document(observationId).getDocument()
        return try document.data(as: Observation.self)
    }
    
    /// Fetches all observations for a user, ordered by timestamp
    func getObservations(userId: String, limit: Int? = nil) async throws -> [Observation] {
        var query: Query = observationsCollection
            .whereField("user_id", isEqualTo: userId)
            .order(by: "timestamp", descending: true)
        
        if let limit = limit {
            query = query.limit(to: limit)
        }
        
        let snapshot = try await query.getDocuments()
        return snapshot.documents.compactMap { document in
            try? document.data(as: Observation.self)
        }
    }
    
    /// Fetches observations for a user filtered by region
    func getObservations(userId: String, regionName: String) async throws -> [Observation] {
        let snapshot = try await observationsCollection
            .whereField("user_id", isEqualTo: userId)
            .whereField("region_name", isEqualTo: regionName)
            .order(by: "timestamp", descending: true)
            .getDocuments()
        
        return snapshot.documents.compactMap { document in
            try? document.data(as: Observation.self)
        }
    }
    
    /// Fetches observations that need to be synced
    func getPendingObservations(userId: String) async throws -> [Observation] {
        let snapshot = try await observationsCollection
            .whereField("user_id", isEqualTo: userId)
            .whereField("sync_status", in: [SyncStatus.pending.rawValue, SyncStatus.failed.rawValue])
            .getDocuments()
        
        return snapshot.documents.compactMap { document in
            try? document.data(as: Observation.self)
        }
    }
    
    /// Fetches observations within a date range
    func getObservations(userId: String, from startDate: Date, to endDate: Date) async throws -> [Observation] {
        let snapshot = try await observationsCollection
            .whereField("user_id", isEqualTo: userId)
            .whereField("timestamp", isGreaterThanOrEqualTo: startDate)
            .whereField("timestamp", isLessThanOrEqualTo: endDate)
            .order(by: "timestamp", descending: true)
            .getDocuments()
        
        return snapshot.documents.compactMap { document in
            try? document.data(as: Observation.self)
        }
    }
    
    // MARK: - Update
    
    /// Updates an observation's sync status
    func updateSyncStatus(observationId: String, status: SyncStatus) async throws {
        try await observationsCollection.document(observationId).updateData([
            "sync_status": status.rawValue,
            "last_modified": FieldValue.serverTimestamp()
        ])
    }
    
    /// Updates the user notes on an observation
    func updateNotes(observationId: String, notes: String?) async throws {
        try await observationsCollection.document(observationId).updateData([
            "notes": notes as Any,
            "last_modified": FieldValue.serverTimestamp()
        ])
    }
    
    /// Marks an identification as user-verified
    func verifyIdentification(observationId: String, identificationId: String) async throws {
        guard var observation = try await getObservation(observationId: observationId) else {
            throw RepositoryError.notFound
        }
        
        if let index = observation.identifications.firstIndex(where: { $0.id == identificationId }) {
            observation.identifications[index].isUserVerified = true
        }
        
        let encoded = try Firestore.Encoder().encode(observation.identifications)
        try await observationsCollection.document(observationId).updateData([
            "identifications": encoded,
            "last_modified": FieldValue.serverTimestamp()
        ])
    }
    
    // MARK: - Delete
    
    /// Deletes an observation and its photos subcollection
    func deleteObservation(observationId: String) async throws {
        let photosSnapshot = try await observationsCollection
            .document(observationId)
            .collection("photos")
            .getDocuments()
        
        let batch = db.batch()
        
        for photoDoc in photosSnapshot.documents {
            batch.deleteDocument(photoDoc.reference)
        }
        
        batch.deleteDocument(observationsCollection.document(observationId))
        
        try await batch.commit()
    }
    
    // MARK: - Real-time Updates
    
    /// Creates a listener for real-time updates to user's observations
    func observeObservations(userId: String, onChange: @escaping (Result<[Observation], Error>) -> Void) -> ListenerRegistration {
        return observationsCollection
            .whereField("user_id", isEqualTo: userId)
            .order(by: "timestamp", descending: true)
            .addSnapshotListener { querySnapshot, error in
                if let error = error {
                    onChange(.failure(error))
                    return
                }
                
                guard let documents = querySnapshot?.documents else {
                    onChange(.success([]))
                    return
                }
                
                let observations = documents.compactMap { document in
                    try? document.data(as: Observation.self)
                }
                
                onChange(.success(observations))
            }
    }
    
    // MARK: - Batch Operations
    
    /// Syncs multiple pending observations in a batch
    func batchSync(observations: [Observation]) async throws {
        let batch = db.batch()
        
        for observation in observations {
            guard let id = observation.id else { continue }
            
            let docRef = observationsCollection.document(id)
            batch.updateData([
                "sync_status": SyncStatus.synced.rawValue,
                "last_modified": FieldValue.serverTimestamp()
            ], forDocument: docRef)
        }
        
        try await batch.commit()
    }
}

// MARK: - Errors

enum RepositoryError: LocalizedError {
    case notFound
    case invalidData
    case syncFailed
    
    var errorDescription: String? {
        switch self {
        case .notFound:
            return "The requested document was not found"
        case .invalidData:
            return "The document data is invalid or corrupted"
        case .syncFailed:
            return "Failed to sync data with the server"
        }
    }
}