import Foundation
import FirebaseFirestore
import FirebaseFirestoreSwift

/// Repository for User data operations
/// Handles all Firestore interactions for the users collection
actor UserRepository {
    
    private let db = Firestore.firestore()
    private let collectionName = "users"
    
    private var usersCollection: CollectionReference {
        db.collection(collectionName)
    }
    
    // MARK: - Create
    
    /// Creates a new user document after authentication
    func createUser(userId: String, email: String, displayName: String) async throws -> User {
        let user = User(
            id: userId,
            email: email,
            displayName: displayName,
            dateCreated: Date(),
            lastLogin: Date(),
            downloadedRegions: [],
            preferences: .default
        )
        
        try usersCollection.document(userId).setData(from: user)
        return user
    }
    
    // MARK: - Read
    
    /// Fetches a user by their ID
    func getUser(userId: String) async throws -> User? {
        let document = try await usersCollection.document(userId).getDocument()
        return try document.data(as: User.self)
    }
    
    /// Checks if a user document exists
    func userExists(userId: String) async throws -> Bool {
        let document = try await usersCollection.document(userId).getDocument()
        return document.exists
    }
    
    // MARK: - Update
    
    /// Updates the user's last login timestamp
    func updateLastLogin(userId: String) async throws {
        try await usersCollection.document(userId).updateData([
            "last_login": FieldValue.serverTimestamp()
        ])
    }
    
    /// Updates the user's display name
    func updateDisplayName(userId: String, displayName: String) async throws {
        try await usersCollection.document(userId).updateData([
            "display_name": displayName
        ])
    }
    
    /// Updates the user's preferences
    func updatePreferences(userId: String, preferences: UserPreferences) async throws {
        let encoded = try Firestore.Encoder().encode(preferences)
        try await usersCollection.document(userId).updateData([
            "preferences": encoded
        ])
    }
    
    /// Adds a region to the user's downloaded regions
    func addDownloadedRegion(userId: String, regionId: String) async throws {
        try await usersCollection.document(userId).updateData([
            "downloaded_regions": FieldValue.arrayUnion([regionId])
        ])
    }
    
    /// Removes a region from the user's downloaded regions
    func removeDownloadedRegion(userId: String, regionId: String) async throws {
        try await usersCollection.document(userId).updateData([
            "downloaded_regions": FieldValue.arrayRemove([regionId])
        ])
    }
    
    // MARK: - Real-time Updates
    
    /// Creates a listener for real-time user document updates
    func observeUser(userId: String, onChange: @escaping (Result<User?, Error>) -> Void) -> ListenerRegistration {
        return usersCollection.document(userId).addSnapshotListener { documentSnapshot, error in
            if let error = error {
                onChange(.failure(error))
                return
            }
            
            guard let document = documentSnapshot else {
                onChange(.success(nil))
                return
            }
            
            do {
                let user = try document.data(as: User.self)
                onChange(.success(user))
            } catch {
                onChange(.failure(error))
            }
        }
    }
}