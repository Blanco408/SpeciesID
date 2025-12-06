import Foundation
import FirebaseFirestore
import FirebaseFirestoreSwift

/// Represents a researcher or app user in the Species Identification system
/// Firestore path: users/{userId}
struct User: Codable, Identifiable, Hashable {
    @DocumentID var id: String?
    var email: String
    var displayName: String
    var dateCreated: Date
    var lastLogin: Date?
    
    /// Regions the user has downloaded models for
    var downloadedRegions: [String]
    
    /// User preferences for the app
    var preferences: UserPreferences?
    
    enum CodingKeys: String, CodingKey {
        case id
        case email
        case displayName = "display_name"
        case dateCreated = "date_created"
        case lastLogin = "last_login"
        case downloadedRegions = "downloaded_regions"
        case preferences
    }
}

struct UserPreferences: Codable, Hashable {
    var defaultRegion: String?
    var autoSavePhotos: Bool
    var highContrastMode: Bool
    var largeTextMode: Bool
    var confidenceThreshold: Double // 0.0 - 1.0
    
    enum CodingKeys: String, CodingKey {
        case defaultRegion = "default_region"
        case autoSavePhotos = "auto_save_photos"
        case highContrastMode = "high_contrast_mode"
        case largeTextMode = "large_text_mode"
        case confidenceThreshold = "confidence_threshold"
    }
    
    static let `default` = UserPreferences(
        defaultRegion: nil,
        autoSavePhotos: true,
        highContrastMode: false,
        largeTextMode: false,
        confidenceThreshold: 0.7
    )
}