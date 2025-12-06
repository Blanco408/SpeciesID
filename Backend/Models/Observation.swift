import Foundation
import FirebaseFirestore
import FirebaseFirestoreSwift

/// Represents a single species observation recorded in the field
/// Firestore path: observations/{observationId}
struct Observation: Codable, Identifiable, Hashable {
    @DocumentID var id: String?
    
    /// Reference to the user who made this observation
    var userId: String
    
    /// When the observation was recorded
    @ServerTimestamp var timestamp: Date?
    
    /// GPS coordinates where the observation was made
    var coordinates: GeoCoordinates
    
    /// Region name for filtering (e.g., "Southern California Tidal Pools")
    var regionName: String?
    
    /// Species identified in this observation (can be multiple)
    var identifications: [SpeciesIdentification]
    
    /// User-provided notes about the observation
    var notes: String?
    
    /// Weather/environmental conditions (optional metadata)
    var conditions: EnvironmentalConditions?
    
    /// Sync status for offline-first functionality
    var syncStatus: SyncStatus
    
    /// When the observation was last modified locally
    var lastModified: Date
    
    enum CodingKeys: String, CodingKey {
        case id
        case userId = "user_id"
        case timestamp
        case coordinates
        case regionName = "region_name"
        case identifications
        case notes
        case conditions
        case syncStatus = "sync_status"
        case lastModified = "last_modified"
    }
}

/// GPS coordinates with precision tracking
struct GeoCoordinates: Codable, Hashable {
    var latitude: Double
    var longitude: Double
    var altitude: Double?
    var horizontalAccuracy: Double? // meters
    
    enum CodingKeys: String, CodingKey {
        case latitude
        case longitude
        case altitude
        case horizontalAccuracy = "horizontal_accuracy"
    }
    
    /// Convert to Firestore GeoPoint for geoqueries
    var geoPoint: GeoPoint {
        GeoPoint(latitude: latitude, longitude: longitude)
    }
}

/// Represents a single species identification within an observation
struct SpeciesIdentification: Codable, Hashable, Identifiable {
    var id: String // UUID string
    var speciesId: String
    var confidenceScore: Double // 0.0 - 1.0
    var modelVersion: String
    var isUserVerified: Bool
    var alternativeSpecies: [AlternativeMatch]?
    
    enum CodingKeys: String, CodingKey {
        case id
        case speciesId = "species_id"
        case confidenceScore = "confidence_score"
        case modelVersion = "model_version"
        case isUserVerified = "is_user_verified"
        case alternativeSpecies = "alternative_species"
    }
}

/// Alternative species matches for ambiguous identifications
struct AlternativeMatch: Codable, Hashable {
    var speciesId: String
    var confidenceScore: Double
    
    enum CodingKeys: String, CodingKey {
        case speciesId = "species_id"
        case confidenceScore = "confidence_score"
    }
}

/// Environmental conditions at time of observation
struct EnvironmentalConditions: Codable, Hashable {
    var tideLevel: TideLevel?
    var lightCondition: LightCondition?
    var weather: String?
    
    enum CodingKeys: String, CodingKey {
        case tideLevel = "tide_level"
        case lightCondition = "light_condition"
        case weather
    }
}

enum TideLevel: String, Codable {
    case high
    case mid
    case low
    case unknown
}

enum LightCondition: String, Codable {
    case bright
    case overcast
    case shaded
    case lowLight = "low_light"
}

/// Tracks sync status for offline-first architecture
enum SyncStatus: String, Codable {
    case pending    // Created offline, not yet synced
    case syncing    // Currently uploading
    case synced     // Successfully synced to cloud
    case failed     // Sync failed, will retry
    case conflict   // Server has different version
}