import Foundation
import FirebaseFirestoreSwift

/// Represents a geographic region with a downloadable species model
/// Firestore path: regions/{regionId}
struct Region: Codable, Identifiable, Hashable {
    @DocumentID var id: String?
    
    /// Human-readable region name
    var name: String
    
    /// Detailed description of the region's coverage
    var description: String?
    
    /// Approximate center point for map display
    var centerCoordinates: GeoCoordinates?
    
    /// Bounding box for the region
    var boundingBox: BoundingBox?
    
    /// Number of species in this region's database
    var speciesCount: Int
    
    /// Species IDs included in this region
    var speciesIds: [String]
    
    /// ML model information
    var modelInfo: ModelInfo
    
    /// When this region's data was last updated
    var lastUpdated: Date
    
    /// Whether this region is actively maintained
    var isActive: Bool
    
    enum CodingKeys: String, CodingKey {
        case id
        case name
        case description
        case centerCoordinates = "center_coordinates"
        case boundingBox = "bounding_box"
        case speciesCount = "species_count"
        case speciesIds = "species_ids"
        case modelInfo = "model_info"
        case lastUpdated = "last_updated"
        case isActive = "is_active"
    }
}

/// Geographic bounding box
struct BoundingBox: Codable, Hashable {
    var northLat: Double
    var southLat: Double
    var eastLng: Double
    var westLng: Double
    
    enum CodingKeys: String, CodingKey {
        case northLat = "north_lat"
        case southLat = "south_lat"
        case eastLng = "east_lng"
        case westLng = "west_lng"
    }
    
    func contains(latitude: Double, longitude: Double) -> Bool {
        latitude >= southLat && latitude <= northLat &&
        longitude >= westLng && longitude <= eastLng
    }
}

/// Information about the ML model for this region
struct ModelInfo: Codable, Hashable {
    /// Semantic version (e.g., "1.2.0")
    var version: String
    
    /// Firebase Storage URL for the Core ML model file
    var downloadUrl: String
    
    /// Model file size in bytes (for download progress)
    var fileSize: Int
    
    /// SHA-256 hash for integrity verification
    var checksum: String
    
    /// Minimum iOS version required
    var minIosVersion: String
    
    /// Model accuracy metrics
    var accuracy: ModelAccuracy?
    
    enum CodingKeys: String, CodingKey {
        case version
        case downloadUrl = "download_url"
        case fileSize = "file_size"
        case checks