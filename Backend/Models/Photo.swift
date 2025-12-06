import Foundation
import FirebaseFirestoreSwift

/// Represents a photo attached to an observation
/// Firestore path: observations/{observationId}/photos/{photoId}
/// Actual image stored in Firebase Storage
struct Photo: Codable, Identifiable, Hashable {
    @DocumentID var id: String?
    
    /// Reference to parent observation
    var observationId: String
    
    /// When the photo was captured
    @ServerTimestamp var capturedAt: Date?
    
    /// Firebase Storage URL for the full-resolution image
    var storageUrl: String?
    
    /// Firebase Storage URL for thumbnail (for gallery performance)
    var thumbnailUrl: String?
    
    /// Local file path (for offline storage)
    var localPath: String?
    
    /// Image metadata
    var metadata: PhotoMetadata?
    
    /// Upload status for offline-first
    var uploadStatus: UploadStatus
    
    /// Order in observation (for multi-angle captures)
    var sortOrder: Int
    
    enum CodingKeys: String, CodingKey {
        case id
        case observationId = "observation_id"
        case capturedAt = "captured_at"
        case storageUrl = "storage_url"
        case thumbnailUrl = "thumbnail_url"
        case localPath = "local_path"
        case metadata
        case uploadStatus = "upload_status"
        case sortOrder = "sort_order"
    }
}

/// EXIF and capture metadata
struct PhotoMetadata: Codable, Hashable {
    var width: Int?
    var height: Int?
    var fileSize: Int? // bytes
    var mimeType: String?
    var deviceModel: String?
    var exposureTime: Double?
    var focalLength: Double?
    var iso: Int?
    
    enum CodingKeys: String, CodingKey {
        case width
        case height
        case fileSize = "file_size"
        case mimeType = "mime_type"
        case deviceModel = "device_model"
        case exposureTime = "exposure_time"
        case focalLength = "focal_length"
        case iso
    }
}

enum UploadStatus: String, Codable {
    case pending    // Waiting to upload
    case uploading  // Currently uploading
    case uploaded   // Successfully uploaded
    case failed     // Upload failed
}