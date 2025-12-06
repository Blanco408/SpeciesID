import Foundation
import FirebaseFirestoreSwift

/// Represents a species in the identification database
/// Firestore path: species/{speciesId}
/// Note: This collection is typically read-only for users, populated by admins/researchers
struct Species: Codable, Identifiable, Hashable {
    @DocumentID var id: String?
    
    /// Scientific binomial nomenclature (e.g., "Pisaster ochraceus")
    var scientificName: String
    
    /// Common name(s) (e.g., "Purple Sea Star", "Ochre Star")
    var commonNames: [String]
    
    /// Taxonomic classification
    var taxonomy: Taxonomy
    
    /// Regions where this species is found
    var regions: [String]
    
    /// Brief description for field identification
    var description: String?
    
    /// Key identifying features
    var identifyingFeatures: [String]?
    
    /// Species that are commonly confused with this one
    var similarSpecies: [String]? // Array of species IDs
    
    /// Conservation status if applicable
    var conservationStatus: ConservationStatus?
    
    /// Reference images stored in Firebase Storage
    var referenceImageUrls: [String]?
    
    /// Whether this species should trigger location anonymization
    var isSensitive: Bool
    
    enum CodingKeys: String, CodingKey {
        case id
        case scientificName = "scientific_name"
        case commonNames = "common_names"
        case taxonomy
        case regions
        case description
        case identifyingFeatures = "identifying_features"
        case similarSpecies = "similar_species"
        case conservationStatus = "conservation_status"
        case referenceImageUrls = "reference_image_urls"
        case isSensitive = "is_sensitive"
    }
}

/// Taxonomic classification hierarchy
struct Taxonomy: Codable, Hashable {
    var kingdom: String
    var phylum: String
    var classname: String // 'class' is reserved in Swift
    var order: String
    var family: String
    var genus: String
    
    enum CodingKeys: String, CodingKey {
        case kingdom
        case phylum
        case classname = "class"
        case order
        case family
        case genus
    }
}

/// IUCN Conservation status
enum ConservationStatus: String, Codable {
    case notEvaluated = "NE"
    case dataDeficient = "DD"
    case leastConcern = "LC"
    case nearThreatened = "NT"
    case vulnerable = "VU"
    case endangered = "EN"
    case criticallyEndangered = "CR"
    case extinctInWild = "EW"
    case extinct = "EX"
    
    var displayName: String {
        switch self {
        case .notEvaluated: return "Not Evaluated"
        case .dataDeficient: return "Data Deficient"
        case .leastConcern: return "Least Concern"
        case .nearThreatened: return "Near Threatened"
        case .vulnerable: return "Vulnerable"
        case .endangered: return "Endangered"
        case .criticallyEndangered: return "Critically Endangered"
        case .extinctInWild: return "Extinct in Wild"
        case .extinct: return "Extinct"
        }
    }
}