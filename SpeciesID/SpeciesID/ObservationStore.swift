import Foundation
import CoreData

struct LocalAlternativeMatch: Codable, Hashable {
    var speciesId: String
    var displayName: String
    var confidenceScore: Double
}

struct LocalBoundingBox: Codable, Hashable {
    var x: Double
    var y: Double
    var width: Double
    var height: Double
}

struct LocalSpeciesIdentification: Codable, Hashable, Identifiable {
    var id: String
    var speciesId: String
    var displayName: String
    var confidenceScore: Double
    var modelVersion: String
    var isUserVerified: Bool
    var alternativeSpecies: [LocalAlternativeMatch]
    var boundingBox: LocalBoundingBox?
}

// MARK: - Core Data Stack

class ObservationStore {
    static let shared = ObservationStore()

    lazy var container: NSPersistentContainer = {
        let container = NSPersistentContainer(name: "ObservationStore", managedObjectModel: Self.model)
        if let description = container.persistentStoreDescriptions.first {
            description.setOption(true as NSNumber, forKey: NSMigratePersistentStoresAutomaticallyOption)
            description.setOption(true as NSNumber, forKey: NSInferMappingModelAutomaticallyOption)
        }
        container.loadPersistentStores { _, error in
            if let error = error {
                print("Core Data error: \(error)")
            }
        }
        return container
    }()

    var context: NSManagedObjectContext {
        container.viewContext
    }

    // Programmatic Core Data model (no .xcdatamodeld file needed)
    static let model: NSManagedObjectModel = {
        let model = NSManagedObjectModel()

        // SavedObservation entity
        let entity = NSEntityDescription()
        entity.name = "SavedObservation"
        entity.managedObjectClassName = "SavedObservation"

        let id = NSAttributeDescription()
        id.name = "id"
        id.attributeType = .UUIDAttributeType

        let timestamp = NSAttributeDescription()
        timestamp.name = "timestamp"
        timestamp.attributeType = .dateAttributeType

        let latitude = NSAttributeDescription()
        latitude.name = "latitude"
        latitude.attributeType = .doubleAttributeType

        let longitude = NSAttributeDescription()
        longitude.name = "longitude"
        longitude.attributeType = .doubleAttributeType

        let speciesId = NSAttributeDescription()
        speciesId.name = "speciesId"
        speciesId.attributeType = .stringAttributeType
        speciesId.isOptional = true

        let confidence = NSAttributeDescription()
        confidence.name = "confidence"
        confidence.attributeType = .doubleAttributeType

        let identificationsJSON = NSAttributeDescription()
        identificationsJSON.name = "identificationsJSON"
        identificationsJSON.attributeType = .stringAttributeType
        identificationsJSON.isOptional = true

        let imagePath = NSAttributeDescription()
        imagePath.name = "imagePath"
        imagePath.attributeType = .stringAttributeType
        imagePath.isOptional = true

        let notes = NSAttributeDescription()
        notes.name = "notes"
        notes.attributeType = .stringAttributeType
        notes.isOptional = true

        let synced = NSAttributeDescription()
        synced.name = "synced"
        synced.attributeType = .booleanAttributeType
        synced.defaultValue = false

        entity.properties = [
            id,
            timestamp,
            latitude,
            longitude,
            speciesId,
            confidence,
            identificationsJSON,
            imagePath,
            notes,
            synced,
        ]
        model.entities = [entity]

        return model
    }()

    // MARK: - CRUD Operations

    func saveObservation(
        latitude: Double,
        longitude: Double,
        speciesId: String?,
        confidence: Double,
        identifications: [LocalSpeciesIdentification] = [],
        imagePath: String?,
        notes: String?
    ) -> UUID {
        let observation = SavedObservation(context: context)
        observation.id = UUID()
        observation.timestamp = Date()
        observation.latitude = latitude
        observation.longitude = longitude
        observation.speciesId = speciesId
        observation.confidence = confidence
        observation.identificationsJSON = encodeIdentifications(identifications)
        observation.imagePath = imagePath
        observation.notes = notes
        observation.synced = false

        save()
        return observation.id!
    }
    func deleteAllObservations() {
        // Delete associated images first
        for obs in getAllObservations() {
            if let path = obs.imagePath {
                ImageStore.shared.deleteImage(at: path)
            }
        }
        
        let request: NSFetchRequest<NSFetchRequestResult> = NSFetchRequest(entityName: "SavedObservation")
        let deleteRequest = NSBatchDeleteRequest(fetchRequest: request)
        
        do {
            try context.execute(deleteRequest)
            context.reset()  // Clear in-memory cached objects
            try context.save()
        } catch {
            print("Failed to delete all observations: \(error)")
        }
    }

    func getAllObservations() -> [SavedObservation] {
        let request = NSFetchRequest<SavedObservation>(entityName: "SavedObservation")
        request.sortDescriptors = [NSSortDescriptor(key: "timestamp", ascending: false)]
        return (try? context.fetch(request)) ?? []
    }

    func getObservation(id: UUID) -> SavedObservation? {
        let request = NSFetchRequest<SavedObservation>(entityName: "SavedObservation")
        request.predicate = NSPredicate(format: "id == %@", id as CVarArg)
        return try? context.fetch(request).first
    }

    func deleteObservation(_ observation: SavedObservation) {
        // Delete associated image
        if let path = observation.imagePath {
            ImageStore.shared.deleteImage(at: path)
        }
        context.delete(observation)
        save()
    }

    func getObservations(from startDate: Date?, to endDate: Date?) -> [SavedObservation] {
        let request = NSFetchRequest<SavedObservation>(entityName: "SavedObservation")
        request.sortDescriptors = [NSSortDescriptor(key: "timestamp", ascending: false)]

        var predicates: [NSPredicate] = []
        if let start = startDate {
            predicates.append(NSPredicate(format: "timestamp >= %@", start as NSDate))
        }
        if let end = endDate {
            predicates.append(NSPredicate(format: "timestamp <= %@", end as NSDate))
        }
        if !predicates.isEmpty {
            request.predicate = NSCompoundPredicate(andPredicateWithSubpredicates: predicates)
        }

        return (try? context.fetch(request)) ?? []
    }

    func getUnsyncedObservations() -> [SavedObservation] {
        let request = NSFetchRequest<SavedObservation>(entityName: "SavedObservation")
        request.predicate = NSPredicate(format: "synced == NO")
        return (try? context.fetch(request)) ?? []
    }

    func markAsSynced(_ observation: SavedObservation) {
        observation.synced = true
        save()
    }

    func identifications(for observation: SavedObservation) -> [LocalSpeciesIdentification] {
        if let json = observation.identificationsJSON,
           let data = json.data(using: .utf8),
           let decoded = try? JSONDecoder().decode([LocalSpeciesIdentification].self, from: data),
           !decoded.isEmpty {
            return decoded.sorted { $0.confidenceScore > $1.confidenceScore }
        }

        // Backward-compatible fallback for records saved before multi-species support.
        if let species = observation.speciesId, !species.isEmpty {
            return [
                LocalSpeciesIdentification(
                    id: UUID().uuidString,
                    speciesId: species,
                    displayName: species,
                    confidenceScore: observation.confidence,
                    modelVersion: "legacy-single-label",
                    isUserVerified: false,
                    alternativeSpecies: [],
                    boundingBox: nil
                )
            ]
        }

        return []
    }

    func primaryIdentification(for observation: SavedObservation) -> LocalSpeciesIdentification? {
        identifications(for: observation).first
    }

    func observationCount(for speciesId: String) -> Int {
        let all = getAllObservations()
        return all.filter { obs in
            if let ids = identifications(for: obs).first {
                return ids.speciesId == speciesId
            }
            return obs.speciesId == speciesId
        }.count
    }

    func observationCounts() -> [String: Int] {
        var counts: [String: Int] = [:]
        for obs in getAllObservations() {
            let species: String
            if let primary = identifications(for: obs).first {
                species = primary.speciesId
            } else if let sid = obs.speciesId {
                species = sid
            } else {
                continue
            }
            counts[species, default: 0] += 1
        }
        return counts
    }

    private func save() {
        if context.hasChanges {
            try? context.save()
        }
    }

    private func encodeIdentifications(_ identifications: [LocalSpeciesIdentification]) -> String? {
        guard !identifications.isEmpty,
              let data = try? JSONEncoder().encode(identifications) else {
            return nil
        }
        return String(data: data, encoding: .utf8)
    }
}

// MARK: - Core Data Entity

@objc(SavedObservation)
public class SavedObservation: NSManagedObject {
    @NSManaged public var id: UUID?
    @NSManaged public var timestamp: Date?
    @NSManaged public var latitude: Double
    @NSManaged public var longitude: Double
    @NSManaged public var speciesId: String?
    @NSManaged public var confidence: Double
    @NSManaged public var identificationsJSON: String?
    @NSManaged public var imagePath: String?
    @NSManaged public var notes: String?
    @NSManaged public var synced: Bool
}
