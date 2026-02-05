import Foundation
import CoreData

// MARK: - Core Data Stack

class ObservationStore {
    static let shared = ObservationStore()

    lazy var container: NSPersistentContainer = {
        let container = NSPersistentContainer(name: "ObservationStore", managedObjectModel: Self.model)
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

        entity.properties = [id, timestamp, latitude, longitude, speciesId, confidence, imagePath, notes, synced]
        model.entities = [entity]

        return model
    }()

    // MARK: - CRUD Operations

    func saveObservation(
        latitude: Double,
        longitude: Double,
        speciesId: String?,
        confidence: Double,
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
        observation.imagePath = imagePath
        observation.notes = notes
        observation.synced = false

        save()
        return observation.id!
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

    func getUnsyncedObservations() -> [SavedObservation] {
        let request = NSFetchRequest<SavedObservation>(entityName: "SavedObservation")
        request.predicate = NSPredicate(format: "synced == NO")
        return (try? context.fetch(request)) ?? []
    }

    func markAsSynced(_ observation: SavedObservation) {
        observation.synced = true
        save()
    }

    private func save() {
        if context.hasChanges {
            try? context.save()
        }
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
    @NSManaged public var imagePath: String?
    @NSManaged public var notes: String?
    @NSManaged public var synced: Bool
}
