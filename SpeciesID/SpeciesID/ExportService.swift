import Foundation
import UIKit
import UniformTypeIdentifiers
import Combine

// MARK: - Export Types

enum ExportFormat: String, CaseIterable {
    case csv = "CSV"
    case json = "JSON"
}

struct ExportOptions {
    var format: ExportFormat = .csv
    var startDate: Date?
    var endDate: Date?
    var includePhotos: Bool = true
}

// MARK: - Export Service

@MainActor
class ExportService: ObservableObject {
    @Published var isExporting = false
    @Published var progress: Double = 0.0
    @Published var exportError: String?

    func exportObservations(options: ExportOptions) async -> URL? {
        isExporting = true
        progress = 0.0
        exportError = nil
        defer { isExporting = false }

        let observations = ObservationStore.shared.getObservations(
            from: options.startDate,
            to: options.endDate
        )

        guard !observations.isEmpty else {
            exportError = "No observations found for the selected date range."
            return nil
        }

        // Snapshot observation data on the main thread before moving to background
        let snapshots = observations.map { makeSnapshot(from: $0) }

        let format = options.format
        let includePhotos = options.includePhotos

        return await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                // Create temp directory
                let exportId = UUID().uuidString
                let tempDir = FileManager.default.temporaryDirectory
                    .appendingPathComponent("SpeciesID_Export_\(exportId)")

                do {
                    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
                } catch {
                    DispatchQueue.main.async { self?.exportError = "Failed to create export directory." }
                    continuation.resume(returning: nil)
                    return
                }

                // Generate data file
                switch format {
                case .csv:
                    self?.generateCSVFromSnapshots(snapshots, directory: tempDir)
                case .json:
                    self?.generateJSONFromSnapshots(snapshots, directory: tempDir)
                }

                DispatchQueue.main.async { self?.progress = 0.3 }

                // Copy photos if requested
                if includePhotos {
                    let photosDir = tempDir.appendingPathComponent("photos")
                    try? FileManager.default.createDirectory(at: photosDir, withIntermediateDirectories: true)

                    for (index, snap) in snapshots.enumerated() {
                        autoreleasepool {
                            if let imagePath = snap.imagePath,
                               let image = ImageStore.shared.loadImage(filename: imagePath) {
                                let photoURL = photosDir.appendingPathComponent(imagePath)
                                if let data = image.jpegData(compressionQuality: 0.8) {
                                    try? data.write(to: photoURL)
                                }
                            }
                        }
                        let p = 0.3 + 0.5 * Double(index + 1) / Double(snapshots.count)
                        DispatchQueue.main.async { self?.progress = p }
                    }
                } else {
                    DispatchQueue.main.async { self?.progress = 0.8 }
                }

                // Create zip archive
                let dateFormatter = DateFormatter()
                dateFormatter.dateFormat = "yyyy-MM-dd"
                let zipFilename = "SpeciesID_Export_\(dateFormatter.string(from: Date())).zip"
                let zipURL = FileManager.default.temporaryDirectory.appendingPathComponent(zipFilename)
                try? FileManager.default.removeItem(at: zipURL)

                let success = self?.createZipArchive(source: tempDir, destination: zipURL) ?? false
                DispatchQueue.main.async { self?.progress = 1.0 }

                // Cleanup temp directory
                try? FileManager.default.removeItem(at: tempDir)

                if success {
                    continuation.resume(returning: zipURL)
                } else {
                    DispatchQueue.main.async { self?.exportError = "Failed to create zip archive." }
                    continuation.resume(returning: nil)
                }
            }
        }
    }

    func observationCount(from startDate: Date?, to endDate: Date?) -> Int {
        ObservationStore.shared.getObservations(from: startDate, to: endDate).count
    }

    // MARK: - Single Observation CSV Export

    func exportSingleObservation(_ observation: SavedObservation) -> URL? {
        exportError = nil

        let snapshot = makeSnapshot(from: observation)
        let csv = Self.csvHeader + Self.csvRows(for: snapshot)
        let filename = Self.csvFilename(for: snapshot)
        let fileURL = FileManager.default.temporaryDirectory.appendingPathComponent(filename)

        try? FileManager.default.removeItem(at: fileURL)

        do {
            try csv.write(to: fileURL, atomically: true, encoding: .utf8)
            return fileURL
        } catch {
            exportError = "Failed to write CSV file."
            return nil
        }
    }

    private func makeSnapshot(from observation: SavedObservation) -> ObsSnapshot {
        ObsSnapshot(
            occurrenceId: observation.id?.uuidString ?? UUID().uuidString,
            eventDate: observation.timestamp,
            latitude: observation.latitude,
            longitude: observation.longitude,
            identifications: ObservationStore.shared.identifications(for: observation),
            imagePath: observation.imagePath,
            notes: observation.notes
        )
    }

    // MARK: - CSV Generation (Darwin Core aligned, safe for background thread)

    private struct ObsSnapshot {
        let occurrenceId: String
        let eventDate: Date?
        let latitude: Double
        let longitude: Double
        let identifications: [LocalSpeciesIdentification]
        let imagePath: String?
        let notes: String?
    }

    /// Maps speciesId → scientific binomial. Loaded once from the bundled metadata file.
    private nonisolated static let scientificNamesByID: [String: String] = {
        guard let url = Bundle.main.url(forResource: "species_metadata", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let file = try? JSONDecoder().decode(SpeciesMetadataFile.self, from: data) else {
            return [:]
        }
        return Dictionary(uniqueKeysWithValues: file.species.map { ($0.id, $0.scientificName) })
    }()

    private nonisolated static let csvColumns: [String] = [
        "occurrence_id",
        "event_date",
        "decimal_latitude",
        "decimal_longitude",
        "geodetic_datum",
        "scientific_name",
        "vernacular_name",
        "species_id",
        "individual_count",
        "basis_of_record",
        "identified_by",
        "model_version",
        "confidence",
        "is_user_verified",
        "alternative_predictions",
        "photo_filename",
        "notes",
    ]

    private nonisolated static let csvHeader = csvColumns.joined(separator: ",") + "\n"

    /// Emits one CSV row per identification (multi-species observations produce multiple rows
    /// sharing the same occurrence_id). Falls back to a single empty-species row when no
    /// identifications exist.
    private nonisolated static func csvRows(for snap: ObsSnapshot) -> String {
        let isoFormatter = ISO8601DateFormatter()
        isoFormatter.formatOptions = [.withInternetDateTime]

        let eventDate = snap.eventDate.map { isoFormatter.string(from: $0) } ?? ""
        let hasCoords = !(snap.latitude == 0 && snap.longitude == 0)
        let lat = hasCoords ? String(format: "%.6f", snap.latitude) : ""
        let lon = hasCoords ? String(format: "%.6f", snap.longitude) : ""
        let datum = hasCoords ? "WGS84" : ""
        let photoFilename = escapeCSV(snap.imagePath ?? "")
        let notes = escapeCSV(snap.notes ?? "")

        guard !snap.identifications.isEmpty else {
            return joinRow([
                snap.occurrenceId, eventDate, lat, lon, datum,
                "", "", "", "1",
                "HumanObservation", "", "", "", "false",
                "",
                photoFilename, notes,
            ])
        }

        var rows = ""
        for ident in snap.identifications {
            let scientificName = scientificNamesByID[ident.speciesId] ?? ""
            let basis = ident.isUserVerified ? "HumanObservation" : "MachineObservation"
            let identifiedBy = ident.isUserVerified ? "User" : "EcoSnap ML"
            let confidence = String(format: "%.4f", ident.confidenceScore)
            let alternatives = ident.alternativeSpecies
                .map { "\($0.displayName) (\(Int(round($0.confidenceScore * 100)))%)" }
                .joined(separator: "; ")

            rows += joinRow([
                snap.occurrenceId,
                eventDate,
                lat,
                lon,
                datum,
                escapeCSV(scientificName),
                escapeCSV(ident.displayName),
                escapeCSV(ident.speciesId),
                "1",
                basis,
                identifiedBy,
                escapeCSV(ident.modelVersion),
                confidence,
                ident.isUserVerified ? "true" : "false",
                escapeCSV(alternatives),
                photoFilename,
                notes,
            ])
        }
        return rows
    }

    private nonisolated static func joinRow(_ values: [String]) -> String {
        values.joined(separator: ",") + "\n"
    }

    private nonisolated static func csvFilename(for snap: ObsSnapshot) -> String {
        let primary = snap.identifications.first?.displayName ?? "Observation"
        let speciesPart = primary
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { !$0.isEmpty }
            .joined(separator: "_")
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd_HHmmss"
        let dateStr = dateFormatter.string(from: snap.eventDate ?? Date())
        return "\(speciesPart)_\(dateStr).csv"
    }

    private nonisolated func generateCSVFromSnapshots(_ snapshots: [ObsSnapshot], directory: URL) {
        let fileURL = directory.appendingPathComponent("observations.csv")
        var csvContent = Self.csvHeader
        for snap in snapshots {
            csvContent += Self.csvRows(for: snap)
        }
        try? csvContent.write(to: fileURL, atomically: true, encoding: .utf8)
    }

    // MARK: - JSON Generation (Darwin Core aligned, safe for background thread)

    private nonisolated func generateJSONFromSnapshots(_ snapshots: [ObsSnapshot], directory: URL) {
        let fileURL = directory.appendingPathComponent("observations.json")

        let isoFormatter = ISO8601DateFormatter()
        isoFormatter.formatOptions = [.withInternetDateTime]

        let observationsPayload: [[String: Any]] = snapshots.map { obs in
            var dict: [String: Any] = [
                "occurrence_id": obs.occurrenceId,
                "event_date": obs.eventDate.map { isoFormatter.string(from: $0) } ?? "",
                "photo_filename": obs.imagePath ?? "",
                "identifications": obs.identifications.map { Self.identificationPayload($0) },
            ]

            let hasCoords = !(obs.latitude == 0 && obs.longitude == 0)
            if hasCoords {
                dict["decimal_latitude"] = obs.latitude
                dict["decimal_longitude"] = obs.longitude
                dict["geodetic_datum"] = "WGS84"
            }
            if let notes = obs.notes, !notes.isEmpty {
                dict["notes"] = notes
            }
            return dict
        }

        let exportDict: [String: Any] = [
            "export_date": isoFormatter.string(from: Date()),
            "observation_count": snapshots.count,
            "observations": observationsPayload,
        ]

        if let jsonData = try? JSONSerialization.data(withJSONObject: exportDict, options: [.prettyPrinted, .sortedKeys]) {
            try? jsonData.write(to: fileURL)
        }
    }

    private nonisolated static func identificationPayload(_ ident: LocalSpeciesIdentification) -> [String: Any] {
        var payload: [String: Any] = [
            "scientific_name": scientificNamesByID[ident.speciesId] ?? "",
            "vernacular_name": ident.displayName,
            "species_id": ident.speciesId,
            "individual_count": 1,
            "basis_of_record": ident.isUserVerified ? "HumanObservation" : "MachineObservation",
            "identified_by": ident.isUserVerified ? "User" : "EcoSnap ML",
            "model_version": ident.modelVersion,
            "confidence": ident.confidenceScore,
            "is_user_verified": ident.isUserVerified,
            "alternative_predictions": ident.alternativeSpecies.map {
                [
                    "scientific_name": scientificNamesByID[$0.speciesId] ?? "",
                    "vernacular_name": $0.displayName,
                    "species_id": $0.speciesId,
                    "confidence": $0.confidenceScore,
                ]
            },
        ]

        if let box = ident.boundingBox {
            payload["bounding_box"] = [
                "x": box.x,
                "y": box.y,
                "width": box.width,
                "height": box.height,
            ]
        }

        return payload
    }

    // MARK: - Zip Archive

    private nonisolated func createZipArchive(source: URL, destination: URL) -> Bool {
        let coordinator = NSFileCoordinator()
        var error: NSError?
        var success = false

        coordinator.coordinate(
            readingItemAt: source,
            options: [.forUploading],
            error: &error
        ) { zipURL in
            do {
                try FileManager.default.copyItem(at: zipURL, to: destination)
                success = true
            } catch {
                print("Zip error: \(error)")
            }
        }

        if let error {
            print("Coordinator error: \(error)")
        }

        return success
    }

    // MARK: - Helpers

    private nonisolated static func escapeCSV(_ string: String) -> String {
        if string.contains(",") || string.contains("\"") || string.contains("\n") {
            let escaped = string.replacingOccurrences(of: "\"", with: "\"\"")
            return "\"\(escaped)\""
        }
        return string
    }
}
