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

        // Create temp directory
        let exportId = UUID().uuidString
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("SpeciesID_Export_\(exportId)")

        do {
            try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        } catch {
            exportError = "Failed to create export directory."
            return nil
        }

        // Generate data file
        let dataFileURL: URL
        switch options.format {
        case .csv:
            dataFileURL = generateCSV(observations: observations, directory: tempDir)
        case .json:
            dataFileURL = generateJSON(observations: observations, directory: tempDir)
        }

        progress = 0.3

        // Copy photos if requested
        if options.includePhotos {
            let photosDir = tempDir.appendingPathComponent("photos")
            try? FileManager.default.createDirectory(at: photosDir, withIntermediateDirectories: true)

            for (index, obs) in observations.enumerated() {
                autoreleasepool {
                    if let imagePath = obs.imagePath,
                       let image = ImageStore.shared.loadImage(filename: imagePath) {
                        let photoURL = photosDir.appendingPathComponent(imagePath)
                        if let data = image.jpegData(compressionQuality: 0.8) {
                            try? data.write(to: photoURL)
                        }
                    }
                }
                progress = 0.3 + 0.5 * Double(index + 1) / Double(observations.count)
            }
        } else {
            progress = 0.8
        }

        // Create zip archive
        let zipFilename = "SpeciesID_Export_\(formattedDate(Date())).zip"
        let zipURL = FileManager.default.temporaryDirectory.appendingPathComponent(zipFilename)

        // Remove existing zip if present
        try? FileManager.default.removeItem(at: zipURL)

        let success = createZipArchive(source: tempDir, destination: zipURL)
        progress = 1.0

        // Cleanup temp directory
        try? FileManager.default.removeItem(at: tempDir)

        if success {
            return zipURL
        } else {
            exportError = "Failed to create zip archive."
            return nil
        }
    }

    func observationCount(from startDate: Date?, to endDate: Date?) -> Int {
        ObservationStore.shared.getObservations(from: startDate, to: endDate).count
    }

    // MARK: - CSV Generation

    private func generateCSV(observations: [SavedObservation], directory: URL) -> URL {
        let fileURL = directory.appendingPathComponent("observations.csv")

        var csvContent = "species_name,date,time,latitude,longitude,confidence,photo_filename,notes\n"

        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd"

        let timeFormatter = DateFormatter()
        timeFormatter.dateFormat = "HH:mm:ss"

        for obs in observations {
            let species = escapeCSV(obs.speciesId ?? "Unidentified")
            let date = obs.timestamp.map { dateFormatter.string(from: $0) } ?? ""
            let time = obs.timestamp.map { timeFormatter.string(from: $0) } ?? ""
            let lat = String(format: "%.6f", obs.latitude)
            let lon = String(format: "%.6f", obs.longitude)
            let confidence = String(format: "%.2f", obs.confidence)
            let photo = obs.imagePath ?? ""
            let notes = escapeCSV(obs.notes ?? "")

            csvContent += "\(species),\(date),\(time),\(lat),\(lon),\(confidence),\(photo),\(notes)\n"
        }

        try? csvContent.write(to: fileURL, atomically: true, encoding: .utf8)
        return fileURL
    }

    // MARK: - JSON Generation

    private func generateJSON(observations: [SavedObservation], directory: URL) -> URL {
        let fileURL = directory.appendingPathComponent("observations.json")

        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd"

        let timeFormatter = DateFormatter()
        timeFormatter.dateFormat = "HH:mm:ss"

        let isoFormatter = ISO8601DateFormatter()

        var obsArray: [[String: Any]] = []

        for obs in observations {
            var dict: [String: Any] = [
                "species_name": obs.speciesId ?? "Unidentified",
                "date": obs.timestamp.map { dateFormatter.string(from: $0) } ?? "",
                "time": obs.timestamp.map { timeFormatter.string(from: $0) } ?? "",
                "latitude": obs.latitude,
                "longitude": obs.longitude,
                "confidence": obs.confidence,
                "photo_filename": obs.imagePath ?? "",
            ]
            if let notes = obs.notes, !notes.isEmpty {
                dict["notes"] = notes
            }
            obsArray.append(dict)
        }

        let exportDict: [String: Any] = [
            "export_date": isoFormatter.string(from: Date()),
            "observation_count": observations.count,
            "observations": obsArray,
        ]

        if let jsonData = try? JSONSerialization.data(withJSONObject: exportDict, options: .prettyPrinted) {
            try? jsonData.write(to: fileURL)
        }

        return fileURL
    }

    // MARK: - Zip Archive

    private func createZipArchive(source: URL, destination: URL) -> Bool {
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

    private func escapeCSV(_ string: String) -> String {
        if string.contains(",") || string.contains("\"") || string.contains("\n") {
            let escaped = string.replacingOccurrences(of: "\"", with: "\"\"")
            return "\"\(escaped)\""
        }
        return string
    }

    private func formattedDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        return formatter.string(from: date)
    }
}
