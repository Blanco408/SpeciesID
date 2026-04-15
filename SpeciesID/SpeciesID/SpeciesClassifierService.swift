@preconcurrency import Vision
import CoreML
import UIKit
import Combine

// MARK: - Classification Models

struct AlternativePrediction: Hashable, Codable {
    let speciesId: String
    let displayName: String
    let confidence: Double
}

struct SupportedSpeciesItem: Identifiable, Hashable {
    let id: String
    let displayName: String
    let scientificName: String
    let iconName: String
    let category: String?
    let referenceImageUrl: String?
    let speciesDescription: String?
}

struct SpeciesDetection: Identifiable, Hashable {
    let id: UUID
    let speciesId: String
    let displayName: String
    let confidence: Double
    /// Normalized to the displayed image frame (0...1, origin at top-left)
    let boundingBox: CGRect
    let alternatives: [AlternativePrediction]
}

struct ClassificationResult {
    let detections: [SpeciesDetection]

    var primaryDetection: SpeciesDetection? {
        detections.first
    }

    // Backward-compatible convenience accessors.
    var speciesId: String {
        primaryDetection?.speciesId ?? "unknown"
    }

    var displayName: String {
        primaryDetection?.displayName ?? "Unidentified"
    }

    var confidence: Double {
        primaryDetection?.confidence ?? 0.0
    }

    var alternatives: [(speciesId: String, displayName: String, confidence: Double)] {
        primaryDetection?.alternatives.map {
            (speciesId: $0.speciesId, displayName: $0.displayName, confidence: $0.confidence)
        } ?? []
    }
}

// MARK: - Species Metadata (loaded from species_metadata.json)

struct SpeciesMetadataFile: Codable {
    let version: String
    let species: [SpeciesMetadataEntry]
}

struct SpeciesMetadataEntry: Codable {
    let id: String
    let displayName: String
    let scientificName: String
    let iconName: String
    let category: String?
    let referenceImageUrl: String?
    let description: String?
}

// MARK: - Species Classifier Service

@MainActor
class SpeciesClassifierService: ObservableObject {
    @Published var isClassifying = false
    @Published var lastResult: ClassificationResult?
    @Published var errorMessage: String?
    @Published private(set) var supportedSpecies: [SupportedSpeciesItem] = []

    private var vnModel: VNCoreMLModel?
    private var speciesMetadata: [String: SpeciesMetadataEntry] = [:]

    private let minimumDetectionConfidence = 0.55
    private let minimumTopMargin = 0.10
    private let fullFrameFallbackConfidence = 0.50
    private let maxDetections = 3
    private let overlapThreshold = 0.30
    /// Maximum entropy ratio (actual/uniform) above which we reject as "unknown"
    private let maxEntropyRatio = 0.65

    init() {
        loadSpeciesMetadata()
        loadModel()
    }

    private func loadSpeciesMetadata() {
        guard let url = Bundle.main.url(forResource: "species_metadata", withExtension: "json") else {
            print("species_metadata.json not found in bundle")
            return
        }
        do {
            let data = try Data(contentsOf: url)
            let file = try JSONDecoder().decode(SpeciesMetadataFile.self, from: data)
            speciesMetadata = Dictionary(uniqueKeysWithValues: file.species.map { ($0.id, $0) })
            print("Loaded species metadata v\(file.version) with \(file.species.count) species")
        } catch {
            print("Failed to load species metadata: \(error)")
        }
    }

    private func loadModel() {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all
            let model = try SpeciesClassifier(configuration: config)
            vnModel = try VNCoreMLModel(for: model.model)
            supportedSpecies = buildSupportedSpecies(from: model.model)
        } catch {
            errorMessage = "Failed to load ML model: \(error.localizedDescription)"
            print("ML model load error: \(error)")
            supportedSpecies = fallbackSpeciesCatalog()
        }
    }

    var isModelLoaded: Bool {
        vnModel != nil
    }

    func classify(image: UIImage) async -> ClassificationResult? {
        guard let vnModel else {
            errorMessage = "Model not loaded"
            return nil
        }
        guard let cgImage = image.cgImage else {
            errorMessage = "Invalid image"
            return nil
        }

        isClassifying = true
        errorMessage = nil
        defer { isClassifying = false }

        let metadata = self.speciesMetadata
        let minimumDetectionConfidence = self.minimumDetectionConfidence
        let minimumTopMargin = self.minimumTopMargin
        let fullFrameFallbackConfidence = self.fullFrameFallbackConfidence
        let maxDetections = self.maxDetections
        let overlapThreshold = self.overlapThreshold
        let maxEntropyRatio = self.maxEntropyRatio

        let result: ClassificationResult? = await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let fullFrame = CGRect(x: 0, y: 0, width: 1, height: 1)
                let windows = Self.candidateWindows()
                var detections: [SpeciesDetection] = []

                let fullFrameObservations = Self.classifyCrop(cgImage, using: vnModel)
                if let fallbackDetection = Self.buildDetection(
                    from: fullFrameObservations,
                    boundingBox: fullFrame,
                    metadata: metadata,
                    minimumConfidence: fullFrameFallbackConfidence,
                    minimumTopMargin: minimumTopMargin,
                    maxEntropyRatio: maxEntropyRatio
                ) {
                    detections.append(fallbackDetection)
                }

                for normalizedWindow in windows {
                    if normalizedWindow == fullFrame {
                        continue
                    }
                    let pixelRect = Self.pixelRect(
                        normalized: normalizedWindow,
                        imageWidth: cgImage.width,
                        imageHeight: cgImage.height
                    )
                    guard let crop = cgImage.cropping(to: pixelRect) else {
                        continue
                    }

                    let observations = Self.classifyCrop(crop, using: vnModel)
                    if let detection = Self.buildDetection(
                        from: observations,
                        boundingBox: normalizedWindow,
                        metadata: metadata,
                        minimumConfidence: minimumDetectionConfidence,
                        minimumTopMargin: minimumTopMargin,
                        maxEntropyRatio: maxEntropyRatio
                    ) {
                        detections.append(detection)
                    }
                }

                let merged = Self.mergeDetections(
                    detections,
                    iouThreshold: overlapThreshold,
                    maxDetections: maxDetections
                )

                let sorted = merged.sorted { $0.confidence > $1.confidence }
                let result = sorted.isEmpty ? nil : ClassificationResult(detections: sorted)
                continuation.resume(returning: result)
            }
        }
        self.lastResult = result
        return result
    }

    func scientificName(for speciesId: String) -> String {
        speciesMetadata[speciesId]?.scientificName ?? Self.prettifySpeciesName(speciesId)
    }

    private func buildSupportedSpecies(from model: MLModel) -> [SupportedSpeciesItem] {
        let labels = model.modelDescription.classLabels?.compactMap { $0 as? String } ?? []
        let sourceLabels = labels.isEmpty ? Array(speciesMetadata.keys).sorted() : labels

        return sourceLabels
            .map { speciesId in
                let entry = speciesMetadata[speciesId]
                return SupportedSpeciesItem(
                    id: speciesId,
                    displayName: entry?.displayName ?? Self.prettifySpeciesName(speciesId),
                    scientificName: entry?.scientificName ?? Self.prettifySpeciesName(speciesId),
                    iconName: entry?.iconName ?? "fish.fill",
                    category: entry?.category,
                    referenceImageUrl: entry?.referenceImageUrl,
                    speciesDescription: entry?.description
                )
            }
            .sorted { $0.displayName < $1.displayName }
    }

    private func fallbackSpeciesCatalog() -> [SupportedSpeciesItem] {
        Array(speciesMetadata.keys)
            .sorted()
            .map { speciesId in
                let entry = speciesMetadata[speciesId]
                return SupportedSpeciesItem(
                    id: speciesId,
                    displayName: entry?.displayName ?? Self.prettifySpeciesName(speciesId),
                    scientificName: entry?.scientificName ?? Self.prettifySpeciesName(speciesId),
                    iconName: entry?.iconName ?? "fish.fill",
                    category: entry?.category,
                    referenceImageUrl: entry?.referenceImageUrl,
                    speciesDescription: entry?.description
                )
            }
    }

    // MARK: - Multi-Species Windows

    private nonisolated static func candidateWindows() -> [CGRect] {
        var windows: [CGRect] = [CGRect(x: 0, y: 0, width: 1, height: 1)]

        // Coarse windows keep coverage high while reducing duplicate detections.
        windows.append(contentsOf: gridWindows(gridSize: 2, overlap: 0.25))

        // Center-focused windows to catch clustered organisms.
        windows.append(CGRect(x: 0.20, y: 0.20, width: 0.60, height: 0.60))

        // Dedupe by rounded coordinates.
        var seen = Set<String>()
        var deduped: [CGRect] = []
        for window in windows {
            let key = String(
                format: "%.3f_%.3f_%.3f_%.3f",
                window.origin.x,
                window.origin.y,
                window.size.width,
                window.size.height
            )
            if !seen.contains(key) {
                seen.insert(key)
                deduped.append(window)
            }
        }
        return deduped
    }

    private nonisolated static func buildDetection(
        from observations: [VNClassificationObservation],
        boundingBox: CGRect,
        metadata: [String: SpeciesMetadataEntry],
        minimumConfidence: Double,
        minimumTopMargin: Double,
        maxEntropyRatio: Double = 0.75
    ) -> SpeciesDetection? {
        guard let top = observations.first else {
            return nil
        }

        let confidence = min(max(Double(top.confidence), 0.0), 1.0)
        guard confidence >= minimumConfidence else {
            return nil
        }

        let secondConfidence = observations
            .dropFirst()
            .first
            .map { min(max(Double($0.confidence), 0.0), 1.0) } ?? 0.0
        let margin = confidence - secondConfidence
        guard margin >= minimumTopMargin else {
            return nil
        }

        // Entropy check: if probabilities are spread too evenly, the model doesn't know
        let numClasses = Double(observations.count)
        if numClasses > 1 {
            let entropy = observations.reduce(0.0) { sum, obs in
                let p = max(Double(obs.confidence), 1e-10)
                return sum - p * log(p)
            }
            let maxEntropy = log(numClasses)
            if maxEntropy > 0 && (entropy / maxEntropy) > maxEntropyRatio {
                return nil
            }
        }

        let alternatives = observations.dropFirst().prefix(2).map {
            AlternativePrediction(
                speciesId: $0.identifier,
                displayName: metadata[$0.identifier]?.displayName ?? Self.prettifySpeciesName($0.identifier),
                confidence: min(max(Double($0.confidence), 0.0), 1.0)
            )
        }

        return SpeciesDetection(
            id: UUID(),
            speciesId: top.identifier,
            displayName: metadata[top.identifier]?.displayName ?? Self.prettifySpeciesName(top.identifier),
            confidence: confidence,
            boundingBox: boundingBox,
            alternatives: alternatives
        )
    }

    private nonisolated static func gridWindows(gridSize: Int, overlap: CGFloat) -> [CGRect] {
        guard gridSize > 0 else { return [] }

        let step = 1.0 / CGFloat(gridSize)
        let windowSize = min(1.0, step * (1.0 + overlap))
        var windows: [CGRect] = []

        for row in 0..<gridSize {
            for col in 0..<gridSize {
                let baseX = CGFloat(col) * step
                let baseY = CGFloat(row) * step
                let x = clamp(baseX - (windowSize - step) / 2.0, min: 0.0, max: 1.0 - windowSize)
                let y = clamp(baseY - (windowSize - step) / 2.0, min: 0.0, max: 1.0 - windowSize)
                windows.append(CGRect(x: x, y: y, width: windowSize, height: windowSize))
            }
        }

        return windows
    }

    private nonisolated static func classifyCrop(_ cgImage: CGImage, using vnModel: VNCoreMLModel) -> [VNClassificationObservation] {
        let request = VNCoreMLRequest(model: vnModel)
        request.imageCropAndScaleOption = .centerCrop

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        do {
            try handler.perform([request])
            return request.results as? [VNClassificationObservation] ?? []
        } catch {
            return []
        }
    }

    private nonisolated static func pixelRect(normalized: CGRect, imageWidth: Int, imageHeight: Int) -> CGRect {
        let x = clamp(normalized.origin.x, min: 0.0, max: 1.0)
        let y = clamp(normalized.origin.y, min: 0.0, max: 1.0)
        let w = clamp(normalized.size.width, min: 0.05, max: 1.0)
        let h = clamp(normalized.size.height, min: 0.05, max: 1.0)

        let px = x * CGFloat(imageWidth)
        let py = y * CGFloat(imageHeight)
        let pw = max(16.0, w * CGFloat(imageWidth))
        let ph = max(16.0, h * CGFloat(imageHeight))

        let clampedX = clamp(px, min: 0.0, max: CGFloat(imageWidth) - pw)
        let clampedY = clamp(py, min: 0.0, max: CGFloat(imageHeight) - ph)

        return CGRect(x: clampedX, y: clampedY, width: pw, height: ph).integral
    }

    private nonisolated static func mergeDetections(
        _ detections: [SpeciesDetection],
        iouThreshold: CGFloat,
        maxDetections: Int
    ) -> [SpeciesDetection] {
        // Keep strongest candidate per species first.
        let bestPerSpecies = detections
            .sorted { $0.confidence > $1.confidence }
            .reduce(into: [String: SpeciesDetection]()) { acc, det in
                if acc[det.speciesId] == nil {
                    acc[det.speciesId] = det
                }
            }

        var remaining = bestPerSpecies.values.sorted { $0.confidence > $1.confidence }
        var selected: [SpeciesDetection] = []

        while !remaining.isEmpty && selected.count < maxDetections {
            let candidate = remaining.removeFirst()
            selected.append(candidate)

            remaining.removeAll { other in
                return iou(other.boundingBox, candidate.boundingBox) >= iouThreshold
            }
        }

        return selected
    }

    private nonisolated static func iou(_ lhs: CGRect, _ rhs: CGRect) -> CGFloat {
        let intersection = lhs.intersection(rhs)
        if intersection.isNull || intersection.isEmpty {
            return 0
        }
        let intersectionArea = intersection.width * intersection.height
        let unionArea = lhs.width * lhs.height + rhs.width * rhs.height - intersectionArea
        if unionArea <= 0 {
            return 0
        }
        return intersectionArea / unionArea
    }

    private nonisolated static func prettifySpeciesName(_ id: String) -> String {
        id
            .replacingOccurrences(of: "_", with: " ")
            .split(separator: " ")
            .map { $0.capitalized }
            .joined(separator: " ")
    }

    private nonisolated static func clamp(_ value: CGFloat, min minValue: CGFloat, max maxValue: CGFloat) -> CGFloat {
        Swift.max(minValue, Swift.min(value, maxValue))
    }
}
