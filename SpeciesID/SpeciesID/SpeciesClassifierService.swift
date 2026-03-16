import Vision
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

// MARK: - Species Classifier Service

@MainActor
class SpeciesClassifierService: ObservableObject {
    @Published var isClassifying = false
    @Published var lastResult: ClassificationResult?
    @Published var errorMessage: String?
    @Published private(set) var supportedSpecies: [SupportedSpeciesItem] = []

    private var vnModel: VNCoreMLModel?

    // This can be expanded without code changes elsewhere.
    private let displayNames: [String: String] = [
        "seahare": "California Seahare",
        "brittlestar": "Brittle Star",
        "sea_cucumber": "Sea Cucumber",
        "bat_star": "Bat Star",
        "ochre_sea_star": "Ochre Sea Star",
        "purple_sea_urchin": "Purple Sea Urchin",
        "red_sea_urchin": "Red Sea Urchin",
        "giant_green_anemone": "Giant Green Anemone",
        "aggregating_anemone": "Aggregating Anemone",
        "california_mussel": "California Mussel",
        "red_abalone": "Red Abalone",
        "owl_limpet": "Owl Limpet",
        "gooseneck_barnacle": "Gooseneck Barnacle",
        "acorn_barnacle": "Acorn Barnacle",
        "kelp_crab": "Kelp Crab",
        "red_rock_crab": "Red Rock Crab",
        "blueband_hermit_crab": "Blueband Hermit Crab",
        "east_pacific_red_octopus": "East Pacific Red Octopus",
        "sea_lemon_nudibranch": "Sea Lemon Nudibranch",
        "spanish_shawl_nudibranch": "Spanish Shawl Nudibranch",
    ]

    private let scientificNames: [String: String] = [
        "seahare": "Aplysia californica",
        "brittlestar": "Ophiuroidea",
        "sea_cucumber": "Holothuroidea",
        "bat_star": "Patiria miniata",
        "ochre_sea_star": "Pisaster ochraceus",
        "purple_sea_urchin": "Strongylocentrotus purpuratus",
        "red_sea_urchin": "Mesocentrotus franciscanus",
        "giant_green_anemone": "Anthopleura xanthogrammica",
        "aggregating_anemone": "Anthopleura elegantissima",
        "california_mussel": "Mytilus californianus",
        "red_abalone": "Haliotis rufescens",
        "owl_limpet": "Lottia gigantea",
        "gooseneck_barnacle": "Pollicipes polymerus",
        "acorn_barnacle": "Balanus glandula",
        "kelp_crab": "Pugettia producta",
        "red_rock_crab": "Cancer productus",
        "blueband_hermit_crab": "Pagurus samuelis",
        "east_pacific_red_octopus": "Octopus rubescens",
        "sea_lemon_nudibranch": "Doriopsilla albopunctata",
        "spanish_shawl_nudibranch": "Flabellinopsis iodinea",
    ]

    private let iconNames: [String: String] = [
        "seahare": "hare.fill",
        "brittlestar": "sparkles",
        "sea_cucumber": "capsule.fill",
        "bat_star": "star.fill",
        "ochre_sea_star": "star.circle.fill",
        "purple_sea_urchin": "circle.hexagongrid.fill",
        "red_sea_urchin": "circle.dashed",
        "giant_green_anemone": "sun.max.fill",
        "aggregating_anemone": "sun.max.circle.fill",
        "california_mussel": "drop.fill",
        "red_abalone": "oval.fill",
        "owl_limpet": "oval.portrait.fill",
        "gooseneck_barnacle": "circle.grid.cross.fill",
        "acorn_barnacle": "circle.grid.2x2.fill",
        "kelp_crab": "water.waves",
        "red_rock_crab": "water.waves.and.arrow.up",
        "blueband_hermit_crab": "snail.fill",
        "east_pacific_red_octopus": "hands.sparkles.fill",
        "sea_lemon_nudibranch": "leaf.fill",
        "spanish_shawl_nudibranch": "flame.fill",
    ]

    private let minimumDetectionConfidence = 0.45
    private let minimumTopMargin = 0.05
    private let fullFrameFallbackConfidence = 0.30
    private let maxDetections = 3
    private let overlapThreshold = 0.30

    init() {
        loadModel()
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

        let displayNames = self.displayNames
        let minimumDetectionConfidence = self.minimumDetectionConfidence
        let minimumTopMargin = self.minimumTopMargin
        let fullFrameFallbackConfidence = self.fullFrameFallbackConfidence
        let maxDetections = self.maxDetections
        let overlapThreshold = self.overlapThreshold

        let result: ClassificationResult? = await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let fullFrame = CGRect(x: 0, y: 0, width: 1, height: 1)
                let windows = Self.candidateWindows()
                var detections: [SpeciesDetection] = []

                let fullFrameObservations = Self.classifyCrop(cgImage, using: vnModel)
                if let fallbackDetection = Self.buildDetection(
                    from: fullFrameObservations,
                    boundingBox: fullFrame,
                    displayNames: displayNames,
                    minimumConfidence: fullFrameFallbackConfidence,
                    minimumTopMargin: 0.0
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
                        displayNames: displayNames,
                        minimumConfidence: minimumDetectionConfidence,
                        minimumTopMargin: minimumTopMargin
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
        scientificNames[speciesId] ?? speciesId
    }

    private func buildSupportedSpecies(from model: MLModel) -> [SupportedSpeciesItem] {
        let labels = model.modelDescription.classLabels?.compactMap { $0 as? String } ?? []
        let sourceLabels = labels.isEmpty ? Array(displayNames.keys).sorted() : labels

        return sourceLabels
            .map { speciesId in
                SupportedSpeciesItem(
                    id: speciesId,
                    displayName: displayNames[speciesId] ?? Self.prettifySpeciesName(speciesId),
                    scientificName: scientificNames[speciesId] ?? Self.prettifySpeciesName(speciesId),
                    iconName: iconNames[speciesId] ?? "fish.fill"
                )
            }
            .sorted { $0.displayName < $1.displayName }
    }

    private func fallbackSpeciesCatalog() -> [SupportedSpeciesItem] {
        Array(displayNames.keys)
            .sorted()
            .map { speciesId in
                SupportedSpeciesItem(
                    id: speciesId,
                    displayName: displayNames[speciesId] ?? Self.prettifySpeciesName(speciesId),
                    scientificName: scientificNames[speciesId] ?? Self.prettifySpeciesName(speciesId),
                    iconName: iconNames[speciesId] ?? "fish.fill"
                )
            }
    }

    // MARK: - Multi-Species Windows

    private static func candidateWindows() -> [CGRect] {
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

    private static func buildDetection(
        from observations: [VNClassificationObservation],
        boundingBox: CGRect,
        displayNames: [String: String],
        minimumConfidence: Double,
        minimumTopMargin: Double
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

        let alternatives = observations.dropFirst().prefix(2).map {
            AlternativePrediction(
                speciesId: $0.identifier,
                displayName: displayNames[$0.identifier] ?? Self.prettifySpeciesName($0.identifier),
                confidence: min(max(Double($0.confidence), 0.0), 1.0)
            )
        }

        return SpeciesDetection(
            id: UUID(),
            speciesId: top.identifier,
            displayName: displayNames[top.identifier] ?? Self.prettifySpeciesName(top.identifier),
            confidence: confidence,
            boundingBox: boundingBox,
            alternatives: alternatives
        )
    }

    private static func gridWindows(gridSize: Int, overlap: CGFloat) -> [CGRect] {
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

    private static func classifyCrop(_ cgImage: CGImage, using vnModel: VNCoreMLModel) -> [VNClassificationObservation] {
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

    private static func pixelRect(normalized: CGRect, imageWidth: Int, imageHeight: Int) -> CGRect {
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

    private static func mergeDetections(
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

    private static func iou(_ lhs: CGRect, _ rhs: CGRect) -> CGFloat {
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

    private static func prettifySpeciesName(_ id: String) -> String {
        id
            .replacingOccurrences(of: "_", with: " ")
            .split(separator: " ")
            .map { $0.capitalized }
            .joined(separator: " ")
    }

    private static func clamp(_ value: CGFloat, min minValue: CGFloat, max maxValue: CGFloat) -> CGFloat {
        Swift.max(minValue, Swift.min(value, maxValue))
    }
}
