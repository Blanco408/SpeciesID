import Vision
import CoreML
import UIKit
import Combine

// MARK: - Classification Result

struct ClassificationResult {
    let speciesId: String
    let displayName: String
    let confidence: Double
    let alternatives: [(speciesId: String, displayName: String, confidence: Double)]
}

// MARK: - Species Classifier Service

@MainActor
class SpeciesClassifierService: ObservableObject {
    @Published var isClassifying = false
    @Published var lastResult: ClassificationResult?
    @Published var errorMessage: String?

    private var vnModel: VNCoreMLModel?

    private let displayNames: [String: String] = [
        "seahare": "California Seahare",
        "brittlestar": "Brittle Star",
        "sea_cucumber": "Sea Cucumber",
    ]

    private let scientificNames: [String: String] = [
        "seahare": "Aplysia californica",
        "brittlestar": "Ophiuroidea",
        "sea_cucumber": "Holothuroidea",
    ]

    init() {
        loadModel()
    }

    private func loadModel() {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all
            let model = try SpeciesClassifier(configuration: config)
            vnModel = try VNCoreMLModel(for: model.model)
        } catch {
            errorMessage = "Failed to load ML model: \(error.localizedDescription)"
            print("ML Model load error: \(error)")
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

        return await withCheckedContinuation { continuation in
            let request = VNCoreMLRequest(model: vnModel) { [weak self] request, error in
                guard let self else {
                    continuation.resume(returning: nil)
                    return
                }

                if let error {
                    self.errorMessage = error.localizedDescription
                    continuation.resume(returning: nil)
                    return
                }

                guard let results = request.results as? [VNClassificationObservation],
                      let topResult = results.first else {
                    continuation.resume(returning: nil)
                    return
                }

                let alternatives = results.dropFirst().prefix(2).map {
                    (speciesId: $0.identifier,
                     displayName: self.displayNames[$0.identifier] ?? $0.identifier,
                     confidence: Double($0.confidence))
                }

                let result = ClassificationResult(
                    speciesId: topResult.identifier,
                    displayName: self.displayNames[topResult.identifier] ?? topResult.identifier,
                    confidence: Double(topResult.confidence),
                    alternatives: alternatives
                )

                self.lastResult = result
                continuation.resume(returning: result)
            }

            request.imageCropAndScaleOption = .centerCrop

            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            do {
                try handler.perform([request])
            } catch {
                self.errorMessage = error.localizedDescription
                continuation.resume(returning: nil)
            }
        }
    }

    func scientificName(for speciesId: String) -> String {
        scientificNames[speciesId] ?? speciesId
    }
}
