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

        let displayNames = self.displayNames

        return await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let request = VNCoreMLRequest(model: vnModel) { request, error in
                    if let error {
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
                         displayName: displayNames[$0.identifier] ?? $0.identifier,
                         confidence: min(Double($0.confidence), 1.0))
                    }

                    let result = ClassificationResult(
                        speciesId: topResult.identifier,
                        displayName: displayNames[topResult.identifier] ?? topResult.identifier,
                        confidence: min(Double(topResult.confidence), 1.0),
                        alternatives: alternatives
                    )

                    continuation.resume(returning: result)
                }

                request.imageCropAndScaleOption = .centerCrop

                let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
                do {
                    try handler.perform([request])
                } catch {
                    continuation.resume(returning: nil)
                }
            }
        }
    }

    func scientificName(for speciesId: String) -> String {
        scientificNames[speciesId] ?? speciesId
    }
}
