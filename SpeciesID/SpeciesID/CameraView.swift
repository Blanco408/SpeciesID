import SwiftUI
import PhotosUI
import CoreLocation
import Combine
import AVFoundation

extension Notification.Name {
    static let observationSaved = Notification.Name("observationSaved")
}

struct CameraView: View {
    @StateObject private var locationManager = LocationManager()
    @StateObject private var classifier = SpeciesClassifierService()

    @State private var selectedImage: UIImage?
    @State private var showImagePicker = false
    @State private var showCamera = false
    @State private var showFullPhotoPreview = false
    @State private var showCameraUnavailable = false
    @State private var notes = ""
    @State private var isSaving = false
    @State private var showSaved = false
    @State private var classificationResult: ClassificationResult?

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Image preview
                ZStack {
                    RoundedRectangle(cornerRadius: 12)
                        .fill(Color(.systemGray6))
                        .frame(height: 250)

                    if let image = selectedImage {
                        DetectionPreview(
                            image: image,
                            detections: classificationResult?.detections ?? []
                        )
                        .frame(height: 250)
                        .cornerRadius(12)
                        .onTapGesture {
                            showFullPhotoPreview = true
                        }
                        .overlay(alignment: .bottomTrailing) {
                            Label("Full", systemImage: "arrow.up.left.and.arrow.down.right")
                                .font(.caption2.weight(.semibold))
                                .padding(.horizontal, 8)
                                .padding(.vertical, 6)
                                .background(.ultraThinMaterial, in: Capsule())
                                .padding(10)
                        }
                    } else {
                        VStack {
                            Image(systemName: "camera.viewfinder")
                                .font(.system(size: 50))
                                .foregroundColor(.gray)
                            Text("Take or select a photo")
                                .foregroundColor(.secondary)
                        }
                    }
                }
                .padding(.horizontal)

                if selectedImage != nil {
                    Text("Tap photo to view full image")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                // Capture buttons
                HStack(spacing: 16) {
                    Button(action: openCamera) {
                        Label("Camera", systemImage: "camera.fill")
                            .frame(maxWidth: .infinity)
                            .frame(height: 50)
                            .background(AppColors.darkGreen)
                            .foregroundColor(.white)
                            .cornerRadius(10)
                    }

                    Button(action: { showImagePicker = true }) {
                        Label("Library", systemImage: "photo")
                            .frame(maxWidth: .infinity)
                            .frame(height: 50)
                            .background(Color(.systemGray5))
                            .foregroundColor(.primary)
                            .cornerRadius(10)
                    }
                }
                .padding(.horizontal)

                // Classification results
                if classifier.isClassifying {
                    HStack(spacing: 8) {
                        ProgressView()
                        Text("Identifying species...")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                    .padding()
                } else if let result = classificationResult {
                    ClassificationResultView(
                        result: result,
                        scientificNameLookup: classifier.scientificName(for:)
                    )
                    .padding(.horizontal)
                } else if let error = classifier.errorMessage, selectedImage != nil {
                    HStack {
                        Image(systemName: "exclamationmark.triangle")
                            .foregroundColor(.orange)
                        Text(error)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding(.horizontal)
                }

                // Notes field
                if selectedImage != nil {
                    TextField("Add notes (optional)", text: $notes)
                        .textFieldStyle(.roundedBorder)
                        .padding(.horizontal)

                    // Location status
                    HStack {
                        Image(systemName: locationManager.location != nil ? "location.fill" : "location.slash")
                        Text(locationManager.locationText)
                    }
                    .font(.caption)
                    .foregroundColor(locationManager.location != nil ? .green : .orange)
                }

                Spacer(minLength: 20)

                // Save button
                if selectedImage != nil {
                    Button(action: saveObservation) {
                        if isSaving {
                            ProgressView().tint(.white)
                        } else {
                            Text("Save Observation")
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .frame(height: 50)
                    .background(AppColors.darkGreen)
                    .foregroundColor(.white)
                    .cornerRadius(10)
                    .padding(.horizontal)
                    .disabled(isSaving || classifier.isClassifying)
                }
            }
            .padding(.vertical)
        }
        .navigationTitle("Capture")
        .navigationBarTitleDisplayMode(.inline)
        .sheet(isPresented: $showImagePicker) {
            PhotoPicker(image: $selectedImage)
        }
        .sheet(isPresented: $showFullPhotoPreview) {
            if let image = selectedImage {
                FullPhotoPreviewView(
                    image: image,
                    detections: classificationResult?.detections ?? []
                )
            }
        }
        .fullScreenCover(isPresented: $showCamera) {
            CameraCapture(image: $selectedImage)
        }
        .alert("Saved!", isPresented: $showSaved) {
            Button("OK") { resetForm() }
        } message: {
            Text("Observation saved locally.")
        }
        .alert("Camera Unavailable", isPresented: $showCameraUnavailable) {
            Button("OK", role: .cancel) {}
        } message: {
            Text("Camera is not available. Try using the photo library instead.")
        }
        .onAppear {
            locationManager.requestPermission()
        }
        .onChange(of: selectedImage) { _, newImage in
            classificationResult = nil
            if let image = newImage {
                Task {
                    classificationResult = await classifier.classify(image: image)
                }
            }
        }
    }

    private func openCamera() {
        guard UIImagePickerController.isSourceTypeAvailable(.camera) else {
            showCameraUnavailable = true
            return
        }

        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            showCamera = true
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                DispatchQueue.main.async {
                    if granted {
                        showCamera = true
                    } else {
                        showCameraUnavailable = true
                    }
                }
            }
        default:
            showCameraUnavailable = true
        }
    }

    private func saveObservation() {
        guard let image = selectedImage else { return }
        isSaving = true

        // Capture state on main thread before dispatching
        let lat = locationManager.location?.coordinate.latitude ?? 0
        let lon = locationManager.location?.coordinate.longitude ?? 0
        let identifications = classificationResult?.detections.map { detection in
            LocalSpeciesIdentification(
                id: UUID().uuidString,
                speciesId: detection.speciesId,
                displayName: detection.displayName,
                confidenceScore: detection.confidence,
                modelVersion: "local-multicrop-1",
                isUserVerified: false,
                alternativeSpecies: detection.alternatives.map {
                    LocalAlternativeMatch(
                        speciesId: $0.speciesId,
                        displayName: $0.displayName,
                        confidenceScore: $0.confidence
                    )
                },
                boundingBox: LocalBoundingBox(
                    x: detection.boundingBox.origin.x,
                    y: detection.boundingBox.origin.y,
                    width: detection.boundingBox.width,
                    height: detection.boundingBox.height
                )
            )
        } ?? []
        let species = identifications.first?.speciesId ?? classificationResult?.speciesId
        let confidence = identifications.first?.confidenceScore ?? classificationResult?.confidence ?? 0.0
        let observationNotes = notes.isEmpty ? nil : notes

        DispatchQueue.global(qos: .userInitiated).async {
            let imagePath = ImageStore.shared.saveImage(image)

            _ = ObservationStore.shared.saveObservation(
                latitude: lat,
                longitude: lon,
                speciesId: species,
                confidence: confidence,
                identifications: identifications,
                imagePath: imagePath,
                notes: observationNotes
            )

            DispatchQueue.main.async {
                isSaving = false
                showSaved = true
                NotificationCenter.default.post(name: .observationSaved, object: nil)
            }
        }
    }

    private func resetForm() {
        selectedImage = nil
        classificationResult = nil
        notes = ""
    }
}

// MARK: - Detection Overlay

private struct DetectionPreview: View {
    let image: UIImage
    let detections: [SpeciesDetection]

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(width: geometry.size.width, height: geometry.size.height)

                DetectionOverlay(
                    detections: detections,
                    imageSize: image.size,
                    canvasSize: geometry.size
                )
            }
        }
    }
}

private struct DetectionOverlay: View {
    let detections: [SpeciesDetection]
    let imageSize: CGSize
    let canvasSize: CGSize

    var body: some View {
        let imageRect = imageRectInCanvas()

        ZStack(alignment: .topLeading) {
            ForEach(detections) { detection in
                let rect = CGRect(
                    x: imageRect.minX + detection.boundingBox.minX * imageRect.width,
                    y: imageRect.minY + detection.boundingBox.minY * imageRect.height,
                    width: detection.boundingBox.width * imageRect.width,
                    height: detection.boundingBox.height * imageRect.height
                )

                RoundedRectangle(cornerRadius: 8)
                    .stroke(AppColors.darkGreen.opacity(0.95), lineWidth: 2)
                    .frame(width: max(20, rect.width), height: max(20, rect.height))
                    .position(x: rect.midX, y: rect.midY)
                    .overlay(alignment: .topLeading) {
                        Text("\(detection.displayName) \(Int(detection.confidence * 100))%")
                            .font(.caption2)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 3)
                            .background(AppColors.darkGreen.opacity(0.9))
                            .foregroundColor(.white)
                            .clipShape(RoundedRectangle(cornerRadius: 6))
                            .offset(x: 4, y: 4)
                    }
            }
        }
    }

    private func imageRectInCanvas() -> CGRect {
        guard imageSize.width > 0, imageSize.height > 0,
              canvasSize.width > 0, canvasSize.height > 0 else {
            return CGRect(origin: .zero, size: canvasSize)
        }

        let imageAspect = imageSize.width / imageSize.height
        let canvasAspect = canvasSize.width / canvasSize.height

        if imageAspect > canvasAspect {
            let width = canvasSize.width
            let height = width / imageAspect
            return CGRect(
                x: 0,
                y: (canvasSize.height - height) / 2.0,
                width: width,
                height: height
            )
        } else {
            let height = canvasSize.height
            let width = height * imageAspect
            return CGRect(
                x: (canvasSize.width - width) / 2.0,
                y: 0,
                width: width,
                height: height
            )
        }
    }
}

private struct FullPhotoPreviewView: View {
    let image: UIImage
    let detections: [SpeciesDetection]
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ZStack {
                Color.black.ignoresSafeArea()
                GeometryReader { geometry in
                    ZStack {
                        Image(uiImage: image)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(width: geometry.size.width, height: geometry.size.height)

                        DetectionOverlay(
                            detections: detections,
                            imageSize: image.size,
                            canvasSize: geometry.size
                        )
                    }
                }
            }
            .navigationTitle("Full Photo")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
}

// MARK: - Photo Picker

struct PhotoPicker: UIViewControllerRepresentable {
    @Binding var image: UIImage?
    @Environment(\.dismiss) var dismiss

    func makeUIViewController(context: Context) -> PHPickerViewController {
        var config = PHPickerConfiguration()
        config.selectionLimit = 1
        config.filter = .images
        let picker = PHPickerViewController(configuration: config)
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: PHPickerViewController, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    class Coordinator: NSObject, PHPickerViewControllerDelegate {
        let parent: PhotoPicker
        init(_ parent: PhotoPicker) { self.parent = parent }

        func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
            parent.dismiss()
            guard let provider = results.first?.itemProvider,
                  provider.canLoadObject(ofClass: UIImage.self) else { return }
            provider.loadObject(ofClass: UIImage.self) { image, _ in
                DispatchQueue.main.async {
                    self.parent.image = image as? UIImage
                }
            }
        }
    }
}

// MARK: - Camera Capture

struct CameraCapture: UIViewControllerRepresentable {
    @Binding var image: UIImage?
    @Environment(\.dismiss) var dismiss

    static var isAvailable: Bool {
        UIImagePickerController.isSourceTypeAvailable(.camera)
    }

    func makeUIViewController(context: Context) -> UIViewController {
        guard UIImagePickerController.isSourceTypeAvailable(.camera) else {
            let vc = UIViewController()
            DispatchQueue.main.async {
                self.dismiss()
            }
            return vc
        }

        let picker = UIImagePickerController()
        picker.sourceType = .camera
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: UIViewController, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let parent: CameraCapture
        init(_ parent: CameraCapture) { self.parent = parent }

        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
            parent.image = info[.originalImage] as? UIImage
            parent.dismiss()
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.dismiss()
        }
    }
}

// MARK: - Location Manager

class LocationManager: NSObject, ObservableObject, CLLocationManagerDelegate {
    private let manager = CLLocationManager()
    @Published var location: CLLocation?

    var locationText: String {
        guard let loc = location else { return "Getting location..." }
        return String(format: "%.4f, %.4f", loc.coordinate.latitude, loc.coordinate.longitude)
    }

    override init() {
        super.init()
        manager.delegate = self
        manager.desiredAccuracy = kCLLocationAccuracyBest
    }

    func requestPermission() {
        manager.requestWhenInUseAuthorization()
        manager.startUpdatingLocation()
    }

    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        location = locations.last
    }
}
