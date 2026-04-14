import SwiftUI
import PhotosUI
import CoreLocation
import Combine
import AVFoundation
import CoreImage

extension Notification.Name {
    static let observationSaved = Notification.Name("observationSaved")
}

struct CameraView: View {
    @StateObject private var locationManager = LocationManager()
    @StateObject private var classifier = SpeciesClassifierService()
    @StateObject private var camera = CameraManager()

    @State private var selectedImage: UIImage?
    @State private var showImagePicker = false
    @State private var showFullPhotoPreview = false
    @State private var notes = ""
    @State private var isSaving = false
    @State private var showSaved = false
    @State private var classificationResult: ClassificationResult?
    @State private var scanLineOffset: CGFloat = 0
    @State private var liveLabel: String = ""
    @State private var liveConfidence: Double = 0

    var body: some View {
        ZStack {
            if selectedImage != nil {
                resultView
            } else {
                liveCameraView
            }
        }
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
        .alert("Saved!", isPresented: $showSaved) {
            Button("OK") { resetForm() }
        } message: {
            Text("Observation saved locally.")
        }
        .onAppear {
            locationManager.requestPermission()
            if selectedImage == nil {
                camera.start()
            }
        }
        .onDisappear {
            camera.stop()
        }
        .onChange(of: camera.frameCount) { _, _ in
            guard selectedImage == nil,
                  !classifier.isClassifying,
                  let frame = camera.latestFrame else { return }
            Task {
                if let result = await classifier.classify(image: frame) {
                    liveLabel = result.displayName
                    liveConfidence = result.confidence
                } else {
                    liveLabel = ""
                    liveConfidence = 0
                }
            }
        }
        .onChange(of: selectedImage) { _, newImage in
            classificationResult = nil
            if let image = newImage {
                camera.stop()
                Task {
                    classificationResult = await classifier.classify(image: image)
                }
            }
        }
    }

    // MARK: - Live Camera View

    private var liveCameraView: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            // Live camera preview
            CameraPreviewView(session: camera.session)
                .ignoresSafeArea()

            // Scanner overlay
            ScannerOverlay(scanLineOffset: $scanLineOffset)
                .ignoresSafeArea()

            // Live classification pill
            if !liveLabel.isEmpty {
                VStack {
                    HStack(spacing: 8) {
                        Image(systemName: "sparkle.magnifyingglass")
                            .font(.caption.weight(.semibold))
                        Text(liveLabel)
                            .font(.subheadline.weight(.semibold))
                        Text("\(Int(liveConfidence * 100))%")
                            .font(.caption.weight(.medium))
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Color.white.opacity(0.25))
                            .cornerRadius(4)
                    }
                    .foregroundColor(.white)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 10)
                    .background(AppColors.darkGreen.opacity(0.85))
                    .cornerRadius(24)
                    .shadow(color: .black.opacity(0.3), radius: 6, y: 2)

                    Spacer()
                }
                .padding(.top, 60)
                .animation(.easeInOut(duration: 0.3), value: liveLabel)
            }

            // Bottom controls
            VStack {
                Spacer()

                // Status bar
                HStack(spacing: 12) {
                    // Model status
                    HStack(spacing: 4) {
                        Circle()
                            .fill(classifier.isModelLoaded ? Color.green : Color.orange)
                            .frame(width: 6, height: 6)
                        Text(classifier.isModelLoaded ? "AI Ready" : "Loading...")
                            .font(.caption2.weight(.medium))
                    }

                    Spacer()

                    // Location status
                    HStack(spacing: 4) {
                        Image(systemName: locationManager.location != nil ? "location.fill" : "location.slash")
                            .font(.caption2)
                        Text(locationManager.location != nil ? "GPS Locked" : "No GPS")
                            .font(.caption2.weight(.medium))
                    }

                    Spacer()

                    // Species count
                    HStack(spacing: 4) {
                        Image(systemName: "leaf.fill")
                            .font(.caption2)
                        Text("\(classifier.supportedSpecies.count) species")
                            .font(.caption2.weight(.medium))
                    }
                }
                .foregroundStyle(.white.opacity(0.8))
                .padding(.horizontal, 24)
                .padding(.vertical, 10)
                .background(.ultraThinMaterial.opacity(0.6))

                // Capture controls
                HStack(alignment: .center, spacing: 40) {
                    // Library button
                    Button(action: { showImagePicker = true }) {
                        VStack(spacing: 4) {
                            Image(systemName: "photo.on.rectangle")
                                .font(.system(size: 22))
                            Text("Library")
                                .font(.caption2)
                        }
                        .foregroundStyle(.white)
                        .frame(width: 60)
                    }

                    // Shutter button
                    Button(action: capturePhoto) {
                        ZStack {
                            Circle()
                                .stroke(.white, lineWidth: 4)
                                .frame(width: 72, height: 72)
                            Circle()
                                .fill(.white)
                                .frame(width: 60, height: 60)
                            Image(systemName: "viewfinder")
                                .font(.system(size: 24, weight: .semibold))
                                .foregroundStyle(AppColors.darkGreen)
                        }
                    }

                    // Flash placeholder for symmetry
                    VStack(spacing: 4) {
                        Image(systemName: "bolt.slash.fill")
                            .font(.system(size: 22))
                        Text("Flash")
                            .font(.caption2)
                    }
                    .foregroundStyle(.white.opacity(0.5))
                    .frame(width: 60)
                }
                .padding(.vertical, 24)
                .padding(.bottom, 8)
            }
        }
        .onAppear {
            withAnimation(.easeInOut(duration: 2.0).repeatForever(autoreverses: true)) {
                scanLineOffset = 1.0
            }
        }
    }

    // MARK: - Result View (after capture)

    private var resultView: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Image preview
                ZStack {
                    if let image = selectedImage {
                        DetectionPreview(
                            image: image,
                            detections: classificationResult?.detections ?? []
                        )
                        .frame(height: 300)
                        .cornerRadius(12)
                        .onTapGesture {
                            showFullPhotoPreview = true
                        }
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
                } else if let error = classifier.errorMessage {
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
                TextField("Add notes (optional)", text: $notes)
                    .textFieldStyle(.roundedBorder)
                    .padding(.horizontal)

                // Location
                HStack {
                    Image(systemName: locationManager.location != nil ? "location.fill" : "location.slash")
                    Text(locationManager.locationText)
                }
                .font(.caption)
                .foregroundColor(locationManager.location != nil ? .green : .orange)

                // Action buttons
                HStack(spacing: 12) {
                    Button(action: { resetForm(); camera.start() }) {
                        Label("Retake", systemImage: "arrow.counterclockwise")
                            .frame(maxWidth: .infinity)
                            .frame(height: 50)
                            .background(Color(.systemGray5))
                            .foregroundColor(.primary)
                            .cornerRadius(10)
                    }

                    Button(action: saveObservation) {
                        if isSaving {
                            ProgressView().tint(.white)
                        } else {
                            Label("Save", systemImage: "checkmark.circle.fill")
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .frame(height: 50)
                    .background(AppColors.darkGreen)
                    .foregroundColor(.white)
                    .cornerRadius(10)
                    .disabled(isSaving || classifier.isClassifying)
                }
                .padding(.horizontal)

                Spacer(minLength: 20)
            }
            .padding(.vertical)
        }
        .navigationTitle("Identify")
    }

    // MARK: - Actions

    private func capturePhoto() {
        camera.capturePhoto { image in
            DispatchQueue.main.async {
                self.selectedImage = image
            }
        }
    }

    private func saveObservation() {
        guard let image = selectedImage else { return }
        isSaving = true

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
        liveLabel = ""
        liveConfidence = 0
    }
}

// MARK: - Camera Manager (AVCaptureSession)

class CameraManager: NSObject, ObservableObject {
    let session = AVCaptureSession()
    private let output = AVCapturePhotoOutput()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let videoQueue = DispatchQueue(label: "com.speciesid.videoOutput", qos: .userInitiated)
    private var photoCompletion: ((UIImage?) -> Void)?
    private var isConfigured = false

    /// The latest frame captured for live classification (published on main actor).
    @Published var latestFrame: UIImage?
    /// Incremented each time a new frame is published, to trigger SwiftUI onChange.
    @Published var frameCount: Int = 0

    /// Timestamp of last frame forwarded for classification (~2 fps throttle).
    private nonisolated(unsafe) var lastFrameTime: CFAbsoluteTime = 0
    private let frameCaptureInterval: CFAbsoluteTime = 0.5 // 2 fps

    func start() {
        guard !session.isRunning else { return }

        if !isConfigured {
            configure()
        }

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.session.startRunning()
        }
    }

    func stop() {
        guard session.isRunning else { return }
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.session.stopRunning()
        }
    }

    func capturePhoto(completion: @escaping (UIImage?) -> Void) {
        photoCompletion = completion
        let settings = AVCapturePhotoSettings()
        output.capturePhoto(with: settings, delegate: self)
    }

    private func configure() {
        session.beginConfiguration()
        session.sessionPreset = .photo

        // Camera input
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input) else {
            session.commitConfiguration()
            return
        }
        session.addInput(input)

        // Photo output
        guard session.canAddOutput(output) else {
            session.commitConfiguration()
            return
        }
        session.addOutput(output)

        // Video data output for live classification
        videoOutput.setSampleBufferDelegate(self, queue: videoQueue)
        videoOutput.alwaysDiscardsLateVideoFrames = true
        if session.canAddOutput(videoOutput) {
            session.addOutput(videoOutput)
        }

        session.commitConfiguration()
        isConfigured = true
    }

    // MARK: - Sample Buffer to UIImage

    private nonisolated func imageFromSampleBuffer(_ sampleBuffer: CMSampleBuffer) -> UIImage? {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return nil }
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return nil }
        return UIImage(cgImage: cgImage)
    }
}

extension CameraManager: AVCapturePhotoCaptureDelegate {
    nonisolated func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        guard let data = photo.fileDataRepresentation(),
              let image = UIImage(data: data) else {
            DispatchQueue.main.async { [weak self] in
                self?.photoCompletion?(nil)
            }
            return
        }
        DispatchQueue.main.async { [weak self] in
            self?.photoCompletion?(image)
        }
    }
}

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        let now = CFAbsoluteTimeGetCurrent()
        guard now - lastFrameTime >= frameCaptureInterval else { return }
        lastFrameTime = now

        guard let image = imageFromSampleBuffer(sampleBuffer) else { return }
        DispatchQueue.main.async { [weak self] in
            self?.latestFrame = image
            self?.frameCount += 1
        }
    }
}

// MARK: - Camera Preview (UIViewRepresentable)

struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> UIView {
        let view = CameraPreviewUIView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        view.backgroundColor = .black
        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {}

    class CameraPreviewUIView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }
}

// MARK: - Scanner Overlay

struct ScannerOverlay: View {
    @Binding var scanLineOffset: CGFloat

    var body: some View {
        GeometryReader { geometry in
            let size = min(geometry.size.width, geometry.size.height) * 0.7
            let center = CGPoint(x: geometry.size.width / 2, y: geometry.size.height / 2 - 40)

            ZStack {
                // Dim area outside scan region
                ScannerMask(center: center, size: size)
                    .fill(style: FillStyle(eoFill: true))
                    .foregroundStyle(.black.opacity(0.4))

                // Corner brackets
                ScannerCorners(center: center, size: size)
                    .stroke(AppColors.darkGreen, style: StrokeStyle(lineWidth: 3, lineCap: .round))

                // Scanning line
                Rectangle()
                    .fill(
                        LinearGradient(
                            colors: [.clear, AppColors.darkGreen.opacity(0.6), .clear],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
                    .frame(width: size - 20, height: 2)
                    .position(
                        x: center.x,
                        y: center.y - size / 2 + scanLineOffset * size
                    )

                // Label
                Text("SPECIES SCAN")
                    .font(.caption.weight(.bold).monospaced())
                    .tracking(3)
                    .foregroundStyle(AppColors.darkGreen)
                    .position(x: center.x, y: center.y + size / 2 + 24)
            }
        }
    }
}

// Mask that dims everything outside the scan area
struct ScannerMask: Shape {
    let center: CGPoint
    let size: CGFloat

    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.addRect(rect)

        let scanRect = CGRect(
            x: center.x - size / 2,
            y: center.y - size / 2,
            width: size,
            height: size
        )
        path.addRoundedRect(in: scanRect, cornerSize: CGSize(width: 16, height: 16))
        return path
    }
}

// Corner bracket shapes
struct ScannerCorners: Shape {
    let center: CGPoint
    let size: CGFloat

    func path(in rect: CGRect) -> Path {
        var path = Path()
        let half = size / 2
        let cornerLen: CGFloat = 30
        let r: CGFloat = 12

        let topLeft = CGPoint(x: center.x - half, y: center.y - half)
        let topRight = CGPoint(x: center.x + half, y: center.y - half)
        let bottomLeft = CGPoint(x: center.x - half, y: center.y + half)
        let bottomRight = CGPoint(x: center.x + half, y: center.y + half)

        // Top-left
        path.move(to: CGPoint(x: topLeft.x, y: topLeft.y + cornerLen))
        path.addLine(to: CGPoint(x: topLeft.x, y: topLeft.y + r))
        path.addQuadCurve(to: CGPoint(x: topLeft.x + r, y: topLeft.y),
                          control: topLeft)
        path.addLine(to: CGPoint(x: topLeft.x + cornerLen, y: topLeft.y))

        // Top-right
        path.move(to: CGPoint(x: topRight.x - cornerLen, y: topRight.y))
        path.addLine(to: CGPoint(x: topRight.x - r, y: topRight.y))
        path.addQuadCurve(to: CGPoint(x: topRight.x, y: topRight.y + r),
                          control: topRight)
        path.addLine(to: CGPoint(x: topRight.x, y: topRight.y + cornerLen))

        // Bottom-left
        path.move(to: CGPoint(x: bottomLeft.x, y: bottomLeft.y - cornerLen))
        path.addLine(to: CGPoint(x: bottomLeft.x, y: bottomLeft.y - r))
        path.addQuadCurve(to: CGPoint(x: bottomLeft.x + r, y: bottomLeft.y),
                          control: bottomLeft)
        path.addLine(to: CGPoint(x: bottomLeft.x + cornerLen, y: bottomLeft.y))

        // Bottom-right
        path.move(to: CGPoint(x: bottomRight.x - cornerLen, y: bottomRight.y))
        path.addLine(to: CGPoint(x: bottomRight.x - r, y: bottomRight.y))
        path.addQuadCurve(to: CGPoint(x: bottomRight.x, y: bottomRight.y - r),
                          control: bottomRight)
        path.addLine(to: CGPoint(x: bottomRight.x, y: bottomRight.y - cornerLen))

        return path
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
            return CGRect(x: 0, y: (canvasSize.height - height) / 2.0, width: width, height: height)
        } else {
            let height = canvasSize.height
            let width = height * imageAspect
            return CGRect(x: (canvasSize.width - width) / 2.0, y: 0, width: width, height: height)
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
                    Button("Done") { dismiss() }
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
