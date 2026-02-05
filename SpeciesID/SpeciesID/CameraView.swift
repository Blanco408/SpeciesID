import SwiftUI
import PhotosUI
import CoreLocation
import Combine
import AVFoundation

struct CameraView: View {
    @Environment(\.dismiss) var dismiss
    @StateObject private var locationManager = LocationManager()

    @State private var selectedImage: UIImage?
    @State private var showImagePicker = false
    @State private var showCamera = false
    @State private var showCameraUnavailable = false
    @State private var notes = ""
    @State private var isSaving = false
    @State private var showSaved = false

    private let darkGreen = Color(red: 0.0, green: 0.5, blue: 0.2)

    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                // Image preview
                ZStack {
                    RoundedRectangle(cornerRadius: 12)
                        .fill(Color(.systemGray6))
                        .frame(height: 250)

                    if let image = selectedImage {
                        Image(uiImage: image)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(height: 250)
                            .cornerRadius(12)
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

                // Capture buttons
                HStack(spacing: 16) {
                    Button(action: openCamera) {
                        Label("Camera", systemImage: "camera.fill")
                            .frame(maxWidth: .infinity)
                            .frame(height: 50)
                            .background(darkGreen)
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

                Spacer()

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
                    .background(darkGreen)
                    .foregroundColor(.white)
                    .cornerRadius(10)
                    .padding(.horizontal)
                    .disabled(isSaving)
                }
            }
            .padding(.vertical)
            .navigationTitle("Capture")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") { dismiss() }
                }
            }
            .sheet(isPresented: $showImagePicker) {
                PhotoPicker(image: $selectedImage)
            }
            .fullScreenCover(isPresented: $showCamera) {
                CameraCapture(image: $selectedImage)
            }
            .alert("Saved!", isPresented: $showSaved) {
                Button("OK") { dismiss() }
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
        }
    }

    private func openCamera() {
        // Check if camera is available (not on simulator)
        guard UIImagePickerController.isSourceTypeAvailable(.camera) else {
            showCameraUnavailable = true
            return
        }

        // Check camera permission
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

        DispatchQueue.global(qos: .userInitiated).async {
            // Save image
            let imagePath = ImageStore.shared.saveImage(image)

            // Save observation
            let lat = locationManager.location?.coordinate.latitude ?? 0
            let lon = locationManager.location?.coordinate.longitude ?? 0

            _ = ObservationStore.shared.saveObservation(
                latitude: lat,
                longitude: lon,
                speciesId: nil, // ML identification will fill this in
                confidence: 0.0,
                imagePath: imagePath,
                notes: notes.isEmpty ? nil : notes
            )

            DispatchQueue.main.async {
                isSaving = false
                showSaved = true
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
        // Check camera availability
        guard UIImagePickerController.isSourceTypeAvailable(.camera) else {
            let alert = UIAlertController(
                title: "Camera Unavailable",
                message: "Camera is not available on this device.",
                preferredStyle: .alert
            )
            alert.addAction(UIAlertAction(title: "OK", style: .default) { _ in
                self.dismiss()
            })
            let vc = UIViewController()
            DispatchQueue.main.async {
                vc.present(alert, animated: true)
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
