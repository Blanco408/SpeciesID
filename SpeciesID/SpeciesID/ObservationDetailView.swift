import SwiftUI
import MapKit

struct ObservationDetailView: View {
    let observation: SavedObservation

    @State private var identifications: [LocalSpeciesIdentification] = []
    @State private var image: UIImage?
    @State private var showDeleteConfirmation = false
    @State private var showShareSheet = false
    @StateObject private var classifier = SpeciesClassifierService()
    @Environment(\.dismiss) private var dismiss

    private var primary: LocalSpeciesIdentification? {
        identifications.first
    }

    private var hasLocation: Bool {
        observation.latitude != 0 || observation.longitude != 0
    }

    private var coordinate: CLLocationCoordinate2D {
        CLLocationCoordinate2D(
            latitude: observation.latitude,
            longitude: observation.longitude
        )
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                photoSection
                speciesInfoSection

                if identifications.count > 1 {
                    allDetectionsSection
                }

                if let primary, !primary.alternativeSpecies.isEmpty {
                    alternativesSection(primary)
                }

                if hasLocation {
                    locationSection
                }

                metadataSection
                actionsSection
            }
            .padding(.bottom, 32)
        }
        .background(Color(.systemGroupedBackground).ignoresSafeArea())
        .navigationTitle("Observation")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear {
            loadData()
        }
        .confirmationDialog(
            "Delete Observation",
            isPresented: $showDeleteConfirmation,
            titleVisibility: .visible
        ) {
            Button("Delete", role: .destructive) {
                ObservationStore.shared.deleteObservation(observation)
                dismiss()
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This will permanently delete this observation and its photo. This cannot be undone.")
        }
        .sheet(isPresented: $showShareSheet) {
            if let image {
                ShareSheet(activityItems: [shareText(), image])
            } else {
                ShareSheet(activityItems: [shareText()])
            }
        }
    }

    // MARK: - Photo Section

    private var photoSection: some View {
        ZStack(alignment: .topLeading) {
            if let image {
                GeometryReader { geo in
                    let imageSize = image.size
                    let displayWidth = geo.size.width
                    let displayHeight = displayWidth * (imageSize.height / imageSize.width)

                    ZStack {
                        Image(uiImage: image)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(width: displayWidth, height: displayHeight)

                        // Bounding boxes
                        ForEach(identifications) { ident in
                            if let box = ident.boundingBox {
                                let rect = CGRect(
                                    x: box.x * displayWidth,
                                    y: box.y * displayHeight,
                                    width: box.width * displayWidth,
                                    height: box.height * displayHeight
                                )
                                Rectangle()
                                    .stroke(boxColor(for: ident), lineWidth: 2)
                                    .frame(width: rect.width, height: rect.height)
                                    .position(
                                        x: rect.midX,
                                        y: rect.midY
                                    )

                                // Label above box
                                Text(ident.displayName)
                                    .font(.caption2.weight(.semibold))
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 2)
                                    .background(boxColor(for: ident).opacity(0.85))
                                    .foregroundColor(.white)
                                    .clipShape(RoundedRectangle(cornerRadius: 4))
                                    .position(
                                        x: rect.midX,
                                        y: rect.minY - 12
                                    )
                            }
                        }
                    }
                    .frame(width: displayWidth, height: displayHeight)
                }
                .aspectRatio(image.size, contentMode: .fit)
            } else {
                ZStack {
                    Color(.systemGray5)
                    Image(systemName: "photo")
                        .font(.system(size: 50))
                        .foregroundColor(Color(.systemGray3))
                }
                .frame(height: 250)
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 0))
    }

    // MARK: - Species Info

    private var speciesInfoSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            if let primary {
                VStack(alignment: .leading, spacing: 4) {
                    Text(primary.displayName)
                        .font(.title2.weight(.bold))

                    Text(classifier.scientificName(for: primary.speciesId))
                        .font(.body)
                        .italic()
                        .foregroundColor(.secondary)
                }

                // Confidence bar
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text("Confidence")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                        Spacer()
                        Text("\(Int(primary.confidenceScore * 100))%")
                            .font(.subheadline.weight(.semibold))
                            .foregroundColor(confidenceColor(primary.confidenceScore))
                    }

                    GeometryReader { geometry in
                        ZStack(alignment: .leading) {
                            RoundedRectangle(cornerRadius: 4)
                                .fill(Color(.systemGray5))
                                .frame(height: 8)

                            RoundedRectangle(cornerRadius: 4)
                                .fill(confidenceColor(primary.confidenceScore))
                                .frame(
                                    width: geometry.size.width * min(primary.confidenceScore, 1.0),
                                    height: 8
                                )
                        }
                    }
                    .frame(height: 8)
                }

                if primary.isUserVerified {
                    HStack(spacing: 4) {
                        Image(systemName: "checkmark.seal.fill")
                            .foregroundColor(AppColors.darkGreen)
                        Text("User Verified")
                            .font(.caption)
                            .foregroundColor(AppColors.darkGreen)
                    }
                }
            } else {
                Text("Unidentified")
                    .font(.title2.weight(.bold))
                    .foregroundColor(.secondary)
            }
        }
        .padding(.horizontal)
    }

    // MARK: - All Detections

    private var allDetectionsSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("All Detections")
                .font(.headline)
                .padding(.horizontal)

            VStack(spacing: 0) {
                ForEach(Array(identifications.enumerated()), id: \.element.id) { index, ident in
                    HStack(spacing: 12) {
                        Circle()
                            .fill(boxColor(for: ident))
                            .frame(width: 10, height: 10)

                        VStack(alignment: .leading, spacing: 2) {
                            Text(ident.displayName)
                                .font(.subheadline.weight(.medium))
                            Text(classifier.scientificName(for: ident.speciesId))
                                .font(.caption)
                                .italic()
                                .foregroundColor(.secondary)
                        }

                        Spacer()

                        Text("\(Int(ident.confidenceScore * 100))%")
                            .font(.subheadline.weight(.medium))
                            .foregroundColor(confidenceColor(ident.confidenceScore))
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 10)

                    if index < identifications.count - 1 {
                        Divider()
                            .padding(.leading, 38)
                    }
                }
            }
            .background(Color(.systemBackground))
            .clipShape(RoundedRectangle(cornerRadius: 12))
            .padding(.horizontal)
        }
    }

    // MARK: - Alternatives

    private func alternativesSection(_ primary: LocalSpeciesIdentification) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Other Possibilities")
                .font(.headline)
                .padding(.horizontal)

            VStack(spacing: 0) {
                ForEach(Array(primary.alternativeSpecies.enumerated()), id: \.element) { index, alt in
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(alt.displayName)
                                .font(.subheadline)
                        }
                        Spacer()
                        Text("\(Int(alt.confidenceScore * 100))%")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 10)

                    if index < primary.alternativeSpecies.count - 1 {
                        Divider()
                            .padding(.leading, 16)
                    }
                }
            }
            .background(Color(.systemBackground))
            .clipShape(RoundedRectangle(cornerRadius: 12))
            .padding(.horizontal)
        }
    }

    // MARK: - Location

    private var locationSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Location")
                .font(.headline)
                .padding(.horizontal)

            VStack(alignment: .leading, spacing: 10) {
                Map(initialPosition: .region(
                    MKCoordinateRegion(
                        center: coordinate,
                        span: MKCoordinateSpan(latitudeDelta: 0.01, longitudeDelta: 0.01)
                    )
                )) {
                    Marker(
                        primary?.displayName ?? "Observation",
                        coordinate: coordinate
                    )
                    .tint(AppColors.darkGreen)
                }
                .frame(height: 180)
                .clipShape(RoundedRectangle(cornerRadius: 10))
                .allowsHitTesting(false)

                HStack(spacing: 4) {
                    Image(systemName: "location.fill")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(String(format: "%.6f, %.6f", observation.latitude, observation.longitude))
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(.horizontal, 4)
            }
            .padding(12)
            .background(Color(.systemBackground))
            .clipShape(RoundedRectangle(cornerRadius: 12))
            .padding(.horizontal)
        }
    }

    // MARK: - Metadata

    private var metadataSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Details")
                .font(.headline)
                .padding(.horizontal)

            VStack(spacing: 0) {
                if let date = observation.timestamp {
                    MetadataRow(
                        icon: "calendar",
                        label: "Date",
                        value: date.formatted(date: .long, time: .shortened)
                    )
                    Divider().padding(.leading, 44)
                }

                if let notes = observation.notes, !notes.isEmpty {
                    MetadataRow(icon: "note.text", label: "Notes", value: notes)
                }

                if identifications.count > 0 {
                    if observation.notes != nil && !observation.notes!.isEmpty {
                        Divider().padding(.leading, 44)
                    }
                    MetadataRow(
                        icon: "number",
                        label: "Detections",
                        value: "\(identifications.count) species detected"
                    )
                }
            }
            .background(Color(.systemBackground))
            .clipShape(RoundedRectangle(cornerRadius: 12))
            .padding(.horizontal)
        }
    }

    // MARK: - Actions

    private var actionsSection: some View {
        VStack(spacing: 12) {
            Button {
                showShareSheet = true
            } label: {
                HStack {
                    Image(systemName: "square.and.arrow.up")
                    Text("Share Observation")
                }
                .font(.body.weight(.medium))
                .foregroundColor(.white)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 14)
                .background(AppColors.darkGreen)
                .clipShape(RoundedRectangle(cornerRadius: 12))
            }

            Button {
                showDeleteConfirmation = true
            } label: {
                HStack {
                    Image(systemName: "trash")
                    Text("Delete Observation")
                }
                .font(.body.weight(.medium))
                .foregroundColor(.red)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 14)
                .background(Color(.systemBackground))
                .clipShape(RoundedRectangle(cornerRadius: 12))
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(Color.red.opacity(0.3), lineWidth: 1)
                )
            }
        }
        .padding(.horizontal)
    }

    // MARK: - Helpers

    private func loadData() {
        identifications = ObservationStore.shared.identifications(for: observation)
        if let imagePath = observation.imagePath {
            image = ImageStore.shared.loadImage(filename: imagePath)
        }
    }

    private func confidenceColor(_ confidence: Double) -> Color {
        if confidence >= 0.8 {
            return .green
        } else if confidence >= 0.5 {
            return .yellow
        } else {
            return .orange
        }
    }

    private func boxColor(for ident: LocalSpeciesIdentification) -> Color {
        let colors: [Color] = [AppColors.darkGreen, .orange, .blue, .purple, .pink]
        guard let index = identifications.firstIndex(where: { $0.id == ident.id }) else {
            return AppColors.darkGreen
        }
        return colors[index % colors.count]
    }

    private func shareText() -> String {
        var text = "Species Observation"
        if let primary {
            text += " - \(primary.displayName)"
            text += " (\(Int(primary.confidenceScore * 100))% confidence)"
        }
        if let date = observation.timestamp {
            text += "\nDate: \(date.formatted(date: .long, time: .shortened))"
        }
        if hasLocation {
            text += "\nLocation: \(observation.latitude), \(observation.longitude)"
        }
        if let notes = observation.notes, !notes.isEmpty {
            text += "\nNotes: \(notes)"
        }
        text += "\n\nIdentified with SpeciesID"
        return text
    }
}

// MARK: - Metadata Row

private struct MetadataRow: View {
    let icon: String
    let label: String
    let value: String

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: icon)
                .font(.body)
                .foregroundColor(.secondary)
                .frame(width: 20)

            VStack(alignment: .leading, spacing: 2) {
                Text(label)
                    .font(.caption)
                    .foregroundColor(.secondary)
                Text(value)
                    .font(.subheadline)
            }

            Spacer()
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }
}

