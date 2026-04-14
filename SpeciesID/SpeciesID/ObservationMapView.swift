import SwiftUI
import MapKit

struct ObservationMapView: View {
    @State private var observations: [SavedObservation] = []
    @State private var selectedObservation: SavedObservation?
    @State private var cameraPosition: MapCameraPosition = .automatic

    // Default to California coast if no observations
    private static let defaultRegion = MKCoordinateRegion(
        center: CLLocationCoordinate2D(latitude: 36.7783, longitude: -119.4179),
        span: MKCoordinateSpan(latitudeDelta: 5.0, longitudeDelta: 5.0)
    )

    var body: some View {
        ZStack(alignment: .bottom) {
            Map(position: $cameraPosition, selection: $selectedObservation) {
                ForEach(geoObservations, id: \.id) { observation in
                    let identifications = ObservationStore.shared.identifications(for: observation)
                    let name = identifications.first?.displayName ?? observation.speciesId ?? "Unknown"

                    Marker(
                        name,
                        coordinate: CLLocationCoordinate2D(
                            latitude: observation.latitude,
                            longitude: observation.longitude
                        )
                    )
                    .tint(AppColors.darkGreen)
                    .tag(observation)
                }
            }
            .mapStyle(.standard(elevation: .realistic))

            // Popup when a marker is selected
            if let selected = selectedObservation {
                ObservationMapPopup(observation: selected)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
                    .padding(.horizontal, 16)
                    .padding(.bottom, 12)
            }
        }
        .animation(.easeInOut(duration: 0.25), value: selectedObservation)
        .onAppear {
            loadObservations()
        }
    }

    private var geoObservations: [SavedObservation] {
        observations.filter { $0.latitude != 0 || $0.longitude != 0 }
    }

    private func loadObservations() {
        observations = ObservationStore.shared.getAllObservations()
        let filtered = geoObservations

        if filtered.isEmpty {
            cameraPosition = .region(Self.defaultRegion)
        } else if filtered.count == 1, let only = filtered.first {
            cameraPosition = .region(MKCoordinateRegion(
                center: CLLocationCoordinate2D(latitude: only.latitude, longitude: only.longitude),
                span: MKCoordinateSpan(latitudeDelta: 0.05, longitudeDelta: 0.05)
            ))
        } else {
            cameraPosition = .region(regionFitting(filtered))
        }
    }

    private func regionFitting(_ observations: [SavedObservation]) -> MKCoordinateRegion {
        var minLat = observations[0].latitude
        var maxLat = observations[0].latitude
        var minLon = observations[0].longitude
        var maxLon = observations[0].longitude

        for obs in observations {
            minLat = min(minLat, obs.latitude)
            maxLat = max(maxLat, obs.latitude)
            minLon = min(minLon, obs.longitude)
            maxLon = max(maxLon, obs.longitude)
        }

        let center = CLLocationCoordinate2D(
            latitude: (minLat + maxLat) / 2,
            longitude: (minLon + maxLon) / 2
        )
        let span = MKCoordinateSpan(
            latitudeDelta: max(0.01, (maxLat - minLat) * 1.4),
            longitudeDelta: max(0.01, (maxLon - minLon) * 1.4)
        )
        return MKCoordinateRegion(center: center, span: span)
    }
}

// MARK: - Map Popup

private struct ObservationMapPopup: View {
    let observation: SavedObservation

    var body: some View {
        let identifications = ObservationStore.shared.identifications(for: observation)
        let primary = identifications.first

        HStack(spacing: 12) {
            // Thumbnail
            if let imagePath = observation.imagePath,
               let image = ImageStore.shared.loadImage(filename: imagePath) {
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(width: 56, height: 56)
                    .cornerRadius(8)
                    .clipped()
            } else {
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color(.systemGray5))
                    .frame(width: 56, height: 56)
                    .overlay(
                        Image(systemName: "photo")
                            .foregroundColor(.gray)
                    )
            }

            VStack(alignment: .leading, spacing: 4) {
                Text(primary?.displayName ?? observation.speciesId ?? "Unidentified")
                    .font(.headline)
                    .lineLimit(1)

                if let date = observation.timestamp {
                    Text(date, style: .date)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                if let primary, primary.confidenceScore > 0 {
                    Text("\(Int(primary.confidenceScore * 100))% confidence")
                        .font(.caption)
                        .foregroundColor(AppColors.darkGreen)
                }
            }

            Spacer()
        }
        .padding(12)
        .background(.regularMaterial)
        .cornerRadius(14)
        .shadow(color: .black.opacity(0.15), radius: 8, y: 4)
    }
}
