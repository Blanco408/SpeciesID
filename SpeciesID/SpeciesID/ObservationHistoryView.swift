import SwiftUI
import CoreData

struct ObservationHistoryView: View {
    @Environment(\.dismiss) var dismiss
    @State private var observations: [SavedObservation] = []

    private let darkGreen = Color(red: 0.0, green: 0.5, blue: 0.2)

    var body: some View {
        NavigationView {
            Group {
                if observations.isEmpty {
                    VStack(spacing: 16) {
                        Image(systemName: "binoculars")
                            .font(.system(size: 60))
                            .foregroundColor(.gray)
                        Text("No observations yet")
                            .font(.title2)
                            .foregroundColor(.secondary)
                        Text("Capture your first species sighting!")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                } else {
                    List {
                        ForEach(observations, id: \.id) { observation in
                            ObservationRow(observation: observation)
                        }
                        .onDelete(perform: deleteObservations)
                    }
                    .listStyle(.plain)
                }
            }
            .navigationTitle("Observations")
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Back") { dismiss() }
                }
            }
            .onAppear {
                loadObservations()
            }
        }
    }

    private func loadObservations() {
        observations = ObservationStore.shared.getAllObservations()
    }

    private func deleteObservations(at offsets: IndexSet) {
        for index in offsets {
            let obs = observations[index]

            // Delete image file if it exists
            if let imagePath = obs.imagePath {
                ImageStore.shared.deleteImage(at: imagePath)
            }

            // Delete from Core Data
            ObservationStore.shared.deleteObservation(obs)
        }

        // Refresh list
        loadObservations()
    }
}

// MARK: - Observation Row

struct ObservationRow: View {
    let observation: SavedObservation

    var body: some View {
        HStack(spacing: 12) {
            // Thumbnail
            if let imagePath = observation.imagePath,
               let image = ImageStore.shared.loadImage(filename: imagePath) {
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(width: 60, height: 60)
                    .cornerRadius(8)
                    .clipped()
            } else {
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color(.systemGray5))
                    .frame(width: 60, height: 60)
                    .overlay(
                        Image(systemName: "photo")
                            .foregroundColor(.gray)
                    )
            }

            // Details
            VStack(alignment: .leading, spacing: 4) {
                // Species or "Unidentified"
                Text(observation.speciesId ?? "Unidentified")
                    .font(.headline)

                // Date
                if let date = observation.timestamp {
                    Text(date, style: .date)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                // Location
                if observation.latitude != 0 || observation.longitude != 0 {
                    HStack(spacing: 4) {
                        Image(systemName: "location.fill")
                            .font(.caption2)
                        Text(String(format: "%.4f, %.4f", observation.latitude, observation.longitude))
                            .font(.caption2)
                    }
                    .foregroundColor(.secondary)
                }

                // Notes preview
                if let notes = observation.notes, !notes.isEmpty {
                    Text(notes)
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .lineLimit(1)
                }
            }

            Spacer()

            // Confidence badge (if identified)
            if observation.confidence > 0 {
                Text("\(Int(observation.confidence * 100))%")
                    .font(.caption)
                    .fontWeight(.medium)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color.green.opacity(0.2))
                    .foregroundColor(.green)
                    .cornerRadius(6)
            }
        }
        .padding(.vertical, 4)
    }
}
