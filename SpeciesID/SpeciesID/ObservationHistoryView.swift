import SwiftUI
import CoreData

struct ObservationHistoryView: View {
    @State private var observations: [SavedObservation] = []
    @State private var showExportSheet = false

    var body: some View {
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
            ToolbarItem(placement: .navigationBarTrailing) {
                Button(action: { showExportSheet = true }) {
                    Image(systemName: "square.and.arrow.up")
                }
                .disabled(observations.isEmpty)
            }
        }
        .sheet(isPresented: $showExportSheet) {
            ExportView()
        }
        .onAppear {
            loadObservations()
        }
    }

    private func loadObservations() {
        observations = ObservationStore.shared.getAllObservations()
    }

    private func deleteObservations(at offsets: IndexSet) {
        for index in offsets {
            ObservationStore.shared.deleteObservation(observations[index])
        }

        loadObservations()
    }
}

// MARK: - Observation Row

struct ObservationRow: View {
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
                Text(primary?.displayName ?? observation.speciesId ?? "Unidentified")
                    .font(.headline)

                if identifications.count > 1 {
                    Text("+\(identifications.count - 1) more species")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                if let date = observation.timestamp {
                    Text(date, style: .date)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                if observation.latitude != 0 || observation.longitude != 0 {
                    HStack(spacing: 4) {
                        Image(systemName: "location.fill")
                            .font(.caption2)
                        Text(String(format: "%.4f, %.4f", observation.latitude, observation.longitude))
                            .font(.caption2)
                    }
                    .foregroundColor(.secondary)
                }

                if let notes = observation.notes, !notes.isEmpty {
                    Text(notes)
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .lineLimit(1)
                }
            }

            Spacer()

            // Confidence badge
            if let primary, primary.confidenceScore > 0 {
                Text("\(Int(primary.confidenceScore * 100))%")
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
