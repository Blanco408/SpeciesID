import SwiftUI

struct HomeView: View {
    @State private var observations: [SavedObservation] = []
    @State private var observationCounts: [String: Int] = [:]
    @State private var modelSpeciesCount: Int = 0

    private let dateFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "EEEE, MMMM d"
        return f
    }()

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                headerSection
                statsRow
                recentObservationsSection
                quickActionsSection
            }
            .padding()
        }
        .background(Color(.systemGroupedBackground).ignoresSafeArea())
        .navigationTitle("SpeciesID")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear {
            loadData()
        }
    }

    // MARK: - Header

    private var headerSection: some View {
        HStack(alignment: .center) {
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 8) {
                    Image(systemName: "leaf.fill")
                        .font(.title2)
                        .foregroundColor(AppColors.darkGreen)
                    Text("SpeciesID")
                        .font(.title.weight(.bold))
                        .foregroundColor(AppColors.darkGreen)
                }
                Text(dateFormatter.string(from: Date()))
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
            Spacer()
        }
        .padding(.top, 8)
    }

    // MARK: - Stats Row

    private var statsRow: some View {
        HStack(spacing: 12) {
            StatCard(
                icon: "binoculars.fill",
                value: "\(observations.count)",
                label: "Observations",
                color: AppColors.darkGreen
            )
            StatCard(
                icon: "sparkles",
                value: "\(observationCounts.keys.count)",
                label: "Species Found",
                color: .orange
            )
            StatCard(
                icon: "cpu",
                value: "\(modelSpeciesCount)",
                label: "Model Species",
                color: .blue
            )
        }
    }

    // MARK: - Recent Observations

    private var recentObservationsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Recent Observations")
                    .font(.headline)
                Spacer()
                if !observations.isEmpty {
                    NavigationLink(destination: ObservationHistoryView()) {
                        Text("See All")
                            .font(.subheadline)
                            .foregroundColor(AppColors.darkGreen)
                    }
                }
            }

            if observations.isEmpty {
                emptyObservationsState
            } else {
                VStack(spacing: 0) {
                    let recentSlice = observations.prefix(5)
                    ForEach(Array(recentSlice.enumerated()), id: \.element.id) { index, observation in
                        NavigationLink(destination: ObservationDetailView(observation: observation)) {
                            RecentObservationRow(observation: observation)
                        }
                        .buttonStyle(.plain)

                        if index < recentSlice.count - 1 {
                            Divider()
                                .padding(.leading, 72)
                        }
                    }
                }
                .padding(.vertical, 8)
                .padding(.horizontal, 12)
                .background(Color(.systemBackground))
                .clipShape(RoundedRectangle(cornerRadius: 12))
            }
        }
    }

    private var emptyObservationsState: some View {
        VStack(spacing: 12) {
            Image(systemName: "camera.macro")
                .font(.system(size: 40))
                .foregroundColor(AppColors.lightGreen)
            Text("No observations yet")
                .font(.headline)
                .foregroundColor(.primary)
            Text("Head out and capture your first species sighting to start building your collection.")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 32)
        .padding(.horizontal, 16)
        .background(Color(.systemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    // MARK: - Quick Actions

    private var quickActionsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Quick Actions")
                .font(.headline)

            HStack(spacing: 12) {
                NavigationLink(destination: CameraView()) {
                    QuickActionButton(
                        icon: "camera.fill",
                        title: "Identify Species",
                        color: AppColors.darkGreen
                    )
                }
                .buttonStyle(.plain)

                NavigationLink(destination: SupportedSpeciesView()) {
                    QuickActionButton(
                        icon: "fish.fill",
                        title: "Browse Species",
                        color: AppColors.lightGreen
                    )
                }
                .buttonStyle(.plain)
            }
        }
    }

    // MARK: - Data Loading

    private func loadData() {
        observations = ObservationStore.shared.getAllObservations()
        observationCounts = ObservationStore.shared.observationCounts()
        let classifier = SpeciesClassifierService()
        modelSpeciesCount = classifier.supportedSpecies.count
    }
}

// MARK: - Stat Card

private struct StatCard: View {
    let icon: String
    let value: String
    let label: String
    let color: Color

    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: icon)
                .font(.title3)
                .foregroundColor(color)
            Text(value)
                .font(.title2.weight(.bold))
                .foregroundColor(.primary)
            Text(label)
                .font(.caption)
                .foregroundColor(.secondary)
                .lineLimit(1)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 14)
        .background(Color(.systemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}

// MARK: - Recent Observation Row

private struct RecentObservationRow: View {
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
                    .frame(width: 50, height: 50)
                    .cornerRadius(10)
                    .clipped()
            } else {
                RoundedRectangle(cornerRadius: 10)
                    .fill(Color(.systemGray5))
                    .frame(width: 50, height: 50)
                    .overlay(
                        Image(systemName: "photo")
                            .foregroundColor(.gray)
                    )
            }

            VStack(alignment: .leading, spacing: 3) {
                Text(primary?.displayName ?? observation.speciesId ?? "Unidentified")
                    .font(.subheadline.weight(.medium))
                    .foregroundColor(.primary)
                    .lineLimit(1)

                if let date = observation.timestamp {
                    Text(date, style: .relative)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }

            Spacer()

            if let primary, primary.confidenceScore > 0 {
                Text("\(Int(primary.confidenceScore * 100))%")
                    .font(.caption.weight(.semibold))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(confidenceColor(primary.confidenceScore).opacity(0.15))
                    .foregroundColor(confidenceColor(primary.confidenceScore))
                    .clipShape(Capsule())
            }

            Image(systemName: "chevron.right")
                .font(.caption)
                .foregroundColor(Color(.tertiaryLabel))
        }
        .padding(.vertical, 8)
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
}

// MARK: - Quick Action Button

private struct QuickActionButton: View {
    let icon: String
    let title: String
    let color: Color

    var body: some View {
        VStack(spacing: 10) {
            Image(systemName: icon)
                .font(.title2)
                .foregroundColor(.white)
            Text(title)
                .font(.subheadline.weight(.medium))
                .foregroundColor(.white)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 20)
        .background(
            LinearGradient(
                colors: [color, color.opacity(0.8)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
        )
        .clipShape(RoundedRectangle(cornerRadius: 14))
        .shadow(color: color.opacity(0.25), radius: 6, x: 0, y: 3)
    }
}

#Preview {
    NavigationStack {
        HomeView()
    }
}
