import SwiftUI

struct SupportedSpeciesView: View {
    @StateObject private var classifier = SpeciesClassifierService()
    @State private var searchText = ""
    @State private var observationCounts: [String: Int] = [:]

    private let columns = [
        GridItem(.adaptive(minimum: 160, maximum: 240), spacing: 12)
    ]

    private var filteredSpecies: [SupportedSpeciesItem] {
        let species = classifier.supportedSpecies
        guard !searchText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return species
        }

        let query = searchText.lowercased()
        return species.filter { item in
            item.displayName.lowercased().contains(query)
            || item.scientificName.lowercased().contains(query)
            || (item.category?.lowercased().contains(query) ?? false)
        }
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header

                if filteredSpecies.isEmpty {
                    emptyState
                } else {
                    LazyVGrid(columns: columns, spacing: 12) {
                        ForEach(filteredSpecies) { species in
                            NavigationLink(destination: SpeciesDetailView(
                                species: species,
                                observationCount: observationCounts[species.id] ?? 0
                            )) {
                                SpeciesCard(
                                    species: species,
                                    observationCount: observationCounts[species.id] ?? 0
                                )
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
            }
            .padding()
        }
        .navigationTitle("Recognizable Species")
        .navigationBarTitleDisplayMode(.inline)
        .searchable(text: $searchText, prompt: "Search species")
        .onAppear {
            observationCounts = ObservationStore.shared.observationCounts()
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("\(classifier.supportedSpecies.count) species available offline")
                .font(.headline)
                .foregroundStyle(AppColors.darkGreen)

            let totalObs = observationCounts.values.reduce(0, +)
            if totalObs > 0 {
                Text("\(totalObs) total observations recorded")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            } else {
                Text("This list comes from the model bundled on your device.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var emptyState: some View {
        VStack(alignment: .leading, spacing: 8) {
            if classifier.supportedSpecies.isEmpty {
                ProgressView("Loading supported species...")
            } else {
                Text("No species match your search.")
                    .foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(Color(.systemGray6))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}

// MARK: - Species Card

private struct SpeciesCard: View {
    let species: SupportedSpeciesItem
    let observationCount: Int

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Reference image
            if let urlString = species.referenceImageUrl, let url = URL(string: urlString) {
                AsyncImage(url: url) { phase in
                    switch phase {
                    case .success(let image):
                        image
                            .resizable()
                            .aspectRatio(contentMode: .fill)
                    case .failure:
                        imagePlaceholder
                    case .empty:
                        ProgressView()
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                    @unknown default:
                        imagePlaceholder
                    }
                }
                .frame(height: 110)
                .clipped()
            } else {
                imagePlaceholder
                    .frame(height: 110)
            }

            // Info section
            VStack(alignment: .leading, spacing: 4) {
                Text(species.displayName)
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.primary)
                    .lineLimit(1)

                Text(species.scientificName)
                    .font(.caption2)
                    .italic()
                    .foregroundStyle(.secondary)
                    .lineLimit(1)

                HStack(spacing: 8) {
                    if let category = species.category {
                        Text(category)
                            .font(.caption2)
                            .foregroundStyle(.white)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(categoryColor(for: category))
                            .clipShape(Capsule())
                    }

                    Spacer()

                    if observationCount > 0 {
                        Label("\(observationCount)", systemImage: "eye.fill")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .padding(10)
        }
        .background(Color(.systemGray6))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private var imagePlaceholder: some View {
        ZStack {
            Color(.systemGray5)
            Image(systemName: species.iconName)
                .font(.system(size: 30))
                .foregroundStyle(Color(.systemGray3))
        }
    }

    private func categoryColor(for category: String) -> Color {
        switch category.lowercased() {
        case "echinoderms": return .orange
        case "mollusks": return .purple
        case "crustaceans": return .red
        case "cnidarians": return .pink
        case "cephalopods": return .indigo
        case "fish": return .blue
        case "algae": return .green
        case "mammals": return .brown
        default: return .gray
        }
    }
}

// MARK: - Species Detail View

struct SpeciesDetailView: View {
    let species: SupportedSpeciesItem
    let observationCount: Int

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                // Hero image
                if let urlString = species.referenceImageUrl, let url = URL(string: urlString) {
                    AsyncImage(url: url) { phase in
                        switch phase {
                        case .success(let image):
                            image
                                .resizable()
                                .aspectRatio(contentMode: .fill)
                        case .failure:
                            heroPlaceholder
                        case .empty:
                            ProgressView()
                                .frame(maxWidth: .infinity)
                                .frame(height: 250)
                        @unknown default:
                            heroPlaceholder
                        }
                    }
                    .frame(height: 250)
                    .frame(maxWidth: .infinity)
                    .clipped()
                    .clipShape(RoundedRectangle(cornerRadius: 16))
                } else {
                    heroPlaceholder
                }

                // Name section
                VStack(alignment: .leading, spacing: 4) {
                    Text(species.displayName)
                        .font(.title2.weight(.bold))
                    Text(species.scientificName)
                        .font(.body)
                        .italic()
                        .foregroundStyle(.secondary)
                }

                // Stats row
                HStack(spacing: 20) {
                    if let category = species.category {
                        StatBadge(
                            icon: "tag.fill",
                            label: category,
                            color: .blue
                        )
                    }
                    StatBadge(
                        icon: "eye.fill",
                        label: observationCount == 1
                            ? "1 observation"
                            : "\(observationCount) observations",
                        color: .green
                    )
                }

                // Description
                if let description = species.speciesDescription {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("About")
                            .font(.headline)
                        Text(description)
                            .font(.body)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.top, 4)
                }
            }
            .padding()
        }
        .navigationTitle(species.displayName)
        .navigationBarTitleDisplayMode(.inline)
    }

    private var heroPlaceholder: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(.systemGray5))
            Image(systemName: species.iconName)
                .font(.system(size: 50))
                .foregroundStyle(Color(.systemGray3))
        }
        .frame(height: 250)
    }
}

private struct StatBadge: View {
    let icon: String
    let label: String
    let color: Color

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: icon)
                .font(.caption)
                .foregroundStyle(color)
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(color.opacity(0.1))
        .clipShape(Capsule())
    }
}

#Preview {
    NavigationStack {
        SupportedSpeciesView()
    }
}
