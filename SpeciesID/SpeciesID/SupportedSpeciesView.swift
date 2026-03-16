import SwiftUI

struct SupportedSpeciesView: View {
    @StateObject private var classifier = SpeciesClassifierService()
    @State private var searchText = ""

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
                            SpeciesCard(species: species)
                        }
                    }
                }
            }
            .padding()
        }
        .navigationTitle("Recognizable Species")
        .navigationBarTitleDisplayMode(.inline)
        .searchable(text: $searchText, prompt: "Search species")
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("\(classifier.supportedSpecies.count) species available offline")
                .font(.headline)
                .foregroundStyle(AppColors.darkGreen)
            Text("This list comes from the model bundled on your device.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
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

private struct SpeciesCard: View {
    let species: SupportedSpeciesItem

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            ZStack {
                Circle()
                    .fill(iconBackgroundColor(for: species.id))
                    .frame(width: 34, height: 34)
                Image(systemName: species.iconName)
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(.white)
            }

            VStack(alignment: .leading, spacing: 4) {
                Text(species.displayName)
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.primary)
                    .lineLimit(2)
                Text(species.scientificName)
                    .font(.caption)
                    .italic()
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }

            Spacer(minLength: 0)
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.systemGray6))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private func iconBackgroundColor(for id: String) -> Color {
        let scalarTotal = id.unicodeScalars.reduce(0) { $0 + Int($1.value) }
        let hue = Double((scalarTotal % 360)) / 360.0
        return Color(hue: hue, saturation: 0.7, brightness: 0.75)
    }
}

#Preview {
    NavigationStack {
        SupportedSpeciesView()
    }
}
