import SwiftUI

struct ClassificationResultView: View {
    let result: ClassificationResult
    let scientificName: String

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Primary prediction
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(result.displayName)
                        .font(.title3)
                        .fontWeight(.bold)
                    Text(scientificName)
                        .font(.subheadline)
                        .italic()
                        .foregroundColor(.secondary)
                }

                Spacer()

                Text("\(Int(result.confidence * 100))%")
                    .font(.title2)
                    .fontWeight(.bold)
                    .foregroundColor(confidenceColor(result.confidence))
            }

            // Confidence bar
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 4)
                        .fill(Color(.systemGray5))
                        .frame(height: 8)

                    RoundedRectangle(cornerRadius: 4)
                        .fill(confidenceColor(result.confidence))
                        .frame(width: geometry.size.width * min(result.confidence, 1.0), height: 8)
                }
            }
            .frame(height: 8)

            // Alternative matches
            if !result.alternatives.isEmpty {
                Divider()

                Text("Other possibilities")
                    .font(.caption)
                    .foregroundColor(.secondary)

                ForEach(result.alternatives.indices, id: \.self) { index in
                    let alt = result.alternatives[index]
                    HStack {
                        Text(alt.displayName)
                            .font(.subheadline)
                        Spacer()
                        Text("\(Int(alt.confidence * 100))%")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
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
