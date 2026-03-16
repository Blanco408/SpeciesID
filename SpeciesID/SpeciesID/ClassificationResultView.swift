import SwiftUI

struct ClassificationResultView: View {
    let result: ClassificationResult
    let scientificNameLookup: (String) -> String

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if let primary = result.primaryDetection {
                HStack(alignment: .top) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(primary.displayName)
                            .font(.title3)
                            .fontWeight(.bold)
                        Text(scientificNameLookup(primary.speciesId))
                            .font(.subheadline)
                            .italic()
                            .foregroundColor(.secondary)
                    }

                    Spacer()

                    Text("\(Int(primary.confidence * 100))%")
                        .font(.title2)
                        .fontWeight(.bold)
                        .foregroundColor(confidenceColor(primary.confidence))
                }

                GeometryReader { geometry in
                    ZStack(alignment: .leading) {
                        RoundedRectangle(cornerRadius: 4)
                            .fill(Color(.systemGray5))
                            .frame(height: 8)

                        RoundedRectangle(cornerRadius: 4)
                            .fill(confidenceColor(primary.confidence))
                            .frame(width: geometry.size.width * min(primary.confidence, 1.0), height: 8)
                    }
                }
                .frame(height: 8)
            }

            if result.detections.count > 1 {
                Divider()
                Text("Detected in photo")
                    .font(.caption)
                    .foregroundColor(.secondary)

                ForEach(result.detections) { detection in
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(detection.displayName)
                                .font(.subheadline)
                            Text(scientificNameLookup(detection.speciesId))
                                .font(.caption2)
                                .foregroundColor(.secondary)
                        }
                        Spacer()
                        Text("\(Int(detection.confidence * 100))%")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                }
            } else if let primary = result.primaryDetection, !primary.alternatives.isEmpty {
                Divider()
                Text("Other possibilities")
                    .font(.caption)
                    .foregroundColor(.secondary)

                ForEach(primary.alternatives, id: \.self) { alt in
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
