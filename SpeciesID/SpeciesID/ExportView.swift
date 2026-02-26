import SwiftUI

struct ExportView: View {
    @StateObject private var exportService = ExportService()
    @Environment(\.dismiss) var dismiss

    @State private var format: ExportFormat = .csv
    @State private var includePhotos = true
    @State private var filterByDate = false
    @State private var startDate = Calendar.current.date(byAdding: .month, value: -1, to: Date()) ?? Date()
    @State private var endDate = Date()
    @State private var exportURL: URL?
    @State private var showShareSheet = false
    @State private var observationCount = 0

    var body: some View {
        NavigationView {
            Form {
                // Format selection
                Section("Export Format") {
                    Picker("Format", selection: $format) {
                        ForEach(ExportFormat.allCases, id: \.self) { fmt in
                            Text(fmt.rawValue).tag(fmt)
                        }
                    }
                    .pickerStyle(.segmented)
                }

                // Date range
                Section("Date Range") {
                    Toggle("Filter by date", isOn: $filterByDate)

                    if filterByDate {
                        DatePicker("From", selection: $startDate, displayedComponents: .date)
                        DatePicker("To", selection: $endDate, displayedComponents: .date)
                    }
                }

                // Options
                Section("Options") {
                    Toggle("Include photos", isOn: $includePhotos)

                    if includePhotos {
                        Text("Photos will be included in the zip archive")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }

                // Preview count
                Section {
                    HStack {
                        Image(systemName: "doc.text")
                            .foregroundColor(AppColors.darkGreen)
                        Text("\(observationCount) observations will be exported")
                            .fontWeight(.medium)
                    }
                }

                // Export button
                Section {
                    if exportService.isExporting {
                        VStack(spacing: 12) {
                            ProgressView(value: exportService.progress)
                                .tint(AppColors.darkGreen)
                            Text("Exporting... \(Int(exportService.progress * 100))%")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    } else {
                        Button(action: startExport) {
                            HStack {
                                Image(systemName: "square.and.arrow.up")
                                Text("Export Observations")
                            }
                            .frame(maxWidth: .infinity)
                            .fontWeight(.semibold)
                        }
                        .disabled(observationCount == 0)
                    }

                    if let error = exportService.exportError {
                        Text(error)
                            .font(.caption)
                            .foregroundColor(.red)
                    }
                }
            }
            .navigationTitle("Export")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") { dismiss() }
                }
            }
            .onAppear { updateCount() }
            .onChange(of: filterByDate) { _, _ in updateCount() }
            .onChange(of: startDate) { _, _ in updateCount() }
            .onChange(of: endDate) { _, _ in updateCount() }
            .sheet(isPresented: $showShareSheet) {
                if let url = exportURL {
                    ShareSheet(activityItems: [url])
                }
            }
        }
    }

    private func updateCount() {
        let start = filterByDate ? startDate : nil
        let end = filterByDate ? endDate : nil
        observationCount = exportService.observationCount(from: start, to: end)
    }

    private func startExport() {
        let options = ExportOptions(
            format: format,
            startDate: filterByDate ? startDate : nil,
            endDate: filterByDate ? endDate : nil,
            includePhotos: includePhotos
        )

        Task {
            if let url = await exportService.exportObservations(options: options) {
                exportURL = url
                showShareSheet = true
            }
        }
    }
}

// MARK: - Share Sheet

struct ShareSheet: UIViewControllerRepresentable {
    let activityItems: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: activityItems, applicationActivities: nil)
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}
