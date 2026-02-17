import SwiftUI

struct SettingsView: View {
    @Binding var isLoggedIn: Bool
    @State private var showLogoutConfirmation = false
    @State private var autoSavePhotos = true
    @State private var confidenceThreshold = 0.7
    @State private var storageUsed: String = "Calculating..."

    var body: some View {
        List {
            // Account
            Section("Account") {
                HStack(spacing: 14) {
                    Image(systemName: "person.circle.fill")
                        .font(.system(size: 40))
                        .foregroundColor(AppColors.darkGreen)
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Guest User")
                            .font(.headline)
                        Text("Sign in for full access")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                .padding(.vertical, 8)
            }

            // Preferences
            Section("Preferences") {
                Toggle(isOn: $autoSavePhotos) {
                    Label("Auto-Save Photos", systemImage: "square.and.arrow.down")
                }
                .tint(AppColors.darkGreen)

                VStack(alignment: .leading, spacing: 8) {
                    Label("Confidence Threshold", systemImage: "gauge.medium")
                    HStack {
                        Slider(value: $confidenceThreshold, in: 0.1...1.0, step: 0.05)
                            .tint(AppColors.darkGreen)
                        Text("\(Int(confidenceThreshold * 100))%")
                            .font(.subheadline)
                            .monospacedDigit()
                            .frame(width: 40)
                    }
                }
                .padding(.vertical, 4)
            }

            // Storage
            Section("Storage") {
                HStack {
                    Label("Photos Storage", systemImage: "internaldrive")
                    Spacer()
                    Text(storageUsed)
                        .foregroundColor(.secondary)
                }

                Button(role: .destructive) {
                    clearAllData()
                } label: {
                    Label("Clear All Data", systemImage: "trash")
                }
            }

            // About
            Section("About") {
                HStack {
                    Label("Version", systemImage: "info.circle")
                    Spacer()
                    Text("1.0.0")
                        .foregroundColor(.secondary)
                }
            }

            // Logout
            Section {
                Button(role: .destructive) {
                    showLogoutConfirmation = true
                } label: {
                    Label("Log Out", systemImage: "rectangle.portrait.and.arrow.right")
                }
            }
        }
        .navigationTitle("Settings")
        .confirmationDialog(
            "Are you sure you want to log out?",
            isPresented: $showLogoutConfirmation,
            titleVisibility: .visible
        ) {
            Button("Log Out", role: .destructive) {
                isLoggedIn = false
            }
            Button("Cancel", role: .cancel) {}
        }
        .onAppear {
            updateStorageInfo()
        }
    }

    private func updateStorageInfo() {
        let bytes = ImageStore.shared.totalStorageUsed()
        if bytes == 0 {
            storageUsed = "No data"
        } else if bytes < 1_024 {
            storageUsed = "\(bytes) B"
        } else if bytes < 1_048_576 {
            storageUsed = String(format: "%.1f KB", Double(bytes) / 1_024)
        } else {
            storageUsed = String(format: "%.1f MB", Double(bytes) / 1_048_576)
        }
    }

    private func clearAllData() {
        let observations = ObservationStore.shared.getAllObservations()
        for obs in observations {
            if let imagePath = obs.imagePath {
                ImageStore.shared.deleteImage(at: imagePath)
            }
            ObservationStore.shared.deleteObservation(obs)
        }
        updateStorageInfo()
    }
}
