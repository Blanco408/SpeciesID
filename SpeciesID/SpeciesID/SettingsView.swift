import SwiftUI

struct SettingsView: View {
    @Environment(\.dismiss) var dismiss
    @Binding var isLoggedIn: Bool
    
    var body: some View {
        NavigationView {
            VStack {
                Spacer()
                Text("Settings")
                    .font(.largeTitle)
                Text("To be updated")
                    .foregroundColor(.secondary)
                Spacer()
            }
            .navigationTitle("Settings")
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Back") { dismiss() }
                }
            }
        }
    }
}

