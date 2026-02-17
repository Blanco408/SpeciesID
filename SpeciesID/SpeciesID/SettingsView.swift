import SwiftUI

struct SettingsView: View {
    @Binding var isLoggedIn: Bool

    var body: some View {
        VStack {
            Spacer()
            Text("Settings")
                .font(.largeTitle)
            Text("To be updated")
                .foregroundColor(.secondary)
            Spacer()

            Button(role: .destructive) {
                isLoggedIn = false
            } label: {
                Label("Log Out", systemImage: "rectangle.portrait.and.arrow.right")
                    .frame(maxWidth: .infinity)
                    .frame(height: 50)
            }
            .buttonStyle(.bordered)
            .padding(.horizontal, 32)
            .padding(.bottom, 40)
        }
        .navigationTitle("Settings")
    }
}
