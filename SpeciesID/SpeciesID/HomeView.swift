import SwiftUI

struct HomeView: View {
    @State private var showCamera = false

    private let darkGreen = Color(red: 0.0, green: 0.5, blue: 0.2)
    private let lightGreen = Color(red: 0.4, green: 0.7, blue: 0.4)

    var body: some View {
        ZStack {
            Color(.systemBackground).ignoresSafeArea()

            VStack(spacing: 20) {
                // Header
                VStack(spacing: 8) {
                    Text("Welcome Back!")
                        .font(.system(size: 32, weight: .bold))
                        .foregroundColor(darkGreen)
                    Text("Ready to identify some species?")
                        .font(.system(size: 16, weight: .medium))
                        .foregroundColor(lightGreen)
                }
                .padding(.top, 40)

                Spacer()

                // Main content area
                VStack(spacing: 24) {
                    // Camera button (main action)
                    Button(action: {
                        showCamera = true
                    }) {
                        VStack(spacing: 12) {
                            Image(systemName: "camera.fill")
                                .font(.system(size: 60))
                                .foregroundColor(.white)
                            Text("Take Photo")
                                .font(.system(size: 20, weight: .semibold))
                                .foregroundColor(.white)
                        }
                        .frame(maxWidth: .infinity)
                        .frame(height: 150)
                        .background(
                            LinearGradient(
                                colors: [darkGreen, lightGreen],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                        .cornerRadius(20)
                        .shadow(color: darkGreen.opacity(0.3), radius: 10, x: 0, y: 5)
                    }
                }
                .padding(.horizontal, 32)

                Spacer()
            }
        }
        .navigationTitle("EcoSnap")
        .navigationBarTitleDisplayMode(.inline)
        .sheet(isPresented: $showCamera) {
            CameraView()
        }
    }
}

#Preview {
    NavigationStack {
        HomeView()
    }
}
