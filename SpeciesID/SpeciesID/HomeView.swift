import SwiftUI

struct HomeView: View {
    @Binding var isLoggedIn: Bool
    @State private var showCamera = false
    @State private var showObservations = false
    @State private var showSettings = false
    
    private let darkGreen = Color(red: 0.0, green: 0.5, blue: 0.2)
    private let lightGreen = Color(red: 0.4, green: 0.7, blue: 0.4)
    
    var body: some View {
        NavigationView {
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
                        
                        // Secondary actions
                        HStack(spacing: 16) {
                            actionButton(
                                icon: "photo.on.rectangle",
                                title: "Gallery",
                                action: {
                                    // TODO: Open photo gallery
                                    print("Gallery tapped")
                                }
                            )
                            
                            actionButton(
                                icon: "list.bullet",
                                title: "Observations",
                                action: {
                                    showObservations = true
                                }
                            )
                            
                            actionButton(
                                icon: "map",
                                title: "Regions",
                                action: {
                                    // TODO: Show regions
                                    print("Regions tapped")
                                }
                            )
                        }
                    }
                    .padding(.horizontal, 32)
                    
                    Spacer()
                }
            }
            .navigationTitle("EcoSnap")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button(action: {
                        showSettings = true
                    }) {
                        Image(systemName: "gearshape.fill")
                            .foregroundColor(darkGreen)
                    }
                }
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: {
                        isLoggedIn = false
                    }) {
                        Image(systemName: "rectangle.portrait.and.arrow.right")
                            .foregroundColor(darkGreen)
                    }
                }
            }
            .sheet(isPresented: $showCamera) {
                CameraView()
            }
            .sheet(isPresented: $showObservations) {
                ObservationHistoryView()
            }
            .sheet(isPresented: $showSettings) {
                SettingsView(isLoggedIn: $isLoggedIn)
            }
        }
    }
    
    private func actionButton(icon: String, title: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            VStack(spacing: 8) {
                Image(systemName: icon)
                    .font(.system(size: 30))
                    .foregroundColor(darkGreen)
                Text(title)
                    .font(.system(size: 14, weight: .medium))
                    .foregroundColor(.primary)
            }
            .frame(maxWidth: .infinity)
            .frame(height: 100)
            .background(Color(.systemGray6))
            .cornerRadius(16)
        }
    }
}

#Preview {
    HomeView(isLoggedIn: .constant(true))
}

