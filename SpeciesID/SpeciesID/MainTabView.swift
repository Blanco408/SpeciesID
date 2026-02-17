import SwiftUI

struct MainTabView: View {
    @Binding var isLoggedIn: Bool
    @State private var selectedTab: Tab = .home

    enum Tab: String {
        case home
        case capture
        case observations
        case settings
    }

    var body: some View {
        TabView(selection: $selectedTab) {
            NavigationStack {
                HomeView()
            }
            .tabItem {
                Label("Home", systemImage: "house.fill")
            }
            .tag(Tab.home)

            NavigationStack {
                CameraView()
            }
            .tabItem {
                Label("Capture", systemImage: "camera.fill")
            }
            .tag(Tab.capture)

            NavigationStack {
                ObservationHistoryView()
            }
            .tabItem {
                Label("Observations", systemImage: "list.bullet")
            }
            .tag(Tab.observations)

            NavigationStack {
                SettingsView(isLoggedIn: $isLoggedIn)
            }
            .tabItem {
                Label("Settings", systemImage: "gearshape.fill")
            }
            .tag(Tab.settings)
        }
        .tint(AppColors.darkGreen)
        .onAppear {
            AppColors.configureTabBarAppearance()
        }
    }
}

#Preview {
    MainTabView(isLoggedIn: .constant(true))
}
