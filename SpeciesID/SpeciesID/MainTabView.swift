import SwiftUI

struct MainTabView: View {
    @EnvironmentObject var authManager: AuthenticationManager
    @EnvironmentObject var syncService: SyncService
    @State private var selectedTab: Tab = .home

    enum Tab: String {
        case home
        case capture
        case species
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
                SupportedSpeciesView()
            }
            .tabItem {
                Label("Species", systemImage: "fish.fill")
            }
            .tag(Tab.species)

            NavigationStack {
                ObservationHistoryView()
            }
            .tabItem {
                Label("Observations", systemImage: "list.bullet")
            }
            .tag(Tab.observations)

            NavigationStack {
                SettingsView()
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
    MainTabView()
        .environmentObject(AuthenticationManager())
        .environmentObject(SyncService())
}
