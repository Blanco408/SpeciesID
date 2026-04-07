import SwiftUI
import FirebaseCore

class AppDelegate: NSObject, UIApplicationDelegate {
    func application(_ application: UIApplication,
                     didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey : Any]? = nil) -> Bool {
        if let path = Bundle.main.path(forResource: "GoogleService-Info", ofType: "plist"),
           FileManager.default.fileExists(atPath: path) {
            FirebaseApp.configure()
        } else {
            print("ERROR: GoogleService-Info.plist not found. Firebase will not work.")
        }
        return true
    }
}

@main
struct SpeciesIDApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) var delegate
    @StateObject private var authManager = AuthenticationManager()
    @StateObject private var syncService = SyncService()

    var body: some Scene {
        WindowGroup {
            Group {
                if authManager.isAuthenticated {
                    MainTabView()
                        .environmentObject(authManager)
                        .environmentObject(syncService)
                } else {
                    LoginView()
                        .environmentObject(authManager)
                }
            }
            .onAppear {
                authManager.setup()
            }
            .onReceive(NotificationCenter.default.publisher(for: UIApplication.willEnterForegroundNotification)) { _ in
                if let uid = authManager.currentUser?.uid {
                    Task { await syncService.sync(userId: uid) }
                }
            }
        }
    }
}
