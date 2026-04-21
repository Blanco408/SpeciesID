import SwiftUI
import FirebaseCore
import GoogleSignIn

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
    func application(_ app: UIApplication,
                     open url: URL,
                     options: [UIApplication.OpenURLOptionsKey: Any] = [:]) -> Bool {
        return GIDSignIn.sharedInstance.handle(url)
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
            .onChange(of: authManager.currentUser?.uid) { _, uid in
                if let uid {
                    Task { await syncService.sync(userId: uid) }
                }
            }
            .onReceive(NotificationCenter.default.publisher(for: UIApplication.willEnterForegroundNotification)) { _ in
                if let uid = authManager.currentUser?.uid {
                    Task { await syncService.sync(userId: uid) }
                }
            }
        }
    }
}
