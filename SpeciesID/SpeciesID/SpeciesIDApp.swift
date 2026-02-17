import SwiftUI
import FirebaseCore

class AppDelegate: NSObject, UIApplicationDelegate {
    func application(_ application: UIApplication,
                     didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey : Any]? = nil) -> Bool {
        // Check if GoogleService-Info.plist exists before configuring Firebase
        if let path = Bundle.main.path(forResource: "GoogleService-Info", ofType: "plist"),
           FileManager.default.fileExists(atPath: path) {
            FirebaseApp.configure()
        } else {
            print("""
            ERROR: GoogleService-Info.plist not found
            
            The app will crash because Firebase cannot be configured.
            
            To fix this:
            1. Go to https://console.firebase.google.com
            2. Open the SpeciesID project
            3. Gear icon → Project Settings → scroll to "Your apps" → iOS app
            4. Download GoogleService-Info.plist
            5. In Xcode: Drag it into the SpeciesID folder (same folder as ContentView.swift)
            6. Make sure "Copy items if needed" is checked
            7. Make sure it's added to the "SpeciesID" target
            """)
            // Don't crash in development - Firebase features just won't work
            // Uncomment the line below if you want the app to crash with a clear error:
            // fatalError("GoogleService-Info.plist is required. See console for instructions.")
        }
        return true
    }
}

@main
struct SpeciesIDApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) var delegate
    @AppStorage("isLoggedIn") private var isLoggedIn: Bool = false
    
    var body: some Scene {
        WindowGroup {
            if isLoggedIn {
                MainTabView(isLoggedIn: $isLoggedIn)
            } else {
                LoginView(isLoggedIn: $isLoggedIn)
            }
        }
    }
}
