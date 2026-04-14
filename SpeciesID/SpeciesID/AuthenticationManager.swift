import Combine
import Foundation
import FirebaseCore
import FirebaseAuth
import FirebaseFirestore
import GoogleSignIn
import AuthenticationServices
import CryptoKit

struct AppUser {
    let uid: String
    let email: String
    let displayName: String
}

@MainActor
final class AuthenticationManager: ObservableObject {
    @Published var currentUser: AppUser?
    @Published var isAuthenticated: Bool = false
    @Published var authError: String?
    @Published var isLoading: Bool = false

    private var authStateHandle: AuthStateDidChangeListenerHandle?
    private var didSetup = false

    // No Firebase calls in init — Firebase isn't configured yet.
    // Call setup() after FirebaseApp.configure() has run (from .onAppear).
    init() {}

    func setup() {
        guard !didSetup, FirebaseApp.app() != nil else { return }
        didSetup = true
        listenForAuthState()
    }

    deinit {
        if let handle = authStateHandle, FirebaseApp.app() != nil {
            Auth.auth().removeStateDidChangeListener(handle)
        }
    }

    // MARK: - Auth State Listener

    private func listenForAuthState() {
        authStateHandle = Auth.auth().addStateDidChangeListener { [weak self] _, firebaseUser in
            Task { @MainActor [weak self] in
                guard let self else { return }
                if let firebaseUser {
                    self.isAuthenticated = true
                    await self.fetchUserProfile(userId: firebaseUser.uid, email: firebaseUser.email)
                } else {
                    self.isAuthenticated = false
                    self.currentUser = nil
                }
            }
        }
    }

    // MARK: - Sign In

    func signIn(email: String, password: String) async {
        guard FirebaseApp.app() != nil else {
            authError = "Firebase is not configured. Check GoogleService-Info.plist."
            return
        }

        isLoading = true
        authError = nil

        do {
            let result = try await Auth.auth().signIn(withEmail: email, password: password)
            let uid = result.user.uid
            // Update last login — don't fail sign-in if this errors
            try? await updateLastLogin(userId: uid)
            await fetchUserProfile(userId: uid, email: email)
        } catch {
            authError = mapFirebaseError(error)
        }

        isLoading = false
    }

    // MARK: - Sign Up

    func signUp(email: String, password: String) async {
        guard FirebaseApp.app() != nil else {
            authError = "Firebase is not configured. Check GoogleService-Info.plist."
            return
        }

        isLoading = true
        authError = nil

        do {
            let result = try await Auth.auth().createUser(withEmail: email, password: password)
            let uid = result.user.uid
            let displayName = email.components(separatedBy: "@").first ?? "User"
            try await createUserDocument(userId: uid, email: email, displayName: displayName)
            currentUser = AppUser(uid: uid, email: email, displayName: displayName)
        } catch {
            authError = mapFirebaseError(error)
        }

        isLoading = false
    }

    // MARK: - Sign Out

    func signOut() {
        guard FirebaseApp.app() != nil else {
            isAuthenticated = false
            currentUser = nil
            return
        }

        do {
            try Auth.auth().signOut()
            currentUser = nil
            isAuthenticated = false
            authError = nil
        } catch {
            authError = "Failed to sign out. Please try again."
            print("Sign out error: \(error.localizedDescription)")
        }
    }

    // MARK: - Guest Login

    func signInAsGuest() {
        isAuthenticated = true
        currentUser = nil
    }

    // MARK: - Firestore Helpers

    private func fetchUserProfile(userId: String, email: String?) async {
        let db = Firestore.firestore()
        do {
            let document = try await db.collection("users").document(userId).getDocument()
            if let data = document.data() {
                let displayName = data["display_name"] as? String ?? "User"
                let userEmail = data["email"] as? String ?? email ?? ""
                currentUser = AppUser(uid: userId, email: userEmail, displayName: displayName)
            } else {
                currentUser = AppUser(uid: userId, email: email ?? "", displayName: "User")
            }
        } catch {
            print("Failed to fetch user profile: \(error.localizedDescription)")
            currentUser = AppUser(uid: userId, email: email ?? "", displayName: "User")
        }
    }

    private func createUserDocument(userId: String, email: String, displayName: String) async throws {
        let db = Firestore.firestore()
        let data: [String: Any] = [
            "email": email,
            "display_name": displayName,
            "date_created": FieldValue.serverTimestamp(),
            "last_login": FieldValue.serverTimestamp(),
            "downloaded_regions": [] as [String],
            "preferences": [
                "auto_save_photos": true,
                "high_contrast_mode": false,
                "large_text_mode": false,
                "confidence_threshold": 0.7
            ]
        ]
        try await db.collection("users").document(userId).setData(data)
    }

    private func updateLastLogin(userId: String) async throws {
        let db = Firestore.firestore()
        try await db.collection("users").document(userId).updateData([
            "last_login": FieldValue.serverTimestamp()
        ])
    }

    // MARK: - Error Mapping

    private func mapFirebaseError(_ error: Error) -> String {
        let nsError = error as NSError
        print("Firebase auth error: \(nsError.code) - \(error.localizedDescription)")
        if let underlyingError = nsError.userInfo[NSUnderlyingErrorKey] {
            print("Underlying error: \(underlyingError)")
        }
        print("Full error details: \(nsError.userInfo)")

        guard nsError.domain == AuthErrorDomain else {
            return "An unexpected error occurred. Please try again."
        }

        switch AuthErrorCode(rawValue: nsError.code) {
        case .invalidEmail:
            return "Please enter a valid email address."
        case .userNotFound:
            return "No account found with this email. Would you like to sign up?"
        case .wrongPassword, .invalidCredential:
            return "Incorrect password. Please try again."
        case .networkError:
            return "No internet connection. Please check your network and try again."
        case .emailAlreadyInUse:
            return "An account with this email already exists. Try signing in instead."
        case .weakPassword:
            return "Password is too weak. Use at least 6 characters."
        case .tooManyRequests:
            return "Too many attempts. Please wait a moment and try again."
        case .userDisabled:
            return "This account has been disabled. Please contact support."
        default:
            return "Something went wrong. Please try again."
        }
    }
}
