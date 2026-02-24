import SwiftUI

struct LoginView: View {
    @EnvironmentObject var authManager: AuthenticationManager
    @State private var email: String = ""
    @State private var password: String = ""
    @State private var showEmailLogin: Bool = false
    @State private var isSignUpMode: Bool = false
    @State private var showErrorAlert: Bool = false

    private let darkGreen = Color(red: 0.0, green: 0.5, blue: 0.2)
    private let lightGreen = Color(red: 0.4, green: 0.7, blue: 0.4)
    private let buttonGray = Color(red: 0.95, green: 0.95, blue: 0.95)

    var body: some View {
        ZStack {
            Color(.systemBackground).ignoresSafeArea()

            VStack(spacing: 10) {
                Spacer()

                VStack(spacing: 16) {
                    ZStack {
                        Image(systemName: "camera.fill")
                            .font(.system(size: 50))
                            .foregroundColor(lightGreen)

                        Image(systemName: "leaf.fill")
                            .font(.system(size: 20))
                            .foregroundColor(lightGreen)
                            .offset(x: 18, y: -18)
                    }

                    VStack(spacing: 4) {
                        Text("EcoSnap")
                            .font(.system(size: 42, weight: .bold))
                            .foregroundColor(darkGreen)
                        Text("Snapshots to Species")
                            .font(.system(size: 16, weight: .medium))
                            .foregroundColor(lightGreen)
                    }
                }

                Spacer()

                VStack(spacing: 8) {
                    Text("Welcome")
                        .font(.system(size: 36, weight: .bold))
                        .foregroundColor(darkGreen)
                    Text("Your species journey awaits")
                        .font(.system(size: 16, weight: .medium))
                        .foregroundColor(lightGreen)
                }
                .padding(.bottom, 30)

                VStack(spacing: 12) {
                    loginButton("Gmail Login", action: handleGoogleSignIn)
                    loginButton("Apple Login", action: handleAppleSignIn)
                    loginButton("Default Login", action: { showEmailLogin = true })
                    guestLoginButton()
                }
                .padding(.horizontal, 32)
                .padding(.bottom, 20)

                Text("By pressing on \"Continue with...\" you agree to our **Terms of Service** and **Privacy Policy**")
                    .font(.system(size: 11))
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 32)
                    .padding(.bottom, 40)
            }

            if authManager.isLoading {
                Color.black.opacity(0.3).ignoresSafeArea()
                ProgressView().scaleEffect(1.2).tint(.white)
            }
        }
        .sheet(isPresented: $showEmailLogin) {
            EmailLogin(
                email: $email,
                password: $password,
                isLoading: $authManager.isLoading,
                errorMessage: $authManager.authError,
                isSignUp: $isSignUpMode,
                onSubmit: handleEmailSubmit,
                onDismiss: {
                    showEmailLogin = false
                    authManager.authError = nil
                }
            )
        }
        .alert("Error", isPresented: $showErrorAlert) {
            Button("OK") { authManager.authError = nil }
        } message: {
            Text(authManager.authError ?? "An unknown error occurred.")
        }
        .onChange(of: authManager.authError) { _, newValue in
            if newValue != nil && !showEmailLogin {
                showErrorAlert = true
            }
        }
    }

    private func loginButton(_ title: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 16, weight: .medium))
                .foregroundColor(.primary)
                .frame(maxWidth: .infinity)
                .frame(height: 50)
                .background(buttonGray)
                .cornerRadius(12)
        }
        .disabled(authManager.isLoading)
    }

    private func guestLoginButton() -> some View {
        Button(action: handleGuestSignIn) {
            Text("Sign in as Guest")
                .font(.system(size: 16, weight: .medium))
                .foregroundColor(.secondary)
                .frame(maxWidth: .infinity)
                .frame(height: 50)
                .background(Color.clear)
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(Color.secondary.opacity(0.3), lineWidth: 1)
                )
                .contentShape(Rectangle())
        }
        .disabled(authManager.isLoading)
    }

    private func handleGoogleSignIn() {
        // Deferred to a later sprint
        print("Gmail Sign In tapped (deferred)")
    }

    private func handleAppleSignIn() {
        // Deferred to a later sprint
        print("Apple Sign In tapped (deferred)")
    }

    private func handleEmailSubmit() {
        Task {
            if isSignUpMode {
                await authManager.signUp(email: email, password: password)
            } else {
                await authManager.signIn(email: email, password: password)
            }
            if authManager.isAuthenticated {
                showEmailLogin = false
                email = ""
                password = ""
                isSignUpMode = false
            }
        }
    }

    private func handleGuestSignIn() {
        authManager.signInAsGuest()
    }
}

#Preview {
    LoginView()
        .environmentObject(AuthenticationManager())
}
