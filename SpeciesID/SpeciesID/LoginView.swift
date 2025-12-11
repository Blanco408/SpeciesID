import SwiftUI

struct LoginView: View {
    @State private var email: String = ""
    @State private var password: String = ""
    @State private var showEmailLogin: Bool = false
    @State private var isLoading: Bool = false
    
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
            
            if isLoading {
                Color.black.opacity(0.3).ignoresSafeArea()
                ProgressView().scaleEffect(1.2).tint(.white)
            }
        }
        .sheet(isPresented: $showEmailLogin) {
            EmailLoginView(
                email: $email,
                password: $password,
                isLoading: $isLoading,
                onSignIn: handleEmailSignIn,
                onDismiss: { showEmailLogin = false }
            )
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
        .disabled(isLoading)
    }
    
    private func handleGoogleSignIn() {
        isLoading = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
            isLoading = false
            print("Gmail Sign In tapped (stubbed)")
        }
    }
    
    private func handleAppleSignIn() {
        isLoading = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
            isLoading = false
            print("Apple Sign In tapped (stubbed)")
        }
    }
    
    private func handleEmailSignIn() {
        isLoading = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
            isLoading = false
            showEmailLogin = false
            print("Email Sign In tapped (stubbed)")
        }
    }
}

struct EmailLoginView: View {
    @Binding var email: String
    @Binding var password: String
    @Binding var isLoading: Bool
    let onSignIn: () -> Void
    let onDismiss: () -> Void
    
    var body: some View {
        NavigationView {
            VStack(spacing: 24) {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Email").font(.system(size: 14, weight: .medium)).foregroundColor(.secondary)
                    TextField("Enter your email", text: $email)
                        .textFieldStyle(.roundedBorder)
                        .keyboardType(.emailAddress)
                        .autocapitalization(.none)
                        .disabled(isLoading)
                }
                
                VStack(alignment: .leading, spacing: 8) {
                    Text("Password").font(.system(size: 14, weight: .medium)).foregroundColor(.secondary)
                    SecureField("Enter your password", text: $password)
                        .textFieldStyle(.roundedBorder)
                        .disabled(isLoading)
                }
                
                Button(action: onSignIn) {
                    Text("Sign In")
                        .font(.system(size: 17, weight: .semibold))
                        .frame(maxWidth: .infinity)
                        .frame(height: 50)
                        .foregroundColor(.white)
                        .background(Color.green)
                        .cornerRadius(12)
                }
                .disabled(isLoading || email.isEmpty || password.isEmpty)
                
                Spacer()
            }
            .padding()
            .navigationTitle("Sign In")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel", action: onDismiss).disabled(isLoading)
                }
            }
        }
    }
}

#Preview {
    LoginView()
}

