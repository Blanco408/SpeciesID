//
//  LoginView.swift
//  SpeciesID
//
//  Login screen with Apple Sign In and Email authentication options
//

import SwiftUI

struct LoginView: View {
    @State private var email: String = ""
    @State private var password: String = ""
    @State private var isSigningIn: Bool = false
    
    var body: some View {
        ZStack {
            // Background gradient - green tones for EcoSnap branding
            LinearGradient(
                colors: [
                    Color(red: 0.2, green: 0.6, blue: 0.4),
                    Color(red: 0.15, green: 0.5, blue: 0.35)
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()
            
            VStack(spacing: 0) {
                Spacer()
                
                // App branding section
                VStack(spacing: 16) {
                    // App icon placeholder (can be replaced with actual logo)
                    Image(systemName: "camera.macro.circle.fill")
                        .font(.system(size: 80))
                        .foregroundColor(.white)
                    
                    Text("EcoSnap")
                        .font(.system(size: 42, weight: .bold, design: .rounded))
                        .foregroundColor(.white)
                    
                    Text("Snapshots to Species")
                        .font(.system(size: 18, weight: .medium))
                        .foregroundColor(.white.opacity(0.9))
                }
                .padding(.bottom, 60)
                
                Spacer()
                
                // Sign in options section
                VStack(spacing: 16) {
                    // Apple Sign In button
                    Button(action: {
                        handleAppleSignIn()
                    }) {
                        HStack {
                            Image(systemName: "applelogo")
                                .font(.system(size: 18, weight: .semibold))
                            Text("Continue with Apple")
                                .font(.system(size: 17, weight: .semibold))
                        }
                        .foregroundColor(.black)
                        .frame(maxWidth: .infinity)
                        .frame(height: 50)
                        .background(Color.white)
                        .cornerRadius(12)
                    }
                    .disabled(isSigningIn)
                    
                    // Email Sign In button
                    Button(action: {
                        handleEmailSignIn()
                    }) {
                        HStack {
                            Image(systemName: "envelope.fill")
                                .font(.system(size: 16, weight: .medium))
                            Text("Continue with Email")
                                .font(.system(size: 17, weight: .semibold))
                        }
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .frame(height: 50)
                        .background(Color.white.opacity(0.2))
                        .cornerRadius(12)
                        .overlay(
                            RoundedRectangle(cornerRadius: 12)
                                .stroke(Color.white.opacity(0.3), lineWidth: 1)
                        )
                    }
                    .disabled(isSigningIn)
                }
                .padding(.horizontal, 32)
                .padding(.bottom, 40)
            }
        }
    }
    
    // MARK: - Actions (stubbed for now - will be implemented by teammate)
    
    private func handleAppleSignIn() {
        isSigningIn = true
        // TODO: Implement Apple Sign In authentication
        // This will be handled by the authentication teammate
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            isSigningIn = false
        }
    }
    
    private func handleEmailSignIn() {
        isSigningIn = true
        // TODO: Implement Email Sign In authentication
        // This will be handled by the authentication teammate
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            isSigningIn = false
        }
    }
}

#Preview {
    LoginView()
}

