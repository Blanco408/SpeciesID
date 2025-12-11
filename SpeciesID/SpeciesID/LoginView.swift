//
//  LoginView.swift
//  SpeciesID
//
//  Created for EcoSnap - Species Identification Capstone
//

import SwiftUI

struct LoginView: View {
    var body: some View {
        VStack(spacing: 0) {
            Spacer()
            
            // Logo and Branding Section
            VStack(spacing: 16) {
                Image(systemName: "camera.macro.circle.fill")
                    .font(.system(size: 80))
                    .foregroundColor(.green)
                
                Text("EcoSnap")
                    .font(.system(size: 42, weight: .bold, design: .rounded))
                    .foregroundColor(.primary)
                
                Text("Snapshots to Species")
                    .font(.system(size: 18, weight: .medium))
                    .foregroundColor(.secondary)
            }
            
            Spacer()
        }
        .padding()
    }
}

#Preview {
    LoginView()
}

