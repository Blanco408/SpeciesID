import SwiftUI

struct CameraView: View {
    @Environment(\.dismiss) var dismiss
    
    var body: some View {
        NavigationView {
            VStack {
                Spacer()
                Text("Camera")
                    .font(.largeTitle)
                Text("To be updated")
                    .foregroundColor(.secondary)
                Spacer()
            }
            .navigationTitle("Camera")
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Back") { dismiss() }
                }
            }
        }
    }
}

