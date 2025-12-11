import SwiftUI

struct ObservationHistoryView: View {
    @Environment(\.dismiss) var dismiss
    
    var body: some View {
        NavigationView {
            VStack {
                Spacer()
                Text("Observation History")
                    .font(.largeTitle)
                Text("To be updated")
                    .foregroundColor(.secondary)
                Spacer()
            }
            .navigationTitle("Observations")
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Back") { dismiss() }
                }
            }
        }
    }
}

