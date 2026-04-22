import SwiftUI

struct SettingsView: View {
    @State private var showCompletedTasks = true
    @State private var useDenseLayout = false

    var body: some View {
        Form {
            Toggle("Show completed tasks", isOn: $showCompletedTasks)
            Toggle("Use dense layout", isOn: $useDenseLayout)
        }
        .padding()
        .navigationTitle("Settings")
    }
}
