import SwiftUI
import FocusCore

struct FocusListRootView: View {
    @Bindable var store: FocusStore

    var body: some View {
        NavigationSplitView {
            List {
                NavigationLink("Inbox") {
                    InboxView(store: store)
                }
                NavigationLink("Projects") {
                    ProjectsView(store: store)
                }
                NavigationLink("Settings") {
                    SettingsView()
                }
            }
            .navigationTitle("FocusList")
        } detail: {
            InboxView(store: store)
        }
    }
}
