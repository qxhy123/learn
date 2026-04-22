import SwiftUI
import FocusCore

struct ProjectsView: View {
    @Bindable var store: FocusStore

    var body: some View {
        List(store.projects) { project in
            Text(project.name)
        }
        .navigationTitle("Projects")
    }
}
