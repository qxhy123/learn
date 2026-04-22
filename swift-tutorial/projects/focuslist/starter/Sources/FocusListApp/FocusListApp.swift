import SwiftUI
import FocusCore

@main
struct FocusListApp: App {
    @State private var store = FocusStore.sample()

    var body: some Scene {
        WindowGroup {
            FocusListRootView(store: store)
        }
    }
}
