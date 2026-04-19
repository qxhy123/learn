import SwiftUI

@main
struct BoardFlowApp: App {
    var body: some Scene {
        WindowGroup {
            BoardHomeView(document: .empty)
        }
    }
}
