import Foundation
import Testing
@testable import FocusCore

@Test func addTaskStoresItInInbox() {
    let store = FocusStore()
    store.addTask(title: "Write first SwiftUI screen")
    #expect(store.inboxTasks.count == 1)
    #expect(store.inboxTasks[0].title == "Write first SwiftUI screen")
}

@Test func completingTaskMarksItDone() {
    let store = FocusStore()
    store.addTask(title: "Review Part 1 draft")
    let task = try! #require(store.inboxTasks.first)
    store.toggleCompletion(task.id)
    #expect(store.inboxTasks[0].isDone)
}

@Test func togglingUnknownTaskDoesNothing() {
    let store = FocusStore()
    let before = store.inboxTasks
    store.toggleCompletion(UUID())
    #expect(store.inboxTasks == before)
}
