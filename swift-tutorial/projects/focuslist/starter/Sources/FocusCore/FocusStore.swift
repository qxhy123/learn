import Foundation
import Observation

@Observable
public final class FocusStore {
    public private(set) var inboxTasks: [FocusTask]
    public private(set) var projects: [FocusProject]

    public init(inboxTasks: [FocusTask] = [], projects: [FocusProject] = []) {
        self.inboxTasks = inboxTasks
        self.projects = projects
    }

    public static func sample() -> FocusStore {
        FocusStore(
            inboxTasks: [
                FocusTask(title: "Review Part 1 storyboard"),
                FocusTask(title: "Draft the FocusCore boundary")
            ],
            projects: [
                FocusProject(name: "Tutorial Rewrite"),
                FocusProject(name: "Release Prep")
            ]
        )
    }

    public func addTask(title: String) {
        guard !title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        inboxTasks.append(FocusTask(title: title))
    }

    public func toggleCompletion(_ id: UUID) {
        guard let index = inboxTasks.firstIndex(where: { $0.id == id }) else { return }
        inboxTasks[index].isDone.toggle()
    }
}
