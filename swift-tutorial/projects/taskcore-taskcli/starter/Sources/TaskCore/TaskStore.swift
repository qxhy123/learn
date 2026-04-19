import Foundation

public enum TaskStoreError: Error, Equatable {
    case emptyTitle
    case taskNotFound(title: String)
    case taskAlreadyDone(title: String)
}

public struct TaskStore: Equatable {
    public private(set) var tasks: [Task]

    public init(tasks: [Task] = []) {
        self.tasks = tasks
    }

    public static func seeded() -> TaskStore {
        TaskStore(tasks: [
            Task(id: 1, title: "read chapter 11"),
            Task(id: 2, title: "split TaskCore from TaskCLI"),
            Task(id: 3, title: "write XCTest coverage"),
        ])
    }

    @discardableResult
    public mutating func add(title: String) throws -> Task {
        let normalized = Task.normalizeTitle(title)
        guard !normalized.isEmpty else {
            throw TaskStoreError.emptyTitle
        }

        let task = Task(id: nextID, title: normalized)
        tasks.append(task)
        return task
    }

    @discardableResult
    public mutating func markDone(title: String) throws -> Task {
        let normalized = Task.normalizeTitle(title)
        guard !normalized.isEmpty else {
            throw TaskStoreError.emptyTitle
        }

        guard let index = tasks.firstIndex(where: { $0.title == normalized }) else {
            throw TaskStoreError.taskNotFound(title: normalized)
        }

        guard tasks[index].markDone() else {
            throw TaskStoreError.taskAlreadyDone(title: normalized)
        }

        return tasks[index]
    }

    public func task(named title: String) -> Task? {
        let normalized = Task.normalizeTitle(title)
        return tasks.first { $0.title == normalized }
    }

    private var nextID: Int {
        (tasks.map(\.id).max() ?? 0) + 1
    }
}
