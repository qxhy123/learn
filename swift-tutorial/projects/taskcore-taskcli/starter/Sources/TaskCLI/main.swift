import Foundation
import TaskCore

struct TaskCLIProgram {
    static func run(arguments: [String], seedStore: TaskStore = .seeded()) -> String {
        var store = seedStore

        guard let command = arguments.first else {
            return usage
        }

        switch command {
        case "list":
            return render(tasks: store.tasks)
        case "add":
            let title = normalizedTitle(from: arguments)
            guard !title.isEmpty else {
                return "Missing task title.\n\(usage)"
            }

            do {
                let task = try store.add(title: title)
                return "Added: \(task.title)\n" + render(tasks: store.tasks)
            } catch {
                return "Could not add task.\n\(usage)"
            }
        case "done":
            let title = normalizedTitle(from: arguments)
            guard !title.isEmpty else {
                return "Missing task title.\n\(usage)"
            }

            do {
                let task = try store.markDone(title: title)
                return "Completed: \(task.title)\n" + render(tasks: store.tasks)
            } catch let error as TaskStoreError {
                switch error {
                case .taskNotFound(let missingTitle):
                    return "Task not found: \(missingTitle)\n" + render(tasks: store.tasks)
                case .taskAlreadyDone(let repeatedTitle):
                    return "Task already done: \(repeatedTitle)\n" + render(tasks: store.tasks)
                case .emptyTitle:
                    return "Missing task title.\n\(usage)"
                }
            } catch {
                return "Could not complete task.\n" + render(tasks: store.tasks)
            }
        default:
            return "Unknown command: \(command)\n\(usage)"
        }
    }

    private static func normalizedTitle(from arguments: [String]) -> String {
        arguments
            .dropFirst()
            .joined(separator: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func render(tasks: [Task]) -> String {
        let lines = tasks.enumerated().map { index, task in
            "\(index + 1). \(task.cliLine)"
        }

        return (["Today's tasks"] + lines).joined(separator: "\n")
    }

    private static let usage = """
    Usage:
      TaskCLI list
      TaskCLI add <title>
      TaskCLI done <title>
    """
}

print(TaskCLIProgram.run(arguments: Array(CommandLine.arguments.dropFirst())))
