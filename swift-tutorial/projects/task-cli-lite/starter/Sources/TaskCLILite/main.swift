import Foundation

struct Task {
    var title: String
    var isDone: Bool
}

struct TaskCLIProgram {
    private static let seedTasks = [
        Task(title: "read chapter 01", isDone: false),
        Task(title: "practice Swift let vs var", isDone: false),
        Task(title: "sketch TaskCLI Lite v1", isDone: false),
    ]

    static func run(arguments: [String]) -> String {
        guard let command = arguments.first else {
            return usage
        }

        switch command {
        case "list":
            return render(tasks: seedTasks)
        case "add":
            let title = arguments.dropFirst().joined(separator: " ").trimmingCharacters(in: .whitespacesAndNewlines)
            guard !title.isEmpty else {
                return "Missing task title.\n\(usage)"
            }

            var tasks = seedTasks
            tasks.append(Task(title: title, isDone: false))
            return "Added: \(title)\n" + render(tasks: tasks)
        case "done":
            let title = arguments.dropFirst().joined(separator: " ").trimmingCharacters(in: .whitespacesAndNewlines)
            guard !title.isEmpty else {
                return "Missing task title.\n\(usage)"
            }

            var tasks = seedTasks
            guard let index = tasks.firstIndex(where: { $0.title == title }) else {
                return "Task not found: \(title)\n" + render(tasks: tasks)
            }

            tasks[index].isDone = true
            return "Completed: \(title)\n" + render(tasks: tasks)
        default:
            return "Unknown command: \(command)\n\(usage)"
        }
    }

    private static func render(tasks: [Task]) -> String {
        let lines = tasks.enumerated().map { index, task in
            let status = task.isDone ? "[x]" : "[ ]"
            return "\(index + 1). \(status) \(task.title)"
        }

        return (["Today's tasks"] + lines).joined(separator: "\n")
    }

    private static let usage = """
    Usage:
      TaskCLILite list
      TaskCLILite add <title>
      TaskCLILite done <title>
    """
}

print(TaskCLIProgram.run(arguments: Array(CommandLine.arguments.dropFirst())))
