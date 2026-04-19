import Foundation

enum Command: String {
    case list
    case add
    case done
}

func usage() -> String {
    """
    TaskCLI Lite
      list
      add <title>
      done <title>
    """
}

let arguments = Array(CommandLine.arguments.dropFirst())

guard let first = arguments.first, let command = Command(rawValue: first) else {
    print(usage())
    exit(0)
}

switch command {
case .list:
    print("No tasks yet.")
case .add:
    let title = arguments.dropFirst().joined(separator: " ")
    print(title.isEmpty ? "Missing task title." : "Added: \(title)")
case .done:
    let title = arguments.dropFirst().joined(separator: " ")
    print(title.isEmpty ? "Missing task title." : "Completed: \(title)")
}
