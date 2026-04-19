import Foundation

public enum TaskStatus: String, Equatable {
    case pending
    case done
}

public struct Task: Equatable {
    public let id: Int
    public private(set) var title: String
    public private(set) var status: TaskStatus

    public init(id: Int, title: String, status: TaskStatus = .pending) {
        self.id = id
        self.title = Task.normalizeTitle(title)
        self.status = status
    }

    public var isDone: Bool {
        status == .done
    }

    public var statusMarker: String {
        isDone ? "[x]" : "[ ]"
    }

    public var cliLine: String {
        "\(statusMarker) \(title)"
    }

    @discardableResult
    public mutating func markDone() -> Bool {
        guard !isDone else {
            return false
        }

        status = .done
        return true
    }

    public static func normalizeTitle(_ title: String) -> String {
        title.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
