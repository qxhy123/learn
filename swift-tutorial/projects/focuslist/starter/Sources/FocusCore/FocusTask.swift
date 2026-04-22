import Foundation

public struct FocusTask: Identifiable, Equatable, Sendable {
    public let id: UUID
    public var title: String
    public var isDone: Bool

    public init(id: UUID = UUID(), title: String, isDone: Bool = false) {
        self.id = id
        self.title = title
        self.isDone = isDone
    }
}
