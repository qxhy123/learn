import Foundation

struct BoardSummary: Identifiable, Equatable {
    let id: UUID
    var title: String
    var cardCount: Int

    init(id: UUID = UUID(), title: String, cardCount: Int) {
        self.id = id
        self.title = title
        self.cardCount = cardCount
    }

    static let samples: [BoardSummary] = [
        BoardSummary(title: "Weekly Planning", cardCount: 8),
        BoardSummary(title: "Product Discovery", cardCount: 14),
        BoardSummary(title: "Research Synthesis", cardCount: 5),
    ]
}

struct BoardDocument: Equatable {
    var title: String
    var boards: [BoardSummary]

    static let empty = BoardDocument(title: "Untitled Board", boards: BoardSummary.samples)
}
