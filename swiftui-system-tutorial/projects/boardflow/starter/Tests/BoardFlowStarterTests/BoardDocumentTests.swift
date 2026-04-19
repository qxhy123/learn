import XCTest
@testable import BoardFlowStarter

final class BoardDocumentTests: XCTestCase {
    func testEmptyDocumentUsesUntitledBoardTitle() {
        XCTAssertEqual(BoardDocument.empty.title, "Untitled Board")
    }

    func testSampleBoardsContainThreeEntries() {
        XCTAssertEqual(BoardSummary.samples.count, 3)
    }
}
