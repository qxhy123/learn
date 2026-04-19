import XCTest
@testable import TaskCLILite

final class TaskCLILiteTests: XCTestCase {
    func testUsageMentionsSupportedCommands() {
        let text = usage()

        XCTAssertTrue(text.contains("list"))
        XCTAssertTrue(text.contains("add <title>"))
        XCTAssertTrue(text.contains("done <title>"))
    }
}
