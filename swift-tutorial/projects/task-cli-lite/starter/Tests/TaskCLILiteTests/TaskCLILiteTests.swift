import XCTest
@testable import TaskCLILite

final class TaskCLILiteTests: XCTestCase {
    func testNoArgumentsShowsUsage() {
        let output = TaskCLIProgram.run(arguments: [])

        XCTAssertTrue(output.contains("Usage:"))
        XCTAssertTrue(output.contains("TaskCLILite list"))
    }

    func testListShowsSeedTasks() {
        let output = TaskCLIProgram.run(arguments: ["list"])

        XCTAssertTrue(output.contains("Today's tasks"))
        XCTAssertTrue(output.contains("[ ] read chapter 01"))
    }

    func testAddAppendsTaskTitle() {
        let output = TaskCLIProgram.run(arguments: ["add", "write", "notes"])

        XCTAssertTrue(output.contains("Added: write notes"))
        XCTAssertTrue(output.contains("[ ] write notes"))
    }

    func testAddWithoutTitleShowsUsage() {
        let output = TaskCLIProgram.run(arguments: ["add"])

        XCTAssertTrue(output.contains("Missing task title."))
        XCTAssertTrue(output.contains("TaskCLILite add <title>"))
    }

    func testDoneMarksMatchingTask() {
        let output = TaskCLIProgram.run(arguments: ["done", "read", "chapter", "01"])

        XCTAssertTrue(output.contains("Completed: read chapter 01"))
        XCTAssertTrue(output.contains("[x] read chapter 01"))
    }

    func testUnknownCommandShowsUsage() {
        let output = TaskCLIProgram.run(arguments: ["remove"])

        XCTAssertTrue(output.contains("Unknown command: remove"))
        XCTAssertTrue(output.contains("TaskCLILite done <title>"))
    }
}
