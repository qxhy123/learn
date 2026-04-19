import XCTest
@testable import TaskCore

final class TaskCoreTests: XCTestCase {
    func testSeededStoreStartsWithThreeTasks() {
        let store = TaskStore.seeded()

        XCTAssertEqual(store.tasks.count, 3)
        XCTAssertEqual(store.tasks.first?.title, "read chapter 11")
        XCTAssertEqual(store.tasks.first?.status, .pending)
    }

    func testAddTrimsTitleAndAssignsNextID() throws {
        var store = TaskStore.seeded()

        let task = try store.add(title: "  build TaskCore + TaskCLI v1  ")

        XCTAssertEqual(task.id, 4)
        XCTAssertEqual(task.title, "build TaskCore + TaskCLI v1")
        XCTAssertEqual(store.tasks.last?.cliLine, "[ ] build TaskCore + TaskCLI v1")
    }

    func testAddEmptyTitleThrowsError() {
        var store = TaskStore.seeded()

        XCTAssertThrowsError(try store.add(title: "   ")) { error in
            XCTAssertEqual(error as? TaskStoreError, .emptyTitle)
        }
    }

    func testMarkDoneUpdatesMatchingTask() throws {
        var store = TaskStore.seeded()

        let task = try store.markDone(title: "split TaskCore from TaskCLI")

        XCTAssertTrue(task.isDone)
        XCTAssertEqual(store.task(named: "split TaskCore from TaskCLI")?.status, .done)
    }

    func testMarkDoneUnknownTitleThrowsNotFound() {
        var store = TaskStore.seeded()

        XCTAssertThrowsError(try store.markDone(title: "missing task")) { error in
            XCTAssertEqual(
                error as? TaskStoreError,
                .taskNotFound(title: "missing task")
            )
        }
    }

    func testMarkDoneTwiceThrowsAlreadyDone() throws {
        var store = TaskStore.seeded()
        _ = try store.markDone(title: "write XCTest coverage")

        XCTAssertThrowsError(try store.markDone(title: "write XCTest coverage")) { error in
            XCTAssertEqual(
                error as? TaskStoreError,
                .taskAlreadyDone(title: "write XCTest coverage")
            )
        }
    }
}
