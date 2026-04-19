# 第12章：用 XCTest 锁定核心行为

> 模块边界立住之后，项目终于可以谈“测试到底该测什么”了。Part 3 的测试重点不是把终端输出拍成一堆字符串快照，而是先锁定 `TaskCore` 的核心行为（core behavior）。

## 为什么这一章现在出现

在单一 executable 的早期阶段，测试通常只能围绕 CLI 入口打转。那时这样做没问题，因为程序太小，系统也还没有明确的共享核心。可一旦你已经拥有 `TaskCore` 和 `TaskCLI`，继续把所有测试都绑在 `swift run` 的最终输出上，就会出现两个问题：

- 你测到的是整条路径的混合结果，很难知道失败到底来自解析、业务规则还是文本格式
- 你会不自觉地把测试重心放在“输出长得像不像”，而不是“任务行为有没有被守住”

教程在这里引入 XCTest，不是为了完成一个“工程化 checklist”，而是为了明确一件事：测试应先包围最稳定、最值得被保护的规则。

对于 `TaskCore + TaskCLI` 当前阶段，这些规则非常具体：

- 新任务标题是否会被规范化（normalize）
- 空标题是否会被拒绝
- 完成任务时是否真的修改了对应任务
- 已完成任务再次完成时，系统能否区分这是另一种失败

如果这些问题没有被测试锁定，后面你做 CLI 重组、存储接缝或 Part 4 runtime 改造时，系统就会变得非常脆。

## 从一个过弱的测试起点开始

很多人会自然写出下面这种测试：

```swift
func testListCommandPrintsTasks() {
    let output = TaskCLIProgram.run(arguments: ["list"])
    XCTAssertTrue(output.contains("Today's tasks"))
}
```

它不是没用，但它很弱。因为它只说明“某次 list 输出里有这行文字”，并没有真正锁住核心行为：

- `TaskStore.add(title:)` 是否分配了正确的 id？
- 标题两端空白是否被去掉？
- `markDone(title:)` 找不到任务时是不是 `taskNotFound`？
- 已完成任务再次完成时是不是 `taskAlreadyDone`？

换句话说，这类测试更像“入口 smoke test”，不够像“核心规则的回归保护网（regression net）”。

## 先决定测试边界：测 `TaskCore`，不是先测所有 CLI 文本

Part 3 的 starter package 把测试目标直接指向 `TaskCore`：

```swift
.testTarget(
    name: "TaskCoreTests",
    dependencies: ["TaskCore"]
)
```

这个决定非常重要。它在工程上等于在说：

“当前阶段，最值得优先保护的是领域行为，不是命令行字符串格式。”

于是我们的测试就能直接写成：

```swift
func testAddTrimsTitleAndAssignsNextID() throws {
    var store = TaskStore.seeded()

    let task = try store.add(title: "  build TaskCore + TaskCLI v1  ")

    XCTAssertEqual(task.id, 4)
    XCTAssertEqual(task.title, "build TaskCore + TaskCLI v1")
    XCTAssertEqual(store.tasks.last?.cliLine, "[ ] build TaskCore + TaskCLI v1")
}
```

这类测试的价值明显更高：

- 它直接触达核心 API
- 它断言了真正重要的状态变化
- 它不会因为某条 usage 文本换行而变得脆弱

注意这里并不是说 CLI 测试完全不重要，而是说当前阶段的优先级应该先落在 core behavior。Part 3 先立“系统规则的护栏”，之后再决定入口层要覆盖到什么程度。

## 测试要围绕行为，不要围绕“看起来像工程”

教程里最常见的一种测试误区，是 performative testing，也就是“表面上写了测试，实际上没有保护关键行为”。

例如：

```swift
func testSeededStoreExists() {
    let store = TaskStore.seeded()
    XCTAssertNotNil(store)
}
```

这几乎没有信息量。`TaskStore.seeded()` 能返回一个值，本来就是类型系统已经保证的事情。

更强的版本应该问：“这个 seeded store 对教程当前阶段到底重要在哪里？”

```swift
func testSeededStoreStartsWithThreeTasks() {
    let store = TaskStore.seeded()

    XCTAssertEqual(store.tasks.count, 3)
    XCTAssertEqual(store.tasks.first?.title, "read chapter 11")
    XCTAssertEqual(store.tasks.first?.status, .pending)
}
```

现在测试开始承接真正的教学语义了：当前 starter package 的初始状态是什么，读者运行 `list` 时到底会面对什么样的任务域。

这就是“测试 concrete behavior，而不是测试类型存在”的差别。

## 用 XCTest 把失败面（failure surface）说清楚

Part 2 已经讨论过错误建模，Part 3 则要求你把它落实到测试里。`TaskStoreError` 之所以值得存在，不是为了显得“更工程”，而是因为测试现在真的要区分不同失败。

比如空标题：

```swift
func testAddEmptyTitleThrowsError() {
    var store = TaskStore.seeded()

    XCTAssertThrowsError(try store.add(title: "   ")) { error in
        XCTAssertEqual(error as? TaskStoreError, .emptyTitle)
    }
}
```

比如找不到任务：

```swift
func testMarkDoneUnknownTitleThrowsNotFound() {
    var store = TaskStore.seeded()

    XCTAssertThrowsError(try store.markDone(title: "missing task")) { error in
        XCTAssertEqual(
            error as? TaskStoreError,
            .taskNotFound(title: "missing task")
        )
    }
}
```

比如重复完成：

```swift
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
```

这些测试带来的最大收益，是系统的 failure surface 开始变成可验证事实，而不是口头上的“后面再处理错误”。

## 什么时候需要 CLI 测试，什么时候先别贪多

现在你也许会问：既然 `TaskCLI` 也是真实目标，为什么 starter package 里没有再建一套 `TaskCLITests`？

因为 Part 3 当前阶段的重点是测试密度而不是测试数量。我们已经知道 CLI 层目前承担的是：

- 读取参数
- 调用 core
- 组织输出

这意味着 CLI 测试当然有价值，但它们的优先级低于 core 行为测试。若当前就把大量时间花在字符串输出断言上，很容易出现一种错觉：测试很多，所以系统很稳。实际上，如果核心规则没有被直接锁住，系统仍然可能在结构调整时悄悄变坏。

更稳的策略是：

1. 先直接测试 `TaskCore`
2. 再用少量 CLI smoke test 补入口验证
3. 让 CLI 的详细组织随着后续章节的命令架构调整一起演进

这是一种测试投资顺序判断，不是“CLI 不值得测试”。

## `swift test` 在教程里真正证明了什么

执行：

```bash
swift test
```

它当然在做构建和测试运行，但在 Part 3 的语境里，它还证明了几件更重要的事：

- `Package.swift` 的 target 关系是正确的
- `TaskCore` 对测试目标可见
- 当前核心行为已经被编码为可回归验证的断言
- 项目不再只是“读一遍代码觉得像是对的”，而是有自动化证据

这就是为什么本章的测试不是教程附属品，而是项目升级为真实工程表面的必要组成。

## 双语关键词

- XCTest：Swift 测试框架
- unit test：单元测试
- core behavior：核心行为
- regression：回归
- regression net：回归保护网
- failure surface：失败面
- smoke test：冒烟测试
- assertion：断言
- test target：测试目标

## 常见错误

### 1. 只测试 CLI 最终字符串

入口测试当然有用，但如果所有测试都只看字符串输出，你很难锁住核心规则，也很难定位失败来源。

### 2. 写“存在型测试”

像 `XCTAssertNotNil(store)` 这种测试信息量极低。测试应该证明行为和规则，而不是证明类型系统本来就保证的事情。

### 3. 不区分不同错误

空标题、找不到任务、重复完成，本来就不是同一种失败。如果测试不区分这些情况，错误建模就只是纸面设计。

### 4. 把测试当成文案快照工具

当前阶段真正值得被优先保护的是 core state transition。文案和换行可以 later adjust，但规则一旦漂移，后续章节会全部变脆。

## English Recap

This chapter shifts testing from CLI-output checking to direct verification of `TaskCore` behavior. The important move is to test domain rules such as title normalization, task completion, and specific failure cases with XCTest. That gives the package a real regression boundary before more CLI refactoring happens.

## Drills

1. 为 `TaskStore.seeded()` 再补一个断言，说明为什么这个断言比 `XCTAssertNotNil` 更有信息量。
2. 写出你认为 `TaskCLI` 未来最值得保留的一个 smoke test，但解释为什么它现在不是优先级最高。
3. 把 `taskNotFound` 和 `taskAlreadyDone` 的差别，用“用户看到什么”和“系统意味着什么”两句话分别说明。

## Project Handoff

核心行为现在已经有了测试护栏，下一步才适合继续讨论系统里的“接缝”问题。下一章我们会把视线从测试转向解析（parsing）、渲染（rendering）和存储（storage）：哪些地方已经出现变化压力，哪些边界该被看见，但又不该被过度设计。
