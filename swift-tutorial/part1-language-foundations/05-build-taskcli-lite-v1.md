# 第5章：构建 TaskCLI Lite v1

> Part 1 不能停在“我已经分别学过几个语法点”。这一章的任务，是把前四章的工具链、值、集合、控制流、函数、Optional、枚举、结构体全部压到一个真实可运行的 SwiftPM 小项目里，并明确落在 `TaskCLI Lite v1`。

## 为什么这个项目现在必须出现

如果教程在第四章结束，读者会处在一种很危险的满足感里：好像每个主题都懂一点，但还没有真正把它们拼成一段完整工程路径。真正能检验你是否把 Swift 基础学进去的，不是再做几题局部练习，而是看你能不能把这些能力收束成一个可以：

- `swift build`
- `swift test`
- `swift run`

的小程序。

`TaskCLI Lite v1` 正适合承担这个角色。它足够小，不会让 Part 1 过早陷入架构噪音；它也足够真，因为命令输入、状态变化、渲染输出、测试验证这些真实程序必须面对的要素一个都没少。

## 从一个过弱版本出发

假设我们直接把前几章的练习代码糊成一个脚本，可能会长这样：

```swift
var tasks = ["read chapter 01", "write notes"]
let command = "list"

if command == "list" {
    print(tasks)
} else if command == "add" {
    tasks.append("new task")
    print(tasks)
} else if command == "done" {
    print("done")
}
```

它的问题不是“跑不起来”，而是它还远远不够成为 Part 1 终点：

- 没有 Swift Package Manager 项目结构
- 没有真正的命令行参数处理
- 任务还只是字符串数组，没有完成状态
- 输出格式粗糙，测试也不存在

所以这一章的工作不是“把脚本再写长一点”，而是把它推进成一个最小但完整的 package。

## 先把工程外壳搭好

Part 1 的 starter package 需要一个清楚但不过度设计的结构：

```text
starter/
├── Package.swift
├── Sources/
│   └── TaskCLILite/
│       └── main.swift
└── Tests/
    └── TaskCLILiteTests/
        └── TaskCLILiteTests.swift
```

这里只有一个 executable target：`TaskCLILite`。这是刻意的。Part 1 的教学目标不是一开始就拆库和可执行入口，而是先让读者在一个足够小的工程壳里看清 SwiftPM、主入口和测试目标的基本关系。

`Package.swift` 的最小版本可以长这样：

```swift
// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "TaskCLILite",
    products: [
        .executable(
            name: "TaskCLILite",
            targets: ["TaskCLILite"]
        )
    ],
    targets: [
        .executableTarget(
            name: "TaskCLILite"
        ),
        .testTarget(
            name: "TaskCLILiteTests",
            dependencies: ["TaskCLILite"]
        )
    ]
)
```

这份 manifest 做的事情非常少，但已经够 Part 1 使用：

- 声明 package 名称
- 提供一个 executable product
- 声明一个测试目标依赖主目标

## 先用测试把最小行为钉住

Part 1 不需要一大堆测试，但至少要有一份真实 XCTest，证明 CLI 的最小行为是可检查的。我们的 starter package 锁定三个核心动作：

- `list`
- `add <title>`
- `done <title>`

测试的价值不只是“防回归”，更重要的是迫使你先说清楚程序应该输出什么。例如：

```swift
func testListShowsSeedTasks() {
    let output = TaskCLIProgram.run(arguments: ["list"])

    XCTAssertTrue(output.contains("Today's tasks"))
    XCTAssertTrue(output.contains("[ ] read chapter 01"))
}
```

这类测试有一个非常适合 Part 1 的优点：它把函数、字符串、集合、命令参数、输出格式这些前面章节的内容重新压到一起，而且足够直观。

## 用最小实现完成 `TaskCLI Lite v1`

接下来把代码收束到 `main.swift`。注意，这里不是“最佳架构展示”，而是“当前教学阶段最合适的实现密度”。

核心数据模型：

```swift
struct Task {
    var title: String
    var isDone: Bool
}
```

程序入口逻辑：

```swift
struct TaskCLIProgram {
    private static let seedTasks = [
        Task(title: "read chapter 01", isDone: false),
        Task(title: "practice Swift let vs var", isDone: false),
        Task(title: "sketch TaskCLI Lite v1", isDone: false),
    ]
}
```

这里我们使用 seed tasks，而不是磁盘持久化。原因很重要：Part 1 要教的是 Swift 语言基础如何落到程序形状上，而不是提前把存储与架构问题塞满。内存态示例已经足够承接本阶段的学习目标。

真正的命令处理则围绕参数数组展开：

```swift
static func run(arguments: [String]) -> String {
    guard let command = arguments.first else {
        return usage
    }

    switch command {
    case "list":
        return render(tasks: seedTasks)
    case "add":
        let title = arguments.dropFirst().joined(separator: " ")
        // ...
    case "done":
        let title = arguments.dropFirst().joined(separator: " ")
        // ...
    default:
        return "Unknown command: \\(command)\n\\(usage)"
    }
}
```

这段代码为什么对 Part 1 来说恰到好处？

- `arguments.first` 让 Optional 处理自然出现
- `switch` 让命令集合保持显式
- `joined(separator: " ")` 把字符串和集合能力接进 CLI
- `render(tasks:)` 则把渲染逻辑从分支里提出来

再看渲染函数：

```swift
private static func render(tasks: [Task]) -> String {
    let lines = tasks.enumerated().map { index, task in
        let status = task.isDone ? "[x]" : "[ ]"
        return "\(index + 1). \(status) \(task.title)"
    }

    return (["Today's tasks"] + lines).joined(separator: "\n")
}
```

这里几乎把 Part 1 的核心语法都串起来了：

- `enumerated()` 让数组迭代带上序号
- 三元表达式（ternary expression）根据 `Bool` 选状态文本
- 字符串插值负责生成最终行输出
- `joined(separator:)` 生成整体文本

## 运行与验证：Part 1 终点必须可证明

一个章节真正结束，不是“代码贴完了”，而是你能在终端里证明它成立：

```bash
cd swift-tutorial/projects/task-cli-lite/starter
swift build
swift test
swift run TaskCLILite list
swift run TaskCLILite add "write chapter notes"
swift run TaskCLILite done "read chapter 01"
```

只有当这些动作都能完成时，`TaskCLI Lite v1` 才算真的落地。Part 1 的项目终点必须是可 build、可 test、可 run 的，这一点比“代码看上去挺像”更重要。

## 为什么这里故意不做更多

走到这里，很多程序员会自然想继续做下面这些事：

- 用文件保存任务
- 拆出独立模块
- 定义更完整的错误类型
- 引入更强的命令解析层

这些方向都没错，但它们不属于 Part 1 的最佳停止点。教程在这里停下，是因为我们要保护一件事：读者对 Swift 基础语言与最小程序骨架之间的直接连接。如果现在继续加工程层复杂度，很容易把本应清楚的语言训练掩盖掉。

## 双语关键词

- 包清单：package manifest
- 可执行目标：executable target
- 测试目标：test target
- 命令参数：command-line arguments
- 渲染：render
- 种子数据：seed data
- 单元测试：unit test / XCTest

## 常见错误

### 1. 过早把 Part 1 写成“小型架构秀”

此时最重要的是程序可解释、可运行、可测试，而不是提早发明一堆未来阶段才需要的抽象。

### 2. 忘记把标题参数重新拼成一个字符串

`add <title>` 和 `done <title>` 的标题通常由多个参数组成，所以需要 `dropFirst()` 后再 `joined(separator: " ")`。

### 3. 以为没有持久化就“不算项目”

Part 1 的项目价值，在于把语言基础落到真实 CLI 形状里，而不是一次完成全部工程层能力。

### 4. 只有代码，没有验证

教程主线要求项目阶段必须能 `build`、`test`、`run`。如果你没有真正执行这些命令，就还没有完成这一章。

## English Recap

This chapter turns all Part 1 concepts into a concrete SwiftPM project. `TaskCLI Lite v1` stays intentionally small: one executable target, one test target, in-memory tasks, and three commands (`list`, `add`, `done`). The point is not architectural sophistication; the point is proving that Swift fundamentals can already produce a real, testable CLI.

## Drills

1. 运行 `swift run TaskCLILite list`，读一遍输出，指出其中哪些部分分别来自数组遍历、字符串插值和布尔状态判断。
2. 给 `TaskCLIProgram.run(arguments:)` 增加一个“缺少标题时返回 usage”的思考题，不必立刻重构，只要写出你认为合理的输出文本。
3. 再写一个 XCTest，验证未知命令时会返回 `Unknown command` 提示。

## Project Handoff

Part 1 在这里明确落到 `TaskCLI Lite v1`。你现在拥有的是一个最小但真实的 Swift CLI：能 build、能 test、能 run，也能清楚解释为什么它还保持单目标、内存态和轻量命令处理。进入 Part 2 之后，教程不会换题，而是沿着这条线继续向前，把这里故意挤在一起的职责逐步演进为更工程化的 `TaskCore + TaskCLI`。
