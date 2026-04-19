# 第11章：Swift Package Manager 与模块边界

> Part 2 把 `TaskCLI Lite` 的类型系统和建模判断推得更稳了，但代码仍然缺少一个真实工程最基本的东西：模块边界（module boundary）。到了 Part 3，项目必须从“会写一些像样的 Swift 代码”升级成“能被维护的 Swift package”。

## 为什么这一章现在出现

前两部分解决的是语言与建模问题：值语义怎么影响 API，错误为什么要被命名，协议边界何时值得出现。可如果这些判断始终停留在单一 executable target 里，它们就会有一个明显上限。

你会遇到这些真实压力：

- 哪些代码属于任务领域本身，哪些只是 CLI 入口细节？
- 哪些行为值得被单独测试，而不是每次都绕着命令行文本跑一整圈？
- 将来一旦引入更强的运行时行为、文件 I/O、乃至 SwiftUI 客户端，哪些东西应该被复用，哪些不该跟着命令行一起移动？

这就是为什么现在必须引入 Swift Package Manager（SPM）与模块边界。这里的重点不是“学会再写一个 `Package.swift`”，而是让项目第一次拥有可解释的物理结构：`TaskCore` 负责领域与核心行为，`TaskCLI` 负责命令行入口和编排。

对已经会别的语言的程序员来说，这一步尤其重要。很多人会把“拆包”误解成一种晚点再做的整理工作，好像现在只要代码还能跑，就先别管结构。Swift 教程如果这样走，后面测试、并发、可靠性和 UI 复用都会失去落点，因为系统根本没有边界可以承接这些主题。

## 从一个过弱的起点开始

先看 Part 1 终点附近那种完全合理、但不该继续扩展的状态：

```text
TaskCLILite/
├── Package.swift
├── Sources/
│   └── TaskCLILite/
│       └── main.swift
└── Tests/
    └── TaskCLILiteTests/
        └── TaskCLILiteTests.swift
```

这个结构在 Part 1 是对的，因为当时最重要的是把 Swift 基础语义压到一个最小 CLI 上。问题在于，当你走到 Part 3，它开始暴露三个工程弱点：

- `main.swift` 同时承载数据模型、命令处理和文本输出，职责开始堆叠
- 测试天然更容易围绕 CLI 字符串，而不是围绕核心行为本身
- 没有一个可以被后续章节继续加强的共享核心（shared core）

这里要特别诚实一点：弱起点不是“错误代码”，而是“已经完成前一阶段使命的代码”。教程真正稳的地方，不是神化旧版本，而是知道它什么时候该退位。

## 先把 package shape 立起来

Part 3 的 starter package 先不追求复杂，而是追求模块含义清楚。我们把项目升级为：

```text
taskcore-taskcli/starter/
├── Package.swift
├── Sources/
│   ├── TaskCore/
│   │   ├── Task.swift
│   │   └── TaskStore.swift
│   └── TaskCLI/
│       └── main.swift
└── Tests/
    └── TaskCoreTests/
        └── TaskCoreTests.swift
```

这棵树真正重要的，不是文件数量变多，而是判断开始被写进目录：

- `TaskCore/` 表示任务领域和核心状态变化
- `TaskCLI/` 表示命令行入口层
- `TaskCoreTests/` 明确说明：我们现在优先锁定 core behavior，而不是把一切都绑定在 CLI 文本快照上

这就是 SPM 在教程里的第一个工程价值。它不是只负责“帮你下载依赖”的工具，而是 Swift 项目表达模块关系的第一语言。

## `Package.swift` 如何把边界写成事实

在这一阶段，manifest 的目标不是花哨，而是把边界表达准确：

```swift
// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "TaskCoreTaskCLI",
    products: [
        .library(name: "TaskCore", targets: ["TaskCore"]),
        .executable(name: "TaskCLI", targets: ["TaskCLI"]),
    ],
    targets: [
        .target(name: "TaskCore"),
        .executableTarget(
            name: "TaskCLI",
            dependencies: ["TaskCore"]
        ),
        .testTarget(
            name: "TaskCoreTests",
            dependencies: ["TaskCore"]
        ),
    ]
)
```

这几行 manifest 表达了三个关键决定：

1. `TaskCore` 是 library product。它不是某个 CLI 文件夹里的内部实现细节，而是一个可以被别的客户端依赖的共享模块。
2. `TaskCLI` 是 executable product。它依赖 `TaskCore`，但不反过来被 core 依赖。
3. 测试目标先依赖 `TaskCore`。这意味着测试的主要对象是核心行为，而不是命令行入口上的每一段文本格式。

这就是“模块边界”真正的含义：不是把文件移进不同文件夹，而是把依赖方向（dependency direction）写死成工程事实。

## `TaskCore` 应该收什么，`TaskCLI` 不该偷什么

边界一旦开始出现，最容易发生的误判有两种。

第一种误判是拆了目录，却没拆职责。比如把 `Task` 放进 `TaskCore`，但又把标题清洗、完成状态规则、查找逻辑继续留在 `main.swift`。这会让 library 看起来存在，实际上核心判断仍然漂在 CLI 层。

第二种误判是抽象过头。一上来就想建 `TaskService`, `CommandCoordinator`, `StorageManager`, `RendererProtocolFactory` 之类的大词。对 Part 3 来说，这会把读者从包工程带进架构模板。

当前版本更稳的分工是：

- `TaskCore.Task`：任务领域模型，表达 `id`、`title`、`status`
- `TaskCore.TaskStore`：核心状态容器和基础行为，例如 `add(title:)` 与 `markDone(title:)`
- `TaskCLI.main.swift`：读取命令、调用 `TaskCore`、把结果渲染成 CLI 文本

例如，下面这种代码就明显属于 core，而不该散在 CLI 入口：

```swift
public struct TaskStore {
    public private(set) var tasks: [Task]

    @discardableResult
    public mutating func add(title: String) throws -> Task {
        let normalized = Task.normalizeTitle(title)
        guard !normalized.isEmpty else {
            throw TaskStoreError.emptyTitle
        }

        let task = Task(id: nextID, title: normalized)
        tasks.append(task)
        return task
    }
}
```

这里表达的是领域规则：标题不能是空白、任务要分配新 id、加入 store 后成为新的 core state。它不是“命令行的特殊癖好”，所以应该待在 `TaskCore`。

相反，这类逻辑则更像 CLI 层工作：

```swift
let command = arguments.first
let title = arguments.dropFirst().joined(separator: " ")
```

因为它们处理的是命令行参数形状，而不是任务领域本身。

## 从“文件分组”进化为“可复用核心”

到这里，你应该开始建立一个更高阶的直觉：模块边界不是为了当前一个命令能不能跑，而是为了未来系统能否沿着正确方向进化。

一旦 `TaskCore` 成型，后续章节会自然受益：

- 第12章可以直接测试核心行为
- 第13章可以识别解析、渲染、存储接缝
- 第14章可以讨论 CLI 组织而不把业务规则埋在入口里
- Part 4 可以继续强化 runtime behavior
- Part 5/6 的 `TaskFlow` 也才有共享核心可以站上去

这就是教程为什么一定要在这里切包。不是因为“真实项目都该多模块”，而是因为我们的共享任务领域终于有资格被当成共享资产对待。

## 双语关键词

- Swift Package Manager：Swift 包管理器
- package：包
- module：模块
- target：目标
- product：产物
- library target：库目标
- executable target：可执行目标
- dependency direction：依赖方向
- module boundary：模块边界
- shared core：共享核心

## 常见错误

### 1. 只拆目录，不拆职责

把 `Task.swift` 放进 `TaskCore/` 不代表边界已经成立。真正的判断标准是：任务规则是否真的从 CLI 入口抽离出来了。

### 2. 一上来就追求复杂架构

Part 3 的目标是清楚边界，不是制造一整套 command framework。能用两个 target 把职责说清楚，就不要提前建一堆层。

### 3. 让 `TaskCore` 反向依赖 CLI 细节

一旦 core 开始知道 usage 文本、命令名、终端输出格式，边界就倒了。记住依赖方向：CLI 依赖 core，不是 core 依赖 CLI。

### 4. 以为 SPM 只是构建工具

`swift build` 当然重要，但在工程判断上，SPM 更关键的价值是把模块、依赖和测试关系写进项目结构。

## English Recap

This chapter turns the project from a single executable into a real Swift package with two targets: `TaskCore` and `TaskCLI`. The key lesson is not “more files,” but clearer dependency direction: domain behavior lives in the library target, while command-line coordination lives in the executable target. That boundary is what makes the rest of Part 3 possible.

## Drills

1. 重新画一遍 `TaskCLI Lite` 和 `TaskCore + TaskCLI` 的目录树，写出每个目录分别代表什么工程判断。
2. 打开 `starter/Package.swift`，用自己的话解释 `products` 和 `targets` 为什么要同时存在。
3. 找出一个你认为明显属于 `TaskCore` 的行为，以及一个明显属于 `TaskCLI` 的行为，说明理由。

## Project Handoff

现在 package shape 已经立住，但“拆包”本身还不够。下一章要回答更尖锐的问题：边界已经有了，哪些行为应该先被 XCTest 锁定，才能防止我们在继续重构 CLI 和接缝时把核心规则改坏。
