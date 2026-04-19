# 第15章：构建 TaskCore + TaskCLI v1

> Part 3 不能停在“我知道该怎么拆包、也知道该怎么写点测试”。这一章的任务，是把前四章全部收束成一个真实可构建、可测试、可运行的 starter package：`TaskCore + TaskCLI v1`。

## 为什么这个项目现在必须落地

如果教程在第14章结束，读者会再次陷入一种危险的中间态：好像每个工程主题都懂了，但系统还没有真正长成一个 package。真正能证明你已经进入 Swift 工程表面的，不是再多谈几个原则，而是把这些原则一起落进：

- `swift build`
- `swift test`
- `swift run TaskCLI ...`

这就是 `TaskCore + TaskCLI v1` 现在必须落地的原因。它不是 Part 3 的附录，而是 Part 3 的结论。只有当 package 真正存在，前面讲过的模块边界、XCTest、解析与渲染接缝、CLI 组织才算从“概念”变成“工程事实”。

## 从一个还不够强的过渡状态出发

假设我们只停留在“知道该拆分”的阶段，项目很可能仍然会有这些弱点：

- 目录里只有想法，没有 starter package
- package manifest 还没把 `TaskCore` 和 `TaskCLI` 写成真实 product
- 测试仍然没有落在 `TaskCore`
- 文档没有清楚说明“为什么拆”和“拆完后当前阶段达到哪里”

这意味着读者虽然知道答案方向，却还没有一个可以自己运行、阅读和验证的工程起点。教程最怕的正是这种“理解停在口头上”的状态。

## starter package 的最小完整形状

当前版本的 starter package 长这样：

```text
swift-tutorial/projects/taskcore-taskcli/starter/
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

它之所以适合作为 Part 3 终点，是因为它同时满足了“真实工程”与“教学可读性”：

- 有真正的 library/executable split
- 有领域模型与核心状态变换
- 有基于 XCTest 的自动化验证
- 有可运行的 CLI 入口
- 又没有提前塞入 Part 4 的 runtime complexity

这就是“最小完整”真正该有的样子。最小，不等于简陋；完整，也不等于做满所有未来需求。

## `TaskCore`：让领域和核心行为站到 package 中心

`Task.swift` 把最基础的领域事实写清楚：

```swift
public enum TaskStatus: String, Equatable {
    case pending
    case done
}

public struct Task: Equatable {
    public let id: Int
    public private(set) var title: String
    public private(set) var status: TaskStatus
}
```

这里没有故意追求复杂字段，也没有提前做持久化标识、时间戳、优先级系统。因为 Part 3 的目标不是扩张领域面，而是让包和行为边界站稳。

`TaskStore.swift` 则把本阶段最重要的 core behavior 收拢起来：

```swift
public struct TaskStore: Equatable {
    public private(set) var tasks: [Task]

    public static func seeded() -> TaskStore { ... }
    public mutating func add(title: String) throws -> Task { ... }
    public mutating func markDone(title: String) throws -> Task { ... }
}
```

这几组 API 足以承接 Part 3 当前的工程主题：

- 测试可以直接锁定核心状态变化
- CLI 可以只做命令输入和输出组织
- 存储与运行时强化仍有明确空间留给后续部分

换句话说，`TaskCore` 已经像一个真正的共享核心，而不是 CLI 内部的一个辅助文件夹。

## `TaskCLI`：保持入口真实，但不要把它写成第二个核心

`main.swift` 当前故意保持简单：

```swift
print(TaskCLIProgram.run(arguments: Array(CommandLine.arguments.dropFirst())))
```

然后由 `TaskCLIProgram` 承担命令行层的工作：

- 读取命令
- 做最基本的参数清洗
- 调用 `TaskStore`
- 把结果渲染成 plain text

例如：

```swift
case "done":
    let title = normalizedTitle(from: arguments)
    guard !title.isEmpty else {
        return "Missing task title.\n\(usage)"
    }

    do {
        let task = try store.markDone(title: title)
        return "Completed: \(task.title)\n" + render(tasks: store.tasks)
    } catch let error as TaskStoreError {
        // 映射为 CLI 文本
    }
```

这个设计的关键不在于“命令够不够多”，而在于 CLI 没有反过来占领核心规则。标题合法性、任务完成状态和失败类型仍然来自 `TaskCore`；CLI 只负责把它们变成用户看得懂的输出。

这就是 Part 3 要的 CLI：真实，但克制。

## 用 XCTest 和验证脚本把工程闭环补齐

Part 3 的 starter package 不是“有代码就行”，而是要有完整验证闭环。除了 `TaskCoreTests.swift` 里的具体单元测试，我们还配了一个专用验证脚本：

```bash
#!/usr/bin/env bash
set -euo pipefail

cd swift-tutorial/projects/taskcore-taskcli/starter
swift build
swift test
printf 'taskcore-taskcli-ok\n'
```

它证明了两件事：

- package 本身可以独立 build/test
- 教程当前阶段有一个清楚、局部、可重复的验证入口

这很重要，因为用户已经明确说过：不要依赖全局 `verify_projects.sh` / `verify_parts.sh` 来替代阶段性进展。Part 3 自己就应该能证明自己的 starter package 是好的。

## 文档为什么也是工程成果的一部分

这一章还必须同时完成三份项目文档：

- `README.md`：解释为什么 split 存在，以及 `TaskCore` / `TaskCLI` 各自负责什么
- `milestones/part3-v1.md`：说明当前阶段已经稳定了哪些 boundary
- `final/README.md`：明确 Part 4 会继续加强 runtime behavior，而不是在 Part 3 假装所有问题都做完了

这不是“教程写作附加题”，而是工程边界的一部分。因为当项目线开始跨 Part 演进时，读者需要一份清楚说明：当前 starter 解决了什么，又故意没解决什么。没有这些文档，Part 3 和 Part 4 的教学边界很容易互相污染。

## 为什么它现在已经足够叫做 `v1`

很多人会下意识把 `v1` 理解成“第一个勉强能跑的版本”。在这里，`TaskCore + TaskCLI v1` 的含义更准确一些：它是第一版真正拥有工程边界的任务系统起点。

它已经具备：

- 可解释的 package boundary
- 可复用的 shared core
- 可回归验证的 core behavior
- 可运行的命令行入口

它还没有具备：

- 文件持久化
- 更复杂的命令集
- 更强的 runtime reliability
- Part 4 要处理的并发与 I/O 压力

这正说明它是一个好 `v1`。因为它清楚知道自己已经稳在哪里，也清楚知道哪些问题还在后面。

## 双语关键词

- starter package：起始包
- shared core：共享核心
- plain text：纯文本
- verification script：验证脚本
- command-line entry：命令行入口
- runtime reliability：运行时可靠性
- milestone：里程碑
- handoff：交接 / 过渡

## 常见错误

### 1. 把 `v1` 理解成“所有功能都要上”

当前 `v1` 的重点是 package engineering。过早追求持久化和复杂命令，只会冲掉 Part 3 的教学重点。

### 2. 让 `TaskCLI` 重新长成第二个核心

如果 `main.swift` 里重新出现大量业务规则，拆包就失去意义了。

### 3. 只有代码，没有阶段文档

跨 Part 项目如果没有清楚文档，很容易让读者误以为“现在已经等于最终版”，或者反过来不知道当前进展到底意味着什么。

### 4. 只跑一次命令，不建立验证闭环

`swift build`、`swift test` 和专用验证脚本一起，才构成 Part 3 当前的最小工程证据。

## English Recap

This chapter assembles the full `TaskCore + TaskCLI v1` starter package. The project now has a real SwiftPM split, direct XCTest coverage on core behavior, a small but usable CLI entry point, and project docs that explain what Part 3 completes and what Part 4 will strengthen later.

## Drills

1. 用你自己的话解释为什么 `TaskStore` 现在放在 `TaskCore`，而不是继续放在 `main.swift`。
2. 运行验证脚本后，写一句话说明它证明了什么，没有证明什么。
3. 假设你要给项目再加一个 `reopen` 命令，先判断它会影响哪些文件，哪些文件当前不该被它碰。

## Project Handoff

到这里，Part 3 的工作才真正完成：项目已经正式变成 `TaskCore + TaskCLI`。下一部分不会推翻它，而是直接在这套边界上加强 runtime behavior、failure surface 和可靠性判断。也就是说，Part 4 的起点不是一个新项目，而是这个已经站稳的 `v1`。
