# 第14章：命令组织与 CLI 架构

> 到了这一步，`TaskCore` 已经提供了核心行为，解析与渲染接缝也被看见了。系统现在真正面临的问题是：CLI 入口要怎样继续生长，才不会重新长回一个大开关脚本？

## 为什么这一章现在出现

在工程化之前，CLI 最容易被写成“一段可以工作的分支逻辑”。这种写法在命令极少时完全合理，但一旦你已经拥有 package boundary 和 core module，命令组织（command organization）就不该再只是 `main.swift` 里越写越长的 `switch`。

此时你会面临几个典型压力：

- `list`、`add`、`done` 只是起点，后面命令数通常会增加
- usage 文本、参数验证、core 调用、输出格式正在逐渐缠在一起
- 一旦想单独理解“这个命令到底做什么”，你不得不先读完整个入口文件

这就是 CLI architecture 现在必须出现的原因。注意，这里的“架构”不是指 command framework，更不是指为了架构感去发明抽象，而是指：**命令行入口怎样保持可读、可改、可验证。**

## 从一个完全可以工作、但已经开始发紧的入口开始

当前 starter package 的入口是：

```swift
struct TaskCLIProgram {
    static func run(arguments: [String], seedStore: TaskStore = .seeded()) -> String {
        var store = seedStore

        guard let command = arguments.first else {
            return usage
        }

        switch command {
        case "list":
            return render(tasks: store.tasks)
        case "add":
            // 参数清洗、调用 store、组织输出
        case "done":
            // 参数清洗、错误分支、组织输出
        default:
            return "Unknown command: \(command)\n\(usage)"
        }
    }
}
```

这个版本的好处是很直接，读者一眼就能看到入口逻辑。它的问题也同样明显：

- 每增加一个命令，`run` 就更拥挤
- 参数解释和命令执行仍然耦合
- 输出和错误文案容易散落成很多局部字符串

也就是说，它已经从“适合教学的短程序”迈进“需要组织策略的入口程序”。

## CLI 架构的第一原则：不要让 `main.swift` 同时承担所有角色

对于当前阶段，我们至少要在脑中区分三件事：

1. 程序入口：拿到 `CommandLine.arguments`
2. 命令解释：把原始参数认成某种命令
3. 命令执行：调用 `TaskCore` 并组织结果文本

一个比较克制、但足够强的进化方式，是把入口保持极薄：

```swift
print(TaskCLIProgram.run(arguments: Array(CommandLine.arguments.dropFirst())))
```

然后让 `TaskCLIProgram` 承接 CLI 层的真正逻辑。这已经比 Part 1 的单文件脚本强很多，因为它至少承认入口和程序逻辑不是一回事。

下一步则是让命令形状更明确。比如：

```swift
enum TaskCommand {
    case list
    case add(title: String)
    case done(title: String)
}
```

有了它之后，CLI 入口就可以从“看字符串做分支”逐步进化成“解释命令，再执行命令”。

## 更强的组织方式：解析和执行分开，但别先建框架

一个适合 Part 3 的最小架构形状可以是：

```swift
struct TaskCLIProgram {
    static func run(arguments: [String], seedStore: TaskStore = .seeded()) -> String {
        var store = seedStore

        guard let command = parse(arguments: arguments) else {
            return usage
        }

        return execute(command: command, store: &store)
    }
}
```

这里真正发生的架构升级只有两件事：

- 解析阶段返回 `TaskCommand`
- 执行阶段只处理已经被识别过的命令值

这会显著改善入口文件的阅读体验。因为你终于可以分别讨论：

- 命令形状是否合理
- 解析失败时返回什么
- 执行阶段怎样和 `TaskCore` 交互

同时它又没有走向过度设计。我们没有发明 command protocol、子命令树、handler registry、容器注入，只是先把最纠缠的职责拆开。

这就是本章最重要的判断：CLI architecture 的目标是压低混乱度，不是提高“架构词汇密度”。

## usage、错误输出与帮助文本应怎样归位

CLI 系统还有一个很现实的组织点：用户文本。随着命令变多，usage 和错误信息很容易散在很多分支里。

当前 starter package 里，这种文本仍然集中在 `TaskCLIProgram` 中，这是合理的，因为它们都属于 CLI 层而不是 core 层。更强一点的做法，是开始把输出意图写得更一致：

- 缺少命令：返回 usage
- 缺少参数：返回具体错误 + usage
- 业务失败：返回领域错误映射后的 CLI 文本
- 成功：返回操作结果 + 当前任务列表

这不是漂亮排版问题，而是用户体验和可维护性问题。CLI 如果没有稳定的文案组织规则，入口层会越来越难改，因为每个分支都在自定义自己的语气和结构。

注意这里仍然不需要一个“国际化文案系统”或“消息工厂”。Part 3 只要求你先把输出组织原则立住。

## 命令组织为什么要服务 core，而不是反过来

这一章很容易被带偏成“CLI 才是主角”。但在我们的项目线上，CLI 是 `TaskCore` 的一个客户端，而不是领域本体。

这意味着命令组织应该做的，是更清楚地调用 core，而不是把核心规则重新拉回 CLI：

- `TaskStore.add(title:)` 的标题合法性判断应留在 core
- `TaskStore.markDone(title:)` 的失败分支定义应留在 core
- CLI 负责把这些行为结果映射成用户可读文本

这条边界一旦守住，CLI 架构再怎么增长，`TaskCore` 仍然能保持为共享核心。将来无论是 Part 4 加强 runtime，还是 Part 5/6 的 `TaskFlow` 复用，都不会被 CLI 绑架。

所以判断标准很简单：如果某段代码删掉 CLI 以后仍然成立，它更可能属于 core；如果它只服务命令行输入输出，它更可能属于 CLI architecture。

## 先让架构“可读”，再让架构“可扩”

很多程序员在工程化 CLI 时，会过早追求 extensibility（可扩展性），结果代码反而先失去 readability（可读性）。Part 3 更推荐的顺序正好相反：

1. 先让命令形状清楚
2. 先让解析和执行职责分离
3. 先让 usage 和错误输出有一致组织
4. 当命令数量和变化速度真的增加时，再讨论更强的抽象

这是因为当前系统的首要风险不是“未来不够可扩”，而是“现在已经开始不好读”。先解决当前混乱，才有资格谈未来扩展。

## 双语关键词

- CLI architecture：命令行架构
- command organization：命令组织
- entry point：入口点
- usage text：用法文本
- command execution：命令执行
- readability：可读性
- extensibility：可扩展性
- handler：处理器
- coordination：编排

## 常见错误

### 1. 把 `main.swift` 重新写成大总管

即使已经有 `TaskCore`，如果入口仍然同时处理参数、业务规则、文本渲染和错误恢复，它很快就会重新变成难以维护的脚本。

### 2. 一开始就发明 command framework

当前只有少量命令时，最需要的是清楚分层，而不是复杂基础设施。框架感不等于工程质量。

### 3. 把业务规则拉回 CLI

参数缺失和 usage 属于 CLI；空标题是否合法、重复完成算什么错误，则属于 core。别让命令层偷走领域判断。

### 4. 把“可扩展”误当成当前第一目标

现在最需要的是可读、可改、可验证。没有这些，所谓 extensibility 只会把复杂度提前释放出来。

## English Recap

This chapter shows how CLI code should grow once the project has a shared core. The important move is to separate entry, parsing, and execution without inventing a full command framework too early. Good CLI architecture keeps `main.swift` thin and keeps domain rules inside `TaskCore`.

## Drills

1. 把当前 `TaskCLIProgram.run` 逻辑分成“入口、解析、执行、渲染”四类，各举一段代码。
2. 说明为什么 `usage` 文本属于 CLI 层，而不属于 `TaskCore`。
3. 设计一个你认为未来可能出现的新命令，并判断它最先会给解析层还是执行层带来压力。

## Project Handoff

命令组织一旦清楚，Part 3 的最后一步就可以落到完整组装：我们已经有 package、core tests、接缝意识和 CLI layering，下一章要把这些收束成真正的 `TaskCore + TaskCLI v1`，并明确它为什么是 Part 3 的终点、Part 4 的起点。
