# 第8章：协议、协议扩展与抽象边界

> 经过前两章，`TaskCLI Lite` 已经不再只是“几个函数拼出一个 CLI”。模型有了自己的行为，值语义也站稳了。接下来真正的问题变成：哪些地方该抽象，哪些地方还不该。

## 为什么这一章现在出现

很多程序员对协议（protocol）的第一印象是“Swift 版 interface”。这个理解不算错，但远远不够，因为它没有回答最关键的问题：**你为什么要在这里引入抽象边界（abstraction boundary）**。

如果抽象太少，代码会很快重新长成一个大团块：

- 解析命令、执行业务、渲染输出全挤在一起
- 测试时很难替换局部行为
- 未来一旦进入 package engineering，就找不到清晰接缝

如果抽象太多，又会走向另一种坏形状：

- 每个小类型都被硬套一个 protocol
- 还没有变化压力，就先造出一层“架构感”
- 代码读起来像工程模板，业务判断却越来越模糊

Part 2 在这里讲协议，不是为了让你学会“到处提接口”，而是为了训练一种工程判断：**协议应该包围真正可能变化、值得替换、值得测试隔离的边界**。

## 先看一个还算能跑、但边界已经开始发糊的版本

假设我们把 `TaskCLI Lite` 继续顺着 Part 1 的自然惯性往下写，很容易长成下面这样：

```swift
struct TaskCLIProgram {
    var tasks: [Task]

    mutating func run(arguments: [String]) -> String {
        guard let command = arguments.first else {
            return "Usage: task-cli-lite <command>"
        }

        switch command {
        case "list":
            return tasks.map(\.cliLine).joined(separator: "\n")
        case "add":
            let title = arguments.dropFirst().joined(separator: " ")
            guard let task = Task(id: tasks.count + 1, title: title) else {
                return "Title cannot be empty"
            }
            tasks.append(task)
            return "Added: \(task.title)"
        default:
            return "Unknown command"
        }
    }
}
```

这段代码的问题不是“写错了”，而是边界已经开始混在一起：

- 参数解析（parsing）
- 模型更新（mutation）
- 输出渲染（rendering）

都在同一个 `run` 里。

短期它仍然可读，长期就会越来越难变。你一旦想换输出格式、想独立测试解析逻辑、想让命令执行层不要直接依赖字符串渲染，马上就会感到阻力。

## 协议不是为了“显得抽象”，而是为了给变化点立边界

最值得先抽象出来的，不是 `Task` 本身，而是那些已经显露出变化压力的职责。

在当前教程阶段，至少有两个边界已经很明显：

- 命令怎么从参数变成领域命令
- 任务列表怎么被渲染成 CLI 输出

先把渲染边界抽出来：

```swift
protocol TaskRendering {
    func render(tasks: [Task]) -> String
}

struct PlainTextTaskRenderer: TaskRendering {
    func render(tasks: [Task]) -> String {
        let lines = tasks.map(\.cliLine)
        return (["Today's tasks"] + lines).joined(separator: "\n")
    }
}
```

现在，“如何把任务显示给用户”已经不再是某个大函数里的顺手拼接，而是一个命名清楚的能力边界。

再把命令解析边界抽出来：

```swift
enum TaskCommand {
    case list
    case add(title: String)
    case done(id: Int)
}

protocol TaskCommandParsing {
    func parse(arguments: [String]) -> TaskCommand?
}

struct TaskCommandParser: TaskCommandParsing {
    func parse(arguments: [String]) -> TaskCommand? {
        guard let command = arguments.first else { return nil }

        switch command {
        case "list":
            return .list
        case "add":
            let title = arguments.dropFirst().joined(separator: " ")
            return .add(title: title)
        case "done":
            guard
                let rawID = arguments.dropFirst().first,
                let id = Int(rawID)
            else { return nil }
            return .done(id: id)
        default:
            return nil
        }
    }
}
```

这里最重要的，不是多了两个 protocol 名字，而是系统开始出现可指认的接缝：

- parser 负责把外部文本变成内部命令值
- renderer 负责把内部状态变成外部文本

中间的模型更新逻辑，终于有机会站在中间而不是被夹碎在字符串处理里。

## 为什么 `Task` 本身通常不需要先抽成 protocol

这是协议章节里最常见的过度反应。

很多读者学到 protocol 后会立刻想写：

```swift
protocol TaskProtocol {
    var id: Int { get }
    var title: String { get }
    var isDone: Bool { get }
}
```

然后再让 `Task` 去遵循它。

这通常没有增加任何真实价值。因为当前阶段并没有多个“任务模型实现”在竞争，也没有哪段代码真的只需要“像任务那样的东西”。你只是把一个具体稳定的领域类型，套上了一层没有变化压力的抽象壳。

这是从 Java / C# / TypeScript 生态迁移过来时非常容易出现的习惯：为了“面向接口编程”而接口化一切。Swift 更讲究抽象的密度。没有变化点，就不要为了形式感先加 protocol。

更准确的经验法则是：

- 具体且稳定的领域值，先保持 concrete type
- 变化频率高、替换价值高、测试隔离价值高的边界，再考虑 protocol

对 `TaskCLI Lite` 来说，`Task` 目前属于前者，`parser` / `renderer` 更接近后者。

## 协议扩展（protocol extension）怎么用，才不是“默认实现垃圾桶”

Swift 协议的一个重要特色，是协议扩展（protocol extension）。这很强大，也很容易被滥用。

先看一个合适的例子。对于所有任务渲染器来说，状态符号逻辑都一样，可以给出默认辅助实现：

```swift
protocol TaskRendering {
    func render(tasks: [Task]) -> String
}

extension TaskRendering {
    func statusSymbol(for task: Task) -> String {
        task.isDone ? "[x]" : "[ ]"
    }
}

struct PlainTextTaskRenderer: TaskRendering {
    func render(tasks: [Task]) -> String {
        let lines = tasks.map { task in
            "\(statusSymbol(for: task)) \(task.title)"
        }
        return (["Today's tasks"] + lines).joined(separator: "\n")
    }
}
```

这里的协议扩展有两个特点：

- 它提供的是和协议职责强相关的共享辅助逻辑
- 它没有偷偷扩大协议的责任边界

这和一种很糟糕的用法不同：把大量和协议主题不直接相关的逻辑，统统塞进 extension，当成隐藏式工具箱。那样代码表面上“很抽象”，实际却更难找责任归属。

你应该把 protocol extension 想成：**给这一类能力提供合理默认值和共享小块，而不是给系统找一个新的杂物间**。

## 抽象边界到底应该切在哪里

这是本章真正重要的判断题。

在 `TaskCLI Lite` 当前阶段，一个抽象边界值得出现，通常要满足至少一个条件：

- 它包住了外部表示与内部模型之间的转换
- 它隔离了未来很可能变化的实现细节
- 它明显改善了测试入口
- 它为 Part 3 的 package 边界提供自然接缝

例如：

- `TaskCommandParsing` 隔离了 CLI 文本输入
- `TaskRendering` 隔离了输出格式

而下面这些就大概率还不值得：

- `TaskProtocol`
- `TaskListProtocol`
- `TaskManagerProtocol`

因为这些名字听起来“很工程”，但并没有包住一个真实不稳定边界。

## 一个更像工程代码、但还没过度设计的版本

把协议边界接到当前程序里，可以得到一个更强的中间形态：

```swift
struct TaskCLIEngine {
    var tasks: [Task]
    let parser: TaskCommandParser
    let renderer: PlainTextTaskRenderer

    mutating func run(arguments: [String]) -> String {
        guard let command = parser.parse(arguments: arguments) else {
            return "Invalid command"
        }

        switch command {
        case .list:
            return renderer.render(tasks: tasks)
        case .add(let title):
            guard let task = Task(id: tasks.count + 1, title: title) else {
                return "Title cannot be empty"
            }
            tasks.append(task)
            return "Added: \(task.title)"
        case .done(let id):
            guard let index = tasks.firstIndex(where: { $0.id == id }) else {
                return "Task not found"
            }
            tasks[index].markDone()
            return "Completed: \(tasks[index].title)"
        }
    }
}
```

注意，这里我仍然先用具体类型 `TaskCommandParser` 和 `PlainTextTaskRenderer` 挂上去，而不是急着把整个 `TaskCLIEngine` 改成一堆泛型或 `any Protocol` 存储。原因很简单：Part 2 此刻的重点是先看清边界，而不是提前把所有使用方式都泛化。

这正是教程节奏里很重要的一点：**先把边界命名清楚，再决定边界如何注入、如何泛化、如何拆 package**。这也是为什么 Part 3 才正式进入 package engineering。

## 协议和继承的思路差别

对于来自传统 OOP 训练的人，这里还有一个需要刻意重建的心智：Swift 协议不是为了先搭一个继承树替代品。

在很多旧习惯里，人们会先想：

- 有没有一个基类 `BaseRenderer`
- `TaskRenderer` 和 `ProjectRenderer` 是否应该继承它

Swift 更自然的方向通常是：

- 先定义一个能力边界 `TaskRendering`
- 让具体类型去遵循这个能力
- 如果有共享默认行为，用协议扩展补上

这会让抽象更贴近“能做什么”，而不是“属于哪棵类树”。

对当前 CLI 教程线来说，这非常重要，因为我们的重点不是造一个类型族谱，而是让命令解析、领域操作、输出渲染这些责任彼此分开。

## 这一步如何为 Part 3 做准备

Part 2 还不拆包（package），这是刻意的。但如果你在 Part 2 不先建立协议与边界判断，到了 Part 3 你会很容易把 package 拆成“目录分组”，而不是语义分组。

真正好的 package 边界，通常来自之前已经存在的抽象接缝。例如：

- 解析层和领域层之间的接缝
- 渲染层和领域层之间的接缝
- 未来存储层和领域层之间的接缝

所以本章的价值不是“学会 protocol 语法”，而是开始形成模块前的边界意识。

## 双语关键词

- 协议：protocol
- 协议扩展：protocol extension
- 抽象边界：abstraction boundary
- 协议遵循：protocol conformance
- 默认实现：default implementation
- 解析：parsing
- 渲染：rendering
- 具体类型：concrete type

## 常见错误

### 1. 为了“面向接口编程”而给每个具体类型都套一层 protocol

没有变化压力的抽象通常只是噪音。`Task` 这样的稳定领域值，当前阶段完全可以保持 concrete。

### 2. 把协议当成继承树替代品来设计

Swift 协议更适合表达能力边界，而不是先搭层级结构。

### 3. 协议扩展写成默认实现垃圾桶

扩展里应该放和该协议职责强相关的共享逻辑，而不是把无处安放的代码统统塞进去。

### 4. 还没看清边界，就先把整个系统改成泛型注入

抽象顺序很重要。先命名边界，再决定注入方式；否则你只是在扩大复杂度。

### 5. 把“拆成多个类型”误以为“边界就已经清楚了”

真正的边界不是文件数量，而是职责是否清晰、替换点是否明确、依赖方向是否稳定。

## English Recap

This chapter introduces protocols as boundary tools, not abstraction theater. In `TaskCLI Lite`, parsing and rendering are real variation points, so they deserve protocols; `Task` itself usually does not. Protocol extensions are useful when they provide small, responsibility-aligned defaults, not when they become hidden utility dumps.

## Drills

1. 把当前练习里的输出拼接逻辑抽成一个 `TaskRendering` 协议和一个 `PlainTextTaskRenderer`。
2. 写一个最小 `TaskCommandParsing`，至少支持 `list` 和 `add <title>`。
3. 检查你写过的某个 protocol：它到底包住了真实变化点，还是只是给具体类型又套了一层名字？

## Project Handoff

到这里，`TaskCLI Lite` 已经开始拥有真正的抽象边界：输入如何进来，输出如何出去，领域模型如何站在中间。下一章会继续把这条线推深，讨论泛型（generics）、关联类型（associated types）和类型驱动 API 设计（type-driven API design）：当边界已经出现时，如何让 API 用类型表达关系，而不是靠复制、`Any` 或文档约定硬撑。
