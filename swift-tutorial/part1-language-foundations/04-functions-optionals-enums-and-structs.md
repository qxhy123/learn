# 第4章：函数、Optional、枚举与结构体

> 前三章已经把工具链、值、集合和控制流搭起来了，但程序仍然太像“会跑的脚本”。这一章的目标，是把它推进到真正可以支撑 `TaskCLI Lite v1` 的形状。

## 为什么这一章现在出现

如果你在上一章停下，程序会出现几个明显问题：

- 命令判断、数据遍历、输出渲染全挤在一段脚本里
- 任务只是字符串，无法自然表达完成状态
- 输入不确定时没有安全处理方式
- 命令种类只能靠魔法字符串横飞

这正是函数（functions）、Optional、枚举（enum）和结构体（struct）必须在现在出现的原因。它们不是独立知识点堆砌，而是四块正好能把弱脚本推进成最小项目的构件：

- function 负责把行为收起来
- Optional 负责表达“可能有，也可能没有”
- enum 负责把离散命令建成受限集合
- struct 负责把任务数据建成真正的值类型

## 先看看一个弱到不该继续扩展的版本

```swift
var tasks = ["read chapter 01", "write notes"]
let command = "done"
let title = "read chapter 01"

if command == "done" {
    for task in tasks {
        if task == title {
            print("done: \(task)")
        }
    }
}
```

这段代码的问题已经非常清楚：

- 任务没有完成状态字段，所谓“done”只是打印一句话
- 命令被表示成裸字符串，写错一个字符就失真
- “找任务”这件事如果找不到怎么办，没有明确表达
- 所有逻辑都堆在顶层，很难继续长

如果你再往这段脚本上补更多 `if` 和更多数组操作，程序会迅速失控。

## 用 `struct` 把任务从字符串升级为数据

Part 1 一个很重要的转折点，是承认任务不是“标题文本”，而是有多个字段的数据：

```swift
struct Task {
    let title: String
    var isDone: Bool
}
```

这几行代码的意义非常大：

- `title` 是任务身份里最直接的一部分
- `isDone` 表达任务状态变化
- 整个 `Task` 是一个值类型（value type），适合 Part 1 的小型数据建模

现在集合也随之升级：

```swift
var tasks = [
    Task(title: "read chapter 01", isDone: false),
    Task(title: "write notes", isDone: false)
]
```

从这里开始，`TaskCLI Lite` 才第一次真正拥有“任务列表”而不是“标题列表”。

## 用 function 把行为从脚本里提出来

顶层脚本过长时，最先该做的不是抽象成复杂架构，而是把明确行为提取为函数：

```swift
func render(tasks: [Task]) {
    for (index, task) in tasks.enumerated() {
        let status = task.isDone ? "[x]" : "[ ]"
        print("\(index + 1). \(status) \(task.title)")
    }
}
```

现在“如何渲染任务列表”被关进了一个有名字的单元。读者不需要盯着所有细节，也能知道这个函数做什么。对于 Part 1 来说，这种函数提炼已经足够重要，因为它把脚本从“所有事情同时发生”推进成“不同职责有了可指认的位置”。

## 用 Optional 表达“不一定存在”

CLI 很快就会遇到这种情况：用户要完成一个标题，但这个标题可能不在列表里。这里最适合引入 Optional：

```swift
func indexOfTask(named title: String, in tasks: [Task]) -> Int? {
    tasks.firstIndex { $0.title == title }
}
```

返回 `Int?` 的意思不是“也许是 Int，也许是别的类型”，而是“可能有一个索引，也可能没有”。这比返回 `-1` 之类的哨兵值（sentinel value）更清楚，也更符合 Swift 的表达习惯。

使用时你必须显式处理：

```swift
if let index = indexOfTask(named: "read chapter 01", in: tasks) {
    tasks[index].isDone = true
} else {
    print("Task not found")
}
```

这正是 Swift 想训练你的地方：不确定性不能被偷偷跳过，程序要明确承认它。

## 用 enum 让命令不再漂在字符串里

上一章用 `switch` 分支命令已经比纯脚本强很多，但命令本身仍然是裸字符串。更稳的做法，是把命令类型收成一个枚举：

```swift
enum Command {
    case list
    case add(String)
    case done(String)
}
```

这个设计立刻带来两个好处：

- 程序认得的命令集合是有限且显式的
- `add` 和 `done` 这类命令可以连同关联值（associated value）一起带上标题

即使 Part 1 还不做完整参数解析，这个 enum 心智也已经非常关键。它告诉你：程序输入不是一堆散乱字符串，而是可建模的命令形状。

## 让弱脚本进化为 `TaskCLI Lite` 的核心骨架

把这些构件组合起来，程序会变成这样：

```swift
struct Task {
    let title: String
    var isDone: Bool
}

enum Command {
    case list
    case add(String)
    case done(String)
}

func render(tasks: [Task]) -> String {
    let lines = tasks.enumerated().map { index, task in
        let status = task.isDone ? "[x]" : "[ ]"
        return "\(index + 1). \(status) \(task.title)"
    }

    return (["Today's tasks"] + lines).joined(separator: "\n")
}
```

这还不是完整成品，但已经和最终 starter package 的结构非常接近。更重要的是，你应该能解释为什么它比前三章的弱脚本强：

- 数据有了真实模型
- 行为开始被函数命名
- 不确定性通过 Optional 显式处理
- 命令有了受限形状

## Part 1 为什么停在这里刚好

你可能会想：“既然已经有 `enum` 和 `struct`，为什么不马上拆更多文件、做更完整架构？”因为本教程的节奏是先把语义和程序骨架做稳，再进入更强的工程边界。Part 1 的职责不是预支 Part 2，而是刚好把 CLI 的第一版立起来。下一章才会把这些材料收束成一个真正可 build、可 test、可 run 的 `TaskCLI Lite v1`。

## 双语关键词

- 函数：function
- Optional：可选值 / `Optional`
- 枚举：enum / enumeration
- 结构体：struct
- 关联值：associated value
- 值类型：value type
- 哨兵值：sentinel value

## 常见错误

### 1. 找不到值时返回魔法数字或空字符串

在 Swift 里，不确定性更适合用 Optional 表达。返回 `-1` 或 `""` 往往会让意图变得模糊。

### 2. 明明是有限命令集合，却仍然坚持用裸字符串到处比较

这会让命令空间越来越脆弱。哪怕 Part 1 的 enum 很简单，它也已经比魔法字符串稳得多。

### 3. 把 `struct` 想成“轻量 class”

Part 1 里更重要的不是语法像不像，而是你开始用值类型来表达任务数据。这个选择会影响后续对状态变化的理解。

### 4. 函数抽取一开始就过度工程化

这一章的目标不是设计完整架构，而是把明显独立的行为命名出来。能清楚表达职责，就已经比脚本堆叠强很多。

## English Recap

This chapter upgrades the program from a loose script to a real shape. `struct` models tasks, functions organize behavior, `Optional` represents uncertainty safely, and `enum` makes commands explicit instead of stringly typed. These four features form the core mental model needed for `TaskCLI Lite v1`.

## Drills

1. 定义一个 `Task` 结构体，包含 `title` 和 `isDone`，创建两个示例任务并打印它们。
2. 写一个函数 `findTask(named:in:) -> Int?`，返回匹配标题的索引；分别测试“找到”和“找不到”两种情况。
3. 定义一个 `Command` 枚举，包含 `list`、`add(String)`、`done(String)`，写一个 `switch` 输出不同说明文本。

## Project Handoff

到这里，Part 1 需要的最后一批核心语言材料已经齐了。下一章我们不再继续拆散概念，而是把前四章所有内容收束到一个真正的 SwiftPM 项目里，完成 `TaskCLI Lite v1`：可 build、可 test、可 run，并且明确作为 Part 1 的项目终点落地。
