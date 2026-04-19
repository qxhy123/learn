# Part 1 综合实验：把 `TaskCLI Lite v1` 真正立起来

## 对应部分与项目阶段

- 对应部分：Part 1 `part1-language-foundations`
- 对应项目阶段：`TaskCLI Lite v1`
- 关联章节：第 1 章到第 5 章

这份 lab 不是给你再做一遍章节 drills，而是检查你是否已经能把 Part 1 的语言基础真正压进一个最小但完整的程序里。做完它以后，你应该能解释：为什么 `TaskCLI Lite v1` 只做 `list`、`add`、`done` 就够了，为什么这里先用值、控制流、函数、`Optional`、`enum`、`struct`，而不是先追求“架构感”。

## 使用方式

建议你直接站在 `swift-tutorial/projects/task-cli-lite/starter` 的语境里完成这份实验，但不要把它理解成“照着 starter 抄”。更好的方式是：

1. 先用自己的话写出当前 `TaskCLI Lite v1` 的输入、状态和输出。
2. 再做综合练习，逼自己把基础语义重新接成一条线。
3. 最后做 debugging / refactoring / challenge，暴露你现在最容易犯的 Part 1 级错误。

## Integrated Exercises

### 综合练习 1：从命令行参数重建最小任务流

目标：不用额外框架，只靠 Part 1 能力，重建一个最小 CLI 流程。

要求：

- 支持 `list`、`add <title>`、`done <title>` 三个命令。
- 使用 `enum Command` 表达命令，而不是在多个 `if argument == ...` 里散着判断。
- 使用 `struct Task` 表达任务，至少包含 `title` 和 `isDone`。
- 对缺参数或未知命令给出清楚的 usage 文本。
- 保持数据为内存态；这里故意不引入文件持久化。

交付物：

- 一段你自己写的 `parseCommand(_:)`。
- 一段 `run(command:tasks:)` 或等价函数，把“解释命令”和“执行行为”分开。
- 一段简短说明：为什么这个阶段的 `TaskCLI Lite v1` 不该先拆成一堆类型和文件。

### 综合练习 2：把弱字符串状态升级为显式数据

给自己一个弱起点：

```swift
var tasks = ["write notes", "read chapter", "[done] fix bug"]
```

把它升级为更强版本：

- 用 `struct Task` 替代字符串拼接协议。
- 用布尔值或 `enum TaskStatus` 表达状态，不要把状态藏在标题前缀里。
- 用单独函数负责渲染 CLI 输出，例如 `renderTaskList(_:)`。
- 解释为什么这一步虽然还简单，但已经在为 Part 2 的建模判断铺路。

### 综合练习 3：把 Optional 当成语义，而不是补丁

设计两个函数：

- `findTask(named:in:) -> Task?`
- `markDone(named:in:) -> Bool` 或 `-> Task?`

然后回答：

- 为什么“找不到任务”更像 `Optional`，而不是立刻 `fatalError`？
- 为什么 `Optional` 只适合表达“可能没有”，还不适合承载更复杂失败？
- 如果用户输入空标题，你会在 Part 1 暂时怎么处理，为什么？

## Debugging Tasks

### 调试任务 1：`done` 命令总是改错任务

观察下面的弱代码：

```swift
for task in tasks {
    if task.title == title {
        task.isDone = true
    }
}
```

你需要指出：

- 这里为什么改不到真正存回数组里的值。
- 这个 bug 和 `struct` 的值语义有什么关系。
- 在 Part 1 语境里，最直接、最可讲清楚的修法是什么。

### 调试任务 2：命令解析吞掉合法输入

观察下面的弱代码：

```swift
let parts = CommandLine.arguments
let command = parts[1]
let title = parts[2]
```

你需要修到至少满足：

- 没有参数时不崩溃。
- `add` 缺标题时给出 usage。
- `add "read chapter 01"` 这种带空格标题仍然工作。

重点不是“写出最花的解析器”，而是证明你已经会用 `Array`、索引检查、`dropFirst()`、`joined(separator:)` 这类 Part 1 能力把程序写稳。

## Refactoring / Design Tasks

### 设计任务 1：把“能跑的大块脚本”切成可解释的骨架

假设你当前所有逻辑都挤在 `main.swift`。请把它收束成至少四个责任点：

- `Task`
- `Command`
- `parseCommand(_:)`
- `run(command:tasks:)` / `render...`

写一段 150 字左右说明，回答：为什么这在 Part 1 已经算改进，但还不等于进入 Part 3 的工程化。

### 设计任务 2：给 usage 文本建立最小一致性

为这三个场景统一输出风格：

- 未知命令
- 参数缺失
- 当前支持命令列表

约束：

- 不要做本地化系统。
- 不要建复杂 formatter 层。
- 只追求“最小一致、容易扩展、阅读成本低”。

## Challenge Tasks

### 挑战 1：新增 `undo <title>`

在不引入类、协议、持久化、外部依赖的前提下，为 `TaskCLI Lite v1` 增加 `undo`，把已完成任务恢复为未完成。

你需要说明：

- 这个功能为什么仍然属于 `TaskCLI Lite v1` 可以承受的范围。
- 它是否会迫使你重新设计 `Command` 和状态表达。
- 你的实现有没有开始暴露 Part 2 才该处理的建模压力。

### 挑战 2：新增 `stats`

输出：

- 总任务数
- 已完成数
- 未完成数

不要为了这一个命令提前引入“分析引擎”。这里真正要练的是：

- `Array` 遍历与聚合
- 小函数分工
- 保持 CLI 输出清楚

## 退出标准

完成这份 lab 时，你至少应能明确说出：

- 为什么 Part 1 的项目阶段名是 `TaskCLI Lite v1`，而不是“迷你架构系统”。
- 为什么 `struct`、`enum`、`Optional`、函数拆分已经足够支撑这个阶段。
- 为什么现在最该修的是语义清晰度与最小可运行闭环，而不是提前抽象。

## 复盘问题

1. 你写的代码里，哪一处最能说明你已经不再把 Swift 当“带分号的脚本语言”？
2. 如果要把当前结果交给 Part 2，你最担心哪一段模型语义还不够稳？
3. 你有没有把 `Optional`、数组和控制流用成“能跑就行”的补丁？如果有，下一轮会怎么收紧？
