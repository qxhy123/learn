# 第7章：类 vs 结构体，以及值语义 vs 引用语义

> 上一章里，`Task` 已经开始有属性、方法和初始化器了。现在问题不再是“模型能不能写出来”，而是“这个模型到底应该以什么语义存在”。

## 为什么这一章现在出现

很多程序员一进入 Swift，会先问一个熟悉的问题：“这个东西是 class 还是 struct？”如果你只是把它当作“语法选型题”，往往会做出看似合理、实际很脆的决定。

Part 2 在这里讲类（class）与结构体（struct），不是为了完成一张语法表，而是因为 `TaskCLI Lite` 已经进入了一个必须认真回答的阶段：

- 一条任务被复制后，修改副本是否应该影响原值？
- 一组任务列表被传来传去时，谁在共享，谁在独占？
- 某个类型代表的是“数据快照”，还是“有身份、有生命周期的对象”？

这背后真正的主题，是值语义（value semantics）和引用语义（reference semantics）。

如果你来自 Python、JavaScript、Java、C#、Ruby 这类“对象默认引用”的世界，很容易把 `class` 当成本能选项。Swift 刻意没有这么设计。它把值类型放在语言中心位置，不是为了显得不同，而是因为很多业务模型天然更适合值语义：复制清楚、修改显式、测试直观、共享更少。

`TaskCLI Lite` 正是这种场景。任务和任务列表，首先是领域值（domain values），不是带身份管理器的小对象。

## 一个来自旧习惯的弱起点：把任务先写成 `class`

先看很多人会下意识写出的版本：

```swift
final class Task {
    let id: Int
    var title: String
    var isDone: Bool

    init(id: Int, title: String, isDone: Bool = false) {
        self.id = id
        self.title = title
        self.isDone = isDone
    }
}
```

如果你来自 Java 或 Kotlin，这看起来非常自然：任务有字段、有构造器、有可变状态，于是就写成 class。

但问题会在“复制”和“传递”时立刻出现：

```swift
let original = Task(id: 1, title: "write chapter 06")
let alias = original

alias.isDone = true

print(original.isDone) // true
```

`alias` 并不是一个独立副本，它只是同一个实例的另一个引用。对很多业务对象来说，这未必是错；但对 `TaskCLI Lite` 当前阶段的任务记录来说，这通常不是你想要的默认行为。

更隐蔽的问题出现在数组里：

```swift
let first = Task(id: 1, title: "read docs")
var today = [first]
var snapshot = today

snapshot[0].isDone = true

print(today[0].isDone) // true
```

很多初学者会以为 `snapshot = today` 已经复制了整个列表，所以修改 `snapshot` 不会影响 `today`。数组本身确实是值类型，但数组里面装的是 class 引用，于是你得到的是“新盒子里装着旧对象地址”。这就是 Swift 学习里最常见的误判之一。

## 任务记录更适合 `struct`，因为它首先是值

把同样的模型写成结构体：

```swift
struct Task {
    let id: Int
    private(set) var title: String
    private(set) var isDone: Bool

    init(id: Int, title: String, isDone: Bool = false) {
        self.id = id
        self.title = title
        self.isDone = isDone
    }

    mutating func markDone() {
        isDone = true
    }
}
```

现在再看复制行为：

```swift
var original = Task(id: 1, title: "write chapter 06")
var copy = original

copy.markDone()

print(original.isDone) // false
print(copy.isDone)     // true
```

这就是值语义的核心体验：**复制之后，各走各的状态演化路径**。

对于 `TaskCLI Lite` 当前阶段，这种行为特别合理，因为一条任务记录通常表达的是“某个时刻的一份状态”。我们在做 CLI 建模、测试、命令处理时，更希望每次修改都清楚落在某个值上，而不是通过共享引用悄悄蔓延。

## 值语义不是“永远复制整份内存”

一提到值类型，很多有 C++ 或 Java 背景的读者会立刻担心性能：“那岂不是每次传数组、传结构体都复制一大块？”

这是理解 Swift 时必须及时纠正的旧直觉。Swift 的很多标准库值类型，包括 `Array`、`Dictionary`、`String`，都带有 copy-on-write（写时复制）策略。你可以先把它理解成：

- 逻辑语义上，它们像值一样工作
- 实现层面，Swift 会尽量延迟真正的数据复制，直到某一方开始修改

这意味着你不应该因为“怕拷贝”就把所有模型都改写成 class。先问语义，再问成本。语义对了，再看性能证据；不要拿可能并不存在的复制成本，去换来真实存在的共享可变状态问题。

在 `TaskCLI Lite` 这种教学型 CLI 里，值语义的可解释性收益远大于潜在复制开销。

## `mutating` 让修改变得显式，这正是优点

对于习惯引用对象的人来说，`mutating` 有时会显得麻烦：为什么我改个字段还要专门声明？

正因为 Swift 想把“我现在正在改变一个值”这件事说清楚。

看一个列表级别的例子：

```swift
struct TaskList {
    private(set) var tasks: [Task]

    mutating func markDone(id: Int) {
        guard let index = tasks.firstIndex(where: { $0.id == id }) else { return }
        tasks[index].markDone()
    }
}
```

这里有两层变化都被明确暴露了：

- `TaskList.markDone` 会改 `TaskList`
- `tasks[index].markDone()` 会改数组中那个 `Task` 值

这种显式感，在小脚本里也许看不出价值；一旦代码变多、测试变多、模块边界变多，它会显著降低“状态到底在哪改掉了”的排查成本。

## 值语义如何改善 `TaskCLI Lite` 的测试直觉

Part 1 里我们已经开始写 XCTest。进入 Part 2 后，值语义会让测试边界更干净。

例如：

```swift
var list = TaskList(tasks: [
    Task(id: 1, title: "read chapter 06"),
    Task(id: 2, title: "write notes")
])

var modified = list
modified.markDone(id: 1)
```

这时你可以非常自然地同时断言原值和修改后值：

- `list` 仍然保持初始状态
- `modified` 代表这次操作后的新状态

这在测试里非常顺手，因为它贴近一种“输入状态 -> 变换 -> 输出状态”的函数式心智。即使你不是函数式编程背景，值语义也会让 CLI 这类程序更容易推理。

## 那什么时候该用 `class`

说到这里，很容易把话走成极端：好像 Swift 里 class 就不该用。不是这样。

更稳的判断是：当一个类型的核心意义在于“共享身份、共享生命周期、共享可变状态”时，class 往往更合适。

在我们的教程主线上，以下这类东西更可能属于 class：

- 需要被多个地方共同持有并持续写入的输出缓冲器
- 有明确生命周期的协调对象（coordinator）
- 包住系统资源的对象，例如文件句柄、进程句柄、观察者中心

举一个简单、贴近 CLI 的例子：

```swift
final class OutputBuffer {
    private(set) var lines: [String] = []

    func write(_ line: String) {
        lines.append(line)
    }
}
```

如果两个不同组件都拿到同一个 `OutputBuffer`，我们通常就是希望它们写进同一份缓冲区，而不是各写各的副本。这时共享引用正是设计目标，不是副作用。

关键不在于“class 高级、struct 轻量”，而在于这个类型表达的是值，还是身份。

## 不要把“有方法”误判成“应该是 class”

这是另一个跨语言迁移里非常常见的误区。很多人会想：

- 有字段，可能是 struct
- 一旦有方法，好像就更像 class

Swift 不是这么分的。`struct` 完全可以有方法、计算属性、初始化器、协议遵循（protocol conformance），也完全可以承载丰富行为。能不能写方法，从来不是 class 的专利。

对于 `Task` 这种领域模型，恰恰是“有行为的 struct”最适合当前阶段。因为这些行为是在维护一个值的局部规则，而不是在管理一个共享对象图。

## 一个更强的阶段性判断

现在把本章的核心判断压成一句工程上有用的话：

`TaskCLI Lite` 当前的任务、任务列表、命令值，更适合默认建成 `struct`，因为它们表达的是可复制、可比较、可预测演化的领域值；只有当某个类型需要稳定身份和共享可变状态时，才应该认真考虑 `class`。

这句话比“Swift 推荐 struct”更有用，因为它告诉你判断依据，而不是告诉你一条教条。

## 值语义会怎样影响后面的设计

这一章不是孤立话题。它会直接影响 Part 2 后半段和 Part 3：

- 讲协议时，你会更清楚哪些抽象是在包围值模型，哪些抽象是在包围资源边界
- 讲泛型时，你会更自然地把 API 设计成“接收和返回值”，而不是让调用方猜共享状态
- 到 Part 3 做 package engineering 时，值语义会让模块边界和测试夹层更容易稳定下来

也就是说，`struct` / `class` 不是早学完就能忘的基础题，它会一路影响你之后对 Swift 工程的判断。

## 双语关键词

- 类：class
- 结构体：struct
- 值语义：value semantics
- 引用语义：reference semantics
- 身份：identity
- 共享可变状态：shared mutable state
- 写时复制：copy-on-write
- 可变方法：`mutating method`

## 常见错误

### 1. 因为别的语言默认对象是引用，就把 Swift 模型也先写成 class

这不是“经验迁移”，而是把旧默认值带进了另一门语言。先看语义需要，再选 `struct` 还是 `class`。

### 2. 以为数组复制了，里面的 class 实例也会自动深拷贝

数组复制的是容器值；如果元素是引用类型，你复制的是引用集合，不是对象内容本身。

### 3. 把“有方法”当成 class 的理由

`struct` 完全可以有丰富行为。方法属于“责任归属”问题，不属于“引用还是值”问题。

### 4. 因为担心性能，就过早放弃值语义

Swift 标准库大量使用 copy-on-write。不要在没有证据的前提下，用假想的复制成本去换共享状态复杂度。

### 5. 把 class 和 struct 的差别理解成“一个重、一个轻”

真正重要的差别不是重量感，而是语义：复制之后是否独立、状态变化是否共享、类型是否靠身份存在。

## English Recap

This chapter reframes `class` vs `struct` as a semantics decision, not a syntax preference. `Task` and `TaskList` fit value semantics because they represent domain state that should copy predictably and mutate explicitly. Use `class` only when shared identity and shared mutable state are the actual design goal.

## Drills

1. 分别用 `class Task` 和 `struct Task` 写一个最小例子，复制后修改副本，观察原值是否被影响。
2. 给 `TaskList` 写一个 `markDone(id:)`，并明确标出为什么它必须是 `mutating`。
3. 想一想你当前 CLI 里有没有某个类型真的需要共享身份；如果有，写一句话说明“为什么它不是普通值”。

## Project Handoff

现在我们已经明确了 `TaskCLI Lite` 当前模型的语义地基：任务和任务列表首先是值，而不是共享对象。下一章会沿着这块地基继续向上搭，讨论协议（protocol）、协议扩展（protocol extension）和抽象边界（abstraction boundary）：既然值模型站稳了，哪些变化点该抽象，哪些变化点还不该提早抽象？
