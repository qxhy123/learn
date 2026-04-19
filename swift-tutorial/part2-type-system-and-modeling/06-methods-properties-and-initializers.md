# 第6章：方法、属性与初始化器

> Part 1 结束时，`TaskCLI Lite v1` 已经能跑，但它仍然带着明显的“脚本后劲”：数据和行为刚刚开始分家，却还没有真正形成稳定模型。Part 2 的第一章，就从这里继续。

## 为什么这一章现在出现

上一部分里，我们已经有了 `struct Task`、`enum Command`、一些函数和一个最小可运行的 CLI。问题是，那套代码虽然足以支撑入门项目，却还不够支撑更长期的建模判断。

你很快会遇到这些压力：

- 任务标题到底应该在哪里做规范化（normalization）？
- “完成任务”这种动作，应该散落在外部函数里，还是收进类型自己负责？
- 一条任务的显示文本，到底是调用方随手拼接，还是模型提供稳定出口？
- 创建对象时怎样保证“这个值一出生就是合法的”？

如果这些问题不在现在处理，后面讲类与结构体、协议、泛型、错误建模时，你会发现所有高级话题都在空中打转。因为真正的类型系统训练，第一步不是“会写更多类型”，而是让一个类型自己承担该承担的状态、不变量（invariant）和行为。

对于来自 Python、JavaScript、Java、Go、Kotlin 的程序员来说，这一步尤其重要。很多人刚进入 Swift 时，会把 `struct` 当成“装字段的小袋子”，把行为继续放在外部函数、管理器（manager）或顶层流程里。那样短期很省事，长期却会让模型越来越空、逻辑越来越散。

## 从一个还能跑、但已经开始发虚的版本出发

先看一个典型弱起点：

```swift
struct Task {
    let id: Int
    var title: String
    var isDone: Bool
}

func renderLine(for task: Task) -> String {
    let mark = task.isDone ? "[x]" : "[ ]"
    return "\(mark) \(task.title)"
}

func markTaskDone(_ task: inout Task) {
    task.isDone = true
}
```

这段代码比 Part 1 前期的脚本当然强很多，但它仍然暴露出三个问题：

- `Task` 自己不知道怎样把自己显示成 CLI 文本
- `Task` 自己也不保护自己的状态变化，谁都可以随手改 `title` 或 `isDone`
- 创建 `Task` 时没有最小合法性检查，空标题、全空白标题都能直接塞进去

这就是一个“数据袋 + 外部工具函数”的形状。很多语言里，这种写法能拖很久，因为类型本身并不强迫你做更稳的设计；Swift 虽然也允许你这么写，但它真正擅长的，是把模型本身设计得更自解释。

## 先让属性（properties）承担状态，而不是只当字段列表

Swift 里的属性（property）不只是“成员变量”的另一种叫法。它有两个更重要的角色：

- 表达类型真正持有的状态
- 表达从状态导出的稳定视图

先把 `Task` 升级成一个稍微像样的模型：

```swift
enum TaskStatus {
    case pending
    case done
}

struct Task {
    let id: Int
    private(set) var title: String
    private(set) var status: TaskStatus

    var isDone: Bool {
        status == .done
    }

    var cliLine: String {
        let mark = isDone ? "[x]" : "[ ]"
        return "\(mark) \(title)"
    }
}
```

这里出现了三个重要升级。

第一，`isDone` 不再是独立存储字段，而是一个计算属性（computed property）。这意味着“是否完成”这个判断来自 `status`，而不是系统里并排躺着两份可能失去同步的信息。你应该开始习惯这种想法：如果一个值可以从更基础的状态推导出来，就优先把它建成 derived state，而不是额外存一份。

第二，`cliLine` 把 CLI 展示逻辑收进了模型最靠近的数据处。不是所有显示逻辑都应该进模型，但像“这条任务在命令行里如何表示”这种和领域意义强相关、变化频率不高的表示，是可以收进去的。它比让调用方到处复制 `"[x]"` / `"[ ]"` 拼接规则稳得多。

第三，`private(set)` 把“谁可以读”和“谁可以写”分开了。外部代码可以读 `title` 和 `status`，但不能绕过模型直接修改。这种写法非常适合 Part 2，因为它在不引入复杂封装层的前提下，已经开始保护状态边界。

如果你来自 Java 或 C#，可能会下意识把这一步理解成“Swift 版 getter”。这还不够准确。更稳的理解是：Swift 属性是在用声明式方式表达“哪些状态是源头，哪些状态是推导结果，哪些写入口需要被收窄”。

## 存储属性（stored property）与计算属性（computed property）怎么分工

一个常见误区，是刚学到计算属性就什么都想写成计算属性，或者反过来，什么都塞成存储属性。判断标准其实很简单：

- 真正需要被保存、复制、比较的状态，用 stored property
- 可以从已有状态稳定推导出来的视图，用 computed property

以 `TaskCLI Lite` 为例：

- `id`、`title`、`status` 是 stored property
- `isDone`、`cliLine`、`statusSymbol` 更适合作为 computed property

比如你还可以再拆一层：

```swift
struct Task {
    let id: Int
    private(set) var title: String
    private(set) var status: TaskStatus

    var statusSymbol: String {
        isDone ? "[x]" : "[ ]"
    }

    var isDone: Bool {
        status == .done
    }

    var cliLine: String {
        "\(statusSymbol) \(title)"
    }
}
```

这里的关键不是“写法更花”，而是模型里的语义开始成形了。`statusSymbol` 不是随手的字符串，而是“状态如何投影到 CLI 视图”的一个命名决定。命名一旦稳定，代码阅读成本就会明显下降。

## 方法（methods）要收的不是“所有逻辑”，而是模型自己的行为

接下来处理更大的问题：行为到底放哪里。

一个来自其他语言的常见旧习惯是，要么把行为全塞进外部工具函数，要么过度反弹，把一切都塞进方法。Swift 更稳的做法，是把“直接维护这个类型状态和规则”的行为收进方法，把更高层的流程编排留给外部。

对于 `Task`，下面这些行为显然属于模型自己：

- 改标题
- 标记完成
- 重新打开任务

可以这样写：

```swift
struct Task {
    let id: Int
    private(set) var title: String
    private(set) var status: TaskStatus

    var isDone: Bool {
        status == .done
    }

    var cliLine: String {
        let mark = isDone ? "[x]" : "[ ]"
        return "\(mark) \(title)"
    }

    mutating func rename(to newTitle: String) -> Bool {
        let normalized = newTitle.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalized.isEmpty else { return false }
        title = normalized
        return true
    }

    mutating func markDone() {
        status = .done
    }

    mutating func reopen() {
        status = .pending
    }
}
```

这里的 `mutating` 非常关键。它提醒你：`struct` 是值类型（value type），修改它的方法必须显式声明“我要改这个值本身”。这和很多以引用语义为默认的语言很不一样。在那些语言里，修改对象经常是默认行为；在 Swift 里，值类型的可变性被放到了台面上。

这会训练出一种更稳的建模直觉：状态变化不是偷偷发生的，而是被类型签名清楚暴露出来。

## 初始化器（initializer）是“建一个值”的地方，不是补洞的地方

接下来进入本章最容易被低估的部分：初始化器（initializer / `init`）。

很多程序员刚写 Swift 时，会直接依赖成员逐个赋值，或者把 `init` 当成“把外面传进来的值抄进来”的样板代码。真正更重要的问题其实是：**一个值在出生那一刻，能不能就满足最小合法性**。

如果 `Task` 的标题允许是空白，那么整个后续系统都要为这个脏状态买单。更稳的做法，是在 `init` 里把问题挡住。

```swift
struct Task {
    let id: Int
    private(set) var title: String
    private(set) var status: TaskStatus

    init?(id: Int, title: String, status: TaskStatus = .pending) {
        let normalizedTitle = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalizedTitle.isEmpty else { return nil }

        self.id = id
        self.title = normalizedTitle
        self.status = status
    }
}
```

这里用了可失败初始化器（failable initializer / `init?`）。它表达的意思很清楚：不是所有输入都能构造出一个合法 `Task`。

这和 Python 里“先建出来再说”、和 JavaScript 里“对象字段先空着”、和很多业务代码里“构造函数里随便收、后面再验证”的习惯都不一样。Swift 的倾向是：如果一个值从出生起就不对，那就不要假装它已经存在。

当然，本章还不打算把所有失败信息建模到最细。`init?` 已经足够说明一个关键判断：初始化器不是礼貌性步骤，它是第一道模型边界。

## 从成员逐个传参，进化到带默认值和命名意图的初始化

除了合法性检查，初始化器还负责另一件事：让“创建一个值”的意图更清楚。

例如，对 CLI 任务来说，大多数新任务初始状态都是 `.pending`。如果每次创建都显式传一遍状态，噪音就会很大。上面的默认参数：

```swift
init?(id: Int, title: String, status: TaskStatus = .pending)
```

已经在做一种很典型的 Swift 设计：把常见路径做短，把非常见路径保留为显式参数。

你还可以继续往前一步，把规范化和构造放到一个更完整的模型中：

```swift
struct TaskList {
    private(set) var tasks: [Task]

    var pendingCount: Int {
        tasks.filter { !$0.isDone }.count
    }

    var completedCount: Int {
        tasks.count - pendingCount
    }

    init(tasks: [Task] = []) {
        self.tasks = tasks
    }

    mutating func add(_ task: Task) {
        tasks.append(task)
    }
}
```

现在你应该能看出 Part 2 的方向了：我们不是在学一些分散的 Swift 语法，而是在让模型开始自己承担局部规则。`TaskList` 负责列表级别状态，`Task` 负责单任务状态，这比“一个大函数处理所有细节”更接近工程代码。

## 一个更强的阶段性版本

把属性、方法、初始化器合起来，`TaskCLI Lite` 的模型层已经可以长成这样：

```swift
enum TaskStatus {
    case pending
    case done
}

struct Task {
    let id: Int
    private(set) var title: String
    private(set) var status: TaskStatus

    var isDone: Bool {
        status == .done
    }

    var cliLine: String {
        "\(isDone ? "[x]" : "[ ]") \(title)"
    }

    init?(id: Int, title: String, status: TaskStatus = .pending) {
        let normalizedTitle = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalizedTitle.isEmpty else { return nil }

        self.id = id
        self.title = normalizedTitle
        self.status = status
    }

    mutating func rename(to newTitle: String) -> Bool {
        let normalized = newTitle.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalized.isEmpty else { return false }
        title = normalized
        return true
    }

    mutating func markDone() {
        status = .done
    }

    mutating func reopen() {
        status = .pending
    }
}

struct TaskList {
    private(set) var tasks: [Task]

    var pendingCount: Int {
        tasks.filter { !$0.isDone }.count
    }

    init(tasks: [Task] = []) {
        self.tasks = tasks
    }

    mutating func add(_ task: Task) {
        tasks.append(task)
    }
}
```

这还不是 `TaskCore`。我们还没有拆 package，没有定义独立模块，也没有做更复杂的错误与测试边界。但它已经明显强于 Part 1 的“结构体 + 一堆外部函数”：

- 状态入口更清楚
- 只读和可写边界开始出现
- 构造时有了最小合法性保证
- 模型自己能表达自己的局部行为

这就是 Part 2 真正要练的东西：**不是多会几个语法，而是让类型承担责任**。

## 为什么现在还不该急着抽出一堆 `Manager`

一旦开始认真建模，很多读者会本能地想补出 `TaskManager`、`TaskService`、`TaskHelper`、`TaskUtils`。这通常不是变强，而是把“责任还不清楚”的代码提前塞进模糊容器。

本章更推荐的顺序是：

1. 先让单个模型自己站稳
2. 再判断哪些行为真的属于列表级别
3. 之后再讨论哪些边界值得抽象成协议或模块

换句话说，Part 2 早期的重点不是造层，而是把层级关系看清。属性、方法、初始化器就是这件事的第一步。

## 双语关键词

- 属性：property
- 存储属性：stored property
- 计算属性：computed property
- 方法：method
- 可变方法：`mutating method`
- 初始化器：initializer / `init`
- 可失败初始化器：failable initializer / `init?`
- 不变量：invariant
- 规范化：normalization

## 常见错误

### 1. 把类型写成纯数据袋，行为全塞到外部函数

这会让状态规则散落在调用方各处。只要某个行为直接维护模型自己的状态，它通常就值得先考虑放进方法里。

### 2. 能推导出的信息还额外存一份

如果 `isDone` 已经能从 `status` 推导出来，就不要再独立存一个布尔值。重复状态会制造同步风险。

### 3. 初始化器只负责“抄参数”，不负责建立合法状态

`init` 的关键价值，是让一个值在创建时就满足最小规则，而不是把脏输入先收进系统，留给后面擦屁股。

### 4. 一学会封装就开始到处造 `Manager`

Part 2 前半段的目标是收紧模型，不是发明层。`TaskManager` 这种名字经常只是“我还没想清楚责任边界”的信号。

### 5. 把计算属性当成“语法版 getter”，不思考它表达的语义

好的计算属性不是把旧 OOP 习惯翻译成 Swift，而是用声明方式表达“这个值是模型稳定导出的视图”。

## English Recap

This chapter upgrades `TaskCLI Lite` from data bags to real models. Properties now separate stored state from derived views, methods own local state transitions, and initializers enforce minimal validity at creation time. The key shift is not “more syntax”, but moving responsibility into the type that owns the data.

## Drills

1. 把你手头的 `Task` 练习版本改成 `private(set)` 保护写入口，并补一个 `cliLine` 计算属性。
2. 写一个 `init?`，要求标题去掉首尾空白后不能为空；分别测试合法标题和全空白标题。
3. 给 `Task` 增加 `reopen()`，然后思考：这个行为为什么更适合做成方法，而不是外部函数？

## Project Handoff

到这里，`TaskCLI Lite` 的模型已经不再只是“能装数据”，而开始对自己的状态和出生条件负责。下一章要解决的是另一个更底层的问题：这些模型为什么默认更适合用 `struct` 而不是 `class`，以及值语义（value semantics）和引用语义（reference semantics）会怎样改变你对状态传播、共享与修改的判断。
