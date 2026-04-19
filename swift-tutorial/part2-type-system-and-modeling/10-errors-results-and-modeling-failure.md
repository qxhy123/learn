# 第10章：错误、Result 与失败建模

> 到了 Part 2 的这个位置，`TaskCLI Lite` 已经有了模型、语义、边界和更强的类型关系。系统越像工程代码，一个问题就越不能继续含糊：失败到底怎么表示。

## 为什么这一章现在出现

Part 1 里，很多失败处理都可以接受保持粗糙：

- 找不到任务就 `print("Task not found")`
- 标题为空就返回 `nil`
- 命令不认识就给一句 `Unknown command`

那时教程重点是先把语法和最小程序骨架立住，这样做没有问题。可一旦进入 Part 2，这些做法就会越来越不够用，因为你开始在做真正的模型和 API 设计。

你会马上遇到几个更严肃的问题：

- “任务不存在”和“任务已完成”都算失败，但它们显然不是同一种失败
- `nil` 到底表示“合法缺省”还是“出错了但我没说原因”
- 某个操作应该立刻抛出错误（`throws`），还是把成功/失败当值传出去（`Result`）
- 用户可读的提示文本，应该直接塞进错误里，还是先保持为领域错误再映射到 CLI 输出

这就是错误建模（error modeling）现在必须出现的原因。Swift 的目标不是让你“少写异常”，而是让你更清楚地区分：

- absence：没有值，但这本身可能是正常情况
- failure：发生了失败，而且失败原因值得表达

很多别的语言会把这两者混在一起。Swift 则在语言层和库层给了你三种常见工具：

- `Optional`
- `throws`
- `Result`

关键不是记住三种语法，而是学会何时用哪一种。

## 先看一个 Part 1 风格、现在已经不够强的版本

```swift
mutating func markDone(id: Int) -> Bool {
    guard let index = tasks.firstIndex(where: { $0.id == id }) else {
        return false
    }

    tasks[index].markDone()
    return true
}
```

返回 `Bool` 在 Part 1 完全能接受，因为它只回答一个简单问题：成功还是失败。

但到了 Part 2，这就开始不够了。调用方没法知道失败到底是因为：

- 没找到任务
- 任务本来就完成了
- `id` 非法

这会让上层代码只能写出含糊逻辑，测试也只能断言“失败了”，却说不清“为什么失败”。

## Optional 适合表达“可能没有”，不适合承载所有失败

先把一个重要边界说清楚。`Optional` 非常有用，但它不是通用错误机制。

例如，下面这个 API 很合理：

```swift
func task(id: Int) -> Task? {
    tasks.first { $0.id == id }
}
```

因为“按 id 查找，可能找不到”在这里更接近正常分支，而不是异常情况。调用方只需要知道：有就拿到，没有就没有。

但如果你写：

```swift
mutating func add(title: String) -> Task? {
    // ...
}
```

那就开始含糊了。返回 `nil` 到底意味着：

- 标题为空
- 标题重复
- 任务上限已满

全都压成 `nil` 之后，系统就失去了分辨能力。

经验法则可以先记成一句话：

- `Optional` 适合 absence without explanation
- `Error` / `Result` 适合 failure with meaning

## 先把失败类型命名出来

Swift 里最常见、也最推荐的错误类型，是 `enum`。

对于 `TaskCLI Lite` 当前阶段，我们可以先写一个任务变更错误：

```swift
enum TaskMutationError: Error, Equatable {
    case taskNotFound(id: Int)
    case taskAlreadyDone(id: Int)
    case emptyTitle
}
```

这段代码很短，却完成了非常重要的建模升级：

- 失败原因被命名了
- 不同失败分支可以被测试区分
- 上层可以根据失败类型决定提示文本

注意，这里我们还没有把错误直接写成用户提示字符串，比如 `"Task #3 not found"`。那是因为领域错误和展示文本最好分层。错误类型负责表达语义，CLI 文本负责面向用户。把两者混死，后面会更难维护。

## `throws` 适合“当前调用链就要处理”的失败

如果一个操作失败后，调用方通常会立刻决定怎么办，那么 `throws` 往往最自然。

把前面的 `markDone` 改强：

```swift
mutating func markDone(id: Int) throws {
    guard let index = tasks.firstIndex(where: { $0.id == id }) else {
        throw TaskMutationError.taskNotFound(id: id)
    }

    guard !tasks[index].isDone else {
        throw TaskMutationError.taskAlreadyDone(id: id)
    }

    tasks[index].markDone()
}
```

这里的语义很清楚：

- 成功时没有额外值，就直接完成修改
- 失败时抛出一个明确错误

调用方也会被 Swift 强迫显式处理：

```swift
do {
    try list.markDone(id: 3)
    print("Completed task 3")
} catch let error as TaskMutationError {
    print(render(error))
}
```

这和 Python / JavaScript 里“想不想接异常，全看调用方随缘”有点像，但 Swift 更强调显式性：只要函数会 `throw`，调用方就必须承认这一点。

对来自 Go 的读者来说，这也很值得比较。Go 倾向于把错误作为返回值并排返回；Swift 选择的是另一条路：把失败控制流从普通返回路径里分开，但仍然要求显式处理。

## `Result` 适合把成功或失败当作值继续传递

那为什么还需要 `Result`？

因为不是所有失败都应该立刻在当前调用点被处理。有时你想把“结果”本身当作值继续传递、收集、组合或测试。

例如，想把多次批量完成任务的结果收集起来：

```swift
func completeAll(
    ids: [Int],
    in list: inout TaskList
) -> [Result<Int, TaskMutationError>] {
    ids.map { id in
        do {
            try list.markDone(id: id)
            return .success(id)
        } catch let error as TaskMutationError {
            return .failure(error)
        } catch {
            fatalError("Unexpected error type: \(error)")
        }
    }
}
```

现在每个完成操作的结果都被保留成一个值。你可以：

- 统计成功数和失败数
- 逐条渲染失败报告
- 在测试里精确断言每个结果

这就是 `Result` 的优势：它把原本只能沿调用栈立刻处理的失败，变成了可以存储、传递、映射（map）和组合的普通值。

在 `TaskCLI Lite` 这个教程阶段，`Result` 很适合拿来表达：

- 批处理结果
- 需要延迟到更外层再解释的命令执行结果
- 希望在测试中直接比对的成功/失败值

## 一个更完整的 CLI 失败面分层

到这里，我们可以开始做一个非常重要的工程区分：不是所有失败都属于同一层。

对于当前项目，至少有三层失败值得区分：

- 输入解析失败：比如命令缺失、ID 不是数字
- 领域操作失败：比如任务不存在、任务已完成
- 展示层输出：把这些失败翻译成用户可理解文本

可以先定义解析错误：

```swift
enum TaskCommandError: Error, Equatable {
    case missingCommand
    case missingTitle
    case invalidID(String)
    case unknownCommand(String)
}
```

解析器：

```swift
struct TaskCommandParser {
    func parse(arguments: [String]) throws -> TaskCommand {
        guard let command = arguments.first else {
            throw TaskCommandError.missingCommand
        }

        switch command {
        case "list":
            return .list
        case "add":
            let title = arguments.dropFirst().joined(separator: " ")
            guard !title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw TaskCommandError.missingTitle
            }
            return .add(title: title)
        case "done":
            guard let rawID = arguments.dropFirst().first else {
                throw TaskCommandError.invalidID("")
            }
            guard let id = Int(rawID) else {
                throw TaskCommandError.invalidID(rawID)
            }
            return .done(id: id)
        default:
            throw TaskCommandError.unknownCommand(command)
        }
    }
}
```

然后在更外层统一映射：

```swift
func render(_ error: TaskCommandError) -> String {
    switch error {
    case .missingCommand:
        return "Usage: task-cli-lite <command>"
    case .missingTitle:
        return "Task title cannot be empty"
    case .invalidID(let raw):
        return "Invalid task id: \(raw)"
    case .unknownCommand(let command):
        return "Unknown command: \(command)"
    }
}

func render(_ error: TaskMutationError) -> String {
    switch error {
    case .taskNotFound(let id):
        return "Task #\(id) not found"
    case .taskAlreadyDone(let id):
        return "Task #\(id) is already completed"
    case .emptyTitle:
        return "Task title cannot be empty"
    }
}
```

这就比“每层都直接 print 一句字符串”强太多了，因为：

- 各层职责清楚
- 测试能对错误语义做断言
- 用户提示可以统一管理

## 什么时候不用 `throws`，什么时候不用 `Result`

学到这里，很多人会进入另一个极端：仿佛所有失败都必须上错误枚举、`throws`、`Result`。

也不是这样。

还是那条判断：

- 如果没有值是正常情况，且不需要说明原因，用 `Optional`
- 如果失败需要当前调用方立刻处理，用 `throws`
- 如果成功/失败本身要被保留成值继续传递，用 `Result`

例如：

- `task(id:) -> Task?` 很合理
- `markDone(id:) throws` 很合理
- `completeAll(ids:) -> [Result<Int, TaskMutationError>]` 也很合理

真正糟糕的情况，是三者职责混掉：

- 找不到值和解析失败都统一返回 `nil`
- 已经要跨层传递结果了，还坚持只用 `throws`
- 当前点就能处理失败，却把一切都包成 `Result` 让调用方多拆一次

## 这一步如何为 Part 3 做准备

错误建模对 Part 3 的帮助非常直接。

等你开始做 package、测试和 CLI 工程化时，最需要稳定的东西之一就是边界契约（boundary contract）。如果边界上的失败只靠打印文本或模糊布尔值表达，包一拆开，测试一细化，系统立刻变脆。

相反，如果你已经在 Part 2 形成这些习惯：

- 解析失败有解析错误类型
- 领域失败有领域错误类型
- 输出文本是外层映射，而不是底层语义本体

那么进入 Part 3 时，很多模块边界已经天然更清楚了。

这也是为什么本章不只是“错误处理技巧”，而是建模主题的一部分。失败面本身就是模型的一部分。

## 双语关键词

- 错误建模：error modeling
- 失败：failure
- 可选值：optional / `Optional`
- 抛出错误：`throws`
- 结果类型：`Result`
- 领域错误：domain error
- 解析错误：parsing error
- 边界契约：boundary contract

## 常见错误

### 1. 把所有失败都压成 `nil` 或 `false`

这会让系统失去区分能力。只要失败原因值得表达，就应该考虑错误类型。

### 2. 在底层模型里直接拼用户提示字符串

领域错误和展示文本最好分层。模型层应该优先表达语义，不要过早绑死输出文案。

### 3. 已经需要立即处理的失败，还硬包成 `Result`

如果当前调用链就会处理，`throws` 往往更自然。

### 4. 只是“可能找不到”也上复杂错误体系

不是所有 absence 都是 failure。`Optional` 依然是 Swift 非常重要、也非常正确的工具。

### 5. `catch` 到错误后只打印，不重新设计边界

如果你发现自己到处 `catch { print(error) }`，通常说明失败语义还没被好好建模，只是被往外推了。

## English Recap

This chapter turns failure into part of the model. `Optional` is for expected absence, `throws` is for failures the current caller should handle immediately, and `Result` is for outcomes that need to travel as values. The main goal is not “more error syntax”, but preserving failure meaning so APIs, tests, and future module boundaries stay clear.

## Drills

1. 把一个返回 `Bool` 的任务操作改写成 `throws`，并定义对应错误枚举。
2. 写一个 `TaskCommandError`，至少覆盖“缺少命令”和“未知命令”两种情况。
3. 设计一个批量完成任务的函数，让它返回 `[Result<Int, TaskMutationError>]`，再思考这为什么比 `[Bool]` 更有信息量。

## Project Handoff

Part 2 到这里完成了从“会写 Swift 语法”到“会设计 Swift 模型与 API”的关键升级：属性和初始化器负责建立合法状态，值语义决定默认建模方向，协议与泛型给出边界和关系，而错误建模把失败也纳入类型系统。下一部分终于可以进入 package engineering，但不是空手进入，而是带着已经成形的模型、边界和契约，把 `TaskCLI Lite` 自然推进到 `TaskCore + TaskCLI`。
