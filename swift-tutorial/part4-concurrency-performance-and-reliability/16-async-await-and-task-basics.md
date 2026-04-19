# 第16章：`async`/`await` 与 `Task` 基础

> Part 3 已经把 `TaskCore + TaskCLI` 变成一个像样的 Swift package，但它的运行时心智仍然很“静态”：程序启动，拿到一个 seeded `TaskStore`，同步执行命令，同步打印结果。这一章要处理的，不是“如何把代码写得更炫”，而是当项目终于要面对真实 I/O 与运行时等待时，Swift 并发模型为什么必须出现。

## 为什么这一章现在出现

Part 3 解决的是 package engineering：模块边界、XCTest、CLI 组织、解析与渲染接缝都已经站住。可一旦系统从“教学内存态”继续走向“真实运行态”，新的压力就会立刻出现：

- 任务列表不可能永远只来自 `TaskStore.seeded()`
- 任务状态迟早要从磁盘读入、写回，而这些操作天然带有等待（waiting）
- CLI 不再只是“调用几个同步函数”，而要面对加载、保存、失败、取消这些 runtime path

对很多有其他语言背景的程序员来说，这里最容易犯的错误，是把 `async`/`await` 理解成“让程序自动并行起来”的语法糖。Swift 在这里真正提供的，首先不是并行（parallelism），而是**挂起点（suspension point）可见化**：你终于能在类型签名和调用点上看见“这里会等”。

这就是为什么并发主题必须放在 Part 4 开头。没有这一步，后面所有关于 Actor、隔离（isolation）、Sendability、取消（cancellation）和可靠性（reliability）的讨论都会失去落点，因为系统甚至还没有明确承认：运行时等待是一等公民。

## 从一个还停在同步心智的起点开始

Part 3 末尾的 CLI 主路径大致是这样的：

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
            let title = normalizedTitle(from: arguments)
            let task = try? store.add(title: title)
            return ...
        case "done":
            let title = normalizedTitle(from: arguments)
            let task = try? store.markDone(title: title)
            return ...
        default:
            return "Unknown command: \(command)\n\(usage)"
        }
    }
}
```

这条路径在 Part 3 是正确的，因为它把重点放在模块和核心行为上，而不是运行时复杂度上。问题在于，这种同步心智一旦继续扩展，就会很快逼出两个坏方向。

第一种坏方向是**阻塞式思维（blocking mindset）**。你会很自然地想：“那我就先把文件读出来，再继续执行。”如果这件事只在一个极小 CLI 里做一次，好像问题不大；但一旦运行时路径变长、操作数变多、保存策略变复杂，你就会把“等待 I/O”误写成“占住线程不动”。

第二种坏方向是**回调式补丁（callback patching）**。很多来自 JavaScript、Java、C# 或传统 Cocoa 风格的人，会下意识写出 completion handler 链：

```swift
repository.load { result in
    switch result {
    case .success(let store):
        ...
    case .failure(let error):
        ...
    }
}
```

这当然不是完全错误，但它会让当前教程最看重的东西变差：调用链的清晰度、错误路径的可读性、以及“哪些点会等待”在代码中的可见性。

所以第16章真正要做的事，是把 `TaskCore + TaskCLI` 的运行时心智从“同步直线”升级成“显式挂起、结构化等待（structured waiting）”。

## `async` / `await` 先改变的是函数签名

Swift 并发模型最重要的一步，不是先写 `Task {}`，而是先让真正会等待的 API 承认自己会等待。

如果我们把当前项目从 seeded memory state 推向更真实的持久化路径，一个自然的边界会是：

```swift
protocol TaskRepository {
    func load() async throws -> TaskStore
    func save(_ store: TaskStore) async throws
}
```

这里的关键信号有两个：

- `async`：调用方必须承认，这里会挂起
- `throws`：调用方必须承认，这里可能失败

这两个信号一起，才构成现代 Swift runtime API 的基本诚实度。与其把“等待”和“失败”偷偷藏在回调里，不如直接写进签名。

于是 CLI 主路径会开始长成下面这种形状：

```swift
struct TaskCLIProgram {
    static func run(
        arguments: [String],
        repository: some TaskRepository
    ) async -> String {
        do {
            var store = try await repository.load()
            return try await execute(arguments: arguments, store: &store, repository: repository)
        } catch {
            return "Failed to load tasks: \(error)"
        }
    }
}
```

这一版不一定已经是最终设计，但它比 Part 3 的同步版本强在一个根本点：**等待不再是隐含副作用，而是 API 结构的一部分。**

对已经会别的语言的人，这里要故意纠正一个常见直觉：`await` 不等于“开了个线程”。它表示的是当前异步函数在此处可能挂起，控制权可以让出去，等结果准备好后再回来继续。Swift 把这个“可能暂停”的事实标记出来，目的正是让你的运行时判断更稳。

## 先把 CLI 入口变成异步入口，而不是到处乱包 `Task`

很多人接触 Swift 并发的第一反应，是把现有同步代码外面套一层：

```swift
Task {
    let output = await TaskCLIProgram.run(arguments: ...)
    print(output)
}
```

这有时能工作，但它经常掩盖了真正该改的地方：**主路径本身就应该是异步的。**

对 `TaskCore + TaskCLI` 这样的命令行项目，更稳的方向是让程序入口直接承认异步：

```swift
@main
struct TaskCLIApp {
    static func main() async {
        let repository = JSONTaskRepository(...)
        let arguments = Array(CommandLine.arguments.dropFirst())
        let output = await TaskCLIProgram.run(arguments: arguments, repository: repository)
        print(output)
    }
}
```

这样做有三个工程收益：

1. 顶层调用链不再靠“额外包一层任务”来绕过同步约束。
2. 异步错误、取消和退出行为更容易被集中处理。
3. 后面引入结构化并发（structured concurrency）时，父子任务关系更清楚。

换句话说，`Task` 不是“把同步代码异步化”的万能胶；真正的一等设计单位仍然是 `async` function。

## `Task` 的角色：桥接边界，而不是逃避设计

那 `Task` 到底什么时候该出现？在当前项目线里，最合理的答案是：**当你已经有清楚的异步 API，只是需要在某个同步边界上启动一段异步工作时。**

例如：

- 从同步 shell entry 过渡到异步主流程
- 在测试或工具代码中临时启动一个异步操作
- 在同一命令执行过程中启动一个受父任务约束的子任务

相比之下，下面这些情况就很危险：

- 因为不知道怎么改函数签名，所以随手 `Task { ... }`
- 为了“后台跑一下”，把核心保存逻辑做成 fire-and-forget
- 为了绕过 actor / sendability 限制，直接 `Task.detached`

在教程的项目线上，主命令路径尤其不应该随便 fire-and-forget。比如 `add` 命令如果执行的是“读入 store -> 追加任务 -> 保存 -> 输出结果”，那保存就不该偷偷在后台悬着跑。因为 CLI 最终要向用户报告的是：这次操作是否真的完成、失败、取消，还是只完成了一半。

这正是工程后果（engineering consequence）与语法示例的区别：并发代码不是能跑就行，它必须说清楚“谁在等谁、谁对完成负责、用户什么时候可以相信结果已经落盘”。

## 从同步命令处理演进到异步命令处理

一旦项目开始接入真实 repository，一个更强但仍然克制的命令路径可能会长成这样：

```swift
enum TaskCommand {
    case list
    case add(title: String)
    case done(title: String)
}

struct TaskCLIProgram {
    static func run(
        arguments: [String],
        repository: some TaskRepository
    ) async -> String {
        guard let command = parse(arguments: arguments) else {
            return usage
        }

        do {
            var store = try await repository.load()

            switch command {
            case .list:
                return render(tasks: store.tasks)

            case .add(let title):
                let task = try store.add(title: title)
                try await repository.save(store)
                return "Added: \(task.title)\n" + render(tasks: store.tasks)

            case .done(let title):
                let task = try store.markDone(title: title)
                try await repository.save(store)
                return "Completed: \(task.title)\n" + render(tasks: store.tasks)
            }
        } catch {
            return renderCLIError(error)
        }
    }
}
```

这版代码虽然仍然简单，但它已经开始暴露 Part 4 真正的运行时主题：

- `load()` 与 `save()` 现在是显式等待点
- 核心规则仍然留在 `TaskCore`
- CLI 需要区分 parse failure、core failure、repository failure
- 命令“完成”的定义不再只是内存里改好了，而是外部状态也成功同步了

你应该特别注意最后一点。Part 3 的 `add` 成功，意思是 `TaskStore` 里 append 成功；Part 4 的 `add` 成功，含义会变得更严格：**用户看到成功文案时，系统应当已经完成了这次状态变更的可靠提交。**

这就是并发章节为什么不是纯语法课。它开始改变“系统完成一次操作”到底是什么意思。

## `await` 会暴露出等待点，也会暴露出设计压力

当你在代码里开始连续看到：

```swift
let store = try await repository.load()
try await repository.save(updatedStore)
```

你就会意识到一件事：原来过去被藏起来的 runtime pressure 终于浮出水面了。

这会直接引出几个更高阶的问题：

- 如果两个命令同时访问同一个 store，谁来保证顺序？
- 如果加载、修改、保存之间发生竞争，如何避免 lost update？
- 如果后台保存尚未完成，哪些值可以安全地跨任务传递？
- 如果用户中途取消命令，当前任务该如何停止？

这些问题本章不会一次做完，但 `async`/`await` 的价值正是把它们逼到台面上。一个好的并发模型不是帮你“自动解决复杂度”，而是先把复杂度变成你能看见、能命名、能分层处理的问题。

下一章的 Actor、隔离和 Sendability，就是在回答这些问题。

## `async let` 和结构化并发：只在真的独立时并行

有些读者会问：既然已经进入 `async`/`await`，是不是应该立刻把很多事情并行起来？答案是：**只有独立、可并行、且结果都会被当前作用域消费的工作，才值得考虑结构化并行。**

假设未来 CLI 启动时既要加载任务，也要加载配置文件，那么你才可能考虑：

```swift
async let store = repository.load()
async let config = configLoader.load()

let (loadedStore, loadedConfig) = try await (store, config)
```

这类代码的前提是：

- 两个操作彼此独立
- 当前作用域确实需要同时等待二者结果
- 失败和取消应当一起受当前父任务管理

相比之下，如果一条命令本来就必须先改 store、再保存、再渲染输出，那它并不是一个适合乱并行的场景。真实工程里，很多所谓“并发优化”其实只是把顺序逻辑写乱了。

所以 Part 4 对结构化并发的第一要求很朴素：先尊重依赖关系，再讨论并行机会。

## 双语关键词

- `async`：异步函数标记
- `await`：挂起等待
- suspension point：挂起点
- structured concurrency：结构化并发
- `Task`：任务
- parent task：父任务
- child task：子任务
- async entry point：异步入口
- blocking：阻塞
- fire-and-forget：发出去就不管
- repository：仓储 / 持久化边界
- runtime path：运行时路径

## 常见错误

### 1. 把 `async` / `await` 理解成“自动并行”

`await` 首先表示“这里可能挂起”，不是“这里一定开新线程”。如果不先建立这个直觉，后面会把很多顺序逻辑误写成看似高级、实则更脆的并发代码。

### 2. 一遇到异步就到处包 `Task { ... }`

如果真正需要等待的 API 仍然是同步签名，只在调用点额外套 `Task`，那通常是在逃避设计。应先让会等待的函数直接成为 `async`。

### 3. 在主命令路径里随便 fire-and-forget

`add` 或 `done` 的保存如果没有被等待完成，CLI 就可能在“显示成功”之后才真正失败。这会直接伤害可靠性。

### 4. 仍然把 repository failure 当成和 core failure 一回事

空标题、任务不存在属于领域或输入问题；加载失败、保存失败属于运行时问题。`async`/`await` 会逼你看清这条边界，别再把它们混成一个 catch-all 文案。

## English Recap

This chapter introduces `async`/`await` as a runtime honesty tool, not a parallelism gimmick. For `TaskCore + TaskCLI`, the important move is to make loading and saving explicit suspension points, turn the CLI entry into an async path, and treat `Task` as a boundary tool rather than a shortcut around design.

## Drills

1. 用一句话区分“同步函数里调用 I/O”与“把 I/O 写成 `async throws`”在工程可读性上的差别。
2. 假设 `TaskCLI` 要从磁盘加载任务后再执行 `done`，写出你认为至少应该出现在哪些函数签名上的 `async` / `throws`。
3. 解释为什么 `Task { try await repository.save(store) }` 对主命令路径来说通常不是一个稳妥设计。

## Project Handoff

现在 `TaskCore + TaskCLI` 已经有资格从“同步 package”进入“异步 runtime”。但只要你认真把 `load` / `save` 写成 `async`，下一个问题就会立刻出现：共享状态到底谁来保护，哪些值能安全跨任务传递，哪些引用会把系统重新拖回 race condition？下一章我们就处理 Actor、隔离与 Sendability。
