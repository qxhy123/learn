# 第17章：Actor、隔离与 Sendability

> 第16章把 `TaskCore + TaskCLI` 从同步心智推进到异步心智，但只要系统开始出现 `async load -> mutate -> save` 这类路径，共享可变状态（shared mutable state）的问题就再也藏不住了。本章要处理的不是“会不会写 `actor` 关键字”，而是 Swift 为什么坚持把并发安全写进类型与隔离边界里。

## 为什么这一章现在出现

只要项目引入异步 repository，上一章那种看似自然的流程就会暴露出一个尖锐事实：**等待点一出现，竞争条件（race condition）就有了现实空间。**

举个很具体的项目压力：

- 命令 A 读取任务列表，准备 `add`
- 命令 B 几乎同时读取同一份任务列表，准备 `done`
- 两边都基于旧快照修改，再分别保存

如果系统没有明确隔离策略，就可能出现 classic lost update：后保存的一方覆盖前一方，导致其中一次变更被悄悄吞掉。

Part 3 的 `TaskStore` 是一个值类型（value type），这很好；但值类型本身不会自动替你解决**共享访问路径**的问题。只要你把某个可变 runtime state 藏进 reference object、全局变量、缓存单例，或者跨多个任务访问的协调器里，并发风险就会回到现场。

所以 Actor 现在出现，不是因为它“新潮”，而是因为 Part 4 终于开始处理一件真实工程事实：异步系统不能只靠“大家小心点不要同时改”来维持正确性。

## 从一个异步了、但仍然共享可变状态的版本开始

假设我们天真地把上一章的 runtime 升级写成：

```swift
final class TaskRuntime {
    private var store: TaskStore
    private let repository: any TaskRepository

    init(store: TaskStore = .seeded(), repository: some TaskRepository) {
        self.store = store
        self.repository = repository
    }

    func add(title: String) async throws -> Task {
        if store.tasks.isEmpty {
            store = try await repository.load()
        }

        let task = try store.add(title: title)
        try await repository.save(store)
        return task
    }
}
```

这段代码的危险之处，恰恰在于它“看起来很正常”。

对于来自 Java、C#、Kotlin 或 JavaScript 的读者，这种 `class + mutable field + async method` 组合非常眼熟。但在 Swift 并发语境里，它正好踩中了最容易出事的形状：

- `store` 是共享可变状态
- `add(title:)` 会跨过 `await`
- 在 `await` 之前和之后，这个对象都可能被别的任务同时访问

这意味着“我以为自己还在改同一份状态”的直觉并不可靠。只要第二个任务在第一个任务挂起期间进来，内部状态就可能被 interleave。

## Actor 的第一价值：把可变状态和访问路径绑在一起

更强的方向不是“加锁到处补洞”，而是把共享状态明确放进 Actor：

```swift
actor TaskRuntime {
    private var store: TaskStore
    private let repository: any TaskRepository

    init(store: TaskStore = .seeded(), repository: some TaskRepository) {
        self.store = store
        self.repository = repository
    }

    func list() async throws -> [Task] {
        if store.tasks.isEmpty {
            store = try await repository.load()
        }

        return store.tasks
    }

    func add(title: String) async throws -> Task {
        if store.tasks.isEmpty {
            store = try await repository.load()
        }

        let task = try store.add(title: title)
        try await repository.save(store)
        return task
    }
}
```

这里最关键的不是“`actor` 会自动加锁”这种表面说法，而是：**Actor 把某块状态的隔离责任（isolation responsibility）写成了语言规则。**

工程上，这带来三个很实际的后果：

1. `store` 不再能被任意同步代码随便碰。
2. 跨 actor 边界访问时，调用方必须用 `await` 明确承认隔离切换。
3. 你会被迫思考哪些数据应留在 actor 内部，哪些数据适合以值快照（value snapshot）形式传出去。

这就是 Swift 并发设计的核心气质：不是让你“更方便地共享状态”，而是让你更难不小心共享错状态。

## 隔离（isolation）不是语法细节，而是边界设计

许多教程会把 actor 讲成“线程安全对象”，这很容易让人低估 Swift 隔离模型的设计价值。对当前项目而言，更准确的理解是：

- `TaskRuntime` actor 负责保护任务运行态
- `TaskCore.TaskStore` 仍然表达领域规则
- `TaskCLI` 作为客户端，通过异步消息式调用和 runtime 交互

于是 CLI 路径会开始长成：

```swift
struct TaskCLIProgram {
    static func run(arguments: [String], runtime: TaskRuntime) async -> String {
        guard let command = parse(arguments: arguments) else {
            return usage
        }

        do {
            switch command {
            case .list:
                let tasks = try await runtime.list()
                return render(tasks: tasks)
            case .add(let title):
                let task = try await runtime.add(title: title)
                let tasks = try await runtime.list()
                return "Added: \(task.title)\n" + render(tasks: tasks)
            case .done(let title):
                let task = try await runtime.markDone(title: title)
                let tasks = try await runtime.list()
                return "Completed: \(task.title)\n" + render(tasks: tasks)
            }
        } catch {
            return renderCLIError(error)
        }
    }
}
```

你会发现一个非常 Swift 的设计后果：CLI 不再直接拿着 `inout store` 到处改，而是通过 actor 定义好的异步接口和运行时交互。也就是说，**命令行层获得的是服务边界（service boundary），不是裸露的共享可变状态。**

这是一个很值得建立的工程直觉。好的隔离边界会迫使上层代码用更明确、更可审计的方式访问系统状态。

## Actor 不是万能安全罩：`await` 之间仍然要小心设计

到这里还不能松懈，因为很多人会马上产生另一个误解：只要用了 actor，一切竞争问题就结束了。

现实没这么简单。Actor 能保护它自己的隔离状态，但它不能替你自动修复不合理的操作序列。比如：

```swift
func refreshThenSave() async throws {
    let freshStore = try await repository.load()
    store = freshStore
    try await repository.save(store)
}
```

这个例子里，逻辑是否正确仍然取决于你怎么定义“刷新”和“保存”的时序意义。Actor 可以帮你避免多个任务同时乱改 `store`，但不能替你决定：

- 到底该先 load 再 mutate，还是先 mutate 再 append journal
- 保存失败时内部状态是否回滚
- 是否应该返回保存前快照还是保存后快照

也就是说，Actor 解决的是**隔离执行**，不是**业务语义自动正确**。这和数据库事务、日志写入、幂等性（idempotency）这些后续运行时话题是相连的。

Part 4 的学习重点因此不是“用了 actor 就放心”，而是“用了 actor 以后，哪些顺序语义终于能被稳定表达出来，哪些仍要你自己设计”。

## Sendable：跨并发边界时，值到底能不能安全传

Actor 讨论完以后，Swift 很快会把另一个概念推到你面前：`Sendable`。

对很多新手来说，这个词看起来像额外的类型噪音；但它处理的是一个非常现实的问题：**某个值跨任务、跨 actor、跨并发边界传递时，语言能否相信它不会偷偷带着共享可变引用一起过去。**

在 `TaskCore + TaskCLI` 里，最天然适合跨边界传递的是值类型：

- `TaskStatus`
- `Task`
- `TaskStore` 的快照
- 解析后的 `TaskCommand`

如果这些类型只由 `Sendable` 成员组成，那么把它们作为异步返回值、传给子任务或跨 actor 返回就更稳。

一个很直接的项目判断是：**越接近领域核心、越经常跨并发边界的模型，越应该优先保持 value-oriented。**

这也是为什么 Part 2 对结构体（struct）、值语义（value semantics）的强调，在 Part 4 并没有过时，反而变得更重要。值语义不只是“建模优雅”，它会直接影响并发安全的可组合性。

## 把 Sendability 设计成项目优势，而不是编译器麻烦

如果你此时把 runtime 设计成大量引用对象互相指着彼此，比如：

- `TaskRuntime` 持有可变 class cache
- repository 返回内部可变 buffer 引用
- parser / renderer / logger 共享某个非线程安全对象

那么跨并发边界时，`Sendable` 检查就会不断提示你：这些东西并不天然适合安全传递。

更强的做法不是一味用 `@unchecked Sendable` 把告警压下去，而是先问：

- 这个类型真的需要共享身份（identity）吗？
- 它是否其实只需要传一个 snapshot？
- 它是不是本来就应该被锁在 actor 内部，不应该外送？

在我们的项目线里，一个很稳的设计方向是：

- `Task`, `TaskStatus`, `TaskCommand`, CLI output model 尽量保持为可发送的值
- repository 或 runtime 这类持有外部资源、需要序列化访问的东西留在 actor / reference owner 内部
- 对外暴露快照或明确结果，而不是暴露“还能继续随便改”的内部引用

这会让系统在后续章节里更容易谈性能、可靠性和 UI 复用。因为安全可传的值，比共享悬空引用更容易被推理。

## `nonisolated`、`detached` 和“为了方便绕过规则”的诱惑

一旦开始跟 actor 共事，很多程序员会迅速发现一些“看起来更省事”的逃生口，比如：

- 把某些成员标成 `nonisolated`
- 用 `Task.detached` 把工作扔出去
- 用 `@unchecked Sendable` 让编译器闭嘴

这些工具不是绝对不能用，但在教程当前阶段，它们最大的风险是：你会在还没有建立稳定并发直觉前，就先学会了怎么绕开安全护栏。

对 `TaskCore + TaskCLI` 这条项目线，更稳的顺序应该是：

1. 先用 actor 把状态归位
2. 先用值快照穿过边界
3. 先让 `Sendable` 约束帮助你发现设计问题
4. 只有在充分理解 trade-off 后，再考虑局部豁免

尤其不要把 `@unchecked Sendable` 当成“Swift 太严格，我先让它过编译”的常规策略。那不是解决问题，而是在把未来的并发 bug 推迟到账。

## 从 `TaskStore` 到 `TaskRuntime`：项目边界真正变强了什么

这一章如果只停留在“把 class 改成 actor”，那就太轻了。真正重要的升级是边界意义发生了变化：

- Part 3：`TaskStore` 是 core behavior 的中心
- Part 4：`TaskRuntime` 开始成为 runtime coordination 的中心

这两个中心并不冲突。`TaskStore` 仍然定义领域规则，如标题规范化、状态转移和错误类型；`TaskRuntime` 负责把这些规则放进异步运行环境里，协调加载、保存、串行访问与任务边界。

这正是一个成熟 Swift 工程该有的层次感：

- 领域模型负责语义正确
- runtime boundary 负责并发正确
- CLI 负责把结果组织成用户可理解的交互

如果你能看清这三层，Actor 就不再是孤零零的语法点，而是项目架构继续变强的必要一环。

## 双语关键词

- actor：参与者 / Actor 隔离对象
- actor isolation：Actor 隔离
- isolation：隔离
- shared mutable state：共享可变状态
- race condition：竞争条件
- lost update：更新丢失
- `Sendable`：可安全跨并发边界传递
- value snapshot：值快照
- `nonisolated`：非隔离成员
- `Task.detached`：脱离父任务的独立任务
- `@unchecked Sendable`：跳过编译器完整检查的可发送声明

## 常见错误

### 1. 以为值类型出现过，就等于并发已经安全

`TaskStore` 是 struct 很好，但只要外层 runtime 仍然把可变状态放在共享引用对象里，并发问题就依然存在。

### 2. 把 actor 理解成“什么都不用想的安全罩”

Actor 保护的是隔离访问，不会替你自动设计出正确的 load / mutate / save 语义，更不会自动解决持久化策略问题。

### 3. 遇到 `Sendable` 约束就先压过去

`@unchecked Sendable` 应该是深思熟虑后的局部决定，而不是日常消音器。多数时候，警告暴露的是边界设计问题。

### 4. 滥用 `Task.detached` 绕开隔离模型

脱离父任务后，取消、优先级和生命周期关系都会改变。对 `TaskCore + TaskCLI` 当前主路径而言，它通常不是默认选择。

## English Recap

This chapter moves the project from “async but still racy” to “async with explicit isolation.” Actors give `TaskCore + TaskCLI` a runtime boundary for shared state, while `Sendable` pushes value-oriented designs that are safer to move across concurrency domains. The lesson is architectural: isolate mutable runtime state, and send snapshots instead of leaking shared references.

## Drills

1. 用自己的话解释为什么 `final class TaskRuntime` 里一边持有 `var store` 一边跨 `await` 修改，会比同步版本更危险。
2. 列出当前项目里你认为最适合保持 `Sendable` 的三个值，并说明原因。
3. 说明为什么“CLI 拿到整个可变 store 自己改”比“CLI 调用 actor 暴露的操作接口”更弱。

## Project Handoff

现在我们已经把 `TaskCore + TaskCLI` 的运行态隔离边界看清楚了。但并发安全还不是全部。只要系统开始持有 actor、repository、后台保存任务和闭包回调，另一个经常被忽略的问题就会出现：谁拥有谁、对象何时释放、闭包和任务会不会把生命周期越拉越长。下一章我们就把视线转向 ARC、内存和 ownership。
