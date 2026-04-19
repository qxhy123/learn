# 第18章：ARC、内存与 ownership 实战

> 并发一出现，很多读者会立刻关注 race condition，却低估了另一个同样真实的工程问题：生命周期（lifetime）。`TaskCore + TaskCLI` 到了 Part 4，已经不再只有几个短命同步值；它开始拥有 runtime actor、repository、后台任务、闭包和潜在的外部资源。这一章要建立的，是 Swift 在真实工程里关于内存、ARC 和 ownership 的判断力。

## 为什么这一章现在出现

Part 2 已经讲过类（class）与结构体（struct）的差别，也讲过值语义与引用语义。可那个阶段的重点主要是建模和 API 判断。到了 Part 4，问题的性质变了：

- 你可能会引入 repository 或 runtime coordinator 这类长期存活对象
- 你可能会持有 `Task` handle、闭包、日志器、缓存、编码器、文件资源
- 你开始写异步代码，而异步闭包经常会悄悄延长对象生命周期

这意味着“引用类型会被 ARC 管理”这句基础知识已经不够了。真正重要的问题变成：

- 哪些东西应该有稳定身份（identity），哪些只该是值快照？
- 谁拥有（own）后台任务？它什么时候结束？
- 一个闭包或 `Task` 强捕获 `self`，会不会让系统比预期活得更久？
- 失败或取消发生时，资源是否能正确清理？

如果不在这里建立 ownership 心智，后面一旦把 runtime 做实，系统就很容易出现另一种“看不见的 bug”：没有 data race，但对象留太久、资源不释放、保存任务悬挂、甚至形成 retain cycle。

## 从“并发安全了就行”的弱直觉开始

很多人学完 actor 以后，会产生一种危险的满足感：只要共享状态都隔离起来，系统就安全了。

这时很容易写出下面这种代码：

```swift
final class AutosaveCoordinator {
    private let runtime: TaskRuntime
    private var pendingSave: Task<Void, Never>?

    init(runtime: TaskRuntime) {
        self.runtime = runtime
    }

    func scheduleSave() {
        pendingSave = Task {
            try? await Task.sleep(for: .milliseconds(300))
            try? await runtime.flush()
        }
    }
}
```

它的问题不一定是 race，而是 lifetime：

- `pendingSave` 持有一个 `Task`
- `Task` 的闭包强捕获 `runtime`
- 如果闭包又间接强捕获 `self`，对象之间的生命周期会被拉长
- 如果没有取消旧任务，新任务会不断覆盖但旧任务仍在后台活着

这类问题不会像类型错误那样立刻在编译期大喊大叫，但它会慢慢把运行时行为变脆：用户以为命令结束了，后台任务还在跑；对象看似该释放了，实际上被悬挂闭包继续留着。

所以第18章的关键不只是“记住 weak self”，而是学会判断：**谁应该拥有这段工作，拥有多久，结束时该怎样清理。**

## ARC 在当前项目里到底管理什么

先把范围说清楚。ARC（Automatic Reference Counting）管理的是引用类型实例的生命周期，比如 class 实例、闭包捕获图中的引用对象等。它不直接管理值类型本身的“引用计数”。

对 `TaskCore + TaskCLI` 来说，这个区分非常关键：

- `Task`、`TaskStore`、`TaskCommand` 这类值类型，重点更多在复制语义与所有权边界
- repository、logger、cache、coordinator 这类引用对象，重点才是 ARC 生命周期

这也是为什么 Swift 工程里经常强调“让领域模型尽量保持 value-oriented”。它不仅让建模更稳，也能减少你需要处理的 ARC 表面面积。

换句话说，如果你把任务系统的核心状态也全部做成 class graph，那么 Part 4 接下来要处理的复杂度会成倍增长：并发隔离难、生命周期难、性能判断也更难。

## 强状态：值承载事实，引用承载资源与身份

一个更成熟的项目判断是：

- 任务事实（task facts）尽量保持为值：`Task`, `TaskStatus`, `TaskStore` snapshot
- 资源持有者（resource owners）才使用引用：repository、runtime coordinator、持久化会话、后台保存器

这条分工的价值非常高。因为一旦“值”和“资源所有者”分工清楚，你会更容易回答这些问题：

- 哪些数据可以安全复制、传递、缓存？
- 哪些对象必须明确 cancel / close / release？
- 哪些边界适合返回 snapshot，而不是暴露活引用？

对当前项目，一个很稳的运行时形状可能是：

```swift
actor TaskRuntime {
    private var store: TaskStore
    private let repository: TaskRepository
}

final class AutosaveCoordinator {
    private let runtime: TaskRuntime
    private var pendingSave: Task<Void, Never>?
}
```

这里 `TaskStore` 仍然表达值世界里的任务事实；`TaskRuntime` 和 `AutosaveCoordinator` 则属于拥有资源、管理时序和生命周期的引用世界。二者职责不同，不应混成一类。

## 闭包与 `Task` 会偷偷改变对象寿命

Swift 新手最容易低估的一点，是闭包和异步任务会改变 retain graph。它们不是“只执行一会儿的代码块”，而是可能在后台持有捕获对象直到任务完成。

例如：

```swift
final class AutosaveCoordinator {
    private var pendingSave: Task<Void, Never>?

    func scheduleSave() {
        pendingSave = Task {
            try? await Task.sleep(for: .milliseconds(300))
            await self.flushNow()
        }
    }
}
```

这里的 `Task` 闭包强捕获 `self`。只要这个任务还活着，`AutosaveCoordinator` 就不会释放。若 `flushNow()` 又依赖其他对象，生命周期会继续向外传导。

更稳的版本通常至少要明确两件事：

1. 旧任务是否应该取消
2. 当前对象若已不再需要，这个后台任务是否还值得继续

例如：

```swift
final class AutosaveCoordinator {
    private let runtime: TaskRuntime
    private var pendingSave: Task<Void, Never>?

    init(runtime: TaskRuntime) {
        self.runtime = runtime
    }

    func scheduleSave() {
        pendingSave?.cancel()

        pendingSave = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(300))
            guard let self else { return }
            try? await self.runtime.flush()
        }
    }

    deinit {
        pendingSave?.cancel()
    }
}
```

这段代码不是“模板答案”，但它体现了 ownership 判断：

- `pendingSave` 的所有者是 `AutosaveCoordinator`
- 旧任务被新任务替换前应取消
- 对象析构时，后台任务也应被取消
- 闭包不再无条件强拉住 `self`

这才是实战中的 ARC 讨论。真正重要的不是记住 `[weak self]` 四个字，而是明白引用关系为什么该长这样。

## retain cycle 不是 UI 专属问题，CLI/runtime 一样会踩

很多人只在 SwiftUI 或 UIKit 里学过 retain cycle，于是误以为命令行项目没这个烦恼。实际上，只要你有 class、closure、长期任务、回调或 observer，就同样会踩。

在 `TaskCore + TaskCLI` 的 Part 4 语境里，下面这些形状都可能出问题：

- coordinator 持有 task handle，task closure 强捕获 coordinator
- repository 持有 completion closure，closure 强捕获 repository owner
- logger / metrics sink 互相引用，谁都不释放
- actor 外围的 class 对象把 cancel token 和自己绑成闭环

所以 retain cycle 不是“界面开发者的毛病”，而是所有 Swift 引用图都会遇到的生命周期问题。CLI 只是可视化更少，不代表风险更小。

## ownership 不止是 ARC：还包括谁暂借、谁复制、谁消费

这一章不能把“ownership”窄化成 ARC，因为 Swift 的 ownership 心智还有更广的一层：谁真正拥有某段值，谁只是短暂借用（borrow-like use），谁在消费（consume）并建立新状态。

对当前项目，几个典型判断是：

- `TaskCLI` 渲染列表时，只需要读取任务快照，不应拥有 runtime 内部可变状态
- repository 保存时，需要的是某个稳定 snapshot，而不是一个还会继续被上层改动的裸引用
- `TaskStore.add` / `markDone` 的 `mutating` 语义，表达的是“在当前所有者上下文里变更值”

这也是为什么 Swift 的值语义、`inout`、`mutating`、actor isolation 会在 Part 4 汇合到一起。它们都在逼你回答同一个问题：**这段状态现在到底归谁负责？**

如果这个问题答不清，后面不是发生数据竞争，就是发生生命周期混乱。

## 资源清理：`deinit`、取消与 `defer`

一旦 runtime 引入文件 I/O、后台保存、日志句柄、临时缓冲等资源，另一个实际问题会马上出现：失败和取消发生时，清理是否可靠。

这里至少要建立三个工程直觉：

1. 有明确所有者的后台任务，应在 `deinit` 或停止路径中取消。
2. 局部临时资源的回收，优先靠作用域与 `defer` 保证。
3. 资源释放语义应与“谁拥有它”一致，不要把 cleanup 分散到很多外部调用点。

例如，当 repository 实现里需要临时写文件再替换正式文件时，`defer` 往往能比 scattered cleanup 更稳；而当某个 coordinator 生命周期结束时，取消 pending task 则应由它自己负责，而不是期待外面有人记得“顺手取消一下”。

这类设计不会直接体现在教程的 starter code 里，但它们决定了你未来写出来的是“偶尔能跑”的并发系统，还是“长期可维护”的并发系统。

## 为什么 Part 4 要把 ownership 拉回 `TaskCore + TaskCLI`

如果你只把 ARC 当成一章独立知识点，很容易觉得它和当前项目线关系不大。可实际上，ownership 正是把前两章串起来的东西：

- 第16章告诉你哪里会等待
- 第17章告诉你谁隔离共享状态
- 第18章告诉你这些对象与任务会活多久、谁拥有谁、谁负责清理

这三者缺一不可。因为真实系统里，“会等待”与“会共享”最终都会落在“会不会活太久、释放太晚、或者在错误时没人收尾”上。

对 `TaskCore + TaskCLI` 来说，ownership 判断还有一个额外价值：它帮助你守住“core 继续值化，runtime 只在必要处引用化”这条项目线。只要这条线不丢，后面的性能与可靠性讨论就会容易很多。

## 双语关键词

- ARC：自动引用计数
- ownership：所有权 / 所有关系
- lifetime：生命周期
- retain cycle：循环引用
- strong capture：强捕获
- weak capture：弱捕获
- `deinit`：析构阶段
- resource owner：资源所有者
- snapshot：快照
- borrow-like use：借用式使用
- cleanup：清理
- `defer`：延迟执行清理

## 常见错误

### 1. 以为 CLI 项目就没有 retain cycle 风险

只要有 class、闭包和长期任务，CLI/runtime 一样会出现生命周期环。没有界面不代表没有引用图。

### 2. 一味把所有对象都改成 class

如果核心任务模型也被引用化，你会同时放大并发、ARC、性能三类复杂度。能保持值语义的地方尽量保持。

### 3. 只记 `[weak self]`，不思考所有权

`weak self` 不是魔法符。真正重要的是：这段任务本该由谁拥有，旧任务何时取消，对象销毁时应如何收尾。

### 4. 忽视后台任务和资源的终止路径

如果没有明确 cancel / cleanup 设计，系统就可能在命令结束后仍保留无意义任务或临时资源。

## English Recap

This chapter grounds ARC and ownership in the project’s runtime evolution. The key idea is to keep task facts as values, keep resource managers as references, and reason explicitly about who owns background work, how closures extend lifetimes, and where cleanup belongs. Concurrency safety is incomplete without lifetime safety.

## Drills

1. 说明为什么 `TaskStore` 继续保持值类型，会让 Part 4 的并发和内存判断都更简单。
2. 找出一个你认为最容易在 runtime 升级里形成 retain cycle 的对象关系，并说明它为什么危险。
3. 用一句话解释 `deinit` 取消后台任务和在外部“记得调用 `stop()`”相比，哪种 ownership 更清楚。

## Project Handoff

现在项目已经有了并发边界和生命周期边界，但还缺少另一个程序员经常口头说、却很少系统建立的能力：性能判断。Swift 的值语义、ARC、Actor hop、数组复制、字符串拼接都会在运行时留下成本，只是规模小的时候不明显。下一章我们就回到 `TaskCore + TaskCLI`，用复制、测量和性能心智把这些成本看清楚。
