# 第29章：高级泛型与协议设计

> Part 2 已经让我们学过 protocol、generics、associated type，但那时重点是“看懂边界”和“学会用类型表达关系”。现在项目里已经同时存在 `TaskCLI Lite` 的早期直觉、`TaskCore + TaskCLI` 的工程边界、以及 `TaskFlow` 的客户端压力，我们终于可以讨论一个更成熟的问题：**什么时候抽象应该继续升级，什么时候应该停手。**

## 为什么这一章现在出现

到了 Part 7，系统里已经有了几类真实压力：

- `TaskCLI` 和 `TaskFlow` 都在消费共享核心，但它们消费方式不同
- Part 3 和 Part 4 让我们看到了 repository、runtime、renderer、parser 这些边界
- Part 6 又把 app model、preview、异步状态和客户端协调层带回了视野

这时如果还只停留在 Part 2 那种“会写一个带 `associatedtype` 的协议”层面，就不够了。现在真正要判断的是：

- 哪些协议值得继续做成协议族（protocol family）
- 哪些泛型关系值得保留，哪些该在边界上被抹平
- `some`、`any`、具体类型（concrete type）和 type erasure 该各自出现在什么位置

也就是说，这一章不是“再学一遍泛型”，而是把泛型与协议设计放回**真实系统的演进压力**里重新理解。

## 从一个较弱起点开始：边界越来越多，但抽象方式开始失控

很多 Swift 工程走到这个阶段，会出现一种表面高级、实际发虚的写法：

```swift
protocol TaskServiceProtocol {
    func list() async throws -> [Task]
    func add(title: String) async throws -> Task
}

protocol TaskListProviderProtocol {
    func list() async throws -> [Task]
}

protocol TaskCommandHandling {
    associatedtype Output
    func handle(_ command: TaskCommand) async throws -> Output
}

struct AnyTaskThing {
    let value: Any
}
```

这里的问题不是“协议数量太多”本身，而是抽象已经开始脱离真实边界：

- `TaskServiceProtocol` 和 `TaskListProviderProtocol` 很可能只是名字不同、职责重复
- `AnyTaskThing` 这种容器没有表达任何可靠关系
- 读者看不出系统到底在保留什么类型信息，又在什么地方主动舍弃

这种弱状态很常见，因为到了高级阶段，人最容易被一种错觉带偏：**只要抽象层数增加，设计就更强。**

事实上，泛型与协议的真正价值从来不是“层数”，而是**边界的精度**。

## 更强的第一步：先把“共享能力”与“客户端表面”分开

对 `TaskCore + TaskCLI + TaskFlow` 这条主线来说，更强的做法通常不是“给所有东西都套 protocol”，而是先问：

- 这是不是跨客户端共享的领域能力
- 这是不是某个客户端自己的交互表面
- 这是不是运行时协调层的内部技术细节

例如，“能根据某种查询拿到任务快照”是一类共享能力；“CLI 把结果渲染成文本”“SwiftUI 把状态喂给 View”则是客户端表面。

这会自然导向一种更稳的协议设计：

```swift
protocol TaskQuerying {
    associatedtype Query
    associatedtype ResultSnapshot

    func fetch(_ query: Query) async throws -> ResultSnapshot
}
```

这个协议比“`TaskServiceProtocol` 大杂烩”强的地方在于，它把真正要保留的关系写清楚了：

- 某种查询能力对应某种输入 `Query`
- 该能力产出某种快照 `ResultSnapshot`
- 这组关系随具体实现而变化

此时 CLI 和 SwiftUI 都能复用“查询”这个抽象方向，但不必被迫共享一整套客户端细节。

## 协议族（protocol family）要围绕关系设计，而不是围绕名词设计

高级 Swift 工程里，一个很重要的升级是：你不再只写单个协议，而开始面对一组相互咬合的协议关系。

例如，对任务系统可以出现这样一组边界：

```swift
protocol TaskSnapshotSource {
    associatedtype Snapshot
    func snapshot() async throws -> Snapshot
}

protocol TaskMutationPerformer {
    associatedtype Mutation
    associatedtype MutationResult

    func perform(_ mutation: Mutation) async throws -> MutationResult
}
```

这时真正重要的，不是协议名字酷不酷，而是你有没有把它们设计成一组**有明确配合方式**的关系：

- `Snapshot` 是读取侧暴露的稳定事实
- `Mutation` 是写入侧接受的受控意图
- `MutationResult` 是变更后的返回信息，而不是随手丢一个 `Bool`

这类协议族的价值，在 `TaskCLI` 和 `TaskFlow` 同时存在时尤其明显：

- CLI 更关心命令转 mutation，再把结果转成文本
- SwiftUI 更关心 mutation 后的新 snapshot 如何回流到状态模型

共享核心无需知道 CLI 怎么排文本，也无需知道 SwiftUI 怎么做动画；它只需要维持这组共享关系稳定。

## `associatedtype` 的高级价值：让边界保留“正确的不同”

Part 2 讲 `associatedtype` 时，重点是“协议可以携带自己的类型关系”。Part 7 要往前走一步，看到它在系统设计里的判断价值：

**不是所有边界都应该统一成同一种输入输出类型。**

看一个弱版本：

```swift
protocol TaskBoundary {
    func call(_ payload: [String: Any]) async throws -> Any
}
```

这个接口当然“通用”，但它摧毁了两种本来值得保留的信息：

- 这条边界到底接收什么意图
- 这条边界到底承诺返回什么结果

更强的版本会承认不同边界携带不同关系：

```swift
protocol BoundaryCall {
    associatedtype Input
    associatedtype Output

    func call(_ input: Input) async throws -> Output
}
```

然后：

- `TaskCLICommandRunner` 可以是 `Input == TaskCommand`
- `TaskFlowActionPerformer` 可以是 `Input == TaskMutation`
- `TaskRepositoryAdapter` 可以是 `Input == PersistenceOperation`

这种设计的关键不是“更抽象”，而是它允许系统在共享结构时保留**正确的不同**。

## 什么时候用 `some`，什么时候用 `any`

到了高级 Swift，很多人会开始被 `some` 和 `any` 困住。对当前教程主线，一个实用判断可以非常直接：

- 你想保留具体实现能力和静态类型关系时，优先考虑 `some`
- 你想在运行时存放“任意符合该协议的值”时，才考虑 `any`

例如：

```swift
func makeTaskRepository() -> some TaskSnapshotSource {
    LiveTaskRepository(seed: .sample)
}
```

这类写法的优势是：调用方知道“我拿到的是某个遵循者”，但不必知道具体实现名；同时编译器仍能保留静态优化和具体类型关系。

而下面这种场景才更像 `any`：

```swift
struct TaskCLIEnvironment {
    let renderer: any TaskRendering
    let logger: any TaskLogging
}
```

因为这里的重点不是保留某个单一底层具体类型，而是把“某个可替换能力”存进运行时环境里。

这条判断很重要，因为许多所谓“高级 Swift 难题”其实都源于一个误区：**在需要保留关系时过早用 `any` 抹平，在只需要替换性时又硬把整个系统做成泛型。**

## type erasure 不是高阶炫技，而是边界修整工具

只要你认真使用带 `associatedtype` 的协议，迟早会碰到一个现实：某些地方确实需要把不同具体实现装进同一容器或环境里。这时 type erasure 才有意义。

例如：

```swift
struct AnyTaskSnapshotSource<Snapshot>: TaskSnapshotSource {
    private let _snapshot: () async throws -> Snapshot

    init<Source: TaskSnapshotSource>(_ source: Source)
    where Source.Snapshot == Snapshot {
        self._snapshot = source.snapshot
    }

    func snapshot() async throws -> Snapshot {
        try await _snapshot()
    }
}
```

这类包装的意义，不是为了显得“很懂 Swift”，而是因为你已经做了一个明确决定：

- 这里的调用方只需要 `Snapshot` 关系
- 具体底层实现可以被隐藏
- 这种隐藏发生在**边界处**，而不是让整个系统从一开始就失去类型信息

对 `TaskFlow` 来说，这尤其有价值。你可能希望 preview、test double、live repository 都能注入同一个 app model，但 model 又只需要“给我任务快照”这件事。那时，type erasure 就是为边界服务，而不是为技术秀服务。

## 高级泛型设计要服务“算法复用”，不是“框架感复用”

在 `TaskCore` 这条线里，真正值得泛型化的，经常不是完整服务对象，而是某些稳定算法。

例如，查询、过滤和排序逻辑可以通过泛型约束复用：

```swift
func grouped<Model: Identifiable, Key: Hashable>(
    _ values: [Model],
    by keyForValue: (Model) -> Key
) -> [Key: [Model]] {
    Dictionary(grouping: values, by: keyForValue)
}
```

对任务系统而言，它可以服务：

- CLI 里的分组输出
- `TaskFlow` 里的 sectioned list
- 将来的统计视图或导出逻辑

这比“先造一个 TaskFramework 基础层”更稳，因为它抽取的是**真正共享的结构能力**，而不是用一个巨型抽象壳把所有场景提前绑住。

## 高级协议设计也要会克制：并不是所有差异都值得抽象

现在必须反过来说一句：当你终于学会高级泛型和协议设计后，更大的风险反而是过度抽象。

对我们的主线项目，下面这些通常不值得在当前阶段抽出来：

- 仅被一个具体功能使用一次的协议
- 只是把一个 concrete type 改了个通用名字的“薄包装”
- 还没有第二个实现体就先做完整 type-erased 基础设施

例如，如果当前只有一个磁盘仓储实现，一个 live preview double，一个 test double，那么：

- 直接保留 `DiskTaskRepository`
- 在需要注入的地方用窄协议或 `some`
- 只有当某个边界真的需要统一存储时再加 erasure

这会比一上来就写 `AnyTaskRepositoryFactoryResolverRegistry` 强得多。

## 把三条项目线串起来看，高级抽象到底在改善什么

把本章放回整套教程主线，它真正改善的是三件事：

### 1. 对 `TaskCLI Lite` 的回看

我们终于能解释，为什么 Part 1 的大量具体代码当时是对的，因为那时抽象压力还不存在。

### 2. 对 `TaskCore + TaskCLI` 的加固

我们可以开始让共享读取、变更、快照和边界能力在类型层面更稳定，而不靠注释和习惯维持。

### 3. 对 `TaskFlow` 的支撑

SwiftUI 客户端需要可注入的查询和变更边界，但它不需要替共享核心继承一整套 UI 细节。高级泛型与协议设计正好帮我们把这条线守住。

所以本章最重要的结论不是“Swift 泛型很强”，而是：**高级抽象的目标是让共享核心与多个客户端之间的关系更清楚，而不是把系统写成泛型迷宫。**

## 双语关键词

- protocol family：协议族
- associated type：关联类型
- generic constraint：泛型约束
- concrete type：具体类型
- opaque type / `some`：不透明返回类型
- existential / `any`：存在类型
- type erasure：类型擦除
- abstraction density：抽象密度
- boundary precision：边界精度

## 常见错误

### 1. 把“高级”理解成“协议越多越高级”

协议只有在包住真实变化关系时才有价值。名字多不等于边界清楚。

### 2. 在需要保留关系时过早使用 `any`

一旦把关系抹平，后面经常只能靠文档和人工记忆补回来。

### 3. 在只需要替换性时把整个系统都泛型化

这会让调用面迅速变重，读者也更难看清真正重要的关系。

### 4. 还没有边界需求就提前做大规模 type erasure

type erasure 是边界修整工具，不是默认基础设施。

### 5. 抽象围绕名词分类，而不是围绕输入输出关系设计

高级协议最怕的不是不够通用，而是通用得没有语义。

## English Recap

Advanced generics and protocol design are valuable when they preserve the right relationships across shared core and multiple clients. Use protocol families and associated types to keep input-output contracts precise, prefer `some` when you want to preserve static relationships, use `any` only where runtime storage truly needs it, and introduce type erasure only at real boundary points.

## Drills

1. 回看你脑中的 `TaskRepository` 设计，判断它到底是在表达“读取快照”还是“读取加变更加协调”的混合职责。
2. 为 `TaskFlow` 设计一个最窄的可注入读取协议，并说明为什么它不该顺手包含 CLI 渲染能力。
3. 找一个你会下意识写成 `any` 的边界，改写成 `some` 或具体类型，再说明取舍理由。

## Project Handoff

现在我们已经把高级泛型和协议设计从“语法能力”推进到了“系统边界判断”。下一章会继续处理另一类很容易被神化的高级 Swift 表面：`result builder`、macro 以及更一般的 API surface judgment。重点仍然不是炫技，而是判断什么值得放进公共表面，什么不值得。
