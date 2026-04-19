# 第26章：持久化与模型集成

> 第25章把 `TaskFlow` 的 app state 和数据流边界立起来了，但一个只会在内存里变化的 app 很快就会暴露教学型脆弱感：重开就丢，异步更新没有来源，客户端和共享核心也只停留在“概念上复用”。因此现在必须把持久化（persistence）和模型集成（model integration）拉进来。

## 为什么这一章现在出现

前面 Part 4 在 `TaskCore + TaskCLI` 里已经认真讨论过 repository、runtime、失败面与可靠性。进入 `TaskFlow` 后，如果我们突然回到“View 自己追加数组，保存以后再说”，那前面的工程主线就被切断了。

这一章现在出现，是为了把两条线重新接上：

- `TaskCore` 继续提供稳定领域模型与规则
- `TaskFlow` 通过面向 app 的数据边界接入这些能力
- 持久化不再是“以后补”的空白，而是 app architecture 的组成部分

## 从一个较弱起点开始：UI 自己就是存储层

SwiftUI 初学阶段很容易写出这种逻辑：

```swift
@Observable
final class TaskListModel {
    var tasks: [Task] = []

    func createTask(title: String) {
        let task = Task(id: tasks.count + 1, title: title, status: .todo)
        tasks.append(task)
    }
}
```

这段代码的弱点很多：

- 任务创建规则直接出现在 app model 中
- ID 策略与共享核心脱节
- 没有加载来源，也没有保存出口
- 一旦加入失败处理或并发刷新，整个 model 会迅速变脆

它和早期 CLI 里“所有事都写在 `main.swift`”是同一种问题，只是换了界面外壳。

## 更强的方向：让 app model 依赖持久化边界，而不是自己变成持久化边界

对 `TaskFlow` 更稳的结构，通常是把持久化和核心集成封装进独立的数据层，例如 repository / runtime / service：

```swift
protocol TaskFlowRepository {
    func loadTasks() async throws -> [Task]
    func createTask(title: String) async throws -> [Task]
    func markDone(id: Task.ID) async throws -> [Task]
}
```

app model 再围绕这个边界组织 UI 状态：

```swift
@Observable
final class TaskListModel {
    private let repository: TaskFlowRepository

    private(set) var tasks: [Task] = []
    private(set) var isLoading = false
    private(set) var errorMessage: String?

    init(repository: TaskFlowRepository) {
        self.repository = repository
    }

    func load() async {
        isLoading = true
        defer { isLoading = false }

        do {
            tasks = try await repository.loadTasks()
            errorMessage = nil
        } catch {
            errorMessage = "Could not load tasks."
        }
    }
}
```

此时 model 终于回到了更合适的职责：面向 UI 暴露状态，而不是自己变成业务规则和存储规则的混合体。

## `TaskFlow` 复用 `TaskCore` 的方式，不是“引用名字”，而是复用规则与模型含义

说“TaskFlow 复用 `TaskCore`”很容易，说清楚怎么复用更重要。

一种较稳的复用方式是：

- `TaskCore` 继续定义 `Task`、`TaskStatus`、`TaskStore` 等共享模型
- 持久化层负责 load / save raw data，并调用核心规则完成变更
- `TaskFlowRepository` 返回给 UI 的仍然是共享核心里的 `Task`

换句话说，app 端不是只“import 了一个类型名”，而是真正站在共享核心的规则语义上。创建任务、完成任务、过滤任务的正确性，不应由 UI 自行定义。

## 持久化集成时要带着 Part 4 的判断回来

进入 SwiftUI 后，一个很大的风险是教程风格突然倒退：仿佛 app 端因为是 UI，所以可以不再认真面对 Part 4 讲过的事情。

实际上本章正是要把那些判断带回来：

- `load` / `save` 是异步边界，应诚实地体现在接口上
- 失败面要区分加载失败、保存失败、领域失败
- 状态提交与 UI 成功提示之间应有清楚契约
- 共享可变状态若存在，应通过明确隔离边界协调

这会让 `TaskFlow` 的 app line 与 CLI line 真正接上。否则所谓“复用共享核心”就只剩名义上的目录关系。

## 让 app model 接受 snapshot，而不是持有裸露可变内部状态

前几部分一直强调 snapshot 和边界，这里同样适用。对 UI 来说，最稳的通常不是拿到一个还能继续随意改写的内部存储对象，而是拿到当前任务事实的稳定快照：

- repository / runtime 内部可以维护自己的协调逻辑
- UI 看到的是 `[Task]` 或其他共享模型快照
- 若要变更，UI 重新发出意图，经数据层处理后拿回新快照

这样做的好处，是把“谁可以改系统状态”控制在更少的地方。SwiftUI View 与 observable model 因此更像状态消费者，而不是随时篡改核心内部状态的参与者。

## 持久化不是只关乎“能不能存”，还关乎 app 启动与恢复体验

当 `TaskFlow` 接入持久化后，app 生命周期中的一些关键节点就开始变得真实：

- 首次启动时如何加载初始任务
- 前台/后台切换时是否需要刷新
- 失败后是回退到旧快照，还是显示空态并提示错误

这些问题虽然看起来更像产品层，但本质上仍是架构问题。因为它们在考验：你的数据边界是否足够清楚，状态恢复语义是否足够稳定。

## 双语关键词

- persistence：持久化
- repository：仓储 / 数据访问边界
- integration：集成
- snapshot：快照
- load / save：加载 / 保存
- storage boundary：存储边界
- domain model：领域模型
- recovery path：恢复路径

## 常见错误

### 1. 让 observable model 直接承担 ID 生成、领域规则和持久化

这会让 UI 协调层变成业务与存储大杂烩。

### 2. 说“复用共享核心”，实际却只复用了类型名

真正的复用应包括规则语义和边界，而不是只把模型复制到 app 层继续手写逻辑。

### 3. 因为是 SwiftUI，就把 Part 4 的异步和失败判断都忘掉

UI client 更需要这些判断，因为用户会直接感受到加载、失败与恢复体验。

### 4. 把 UI 直接绑到可随意变动的内部可变状态上

更稳的做法通常是通过 snapshot 与明确命令边界交流。

## English Recap

Persistence in `TaskFlow` should be introduced through a clear data boundary, not by turning the UI model into a storage engine. The SwiftUI app still reuses `TaskCore` for domain meaning, while repositories or runtimes handle async loading, saving, and state coordination before UI models expose snapshots to views.

## Drills

1. 说明为什么 `TaskListModel` 不应自己生成任务 ID 并直接追加数组。
2. 画出一条“app 启动 -> 加载持久化数据 -> 更新 UI”路径，标出共享核心出现的位置。
3. 如果保存失败，你会把错误归入哪一层？UI 层应该如何知道这件事？

## Project Handoff

现在 `TaskFlow` 不再只是消费内存中的任务数组，而开始有资格谈真正的 app 数据集成。下一章继续往前走：一旦加载、保存和刷新都变成异步工作，SwiftUI 如何更新界面、如何做预览、又如何为这些行为建立基础测试。
