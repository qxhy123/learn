# 第22章：状态、`Binding` 与 Observable Models

> 第21章先把 SwiftUI 的第一件大事讲清楚了：View 是状态的声明式描述，而不是可变控件集合。可如果只停在这里，你仍然只能写“长得像 SwiftUI”的静态界面。真正让 `TaskFlow` 活起来的，是状态流（state flow）本身：谁拥有状态，谁只借用状态，谁负责把用户输入变成领域变更。

## 为什么这一章现在出现

一旦你接受了“UI 由状态驱动”，下一个问题就会立刻冒出来：

- 草稿标题应该存在哪里？
- 列表筛选条件是谁在拥有？
- 某一行的勾选动作怎么安全地改回共享任务状态？
- `TaskFlow` 如何在不重写领域规则的前提下，把 UI 事件连接到 `TaskCore`

这就是为什么 `@State`、`Binding`、可观察模型（observable model）必须在此时出现。前一章解决的是“SwiftUI 视图长什么样”，这一章要解决的是“这些视图到底围绕什么状态在运转”。

## 从一个较弱起点开始：每层都各自保存一份数据

初学者最常见的 SwiftUI 状态错误，是很自然地在每一层都拷一份“自己要用的数据”：

```swift
struct TaskListView: View {
    let tasks: [Task]
    @State private var draftTitle = ""

    var body: some View {
        VStack {
            TaskComposerView(draftTitle: draftTitle)
            TaskRowsView(tasks: tasks)
        }
    }
}

struct TaskComposerView: View {
    @State var draftTitle: String

    var body: some View {
        TextField("New task", text: $draftTitle)
    }
}
```

问题在于，`TaskListView` 和 `TaskComposerView` 都在持有一份标题状态；子视图改的是自己的局部副本，不是父视图真正关心的那份状态。类似错误也会出现在：

- 任务列表在上层一份、详情页又复制一份
- 筛选条件在 toolbar 一份、列表又自己保留一份
- 领域模型在 observable object 一份、View 里又自己 cache 一份

这些错误的共同后果，是系统失去单一事实源，UI 表面还能动，但数据关系开始发散。

## `@State`：拥有局部、短生命周期的 View 状态

`@State` 最适合保存“这个 View 自己拥有、而且主要为了本地交互存在”的状态，例如：

- 输入框当前草稿
- 是否展开一个临时 section
- 当前选中的 tab
- 本地排序开关

对 `TaskFlow`，新增任务表单里的标题草稿就是典型例子：

```swift
struct TaskComposerView: View {
    @State private var draftTitle = ""
    let onSubmit: (String) -> Void

    var body: some View {
        HStack {
            TextField("New task title", text: $draftTitle)
            Button("Add") {
                let title = draftTitle.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !title.isEmpty else { return }
                onSubmit(title)
                draftTitle = ""
            }
        }
    }
}
```

这里 `draftTitle` 是局部 UI 状态，放在 `@State` 很合理；真正的任务新增规则则通过 `onSubmit` 往外传，让更高层决定如何调用共享核心。

## `Binding`：子视图借用父状态，而不是复制父状态

如果一个子 View 需要编辑父 View 持有的状态，那通常应该传 `Binding`，而不是再开一份 `@State`。

例如筛选器选择器：

```swift
enum TaskFilter: String, CaseIterable {
    case all
    case openOnly
    case doneOnly
}

struct TaskFilterPicker: View {
    @Binding var filter: TaskFilter

    var body: some View {
        Picker("Filter", selection: $filter) {
            Text("All").tag(TaskFilter.all)
            Text("Open").tag(TaskFilter.openOnly)
            Text("Done").tag(TaskFilter.doneOnly)
        }
        .pickerStyle(.segmented)
    }
}
```

父视图则拥有真实状态：

```swift
struct TaskListScreen: View {
    @State private var filter: TaskFilter = .all

    var body: some View {
        VStack {
            TaskFilterPicker(filter: $filter)
            TaskListView(tasks: filteredTasks)
        }
    }
}
```

这正是 `Binding` 的语义价值：子视图在“借用可写访问”，而不是偷偷持有第二份事实。

## Observable Model：当状态不再只是某个 View 的小局部

随着 `TaskFlow` 从静态列表推进到真正的 app client，状态会逐渐超出单个 View 的承受范围。比如：

- 任务列表来自共享核心或 repository
- 新增 / 完成任务会触发领域变更
- 加载中、失败中、刷新中这些状态要被多个 View 观察

这时就需要可观察模型（observable model）承担“屏幕级或功能级状态拥有者”的角色。现代 SwiftUI 里你会看到两类常见说法：

- `ObservableObject` / `@Published`：较早期、仍然常见的组合
- `@Observable` / `@Bindable`：Observation 框架下更现代的组合

本教程在概念上更看重的是职责，而不是站队某一个语法版本。无论你用哪种观察机制，都应守住一个判断：**observable model 是 UI state coordinator，不是把整个 `TaskCore` 规则搬进 View 层。**

例如：

```swift
@Observable
final class TaskListModel {
    private(set) var tasks: [Task] = []
    private(set) var isLoading = false
    private(set) var errorMessage: String?

    let runtime: TaskFlowRuntime

    init(runtime: TaskFlowRuntime) {
        self.runtime = runtime
    }

    func load() async {
        isLoading = true
        defer { isLoading = false }

        do {
            tasks = try await runtime.loadTasks()
            errorMessage = nil
        } catch {
            errorMessage = "Could not load tasks."
        }
    }
}
```

这里 model 管的是列表界面的状态和交互协调；领域规则仍应落在共享核心和明确的数据边界里。

## 从 `TaskCore` 到 `TaskFlow`：状态流不应切断共享领域

因为 `TaskFlow` 是共享核心 client，所以状态流设计必须刻意避免一个陷阱：为了迁就 UI 写法，重新定义一套“只在 app 里成立的任务规则”。

更稳的层次通常是：

- `TaskCore`：任务模型与核心规则
- `TaskFlowRuntime` 或数据服务层：面向 UI 的加载、保存、命令协调
- SwiftUI observable model：把 UI 关注的状态包装成可观察表面
- View：只消费状态并触发意图（intent）

这样做的好处很实际。CLI 线和 SwiftUI 线虽然客户端体验不同，但它们依赖的是同一条领域主线。用户新增任务，不应在 CLI 是一套规则、到 UI 又变成另一套规则。

## 局部状态、共享状态与派生状态要分清

一套能长期扩展的 SwiftUI app，通常至少要能区分三类状态：

- 局部状态（local state）：例如输入框草稿、sheet 开关
- 共享状态（shared feature/app state）：例如当前任务列表、同步状态、全局筛选
- 派生状态（derived state）：例如 `openTasks`, `doneTasks`, `isSubmitDisabled`

派生状态往往不该被单独存储一份，而应从现有事实推导出来。对 `TaskFlow` 来说，如果你已经有 `tasks` 和 `filter`，那么 `filteredTasks` 更像计算结果，而不是另一块独立可写状态。

这和前面讲值语义、模块边界时的思路完全一致：能推导出来的东西，不要额外制造一份真假难辨的缓存。

## 双语关键词

- state：状态
- `@State`：局部视图状态包装
- `Binding` / `@Binding`：绑定 / 借用可写访问
- observable model：可观察模型
- `ObservableObject`：可观察对象
- `@Observable`：Observation 宏
- derived state：派生状态
- single source of truth：单一事实源
- intent：用户意图

## 常见错误

### 1. 子视图需要编辑父状态时，再开一份 `@State`

这通常会立刻制造第二份事实源。应优先考虑 `Binding`。

### 2. 把所有状态都塞进一个巨型 model

observable model 应服务明确边界，不是把局部草稿、全局设置、导航、网络、缓存全部塞进一个 God object。

### 3. 把派生状态也持久保存成独立可写字段

如果 `filteredTasks` 可以由 `tasks + filter` 推导，就不要再维护第三份手工同步的数据。

### 4. 在 UI 层重新实现 `TaskCore` 规则

UI 协调可以在 app 层，但“标题是否合法”“任务如何完成”这类领域规则不应变成 View 特供逻辑。

## English Recap

`@State` is for local view-owned state, `Binding` lets child views edit parent-owned state without copying it, and observable models coordinate screen-level state that outgrows a single view. In `TaskFlow`, the key is to keep a single source of truth while still reusing `TaskCore` for real task rules.

## Drills

1. 列出 `TaskFlow` 中三个适合放在 `@State` 的状态，以及两个更适合放在 observable model 的状态。
2. 解释为什么筛选器子视图更适合接收 `Binding<TaskFilter>`，而不是 `TaskFilter` 加一个回调副本组合。
3. 画出你理解的状态流层次：`TaskCore`、运行时/服务层、observable model、View 分别负责什么。

## Project Handoff

现在 `TaskFlow` 已经不只是“会显示任务的界面”，而开始具备清楚的状态拥有关系。下一章要继续把这些状态放进更真实的容器中：列表（`List`）、表单（`Form`）和基础导航（navigation），并让任务领域的常见交互形成一条完整的 app 流程。
