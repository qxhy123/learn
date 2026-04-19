# 第25章：应用状态与数据流

> Part 5 让 `TaskFlow` 长出了 SwiftUI 客户端的基本形态，但“会显示、会导航”还不等于“架构站住了”。一旦 app 要持续运行、跨多个 screen 共享状态、承接异步更新与错误反馈，你就必须开始认真讨论应用状态（app state）和数据流（data flow）。这正是 Part 6 的起点。

## 为什么这一章现在出现

到了 `TaskFlow v1`，很多状态还可以局限在单个 screen 或局部 view 中。但只要继续扩展，很快就会遇到这些问题：

- 首页列表和详情页都需要看到同一份任务事实
- 创建、完成、刷新任务后，多个界面都可能要同步更新
- 加载中、错误提示、筛选条件、当前选择项开始在 app 内跨边界流动

如果没有更明确的数据流设计，SwiftUI app 很容易进入一种“哪里能改就在哪里改”的脆弱状态。界面看似响应迅速，实际却慢慢长出多个事实源和大量 patch 式同步。

## 从一个较弱起点开始：到处传值，到处顺手改

在小 demo 阶段，下面这种写法看起来很自然：

```swift
struct TaskHomeScreen: View {
    @State private var tasks: [Task] = []

    var body: some View {
        TaskListView(tasks: tasks)
        TaskComposerView { title in
            tasks.append(Task(id: tasks.count + 1, title: title, status: .todo))
        }
    }
}
```

它的问题并不是“不能工作”，而是它把 app state、领域规则和界面动作全挤在一个地方：

- `TaskHomeScreen` 同时负责状态拥有、任务创建、ID 生成和界面装配
- 新增任务直接绕开共享核心
- 一旦详情页、筛选页、同步状态也加入，这个 screen 会迅速膨胀

这和早期单文件 CLI 的问题很像。只是现在压力点从命令入口转移到了 app state。

## 更强的起点：先承认 app state 是一个边界

Part 6 的第一个重要判断是：**app state 不是“几个 `@State` 字段的总和”，而是一层需要被设计的边界。**

对 `TaskFlow`，你至少要开始区分：

- feature state：某个 screen 或 feature 自己关心的状态
- app-level coordination state：跨 screen 共享的状态与流程
- domain state：由共享核心定义的任务事实

例如，一个更成熟的组织方式会让列表功能拥有自己的 model，但该 model 背后依赖统一的 app/runtime 层：

```swift
@Observable
final class TaskFlowAppModel {
    var selectedFilter: TaskFilter = .all
    var taskList = TaskListModel(...)
    var selectedTaskID: Task.ID?
}
```

这里重点不是具体语法，而是你终于在承认：应用状态不是到处零散发生，而是需要边界和层次。

## 数据流应尽量保持单向可解释

SwiftUI 项目一旦开始长大，最珍贵的可维护性资产之一就是**单向、可解释的数据流**：

1. 用户在 View 中触发意图（intent）
2. model / coordinator 接收意图并调用共享核心或运行时
3. 共享核心产生新的领域事实
4. 新状态回流到 observable model
5. View 重新根据状态渲染

把这条链写成口号很容易，真正难的是抵抗“图省事直接在 View 里改”的诱惑。对 `TaskFlow`，一个比直接 `tasks.append(...)` 更稳的路径可能是：

```swift
TaskComposerView { title in
    Task {
        await model.createTask(title: title)
    }
}
```

而 `model.createTask(title:)` 再去调用 runtime / repository。这样做并不是为了多加一层，而是为了让数据流仍然能被追踪、测试和复用。

## App state 与 feature state 的边界不要混淆

不是所有状态都值得提升到 app 级。若一看到“共享状态”就把一切都塞到最顶层，系统也会变得笨重。

对 `TaskFlow`，一个实用判断是：

- 只影响局部输入体验的状态，留在 feature 或 View
- 会影响多个 screen 或需要跨流程保持一致的状态，才考虑提升

例如：

- “新增任务表单的当前草稿”通常是 feature-local
- “当前任务列表、同步状态、当前过滤器”更可能是 feature-shared 或 app-shared
- “当前选中的导航路径”则常常需要更高层协调

成熟的数据流不是把所有状态抬到顶层，而是让每类状态停在刚好够用的层级。

## 共享核心如何进入 app data flow

因为 `TaskFlow` 明确是共享核心 client，所以数据流里必须保留 `TaskCore` 的位置，而不是让 app state 层直接重写规则。

更稳的路径通常是：

- View 发出“创建任务”“完成任务”“刷新任务”的 UI 意图
- feature/app model 调用 runtime service
- runtime service 调用 `TaskCore` 规则或共享模型变更
- 结果再回到 model，更新 `tasks`, `error`, `isLoading` 等 app-facing state

这会让 CLI 线和 SwiftUI 线继续共享同一套任务事实。两条客户端可以长得不同，但不应各自偷偷拥有一份不同的领域现实。

## 失败与加载状态也是 app state 的一部分

很多初学者一说“状态”，首先想到的是内容数据本身，例如 `[Task]`。到了 Part 6，要把另一类状态也纳入同等重要的位置：

- `isLoading`
- `errorMessage`
- `lastRefreshDate`
- `isSyncing`

这些状态不一定属于领域模型，但它们属于 app 行为现实。如果不把它们清楚建模，你最终就会回到老路：这里 show 一个 spinner，那里弹一句错误文案，却没有地方能解释“系统当前到底处于什么运行态”。

## 双语关键词

- app state：应用状态
- data flow：数据流
- feature state：功能状态
- coordinator：协调者
- intent：意图
- single-direction flow：单向流动
- shared state：共享状态
- loading state：加载状态
- error state：错误状态

## 常见错误

### 1. 只把“内容数据”当状态，忽略加载与失败状态

没有运行态状态，app 表面会动，但行为不可解释。

### 2. 一看到共享，就把所有状态提升到最顶层

这会让 app model 变成巨型状态仓库，反而难以维护。

### 3. 在 View 中直接重写领域规则

View 应表达意图，不应成为 `TaskCore` 规则的平行实现地。

### 4. 数据流既可以从上往下，也可以从中间横向 patch

越是跨 screen 的 app，越要避免“随手就改”的横向状态污染。

## English Recap

Part 6 begins by treating app state as a real architectural boundary. `TaskFlow` needs clear separation between local feature state, shared app coordination state, and domain state from `TaskCore`, with a mostly one-way data flow from user intent to runtime to updated UI state.

## Drills

1. 把 `TaskFlow` 当前可能存在的状态分成三类：局部、feature 共享、app 级协调。
2. 用五步描述一次“创建任务”在 app 中的数据流。
3. 解释为什么 `isLoading` 和 `errorMessage` 不能被当成“只是显示细节”，而应被视为 app state。

## Project Handoff

现在我们终于把 `TaskFlow` 的状态从“若干 UI 局部变量”推进到了真正的 app data flow。下一章会继续处理一个更现实的问题：这些状态如何和持久化、共享模型以及 `TaskCore` 集成，让 app 不只是内存里的漂亮壳子。
