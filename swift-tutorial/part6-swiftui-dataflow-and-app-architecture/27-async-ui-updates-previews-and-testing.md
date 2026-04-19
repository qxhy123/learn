# 第27章：异步 UI 更新、预览与测试

> 第26章让 `TaskFlow` 开始接入真实数据边界后，SwiftUI app 就再也不是“同步假世界”了。加载任务、提交修改、失败恢复、刷新列表，都可能成为异步工作。于是本章的重点，不只是 `await` 在 View 里怎么写，而是：异步更新怎样让 UI 保持可解释，预览（preview）如何服务设计判断，测试又如何保护这些行为。

## 为什么这一章现在出现

如果没有上一章的持久化与模型集成，异步 UI 更新只会显得像一组零散技巧：`.task`、`refreshable`、`Task {}`、preview data、UI test。可一旦 app 真正通过 repository 或 runtime 加载任务，下面这些问题就全部变成现实：

- 页面第一次出现时如何触发加载
- 刷新期间 UI 应该显示什么
- 错误发生时如何避免界面处于半更新状态
- preview 怎样脱离真实存储仍能表达结构
- 测试如何验证状态流而不依赖截图叙事

因此本章出现的时机正好：数据边界已经有了，异步 UI 行为才不是空中楼阁。

## 从一个较弱起点开始：在 View 事件里零散发起异步工作

SwiftUI 初学者很容易写出这样的代码：

```swift
Button("Reload") {
    Task {
        tasks = try await repository.loadTasks()
    }
}
```

这段代码的弱点在于：

- 加载中状态没有被建模
- 错误路径被吞掉或只在局部打印
- View 直接依赖 repository，绕过了上层状态协调
- 同一个 screen 的首次加载、下拉刷新、重试可能各写一套异步逻辑

也就是说，它当然可以“把数据拉回来”，但它还没有形成稳定的异步 UI 契约。

## 更强的方向：异步更新先经过 model，再回到 View

对 `TaskFlow`，更稳的路径通常是：

```swift
struct TaskHomeScreen: View {
    @State private var model = TaskListModel(...)

    var body: some View {
        TaskListView(tasks: model.tasks)
            .overlay {
                if model.isLoading {
                    ProgressView("Loading tasks...")
                }
            }
            .task {
                await model.load()
            }
            .refreshable {
                await model.refresh()
            }
            .alert("Could not load tasks", isPresented: $model.hasError) {
                Button("OK", role: .cancel) {}
            } message: {
                Text(model.errorMessage ?? "")
            }
    }
}
```

这里的核心不是 API 数量，而是运行路径终于一致了：

- 首次出现时通过 `.task` 加载
- 用户主动刷新时通过 `.refreshable` 走同一条状态协调路径
- View 不直接决定存储逻辑，只消费 `isLoading`、`tasks`、`errorMessage`

异步更新因此进入了可解释的模型，而不是零散副作用集合。

## `.task`、`refreshable` 与用户意图的关系

很多教程会把 `.task` 和 `.onAppear` 简单并列。对本教程，更重要的是理解它们所承载的语义：

- `.task` 更像“当这个界面进入活跃显示语境时，应触发的一段异步工作”
- `.refreshable` 更像“用户明确发起刷新意图时，应执行的一段异步工作”

这两者虽然都可能调用 `model.load()`，但语义并不完全相同。区分语义的价值在于，你后面才能清楚回答：

- 首次加载失败后，重试是否走同一逻辑
- 下拉刷新时是否需要保留旧快照
- 某些屏幕是否只应首次加载一次

也就是说，异步 API 不只是“在哪里写 `await`”，而是在表达 UI 里的不同运行时事件。

## 预览（Preview）不是截图替代品，而是结构与状态检查器

SwiftUI 世界里，preview 常被误解成“快速看到界面长什么样”。这当然成立，但对工程写作来说，preview 的更大价值是：**用受控状态快速检查结构与边界是否合理。**

对 `TaskFlow`，好的 preview 不应依赖真实持久化环境，而应提供若干稳定状态样本：

```swift
#Preview("Loaded Tasks") {
    TaskHomeScreen(
        model: .previewLoaded
    )
}

#Preview("Empty State") {
    TaskHomeScreen(
        model: .previewEmpty
    )
}

#Preview("Load Failure") {
    TaskHomeScreen(
        model: .previewFailure
    )
}
```

这样的 preview 在教程中比截图驱动更有价值，因为它明确告诉读者：

- 当前 screen 依赖哪些状态
- 这些状态如何影响 UI 结构
- 哪些情况需要被设计，而不是等真机运行时“碰到了再说”

## 测试也应沿着状态流写，而不是只盯 UI 表象

SwiftUI 测试常常让初学者陷入两个极端：

- 要么完全不测，因为“UI 太难测”
- 要么只想做截图式或点击式 end-to-end 测试

对当前教程，更稳的入手点通常是状态协调层测试，也就是 observable model / repository interaction 测试。因为这层最能锁住真正重要的行为：

- `load()` 调用前后 `isLoading` 如何变化
- repository 成功返回后 `tasks` 是否更新
- repository 失败时 `errorMessage` 是否设置
- `createTask()` 后是否拿到新快照并清空提交态

你会发现，这跟 Part 3 在 CLI/Core 里优先锁核心行为而不是只盯输出字符串，是同一种测试哲学。

## 预览、假数据与测试替身（test double）之间的关系

为了让 preview 和测试都服务架构，而不是反过来绑架架构，一个很有价值的习惯是：让数据边界支持受控替身。

例如：

- preview repository：始终返回固定任务样本
- failing repository：始终抛出某个加载错误
- delayed repository：模拟异步等待

这类替身的意义，不是“为了方便造假”，而是为了让 UI 结构、状态流和错误路径都能被独立观察。对描述性项目文档来说，它也比“必须先搭一整套环境才能理解”更符合教程节奏。

## 双语关键词

- async UI update：异步界面更新
- `.task`：任务修饰器
- `refreshable`：可刷新修饰器
- preview：预览
- preview data：预览数据
- test double：测试替身
- loading indicator：加载指示
- error presentation：错误呈现

## 常见错误

### 1. 在 View 里随手 `Task {}` 调 repository，然后把状态更新散在各处

这会让异步逻辑难以追踪，也难以测试。

### 2. preview 直接依赖真实持久化或真实网络环境

这样 preview 很快失去“快速检查结构”的价值，变成脆弱集成入口。

### 3. 认为 SwiftUI 测试只能做截图或端到端点击

对当前阶段，更重要的是先把 model 层状态流锁住。

### 4. 加载、失败、刷新共用一团模糊状态

没有明确状态建模，UI 表面会更新，但运行语义仍然不清楚。

## English Recap

Async SwiftUI work should flow through state models, not be scattered across view event handlers. In `TaskFlow`, `.task`, `refreshable`, previews, and tests all become useful once they are tied to explicit loading, success, and failure states driven by reusable repositories or runtime adapters.

## Drills

1. 解释为什么 `.task { await model.load() }` 比 Button 内部直接改 `tasks` 更稳。
2. 设计三个 `TaskFlow` preview 状态，并说明它们各自验证什么结构判断。
3. 写出你会为 `TaskListModel.load()` 验证的两个关键测试断言。

## Project Handoff

现在 `TaskFlow` 已经不只是“有数据的 app”，而且开始具备异步更新、预览和测试入口。下一章将把这些部分收束成更完整的应用架构判断：当 feature 继续增长时，`TaskFlow` 应如何扩展而不失去与共享核心之间的清晰关系。
