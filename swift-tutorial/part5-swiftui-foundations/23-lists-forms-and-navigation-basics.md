# 第23章：`List`、`Form` 与导航基础

> 第21章解决了 SwiftUI 的视图心智，第22章把状态拥有关系理顺了。可 `TaskFlow` 仍然只是在若干局部 View 里“会显示、会编辑”而已。真正像一个 app client 的界面，需要把任务列表、录入入口和页面流转组织成稳定结构。于是 `List`、`Form` 和 navigation 现在才真正有落点。

## 为什么这一章现在出现

如果没有前两章，`List`、`Form`、`NavigationStack` 很容易被学成三组 API 清单：

- `List` 用来滚动展示
- `Form` 用来放输入控件
- `NavigationStack` 用来点一下跳下一页

但那样的学习方式会忽略最关键的问题：这些容器到底在承载什么任务领域语义？为什么列表是一种结构边界，表单是一种输入边界，导航是一种用户流边界？

对 `TaskFlow` 来说，这一章的目标不是“做出几个页面”，而是让任务管理领域第一次长成像 app 的交互骨架。

## 从一个较弱起点开始：所有内容挤在一个大 `VStack`

初学 SwiftUI 时，很多人会把整个界面都塞进一个大 `VStack`：

```swift
struct TaskHomeView: View {
    var tasks: [Task]
    @State private var draftTitle = ""

    var body: some View {
        VStack {
            Text("TaskFlow")
            TextField("New task", text: $draftTitle)
            Button("Add") { ... }

            ForEach(tasks) { task in
                TaskRowView(task: task)
            }
        }
    }
}
```

这不是“绝对错误”，但它很快会遇到几个现实问题：

- 列表滚动、编辑、删除、选择都没有自然容器
- 输入控件和展示控件混在一起，结构层次模糊
- 详情流转只能靠额外的 if/else patch

也就是说，它能展示内容，却还没形成 app 的交互结构。

## `List`：让任务集合进入真正的“集合界面”语境

`List` 的意义不只是“可滚动”。对任务领域更重要的是，它承认当前界面正在呈现一个有身份（identity）、可枚举、可交互的项目集合。

```swift
struct TaskListView: View {
    let tasks: [Task]

    var body: some View {
        List(tasks) { task in
            NavigationLink(value: task) {
                TaskRowView(task: task)
            }
        }
    }
}
```

相比手工 `VStack + ForEach`，`List` 在当前阶段带来的直觉升级是：

- 你开始把任务看成 app 中的一等集合对象
- 行级交互（selection、swipe actions、删除、移动）有了自然生长空间
- 导航与集合结构能更自然地对接

这和 CLI 线的差异也很清楚。CLI 的主结构是命令与输出；SwiftUI app 的主结构则是列表、输入、导航和状态反馈。

## `Form`：表单不是“若干控件”，而是输入契约

`Form` 初学时常被当成“把 `TextField`、`Toggle`、`Picker` 塞进去的地方”。更稳的理解是：**Form 表达的是一组需要一起解释、一起提交、一起校验的用户输入。**

对 `TaskFlow`，新增任务可以从简单输入框升级成更像输入契约的表单：

```swift
struct TaskEditorView: View {
    @Binding var draftTitle: String
    @Binding var draftPriority: TaskPriority
    let onSubmit: () -> Void

    var body: some View {
        Form {
            Section("Task") {
                TextField("Title", text: $draftTitle)
                Picker("Priority", selection: $draftPriority) {
                    Text("Low").tag(TaskPriority.low)
                    Text("Medium").tag(TaskPriority.medium)
                    Text("High").tag(TaskPriority.high)
                }
            }

            Section {
                Button("Create Task", action: onSubmit)
            }
        }
    }
}
```

即使当前共享核心还没正式引入 `TaskPriority`，这个例子也在传达一件更重要的事：表单是输入边界，边界外层应决定如何校验、如何转成领域意图，而不是让每个控件各自偷偷改系统状态。

## 基础导航：让用户流不再依赖局部 patch

任务系统一旦进入 app 语境，就很自然会出现至少两类页面：

- 集合页：列表、筛选、概览
- 详情页：单个任务的信息与操作

如果没有稳定导航结构，你很容易退回到局部布尔值切换：

```swift
if selectedTask != nil {
    TaskDetailView(task: selectedTask!)
}
```

更强的方向是让导航本身成为 app 结构的一部分：

```swift
struct TaskHomeScreen: View {
    let tasks: [Task]

    var body: some View {
        NavigationStack {
            TaskListView(tasks: tasks)
                .navigationTitle("TaskFlow")
                .navigationDestination(for: Task.self) { task in
                    TaskDetailView(task: task)
                }
        }
    }
}
```

这样写的收益不只是在“能跳转”，而在于：

- 用户流开始有清楚骨架
- 详情页成为明确的目的地（destination）
- 列表行和详情页之间通过任务身份连接，而不是通过到处散落的布尔开关

## 详情页也应继续复用共享核心模型

一到导航和详情，很多教程会突然开始重新造一套“给界面用”的临时数据模型。对本教程，这正是应该刻意避开的路线。

`TaskDetailView` 的意义，不是脱离共享领域去展示一些 UI 私有字段；它应继续围绕 `TaskCore` 里的 `Task` 来解释任务事实：

```swift
struct TaskDetailView: View {
    let task: Task

    var body: some View {
        Form {
            Section("Summary") {
                Text(task.title)
                Text(task.status.displayName)
            }

            Section("Actions") {
                Button("Mark Done") { ... }
                    .disabled(task.status == .done)
            }
        }
        .navigationTitle("Task Detail")
    }
}
```

这里使用 `Form` 而不是单纯 `VStack`，也在表达一个判断：详情页通常不只是信息展示，还承载“查看并做出一个或多个任务操作”的语义。

## 列表、表单、导航在 `TaskFlow v1` 中如何协同

到了这一章，`TaskFlow v1` 已经应该具备一个最小但完整的 app 流程：

1. 在首页看到任务列表
2. 使用局部输入或表单创建新任务
3. 点击某个任务进入详情页
4. 在详情页查看状态，并进行基础动作

注意这里的重点不是 feature 数量，而是结构连续性。它表明 SwiftUI 客户端已经不只是若干孤立 View，而是成为了共享任务领域上的一条完整交互线。

## 双语关键词

- `List`：列表容器
- `Form`：表单容器
- `NavigationStack`：导航栈
- `NavigationLink`：导航链接
- destination：目的地视图
- selection：选择
- detail view：详情视图
- input contract：输入契约

## 常见错误

### 1. 把 `List` 当成“会滚动的 `VStack`”

`List` 的更大价值在于它表达集合界面语义，而不是只是多了滚动能力。

### 2. 在表单控件变化时立刻散射式修改系统状态

表单更适合收集、校验和提交一组输入，不应让每个控件都各自直接改共享领域状态。

### 3. 用一堆布尔开关手工拼导航

局部 patch 式导航在页面稍多时会迅速失控。应尽早让导航结构化。

### 4. 进入详情页后重新定义一套 UI 专用任务模型

详情页仍然是共享任务领域的客户端，不是另一个平行数据宇宙。

## English Recap

`List`, `Form`, and navigation are not just UI containers; they define collection structure, input boundaries, and user flow. In `TaskFlow`, they turn isolated SwiftUI views into a coherent app client that still reuses `TaskCore` models instead of inventing a separate UI-only task system.

## Drills

1. 说明为什么 `TaskListView` 进入 `List` 语境后，比 `VStack + ForEach` 更适合承载 app 级任务集合。
2. 假设“新增任务”需要标题和截止日期，解释为什么 `Form` 比零散控件更适合承载这组输入。
3. 画出一个最小 `TaskFlow` 用户流：列表页、详情页、返回路径，说明各自消费哪些状态。

## Project Handoff

到这里，`TaskFlow` 的 UI 骨架已经站住：列表、输入、导航都有了明确角色。下一章要把这些基础拼成 Part 5 的项目里程碑，也就是 `TaskFlow v1`：一个明确复用共享核心、但仍然保持描述性文档边界的 SwiftUI 客户端版本。
