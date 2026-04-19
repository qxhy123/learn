# 第21章：SwiftUI 心智模型与 View Composition

> Part 4 已经把 `TaskCore + TaskCLI` 推进到更像真实 Swift 工程的状态：我们讨论了异步、Actor、ownership、性能和可靠性。现在终于进入 SwiftUI，但切换的不是“课程宇宙”，而只是客户端形态。`TaskFlow` 不是拿来替换共享核心的新 demo，而是站在 `TaskCore` 之上的第一个 UI client。

## 为什么这一章现在出现

很多人第一次接触 SwiftUI 时，学到的是一组看起来很快见效的表面动作：拖几个控件、拼几个 modifier、让预览里出现一张列表截图。这样当然会得到“我做出了界面”的即时反馈，但它也会把 SwiftUI 学成一种过度依赖结果图的表层经验。

这套教程现在才讲 SwiftUI，是因为前四部分已经把更重要的东西立住了：

- 你知道 `Task`、`TaskStatus`、`TaskStore` 这种领域模型为什么应该稳定
- 你知道模块边界和共享核心为什么存在
- 你知道状态变化、异步工作和失败面不该被随手塞进 UI 事件回调

因此这一章的任务，不是先去追“漂亮页面”，而是先把 SwiftUI 的运行方式看清楚：View 到底是什么，`body` 在表达什么，组合（composition）为什么比堆叠页面细节更重要。

## 从一个较弱的起点开始：把 SwiftUI 当成命令式 UI

如果你来自 UIKit、Android View、React 初学阶段，甚至桌面 GUI 工具包，你很容易带着一种旧直觉进入 SwiftUI：界面是一堆活着的控件对象，我要在某个时刻命令它们改变自己。

于是代码会不自觉地长成这种思路：

```swift
struct TaskListView: View {
    var tasks: [Task]

    var body: some View {
        var rows: [TaskRowView] = []

        for task in tasks {
            rows.append(TaskRowView(task: task))
        }

        return VStack {
            Text("TaskFlow")
            ForEach(rows.indices, id: \.self) { index in
                rows[index]
            }
        }
    }
}
```

这段代码的问题不在于“编译一定过不了”，而在于它反映了错误心智：

- 把 View 当成要手工构造和缓存的对象图
- 把 `body` 当成“搭建一次界面”的施工现场
- 把组合理解成“先拼好子控件数组，再塞回容器”

SwiftUI 更接近一种声明式（declarative）描述：给定当前状态，界面应该是什么。View 更像轻量的值描述（value-like description），而不是你要长期持有并手工驱动的控件实例。

## 更强的理解：`body` 是状态到界面的映射

对当前教程最重要的 SwiftUI 心智，可以先压缩成一句话：

**`body` 不是“把界面搭起来一次”，而是“在当前状态下声明界面应该长什么样”。**

如果 `TaskFlow` 要显示当前任务列表，那么界面描述应直接贴着任务状态来写：

```swift
struct TaskListView: View {
    let tasks: [Task]

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("TaskFlow")
                .font(.title.bold())

            ForEach(tasks) { task in
                TaskRowView(task: task)
            }
        }
        .padding()
    }
}
```

这里的重要变化不是语法简洁，而是语义更诚实：

- `tasks` 是输入数据
- `body` 根据输入描述输出界面
- `TaskRowView` 是组合单元，而不是手工缓存的子控件

这和前面 Part 2、Part 3 的思路是一致的。我们一直在强调“让边界表达真实职责”，而 SwiftUI 只是把这件事带进 UI 层。

## SwiftUI 的“刷新”不是你手动重绘，而是状态变了、描述重算

很多初学者会问：如果 View 只是描述，那界面更新到底发生在哪里？

答案不是“你去调用 `reloadData()`”，而是：当驱动 `body` 的状态变化时，SwiftUI 会重新求值（re-evaluate）相关描述，并计算需要更新的界面结果。

这就是为什么 `TaskFlow` 不应把业务规则写成“点击按钮后直接改几段 UI 文本”。真正应该变化的是任务状态本身，例如：

- 某个 `Task` 的 `status` 从 `.todo` 变成 `.done`
- 当前筛选条件从 `.all` 变成 `.openOnly`
- 输入表单里的草稿标题从空字符串变成用户刚输入的文本

当状态变化被明确建模，UI 才会自然跟着变。否则你就会回到命令式 patching：这里顺手改 label，那边顺手藏 row，过几周后谁也说不清系统的单一事实源（single source of truth）在哪。

## View Composition：不是“拆小一点”，而是“按语义拆边界”

一说组合，很多人会立刻想到“把大 View 拆成几个小 View”。这当然常常是对的，但真正重要的是：**你按什么边界来拆。**

对 `TaskFlow`，一个比“视觉块”更稳的拆法通常是“任务领域语义 + 交互职责”：

```swift
struct TaskRowView: View {
    let task: Task

    var body: some View {
        HStack {
            Image(systemName: task.status == .done ? "checkmark.circle.fill" : "circle")
            VStack(alignment: .leading) {
                Text(task.title)
                Text(task.status.displayName)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }
}
```

这样拆的收益有三层：

- UI 结构更容易读，因为 `TaskRowView` 就是在表达“一个任务行”
- 共享核心更容易复用，因为子 View 直接消费 `TaskCore` 里的稳定模型
- 后续状态流更容易加，因为交互边界已经开始变清楚

换句话说，组合不是“为了让文件短一点”，而是为了让界面结构和任务领域结构互相对齐。

## `some View`、modifier 与布局：先理解组合，不急着追 modifier 数量

SwiftUI 初学阶段还有一个常见误区：把学习重点放在 modifier 清单，好像知道更多 `.padding()`、`.background()`、`.toolbar()` 就算掌握框架。

这些 API 当然要会，但对当前阶段更关键的判断是：

- 哪些 View 是真正的语义单元
- 哪些 modifier 在表达布局、样式或交互
- 哪些数据应该作为输入传进来，而不是在 View 内部硬编码

例如对 `TaskFlow` 来说，下面这种写法虽然能跑，却是弱设计：

```swift
Text(task.title)
    .padding(8)
    .background(task.status == .done ? .green : .gray)
    .cornerRadius(12)
```

因为它把“任务状态如何呈现”直接散在局部样式里。更稳的方向通常是先形成一个小的语义单元，例如 `TaskStatusBadge`，再把样式集中表达。你会发现，这跟前几部分一直讲的抽象边界没有本质区别。

## `TaskFlow` 在 Part 5 的角色：共享核心之上的第一个 SwiftUI 客户端

到这里需要刻意强调一次：`TaskFlow` 不是“SwiftUI 版重新实现任务系统”。它是共享任务领域的一个新 client。

因此当前章节里的 View 设计，应默认建立在这些前提上：

- `Task`、`TaskStatus` 等基础模型来自共享核心，而不是 UI 自己重新定义一套
- SwiftUI 层负责呈现、输入采集和用户流转
- 领域规则仍应尽量落在 `TaskCore` 或后续明确的数据层边界上

这会让 Part 5 和前四部分自然接起来。你不是突然离开 `TaskCore + TaskCLI` 去学另一个课程，而是在同一个任务领域里新增一条 Apple/SwiftUI 专项客户端线。

## 双语关键词

- SwiftUI：Apple 的声明式 UI 框架
- declarative UI：声明式界面
- `View`：视图描述类型
- `body`：视图主体描述
- view composition：视图组合
- modifier：修饰器
- value-like description：值式描述
- single source of truth：单一事实源
- re-evaluate：重新求值

## 常见错误

### 1. 把 View 当成长期存活的控件对象

这样会让你不断想“什么时候手动更新它”。SwiftUI 更希望你关注输入状态，而不是手工驱动控件实例。

### 2. 在 `body` 里手工组装可变中间对象

`body` 应优先表达声明式结构。若总想着先创建数组、再 patch View，你多半还停在命令式心智。

### 3. 按截图区域拆 View，而不是按语义职责拆

截图块当然能帮助你看视觉布局，但真正稳的分解边界通常来自领域对象和交互职责。

### 4. 以为学 SwiftUI 就等于学 modifier 词典

modifier 很重要，但它们是组合表达的一部分，不是框架心智本身。

## English Recap

SwiftUI works best when you treat a `View` as a lightweight description of UI for the current state, not as a mutable widget object. In `TaskFlow`, view composition should follow task-domain meaning, reuse `TaskCore` models, and keep UI changes driven by state changes rather than imperative patching.

## Drills

1. 用自己的话解释：为什么 `body` 更像“状态到界面的映射”，而不是“只执行一次的搭建过程”？
2. 如果 `TaskFlow` 有“任务行”和“状态徽章”两个界面片段，你会按什么语义边界拆成子 View？
3. 说明为什么在 SwiftUI 客户端里复用 `TaskCore.Task`，比重新定义一个只给 UI 用的 `UITask` 更稳。

## Project Handoff

现在你已经有了 SwiftUI 的第一层心智：View 是描述，组合按语义切，`TaskFlow` 是共享核心之上的 client。下一章要解决的，就是让这些描述真正跟状态流连接起来：哪些状态留在 View 内，哪些通过 `Binding` 传递，哪些该交给可观察模型（observable model）承担。
