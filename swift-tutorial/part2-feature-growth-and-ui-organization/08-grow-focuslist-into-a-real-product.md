# 第 8 章：把 FocusList 推进成真正的产品界面

## Part 2 的收尾不该只是“功能越来越多”

做到这一章时，你已经有了项目、标签、编辑流、搜索和筛选。如果教程现在直接跳去持久化或模块化，会留下一个巨大问题：这些能力到底有没有被组织成一个像样的产品？

第二部分的真正目标，是让 `FocusList` 从“会动的 demo”变成“用户一眼能看懂的工作界面”。

## 先用产品眼光重新审视入口

打开根视图，重新看一遍侧栏。如果它还是按“先写完哪个页面就先放哪个链接”的方式排列，就说明系统还没真正进入产品状态。一个更像产品的整理方式通常会有：

- 明确的一级工作区，例如 `Inbox`、`Today`
- 清楚的结构区，例如 `Projects`
- 辅助入口，例如 `Settings`

你可以先把侧栏整理成这种形态：

```swift
List(selection: $selection) {
    Section("Focus") {
        NavigationLink(value: Route.inbox) { Label("Inbox", systemImage: "tray") }
        NavigationLink(value: Route.today) { Label("Today", systemImage: "sun.max") }
    }

    Section("Projects") {
        ForEach(store.projects) { project in
            NavigationLink(value: Route.project(project.id)) {
                Label(project.name, systemImage: "folder")
            }
        }
    }

    Section("Manage") {
        NavigationLink(value: Route.tags) { Label("Tags", systemImage: "tag") }
        NavigationLink(value: Route.settings) { Label("Settings", systemImage: "gear") }
    }
}
```

这个重排的意义不在于 `Section` 或 `Label`，而在于你终于开始把入口按产品语义组织。

## 详情区必须解释“我当前正在看什么”

只有侧栏还不够。详情区也要开始承担上下文解释。一个可靠的做法，是让每个大视角都包含：

- 当前视角标题
- 结果数量或筛选状态摘要
- 主要操作入口
- 没有数据时的空状态说明

例如 `Inbox` 页面的顶部不该只是一张列表。它至少应该能回答：

- 我现在看的是所有未归档任务，还是筛选后的任务？
- 当前搜索词是什么？
- 用户下一步最常做的动作是什么？

一旦这些上下文缺席，产品就会看起来像“很多能点的控件”，而不是一个工作流。

## 用一个“空状态”暴露你的产品判断

空状态非常适合检查你是不是真的在做产品，而不是做页面拼装。假设 `Today` 没有任务，空状态可以这样写：

```swift
ContentUnavailableView(
    "No tasks for today",
    systemImage: "checkmark.circle",
    description: Text("Capture work in Inbox or assign a due date to see it here.")
)
```

这段内容的价值，不在于视图名字，而在于它把产品语义讲清楚了：

- 当前为什么为空。
- 用户接下来可以做什么。
- 这个页面和别的页面之间是什么关系。

## Part 2 结束时应该达到什么程度

你现在不需要一个满配任务管理器，但至少应该达到下面这个状态：

- 用户进入应用后，能分清一级入口。
- 任务信息不再只有标题，而开始携带结构上下文。
- 搜索、筛选和编辑流都能落在清楚的位置上。
- 空状态、默认详情和常用动作已经像同一个产品，而不是临时拼装。

如果你还做不到这些，先别急着进入持久化。因为 Part 3 会把复杂度再抬一层，结构不稳时只会雪上加霜。

## 一次收束检查

在进入下一部分之前，自己走一遍完整流程：

1. 启动应用，确认默认进入哪个主视角。
2. 新建一个任务并归入项目。
3. 用搜索或筛选找回它。
4. 切到另一个入口，再回到原视角。
5. 观察产品是否仍然像一个连续工作流。

如果第 5 步的答案是否定的，你需要继续整理产品界面，而不是继续加新技术。

## 本章小结

Part 2 的价值不是“把功能表写长”，而是让 `FocusList` 终于开始像一个产品。只有当产品入口、详情区和交互流都说同一种语言时，后面的模型、持久化和共享核心才会站在稳地基上。
