# 第 5 章：设计任务分组与标签

## 这一章解决的不是“多几个字段”

Part 1 的 `FocusList v1` 已经能新增任务、浏览任务、切换完成状态，但它仍然只有一条扁平路径：所有东西都落进 `Inbox`。只要任务数量开始增长，用户马上会遇到两个问题：

- 这些任务彼此是什么关系？
- 我怎么从“几十条任务”里快速找回当前关注点？

所以 Part 2 的第一步不应该是先画一个更花的界面，而是先给产品补上信息结构。你现在要让 `FocusList` 学会三种语义：任务、项目、标签。

## 先从现有 starter 找切口

当前 `FocusCore` 里已经有 `FocusTask` 和 `FocusProject`，但任务和项目之间还没有连接，标签也不存在。先别急着做复杂模型，先把当前产品真正需要的结构补进去：

```swift
public struct FocusTask: Identifiable, Equatable, Sendable {
    public let id: UUID
    public var title: String
    public var projectID: UUID?
    public var tags: [String]
    public var isDone: Bool

    public init(
        id: UUID = UUID(),
        title: String,
        projectID: UUID? = nil,
        tags: [String] = [],
        isDone: Bool = false
    ) {
        self.id = id
        self.title = title
        self.projectID = projectID
        self.tags = tags
        self.isDone = isDone
    }
}
```

这个模型只做了三件事：

1. 允许任务归属某个项目。
2. 允许任务拥有一组轻量标签。
3. 保持 `Inbox` 仍然能接住“还没归档到项目”的任务。

这三个决定已经足够支撑后面的筛选、搜索和持久化。不要在这里提前把颜色、排序规则、层级标签、子任务关系一次性全做完。

## 先做“看得见结构”的界面变化

建模之后，下一步不是立刻做复杂编辑器，而是让结构先在界面上可见。你至少应该完成两处改动：

1. 任务行能展示项目名或标签摘要。
2. 侧栏开始出现按项目或视角分组的入口。

例如，先让任务行把项目和标签显示出来：

```swift
struct TaskRow: View {
    let task: FocusTask
    let projectName: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(task.title)
            HStack(spacing: 6) {
                if let projectName {
                    Text(projectName)
                }
                ForEach(task.tags, id: \.self) { tag in
                    Text("#\(tag)")
                }
            }
            .font(.caption)
            .foregroundStyle(.secondary)
        }
    }
}
```

这个组件的价值不是“界面更丰富”，而是它逼你回答一个工程问题：项目名和标签信息应该由谁提供？一个稳妥的答案通常是，`TaskRow` 只负责显示，具体传什么数据由上层页面决定。

## 侧栏分组要表达产品语义

一条非常容易做对的规则是：先把入口按“用户组织注意力的方式”分组，而不是按“哪个页面先写出来”排列。

```swift
List {
    Section("Focus") {
        NavigationLink("Inbox") { InboxView(store: store) }
        NavigationLink("Today") { TodayView(store: store) }
    }

    Section("Projects") {
        ForEach(store.projects) { project in
            NavigationLink(project.name) {
                ProjectDetailView(store: store, project: project)
            }
        }
    }

    Section("Browse") {
        NavigationLink("All Tags") { TagBrowserView(store: store) }
        NavigationLink("Settings") { SettingsView() }
    }
}
```

现在的重点不是把 `TodayView` 或 `TagBrowserView` 一次性做完，而是建立一种稳定的产品地图：用户可以按任务视角、项目视角、标签视角进入系统。

## 一次手动检查，确认你不是在写提纲代码

做完这一章后，自己走一遍这条路径：

1. 创建一个带项目归属的任务。
2. 创建一个只有标签、没有项目的任务。
3. 在 `Inbox` 里确认两者都能被看懂。
4. 进入某个项目入口，确认只显示相关任务。
5. 回头问自己：如果去掉项目或标签，哪些界面马上会重新变得模糊？

如果你已经能回答最后一个问题，说明这章不是“又加两个字段”，而是真的让 `FocusList` 拥有了信息结构。

## 本章最容易犯的错

### 错误 1：把项目和标签做成同一种东西

项目是容器，标签是跨容器的横向语义。这两者混在一起，后面的过滤和持久化都会混乱。

### 错误 2：结构已经建好了，却不让用户看见

如果任务行和侧栏都看不出项目/标签的存在，这套结构对用户来说仍然是不存在的。

## 本章小结

Part 2 的起点不是“更多页面”，而是“更清楚的产品结构”。当任务、项目和标签三层语义站稳之后，后面的编辑流、筛选和搜索才有真实落点。
