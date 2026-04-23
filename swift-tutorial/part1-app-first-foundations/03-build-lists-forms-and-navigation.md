# 第 3 章：搭建列表、表单与基础导航

## 现在产品还缺什么

前两章让你理解了应用入口和状态，但一个任务应用至少还要具备三种用户能力：

- 看见一组任务
- 输入和修改信息
- 知道自己当前在哪个页面结构里

这就是 `List`、`Form` 和导航容器要解决的事。

## 列表：产品不是一块静态文本

打开 `InboxView`，最重要的部分是这个 `List`：

```swift
List(store.inboxTasks) { task in
    Button {
        store.toggleCompletion(task.id)
    } label: {
        HStack {
            Image(systemName: task.isDone ? "checkmark.circle.fill" : "circle")
            Text(task.title)
        }
    }
    .buttonStyle(.plain)
}
```

这里你要关注的不是“List 的参数怎么写”，而是两件事情：

1. 界面在读共享状态 `store.inboxTasks`
2. 用户点击之后，不是直接改图标，而是通过 `store.toggleCompletion` 改状态

这就是 SwiftUI 最核心的节奏：**用户操作改变状态，状态变化再驱动界面更新。**

## 表单：输入不是随便放几个控件

虽然当前 `SettingsView` 还很简单，但它已经在训练你把输入组织成结构：

```swift
Form {
    Toggle("Show completed tasks", isOn: $showCompletedTasks)
    Toggle("Use dense layout", isOn: $useDenseLayout)
}
```

`Form` 的价值，不是因为它长得像设置页，而是因为它表达了一种用户心智：这里是一组可以被编辑、保存、取消或重置的输入项。

也就是说，当你后面做任务编辑和项目编辑时，应该优先思考“这是不是一个结构化输入场景”，而不是只想着“放几个控件进去就行”。

## 导航：产品骨架从这里开始

在 starter 里，根视图使用的是 `NavigationSplitView`：

```swift
NavigationSplitView {
    List {
        NavigationLink("Inbox") {
            InboxView(store: store)
        }
        NavigationLink("Projects") {
            ProjectsView(store: store)
        }
        NavigationLink("Settings") {
            SettingsView()
        }
    }
} detail: {
    InboxView(store: store)
}
```

为什么这里不直接塞一个 Tab 或一个大页面？

因为 `FocusList` 从第一天起就要被训练成一个有信息架构的产品。侧栏、列表和详情的关系越早建立，后面加入标签、分组、筛选和搜索时就越不容易塌。

## 跟着做一个最小扩展

给侧栏再加一个占位入口，例如：

```swift
NavigationLink("Today") {
    Text("Today's agenda will live here.")
}
```

然后重新构建，确认：

- 导航结构仍然清楚
- 详情区仍然有默认内容
- 新入口没有破坏原有页面关系

这个动作很简单，但它在训练你：**增加入口时，优先考虑产品骨架是否仍然稳定。**

## 本章最容易犯的错

### 错误 1：所有内容先堆到一个页面里

这样短期省事，长期会让产品结构完全不可解释。等功能多了之后，你只会得到一个“什么都能放一点”的大页面。

### 错误 2：把导航当成装饰

导航不是为了看起来完整，而是产品结构本身。如果入口和详情关系一开始就不清楚，后面越加功能越乱。

## 本章小结

现在的 `FocusList` 已经拥有了最小产品骨架：

- `List` 负责展示动态集合
- `Form` 负责表达结构化输入
- `NavigationSplitView` 负责建立稳定入口关系

Part 1 的最后一章，我们会把这些局部能力收束成第一个真正可用的 `FocusList v1`。
