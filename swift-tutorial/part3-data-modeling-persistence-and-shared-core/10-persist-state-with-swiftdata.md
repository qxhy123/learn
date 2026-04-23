# 第 10 章：使用 SwiftData 持久化状态

## 现在的核心变化是“数据拥有生命周期”

前两部分里，`FocusList` 可以靠内存数组推进学习，因为重点在应用骨架和产品结构。到了 Part 3，这个前提已经不够了。任务管理产品天然要求一件事：用户下次打开应用时，任务还在。

一旦你接受这一点，系统里就立刻出现两层状态：

- 当前正在驱动界面的内存状态。
- 需要跨会话保存和恢复的持久化状态。

这就是 `SwiftData` 要进入主线的原因。

## 第一步：先定义存储模型，而不是直接把 View 绑上去

你可以先建立一组和领域模型相邻、但职责不同的存储类型：

```swift
import SwiftData

@Model
final class StoredTask {
    @Attribute(.unique) var id: UUID
    var title: String
    var note: String
    var dueDate: Date?
    var isDone: Bool
    var projectID: UUID?
    var tags: [String]

    init(
        id: UUID = UUID(),
        title: String,
        note: String = "",
        dueDate: Date? = nil,
        isDone: Bool = false,
        projectID: UUID? = nil,
        tags: [String] = []
    ) {
        self.id = id
        self.title = title
        self.note = note
        self.dueDate = dueDate
        self.isDone = isDone
        self.projectID = projectID
        self.tags = tags
    }
}
```

这里的关键不是 `@Model` 注解，而是一个边界判断：`StoredTask` 代表“怎么存”，不一定等于“领域层怎么说话”。

## 第二步：明确领域模型和存储模型的转换

不要让所有页面都直接理解 `StoredTask`。更稳的做法是集中处理转换：

```swift
extension FocusTask {
    init(stored: StoredTask) {
        self.id = stored.id
        self.title = stored.title
        self.note = stored.note
        self.projectID = stored.projectID
        self.tags = stored.tags
        self.dueDate = stored.dueDate
        self.isDone = stored.isDone
    }
}
```

这样做的好处很直接：

- 页面继续使用领域语言，而不是存储语言。
- 你以后替换存储实现时，影响面更小。
- 测试可以更容易地围绕 `FocusTask` 和 `FocusStore` 展开。

## 第三步：给读取和保存安排明确位置

到了这里，别把 `@Query` 随手撒进每个页面。先决定谁拥有持久化流程。对这套教程来说，一个合理阶段目标通常是：

- `View` 发出用户意图，例如“新增任务”。
- 一个存储边界对象读写 `SwiftData`。
- 再把结果喂回 `FocusStore` 或共享核心。

如果你跳过这一层，页面很快就会同时承担：

- 读取
- 过滤
- 保存
- 失败处理
- 用户反馈

这会把整个产品重新拉回“页面替系统背锅”的状态。

## 最小读取长什么样

在某个专门承接持久化的边界对象里，你可能会写：

```swift
@Query(sort: \StoredTask.title) private var tasks: [StoredTask]
```

但真正重要的问题不是查询语法，而是：

- 这个排序是产品规则，还是页面当前需要？
- 为空时它代表正常空状态，还是恢复失败？
- 它拿到结果之后，是直接给 UI 用，还是先转回领域模型？

本章真正训练的，是这些判断。

## 先不要持久化什么

持久化很容易被做过头。下面这些通常不该在这一阶段写入长期存储：

- 正在输入中的标题草稿
- 当前 sheet 是否打开
- 当前页面选择的是哪个 segment
- 临时搜索词

这些都属于交互状态。它们离开当前会话后通常没有保留价值。

## 本章小结

`SwiftData` 的价值不是帮你“把数组存起来”，而是迫使你重新划分系统边界。只有当你能说清哪些数据要跨会话存在、谁负责把它们读回来、谁负责解释失败，持久化才算真的进入了产品主线。
