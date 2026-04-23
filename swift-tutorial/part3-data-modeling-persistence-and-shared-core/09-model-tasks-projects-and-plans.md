# 第 9 章：建模任务、项目与计划

## 模型不是字段清单，而是产品语义

Part 2 结束时，`FocusList` 已经像一个产品了，但它的模型还可能停留在“先让页面能跑”的水平。到了 Part 3，这已经不够。因为从现在开始，你要处理的不是单页交互，而是更长生命周期的数据：

- 用户关掉应用后还要再回来。
- 同一条任务会在 `Inbox`、`Today`、`Projects` 之间被不同方式阅读。
- 后续的 CLI 和测试也要理解同一套规则。

这意味着模型必须先稳住，否则持久化和共享核心只会把混乱搬进另一个目录。

## 先把三个概念真正分开

这一阶段至少要把任务、项目和计划建成不同语义：

```swift
struct FocusTask: Identifiable, Equatable, Sendable {
    let id: UUID
    var title: String
    var note: String
    var projectID: UUID?
    var tags: [String]
    var dueDate: Date?
    var planID: UUID?
    var isDone: Bool
}

struct FocusProject: Identifiable, Equatable, Sendable {
    let id: UUID
    var name: String
    var colorName: String
}

struct FocusPlan: Identifiable, Equatable, Sendable {
    let id: UUID
    var name: String
    var taskIDs: [UUID]
}
```

这里最重要的不是字段本身，而是关系：

- 任务是被完成、被推迟、被搜索的主体。
- 项目是稳定容器，用来表达“属于同一件事”的任务集合。
- 计划是工作视角，用来表达时间或行动安排，不是另一个项目。

如果你把计划也做成项目，或者把项目硬塞进标签，本章就还没做完。

## 用真实页面来验证模型是不是说得通

拿三个入口做检查最有效：

### `Inbox`

它应该能接住没有项目归属、也暂时没进计划的任务。
如果一条新任务必须先选项目才能存在，说明模型过早强制结构。

### `Projects`

它应该主要表达“哪些任务属于同一个工作容器”。
如果这里开始关心当前搜索词、面板展开状态、编辑弹窗开没开，说明 UI 状态正在污染模型。

### `Today`

它应该来自任务的到期信息或计划安排，而不是额外造一个“TodayTask”新类型。
如果一个视角一出现，你就想发明一种新任务结构，通常说明原模型边界还没想清。

## 不要让页面替模型背锅

当模型没站稳时，页面里会出现很多信号：

- 大量 `if mode == ...` 分支。
- 同一条任务在不同页面要被不同方式“解释”。
- 为了得到某个结果，先在 View 里拼很多临时过滤逻辑。

遇到这些症状，优先回模型，不要先怪 SwiftUI。因为这通常不是 UI 的问题，而是产品语义还没表达清楚。

## 一次建模演练

拿“把一条任务加入今天计划”这个动作来检验：

1. 这件事是修改任务本身，还是修改计划和任务之间的关系？
2. 它会不会改变项目归属？通常不会。
3. 如果用户把任务从 `Today` 移除，它是否应该从系统彻底消失？当然不应该。

只要这三问你答得清楚，说明任务、项目和计划的职责已经开始分开。

## 本章小结

Part 3 不是从 `SwiftData` 开始，而是从模型开始。只有当任务、项目和计划真正变成三种不同的产品语义时，后面的持久化、查询和共享核心才不会只是“把混乱序列化”。
