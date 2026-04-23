# 第 11 章：设计查询、存储边界与失败路径

## 有了持久化之后，难点才刚开始

前一章让数据可以长期存在，但这只是起点。真正困难的问题马上会出现：

- 查询逻辑写在哪里？
- 排序和过滤是页面责任还是共享责任？
- 保存失败、读取失败、坏数据恢复失败分别由谁解释？

如果这些问题都让 `View` 自己处理，`FocusList` 很快又会退化成一堆局部补丁。所以这一章要给系统补上真正的边界。

## 先定义一个存储接口，而不是让页面直接碰底层

哪怕当前实现只有 `SwiftData`，也先把页面挡在外面：

```swift
protocol TaskRepository {
    func loadInboxTasks() throws -> [FocusTask]
    func loadTasks(for projectID: UUID) throws -> [FocusTask]
    func save(_ task: FocusTask) throws
    func delete(_ id: UUID) throws
}
```

这个协议的价值有三层：

1. 它把“页面需要什么结果”说清楚。
2. 它把底层实现细节关进边界里。
3. 它给测试和 CLI 留出了稳定接口。

一旦你有了这层协议，查询逻辑和失败处理都开始有真正归位的地方。

## 失败先翻译成产品语言

不要把错误永远停留在技术术语。先定义产品能理解的失败：

```swift
enum TaskStoreError: LocalizedError {
    case loadFailed
    case saveFailed
    case corruptedData

    var errorDescription: String? {
        switch self {
        case .loadFailed:
            "Could not load tasks."
        case .saveFailed:
            "Could not save your changes."
        case .corruptedData:
            "Stored task data is invalid."
        }
    }
}
```

这里最重要的不是字符串，而是分类。取消、空状态、网络中断、坏数据都不该被塞进同一个“出错了”。

## 查询边界的职责是什么

在 `FocusList` 里，查询边界至少要负责三件事：

- 把底层数据读成领域模型。
- 统一过滤、排序和分页规则。
- 把失败翻译成上层可以处理的错误。

页面的职责则应该更单纯：

- 提供用户意图。
- 持有局部交互状态。
- 呈现当前查询结果、空状态和错误提示。

如果你发现页面里已经出现长串 `filter -> sort -> map -> catch -> alert`，基本说明边界还没立好。

## 用一个查询对象避免“条件四处飘”

当筛选条件变多时，别把参数散成一堆布尔值。可以先把查询意图收成一个对象：

```swift
struct TaskQuery: Equatable, Sendable {
    var projectID: UUID?
    var searchText: String = ""
    var includeCompleted = true
    var dueTodayOnly = false
}
```

这样页面能表达“我要什么结果”，而存储边界负责解释“怎么得到结果”。这正是你后面做共享核心时最需要的能力。

## 一次失败路径演练

拿“保存编辑后的任务”来做检查：

1. 用户点击保存。
2. 页面把 `TaskDraft` 转回 `FocusTask`。
3. 仓储尝试写入。
4. 如果失败，上抛 `TaskStoreError.saveFailed`。
5. 页面保留草稿，并提示用户重试。

这条路径里，页面负责交互和反馈，边界负责写入和失败翻译。只要职责混掉，你就会得到“用户都不知道到底有没有保存成功”的产品体验。

## 本章小结

持久化之后，真正的工程价值来自清楚边界。查询不是页面的私活，错误也不该只活在日志里。只有把查询、存储和失败各自放回正确位置，`FocusCore` 才有理由在下一章登场。
