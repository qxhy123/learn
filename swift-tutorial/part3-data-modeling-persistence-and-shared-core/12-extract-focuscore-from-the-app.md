# 第 12 章：从应用里抽出 FocusCore

## 到了这里，抽模块才终于不是表演

很多教程一上来就爱谈“核心层”“领域层”，但在产品压力还没出现时，这通常只是换一种方式制造样板代码。`FocusList` 走到 Part 3，情况已经不同了：

- 任务、项目、计划已经形成稳定语义。
- 持久化和查询开始有边界。
- 错误和失败恢复不再只是局部页面细节。
- Part 4 还要引入测试和 CLI。

这时把共享规则从 App 中抽出来，才是真正有理由的。

## 先决定什么值得进入 FocusCore

`FocusCore` 适合承接的是那些满足下面两个条件的代码：

1. 它表达的是产品规则，而不是单页展示技巧。
2. App 与 CLI 都可能复用它。

所以你可以把下面这些东西移进去：

- `FocusTask`、`FocusProject`、`FocusPlan`
- 任务新增、完成、归档等核心操作
- 查询对象、排序规则、基础校验
- 部分仓储接口或服务协议

而下面这些通常不该进去：

- sheet 是否打开
- 当前 segment 选中了什么
- 某个页面的搜索框占位文案
- 某个平台独有的 toolbar 排版

## 给核心一个像样的公开接口

共享核心不是“把文件挪走”。它需要有清楚的 API 表面：

```swift
public final class FocusStore {
    public private(set) var inboxTasks: [FocusTask]
    public private(set) var projects: [FocusProject]

    public func addTask(_ draft: TaskDraft) throws
    public func updateTask(id: UUID, using draft: TaskDraft) throws
    public func toggleCompletion(_ id: UUID)
    public func tasks(matching query: TaskQuery) -> [FocusTask]
}
```

这里最重要的不是方法个数，而是风格一致：

- 让核心回答“系统怎么变化”。
- 让 UI 决定“怎样让用户触发这些变化”。

## 抽取的顺序很关键

不要一口气把所有东西搬过去。更稳的顺序通常是：

1. 先抽纯模型与纯规则。
2. 再抽不会依赖 UI 的查询逻辑。
3. 最后再决定哪些存储接口也应该由核心暴露。

每移动一步，都重新问一次：这段代码是否仍然不依赖 `SwiftUI`？
只要答案是否定的，它大概率还不该进入共享核心。

## 用 `focusctl` 反向验证核心边界

一个很好用的自检方法是：如果这段规则真的属于核心，CLI 应该也能复用。例如：

```swift
import FocusCore

let store = FocusStore.sample()
for task in store.tasks(matching: TaskQuery(includeCompleted: true)) {
    print(task.title)
}
```

如果你发现 CLI 为了拿到任务列表，不得不依赖某个 SwiftUI 视图或页面私有状态，那就说明你还没真正把共享规则抽干净。

## 一次抽取后的检查

做完这一章后，至少检查三件事：

1. `FocusCore` 不依赖 `SwiftUI`。
2. `FocusListApp` 依赖 `FocusCore`，但反向依赖不存在。
3. 你能指出三条“现在已经可以被 App 和 CLI 共用”的规则。

只有这三点都成立，抽模块才算真完成。

## 本章小结

`FocusCore` 在这里出现，不是因为教程突然想讲架构，而是因为产品复杂度已经证明它有必要存在。真正高级的判断，不是“什么时候能抽模块”，而是“什么时候抽模块刚刚好”。
