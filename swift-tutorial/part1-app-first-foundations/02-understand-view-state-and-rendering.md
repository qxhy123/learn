# 第 2 章：理解 View、状态与重新渲染

## 这一章为什么关键

SwiftUI 最容易让人误解的地方，不是某个控件怎么写，而是 `View` 到底是什么。很多人会把 `View` 想成一个会一直活着的小对象，里面保存了大量内部状态。这样理解，后面几乎一定会把数据放错地方。

本章要做的，是把这件事讲清楚：**`View` 是对界面的描述，界面变化来自状态变化，而不是你在某个对象里“手动改 UI”。**

## 先看 starter 里的真实例子

打开 `InboxView.swift`：

```swift
import SwiftUI
import FocusCore

struct InboxView: View {
    @Bindable var store: FocusStore
    @State private var draftTitle = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                TextField("Add a task", text: $draftTitle)
                Button("Add") {
                    store.addTask(title: draftTitle)
                    draftTitle = ""
                }
            }

            List(store.inboxTasks) { task in
                Button {
                    store.toggleCompletion(task.id)
                } label: {
                    HStack {
                        Image(systemName: task.isDone ? "checkmark.circle.fill" : "circle")
                        Text(task.title)
                    }
                }
            }
        }
    }
}
```

这里同时出现了两类状态：

- `draftTitle`：当前输入框里的草稿文本
- `store`：整个产品共享的任务状态

这两者的区别，就是本章最重要的判断训练。

## `@State` 该放什么

`@State` 适合放当前 View 自己拥有、别人不需要共享、生命周期跟着当前界面走的状态。

在 `InboxView` 里，`draftTitle` 就是典型例子：

- 它只服务于当前输入框
- 任务成功提交后它就应该被清空
- 其他页面根本不需要知道它现在写了什么

如果你把这类状态也塞进 `FocusStore`，后面产品会很快变脏：共享模型会开始携带大量只属于局部编辑流的临时信息。

## `@Bindable` 又是在干什么

`@Bindable` 出现的前提，是 `FocusStore` 本身已经是一个可观察模型。它的作用，是让 `View` 能自然地读取、修改这个共享模型中的数据，而不用自己手写一堆中转逻辑。

可以先把它理解成一句话：

**`@Bindable` 让 `View` 能站在共享产品状态旁边工作，而不是把共享状态复制进来。**

## 自己做一个状态判断题

现在问你三个状态，它们应该放哪？

1. 输入框当前草稿文本
2. 所有任务数组
3. 当前用户是否打开“显示已完成任务”开关

建议答案是：

- 1 放在局部 `@State`
- 2 放在 `FocusStore`
- 3 取决于它是页面局部偏好，还是多个页面共享的产品设置

也就是说，状态设计不是背包装器名字，而是先判断所有权。

## 一个很小但很有价值的实验

把 `draftTitle` 改成下面这样：

```swift
@State private var draftTitle = "Write a better Part 1"
```

然后重新构建，观察输入框默认值。
再点击 `Add`，看任务是否新增且输入框是否清空。

这个实验在训练你确认两件事：

- 局部状态确实在驱动当前界面
- 共享状态变化后，列表会跟着刷新

## 本章最容易犯的错

### 错误 1：所有东西都往 `@State` 里塞

这样做短期看起来很方便，但一旦多个页面都需要同一份数据，你就会开始复制状态、写同步逻辑，最后把数据流搞得很难解释。

### 错误 2：局部草稿状态也急着进共享核心

这和上一个错误正好相反，但一样糟糕。共享核心应该承接稳定规则，而不是替页面保存每一次输入框击键。

## 本章小结

做完这一章后，你应该已经能说清：

- `View` 更像描述而不是“永久活着的 UI 对象”
- 什么状态适合 `@State`
- 什么状态更适合留在 `FocusStore`

如果这三点还不稳，Part 2 的产品化和 Part 3 的共享核心都会变得很容易跑偏。
