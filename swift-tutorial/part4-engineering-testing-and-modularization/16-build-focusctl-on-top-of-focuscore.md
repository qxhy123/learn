# 第 16 章：在 FocusCore 之上构建 focusctl

## CLI 在这套教程里是一面镜子

`focusctl` 不是为了把课程变成“双主线产品”，它的任务更直接：验证 `FocusCore` 里的东西到底是不是真的共享。如果共享核心只是把 UI 的实现细节挪了个目录，CLI 很快就会暴露这个问题。

starter 里的 CLI 目前非常小：

```swift
import FocusCore

let store = FocusStore.sample()
print("FocusList inbox")
for task in store.inboxTasks {
    let mark = task.isDone ? "[x]" : "[ ]"
    print("\\(mark) \\(task.title)")
}
```

这段代码短得几乎像示例，但它已经说明一件事：只要核心边界清楚，不依赖 SwiftUI 的第二表面就能自然长出来。

## 先给 CLI 一个最小命令面

与其一下子做复杂参数解析，不如先做三个稳定命令：

- `focusctl list`
- `focusctl add "Write release notes"`
- `focusctl complete <id>`

一个朴素入口就足够开始：

```swift
enum Command {
    case list
    case add(String)
    case complete(UUID)
}
```

重点不在于命令行框架，而在于命令执行阶段不要自己重写业务逻辑。CLI 应该把用户意图翻译成 `FocusCore` 能理解的调用：

```swift
switch command {
case .list:
    for task in store.tasks(matching: TaskQuery()) {
        print(task.title)
    }
case .add(let title):
    try store.addTask(TaskDraft(title: title))
case .complete(let id):
    store.toggleCompletion(id)
}
```

## 为什么 CLI 对边界如此敏感

如果你在实现 `focusctl` 时发现自己需要：

- 导入 `SwiftUI`
- 复制一套筛选逻辑
- 重新写一份任务校验规则

那基本说明共享核心还不够共享。CLI 的价值恰好在这里，它会逼你把边界里那些“看起来已经抽好了，其实只是换地方”的问题暴露出来。

## CLI 也能成为验证面

CLI 不是只有工程师自己用，它还是一条极好的快速验证路径。比如：

```bash
swift run focusctl list
swift run focusctl add "Review accessibility labels"
swift run focusctl complete 9B1C1A7D-...
```

这些命令让你在不打开 UI 的情况下也能快速检查核心规则是不是仍然成立。到了后面的发布准备阶段，这会非常有价值。

## 做完这一章后的检查点

至少确认：

1. CLI 没有复制 App 中的业务逻辑。
2. CLI 的命令执行完全建立在 `FocusCore` 上。
3. 你能指出至少两条“App 和 CLI 现在真正共享的规则”。

只要这三点站稳，`focusctl` 就已经完成了它在教程里的主要使命。

## 本章小结

`focusctl` 的作用不是让课程显得更大，而是让共享核心接受真正的复用检验。只要 CLI 能自然站在 `FocusCore` 之上，你的模块边界通常就已经开始可靠了。
