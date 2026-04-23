# FocusList Starter：起始工程

## 这份 starter 的定位

它不是答案，也不是演示用成品，而是整套教程共同依赖的一份可验证基线。你应该把它理解成：Part 1 开始时，你自己已经拿到的一份最小代码库。它足够简单，能让你看清 `App`、状态、导航和共享模型；也足够真实，能承接后面六个部分的持续增长。

## 当前工程里已经有什么

### `FocusListApp`

一个能编译的 SwiftUI 应用壳，负责承接前两部分的 UI 主线。你会在这里练习：

- `App`、`Scene`、`WindowGroup`
- 根视图和导航骨架
- 页面局部状态与共享状态的协作

### `FocusCore`

一个最小共享核心，里面已经有任务、项目和 `FocusStore` 的起点。它现在刻意保持很轻，因为教程要让你亲眼看到：共享核心为什么应该在 Part 3 才真正长起来。

### `focusctl`

一个非常小的 CLI 表面。前期它只是验证切面，Part 4 以后才会变成你检查共享规则是否真的共享的工具。

## 现在先做什么

第一步只做健康检查：

```bash
cd swift-tutorial/projects/focuslist/starter
swift test
swift build --product FocusListApp
swift build --product focusctl
```

只有这里通过，后面正文和 labs 的所有讨论才有意义。

## 读 starter 时重点看哪几个文件

- `Sources/FocusListApp/FocusListApp.swift`
- `Sources/FocusListApp/Root/FocusListRootView.swift`
- `Sources/FocusListApp/Features/Inbox/InboxView.swift`
- `Sources/FocusCore/FocusStore.swift`
- `Tests/FocusCoreTests/FocusCoreTests.swift`

先把这些文件看懂，再进入正文，学习速度会快很多。

## 一条使用约定

修改 starter 时，始终记住它是教程基线。你做的每个改动都应该能回答：这一步现在为什么值得做，它怎样改变了产品结构或工程边界。
