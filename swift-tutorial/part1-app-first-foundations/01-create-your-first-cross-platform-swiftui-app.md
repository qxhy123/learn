# 第 1 章：创建第一个跨平台 SwiftUI 应用

## 这一章要解决什么

如果教程一开始就把 Swift 拆成一张语法清单，读者通常会在进入真实 App 时再次迷路。你知道 `struct`、`enum`、`func` 是什么，但不知道这些东西为什么会和 `App`、`Scene`、`WindowGroup` 出现在同一段代码里。本章的任务，就是先把第一个 `FocusList` 应用壳真正跑起来，再回头解释这段代码到底在表达什么。

## 先把 starter 跑起来

先确认工程能构建：

```bash
cd swift-tutorial/projects/focuslist/starter
swift test
swift build --product FocusListApp
```

如果这里过不了，不要继续往下读。因为后面所有关于状态和界面的讨论，都建立在这份 starter 代码能正常工作的前提上。

## 看懂第一个 SwiftUI 应用入口

打开 `Sources/FocusListApp/FocusListApp.swift`，你会看到这样的结构：

```swift
import SwiftUI
import FocusCore

@main
struct FocusListApp: App {
    @State private var store = FocusStore.sample()

    var body: some Scene {
        WindowGroup {
            FocusListRootView(store: store)
        }
    }
}
```

这里有四个你现在必须搞清楚的点：

1. `@main` 告诉 Swift，这就是应用入口。
2. `App` 不是某个页面，它描述的是整个应用。
3. `body` 返回的不是“执行过的 UI”，而是应用的场景描述。
4. `WindowGroup` 是用户真正会看到的窗口或界面容器。

也就是说，这段代码不是在“命令式地创建窗口”，而是在声明：这个应用应该以怎样的场景启动，并把哪一个根视图交给系统显示。

## 为什么这里会出现 `@State`

很多刚接触 SwiftUI 的人会问：应用入口里为什么会有状态？

原因很直接。`FocusList` 不是一张静态页面，而是一个会随着用户操作变化的产品。`FocusStore.sample()` 提供了 starter 的最小产品状态，`@State` 则让这个状态成为当前应用壳能持有、并驱动界面变化的东西。

先记住一句话：**SwiftUI 的应用入口不是“只负责启动”，它也负责把产品的初始状态交给根视图。**

## 跟着代码走到根视图

继续打开 `Sources/FocusListApp/Root/FocusListRootView.swift`：

```swift
import SwiftUI
import FocusCore

struct FocusListRootView: View {
    @Bindable var store: FocusStore

    var body: some View {
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
            .navigationTitle("FocusList")
        } detail: {
            InboxView(store: store)
        }
    }
}
```

这段代码已经让 `FocusList` 具备了最小产品骨架：

- 左侧有导航入口
- 右侧有默认详情区
- `Inbox` 和 `Projects` 共享同一个 `store`
- `Settings` 暂时还是轻量入口

现在你应该意识到，这不是“为了演示 `NavigationSplitView` 而拼出来的页面”，而是一个真正产品骨架的起点。

## 亲手做一个最小改动

把侧栏标题改成你自己的版本，例如：

```swift
.navigationTitle("FocusList Starter")
```

然后重新构建：

```bash
swift build --product FocusListApp
```

你现在做的事情很小，但它已经是一次真正的产品修改流程：改代码、重新构建、确认界面骨架仍然成立。

## 这章最容易犯的错

### 错误 1：把 `App` 当成“只会出现一次的特殊魔法文件”

它当然是入口，但它不是和产品无关的模板。后面你会不断回到这里，重新思考应用如何注入共享状态、如何组织场景，以及不同平台怎样共享或分化。

### 错误 2：一上来就想把所有状态抽成完美架构

现在这么做只会让你在还没有产品压力时先制造复杂度。Part 1 的任务不是设计终局，而是建立一条足够清楚的产品骨架。

## 本章小结

做完这一章后，你至少应该能回答：

- `App`、`Scene`、`WindowGroup` 各自负责什么
- 为什么应用入口里会出现产品状态
- `FocusListRootView` 为什么已经算一个真实产品骨架，而不是练习页面

如果这些问题你还回答不稳，先不要急着往后跳。Part 1 的所有内容，都建立在这个应用壳已经被你真正看懂的前提上。
