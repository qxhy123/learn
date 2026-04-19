# 第1章：SwiftUI App 到底在写什么

## 为什么这一章现在出现

很多人学 SwiftUI 时，第一反应是把它当成“换一种语法写界面”。这当然不完全错，但如果只停在这个层面，后面一进复杂工作台、状态流、画布交互，就会迅速失控。你会写出一堆能跑的界面，却解释不清：

- 为什么 `body` 看起来像函数，却能驱动整个界面
- 为什么界面变化不是你手工去“刷新控件”
- 为什么 `App`、`Scene`、`WindowGroup` 这些东西不像 UIKit/AppKit 时代那样是显式的启动流水线

`BoardFlow` 作为这套教程的主项目，第一步还不是画布，不是拖拽，也不是 `Canvas`。第一步只是把一个 Mac SwiftUI app 壳建立起来，让你先看清：**SwiftUI 程序到底在表达什么。**

## 从一个较弱起点开始：把 SwiftUI 当成控件初始化脚本

初学者经常在脑子里默认这样的隐含模型：

1. `@main` 里的 `App` 像传统入口函数
2. `body` 是“启动时执行一次”的界面搭建代码
3. `WindowGroup` 是“给我生成一个窗口对象”
4. `BoardHomeView` 是“创建并长期保存着的页面实例”

于是他们会不自觉地把 SwiftUI 理解成另一种写法的初始化脚本。比如脑内模型会变成：

```swift
@main
struct BoardFlowApp: App {
    var body: some Scene {
        WindowGroup {
            // 创建首页控件
            // 填进去一些初始数据
            // 之后这个页面对象就一直活着
            BoardHomeView(document: .empty)
        }
    }
}
```

问题不是这段代码不能运行，而是这个理解方式太弱。它会把你后面所有设计都拉向命令式补丁：

- 想“什么时候重新创建 view”
- 想“怎么保存这个 view 实例”
- 想“哪个地方去命令式地刷新某块 UI”

这和 SwiftUI 真正鼓励的写法是错位的。

## 更强的理解：SwiftUI 在声明当前状态下的界面结构

对本教程来说，第一原则可以压缩成一句话：

**SwiftUI 代码的核心任务不是手工造控件，而是在当前状态下声明界面应该长什么样。**

看 `BoardFlow` starter：

```swift
@main
struct BoardFlowApp: App {
    var body: some Scene {
        WindowGroup {
            BoardHomeView(document: .empty)
        }
    }
}
```

这段代码更准确的理解不是：

- “先创建一个窗口”
- “再把首页对象塞进去”

而是：

- 这个 app 有一个主场景
- 主场景使用窗口组承载内容
- 当这个场景需要内容时，内容由 `BoardHomeView(document: .empty)` 这段声明来描述

也就是说，`App` 不只是程序入口，它还是**应用级界面结构的声明表面**。`Scene` 不只是窗口容器，它是系统理解你这个 app 如何组织显示语境的方式。`WindowGroup` 不只是“帮我开窗”，它是在说：**这类内容存在于窗口场景里。**

这正是后面为什么我们可以自然扩展到多窗口、多文档、工具面板和桌面工作台结构。因为从一开始，SwiftUI 的入口就不是“初始化一个对象图”，而是“声明 app 的显示组织形式”。

## `body` 为什么看起来像普通属性，却能驱动 UI

这是 SwiftUI 新手最容易模糊的一点。

`body` 看起来像普通计算属性：

```swift
var body: some Scene { ... }
```

或者：

```swift
var body: some View { ... }
```

但它承载的不是“我要返回一个永久存在的界面对象”，而是“我给出当前状态下的界面描述”。这个差别非常重要：

- 你写的是描述，不是手工维护的控件树
- 你交出去的是结构，不是长期缓存的实例
- 你真正关心的是输入状态和输出界面的关系

所以后面一旦状态变化，SwiftUI 的关键问题不是“你要不要重新 new 一个 view”，而是“基于新状态，界面描述现在是什么”。理解了这点，后面讲状态、列表 identity、动画、画布、性能时，你就不会总往控件生命周期上绕。

## `BoardFlow` 在本章的落点

本章结束时，`BoardFlow` 还只是最小应用壳，但它已经清楚表达了三件事：

1. 它是一个 Mac SwiftUI app，而不是 Xcode 模板残骸
2. 它的主内容是一个明确的首页 `BoardHomeView`
3. 首页的数据来源不是“UI 自己偷偷拼出来的文本”，而是一个 `BoardDocument`

这里的价值不在于功能多，而在于边界开始诚实：

- `BoardFlowApp` 负责应用级入口和场景
- `BoardHomeView` 负责首页内容描述
- `BoardDocument` 负责提供首页所需的最小领域数据

这就是后面能继续长出工作台、导航、多面板和画布层的基础。

## 双语关键词

- app entry：应用入口
- scene：场景
- `WindowGroup`：窗口组
- `body`：界面描述属性
- declarative：声明式
- value-like description：值式描述
- application structure：应用结构
- rendering context：显示语境

## 常见错误

### 1. 把 `App` 当成一次性启动脚本

这样会导致你总想把副作用、依赖创建、状态初始化全塞在入口里，后面很快失去结构分层。

### 2. 把 `body` 当成“执行一次的搭建过程”

这会让你总想缓存视图、存控件引用、手工 patch UI。

### 3. 把 `WindowGroup` 理解成“唯一主窗口对象”

它描述的是一类窗口场景，不是一个你亲手掌控的 NSWindow 实例。

### 4. 在应用入口层就塞进大量全局可变状态

入口层应该先表达结构；复杂状态协调要有更合适的层次，不要一开始就糊成单例中心。

## English Recap

SwiftUI app code is not mainly a control-construction script. `App`, `Scene`, and `WindowGroup` describe how the application presents its content, while `body` describes what the UI should be for the current state. In `BoardFlow`, the first win is not feature depth but honest structure: app entry, home view, and document data already have distinct roles.

## Drills

1. 用你自己的话解释 `WindowGroup` 为什么不是“一个主窗口对象变量”。
2. 说明为什么把 `body` 理解成“搭建一次 UI”会在后面造成状态设计问题。
3. 对照 `BoardFlowApp`，指出应用结构声明、页面内容声明、数据模型这三层分别落在哪。

## Project Handoff

现在你应该把 `BoardFlow` 看成一个**已经有应用结构边界的 SwiftUI 项目**，而不是一个还没开始的空壳。下一章要继续解决另一个根问题：既然我们写的是界面描述，那这些描述是如何通过组合组织起来的？这就会进入 `View Composition` 和三大基础布局。
