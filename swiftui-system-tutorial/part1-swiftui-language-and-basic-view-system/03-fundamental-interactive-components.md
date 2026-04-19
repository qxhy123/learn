# 第3章：最基本的可交互组件

## 为什么这一章现在出现

前两章解决的是“应用结构”和“界面结构”。但界面结构还只是骨架。真正让首页成立的，是那些最基础却最常被低估的组件：

- `Text`
- `Image`
- `Button`
- `Label`

很多人会嫌这些组件太简单，急着往后学列表、导航、动画、画布。可现实里，复杂 UI 不是靠高级 API 凭空出现的，而是由这些基础单元逐步组织出来。它们决定了页面的阅读节奏、可点击语义、图标表达和动作入口。

## `Text`：不是把字符串显示出来就完了

`Text` 是 SwiftUI 里最基础的展示组件，但它真正承担的是**信息层级表达**。

在 `BoardFlow` 首页 detail 区里：

```swift
Text(document.title)
    .font(.largeTitle.bold())

Text("BoardFlow starter for Part 1 and Part 2")
    .foregroundStyle(.secondary)

Text("Recent boards")
    .font(.headline)
```

这里不是三个字符串而已，它们分别在表达：

- 页面主标题
- 次级说明
- 区块标题

如果没有这种层级判断，页面很快会变成“所有文字都只是文字”。而一旦后面进入工作台、Inspector、属性面板，文本层级混乱会直接拖垮信息可读性。

## `Image` 与图标：视觉提示不只是装饰

虽然 starter 当前实现还没单独放 `Image(systemName:)`，但 `BoardFlow` 的界面很快会大量依赖图标语义：

- 白板图标
- 工具栏动作
- Inspector 状态提示
- 连接、吸附、选择等视觉符号

这也是为什么 SwiftUI 里图像组件不能只被看成“放图片”。在工程上，它往往承载：

- 动作类别提示
- 状态提示
- 空间关系提示

后面从工作台进入画布时，这种视觉语义会越来越重要。

## `Button`：不是“点击后执行代码”，而是用户意图入口

初学者对 `Button` 最容易有两个问题：

1. 觉得它只是“包一段点击逻辑”
2. 把所有行为直接塞进按钮闭包

更强的理解是：`Button` 首先是在表达**这里有一个用户意图入口**。

例如后面 `BoardFlow` 首页会自然出现：

```swift
Button("Create Board") {
    // 触发创建白板意图
}
```

真正重要的不只是这段代码有没有执行，而是它在结构上表明：

- 用户可以从这里发起新建动作
- 这个动作是首页的一等入口
- 后续状态变化应由明确状态拥有者协调，而不是任意在闭包里乱改

这就是为什么本教程一直强调 intent。`Button` 不是“方便你写点击代码”，而是把“用户想做什么”变成一个明确界面节点。

## `Label`：把文字和图标语义绑在一起

在 starter 的左侧 board 列表中：

```swift
Label(board.title, systemImage: "square.on.square")
```

`Label` 的价值，不在于少写了一个 `HStack`，而在于它明确表达了“图标 + 文字”是一个共同语义单位。这种组件尤其适合：

- 列表项
- 菜单项
- 工具栏按钮
- 侧边栏导航项

如果你后面总是手工 `HStack { Image ... Text ... }`，不是绝对错误，但很多场景下你其实是在绕开一个更诚实的语义组件。

## 从一个较弱起点开始：所有交互都只是“点一下执行点代码”

弱设计的写法往往长这样：

```swift
Button("Create Board") {
    boards.append(...)
    selection = ...
    showSidebar = true
    logAction()
    saveIfNeeded()
}
```

这类代码的问题不只是“闭包太长”，而在于：

- 组件失去单一职责
- 用户意图、状态更新、持久化、副作用全混在一起
- 后面越多入口，状态越容易发散

更强的方向是：组件表达入口，状态层协调结果。Part 2 会把这一点讲得更系统，但这里你要先建立一个基本警觉：**别把基础组件当成万能逻辑收纳盒。**

## `BoardFlow` 在本章的落点

本章结束时，你应该把 `BoardFlow` 首页里的基础组件理解成这些角色：

- `Text`：建立信息层级
- `Label`：建立图文合一的列表项语义
- `Button`：建立动作入口
- `Image`：为后续工作台和工具系统预留视觉语义

这不是低阶内容。恰恰相反，如果基础组件的语义意识不立住，后面再多高级能力都会长在一块不稳定的 UI 土壤上。

## 双语关键词

- text hierarchy：文本层级
- iconography：图标语义
- interaction entry：交互入口
- intent：用户意图
- label：图文标签组件
- affordance：可供性 / 操作提示

## 常见错误

### 1. 所有 `Text` 都用同一层级

读者能看到内容，但看不出信息结构。

### 2. 在 `Button` 里直接堆业务逻辑

短期省事，后面一定让状态协调变脏。

### 3. 总是手工 `Image + Text`，不判断 `Label` 是否更合适

这会让很多导航和列表语义表达变弱。

### 4. 把图标当纯装饰

对工作台和创作工具来说，图标很快会变成操作语言的一部分。

## English Recap

Basic SwiftUI components are not trivial filler. `Text` defines information hierarchy, `Button` defines intent entry points, and `Label` combines icon and text into one semantic unit. `BoardFlow` uses these components to turn a plain layout skeleton into a readable, actionable home screen.

## Drills

1. 为什么 `Label` 在 sidebar 或列表项里往往比手写 `HStack` 更诚实？
2. 举例说明 `Button` 为什么不该直接承担持久化和复杂状态协调。
3. 对 `BoardFlow` 首页，指出哪几段 `Text` 分别对应主标题、说明和区块标题。

## Project Handoff

现在 `BoardFlow` 已经不只是有结构的静态页面，而是开始具备清楚的阅读层级和交互入口。下一章要把一个更关键的问题讲透：这些组件为什么会“动起来”？答案不在控件本身，而在状态如何驱动界面。
