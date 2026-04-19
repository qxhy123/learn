# 第2章：View Composition 与三大基础布局

## 为什么这一章现在出现

理解了 `App` 和 `body` 的声明式角色之后，下一步就不是去背 modifier 清单，而是先解决一个更基础的问题：

**一个 SwiftUI 界面到底怎样被组织出来？**

很多教程一上来就铺一串 API：`padding`、`frame`、`foregroundStyle`、`background`。但如果你没有先抓住组合结构，学到的只会是一堆局部写法，最后页面一复杂就只剩下 modifier 堆叠。对 `BoardFlow` 这样的创作工具主线，这种习惯后面会直接把工作台结构写烂。

所以本章先收住，不急着追复杂样式，而是先把最基本的组合语言讲清楚。

## 三种基础布局不是“语法容器”，而是三种组织关系

SwiftUI 里最常见的三类基础容器是：

- `VStack`
- `HStack`
- `ZStack`

如果只背语法，它们只是“竖排 / 横排 / 叠放”。但更有价值的理解是：

- `VStack` 表达垂直阅读顺序
- `HStack` 表达并列关系
- `ZStack` 表达层叠关系

这不是文字游戏，而是影响你如何拆界面语义。

对 `BoardFlow` 首页 detail 区来说：

```swift
VStack(alignment: .leading, spacing: 16) {
    Text(document.title)
        .font(.largeTitle.bold())

    Text("BoardFlow starter for Part 1 and Part 2")
        .foregroundStyle(.secondary)

    Text("Recent boards")
        .font(.headline)

    ForEach(document.boards) { board in
        HStack {
            Text(board.title)
            Spacer()
            Text("\(board.cardCount) cards")
                .foregroundStyle(.secondary)
        }
    }

    Spacer()
}
```

这里不是“用了两个 stack，所以界面能摆出来”。更关键的是：

- 页面总体内容按垂直阅读顺序排列，所以外层是 `VStack`
- 每一行白板摘要都在表达“左侧标题 + 右侧计数”的并列关系，所以内层是 `HStack`
- 当前这个页面还不需要叠层，因此没有用 `ZStack`

这就是组合不是“为了排版”，而是为了表达结构。

## 从一个较弱起点开始：按视觉切块，而不是按职责组合

很多初学者会这样写：

```swift
VStack {
    Text("BoardFlow")
    Text("Recent boards")
    Text("Weekly Planning")
    Text("8 cards")
    Text("Product Discovery")
    Text("14 cards")
}
```

界面当然能显示，但结构完全没有被表达出来。问题在于：

- “标题”和“列表项”是不同语义层次，却被平铺在一起
- 同一条列表项内部的并列关系没有被单独表达
- 后面一旦要加图标、交互、状态、选择态，就会迅速长成一坨

更强的写法并不是“文件拆得更碎”，而是**把组合边界和界面职责对齐**。

## View Composition：不是把大文件拆小，而是把语义拆清楚

“组合”在 SwiftUI 里最容易被误解成“组件化”。很多人会机械地把任何 `HStack` 都抽成一个子 View，结果文件是多了，但结构没变清楚，甚至更难追踪。

真正重要的是三个判断：

1. 这部分界面是否承担独立语义？
2. 它是否拥有清楚的输入？
3. 拆出来之后，是否更容易读、测、复用？

对 `BoardFlow` 来说，`Recent boards` 列表区域以后很可能会长成一个独立的摘要区，这时把它抽成 `BoardSummaryListView` 就是有意义的。相反，如果只是为了“看起来更组件化”把一个只有一行文字的 `HStack` 单独拆文件，收益就很低。

## `ZStack` 不该晚学，但也不该乱学

初学者对 `ZStack` 通常有两个极端：

- 要么完全不用，所有浮层效果都硬写成奇怪的 `overlay`
- 要么什么都叠，结果界面层级一混乱就失控

更稳的理解是：`ZStack` 表达的是**层叠空间里的前后关系**。它最适合后面这些场景：

- 浮层按钮
- 局部高亮遮罩
- 选中态描边
- 画布上的辅助层

`BoardFlow` 当前 Part 1 还没正式进入这些场景，所以本章只先建立语义意识：当界面里开始出现“前景内容 + 背景基底 + 局部浮层”时，你就进入 `ZStack` 和层级设计的语境了。Part 3 和 Part 4 会大规模用到它。

## Modifier 的价值是修饰已有结构，不是代替结构

SwiftUI 另一个常见误区是把 modifier 当成主角，好像界面水平取决于你记住多少个点语法。实际上，modifier 的位置应该很清楚：

- 组合先决定结构
- modifier 再细化布局、样式、交互

例如：

```swift
Text(document.title)
    .font(.largeTitle.bold())
    .foregroundStyle(.primary)
```

这些 modifier 有价值，但前提是 `Text(document.title)` 自己已经是合理的语义节点。如果你一开始结构就错了，再多 modifier 也只是让错误结构更复杂。

## `BoardFlow` 在本章的落点

本章结束时，你应该能把 `BoardFlow` 首页看成一个清楚的组合结构：

- 左侧：board 集合导航
- 右侧：detail 内容区
- 右侧内部：标题、说明、列表摘要按垂直语义组织
- 每条摘要：标题和统计按水平并列组织

这看起来很基础，但它其实已经为后面工作台、Inspector、浮层和画布结构打下了语义地基。

## 双语关键词

- composition：组合
- layout stack：布局栈
- vertical flow：垂直流
- horizontal relation：水平并列关系
- layering：层叠
- modifier：修饰器
- semantic boundary：语义边界

## 常见错误

### 1. 把所有内容平铺在一个 `VStack`

短期能跑，长期一定难扩展。

### 2. 为了“组件化”机械拆 View

拆分应该服务语义边界，而不是服务形式上的文件数量。

### 3. 用 modifier 代替结构设计

modifier 只能细化已有结构，不能弥补错误结构。

### 4. 过早滥用 `ZStack`

层叠关系一旦模糊，后面浮层、点击区域、命中测试都会变难解释。

## English Recap

`VStack`, `HStack`, and `ZStack` are not just layout tricks. They encode vertical reading flow, horizontal relationships, and layered spatial structure. In `BoardFlow`, understanding composition early prevents the UI from becoming a pile of modifiers before the project even reaches workbench and canvas complexity.

## Drills

1. 说明为什么 `BoardFlow` detail 区适合外层 `VStack`、内层列表项用 `HStack`。
2. 举一个你觉得以后 `BoardFlow` 会用到 `ZStack` 的场景，并解释原因。
3. 判断什么情况下应该把一块 UI 抽成独立子 View，而不是继续留在当前文件。

## Project Handoff

现在你已经知道 `BoardFlow` 首页是如何通过组合建立起来的。下一章要把这个结构真正“点亮”起来：不是继续讨论抽象组合，而是进入最基本的可交互组件，先让首页拥有清楚的可读内容和入口语义。
