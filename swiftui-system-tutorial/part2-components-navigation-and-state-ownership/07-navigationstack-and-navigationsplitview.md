# 第7章：NavigationStack 与 NavigationSplitView

## 为什么 Mac 教程必须讲 Split View

很多 SwiftUI 教程的导航叙事都默认从 iPhone 开始，所以重点会落在：

- `NavigationStack`
- `NavigationLink`
- push/pop 风格的页面流

这在移动端非常自然。但 `BoardFlow` 是 `Mac-first` 创作工具，主界面不是“从列表点进去详情”这么简单，而更像：

- 左侧资源或集合区
- 中间主内容区
- 右侧可能还有 Inspector 或细节区

如果一开始只用 `NavigationStack` 的脑回路来想，你后面写出来的会更像“多页跳转应用”，而不是“桌面工作台”。所以这里必须尽早引入 `NavigationSplitView`。

## `NavigationStack`：层级流的主力

`NavigationStack` 最适合的语境是：

- 从一个页面进入下一级页面
- 路径有明显先后层级
- 用户理解的是“深入”和“返回”

例如：

- 设置页进入某个子页
- 文档列表进入单文档详情
- 检查一组层级关系明确的内容

它的问题不是不好，而是当你的桌面应用主结构本来就应该并行呈现多个区域时，纯 `NavigationStack` 会显得太单线。

## `NavigationSplitView`：多栏结构不是导航特效，而是工作台骨架

对 `BoardFlow`，更强的方向是：

```swift
NavigationSplitView {
    List(model.boards, selection: $model.selection) { board in
        Label(board.title, systemImage: "square.on.square")
    }
} detail: {
    BoardDetailPanel(selection: model.selectedBoard)
}
```

这段代码比“列表点进去详情页”更重要的地方在于：

- 左右区域是并存的，不是 push 出来的
- 当前选择本身就是工作台状态的一部分
- detail 不是“新页面”，而是同一工作台里的主解释区

这正是创作工具、开发者工具、管理后台和桌面应用里特别常见的界面组织方式。

## 为什么 `BoardFlow` Starter 现在就已经用了 `NavigationSplitView`

你可能注意到了：starter 当前已经是 `NavigationSplitView`。这不是“提早炫高级 API”，而是在结构上避免走弯路。

如果教程第一阶段就让读者习惯单页或单栈页面，后面再转成工作台，读者会以为那是“突然换一种写法”。而实际上，对 `BoardFlow` 这种目标产品来说，多栏结构从一开始就是更诚实的界面骨架。

所以本章的重点不是背 API，而是理解：

- 对什么类型的问题，`NavigationStack` 是自然解
- 对什么类型的问题，`NavigationSplitView` 才是自然解

## `NavigationLink` 仍然重要，但它不是这套教程的主结构

这并不意味着 `NavigationLink` 不重要。以后 `BoardFlow` 里仍然可能有：

- 从某个列表项进入更深层次的配置页
- 从检查器跳到某个详情流
- 从资源库进入模板详情

这些地方 `NavigationStack` / `NavigationLink` 依然合适。但它们是**工作台内部的层级流**，而不是整个应用的主骨架。

这也是本教程要建立的系统感：  
不是哪个 API 高级，而是**哪个结构更适合当前场景**。

## `BoardFlow` 在本章的落点

经过这章，你应该已经能明确判断：

- `BoardFlow` 的主结构更像 `NavigationSplitView`
- 当前选中的 board 是工作台状态，不只是临时页面跳转结果
- 以后即使引入 Inspector、多面板和画布，中间主区域依然是工作台中心，而不是“某个 pushed 页面”

这一步一旦想清楚，后面状态拥有关系就更容易讲。因为“当前选中哪个 board”已经不再是某个局部视图的小状态，而是整个屏幕结构都在依赖的事实。

## 双语关键词

- navigation stack：导航栈
- split view：分栏导航 / 分栏工作台
- detail pane：详情面板
- selection：当前选择
- workbench：工作台
- hierarchical flow：层级流

## 常见错误

### 1. 在 Mac 工作台场景里硬套移动端 push 流

结果通常是页面切来切去，却没有稳定桌面骨架。

### 2. 把 `NavigationSplitView` 当成“视觉上分三栏”的 layout 技巧

它首先是在表达结构，不只是布局。

### 3. 以为用了 `NavigationStack` 就不能有工作台

实际上它仍然适合工作台内部的深层流转，只是不是主骨架。

### 4. 把当前选择写成局部临时值

对工作台来说，selection 往往是屏幕级状态。

## English Recap

`NavigationStack` is best for hierarchical page flow. `NavigationSplitView` is better for desktop workbench structure where selection and detail coexist. `BoardFlow` uses split-view thinking early because its real destination is not a phone-style navigation app but a multi-panel creative tool.

## Drills

1. 为什么说 `BoardFlow` 的 detail 区更像工作台解释区，而不是 pushed page？
2. 举一个你认为以后 `BoardFlow` 里仍然适合 `NavigationStack` 的内部子流程。
3. 解释“selection 是工作台状态”这句话对后面状态设计意味着什么。

## Project Handoff

现在主结构已经清楚了：`BoardFlow` 是一个多栏工作台，而不是单线页面流。下一章要真正进入最容易写乱的一层：谁拥有状态，谁只是编辑状态，以及为什么 `Binding` 不是“能双向同步就全都传”。
