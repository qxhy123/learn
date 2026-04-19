# 第9章：Observable Model 与屏幕级状态协调

## 为什么 Part 2 需要可观察模型

如果 `BoardFlow` 只是一页静态首页，那 Part 1 的局部状态和简单输入就够了。  
但一旦进入桌面骨架，状态就开始超出单个 view 的承受范围：

- 白板列表要共享给 sidebar 和 detail
- 当前选中项会影响多个区域
- 创建表单和当前集合要在一个屏幕上协调
- 后面还会接到持久化、撤销、自动保存和工作台面板

这时如果你还只靠“父 view 一层层传下去”，很快就会写出一个超级大 view，状态到处乱串。于是需要一个更明确的角色：**屏幕级状态协调者。**

## Observable Model 的职责：协调 UI 状态，不是吞掉所有业务规则

本教程用“observable model”这个说法，是为了强调职责，而不是押注某个特定语法。你可以用：

- `@Observable`
- `ObservableObject`

关键不是站队哪种 API，而是把职责守住：

**observable model 负责组织和暴露屏幕所需状态，并协调用户意图进入更底层系统。它不是把整个 app 的业务规则都塞进 UI 层。**

例如：

```swift
@Observable
final class BoardWorkbenchModel {
    var boards: [BoardSummary] = BoardSummary.samples
    var selection: BoardSummary.ID?
    var draftTitle = ""
    var draftTemplate: BoardTemplate = .blank
    var showDotGrid = true

    var selectedBoard: BoardSummary? {
        boards.first { $0.id == selection }
    }
}
```

这里 model 的意义在于：

- 暴露集合状态
- 暴露当前选择
- 暴露表单草稿
- 提供派生状态 `selectedBoard`

但它还不是最终的文档系统，也不是持久化层，更不是全部领域逻辑。

## `@Observable` 和 `ObservableObject`：本教程怎么处理

SwiftUI 工程现实里你会同时看到新旧两套观察风格：

- `@Observable`
- `ObservableObject` + `@Published`

本教程在概念上会优先解释观察模型的职责，再在代码层根据阶段选择更现代、更适合当前示例的写法。原因很简单：如果你一开始只关心语法，不关心职责，你很容易写出“虽然会刷新，但分层非常差”的代码。

所以这里你要先掌握的是：

- 什么时候状态已经超出局部 view
- 什么时候该引入屏幕级 model
- 它该持有什么，不该持有什么

## `BoardFlow` 的屏幕级 model 应该怎样理解

对当前 v1 桌面骨架来说，一个比较稳的层次是：

- `BoardDocument`：页面要消费的基本事实模型
- `BoardWorkbenchModel`：当前屏幕的状态协调层
- 子视图：消费状态、触发输入和意图

例如 sidebar 可能只关心：

- `boards`
- `selection`

创建表单可能只关心：

- `draftTitle`
- `draftTemplate`
- `showDotGrid`

detail 区可能关心：

- `selectedBoard`

如果没有 observable model，你会被迫：

- 要么把所有状态塞进一个超级父 view
- 要么让每个子 view 都拿自己的一份状态

前者会膨胀，后者会分叉。

## 派生状态应该优先算出来，而不是另存一份

observable model 很容易走偏的一点是：  
一旦它成了状态中心，就开始什么都存。

这是错误的。能推导出来的，优先推导。比如：

```swift
var selectedBoard: BoardSummary? {
    boards.first { $0.id == selection }
}
```

这比你再单独存一份 `currentBoard` 更稳，因为：

- 少一份需要保持同步的事实
- 更容易解释谁是真源
- 更不容易出现 sidebar 选中和 detail 内容不一致

这条原则后面在大画布、Inspector、工具系统里会越来越重要。

## `BoardFlow` 在本章的落点

本章结束时，你应该已经能看懂 Part 2 的完整状态分层：

- 页面需要集合、选择、表单草稿和派生详情
- 这些状态不该散在一堆子 view 里
- 它们需要一个可观察的屏幕级协调层

这就意味着 `BoardFlow` 已经开始具备“桌面应用骨架”的状态基础，而不只是几个能显示内容的 view。

## 双语关键词

- observable model：可观察模型
- screen state：屏幕级状态
- coordination：协调
- derived value：派生值
- observation：观察机制
- state surface：状态表面

## 常见错误

### 1. 把 observable model 当万能业务层

它应该协调 UI 状态，而不是吞掉所有底层规则。

### 2. 把所有可推导值都存下来

存得越多，同步成本越高。

### 3. 不愿引入屏幕级状态层，继续硬顶超级父 view

结果通常是参数越来越多，view 越来越肿。

### 4. 把每个区域都做成各自的小状态中心

这样很快就会出现多份平行事实。

## English Recap

Once `BoardFlow` becomes a real desktop skeleton, local state and parent-to-child passing are no longer enough. An observable screen model should coordinate shared list state, selection, and form drafts, while still keeping derived values derived and leaving deeper business rules outside the UI coordination layer.

## Drills

1. 说明为什么 `selection` 更像屏幕级状态，而不是某个列表项局部状态。
2. 为什么 `selectedBoard` 更适合作为派生值，而不是额外存一份？
3. 解释 observable model 为什么不应该直接吞掉持久化和全部业务规则。

## Project Handoff

Part 2 的最后一章要把这里的组件、导航和状态层全部合起来，真正给 `BoardFlow` 一个 v1 桌面应用骨架。那会是 Part 1 到 Part 2 的第一次完整收束。
