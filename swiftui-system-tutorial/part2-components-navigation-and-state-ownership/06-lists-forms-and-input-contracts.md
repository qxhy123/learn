# 第6章：列表、表单与输入契约

## 为什么这一章现在出现

Part 1 解决的是“SwiftUI 在写什么”和“首页是如何被组织出来的”。但那还不够。因为一个真正的桌面应用不会长期停留在“标题 + 几块静态内容”的阶段，它很快就会长出两类东西：

- 有身份的集合内容
- 需要解释、校验、提交的输入内容

对 `BoardFlow` 来说，这两类东西分别就是：

- 白板列表
- 新建或编辑白板的输入区域

因此 Part 2 不能从随便几个输入组件开始，而必须先建立一个更稳的判断：**列表不是一堆重复行，表单也不是若干控件堆在一起。**

## `List`：把数据集合放进真正的集合界面语境

很多人一开始会继续沿用 `VStack + ForEach`：

```swift
VStack {
    ForEach(document.boards) { board in
        Text(board.title)
    }
}
```

这种写法当然能显示数据，但它没有明确表达“这是一个可导航、可选择、可演化的集合界面”。`List` 的价值就在这里：

- 它承认你当前正在处理一个有 identity 的集合
- 它天然更贴近导航、选择、删除、移动这类集合级交互
- 它让桌面应用的 sidebar / outline / list 思维更自然

对 `BoardFlow`，左侧白板区本质上就是一个工作台里的集合视图，所以 `List` 比手工垂直堆叠更诚实。

## `Form`：表单不是控件容器，而是输入契约

SwiftUI 初学者最常见的弱理解是：“`Form` 就是把 `TextField`、`Toggle`、`Picker` 这些东西装进去。”  
这很像把 UI 理解成外观拼装，而不是把输入理解成一份契约。

更强的理解是：

**Form 表达的是一组应该被一起解释、一起校验、一起提交的输入。**

比如 `BoardFlow` 的最小新建白板表单，可以长这样：

```swift
Form {
    Section("Board") {
        TextField("Title", text: $draftTitle)

        Picker("Template", selection: $draftTemplate) {
            Text("Blank").tag(BoardTemplate.blank)
            Text("Planning").tag(BoardTemplate.planning)
            Text("Research").tag(BoardTemplate.research)
        }

        Toggle("Show dot grid", isOn: $showDotGrid)

        Stepper("Starter cards: \(starterCardCount)", value: $starterCardCount, in: 0...12)
    }
}
```

这里最重要的不是控件种类，而是它们共同表达一件事：**“我要创建一个什么样的 board”**。一旦你这样看，后面校验、提交、保存路径就都有了稳定语义。

## `Section`：不是排版留白，而是输入和内容的语义分区

`Section` 很容易被误解成“加个标题让界面更好看”。实际上它的真正价值在于：

- 帮你划分输入边界
- 帮你划分内容职责
- 让读者知道哪些字段是同一组解释单位

例如：

- `Board`：标题、模板、初始配置
- `View Options`：网格、缩放起点、默认面板可见性
- `Actions`：创建、取消

如果没有 `Section`，表单会迅速退化成“好多控件顺着排”。这对桌面工作台来说尤其危险，因为桌面 UI 很容易长得复杂，没有语义分组就会很快失控。

## 输入组件的角色，不只是“会收值”

本章要系统化的还有一个误区：别把基础输入组件只当成“拿到一个值”的工具。它们各自有更适合的输入语义。

### `TextField`

适合文本草稿、标题、名称这类直接编辑的字符串输入。

### `Toggle`

适合明确的布尔开关，比如是否显示网格、是否在新板中包含示例卡片。

### `Picker`

适合有限离散选项，比如模板类型、默认布局模式、排序方式。

### `Stepper`

适合小范围递增递减输入，比如初始卡片数量、缩放步长、层级数量。

这不是死规矩，但它能帮助你把输入表达得更诚实。工程里最怕的不是“用了错误控件”，而是所有输入都长得像临时凑出来的。

## `BoardFlow` 在本章的落点

这一章结束时，你应该能把 `BoardFlow` 的下一步结构明确想成：

- 左侧：`List` 承载白板集合
- 右侧：详情区解释当前板或空态
- 创建动作：通过 `Form` 和 `Section` 组织输入契约

也就是说，`BoardFlow` 从这一章开始，不再只是“首页上显示一些白板”，而是进入**集合 + 输入**的桌面应用语境。

## 双语关键词

- list：列表
- collection UI：集合界面
- form：表单
- input contract：输入契约
- section：语义分区
- discrete options：离散选项

## 常见错误

### 1. 用 `VStack + ForEach` 硬顶所有列表场景

短期能跑，长期会让选择、导航和集合交互都缺乏自然容器。

### 2. 把 `Form` 当成“输入控件摆放区域”

这会让输入字段失去统一解释和提交边界。

### 3. 不区分 `Picker`、`Toggle`、`Stepper` 的输入语义

所有输入组件都能“改值”，但不是都在表达同一种用户意图。

### 4. 把 `Section` 当纯视觉装饰

真正的收益在于语义分区，而不是样式。

## English Recap

`List` puts data into a real collection UI context, while `Form` defines an input contract rather than just holding controls. In `BoardFlow`, these components mark the transition from a readable home shell to an actual desktop app structure with collections and structured creation flows.

## Drills

1. 解释为什么 `BoardFlow` 的白板区更像 `List` 场景，而不是一组手工堆叠的文本行。
2. 说明为什么“模板、网格、初始卡片数量”更适合被看作一组输入契约。
3. 试着为 `BoardFlow` 的创建表单设计两个 `Section`，并说明它们的分组依据。

## Project Handoff

现在你已经有了集合和输入的基本容器语义。下一章要继续把它们放进桌面应用真正的骨架里：同样是导航，`NavigationStack` 和 `NavigationSplitView` 在 Mac 工作台中的含义并不一样。
