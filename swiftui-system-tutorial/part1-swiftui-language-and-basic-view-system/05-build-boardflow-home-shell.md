# 第5章：做出 BoardFlow 的最小工作台首页

## 本章交付

到这一章，Part 1 不再只是解释 SwiftUI 的几个基本概念，而是把前四章压成一个真实的最小交付。当前 `BoardFlow` starter 要达到的不是“功能完整”，而是“结构完整”。也就是说，它至少要已经具备：

- 一个清楚的 app 入口
- 一个清楚的首页内容 view
- 一份明确传入首页的数据模型
- 一块能读、能看、能继续生长的首页内容区

这就是最小工作台首页。

## 当前工程结构

当前 starter 的核心文件很少，但分工已经明确：

### `BoardFlowApp.swift`

它负责应用入口和主场景定义：

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

### `BoardDocument.swift`

它负责首页所需的最小数据事实：

```swift
struct BoardSummary: Identifiable, Equatable {
    let id: UUID
    var title: String
    var cardCount: Int
}

struct BoardDocument: Equatable {
    var title: String
    var boards: [BoardSummary]

    static let empty = BoardDocument(title: "Untitled Board", boards: BoardSummary.samples)
}
```

### `BoardHomeView.swift`

它负责首页结构描述：

```swift
NavigationSplitView {
    List(document.boards) { board in
        Label(board.title, systemImage: "square.on.square")
    }
    .navigationTitle("Boards")
} detail: {
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
    .padding(24)
}
```

## 本章怎么串起 Part 1

这一页之所以重要，不是因为它复杂，而是因为它把 Part 1 的四条主线都落了地。

### 1. 应用结构落地

`BoardFlowApp` 说明你已经不再是在写一堆孤立 view，而是在声明一个真实 app 的入口与场景。

### 2. 组合结构落地

首页 detail 区通过 `VStack` 和 `HStack` 建立起清楚的信息层次，不是把所有内容乱堆在一起。

### 3. 基础组件落地

`Text`、`Label`、列表行摘要都已经承担了各自语义角色，页面不是抽象骨架，而是一个可阅读界面。

### 4. 状态驱动意识落地

虽然当前 starter 还克制，但它已经明确通过 `document` 输入驱动首页，而不是让 view 自己拼出一堆临时字符串。后面一旦加入筛选、创建、选择、导航，你就知道该从状态层往上推，而不是从控件补丁往下糊。

## 为什么现在就引入 `NavigationSplitView`

你可能会问：Part 1 不是才刚开始吗，为什么首页就已经用了 `NavigationSplitView`？

因为这套教程是 `Mac-first`，而 `BoardFlow` 的终点不是一个单列表 app，而是一个桌面创作工具。早点让你看见“sidebar + detail”这种结构，比让你先沉迷单页 demo 更诚实。当前它仍然保持简单，只是在结构上提前对齐后续工作台。

换句话说，这不是提前讲复杂导航，而是在防止教程从一开始就走错叙事路径。

## 自查清单

写完这一章后，你应该能回答：

1. 为什么 `BoardFlowApp` 负责入口，而不是首页自己负责应用结构？
2. 为什么 `BoardDocument` 比直接在 view 里写一堆字符串更稳？
3. 为什么 detail 区的主结构是 `VStack`？
4. 为什么白板摘要行是 `HStack`？
5. 为什么这个 starter 虽然小，但已经具备后续长成工作台的可能？

只要这五个问题答不清，说明 Part 1 还没有真正学透。

## 常见错误

### 1. 为了“更简单”把数据直接写死在 view 里

这样做会让 starter 看起来更短，但会切断后面状态和模型的主线。

### 2. 过早追求视觉花样

Part 1 的目标是结构清楚，不是做炫技首页。

### 3. 把 `NavigationSplitView` 误解成“已经在讲复杂导航系统”

当前它只是桌面结构的诚实起点。

### 4. 看见页面能显示就以为 Part 1 完成了

真正完成的标准不是“能跑”，而是你能解释清楚这套结构为什么这样组织。

## English Recap

Part 1 ends with a small but honest delivery: `BoardFlow` now has an app entry, a document input model, and a readable home shell. The point is not feature depth yet. The point is that the project already follows the right structural rules for a Mac-first SwiftUI system tutorial.

## Drills

1. 用一句话说明 `BoardFlowApp`、`BoardDocument`、`BoardHomeView` 三者的职责边界。
2. 为什么说当前 starter 的价值在“结构完整”，而不是“功能完整”？
3. 如果你要在首页增加一个“Create Board”按钮，它首先应该被理解成什么角色？

## Project Handoff

Part 1 到这里结束。你已经完成了从“SwiftUI 是什么”到“能写出一个结构诚实的首页壳”的过渡。下一部分要真正开始补齐你前面指出的核心短板：常用 UI 组件、列表与表单、导航结构，以及状态拥有关系，都会进入系统讲解。
