# 第10章：做出 BoardFlow v1 桌面应用骨架

## 本章交付

Part 2 不是把若干组件解释完就结束，而是要给 `BoardFlow` 一个清楚的 v1 桌面骨架。它至少要回答这几个问题：

- 左侧什么在列？
- 中间什么在解释？
- 当前选择是什么？
- 新建或编辑动作从哪里进入？
- 表单和列表怎么共存而不互相打架？

到这一章，`BoardFlow v1` 应该被理解成一个真正的桌面应用骨架，而不再只是“首页 + 一些示例内容”。

## 一个更完整的 v1 骨架应该长什么样

对当前阶段，一个合理的骨架大致会是：

```swift
NavigationSplitView {
    List(model.boards, selection: $model.selection) { board in
        Label(board.title, systemImage: "square.on.square")
    }
    .navigationTitle("Boards")
} detail: {
    VStack(alignment: .leading, spacing: 20) {
        if let board = model.selectedBoard {
            Text(board.title)
                .font(.largeTitle.bold())
            Text("Cards: \(board.cardCount)")
                .foregroundStyle(.secondary)
        } else {
            Text("Select a board")
                .font(.title2)
            Text("Choose a board from the sidebar or create a new one.")
                .foregroundStyle(.secondary)
        }

        Divider()

        Form {
            Section("Create Board") {
                TextField("Title", text: $model.draftTitle)
                Picker("Template", selection: $model.draftTemplate) {
                    Text("Blank").tag(BoardTemplate.blank)
                    Text("Planning").tag(BoardTemplate.planning)
                    Text("Research").tag(BoardTemplate.research)
                }
                Toggle("Show dot grid", isOn: $model.showDotGrid)
            }
        }
    }
    .padding(24)
}
```

这里重要的不只是多了几个组件，而是桌面应用骨架已经开始成形：

- `List` 负责集合语境
- `selection` 负责工作台当前焦点
- detail 区负责解释当前选择
- `Form` 负责结构化输入

## 本章如何串起 Part 2

### 1. 容器组件串起来了

`List`、`Form`、`Section` 不再是分散 API，而是分别占住了集合区和输入区的角色。

### 2. 导航结构串起来了

`NavigationSplitView` 不再只是示意，而成为 Mac 工作台的主骨架。

### 3. 状态所有权串起来了

列表选择、创建草稿、详情显示都围绕同一套屏幕级状态组织，而不是各自私有一份。

### 4. 可观察模型串起来了

屏幕级 model 承担协调作用，让 sidebar、detail 和表单之间的状态关系保持清楚。

## 为什么这一步对后面特别关键

因为从下一部分开始，教程会进入真正的桌面工作台语境：

- toolbar
- inspector
- environment
- focus
- 多面板层次

如果现在还没有一个稳定的 v1 骨架，后面这些能力只会像功能堆叠，而不会形成系统。所以 Part 2 的结尾必须先把“常规桌面应用”这一层站稳。

## 自查清单

你现在应该能回答：

1. 为什么 `List` 和 `Form` 在 `BoardFlow` 里是两种不同职责，而不是同类控件容器？
2. 为什么 `NavigationSplitView` 比单纯 `NavigationStack` 更适合当前主结构？
3. 为什么 `selection` 是工作台状态？
4. 为什么创建表单应该围绕草稿和输入契约，而不是直接在按钮里乱改集合？
5. 为什么 observable model 在这里是协调层，而不是业务黑洞？

## 常见错误

### 1. 做出了侧栏和详情区，但状态仍然各管各的

这样只是看起来像工作台，实际上还是拼装页面。

### 2. 把创建表单写成一组零散输入，不承认它是统一契约

后面校验、提交和取消都会很难做。

### 3. 以为 v1 骨架就该把所有高级特性都塞进去

当前阶段的目标是结构成立，不是 feature 贪多。

### 4. 组件会用了，却解释不清它们为什么在这里协同

这正是“学过 API，但系统感不够”的典型症状。

## English Recap

`BoardFlow v1` is the point where Part 2 becomes real: collection UI, form input, split-view structure, bindings, and observable coordination now align into one desktop skeleton. The value is not feature richness yet. The value is that the app finally behaves like a real Mac workbench baseline.

## Drills

1. 用自己的话总结 `BoardFlow v1` 的四块核心结构。
2. 为什么说 v1 的目标是“结构成立”，而不是“功能做满”？
3. 如果你要在下一阶段加 Inspector，为什么现在的 split-view 骨架已经在帮你铺路？

## Project Handoff

Part 2 到这里结束。你现在已经不只是会写几个 SwiftUI 组件，而是拥有了一条从基础视图语言到桌面应用骨架的完整主线。下一部分要进入真正的 `Mac workbench` 结构：toolbar、inspector、environment、focus 和多面板层次，都会开始系统展开。
