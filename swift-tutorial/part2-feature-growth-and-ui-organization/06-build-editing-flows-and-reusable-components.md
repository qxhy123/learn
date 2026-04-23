# 第 6 章：构建编辑流与可复用组件

## “能改”不等于“有编辑流”

很多初学者会把编辑功能做成两种极端：

- 直接在列表里随手改字段，导致半成品状态一路泄漏。
- 抽一个巨大的“通用编辑器”，把所有业务规则和 UI 条件塞进去。

这两种都不稳。真正的编辑流至少要回答四件事：

1. 用户什么时候进入编辑。
2. 编辑中的草稿状态放哪里。
3. 保存和取消分别意味着什么。
4. 哪一层负责真正写回产品数据。

## 先把“草稿”和“已提交状态”拆开

以任务编辑为例，不要直接在 `FocusTask` 上边输入边修改。先给编辑器准备一个草稿结构：

```swift
struct TaskDraft: Equatable {
    var title: String = ""
    var projectID: UUID?
    var tags: [String] = []

    init(task: FocusTask? = nil) {
        self.title = task?.title ?? ""
        self.projectID = task?.projectID
        self.tags = task?.tags ?? []
    }
}
```

这个 `TaskDraft` 很重要，因为它把“用户正在输入什么”与“系统当前真实保存了什么”拆开了。没有这层草稿，你就很难做取消、校验和失败恢复。

## 让编辑器只负责编辑，不负责拍板

接下来做一个职责清楚的 `TaskEditor`。它只接受草稿和回调，不直接拥有全局状态：

```swift
struct TaskEditor: View {
    @Binding var draft: TaskDraft
    let availableProjects: [FocusProject]
    let onSave: () -> Void
    let onCancel: () -> Void

    var body: some View {
        Form {
            TextField("Task title", text: $draft.title)

            Picker("Project", selection: $draft.projectID) {
                Text("Inbox").tag(Optional<UUID>.none)
                ForEach(availableProjects) { project in
                    Text(project.name).tag(Optional(project.id))
                }
            }
        }
        .toolbar {
            Button("Cancel", action: onCancel)
            Button("Save", action: onSave)
                .disabled(draft.title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        }
    }
}
```

这里最关键的不是 `Form` 或 `Picker` 的语法，而是边界：

- 编辑器只管展示和收集输入。
- 保存策略由调用方决定。
- 校验规则先做最小集：标题不能为空。

## 谁来持有编辑流程

编辑器本身不该决定它是 sheet、popover 还是导航推进页。通常由当前页面持有这条流程：

```swift
@State private var draft = TaskDraft()
@State private var isPresentingEditor = false
@State private var editingTaskID: UUID?
```

一个稳定的动作顺序通常是：

1. 点击“新建任务”或“编辑任务”。
2. 用当前任务数据初始化 `draft`。
3. 展示编辑器。
4. 用户点击保存时，再把 `draft` 提交给 `FocusStore` 或后续的 `FocusCore`。
5. 点击取消时，直接丢弃 `draft`。

这条流程保证了页面状态和产品状态不会相互污染。

## 什么样的组件值得抽

现在你会开始看到 `TaskRow`、`TagChip`、`EmptyStateView` 这种可复用组件候选。判断标准不是“重复了两次”，而是下面三条：

- 它重复的是稳定的视觉结构或交互协议。
- 调用方不会因为它失去对业务规则的控制。
- 抽出后，接口更清楚，不是更模糊。

例如 `TaskRow` 值得抽，是因为任务展示的骨架已经稳定。
而“任务编辑 + 保存 + 删除 + 错误提示”一把全抽成 `UniversalTaskFeatureView`，通常就是把复杂度藏起来，而不是降低复杂度。

## 做完这一章后的手动验收

至少走通这几条路径：

1. 从列表进入编辑。
2. 修改标题后点击取消，原任务不应变化。
3. 修改标题后点击保存，列表应立即反映新值。
4. 新建任务时，空标题保存按钮应禁用。

如果你做完后发现“取消其实也会改到真数据”，说明草稿层还没真正站稳。

## 本章小结

编辑流真正训练你的，是“状态在什么时候才算真的提交”。而组件抽取训练你的，是“哪些重复值得抽，哪些复杂度只是在换地方藏”。这两条判断，后面会直接影响测试、持久化和失败处理。
