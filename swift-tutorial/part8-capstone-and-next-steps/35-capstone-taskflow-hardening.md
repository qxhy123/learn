# 第35章：Capstone `TaskFlow` 加固

> 现在共享核心和 CLI 路径已经在 capstone 里先被收紧，`TaskFlow` 终于可以接入一套更稳定的 shared foundation。本章的重点不是“把 UI 做得更花”，而是让 SwiftUI 客户端真正站在 hardened core 之上，补齐数据流、恢复、预览与验证的最后一段。

## 为什么这一章现在出现

如果在第34章之前就急着强化 `TaskFlow`，很容易出现一个假成熟状态：

- app model 表面很完整
- UI 状态很多，看起来很工程化
- 但底层共享 contract 仍然不稳定

那样的结果通常是 `TaskFlow` 被迫自己做很多补偿：

- 自己推断 mutation 成功后要不要刷新
- 自己定义一套错误分层
- 自己缓存或拼接本应由 shared foundation 提供的快照语义

现在顺序终于对了。`TaskFlow` hardening 之所以放在这里，正是因为它不该再做平行系统，而应做**共享 contract 的图形客户端实现**。

## 从一个较弱起点开始：UI 看似完整，但实际上在补核心空白

弱状态常常长这样：

- `TaskFlow` 有自己的 `TaskListModel`
- `TaskFlow` 有自己的失败文案和刷新逻辑
- `TaskFlow` 也许通过 repository 在工作
- 但它和 CLI 使用的共享语言还不完全一致

表现出来的症状通常包括：

- mutation 之后 UI 决定“自己再 load 一次”
- preview 用的是一套平行样本类型
- 错误映射更多是 UI 补丁，而不是共享失败 contract 的呈现
- 恢复路径和持久化入口停在 app 层，难以与 core 语义对齐

这不是说 `TaskFlow` 之前写错了，而是说它现在终于到了可以“停止补洞，开始对齐”的阶段。

## 更强的第一步：让 app model 直接围绕共享 snapshot / mutation 工作

capstone 阶段的 `TaskFlow` 更稳的形状，通常不是“View 自己协调更多状态”，而是让 app model 的核心职责变得更明确：

- 持有共享 snapshot 的 app-facing 视图态
- 发送共享 query / mutation
- 把 shared failure 映射为 UI 状态

例如：

```swift
@Observable
final class TaskListModel {
    private let runtime: TaskRuntime

    private(set) var snapshot: TaskSnapshot?
    private(set) var isLoading = false
    private(set) var failure: TaskRuntimeFailure?

    func load() async {
        isLoading = true
        defer { isLoading = false }

        do {
            snapshot = try await runtime.fetch(.all)
            failure = nil
        } catch let runtimeFailure as TaskRuntimeFailure {
            failure = runtimeFailure
        }
    }

    func perform(_ mutation: TaskMutation) async {
        do {
            let result = try await runtime.perform(mutation)
            snapshot = result.snapshot
            failure = nil
        } catch let runtimeFailure as TaskRuntimeFailure {
            failure = runtimeFailure
        }
    }
}
```

此时，`TaskFlow` 不再自己定义另一套“成功长什么样”，而是把共享 contract 翻译成 UI 可渲染状态。

## `TaskFlow` hardening 的重点不是状态更多，而是状态语义更清楚

到了 capstone，这一点尤其关键。SwiftUI 项目一变复杂，很多人会本能地继续往 model 里加状态字段。但真正更强的方向通常是：

- 不是加更多状态
- 而是明确哪些状态来自 shared foundation，哪些是 UI 局部状态

例如：

### 更适合来自 shared foundation 的状态

- 当前任务 snapshot
- mutation 是否完成
- runtime failure
- 最后一次成功刷新时间

### 更适合留在客户端的状态

- 当前表单草稿
- 当前 sheet 是否展开
- 当前导航路径
- 当前局部动画或交互提示

这条边界一旦守住，`TaskFlow` 就能既共享核心语义，又保留 SwiftUI 客户端自己的节奏。

## Preview 在 capstone 中的角色会升级

Part 6 已经让我们知道 preview 是结构检查器。到了 capstone，它更进一步成为**共享 contract 接入正确与否的快速证据**。

更成熟的 preview 组织方式会是：

- 用共享 snapshot 构造 loaded state
- 用共享 failure 构造 error state
- 用局部 UI 状态叠加在共享状态之上，观察 screen 是否仍然清楚

例如：

```swift
#Preview("Loaded") {
    TaskHomeScreen(model: .previewLoadedSnapshot)
}

#Preview("Persistence Failure") {
    TaskHomeScreen(model: .previewFailure(.saveFailed))
}
```

这类 preview 的价值在于，它们不再只是“给 UI 凑几条假任务”，而是在验证：

- app model 是否真的围绕共享 contract 工作
- UI 是否能诚实呈现 shared failure
- 状态切面是否还清楚

## `TaskFlow` hardening 也要补齐自己的验证链

和 CLI/Core 一样，UI 客户端的毕业标准也不能只靠“看起来能动”。对 capstone 更值得补的验证包括：

- app model 对 shared snapshot 的状态转换测试
- mutation 成功后 UI 状态如何变化
- shared failure 到 UI 呈现的映射测试
- preview / test double 是否真正复用了 shared contract

这里的重点不是追求重量级 UI 自动化，而是让 `TaskFlow` 的状态流与共享 contract 之间有足够证据。

## 恢复（recovery）与持久化路径要在这里真正说清楚

前面的章节已经讲过持久化和 app state。本章要把它们放进 capstone 语境里重新收紧：

- app 启动时从哪里拿到初始 snapshot
- mutation 失败后 UI 是否保留旧 snapshot
- app 恢复时哪些状态来自持久化，哪些只是临时 UI 状态
- 用户看到的错误是否对应 shared failure 的真实层级

这一步之所以重要，是因为 SwiftUI 客户端很容易用视觉连贯性掩盖语义模糊。Capstone 不允许这样做。你必须能说清：

- 失败后系统状态是什么
- UI 为什么显示成现在这样
- 这个结果和 CLI/Core 的 contract 是否一致

## `TaskFlow` hardening 不是让 UI 取代 CLI，而是证明共享核心真的可复用

整套教程在这里有一个必须守住的工程判断：`TaskFlow` 的成熟不是通过消灭 CLI 来证明的。

恰恰相反，真正成熟的 `TaskFlow` 会让你更清楚地看到：

- CLI 适合脚本化与文本入口
- SwiftUI 适合交互、导航与持续状态可视化
- 两者共享同一 foundation 时，系统整体更强

所以本章真正交付的不是“UI 胜利”，而是**多客户端共享核心终于被证实可行**。

## 一个更强的 capstone 版 `TaskFlow` 应达到什么状态

本章结束时，一个更成熟的 `TaskFlow` 应该至少具备这些判断特征：

- app model 以共享 snapshot / mutation / failure 为中心工作
- UI 局部状态与共享状态边界清楚
- preview 和 test double 站在共享 contract 上，而不是平行模型上
- 恢复与错误呈现有明确语义，不靠“视觉上差不多”混过去
- SwiftUI 客户端不再偷偷重写核心规则

这就足以让 `TaskFlow` 真正成为 capstone 系统的一部分，而不是共享核心外面的一层漂亮壳子。

## 本章不追求什么

同样，为了控制范围，这里也要明确不追求：

- 不追求再扩一轮新 feature
- 不追求用复杂动画或导航技巧制造“完成感”
- 不追求把所有 UI 状态都抬到共享层
- 不追求用 SwiftUI 的便利性反过来挤压 core contract

毕业级系统的 UI 加固，重点始终是**与共享核心对齐的质量**。

## 双语关键词

- app model hardening：应用状态模型加固
- shared snapshot：共享快照
- state projection：状态投影
- recovery path：恢复路径
- failure presentation：失败呈现
- preview double：预览替身
- state transition：状态转换
- multi-client reuse：多客户端复用

## 常见错误

### 1. 继续让 `TaskFlow` 自己补共享 contract 的空白

capstone 的目标是对齐，不是继续补丁式自救。

### 2. 把 UI 状态和共享状态混成一团

SwiftUI 的局部交互状态不应反向污染共享语义。

### 3. preview 仍然建立在平行模型或临时假数据上

那会让你错过最宝贵的快速验证机会。

### 4. 只做视觉验证，不做状态流验证

UI 之所以容易失控，就是因为“看起来差不多”常常掩盖 contract 已经跑偏。

### 5. 把 `TaskFlow` 的成熟理解为“终于不需要 CLI”

真正的成熟是证明共享核心能同时服务不同客户端。

## English Recap

TaskFlow hardening should connect the SwiftUI app directly to the hardened shared contract. App models should project shared snapshots, mutations, and failures into UI state while keeping local interaction state separate. Previews, tests, recovery, and failure presentation all become stronger once the app stops compensating for core ambiguity and instead reuses the same foundation as the CLI.

## Drills

1. 列出 `TaskFlow` 中三类必须来自 shared foundation 的状态，以及三类必须留在 UI 客户端的状态。
2. 设计两个 capstone preview，一个验证共享 snapshot，一个验证 shared failure 到 UI 的映射。
3. 解释为什么 mutation 之后“总是自己 reload 一次”在 capstone 阶段通常意味着 contract 还不够清楚。

## Project Handoff

到这里，CLI/Core 和 `TaskFlow` 已经在 capstone 中重新接回同一套 shared foundation。最后一章不会再讲新技术，而会做整套课程的真正收束：总结三条项目线如何统一，明确你现在已经具备的能力边界，以及之后继续深入 Swift 的实际路线图。
