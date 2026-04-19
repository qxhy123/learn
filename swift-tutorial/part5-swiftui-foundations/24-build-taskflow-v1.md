# 第24章：构建 `TaskFlow v1`

> Part 5 的前三章分别解决了 SwiftUI 的三个基础问题：View 如何思考、状态如何流动、列表/表单/导航如何组成 app 结构。现在需要把这些能力收束成一个真正的阶段性项目结果：`TaskFlow v1`。它不是“另起炉灶的新任务系统”，而是共享任务领域上的第一个 SwiftUI 客户端里程碑。

## 为什么这一章现在出现

如果 Part 5 停在概念讲解，读者很容易把前面三章理解成三组互不相干的 SwiftUI 说明文。项目章的作用，就是把这些局部能力重新压回同一条任务领域主线。

更重要的是，这一章要刻意说明 `TaskFlow` 的身份：

- 它复用 `TaskCore` 的模型和核心规则
- 它在用户体验上不同于 `TaskCLI`
- 它目前是一条描述性项目线，而不是要求你在本章里先拿到一份完整 Xcode scaffold 才能继续

这能避免一个常见误解：好像一进入 SwiftUI，前面的 package 工程线就失效了。事实正好相反，`TaskFlow` 的价值正来自前面已经存在的共享核心。

## 从一个较弱起点开始：把 SwiftUI 版任务 app 当成完全重写

如果没有刻意守住项目连续性，很多人会在这里做出一个表面合理、实际很弱的选择：直接为 SwiftUI app 重写一套任务模型、列表逻辑和状态规则。

表面上这会更快，因为 UI 写法看起来更顺手；但它会立刻制造几个长期问题：

- CLI 和 UI 的任务规则开始分叉
- 共享核心失去复用意义
- 后续持久化、同步、测试都要重复写两套

所以 `TaskFlow v1` 的真正目标不是“最短路径做出一个 SwiftUI 列表”，而是**在不牺牲共享核心的前提下，让 SwiftUI 客户端形成第一条完整 app 流程。**

## `TaskFlow v1` 应该具备什么

到 Part 5 结束时，一个强而克制的 `TaskFlow v1` 通常应至少具备这些能力：

- 读取共享任务模型并渲染列表
- 提供一个最小新增任务入口
- 提供基础筛选或状态区分能力
- 允许进入任务详情或二级界面
- 让“完成任务”之类的基础动作在 UI 中有明确落点

这五点看上去并不夸张，但它们刚好能证明三件重要的事：

1. SwiftUI 客户端已经能消费共享领域模型
2. 状态、绑定与组合不再只是局部示例
3. App 结构开始有自己的 UI 节奏，而不是 CLI 输出的视觉改写

## `TaskCLI` 线与 `TaskFlow` 线的差异，不在领域规则，而在客户端职责

这一章必须把两条项目线的差异说清楚。否则读者会很容易把 SwiftUI app 理解成“把 CLI 搬到图形界面”。

实际上两条线共享的是领域与核心行为，不共享的是客户端交互表面：

- `TaskCLI` 以命令、参数、文本输出为中心
- `TaskFlow` 以列表、输入控件、导航、反馈状态为中心

也就是说，`TaskCore` 回答的是“任务系统本身如何工作”，而不同 client 回答的是“用户通过什么媒介与这个系统交互”。这是项目连续性的关键。

## 一个更稳的 `TaskFlow v1` 分层

为了让 SwiftUI 客户端站在共享核心上，而不是压扁边界，一个更成熟的 v1 分层通常是：

- `TaskCore`：任务模型、规则、核心变更语义
- `TaskFlowRuntime` 或 adapter：把共享核心能力包装成更适合 app 调用的接口
- SwiftUI model / screen state：管理列表页、编辑页、详情页的状态与交互
- SwiftUI views：声明式界面结构

例如，列表页的“新增任务”动作不应直接在 Button 内部手工追加数组，而应经过一个更明确的意图流：

```swift
Button("Add") {
    model.createTask()
}
```

而 `model.createTask()` 再去调用共享核心或 runtime 层。这样的设计比“View 里自己顺手改数据”稳得多，因为它保留了 app client 与共享核心之间的清晰边界。

## v1 里先不要过度承诺的东西

Part 5 是 SwiftUI 基础部分，因此 `TaskFlow v1` 还不需要把所有后续架构话题一次做完。当前阶段可以有意识地保持克制：

- 可以先以描述性项目文档说明结构，而不是假装本章已经 build-verified
- 可以先聚焦单窗口/单主流任务管理路径
- 可以先让持久化与更复杂的 async 更新留到 Part 6

这种克制不是退让，而是教学边界。就像 Part 3 不会在一开始就把所有 runtime 风险做满一样，Part 5 也应先把 SwiftUI 的基础客户端形态立住。

## `TaskFlow` 项目文档怎么看

从这一章开始，项目目录下会出现一条与 `TaskCore + TaskCLI` 平行、但并不割裂的文档线：

- `projects/taskflow/README.md`：说明 `TaskFlow` 的整体定位
- `starter/README.md`：说明读者在 Part 5 起点应理解的状态
- `milestones/part5-v1.md`：记录 Part 5 结束时的 v1 里程碑
- Part 6 再继续进入 architecture milestone 与 final 说明

这些文档在本任务里是**描述性文档资产**，不是要求你此刻必须拿到一整套可构建 app 工程。你应把它们当成项目意图和阶段成果的说明层，而不是 build output。

## `TaskFlow v1` 的 stronger state

如果把本章收束成一个更强版本，它应大致呈现出这样的判断：

- SwiftUI app 没有重新发明任务领域，而是在消费共享核心
- View 组合围绕任务语义组织
- 状态拥有关系已经清楚到足以支持列表、编辑和导航
- 项目文档能解释 starter、里程碑和后续架构生长路径

这就够把 `TaskFlow` 从“会显示几行任务的 SwiftUI demo”提升到“共享核心上的第一代 app client”。

## 双语关键词

- client：客户端
- shared core：共享核心
- app runtime：应用运行时
- adapter：适配层
- milestone：里程碑
- starter state：起始状态
- final state：最终阶段状态
- descriptive docs：描述性文档

## 常见错误

### 1. 为了让 SwiftUI 写起来顺手，直接重写一套任务模型

这会让 `TaskFlow` 从共享核心 client 退化成平行项目。

### 2. 以为 UI client 和 CLI client 的差异意味着领域规则也应该分叉

客户端交互可以不同，但任务规则和共享核心不该被拆成两份现实。

### 3. 在 Part 5 就把所有持久化、同步、复杂架构全部做满

这样会冲淡当前章节真正要训练的 SwiftUI 基础判断。

### 4. 把项目文档误解为“必须先有完整 Xcode 工程才能阅读”

本任务中的 `TaskFlow` 文档是描述性的教程资产，重点是解释结构与阶段，而不是要求先准备一套特定 IDE scaffold。

## English Recap

`TaskFlow v1` is the first SwiftUI client built on top of the shared task domain, not a replacement for `TaskCore + TaskCLI`. The goal of Part 5 is to establish view composition, state ownership, list/form/navigation flow, and clear project documentation without pretending that the entire app architecture is already finished.

## Drills

1. 用自己的话区分：`TaskCLI` 与 `TaskFlow` 共享什么，不共享什么？
2. 写出一个你认可的 `TaskFlow v1` 分层，并解释每层为什么存在。
3. 假设有人建议“SwiftUI 比较特殊，所以 UI 端重新定义任务模型更方便”，请反驳这个建议。

## Project Handoff

Part 5 到这里结束时，`TaskFlow v1` 已经建立起第一代 SwiftUI 客户端形态。下一部分不会推翻它，而是继续处理 app 真正绕不开的问题：应用状态、数据流、持久化、异步更新、预览与测试，以及架构如何随着功能增长而不失控。
