# TaskFlow

## 这个项目为什么现在出现

`TaskFlow` 是 Swift 教程在 Part 5 到 Part 6 的 SwiftUI 应用线，但它不是一条与前文脱节的“第二门课”。它出现的前提，是读者已经完成了 `TaskCLI Lite -> TaskCore + TaskCLI` 这条共享任务领域主线，已经知道：

- 为什么核心模型需要稳定边界
- 为什么客户端不该重新发明领域规则
- 为什么并发、可靠性和测试会影响系统真实形状

因此 `TaskFlow` 的角色不是替代前面的 CLI 线，而是把同一个任务领域带进 Apple / SwiftUI 客户端语境。

## `TaskFlow` 如何复用 `TaskCore`

这条项目线的核心判断非常简单，但必须反复强调：`TaskFlow` 是 `TaskCore` 的客户端（client）。

这意味着：

- `Task`、`TaskStatus` 等共享模型应沿用 `TaskCore` 的含义
- 新增、完成、过滤等任务行为应优先建立在共享核心规则之上
- SwiftUI 层主要负责界面组合、状态流、导航和用户交互

换句话说，`TaskFlow` 复用的不是“同名类型”而已，而是共享任务核心已经沉淀出的模型语义和行为边界。

## `TaskFlow` 与 `TaskCLI` 有什么不同

`TaskFlow` 和 `TaskCLI` 的差异，不在领域规则，而在客户端表面：

- `TaskCLI` 以命令、参数、文本输出和脚本化调用为中心
- `TaskFlow` 以列表、表单、导航、异步反馈和 app 状态为中心

两者都应站在同一个共享任务核心之上。这样课程后半段讨论 app architecture 时，读者不会误以为“做了 UI 就可以抛弃 core”。

## 目录说明

- `starter/`：Part 5 起点说明。它描述 SwiftUI 线在刚进入时应具备怎样的最小理解。
- `milestones/part5-v1.md`：Part 5 结束时的 `TaskFlow v1` 里程碑。
- `milestones/part6-architecture.md`：Part 6 结束时的数据流与架构里程碑。
- `final/README.md`：站在当前 SwiftUI 线阶段终点回看，说明 `TaskFlow` 已经达到什么成熟度，以及它如何继续与 CLI/core 主线并存。

## Starter、Milestone、Final 分别表示什么

这条项目线的文档故意按阶段组织，而不是只放一个“最终版本说明”：

- starter state 说明读者在本阶段开头应该理解什么，还没有解决什么
- milestone 说明一个 part 结束时系统在哪些关键能力上明显变强
- final state 则总结当前项目线为何已经具备继续进入下一阶段的资格

这样设计是为了让教程保持连续性。读者看到的不是“突然多了一个 SwiftUI 项目”，而是一条清楚演进的客户端线。

## 这些文档的定位

本目录下的文档是**描述性教程资产**，不是本任务里需要先 build-verify 的工程脚手架说明。它们用来解释：

- `TaskFlow` 在整套 Swift 教程中的位置
- 当前阶段的架构与交互边界
- 它如何复用 `TaskCore`
- 它为何与 `TaskCLI` 并存而不是互相取代

因此本任务不会要求你先拿到某个特定 Xcode 工程骨架，才能理解 Part 5/6 的项目线。重点是把 app 线的结构和判断讲清楚。
