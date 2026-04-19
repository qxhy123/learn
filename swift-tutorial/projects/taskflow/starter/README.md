# TaskFlow Starter

## Starter state 的意义

`TaskFlow` 的 starter state 不是“空白 app”，也不是“已经有完整 SwiftUI 工程只差补几行代码”。它描述的是读者在刚进入 Part 5 时的项目理解起点：

- 你已经拥有共享任务领域的主线经验
- 你知道 `TaskCore + TaskCLI` 为什么存在
- 你还没有把这些能力系统地转成 SwiftUI app 的状态流和界面结构

所以 starter 的重点不是工程脚手架数量，而是认知起点是否对齐。

## 在 starter 阶段，读者应该带着什么进入 `TaskFlow`

进入这条线时，较稳的前置理解应包括：

- `TaskCore` 已经承接共享任务模型与基础规则
- SwiftUI 不应重新实现一套平行任务世界
- View 应围绕状态描述界面，而不是命令式 patch 控件
- app 客户端的职责会和 CLI 不同，但共享核心不应变化

如果这些前提不成立，后续 `TaskFlow v1` 很容易退化成“SwiftUI 外壳包住一套新的业务逻辑”。

## Starter 阶段还没有完成什么

在 starter state，下面这些能力还没有被系统做实：

- app 级状态流如何组织
- 列表、表单、导航怎样形成完整交互路径
- 数据如何从共享核心和持久化边界流向 UI
- 异步更新、preview、测试如何围绕 app 状态工作

也就是说，starter 并不是一个“已经做完一半界面”的中间成品，而是一个明确等待 Part 5/6 继续塑形的起点。

## 它和 `TaskCore + TaskCLI` starter / final 的关系

不要把 `TaskFlow` starter 理解成要重复走一遍 Part 3 的 package 起步。`TaskFlow` 的 starter 是建立在已有共享核心之上的客户端起步：

- 前面的 starter 关注 package、module boundary、CLI entry
- 这里的 starter 关注 SwiftUI 心智、状态拥有和 app 结构起点

两者面对的是同一个任务领域的不同问题，而不是互相覆盖。

## 本目录文档的阅读方式

建议把本 README 当成 Part 5 的项目入口说明：

1. 先确认 `TaskFlow` 是共享核心 client，而不是重写项目
2. 再进入 Part 5 四章，理解 SwiftUI 基础与 `TaskFlow v1`
3. 随后看 `milestones/part5-v1.md`，确认第一阶段 SwiftUI 客户端已经站住

本 starter 文档本身是描述性的，不要求先准备特定 IDE 工程再继续阅读。
