# TaskCore + TaskCLI Part 3 v1 Milestone

## 当前阶段稳定了什么

Part 3 的 `v1` 版本完成了项目线最关键的一次工程升级：从单 executable 的 `TaskCLI Lite`，进入有清楚模块边界的 `TaskCore + TaskCLI`。

当前里程碑已经稳定的部分有：

- SwiftPM manifest 同时声明了 library product `TaskCore` 与 executable product `TaskCLI`
- 任务领域模型和基础状态变换已经进入 `TaskCore`
- XCTest 直接锁定 `TaskCore` 的行为，而不是只围绕 CLI 文本做表面验证
- `TaskCLI` 负责命令行参数和输出组织，但不再持有核心规则
- 整个 starter package 可以通过 `swift build` 与 `swift test`

## 当前阶段故意没做满什么

这个里程碑还不是“完整任务系统”。它刻意保留了几个后续阶段要继续强化的地方：

- 仍然使用内存 seeded state，而不是文件存储
- 解析、渲染、存储接缝刚刚被看见，还没有发展成更强 runtime surface
- CLI 命令组织保持简单，没有提前引入复杂 command tree
- Part 4 需要继续处理失败面、可靠性与可能出现的并发/I/O 压力

## 为什么这就足够作为 Part 3 的落点

因为 Part 3 的目标不是做出最终产品，而是把 Swift 工程表面立住。只要读者现在已经能清楚回答“哪些代码属于 core，哪些属于 CLI，哪些行为应该先被 XCTest 锁住”，这一部分就完成了它的教学任务。接下来系统才有资格讨论更强的运行时行为。
