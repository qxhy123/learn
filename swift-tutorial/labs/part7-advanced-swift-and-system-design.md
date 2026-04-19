# Part 7 综合实验：在 `Capstone` 前完成一次有克制的系统重设计

## 对应部分与项目阶段

- 对应部分：Part 7 `part7-advanced-swift-and-system-design`
- 对应项目阶段：`Capstone` 预备重设计阶段
- 关联章节：第 29 章到第 32 章

Part 7 的危险，在于你终于拿到了高级 Swift 的工具箱，于是很容易开始“把抽象当成绩效”。这份 lab 的目标，不是让你堆更多 `some` / `any` / macro / type erasure，而是逼你回答：当系统已经有 CLI、core、SwiftUI 三条表面时，哪些抽象真正帮助统一语义，哪些抽象只是在扩张 API 表面。

## 使用方式

你做这份 lab 时，必须先假设两件事：

1. 现在的系统已经能工作，所以 redesign 不是为了炫耀重写能力。
2. 高级语言工具只有在帮助边界更清楚时才值得引入。

每做一个设计动作，都先问：

- 这个抽象解决的是“重复实现”，还是“重复命名”？
- 这个高级特性会收紧 API，还是扩大 API？

## Integrated Exercises

### 综合练习 1：重画共享抽象地图

请从 CLI、`TaskCore`、`TaskFlow` 三条线同时出发，列出真正共享的语义：

- snapshot
- mutation
- domain error
- persistence intent
- filtering / sorting semantics

然后判断：

- 哪些应成为共享 contract
- 哪些应只停留在单个客户端
- 哪些曾经共享过，但现在其实该删除

### 综合练习 2：为一个高级抽象写“准入审查”

任选一个主题：

- 高级泛型
- protocol family
- type erasure
- result builder
- macro

请写出一份准入清单，至少回答：

- 当前具体重复是什么
- 不用高级特性能否更简单地解决
- 引入后 API 对调用者是更清楚还是更神秘

要求：必须给出一个“这次不该用”的反例，不允许只写成功案例。

### 综合练习 3：做一次系统 redesign 提案

以 `Capstone` 为目标，提出一个最小 redesign：

- 收紧一个共享 contract
- 删除一个多余抽象
- 把一个系统 API 依赖移到更合理的边界

输出格式：

`Current Weakness` / `Proposed Change` / `Why Now` / `Risk`

## Debugging Tasks

### 调试任务 1：`any` 用上了，类型关系却全丢了

症状：

- API 看起来更统一
- 调用处却需要大量 downcast 或手动补类型信息

你的任务：

- 解释这是不是 existential 被错误用于保留关系的问题。
- 判断这里该回到泛型、`some`，还是具体类型。
- 说明“接口更统一”为什么不等于“设计更强”。

### 调试任务 2：macro 消掉样板后，关键逻辑也看不见了

常见症状：

- 生成代码替代了作者对状态和依赖的解释
- 读者知道它能跑，但不知道 contract 在哪

请回答：

- 哪类逻辑适合交给 macro 消除机械样板
- 哪类逻辑必须继续显式保留在源代码里
- 在教程语境里，为什么可读性和可解释性比“更少手写行数”更重要

## Refactoring / Design Tasks

### 设计任务 1：做一次 package boundary 逆向修剪

从“减少暴露面”出发，审查以下东西是否真的应公开：

- shared protocol
- utility extension
- system adapter
- preview helper

要求：

- 能降为 internal 的先降为 internal。
- 能回到具体类型的先回到具体类型。
- 能删的优先删，不要用更高级抽象包住历史包袱。

### 设计任务 2：让系统 API 进入方式更语义化

挑一项系统能力，例如：

- 日期与时间
- 文件路径
- 通知
- URL 打开

写出一个 semantic adapter 方案，说明：

- 为什么系统 API 不应直接到处渗透
- 为什么也不必为了“纯净”而完全拒绝 `Foundation`

## Challenge Tasks

### 挑战 1：为跨客户端查询语义设计最小 DSL

目标：CLI 和 `TaskFlow` 都能表达过滤 / 排序 / 搜索，但不复制业务规则。

约束：

- 优先从具体需求抽象，而不是先造“查询框架”。
- 如果 result builder 真的能让调用更清楚，可以用；否则不要硬上。
- 需要解释你的 DSL 怎样避免把领域语义藏进语法糖。

### 挑战 2：设计一次“删除优先”的 capstone 预重构

规则：

- 至少删除一个你认为不再值得存在的抽象层
- 删除后仍要保持 CLI / core / UI 三方边界更清楚
- 写出回滚策略，说明如果这次删除判断错了，怎样低风险恢复

## 退出标准

完成这份 lab 后，你应该能明确说明：

- 为什么 Part 7 对应的是 `Capstone` 预备阶段，而不是“高级特性秀场”。
- 为什么真正强的系统设计往往体现在删除、收紧、重命名，而不是新增层数。
- 为什么高级 Swift 只有在服务边界和 contract 时才值得进入系统。

## 复盘问题

1. 你最想引入的那个高级工具，真的在解决当前系统的第一痛点吗？
2. 你删除的抽象里，哪一个最能证明“少即是稳”？
3. 如果明天进入 Part 8，哪三个 contract 已经足够成熟，可以当作 `Capstone` 的输入资产？
