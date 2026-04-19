# 第33章：Capstone 重建计划

> Part 8 不是“再学四个新知识点”，而是把整套教程真正收束成一条能执行的毕业路径。到了这里，最危险的做法反而是马上动手重写，因为高级阶段最容易把 capstone 误做成一次情绪化升级。本章的任务，就是先把重建计划（rebuild plan）写清楚。

## 为什么这一章现在出现

前七部分结束后，我们已经拥有三条阶段性成果：

- `TaskCLI Lite` 代表最初的语言与建模直觉
- `TaskCore + TaskCLI` 代表共享核心、测试、并发与可靠性判断
- `TaskFlow` 代表站在共享核心之上的 SwiftUI 客户端

如果现在不先写清 capstone 计划，读者极容易掉进两种路径：

- 路径一：把它当成“大重写冲刺”，想到什么改什么
- 路径二：把它当成“把前面再复述一遍”，最后没有真正交付路径

Capstone 的正确起点不是编码，而是明确：

- 我们到底要统一什么
- 哪些行为必须先锁住
- 哪些阶段可以独立推进
- CLI/Core/SwiftUI 三条线如何在本次重建里重新汇合

## 从一个较弱起点开始：一上来就凭印象重构

弱状态通常很像这样：

1. 想到共享抽象，于是先改 `TaskCore`
2. 想到 SwiftUI 体验，于是顺手改 `TaskFlow`
3. 想到 CLI 文案不统一，又回头改命令入口
4. 改着改着才发现测试没有覆盖、术语也不一致

这种方式的问题不是“动作不努力”，而是缺少 capstone 应有的重建顺序：

- 没有先定义共享 contract
- 没有先明确最小成功标准
- 没有先决定验证证据
- 没有区分“硬化已有行为”和“新增系统能力”

最后产出的往往不是一个更稳的系统，而是一堆彼此拖拽的半成品。

## 更强的起点：先把 capstone 目标收缩成三件事

对当前教程主线，一个强而克制的 capstone 目标可以收束为三件事：

### 1. 统一共享语言

把查询、快照、变更、失败面等共享 contract 说清楚，让 CLI 和 `TaskFlow` 不再各自发明术语。

### 2. 硬化共享核心与 CLI 路径

先把最靠近行为真相的一层稳定下来，包括运行时 contract、持久化边界、错误映射与验证面。

### 3. 让 `TaskFlow` 真正站在 hardened core 上

不是重新做一个 UI demo，而是让 app state、持久化、预览、测试和恢复路径都接到新的共享 contract 上。

你会发现，这三个目标没有要求“做一个更大的功能系统”。Capstone 的重点是**系统统一与工程质量**，不是 feature parade。

## Capstone 的输入资产：不要忘记前面所有阶段都是设计材料

很多人做毕业项目时，会错误地把前面内容当作“已经过去的练习”。对本教程不是这样。

Capstone 的输入资产恰恰来自前面所有阶段：

- `TaskCLI Lite` 提供早期直白 API 的可读性标准
- Part 2 提供类型与抽象边界判断
- Part 3 提供 package、测试与 CLI 组织经验
- Part 4 提供并发、可靠性和 failure surface 判断
- Part 5/6 提供 `TaskFlow` 的状态流、客户端边界与 preview/test 经验

所以本章的计划，必须是“汇总既有判断再重建”，而不是把历史全部归零。

## Capstone 的第一步：做一份现状盘点（inventory）

真正开始之前，先盘点当前系统的三个面：

### 共享 contract 面

- 现在有哪些共享类型是真正跨 CLI 和 SwiftUI 复用的
- 哪些错误类型和 snapshot 语义已经稳定
- 哪些命名仍然各说各话

### 客户端面

- CLI 当前通过什么命令组织工作
- `TaskFlow` 当前通过什么 app state 和 model 运转
- 哪些能力只属于某一客户端，不应误抽进共享层

### 基础设施面

- 持久化、日志、时间、配置这些系统依赖停在哪一层
- 哪些 adapter 已经存在
- 哪些系统依赖还直接渗透进了错误位置

这份盘点的目的不是写报告，而是为了确定“哪里需要 redesign，哪里只需要 hardening”。

## Capstone 的第二步：定义最小成功标准（minimum success bar）

一个可执行的 capstone 必须有可验证的完成标准。对本课程，更合理的最小成功标准是：

- CLI 和 `TaskFlow` 使用同一套共享 query / mutation / snapshot 语言
- 共享核心有清楚的运行时失败面和基础验证
- CLI 路径在错误、取消、持久化上不再模糊
- `TaskFlow` 不再依赖平行实现，而是真正消费 hardened core contract
- 章节、README 和项目说明能清楚解释 starter、阶段成果与 capstone 的关系

注意，这些标准都不是“做出更多页面”或“支持更多命令”。它们是系统收束标准。

## 更强的 capstone 分阶段计划

把本次重建拆成三个阶段会更稳：

### 阶段一：统一 contract

- 确定共享 snapshot、query、mutation、failure 语言
- 删除重复命名和重复抽象
- 明确哪些客户端状态不进入共享层

### 阶段二：CLI/Core hardening

- 让命令映射共享 contract
- 硬化持久化、错误、取消与验证面
- 确定 CLI 的文本输出和退出语义站在哪一层

### 阶段三：`TaskFlow` hardening

- 让 app model 消费同一共享 contract
- 重整 preview、test double、持久化入口和恢复路径
- 确保 UI 不再平行重写核心语义

这三阶段顺序非常关键。先做 CLI/Core，不是因为 CLI 比 UI 重要，而是因为**共享行为真相要先站稳，客户端才能可靠复用。**

## 计划里必须显式写上的验证面

Capstone 不是 brainstorming，也不是 purely architectural rewriting。它必须从一开始就带上验证面：

- 共享 contract 是否有足够测试保护
- CLI 命令是否能覆盖关键成功和失败路径
- `TaskFlow` 的 model / preview / test double 是否能表达主要状态
- 文档结构是否能解释这次重建为什么成立

换句话说，计划里应该明确“用什么证据证明系统更强了”，而不是只写“要重构某某层”。

## 一个具体可执行的章节内路线

为了让读者真的能沿教程推进，Part 8 的章内路线可以理解为：

1. 第33章先写计划，冻结术语与阶段目标
2. 第34章专注 CLI/Core 的 contract 和 runtime hardening
3. 第35章再把 `TaskFlow` 接上 hardened core
4. 第36章最后总结统一后的项目线，并给出真实下一步路线

这种安排的价值是：每一章都在上一章交付物上继续工作，而不是“每章都像新项目开头”。

## Capstone 里最该避免的三种冲动

### 1. 新建一个“更高级 demo”

用户已经明确要求不能另起项目线。Capstone 必须统一既有系统，而不是另造一个秀场工程。

### 2. 为了展示高级 Swift 而加技术装饰

macro、builder、复杂泛型只有在真的改善边界时才值得出现，不应成为毕业作品的装饰柜。

### 3. 一边设计一边失去术语纪律

Capstone 最大的收益之一就是统一语言；如果做到最后还是 `TaskService`、`TaskManager`、`TaskRuntimeAPI` 混用，那说明计划没有起作用。

## 从教学角度看，为什么 capstone 必须先计划后实现

因为 Part 8 真正要教给读者的，不只是“完成一个项目”，而是：

- 当系统已经有历史包袱时，如何规划重建
- 如何区分共享 contract 和客户端表面
- 如何让验证和设计一起前进
- 如何在不另起炉灶的情况下提高系统质量

这其实是一种比单次实现更重要的工程能力。很多人会写代码，但不会写“正确的重建顺序”。本章正是在补这块能力。

## 双语关键词

- capstone：毕业项目 / 综合收束项目
- rebuild plan：重建计划
- inventory：现状盘点
- minimum success bar：最小成功标准
- hardening：硬化 / 加固
- staged delivery：分阶段交付
- contract freeze：契约冻结
- integration order：集成顺序
- verification surface：验证面

## 常见错误

### 1. 一开始就同时改 CLI、Core、SwiftUI，缺少阶段顺序

多线同时动手很容易让共享 contract 在半途中不断飘移。

### 2. 把 capstone 目标写成功能清单，而不是系统收束目标

Part 8 的重点是统一与硬化，不是堆更多 feature。

### 3. 忘记前面章节本身就是设计材料

Capstone 不是从零开始，而是站在整套教程累积判断之上。

### 4. 没有明确验证标准就开始重建

没有证据标准的重构，最后只能靠主观感觉宣布完成。

### 5. 把重建计划写成纯架构口号

计划必须能落到实际阶段、实际边界和实际验证任务上。

## English Recap

The capstone should begin with a rebuild plan, not a rewrite impulse. A strong plan inventories the current system, defines a minimum success bar, and stages the work into shared contract alignment, CLI/Core hardening, and TaskFlow hardening. The goal is not more features, but a unified and verifiable system.

## Drills

1. 用三句话分别说明：capstone 要统一什么、要保留什么、不要做什么。
2. 为你心里的任务系统写一个最小成功标准，确保它包含 CLI、core、SwiftUI 三个面。
3. 试着把当前系统问题分成“需要 redesign”和“只需要 hardening”两类。

## Project Handoff

计划已经立住后，下一章会正式进入第一阶段实施：先强化 `TaskCore + TaskCLI` 这条共享行为主线，明确命令如何映射共享 contract，持久化和失败面如何收紧，验证证据如何补齐。等这一层站稳，`TaskFlow` 才有资格接入真正 hardened 的核心。
