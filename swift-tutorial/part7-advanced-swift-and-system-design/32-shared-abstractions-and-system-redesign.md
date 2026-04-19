# 第32章：共享抽象与系统重设计

> Part 7 的最后一章必须回答一个真正决定课程成色的问题：学了这么多高级 Swift 之后，我们到底要不要重设计系统（redesign）？如果要，重设计应该围绕哪些共享抽象发生；如果不要，又该如何抵抗“学完高级特性就想把一切重写一遍”的冲动。

## 为什么这一章现在出现

现在的项目主线已经走到一个非常典型的工程节点：

- `TaskCLI Lite` 提供了最早的任务领域直觉
- `TaskCore + TaskCLI` 建立了模块、测试、并发和可靠性基础
- `TaskFlow` 则证明同一共享核心可以服务图形客户端

这时所有高级话题都会自然汇聚到一个问题上：

**到底哪些东西已经稳定到值得共享，哪些东西只是各自客户端的局部合理性？**

这不是一章“教你大重构”的鼓动文。恰恰相反，本章的目标是让你学会一种更成熟的系统 redesign 判断：

- 不是为了把代码写得更新潮
- 不是为了把每章学过的特性都塞进去
- 而是为了让系统在多个客户端和未来 capstone 压力下更清楚、更稳、更容易继续演进

## 从一个较弱起点开始：三条项目线都能解释自己，但合起来不够顺

这正是很多教程后期最容易出现的问题。每条线单看都合理：

- CLI 有自己的命令入口和输出组织
- `TaskCore` 有自己的 store、runtime、repository、错误模型
- `TaskFlow` 有自己的 app state、preview、持久化协调

可一旦把它们合起来看，就会暴露一些“局部合理，整体发紧”的地方：

- CLI 和 SwiftUI 各自有一套读取/变更协调命名
- 共享快照与客户端状态模型的关系还不够统一
- 某些系统依赖仍然散在多个边界
- 有些抽象只服务一个客户端，却被误放进共享核心

这就是 redesign 的真正触发点。不是“看起来旧”，而是**整条系统线的语言开始不一致**。

## 更强的第一步：把“真正共享的东西”重新命名清楚

系统 redesign 最先要做的，不是搬文件，而是重新命名共享抽象。

对当前项目，一组比较值得重新确认的共享对象可能是：

- `TaskSnapshot`：某一时刻的任务事实快照
- `TaskQuery`：读取意图
- `TaskMutation`：写入意图
- `TaskMutationResult`：变更结果
- `TaskRuntimeFailure`：运行时失败

这些名字重要，不是因为“更架构”，而是因为它们能同时服务 CLI 和 SwiftUI：

- CLI 可以把 `TaskCommand` 翻译成 `TaskQuery` 或 `TaskMutation`
- `TaskFlow` 可以把用户动作翻译成同样的变更意图
- 共享核心和适配层都能围绕同一套词汇工作

这一步看起来只是命名，其实是在做整个系统 redesign 的语言清理。

## redesign 的目标应该是“统一语义”，不是“统一实现”

这是本章最需要说清楚的一点。许多重构失败，正是因为它们误把“共享抽象”理解成“所有实现都应该完全一样”。

对我们的项目线，正确的共享对象更可能是：

- 查询与变更的语义
- 快照与失败面的含义
- 持久化和系统适配层看到的核心数据形状

而不该被强行统一的通常包括：

- CLI 的 usage 文本和退出码语义
- SwiftUI 的导航结构和本地 UI 状态
- preview、表单草稿、瞬时选择状态

所以 redesign 的高级判断，是在统一语义的同时允许客户端保留自己的表面节奏。

## 一个较强的系统形状：让三条线围绕共享操作模型对齐

如果把前几章的高级判断收束成一个更成熟的中间形态，系统可能会更像这样：

```swift
struct TaskSnapshot {
    let tasks: [Task]
    let lastUpdatedAt: Date?
}

enum TaskQuery {
    case all
    case filtered(TaskFilter)
}

enum TaskMutation {
    case create(title: String)
    case markDone(id: Task.ID)
    case delete(id: Task.ID)
}

enum TaskMutationResult {
    case snapshot(TaskSnapshot)
    case created(Task, snapshot: TaskSnapshot)
    case updated(Task, snapshot: TaskSnapshot)
}
```

这类设计的价值不在于“类型更多”，而在于：

- CLI 和 `TaskFlow` 不再各自发明“成功返回什么”的说法
- repository、runtime、preview、测试都能围绕同一套共享语义写
- 未来 capstone 的强化工作可以有清楚抓手，而不是沿着历史偶然命名继续堆

## redesign 还要学会删除：共享抽象越多，不一定越好

到了高级阶段，另一个必须建立的判断是：真正好的 redesign 往往伴随删除。

例如，你可能会发现：

- `TaskServiceProtocol`、`TaskListProviding`、`TaskRuntimeAPI` 三者职责重叠
- 某些针对 CLI 的渲染工具不该留在共享层
- 某些 `TaskFlow` 专属状态包装其实不需要被共享

这时更强的动作不是再加一层总抽象，而是：

- 删除重复协议
- 缩窄公共表面
- 让局部能力回到局部客户端

换句话说，redesign 的成熟度，往往体现在你敢不敢**减少抽象噪音**，而不是继续堆。

## redesign 的顺序应从 contract 开始，而不是从目录开始

这是 capstone 前非常重要的实践顺序：

1. 先确定共享 contract
2. 再确定客户端如何翻译到 contract
3. 再决定哪些实现要搬动
4. 最后才调整目录与 package

如果顺序反过来，先挪包、先改文件树，通常会出现一种假进展：结构看起来更新了，但系统真正共享的语义仍然模糊。

对当前教程主线，这意味着：

- 先定义 `TaskSnapshot` / `TaskMutation` / `TaskQuery` 是否真的是 capstone 共同语言
- 再看 CLI 命令如何映射进去
- 再看 `TaskFlow` 的 app action 如何映射进去
- 最后才决定哪些 README、模块边界、测试结构要一起调整

## redesign 必须保留“弱起点为何曾经合理”的尊重

这是一种经常被忽略的高级判断。教程到了后期，很容易把早期代码看成“全都太弱，应该推翻”。其实不对。

Part 1 到 Part 6 的很多写法，在当时都合理，因为它们服务的是当时的教学目标和工程压力：

- Part 1 需要具体、直白，而不是高度抽象
- Part 3 需要先建立 package 和 test seam，而不是先做统一操作模型
- Part 6 需要先把 app state 站稳，而不是先重写共享 runtime contract

真正成熟的 redesign，不会把过去视为错误，而会把过去视为**阶段性正确**。现在要做的，是为更高层压力重新组织，而不是否定前面的学习路径。

## 共享抽象设计也要照顾验证方式

只要 redesign 真的成功，验证方式会变得更清楚。

例如：

- core tests 可以围绕 `TaskQuery` / `TaskMutation` 写稳定 contract 测试
- CLI tests 可以聚焦“命令如何映射共享 contract，再映射文本”
- `TaskFlow` tests 可以聚焦“UI 意图如何驱动共享 contract，再更新 app state”

这正是“共享抽象是否真的共享”的一个非常现实的判据：如果 redesign 后测试还是各写各的语言，那共享很可能还只是表面。

## Capstone 前的 redesign 不该做什么

为了把范围守住，本章还要明确说出不该做的事：

- 不要为了展示高级特性，把 macro、builder、泛型技巧全部灌进 redesign
- 不要把所有客户端细节抽进共享核心
- 不要在还没有行为保护时做大范围无测试重写
- 不要把 redesign 写成“另起一个全新 demo 系统”

用户已经明确要求课程主线必须连续，所以 Part 7 的 redesign 只能服务既有项目线：`TaskCLI Lite -> TaskCore + TaskCLI -> TaskFlow -> Capstone`。

## 从 redesign 走向 capstone，课程现在真正准备好了什么

走到这里，Part 7 应该完成两件最关键的准备：

### 1. 概念准备

读者终于能用高级 Swift 的视角判断共享抽象，而不是把高级特性当展示柜。

### 2. 项目准备

系统已经具备一次 capstone 级重整的语言基础：知道该统一什么、该保留什么、该删除什么。

这正是 Part 8 需要的起点。Capstone 不应再是“想到哪改到哪”，而应沿着本章收束出的共享 contract 与 redesign 顺序展开。

## 双语关键词

- redesign：重设计
- shared abstraction：共享抽象
- contract：契约 / 合同
- semantic alignment：语义对齐
- mutation：变更意图
- snapshot：快照
- client translation：客户端翻译层
- abstraction pruning：抽象修剪
- staged correctness：阶段性正确

## 常见错误

### 1. 把 redesign 理解成“把学过的高级特性全用上”

高级判断力的核心是克制，而不是展示。

### 2. 误把“共享抽象”当成“共享所有实现”

共享的是共同语义，不是每个客户端的具体表面。

### 3. 从目录调整开始，而不是从 contract 开始

这通常只能带来表面整齐，不能带来真实系统收束。

### 4. 发现重复后继续叠新协议，不敢删除旧抽象

成熟 redesign 必须愿意做抽象修剪。

### 5. 把前几部分的弱状态一律视为错误

过去的设计是阶段性正确；现在的 redesign 是为新压力重新组织，而不是全盘否定。

## English Recap

System redesign at this stage should align shared semantics, not flatten all implementations into one shape. The right move is to clarify shared contracts such as snapshots, queries, mutations, and failures, remove redundant abstractions, and let CLI and SwiftUI remain distinct clients that translate into the same core language. That gives the project a clean foundation for the capstone.

## Drills

1. 写出你认为最值得在 capstone 前统一的三个共享概念，并说明它们为什么同时服务 CLI 与 SwiftUI。
2. 找一个你直觉上想抽进共享层的客户端细节，解释为什么它其实应该留在客户端。
3. 试着按“contract -> translation -> implementation -> package”顺序，口头描述一次你会如何重整当前任务系统。

## Project Handoff

Part 7 到这里完成后，高级 Swift 已经重新落回系统设计与共享抽象判断。接下来的 Part 8 不会另起一个新项目，而是沿着这里收束出的 contract 和 redesign 方向，正式进入 capstone rebuild：先立计划，再分阶段硬化 CLI/Core 和 `TaskFlow`，最后把整套课程收束成清楚的毕业路线图。
