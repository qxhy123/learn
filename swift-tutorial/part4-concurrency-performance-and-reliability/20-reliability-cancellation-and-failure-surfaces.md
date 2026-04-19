# 第20章：可靠性、取消与 failure surface

> Part 4 到这里终于要收束成一个真正的 runtime 工程判断：一个系统不仅要“能执行”，还要在失败、取消、部分完成和资源压力下保持可解释。`TaskCore + TaskCLI` 之前已经有了 `TaskStoreError`，但那只是可靠性故事的一部分。本章要处理的是更完整的 failure surface，以及 Swift 并发语境下取消（cancellation）为什么必须被认真对待。

## 为什么这一章现在出现

如果你回看 Part 3，会发现项目当时的失败面相对简单：

- 标题为空
- 任务不存在
- 任务已经完成
- CLI 命令未知或缺少参数

这些错误当然重要，但它们主要属于输入验证和领域规则。而 Part 4 一旦引入异步 runtime，就会马上出现另一批失败：

- 加载任务失败
- 保存任务失败
- 文件内容损坏或 schema 不匹配
- 命令执行到一半被取消
- 后台保存任务还没完成，用户已经结束流程

这时如果你仍然只会写一句 `"Could not add task"`，系统虽然形式上“处理了错误”，工程上却是不可靠的。因为用户不知道失败发生在哪层，开发者也很难判断是否有部分状态已经提交，测试更无法验证系统在异常路径上到底承诺了什么。

这就是为什么可靠性要成为 Part 4 的最后一章。前面的并发、安全、ownership、性能，最终都要汇到这里：系统到底如何面对不顺利的运行时。

## 从一个 catch-all 错误文案的弱状态开始

看 Part 3 的 CLI 入口，你会发现这种写法：

```swift
do {
    let task = try store.add(title: title)
    return "Added: \(task.title)\n" + render(tasks: store.tasks)
} catch {
    return "Could not add task.\n\(usage)"
}
```

在 Part 3 这没有问题，因为那时我们还没把运行态做实，重点是模块边界与核心行为。可一旦 Part 4 加入 repository、actor、异步保存，这类 catch-all 文案就会迅速变成工程弱点：

- 是输入错了，还是磁盘坏了？
- 是保存还没开始，还是保存失败后内存已修改？
- 是用户取消了，还是系统自己崩掉了？

如果这些区别都被抹平，系统就只剩“成功”或“某种失败”两种表情。那对真实工程来说远远不够。

## 可靠性的第一步：把 failure surface 分层

对当前项目，一个更成熟的失败面应该至少分成几层：

### 1. 输入 / CLI 层

- 未知命令
- 缺少参数
- usage 文本指导

### 2. 领域 / core 层

- `TaskStoreError.emptyTitle`
- `TaskStoreError.taskNotFound`
- `TaskStoreError.taskAlreadyDone`

### 3. 运行时 / persistence 层

- 无法读取数据
- 无法写入数据
- 数据损坏、解码失败
- 状态不一致、提交失败

### 4. 并发控制层

- 任务取消
- 超时
- 后台任务被停止

一旦这样分层，CLI 才能把错误映射成更有信息量的输出；测试也才能验证“这个失败到底属于哪一类”。

这就是 failure surface 的工程意义：不是“错误种类更多”，而是系统开始能解释自己在哪里失败、失败后处于什么状态。

## 取消（cancellation）不是异常插曲，而是运行时正常路径

很多程序员把取消当成罕见事件，仿佛只有 UI 才需要考虑它。其实对异步 CLI 和 runtime 服务来说，取消同样是正常控制流。

对 `TaskCore + TaskCLI`，取消可能来自很多场景：

- 用户中断当前命令
- 上层调用者决定放弃等待
- 后台自动保存被新保存任务替代
- 测试在超时或 teardown 时终止未完成任务

Swift 的取消模型有一个很重要的工程特点：**取消是协作式（cooperative）而不是强制式（preemptive）**。这意味着任务不会凭空在任意指令间被硬杀；你的代码需要在合适位置检查取消，并决定如何收尾。

例如，在保存前后，你可能会显式检查：

```swift
func add(title: String) async throws -> Task {
    try Task.checkCancellation()

    let task = try store.add(title: title)

    try Task.checkCancellation()
    try await repository.save(store)

    return task
}
```

这段代码表达的是非常成熟的运行时态度：取消不是“奇怪失败”，而是这条命令的合法结局之一。你需要决定它应在哪些点被尊重，以及尊重后系统状态如何保持一致。

## 更强的状态：把“成功”的定义收紧

Part 3 的成功含义偏向“内存里改好了”；Part 4 更成熟的成功含义应该是：

- 输入合法
- 领域规则通过
- 持久化提交完成
- 若过程中出现取消或失败，系统没有留下未解释的半成品状态

这会直接改变命令路径的设计。

例如，`add` 命令不再只是：

```swift
let task = try store.add(title: title)
return "Added: \(task.title)"
```

而更像：

```swift
func add(title: String) async throws -> Task {
    try Task.checkCancellation()

    let task = try store.add(title: title)

    do {
        try await repository.save(store)
        return task
    } catch {
        // 这里必须明确：是否回滚？是否保持 dirty state？是否上抛？
        throw RuntimeFailure.saveFailed(underlying: error)
    }
}
```

这段代码背后的工程问题比语法重要得多：

- 保存失败时，`store` 要不要保留已修改值？
- CLI 文案是否应该说“添加失败”，还是“添加尚未持久化”？
- 下次启动时用户看到的是旧状态还是新状态？

这就是为什么可靠性不是简单的 `do/catch` 语法问题。它逼你定义系统承诺。

## 取消与失败要能区分，不要都揉成“出错了”

一个成熟 runtime 非常需要区分 cancellation 和 ordinary failure。

原因很简单：

- 被取消不一定说明系统坏了
- 用户取消通常不是 bug
- 重试策略、日志级别、CLI 输出都可能因此不同

对当前项目，更强的 CLI 映射可能会是：

- 领域错误：给出具体提示，例如 `Task already done`
- persistence 错误：说明加载或保存失败
- cancellation：说明操作已取消，不把它伪装成系统错误

例如：

```swift
func renderCLIError(_ error: Error) -> String {
    switch error {
    case is CancellationError:
        return "Operation cancelled."
    case let error as TaskStoreError:
        return renderCoreError(error)
    case let error as RuntimeFailure:
        return renderRuntimeError(error)
    default:
        return "Unexpected runtime failure."
    }
}
```

这看起来像只是多写了几个 case，但它真正加强的是：用户体验、日志判断、测试覆盖和未来可维护性同时变清楚了。

## 原子性（atomicity）与部分完成：可靠性真正难的地方

一旦系统涉及 load/save，最棘手的问题往往不是“有没有错误”，而是“错误发生时系统已经做到哪一步”。

对于 `TaskCore + TaskCLI` 的 runtime 升级，你至少应该开始问：

- 写文件时，是否先写临时文件再原子替换？
- 保存失败后，内存状态和磁盘状态是否可能分叉？
- list 命令看到的是上次成功提交，还是本次未提交内存态？

这些问题目前不要求你把完整存储系统都做完，但它们决定了 failure surface 是否真实。

一个只会说“save failed”的系统还不够可靠；一个能回答“失败后哪些状态保证没变、哪些状态可能已经变了”的系统才开始像工程。

这也是为什么 Part 4 的可靠性讨论必须和前面的 Actor、ownership 连起来：

- Actor 帮你定义谁串行地改状态
- ownership 帮你定义谁负责提交、谁负责 cleanup
- reliability 则要求你说清楚提交失败后世界处于什么状态

## 重试（retry）与恢复（recovery）不是默认福利

很多人一谈可靠性就想“失败了就重试”。这在某些系统里是对的，但在教程当前阶段，更重要的是建立一个更严格的顺序：

1. 先明确失败类型
2. 先明确操作是否幂等（idempotent）
3. 先明确重试是否可能造成重复写入或旧状态覆盖
4. 然后才决定要不要自动恢复

例如：

- `load` 因临时读错误失败，可能值得重试
- `save` 若已经部分写入，不一定适合盲目重试
- `add` 命令若内部没有稳定操作 ID，重试可能导致重复任务

这说明“可靠性”并不等于“多写几次试试”。成熟系统先定义承诺，再定义恢复策略。

## 测试可靠性，不只是测试 happy path

Part 3 已经让读者知道 XCTest 应优先锁核心行为。到了 Part 4，测试也需要继续升级心智。

对运行时可靠性，更值得补的测试不是“又成功添加了一条任务”，而是：

- save 失败时，CLI 返回哪类错误
- cancellation 发生在保存前后，不同点位行为是否一致
- list 在 load 失败时如何表现
- 重复完成、取消、持久化失败之间是否能被清楚区分

也就是说，Part 4 的测试重点会逐步从“行为存在”转向“failure contract 清楚”。一个工程系统真正稳的地方，往往不是成功路径写得多顺，而是失败路径同样被定义得清楚。

## 从 `TaskStoreError` 到更完整的 runtime contract

这一章最值得带走的判断，是不要把 Part 3 的 `TaskStoreError` 误认为已经等于“项目的全部错误模型”。

Part 3 的错误模型很重要，它把领域失败命名出来了；但 Part 4 要求你再往前走一步：

- 领域失败只是 failure surface 的一层
- runtime failure、cancellation、partial completion 同样需要被建模
- CLI 要对这些层次做准确映射，而不是全部扁平成 generic 文案

换句话说，项目现在真正升级的不只是“错误种类更多”，而是**系统开始具备可解释的运行时契约（runtime contract）**。

当你能回答“这条命令什么时候算成功、取消算什么、失败后状态如何、哪些层错误可以恢复”时，`TaskCore + TaskCLI` 才真的从教程 demo 进入了现代 Swift 工程的门槛。

## 双语关键词

- reliability：可靠性
- failure surface：失败面
- runtime contract：运行时契约
- cancellation：取消
- cooperative cancellation：协作式取消
- `CancellationError`：取消错误
- atomicity：原子性
- partial completion：部分完成
- recovery：恢复
- retry：重试
- idempotency：幂等性
- happy path：顺利路径

## 常见错误

### 1. 把所有失败都揉成一句通用文案

如果 CLI 只能说“操作失败”，用户与开发者都无法判断失败层级，测试也难以验证系统承诺。

### 2. 把取消当成异常噪音

取消是异步系统的正常路径，不应被伪装成普通错误，更不应被完全吞掉。

### 3. 默认保存失败后“应该没事”

没有原子性与提交语义设计，保存失败后到底留下什么状态并不显然。不要用想当然替代契约。

### 4. 没定义幂等性就贸然重试

自动重试如果没有稳定语义支持，可能把一次失败放大成重复写入或状态覆盖问题。

## English Recap

This chapter completes Part 4 by treating reliability as a first-class runtime concern. `TaskCore + TaskCLI` now needs a layered failure surface, explicit cancellation handling, and a stricter definition of success that includes durable completion, not just in-memory mutation. Reliable systems explain what failed, where it failed, and what state remains afterward.

## Drills

1. 把当前项目可能出现的失败分成“输入层、领域层、运行时层、取消层”四类，各写一个例子。
2. 解释为什么“保存失败”与“用户取消”不该共用同一条 CLI 文案。
3. 假设 `done` 在内存中已成功修改，但保存失败了。请写出你认为系统至少要说明清楚的两个状态事实。

## Project Handoff

Part 4 到这里真正完成了它的任务：`TaskCore + TaskCLI` 不再只是一个有 package、有测试的 Swift 教学项目，而是开始具备现代 Swift runtime 的基本判断力。进入下一部分时，`TaskFlow` 或其他客户端将不只是复用一个“会跑的 core”，而是站在一个更懂并发、ownership、性能和可靠性的共享核心之上。
