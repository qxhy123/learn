# 第34章：Capstone CLI 与核心加固

> 第33章已经明确了 capstone 的顺序：先统一 contract，再做 CLI/Core hardening，再让 `TaskFlow` 接上来。本章的任务，就是落实这条顺序中的第一段实施工作。重点不是“给 CLI 加更多命令”，而是把共享核心与命令行路径真正硬化到可被整个系统依赖。

## 为什么这一章现在出现

在多客户端系统里，最危险的事情之一就是 UI 比核心更先稳定。因为那通常意味着：

- 共享 contract 仍然摇摆
- CLI 与 UI 各自维护一套成功/失败语言
- 测试覆盖停留在局部 happy path

所以 capstone 必须先处理 `TaskCore + TaskCLI`。这不是偏心 CLI，而是承认一个事实：

**共享行为真相越靠近 core，越应该先被收紧。**

如果这一层还不稳，`TaskFlow` 后面接上的只会是一块继续变动的地基。

## 从一个较弱起点开始：CLI 能跑，但 contract 仍然松散

Part 3 和 Part 4 已经让 CLI/Core 比早期强很多，但放到 capstone 标准下，仍可能存在这些弱点：

- 命令与共享 query / mutation 语言没有完全对齐
- 成功返回值有时是任务，有时是字符串拼接，有时是模糊结果
- 持久化失败、取消、领域错误在 CLI 层的映射还不够稳定
- 某些验证只锁住了局部函数，没有锁住整体 contract

这类系统“能工作”，却还不够像毕业级工程。Capstone 的 hardening，正是要把“能工作”推进到“行为边界可解释、可验证”。

## 更强的第一步：让 CLI 明确成为共享 contract 的翻译层

对 capstone 版本，更成熟的 CLI 角色不是“自己直接操作 store”，而是：

1. 解释命令行输入
2. 把输入翻译成共享 `TaskQuery` / `TaskMutation`
3. 调用 hardened runtime
4. 把返回的 snapshot / result / failure 映射成 CLI 输出

这意味着 CLI 的核心职责被收紧成一个翻译层：

```swift
enum TaskCommand {
    case list(filter: TaskFilter?)
    case add(title: String)
    case done(id: Task.ID)
}

func translate(_ command: TaskCommand) -> TaskOperation {
    switch command {
    case .list(let filter):
        return .query(.filtered(filter ?? .all))
    case .add(let title):
        return .mutation(.create(title: title))
    case .done(let id):
        return .mutation(.markDone(id: id))
    }
}
```

此时 CLI 的强点，不再是“自己知道所有业务怎么做”，而是“它能稳定地把用户文本入口对齐到共享语言”。

## Core hardening 的第一项：收紧 snapshot 与 mutation 结果

在 capstone 前，系统很可能已经有多种返回形状：

- 某些 API 返回 `[Task]`
- 某些返回 `Task`
- 某些返回 `Void`
- 某些再额外依赖调用方重新 list 一次

这在局部阶段可以接受，但 capstone 里更强的做法通常是让 mutation contract 更稳定：

- 要么 mutation 明确返回 `TaskMutationResult`
- 要么它至少承诺能提供变更后的 `TaskSnapshot`

这样带来的好处非常实际：

- CLI 成功路径更容易统一渲染
- `TaskFlow` 后续接入时不必再平行做“变更后自己刷新”的另一套语义
- 测试更容易围绕 contract 写断言，而不是围绕偶然返回值写断言

## Core hardening 的第二项：把 failure surface 写成运行时承诺

本章的 hardening 不是只改成功路径。真正更关键的，往往是 failure contract。

对 capstone 级 `TaskCore + TaskCLI`，至少应能清楚区分：

- 输入错误：参数缺失、命令未知、格式非法
- 领域错误：空标题、找不到任务、重复完成
- 持久化错误：加载失败、保存失败、数据损坏
- 取消或中断：任务被取消、操作中止

更成熟的 CLI 输出因此不应只有一句“operation failed”，而要能映射成稳定、可测试的表面：

```swift
func renderCLIError(_ failure: TaskRuntimeFailure) -> String {
    switch failure {
    case .invalidInput(let message):
        return "Input error: \(message)"
    case .domain(let error):
        return renderCoreError(error)
    case .persistenceLoadFailed:
        return "Could not load tasks."
    case .persistenceSaveFailed:
        return "Could not save tasks."
    case .cancelled:
        return "Operation cancelled."
    }
}
```

这一步的真正价值是：系统终于开始对失败也承担表述责任，而不只是成功时才有清楚语言。

## Core hardening 的第三项：补齐最关键的验证链

Capstone 不是把代码弄得更复杂，而是把验证链补齐。对 CLI/Core，更值得补的不是更多 happy path demo，而是这些验证：

- query / mutation contract 测试
- 持久化失败路径测试
- cancellation 或中途中止路径测试
- CLI 命令到共享 operation 的翻译测试
- CLI 输出映射测试

换句话说，本章真正要让系统更强的方式，是让“命令输入 -> 共享 contract -> runtime 行为 -> CLI 输出”这条链被锁住。

## hardening 也包括系统依赖位置的收紧

除了 contract 和 tests，CLI/Core 还需要做另一件事：确认系统依赖停在正确位置。

一个较强的 capstone 版本应该满足：

- `TaskCore` 不直接知道文件路径、标准输出、退出环境
- persistence adapter 负责接触文件系统或 Foundation 编解码
- CLI 层负责 usage、输出、退出语义
- runtime 层负责协调 snapshot、mutation 和失败面

只要这条边界收紧了，后面 `TaskFlow` 接入时就不会被 CLI 的历史偶然细节拖住。

## CLI hardening 不是“让 CLI 更重要”，而是让多客户端系统更可信

这里必须特别说明，因为很多读者一看到本章会误以为课程最后又退回命令行中心。其实不是。

CLI/Core 先 harden 的理由是：

- CLI 是最直接暴露共享 contract 的客户端
- 它的表面简单，适合先把结果和失败语义说清楚
- 一旦这条线稳住，`TaskFlow` 的 app state 才能减少自己的补偿逻辑

也就是说，本章是在为 UI 线路清路，而不是与 UI 竞争。

## 一个更强的中间状态应该长什么样

如果把本章的成果写成系统判断，它至少应呈现为：

- CLI 命令是共享 query / mutation 的文本入口
- runtime 对成功、失败、取消有清楚承诺
- snapshot / result 形状足够稳定，能被多个客户端消费
- 持久化和系统依赖停在 adapter 边界
- 验证覆盖不只锁成功，还锁失败与映射

这时的 `TaskCore + TaskCLI` 才开始像真正可被信任的 shared foundation。

## 本章不追求什么

为了守住范围，这里也要明确说不追求什么：

- 不追求把 CLI 变成完整 command framework
- 不追求引入大量新命令和新 feature
- 不追求把所有客户端逻辑都提前搬进 core
- 不追求为了“更像框架”而做过度泛型化

Capstone 的 hardening 重点始终是 contract、边界与验证。

## 从教程角度看，本章到底让哪三条线重新对齐了

### `TaskCLI Lite`

它最早教会我们的，是文本入口应该直白、行为应该可读。本章保留了这份可读性，但把底层 contract 收紧了。

### `TaskCore + TaskCLI`

它提供了现在真正被 harden 的主体：共享核心、runtime、failure surface、CLI 翻译层。

### `TaskFlow`

虽然本章还没正式处理 UI，但已经为它准备好了更稳定的共享操作模型和快照语言。

这正是 capstone 阶段的项目统一感。

## 双语关键词

- hardening：硬化 / 加固
- shared contract：共享契约
- command translation：命令翻译
- runtime failure：运行时失败
- output mapping：输出映射
- persistence adapter：持久化适配器
- verification chain：验证链
- client surface：客户端表面

## 常见错误

### 1. 把 CLI hardening 做成“加更多命令”

命令数量增加不等于系统更稳，contract 清楚才是关键。

### 2. 只收紧成功路径，不收紧失败面

毕业级系统最容易露馅的地方，正是在失败时说不清自己承诺了什么。

### 3. 让 CLI 继续直接碰共享可变状态或系统依赖

CLI 应是翻译层，不应再次退化成总管脚本。

### 4. 把 UI 的历史补丁逻辑提前塞进 core

shared foundation 应服务多个客户端，而不是被某个客户端反向主导。

### 5. 验证只补 CLI 字符串，不补共享 contract

没有 contract 级验证，CLI 测试很容易只锁住表面排版而不是系统意义。

## English Recap

Capstone hardening starts by making CLI a clean translator into the shared task contract. The core should expose stable snapshots, mutations, and failures, while persistence and environment details stay in adapters. Verification must cover the whole chain from command input to contract execution to CLI output, including failure and cancellation paths.

## Drills

1. 画出一条 `task done 3` 的 capstone 路径，标出命令翻译、共享 mutation、runtime、输出映射四个点。
2. 写出你认为 CLI/Core 在 capstone 中最该补的三个 failure contract 测试。
3. 说明为什么“让 mutation 返回清楚的结果或 snapshot”会同时帮助 CLI 和 `TaskFlow`。

## Project Handoff

共享核心与 CLI 路径现在已经被收紧到足以充当 shared foundation。下一章会沿着同一套 contract 进入 `TaskFlow` hardening：不是做另一个任务系统，而是让 SwiftUI app 真正消费这套 hardened core，并把 preview、测试、恢复和 app state 一起接上来。
