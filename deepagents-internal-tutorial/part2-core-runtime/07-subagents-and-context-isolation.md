# 第7章：Subagents、任务交接与上下文隔离

## 本章回答什么

- `SubAgent`、`CompiledSubAgent`、`AsyncSubAgent` 三种形态分别由谁装配、适合什么边界
- `task` 为什么是 parent -> child handoff 的正式入口，而不是隐式捷径
- general-purpose subagent 为什么默认存在，以及它在 harness 里的职责
- parent state、child state、returned state 为什么必须分开看
- 子代理执行完之后，主线程真正拿回来的是什么，为什么通常是压缩后的 return surface
- 父级审批/权限规则在哪些边界内会继承，在哪些边界外必须由子代理自己负责

## 在整套系统中的位置

- 横向主题：`Execution`、`Isolation`
- 前置章节：[README](../README.md)、[前言：如何使用本教程](../00-preface.md)、[第3章：create_deep_agent 作为装配根](../part1-foundations/03-create-deep-agent-as-assembly-root.md)、[第5章：Tools 作为 Runtime Surface](./05-tools-as-runtime-surface.md)、[第6章：Memory、Skills、Prompt Layering 与 Config 传播](./06-memory-skills-and-system-prompt-layering.md)
- 后续章节：[第8章：Summarization、Permissions 与安全边界](./08-summarization-permissions-and-safety-boundaries.md)

这一章只回答一件事：子代理是怎样被挂进主 agent、怎样接收任务、怎样与主线程隔离、怎样把结果返回。callback tree、stream consumer 可见性、token 级传播矩阵不再在这里展开。

## 静态结构

建议同时打开这些实现文件：

- `deepagents/libs/deepagents/deepagents/graph.py`
- `deepagents/libs/deepagents/deepagents/middleware/subagents.py`
- `deepagents/libs/deepagents/deepagents/middleware/async_subagents.py`
- `deepagents/libs/deepagents/tests/unit_tests/test_subagents.py`
- `deepagents/libs/deepagents/tests/unit_tests/test_async_subagents.py`

### 三种 subagent form

| 形态 | 谁负责真正装配 | 默认执行边界 |
| --- | --- | --- |
| `SubAgent` | `graph.py` + `SubAgentMiddleware._get_subagents()` | Deep Agents 为它补默认 middleware 栈，再构出一个本地 child graph |
| `CompiledSubAgent` | 调用方自己提供 `runnable` | Deep Agents 只把现成 runnable 接进 `task`，内部逻辑按 runnable 自己的 contract 跑 |
| `AsyncSubAgent` | `AsyncSubAgentMiddleware` | 本地只发起远端异步任务，不在本地构完整 child graph |

### `task` 是正式 handoff surface

主 agent 不会“直接调用某个子代理对象”。它调用的是 `task` 这个工具面，`task` 再根据 `subagent_type` 找到对应的 child execution surface。

这层抽象很关键，因为它决定了：

- parent-child 协作仍然遵守标准 tool-calling 语义
- delegation 是一次可审计、可拦截、可返回结果的正式运行时动作
- 子代理的返回面最终要回到一次工具调用结果，而不是强行把 child graph 嵌进 parent history

### 三份状态要分开记

| 状态面 | 由谁持有 | 本章关心什么 |
| --- | --- | --- |
| parent state | 主线程当前图 | 哪些字段会被拿来派生 child 输入 |
| child state | 子代理自己的运行时状态 | 子代理执行中如何演化，不等于 parent 自动可见 |
| returned state | child 结束后允许折返给 parent 的结果面 | 哪些字段会被压缩、过滤、转成 `ToolMessage` 或有限的 update |

`subagents.py` 里的 `_EXCLUDED_STATE_KEYS` 之所以存在，就是为了让这三份状态不要被误当成一份。

## 运行时链路

### 1. `graph.py` 先决定子代理属于哪条装配路径

`create_deep_agent()` 处理 `subagents=[...]` 时，先按声明形态做分流：

1. 声明式 `SubAgent` 会被保留为 spec，等待 `SubAgentMiddleware` 在后续构图时真正实例化。
2. `CompiledSubAgent` 会把现成 `runnable` 作为 inline subagent surface 挂进去。
3. `AsyncSubAgent` 会交给 `AsyncSubAgentMiddleware`，本地只准备异步 delegation 能力。

这一步的核心不是类型命名，而是执行边界：

- declarative form 允许 Deep Agents 继续补自己的默认 harness
- compiled form 明确告诉框架“内部图已经由调用方决定”
- async form 明确告诉框架“真正执行边界不在本地线程内”

### 2. `task` 把 handoff 变成一次正式工具调用

`SubAgentMiddleware` 会构造 `task` 工具。一次典型 handoff 是：

1. parent 触发 `task`
2. `task` 根据 `subagent_type` 找到目标 child surface
3. `_validate_and_prepare_state()` 从 `runtime.state` 派生 `subagent_state`
4. `_EXCLUDED_STATE_KEYS` 被过滤掉，避免 parent 的私有/高噪声状态直接泄漏进去
5. 当前任务说明被包装成新的 `HumanMessage`，作为 child 的直接工作输入
6. child 运行结束后，结果通过 `_return_command_with_state_update()` 折返

对维护者来说，最重要的结论是：

> 子代理执行不是“主线程继续往下跑的同一段 messages”，而是一次经过 `task` 切换的独立执行片段。

### 3. general-purpose subagent 是默认 harness 元素

如果你没有显式提供名为 `general-purpose` 的子代理，`graph.py` 会补一个默认子代理。

它的作用不是教程演示，而是给主 agent 一个随时可用的隔离执行容器：

- 当任务复杂、上下文噪声高、又不需要专用领域子代理时，主 agent 仍然能把工作切出去
- 这保证了“隔离执行”是框架默认能力，而不是只有高级用户才有的附加配置

### 4. child 拿到的不是 parent 全量状态

`_EXCLUDED_STATE_KEYS` 至少会挡住这些典型字段：

- `messages`
- `todos`
- `structured_response`
- `skills_metadata`
- `memory_contents`

这样做的目的很具体：

- 避免把 parent 全历史直接塞进 child，打破上下文隔离
- 避免把 parent 的私有 prompt 材料直接转借给 child
- 避免那些没有清晰 reducer contract 的字段被误传播

因此“child 能访问 parent 的哪些东西”必须通过 handoff 过滤逻辑来回答，而不是通过“它们都在同一线程里”来回答。

### 5. 返回给 parent 的通常是压缩后的 return surface

子代理结束后，parent 通常不会拿到整个 child state。常见返回面是：

- 一个最终 `ToolMessage`
- 少量允许冒泡的 state update
- 若有 `structured_response`，则会先被序列化成 parent 可消费的结果内容

这一步是有意压缩，不是实现偷懒。主线程通常只需要：

- 任务完成了吗
- 子代理给出的结论是什么
- 是否有明确允许回传的结构化结果

而不需要 child 的完整内部轨迹。

### 6. 审批/权限继承只在执行边界内成立

这一章只保留与执行边界直接相关的最小结论：

- 声明式 `SubAgent` 可以继承并本地化父级的一部分 harness 配置，例如 `interrupt_on` / `permissions`
- `CompiledSubAgent` 不会自动吃到父级默认 middleware 栈；如果它内部需要审批或权限策略，必须在 runnable 自己那层实现
- `AsyncSubAgent` 的远端执行环境更不由本地 parent 自动兜底；本地最多控制“是否发起 delegation”以及“如何处理返回”

所以“父级规则是否生效”首先不是传播问题，而是 child 是谁构的、规则被装配到了哪一层的问题。

## 传播 / 可见性 / 拦截点

这一节只保留执行边界所需的最小判断，不再承担传播理论总论。

### 1. parent 可以拦截的是 `task` 这次 delegation

对 parent 来说，最稳定的拦截点是 `task` 自身：

- 可以决定是否允许发起某次 handoff
- 可以在 delegation 发生前暂停或审批

但这不等于 parent 自动深入 child 内部每个节点。

### 2. child 内部规则是否存在，取决于 child 自己的装配方式

- declarative `SubAgent`：Deep Agents 可以在 child graph 构建期把对应规则一起装进去
- `CompiledSubAgent`：内部运行时规则由 runnable 自己负责
- `AsyncSubAgent`：远端权限、审批、执行环境由远端 contract 负责

### 3. parent 真正稳定可见的是返回面，而不是 child 全状态

从执行 contract 来看，parent 最终稳定能依赖的是：

- `task` 是否成功结束
- `ToolMessage` 的结果内容
- 允许折返的有限 state update

如果你现在关心的是传播、stream consumer 可见性、或者 callback tree 的形状，而不是本章的运行时职责，请跳到 Part 3。

## 扩展接口

### 1. 新增声明式子代理

- 在 `subagents=[...]` 中新增 `SubAgent` spec
- 明确它自己的 `tools`、`system_prompt`、`skills`、`permissions`、`interrupt_on`
- 如果它需要专属规则，不要假设会自动从 parent 全量继承

### 2. 接入现成 runnable

- 用 `CompiledSubAgent` 暴露已有 runnable
- 把 child 内部审批、权限、工具限制放进 runnable 自己的装配中
- 只把 parent-child handoff 和结果回传交给顶层 `task`

### 3. 接入远端异步子代理

- 用 `AsyncSubAgent` 表达“本地发起、远端执行、稍后查询结果”的模式
- 本地章节内只维护 task 发起与结果接回的 contract
- 远端安全策略必须在远端 agent/runtime 自己定义

### 4. 调整 return surface

- 需要 parent 只拿摘要时，保持 `ToolMessage` + 受控 state update 的默认收口
- 需要更多结构化结果时，优先明确 `structured_response` 的序列化与回传 contract
- 不要把 child 的临时 state 直接扩成 parent 默认可见面

## 常见问题与排障入口

- 主 agent 为什么总能看到 `task`，却看不到 child 的完整内部状态：因为 `task` 是正式 handoff surface，child 完整状态默认不会整包回传。
- 自定义 `SubAgent` 为什么没拿到主线程的 memory / skills：这是 handoff 过滤的设计结果，不是加载失败；先看 `_EXCLUDED_STATE_KEYS` 和子代理自身配置。
- compiled 子代理为什么没有继承父级审批或权限：因为它走的是 use-as-is runnable 路径；要修 runnable 自己的装配层。
- async 子代理为什么顶层 permissions 挡不住远端行为：因为本地只守 delegation 入口，远端环境必须自己定义安全边界。
- parent 拿到的结果为什么只有摘要：先看 `_return_command_with_state_update()`；默认 contract 就是压缩返回面。
- 想改“哪些字段能回传到 parent”应该查哪里：先查 `subagents.py` 里的状态过滤和返回折返逻辑，再决定是否要改 child 自己的状态 schema。

## 本章结论

- 谁提供：`graph.py` 决定三种 subagent form 的装配路径，`SubAgentMiddleware` / `AsyncSubAgentMiddleware` 提供 handoff 与执行桥接，`task` 提供正式 parent-child 交接面。
- 如何传播：parent 通过 `task` 发起 delegation，框架从 parent state 派生受过滤的 child 输入，child 执行后再以压缩后的 `ToolMessage` 与有限 update 回传。
- 修在哪层：声明式子代理的装配、handoff、返回面问题修 `Deep Agents`；compiled 子代理内部规则修 runnable 自己那层；async 子代理的远端边界修远端 agent/runtime。
