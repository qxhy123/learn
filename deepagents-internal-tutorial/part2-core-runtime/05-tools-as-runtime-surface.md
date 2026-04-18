# 第5章：Tools 作为 Runtime Surface

## 本章回答什么

- 模型看到的 tool surface、真正执行 tool 的 runtime surface、以及最终回到 state / message / parent 的 return surface 分别是什么
- `BaseTool.run()`、`ToolRuntime`、`ToolNode`、backend、subagent handoff 各自负责哪一段链路
- 为什么 `task` 不是“特殊内建魔法”，而是 parent-child delegation 的正式 tool surface
- permissions 为什么首先是 tool-surface policy，而不是整套 graph 的万能控制面

## 在整套系统中的位置

- 横向主题：`Execution`、`Propagation`
- 前置章节：[README](../README.md)、[前言：如何使用本教程](../00-preface.md)、[第3章：create_deep_agent 作为装配根](../part1-foundations/03-create-deep-agent-as-assembly-root.md)、[第4章：Filesystem 与状态模型](./04-filesystem-and-state-model.md)
- 后续章节：[Memory、Skills、Prompt Layering 与 Config 传播](./06-memory-skills-and-system-prompt-layering.md)、[Subagents、拦截边界与上下文隔离](./07-subagents-and-context-isolation.md)、[Summarization、Permissions 与安全边界](./08-summarization-permissions-and-safety-boundaries.md)、[附录 D：传播与可见性速查表](../appendix/propagation-and-visibility-cheatsheet.md)

## 静态结构

这一章盯住的不是“某个工具函数怎么写”，而是 tools 在整套系统里承担的 runtime surface。

对维护者来说，tool 至少同时有三层含义：

1. 它是模型可见的 capability surface。模型只能通过 tool schema、tool description、system prompt 里关于工具的说明来决定是否发起调用。
2. 它是 runtime bridge。模型一旦产出 tool call，执行就会从 chat model surface 切到 `BaseTool.run()`、`ToolNode`、`ToolRuntime`、backend 或 subagent。
3. 它是结果折返面。tool output 不只是文本结果；它还可能变成 `ToolMessage`、`Command(update=...)`、state update，或者 parent-child handoff 的回传结果。

因此，filesystem、memory 更新、subagent delegation、permissions 虽然看起来属于不同专题，但在执行路径上都要先经过同一个问题：

> 这次能力到底是以什么 tool surface 暴露给模型，又是沿哪条 runtime 链路被执行的。

### 代码在哪里

建议同时打开：

- `deepagents/libs/deepagents/deepagents/graph.py`
- `deepagents/libs/deepagents/deepagents/middleware/filesystem.py`
- `deepagents/libs/deepagents/deepagents/middleware/subagents.py`
- `deepagents/libs/deepagents/deepagents/middleware/permissions.py`
- `langchain/libs/core/langchain_core/tools/base.py`
- `langchain/libs/core/langchain_core/runnables/config.py`
- `langgraph/libs/prebuilt/langgraph/prebuilt/tool_node.py`

### 三层 ownership

| 层 | 对 tools 真正负责什么 |
| --- | --- |
| `LangChain` | 定义 `BaseTool.run()` / `arun()` 的生命周期、输入输出归一化、callback tree、child config patch |
| `LangGraph` | 在 tool 执行期注入 `ToolRuntime`，并把 tool result 接回 graph state、messages、checkpoint、stream |
| `Deep Agents` | 决定默认暴露哪些工具、哪些工具描述和 prompt 规则会进入模型视野、哪些 tool surface 需要权限策略或 backend 能力 |

### 这章里的核心判断

维护时不要把以下三件事混写：

- “模型能看到什么工具”
- “工具执行时拿到了什么 runtime 上下文”
- “工具执行完后什么会回到 parent / state / stream”

这三件事分别发生在不同层，也因此对应不同的排障入口。

## 运行时链路

### 1. tool call 如何从模型输出进入 tool surface

`create_deep_agent()` 先通过 middleware 和 profile 把 tool surface 装出来，模型看到的通常是：

- filesystem 相关工具，例如 `read_file`、`write_file`、`edit_file`、`glob`、`grep`
- parent-child delegation 用的 `task`
- 其他显式注入的业务工具

这里真正重要的是，Deep Agents 不会绕开上游 agent contract 自己发明一套“特殊工具协议”。模型仍然是在标准 tool-calling surface 上工作。

因此一条典型链路是：

1. `FilesystemMiddleware`、`SubAgentMiddleware` 等把工具和对应说明接进 agent。
2. model 根据当前 prompt 和 tool descriptions 产出 tool call。
3. LangGraph 的 `ToolNode` 接住这次 tool call，并按工具名找到对应的 tool 实现。
4. 后续执行正式离开“模型输出文本”阶段，进入 tool runtime。

`task` 也遵守同一条线。它不是父子代理之间的隐式快捷通道，而是 parent 要显式调用的一次 tool invocation。这就是为什么 parent-child delegation 也必须从 tool surface 来理解。

同理，`execute` 是否真的可用，也不由模型说了算。模型最多只能看见这个工具；真正能不能执行，还要看 middleware 是否暴露了它，以及 backend 是否支持对应能力。

### 2. `BaseTool.run()`、`ToolRuntime`、graph runtime 如何衔接

工具真正开始执行时，要把 `LangChain` 的 tool lifecycle 和 `LangGraph` 的 graph context 拼起来。

这一步的职责分工可以收口成一句话：

> `BaseTool.run()` 管 lifecycle 与 callback/config，`ToolRuntime` 管 graph-aware execution context，Deep Agents 工具实现再利用这两个面去碰 backend 或 subagent。

维护者最该记住的几个点：

- `BaseTool.run()` / `arun()` 仍是工具调用的基础入口。输入归一化、tool run 事件、child callback manager、config patch 都在这里发生。
- `patch_config()` / `get_child()` 让 tool run 在 callback tree 里成为父 run 下面的一个 child。这解释了为什么 tool 调用通常能在 tracing / callback 里被看到。
- `ToolRuntime` 不是 Deep Agents 自己的发明，而是 LangGraph 注入给工具的 graph runtime surface。它把 `state`、`context`、`config`、`tool_call_id`、`stream_writer` 等执行期信息带进工具。
- Deep Agents 的 filesystem 工具会通过 `runtime` 解析 backend，并把 thread-scoped 的文件状态与 backend 介质接起来。
- `task` 工具会通过 `runtime.state` 构造 child 输入状态，把父线程当前允许透传的上下文整理成一次 handoff。

这也解释了两个经常被误判的问题：

1. 工具里能拿到 `config` 或 thread 相关上下文，不等于 Deep Agents 自己维护了一套独立 config 传播系统；多数语义来自 `LangChain` + `LangGraph`。
2. tool run 出现在 callback tree 里，不等于父级 middleware 自动包住了 compiled subagent 内部所有 model/tool 调用；`task` 只正式包住 handoff 自身。

### 3. tool output 如何回到 state / message / parent

tool 并不只会“返回一段字符串给模型”。在这套栈里，tool output 至少有三种常见回路：

1. 返回普通结果，进入 `ToolMessage`，继续成为后续模型推理可见的 message history。
2. 返回带 `update` 的结果，合并进 graph state。filesystem 场景里，这通常意味着文件相关更新最终通过 state channel 和 reducer 回到 `files`。
3. 返回 parent-child handoff 的折返结果。`task` 在 child 完成后，不会把整个 child graph state 原样抛回 parent，而是压成有限的 `ToolMessage` 和少量允许冒泡的 state update。

这条 return surface 决定了“哪个结果能被谁看到”：

- 模型下一轮通常看的是 `ToolMessage`
- graph / checkpoint 看的是 update 后的 state
- parent agent 对 subagent 通常只拿到压缩后的任务结果，而不是 child 内部全部步骤

所以看到某个结果“出现在工具返回里”，还不够。维护者还要继续问：

- 它最终是 message、state update，还是 handoff return
- 它会不会进入 checkpoint
- 它是不是只对当前 parent 可见，而不是对外层流消费者天然可见

## 传播 / 可见性 / 拦截点

tools 是执行入口，也是传播边界最容易被写错的地方。

### 1. tool surface 是模型可见面，不是全部执行面的总开关

system prompt、tool descriptions、tool schema 影响的是模型“是否选择调用”。这些属于 tool surface 的建模面，不等于 runtime 里已经发生了某次执行，更不等于这次执行拥有任何额外权限。

### 2. callback tree 能看到 tool run，但看见不等于能穿透 child graph

`BaseTool.run()` 通常会让这次 tool invocation 成为 callback tree 里的 child run。这解释了为什么 tracing 往往能看到 `read_file` 或 `task`。

但如果 `task` 里面调用的是 compiled subagent，父级正式拦住的仍然是这次 handoff tool call；child graph 内部是否继续暴露完整 callback / stream 事件，要回到 subgraph runtime 和具体实现判断。

### 3. `ToolRuntime` 暴露的是 graph 上下文，不是全局魔法变量

工具里看得到 `state`、`context`、`stream_writer`，说明这次调用发生在 LangGraph runtime 下；如果脱离 graph context 直接调用某些 backend 或工具实现，很多 graph-aware 语义根本不会成立。

### 4. permissions 首先是 tool-surface policy

permissions 最稳妥的理解方式是：

- 它主要限制模型在当前 harness 里能不能用、怎样用某些工具
- 它主要收口的是 Deep Agents 暴露出来的本地 tool surface
- 它不是对任意 graph node、任意 compiled runnable、任意远端执行环境的通用总控

这就是为什么“给顶层 agent 配了 permissions”不能自动推出以下结论：

- compiled subagent 内部所有逻辑都受同样约束
- remote async subagent 的远端执行环境也被同样保护
- backend 天然就提供了安全的 execute 介质

### 5. 传播问题必须继续拆成四条线

如果你现在关心的是 tool 调用之后 callback、stream、state、parent return 分别怎么走，最安全的做法是把问题拆开：

- 执行线：tool call 是如何被实际执行的
- 观测线：callback tree / tracing 能看到什么
- 流输出线：`messages`、`updates`、`custom` 对外暴露什么
- 结果折返线：什么真正回到 parent / state / checkpoint

这一章先把 tools 作为 runtime surface 讲清。更细的传播矩阵，继续看 [Memory、Skills、Prompt Layering 与 Config 传播](./06-memory-skills-and-system-prompt-layering.md)、[Subagents、拦截边界与上下文隔离](./07-subagents-and-context-isolation.md)、[Summarization、Permissions 与安全边界](./08-summarization-permissions-and-safety-boundaries.md) 与 [附录 D](../appendix/propagation-and-visibility-cheatsheet.md)。

## 扩展接口

从维护动作看，tools 相关扩展通常落在这几个入口：

### 1. 扩 tool surface

- 在 middleware 或 agent 装配处新增工具
- 定义 schema、description、是否进入默认 tool set
- 明确这是模型可见能力，还是仅供内部 runnable 使用

### 2. 扩 graph-aware tool implementation

- 需要 thread state、context、stream writer 时，优先沿 `ToolRuntime` 取值
- 需要 child callback tree 时，尊重 `BaseTool.run()` 的 lifecycle，而不是绕开 tool primitive 直接自造执行入口

### 3. 扩 return surface

- 只需要让模型看到结果时，返回普通 tool content 即可
- 需要更新 graph state 时，显式走 update / reducer 语义
- 需要 parent-child handoff 时，仿照 `task` 的“压缩后折返”策略，不要把 child 内部状态整包冒泡

### 4. 扩 policy，而不是假装扩 runtime contract

- 想限制某类工具，优先改 permissions 或 tool injection policy
- 想改变真正的 graph runtime、callback、stream 语义，应回到 `LangChain` / `LangGraph`
- 想改变文件能力落到哪里，应改 backend，而不是只改工具文案

## 常见问题与排障入口

- 工具明明在 prompt 里可见，但运行时报 capability 不存在：先查 middleware 是否真的注入了该工具，再查 backend 是否支持对应能力。
- `execute` 有时出现、有时没有：先查 `FilesystemMiddleware` 的过滤逻辑，再查 backend 是否实现了 `SandboxBackendProtocol`。
- 工具里拿不到预期的 `state` / `context`：先确认它是不是经由 LangGraph `ToolNode` 执行，而不是被脱离 graph context 直接调用。
- 我能看到 `task`，为什么看不到子代理内部所有调用：因为 `task` 是正式 handoff surface，child graph 内部可见性要另看 subgraph runtime、stream mode 和具体 runnable。
- 顶层 permissions 为什么没挡住 compiled subagent 里的危险逻辑：因为 permissions 主要守的是当前 harness 暴露出来的 tool surface，不是 compiled runnable 的通用安全模型。
- 文件工具返回成功，但状态没有按预期持久：回查 [第4章：Filesystem 与状态模型](./04-filesystem-and-state-model.md)，确认 backend、`files` state channel、reducer、checkpoint 语义是否吻合。
- tracing 里看得到 tool run，但 UI 或外层流看不全：把 callback 可见性、stream visibility、parent return 分开排；先看 [附录 D：传播与可见性速查表](../appendix/propagation-and-visibility-cheatsheet.md)。

## 本章结论

- 谁提供：`LangChain` 提供 tool primitive 与 `BaseTool.run()` 生命周期，`LangGraph` 提供 `ToolRuntime` 和 tool-to-graph 的执行桥，`Deep Agents` 提供默认 tool surface、handoff 规则、backend 耦合与 permissions policy。
- 如何传播：模型先在 tool surface 上产出 tool call，随后由 `ToolNode` 与 `BaseTool.run()` 进入执行，再借助 `ToolRuntime` 读取 graph 上下文，最后以 `ToolMessage`、state update 或 parent-child handoff 结果折返。
- 修在哪层：tool schema、默认暴露、permissions、backend 耦合问题优先修 `Deep Agents`；tool lifecycle、callback/config 语义优先看 `LangChain`；state merge、subgraph execution、stream/runtime 上下文优先看 `LangGraph`。
