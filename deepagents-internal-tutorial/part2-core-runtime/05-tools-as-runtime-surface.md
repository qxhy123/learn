# 第5章：Tools 作为 Runtime Surface

## 本章回答什么

- 模型看到的 tool surface、真正执行 tool 的 runtime surface、以及最终回到 state / message / parent 的 return surface 分别是什么
- `BaseTool.run()`、`ToolRuntime`、`ToolNode`、backend、subagent handoff 各自负责哪一段链路
- 为什么 `task` 不是“特殊内建魔法”，而是 parent-child delegation 的正式 tool surface
- permissions 为什么首先是 tool-surface policy，而不是整套 graph 的万能控制面

## 在整套系统中的位置

- 横向主题：`Execution`、`Propagation`
- 前置章节：[README](../README.md)、[前言：如何使用本教程](../00-preface.md)、[第3章：create_deep_agent 作为装配根](../part1-foundations/03-create-deep-agent-as-assembly-root.md)、[第4章：Filesystem 与状态模型](./04-filesystem-and-state-model.md)
- 后续章节：[Memory、Skills、Prompt Layering 与 Config 传播](./06-memory-skills-and-system-prompt-layering.md)、[Subagents、任务交接与上下文隔离](./07-subagents-and-context-isolation.md)、[Summarization、Permissions 与安全边界](./08-summarization-permissions-and-safety-boundaries.md)、[附录 D：传播与可见性速查表](../appendix/propagation-and-visibility-cheatsheet.md)

## 静态结构

这一章盯住的不是“某个工具函数怎么写”，而是 tools 怎样被放回 Pregel 主执行路径里理解。

如果把第 4 章当作“状态怎么存、怎么合并”的章节，这一章的工作就是回答另一件事：

> 一次 graph 执行到底怎样从 compile 后的静态图，进入 Pregel loop，再落到 node / tool / subgraph task，最后把结果分别送回 output、stream 与 parent return。

tools 在这条路里不是孤立专题，而是 runtime surface 的一个具体切面。filesystem、subagent delegation、permissions、backend 能力都仍然重要，但都必须放回同一条 Pregel 执行链路里看。

### 代码在哪里

建议同时打开：

- `langgraph/libs/langgraph/langgraph/graph/state.py`
- `langgraph/libs/langgraph/langgraph/pregel/main.py`
- `langgraph/libs/langgraph/langgraph/pregel/_loop.py`
- `langgraph/libs/langgraph/langgraph/pregel/_runner.py`
- `langgraph/libs/langgraph/langgraph/pregel/__init__.py`（只作为 `Pregel` / `NodeBuilder` 的 re-export surface，不是主逻辑入口）
- `deepagents/libs/deepagents/deepagents/graph.py`
- `deepagents/libs/deepagents/deepagents/middleware/filesystem.py`
- `deepagents/libs/deepagents/deepagents/middleware/subagents.py`
- `deepagents/libs/deepagents/deepagents/middleware/permissions.py`
- `langchain/libs/core/langchain_core/tools/base.py`
- `langchain/libs/core/langchain_core/runnables/config.py`
- `langgraph/libs/prebuilt/langgraph/prebuilt/tool_node.py`
- `langgraph/libs/langgraph/langgraph/runtime.py`

### 三层 ownership

| 层 | 对 tools 真正负责什么 |
| --- | --- |
| `LangChain` | 定义 `BaseTool.run()` / `arun()` 的生命周期、输入输出归一化、callback tree、child config patch |
| `LangGraph` | 负责把 compiled graph 变成 Pregel 执行体，在 task 运行阶段注入 `Runtime` / `ToolRuntime`，并把 tool result 接回 graph state、messages、checkpoint、stream |
| `Deep Agents` | 决定默认暴露哪些工具、哪些工具描述和 prompt 规则会进入模型视野、哪些 tool surface 需要权限策略、backend 能力与 subagent handoff |

### 为什么 tool runtime 必须放回 Pregel 主路径里理解

维护时最容易犯的错，是把 “tool 在哪里被定义” 误当成 “tool 怎样被执行”。

真正需要分开的，是以下三件事：

- “模型能看到什么工具”
- “Pregel 在当前 step 里实际跑了什么 task”
- “工具执行完后什么会回到 output / state / stream / parent”

这三件事都与 tool 有关，但它们不是同一层。

如果脱离 Pregel 主路径去谈 `ToolRuntime`，会得到两个错误结论：

1. 误以为 `ToolRuntime` 是 Deep Agents 自己额外发明的一套上下文系统。
2. 误以为 tool output、state update、stream output 本来就是同一条返回线。

本章的判断标准因此很简单：先把 compile、defaults、loop、runner、runtime injection、result surfaces 串起来，再把 Deep Agents 的 tools / subagents / backend 装配插回去看。

## 运行时链路

### 从 `StateGraph.compile()` 到 Pregel：compile 固化了什么

`StateGraph.compile()` 固化 graph shape、output keys、interrupt 配置，但不是执行本体。

这一层要记住的不是“compile 之后图就开始跑了”，而是 compile 把运行时真正要用的静态骨架钉死下来：

- graph 里有哪些 node、channel、edge、branch
- 哪些 key 是 graph output key，哪些只是内部 state key
- interrupt before / after 的断点配置
- 哪些 node 以后会被当成 subgraph、tool node、普通 runnable node 来调度

compile 的意义是把 “业务图长什么样” 固化成一个 Pregel 可执行对象；它本身不推进 step，也不亲自跑 tool。

因此看 Chapter 5 时，`StateGraph.compile()` 更像是 runtime 的边界线：

- compile 之前，你还在定义图。
- compile 之后，你才进入 Pregel 执行路径。

### `Pregel._defaults()` 在运行前装配了什么

`_defaults()` 在进入 loop 前计算并返回 stream modes、output keys、interrupt 设置、checkpointer、store、cache、durability，以及本轮执行要沿用的相关 pre-loop defaults。

这是 Pregel 真正开始跑之前的默认值整理层。它不定义 graph shape，也不等于把后续 runtime assembly 一次做完。

维护者可以把它理解成“先把 loop 需要的默认参数算齐”的阶段。到这里，Pregel 会先确定：

- 这次要不要开 `messages` / `updates` / `custom` 等 stream mode
- 哪些 key 会作为本轮 output keys 被使用
- checkpointer、store、cache、durability 是否存在以及各自取什么值
- interrupt 与其他 pre-loop defaults 怎样传给后续 loop

subgraph stream handling，以及 `Pregel.stream()` / `astream()` 里对 `Runtime` 的构造与 merge，都发生在 `_defaults()` 返回之后、loop 启动之前。

这就是为什么很多“为什么这里能拿到 store / stream / interrupt 配置”的问题，不能直接怪到 tool 实现头上。`_defaults()` 先把 pre-loop defaults 算出来，后面才轮到 `stream()` / `astream()` 继续装 runtime，再交给 loop 和 runner 消费。

### `SyncPregelLoop` / `AsyncPregelLoop` 如何推进 step

`SyncPregelLoop` / `AsyncPregelLoop` 负责推进 Pregel step，不负责定义业务节点。

loop 做的事情，是不断重复 Pregel 的标准节奏：

1. 读取当前可运行的 node / pending work。
2. 基于 state、messages、interrupt 与 scheduler 状态决定本 step 要跑哪些 task。
3. 把这些 task 交给 runner。
4. 收集本 step 的 writes、updates、events、errors。
5. 应用本 step 的结果，再决定是否进入下一步或结束。

这意味着 loop 是“推进器”，不是“业务定义器”：

- 哪个节点会调用模型，来自 compile 后的 graph shape。
- 哪个工具会暴露给模型，来自 agent / middleware 装配。
- 哪个 subgraph 会被 handoff，来自具体 node / tool 的实现。

所以不要把 `SyncPregelLoop` / `AsyncPregelLoop` 误读成“定义业务图的地方”；它们只负责持续推进 step。

### `PregelRunner` 如何把 node、tool、subgraph 变成可执行 task

`PregelRunner` 负责把当前 step 的 node、tool、subgraph 任务真正跑起来。

真正从 “这一步应该跑什么” 走到 “这个东西开始执行了” 的桥梁，在 runner 上。

对 Chapter 5 来说，最关键的是把三类东西统一看成 task：

- 普通 node task：例如模型节点、业务 runnable 节点
- tool task：`ToolNode` 解析模型吐出的 tool call 后，按工具名找到具体 `BaseTool`
- subgraph task：某个节点或 handoff 把 compiled subgraph 当成 child runnable 跑起来

这也是为什么工具不该被看成 Pregel 之外的补充机制。对 runner 来说，tool 不是旁路，它就是当前 step 里被调度起来的一类 task。

`task` 也因此不是“父子代理之间的隐式快捷通道”，而是一种正式 task：parent 先经由标准 tool surface 发起一次 tool invocation，runner 再真正把 handoff 跑起来。

### `Runtime` / `ToolRuntime` 是在哪个阶段注入的

`Runtime` / `ToolRuntime` 是 Pregel 执行路径里的注入面，不是 Deep Agents 私有上下文系统。

它们出现的时机不在 compile，也不在 `_defaults()` 这种运行前准备层，而是在 runner 把当前 step 的 task 真正执行起来的时候。

这里建议把两层运行时分开看：

- `Runtime`：更通用的 graph execution context，面向 runnable / node / graph 执行面
- `ToolRuntime`：tool-specific 的 graph runtime surface，把 `state`、`context`、`config`、`tool_call_id`、`stream_writer` 等信息带给工具实现

工具真正开始执行时，`LangChain` 与 `LangGraph` 才会在这里接起来：

- `BaseTool.run()` / `arun()` 仍是工具调用的基础入口，负责输入归一化、callback tree、config patch、tool lifecycle
- `ToolNode` 决定本次 tool call 要落到哪个 tool 实现
- LangGraph 在 task 执行阶段把 `ToolRuntime` 注入进去
- Deep Agents 的 filesystem 工具、`task` handoff 工具、业务工具再沿着这个 runtime 去碰 backend、child graph 或外部能力

所以这里必须把一句话说死：

- `Runtime` / `ToolRuntime` 是 Pregel 执行路径里的注入面，不是 Deep Agents 私有上下文系统。

### tool output、state update、stream output 分别从哪条路径出来

运行结束后，不同结果不会沿同一条线返回。

tool output 最常见的三条路径是：

1. 普通 tool result 先被包装回 `ToolMessage` 或等价消息结果，进入后续模型可见的 messages 历史。
2. 带 `update` 的返回值会进入 graph state merge 路径，通过 channel / reducer / checkpoint 回到图状态。
3. 运行时事件会沿 stream writer 或 stream mode 进入 `messages` / `updates` / `custom` 等流输出面。

如果是 `task` 这种 handoff 工具，还要再加第四条：

4. child 完成后的折返结果会作为 parent 真正接回的 handoff return surface，被压成有限的消息与允许冒泡的 update，而不是整个 child graph state 原样上浮。

这也是为什么“工具返回了结果”本身没有回答问题。还要继续问：

- 它是 graph-visible output，还是仅仅进入了 message history
- 它是 stream 中途事件，还是最终结果
- 它是否会进入 checkpoint
- 它是否只对 parent 可见

### Deep Agents 的 tool / subagent / backend 装配插在 Pregel 路径的哪一段

Deep Agents 主要插在 compile 之前与 task 执行期间之间的两端，而不是替代 Pregel 本身。

第一段是 execution surface 装配：

- `create_deep_agent()` 通过 middleware、profile、system prompt、tool descriptions 把模型可见的 tool surface 装出来
- `FilesystemMiddleware`、`SubAgentMiddleware`、permissions 相关 middleware 决定哪些工具会出现在模型视野里
- `execute` 是否能暴露，不只看文案，也取决于 backend 是否真的支持对应能力

第二段是 task 运行时落地：

- 模型先在标准 tool-calling surface 上产出 tool call
- `ToolNode` 在当前 Pregel step 中接住这次 tool call
- runner 把对应 tool task 真正跑起来
- Deep Agents 的工具实现借助 `ToolRuntime` 去访问 backend、状态、child graph 或 handoff 上下文

放到具体例子里更容易看清：

- filesystem 工具不是绕开 Pregel 的特殊 IO 系统；它是在 tool task 里通过 `runtime` 找到 backend，并把文件结果接回 state / messages / stream
- `task` 不是额外的“内建代理切换魔法”；它是在标准 tool surface 上暴露的一次 handoff tool，然后在 runner 里被当成 child execution task 跑起来
- permissions 不是图执行的万能总控；它优先限制的是 Deep Agents 暴露给模型的 tool surface，以及这些 tool surface 在当前 harness 里的可用方式

## 传播 / 可见性 / 拦截点

tools 是执行入口，也是传播边界最容易被写错的地方。

### output surface、stream surface、result return surface 不是一条线

- output surface：本轮执行最终产出的 graph-visible 结果。
- stream surface：运行过程中向 consumer 暴露的 `messages` / `updates` / `custom`。
- result return surface：parent、tool caller 或外层 harness 真正接回的折返结果。

这里最容易写错的地方，是把 “stream 里看见了” 误当成 “graph output 已经更新”，或者把 “parent 收到了折返结果” 误当成 “整个 child state 对外可见”。

对维护者来说，最稳妥的排查顺序是：

- 先问这条信息在哪个 surface 上出现
- 再问它是 loop 结束时产出的最终 output，还是 task 执行中途写出去的 stream event
- 最后再问 parent / caller 真正拿回来的 return payload 是什么

### tool surface 是模型可见面，不是全部执行面的总开关

system prompt、tool descriptions、tool schema 影响的是模型“是否选择调用”。这些属于 tool surface 的建模面，不等于 runtime 里已经发生了某次执行，更不等于这次执行拥有任何额外权限。

### callback tree 能看到 tool run，但看见不等于能穿透 child graph

`BaseTool.run()` 通常会让这次 tool invocation 成为 callback tree 里的 child run。这解释了为什么 tracing 往往能看到 `read_file` 或 `task`。

但如果 `task` 里面调用的是 compiled subagent，父级正式拦住的仍然是这次 handoff tool call；child graph 内部是否继续暴露完整 callback / stream 事件，要回到 subgraph runtime 和具体实现判断。

### `ToolRuntime` 暴露的是 graph 上下文，不是全局魔法变量

工具里看得到 `state`、`context`、`stream_writer`，说明这次调用发生在 LangGraph runtime 下；如果脱离 graph context 直接调用某些 backend 或工具实现，很多 graph-aware 语义根本不会成立。

### permissions 首先是 tool-surface policy

permissions 最稳妥的理解方式是：

- 它主要限制模型在当前 harness 里能不能用、怎样用某些工具
- 它主要收口的是 Deep Agents 暴露出来的本地 tool surface
- 它不是对任意 graph node、任意 compiled runnable、任意远端执行环境的通用总控

这就是为什么“给顶层 agent 配了 permissions”不能自动推出以下结论：

- compiled subagent 内部所有逻辑都受同样约束
- remote async subagent 的远端执行环境也被同样保护
- backend 天然就提供了安全的 execute 介质

### 传播问题必须继续拆成四条线

如果你现在关心的是 tool 调用之后 callback、stream、state、parent return 分别怎么走，最安全的做法是把问题拆开：

- 执行线：tool call 是如何被实际执行的
- 观测线：callback tree / tracing 能看到什么
- 流输出线：`messages`、`updates`、`custom` 对外暴露什么
- 结果折返线：什么真正回到 parent / state / checkpoint

这一章先把 tools 作为 runtime surface 讲清。更细的传播矩阵，继续看 [Memory、Skills、Prompt Layering 与 Config 传播](./06-memory-skills-and-system-prompt-layering.md)、[Subagents、任务交接与上下文隔离](./07-subagents-and-context-isolation.md)、[Summarization、Permissions 与安全边界](./08-summarization-permissions-and-safety-boundaries.md) 与 [附录 D](../appendix/propagation-and-visibility-cheatsheet.md)。

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
- 我以为 compile 就开始执行了，为什么看不到运行期对象：因为 `StateGraph.compile()` 只固化 graph shape、output keys 与 interrupt 配置，真正执行要等 Pregel loop 开始。
- 为什么这次 invocation 能拿到 stream/store/checkpointer：先查 `Pregel._defaults()` 怎样整理了本轮 execution envelope，不要先把锅甩给 tool 实现。
- 为什么某个 node 看起来“突然跑起来了”：先查当前 step 里 runner 生成了哪些 task，而不是只盯 node 定义文件。
- `execute` 有时出现、有时没有：先查 `FilesystemMiddleware` 的过滤逻辑，再查 backend 是否实现了 `SandboxBackendProtocol`。
- 工具里拿不到预期的 `state` / `context`：先确认它是不是经由 LangGraph `ToolNode` 执行，而不是被脱离 graph context 直接调用。
- 我能看到 `task`，为什么看不到子代理内部所有调用：因为 `task` 是正式 handoff surface，child graph 内部可见性要另看 subgraph runtime、stream mode 和具体 runnable。
- 顶层 permissions 为什么没挡住 compiled subagent 里的危险逻辑：因为 permissions 主要守的是当前 harness 暴露出来的 tool surface，不是 compiled runnable 的通用安全模型。
- 文件工具返回成功，但状态没有按预期持久：回查 [第4章：Filesystem 与状态模型](./04-filesystem-and-state-model.md)，确认 backend、`files` state channel、reducer、checkpoint 语义是否吻合。
- tracing 里看得到 tool run，但 UI 或外层流看不全：把 callback 可见性、stream visibility、parent return 分开排；先看 [附录 D：传播与可见性速查表](../appendix/propagation-and-visibility-cheatsheet.md)。

## 本章结论

- 谁提供：`StateGraph.compile()` 固化静态图骨架，`Pregel._defaults()` 组装本轮执行参数，`SyncPregelLoop` / `AsyncPregelLoop` 推进 step，`PregelRunner` 真正跑 task，`LangChain` 提供 `BaseTool.run()` 生命周期，`LangGraph` 提供 `Runtime` / `ToolRuntime` 注入面，`Deep Agents` 提供默认 tool surface、handoff 规则、backend 耦合与 permissions policy。
- 如何传播：模型先在 tool surface 上产出 tool call，随后由 runner 在当前 Pregel step 中把 node / tool / subgraph 作为 task 跑起来，再借助 `Runtime` / `ToolRuntime` 读取 graph 上下文，最后分别落到 output surface、stream surface 与 result return surface。
- 修在哪层：tool schema、默认暴露、permissions、backend 耦合问题优先修 `Deep Agents`；tool lifecycle、callback/config 语义优先看 `LangChain`；compile、loop、runner、state merge、subgraph execution、stream/runtime 上下文优先看 `LangGraph`。
