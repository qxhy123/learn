# 第1章：这一栈到底在构建什么

## 本章回答什么

- `LangChain`、`LangGraph`、`Deep Agents` 各自在解决什么问题，为什么不能只看 `deepagents/`
- Deep Agents 为什么不是“重新发明 agent runtime”，而是把上游能力约束成默认 harness
- 维护者第一次定位问题时，为什么必须先用三层 ownership 视角，而不是直接钻某个类或某个仓库

## 在整套系统中的位置

- 横向主题：`Assembly`
- 前置章节：[README](../README.md)、[前言：如何使用本教程](../00-preface.md)
- 后续章节：[第2章：仓库地图与包边界](./02-repo-map-and-package-boundaries.md)、[第3章：create_deep_agent 作为装配根](./03-create-deep-agent-as-assembly-root.md)、[第4章：Filesystem 与状态模型](../part2-core-runtime/04-filesystem-and-state-model.md)

## 静态结构

这一章是全书的 ownership 起点。它先不追某个具体 bug，而是先把三层栈的职责边界摆正，避免后续把所有行为都错误归因到 Deep Agents。

### 三层各自负责什么

#### `LangChain`

负责：

- `BaseChatModel` / `BaseTool` / `Runnable`
- `RunnableConfig`、tags、metadata、callbacks 的传播
- agent middleware hook surface，例如 `wrap_model_call`、`before_model`、`wrap_tool_call`

#### `LangGraph`

负责：

- `StateGraph` / `CompiledStateGraph`
- Pregel step 执行、reducer、checkpoint、subgraph
- `stream_mode`、`subgraphs=True`、`Runtime` / `ToolRuntime`

#### `Deep Agents`

负责：

- `create_deep_agent()` 默认装配
- filesystem / todo / skills / memory / subagent / permissions / profiles 等 middleware 组合
- 默认 general-purpose subagent
- backend adapter 与本地策略边界

### 代码在哪里

建议同时打开：

- `deepagents/libs/deepagents/deepagents/graph.py`
- `langchain/libs/langchain_v1/langchain/agents/factory.py`
- `langchain/libs/core/langchain_core/tools/base.py`
- `langgraph/libs/langgraph/langgraph/pregel/main.py`
- `langgraph/libs/prebuilt/langgraph/prebuilt/tool_node.py`

## 运行时链路

### 1. `LangChain` 提供的是 primitive，不是最终 harness

在上游，`BaseChatModel.invoke()` / `stream()`、`BaseTool.run()`、`CallbackManager.configure()`、`RunnableConfig.ensure_config()` 这些 primitive 负责把一次模型调用、一次工具调用、一次 callback tree 跑起来。

这些 primitive 解决的是：

- 模型怎么调
- 工具怎么调
- config 怎么传
- callback 怎么树状扩散

但它们不直接回答：

- 默认要不要有 filesystem tools
- subagent 应该怎么组织
- permissions 怎么收口

### 2. `LangGraph` 把 primitive 变成可编排 runtime

当这些 primitive 被放进 graph 里，LangGraph 开始负责：

- state schema 与 reducer
- 每一步 step 的调度
- subgraph 的 namespace 与 checkpoint
- `stream_mode="messages" / "updates" / "custom"`

所以“为什么能流式看到子图 token”“为什么某个 `Command(update=...)` 会冒泡回父图”这类问题，通常先看 LangGraph。

### 3. `Deep Agents` 的价值在于默认装配，不在于重写底层语义

`create_deep_agent()` 做的关键事不是造一套新 runtime，而是把上游 primitive 组装成一个默认 harness：

- 先选 model/profile
- 再拼 middleware 栈
- 再注入 default tools 与 general-purpose subagent
- 最后调用上游 `create_agent()` 产出 compiled graph

也就是说，Deep Agents 的主战场是：

- 默认行为选择
- 装配顺序
- 本地策略边界

### 4. “深”不是模型能力，而是 harness 结构

Deep Agents 之所以叫 “deep”，不是因为默认模型更强，而是因为它默认允许：

- 大上下文任务拆解成 subagent
- 文件系统与状态持续进入 graph
- memory / skills / summarization / permissions 共同工作

这是一种 harness 深度，而不是单次 LLM 调用深度。

## 传播 / 可见性 / 拦截点

### 为什么这不是“几个 feature 的堆叠”

如果它只是 feature 堆叠，那么：

- callback 传播不需要考虑上游 config tree
- streaming 可见性不需要考虑 `StreamMessagesHandler`
- compiled subagent 的边界也不需要区分“看得见”和“被拦截”

但实际代码里，这些问题都必须跨三层解释，所以 Deep Agents 更像：

> 一个把上游 runtime 约束成默认工作流的 assembly layer。

### 维护者在这一章要先记住的三个边界

- callback / config 传播主要属于 `langchain_core`，不是 Deep Agents 自己发明的新机制
- token 是否能被外层看到，优先看 LangGraph 的 stream observer，而不是先猜 Deep Agents middleware
- subagent 是否会被主 agent 拦截，和 subagent 的事件是否会被外层观测到，不是同一个问题

## 扩展接口

这一章不展开完整 API 清单，但需要先把扩展面落到对的层：

- `LangChain`：`BaseTool`、`BaseChatModel`、agent middleware hook surface
- `LangGraph`：`StateGraph`、`Runtime` / `ToolRuntime`、stream mode、checkpoint / subgraph 能力
- `Deep Agents`：`create_deep_agent()` 装配、middleware 顺序、backend / profile / permissions / subagent policy

如果你还没判断清楚自己在改 primitive、runtime 还是本地装配，就不该直接开始改 Deep Agents 代码。

## 常见问题与排障入口

- 坑 1：把 `task`、`nostream`、checkpoint、callback manager 全都说成是 Deep Agents 的机制。实际上这些点分别落在不同上游层。
- 坑 2：看到 Deep Agents 有自己的 middleware，就以为上游 agent middleware 不重要。实际上 Deep Agents 正是建立在 LangChain agent middleware surface 之上。
- 坑 3：把 bug 归因写成“Deep Agents stream 有问题”，但真正的问题在 LangGraph runtime 或 `langchain_core` callback/config 传播。

排障入口建议这样选：

- 怀疑 config、callbacks、run tree：先看 `langchain/libs/core/langchain_core/`
- 怀疑 stream、subgraph、checkpoint、可见性：先看 `langgraph/libs/langgraph/langgraph/`
- 怀疑默认工具、permissions、subagent policy、profile 装配：再回到 `deepagents/libs/deepagents/deepagents/`

## 本章结论

- 谁提供：`LangChain` 提供 primitive，`LangGraph` 提供 runtime，`Deep Agents` 提供默认 harness 与 assembly contract。
- 如何传播：行为先沿 model / tool / callback / config / state / stream 这些上游机制传播，再被 Deep Agents 通过 middleware、profile、backend 重新装配成默认工作流。
- 修在哪层：先判断问题属于 primitive、runtime 还是本地装配；只有默认顺序、策略和 harness wiring 才优先改 `deepagents`。
