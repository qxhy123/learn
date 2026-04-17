# 第1章：这一栈到底在构建什么

## 学习目标

学完本章，你应该能回答：

1. `LangChain`、`LangGraph`、`Deep Agents` 各自在解决什么问题
2. 为什么 Deep Agents 不是一个“重新发明 agent runtime”的项目
3. 维护者为什么必须用三层视角，而不能只看 `deepagents/`

---

## 问题是什么

第一次读 Deep Agents 时，最容易出现两种误判：

- 误判 1：把所有行为都算到 Deep Agents 头上
- 误判 2：把 Deep Agents 误看成“只是一些 prompt 和工具的拼装”

这两种看法都不对。

更准确的说法是：

- `LangChain` 提供 agent primitive
- `LangGraph` 提供 stateful runtime
- `Deep Agents` 提供 opinionated harness

真正要理解的不是某个类名，而是三层怎么接起来。

---

## 哪一层负责什么

### `LangChain`

负责：

- `BaseChatModel` / `BaseTool` / `Runnable`
- `RunnableConfig`、tags、metadata、callbacks 的传播
- agent middleware hook surface，例如 `wrap_model_call`、`before_model`、`wrap_tool_call`

### `LangGraph`

负责：

- `StateGraph` / `CompiledStateGraph`
- Pregel step 执行、reducer、checkpoint、subgraph
- `stream_mode`、`subgraphs=True`、`Runtime` / `ToolRuntime`

### `Deep Agents`

负责：

- `create_deep_agent()` 默认装配
- filesystem / todo / skills / memory / subagent / permissions / profiles 等 middleware 组合
- 默认 general-purpose subagent
- backend adapter 与本地策略边界

---

## 代码在哪里

建议同时打开：

- `deepagents/libs/deepagents/deepagents/graph.py`
- `langchain/libs/langchain_v1/langchain/agents/factory.py`
- `langchain/libs/core/langchain_core/tools/base.py`
- `langgraph/libs/langgraph/langgraph/pregel/main.py`
- `langgraph/libs/prebuilt/langgraph/prebuilt/tool_node.py`

---

## 实现怎么工作

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

---

## 为什么这不是“几个 feature 的堆叠”

如果它只是 feature 堆叠，那么：

- callback 传播不需要考虑上游 config tree
- streaming 可见性不需要考虑 `StreamMessagesHandler`
- compiled subagent 的边界也不需要区分“看得见”和“被拦截”

但实际代码里，这些问题都必须跨三层解释，所以 Deep Agents 更像：

> 一个把上游 runtime 约束成默认工作流的 assembly layer。

---

## 容易踩什么坑

- 坑 1：把 `task`、`nostream`、checkpoint、callback manager 全都说成是 Deep Agents 的机制。
  实际上这些点分别落在不同上游层。

- 坑 2：看到 Deep Agents 有自己的 middleware，就以为上游 agent middleware 不重要。
  实际上 Deep Agents 正是建立在 LangChain agent middleware surface 之上。

- 坑 3：把 bug 归因写成“Deep Agents stream 有问题”，但真正的问题在 LangGraph runtime 或 `langchain_core` callback/config 传播。

---

## 本章小结

- `LangChain` 提供 primitive。
- `LangGraph` 提供 runtime。
- `Deep Agents` 提供默认 harness。
- 维护者读 Deep Agents 时，第一步不是钻实现，而是先认清这三层各自负责什么。
