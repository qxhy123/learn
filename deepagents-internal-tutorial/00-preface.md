# 前言：如何使用本教程

## 教程设计理念

这份教程的核心原则只有一句：

> 先分清三层 ownership，再讨论实现细节。

如果你只盯住 `deepagents/libs/deepagents`，很多现象会解释不通。因为 Deep Agents 的大量行为其实是三层协作结果：

- `LangChain / langchain-core` 决定 tool、model、callback manager、`RunnableConfig` 怎么跑
- `LangGraph` 决定 graph、subgraph、checkpoint、stream mode、`Runtime` / `ToolRuntime` 怎么跑
- `Deep Agents` 决定默认 middleware、backend、profile、subagent policy、permissions 怎么装配

因此，本教程不把 LangGraph / LangChain 当成“背景知识”，而把它们当成 Deep Agents 内部教程的一部分。

---

## 推荐的阅读姿势

### 1. 先看 ownership，再看调用链

每章先回答：

- 这个行为属于哪一层
- 这一层给下游暴露了什么 surface
- Deep Agents 是在“直接提供能力”，还是“只是把上游能力装配成默认策略”

### 2. 同时打开上游源码

本教程会频繁同时引用：

- `deepagents/libs/deepagents/deepagents/graph.py`
- `langgraph/libs/langgraph/langgraph/pregel/main.py`
- `langchain/libs/core/langchain_core/runnables/config.py`

不要只看 Deep Agents 本地代码，再靠记忆猜上游行为。

### 3. 把“传播”和“拦截”分开

很多维护者第一次读 subagent / streaming / callbacks 相关代码时，最容易把下面两件事混为一谈：

- 一个事件是不是对外部观察者可见
- 一个行为是不是被父级 middleware 真正包裹或拦截

本教程会反复区分这两件事。

---

## 每章的组织方式

每章都尽量保持同一骨架：

### 问题是什么

先定义维护者真正要判断的问题，而不是先解释 API。

### 哪一层负责什么

显式写出：

- `LangChain`
- `LangGraph`
- `Deep Agents`

### 代码在哪里

给出应同时打开的本地文件和上游文件。

### 实现怎么工作

按调用链解释数据、state、callbacks、stream、middleware 是怎么穿过去的。

### 容易踩什么坑

只保留维护者最容易误判的边界问题。

---

## 学习目标

读完整份教程后，你应该能稳定回答：

1. 一个行为到底属于 LangChain、LangGraph 还是 Deep Agents
2. `create_deep_agent()` 到底是“新 runtime”还是“上游 runtime 的装配根”
3. 为什么 subagent / streaming / callbacks 这类问题必须跨仓库分析
4. 什么时候应该去修上游，什么时候应该只改 Deep Agents 本地策略
5. 做一项功能改动时，最小测试闭环该怎么补

---

## 建议的并行阅读材料

按主题推荐：

### 装配

- `deepagents/libs/deepagents/deepagents/graph.py`
- `langchain/libs/langchain_v1/langchain/agents/factory.py`

### Subagent / Tool Runtime

- `deepagents/libs/deepagents/deepagents/middleware/subagents.py`
- `langgraph/libs/prebuilt/langgraph/prebuilt/tool_node.py`
- `langchain/libs/core/langchain_core/tools/base.py`

### Callback / Config

- `langchain/libs/core/langchain_core/runnables/config.py`
- `langchain/libs/core/langchain_core/callbacks/manager.py`
- `langchain/libs/core/langchain_core/language_models/chat_models.py`

### Streaming / Subgraphs / Checkpoints

- `langgraph/libs/langgraph/langgraph/pregel/main.py`
- `langgraph/libs/langgraph/langgraph/pregel/_messages.py`
- `langgraph/libs/langgraph/langgraph/constants.py`

---

## 范围声明

本教程的重点不是：

- 外部产品文档
- CLI 使用说明
- LangSmith trace UI 教学
- 某个 example 的业务逻辑细节

本教程的重点是：

- 三层栈如何协作
- Deep Agents 在这三层里添加了哪些本地策略
- 这些策略靠什么测试、回调、stream、state contract 被锁住

如果你读到某一章发现问题已经不再是 Deep Agents 本地策略，而是上游 primitive 行为，那么那一章就应该把你带到上游源码，而不是继续在本地仓库里兜圈子。
