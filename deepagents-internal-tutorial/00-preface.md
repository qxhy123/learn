# 前言：如何使用本教程

## 教程设计理念

这份教程的基本原则只有一句：先分清三层 ownership，再讨论具体实现。

Deep Agents 的大量行为都不是单仓库事实，而是三层协作结果：

- `LangChain / langchain-core` 决定 tool、model、callback manager、`RunnableConfig` 和 middleware surface
- `LangGraph` 决定 graph runtime、subgraph、checkpoint、stream mode、`Runtime` / `ToolRuntime`
- `Deep Agents` 决定默认 middleware、backend、profile、subagent policy、permissions 等本地装配策略

因此，本教程不会把 LangGraph / LangChain 当成背景知识，而是把它们视为解释 Deep Agents 的必要组成部分。

## 阅读契约

阅读本教程时，默认接受以下约定：

1. 不把 `deepagents` 当成孤立仓库阅读；遇到边界问题时必须同时看上游
2. 不把“看得见”误当成“被父级拦截”，也不把“当前实现如此”误写成硬契约
3. 每一章都优先回答维护问题：谁负责、怎样传播、哪里可观测、应修哪层、怎样验证
4. 当教程明确标记某个行为是 `Current implementation` 或 `Known limitation` 时，不要把它当成稳定承诺
5. 当教程给出最小测试闭环时，默认那是维护者做改动前后的验证基线

如果你只想记 API 名称而不关心边界、传播和验证，这套教程不会高效。

## 来源标签与稳定性标签

后续章节会同时标记“行为来自哪一层”和“它有多稳定”。

| 标签 | 含义 |
| --- | --- |
| `LC` | LangChain / langchain-core 提供的 primitive、callback、config 传播语义 |
| `LG` | LangGraph 提供的 graph runtime、streaming、subgraph、checkpoint 语义 |
| `DA` | Deep Agents 本地装配、middleware、policy、profile 语义 |
| `Stable mechanism` | 已被上游 contract 或长期实现稳定支持 |
| `Current implementation` | 当前实现如此，但不应写成硬契约 |
| `Known limitation` | 已知缺口或刻意保留的不完整能力 |
| `Test-backed behavior` | 当前有测试或明确代码证据支持 |

使用这些标签的目的，是把“来源”和“稳定性”分开说清，避免混写。

## 每章的统一章法

除少数特例外，每章都尽量沿着同一条章法展开：

1. 先定义维护者真正要判断的问题，而不是先堆 API
2. 显式写出 `LC`、`LG`、`DA` 三层分别负责什么
3. 给出建议同时打开的本地文件和上游文件
4. 按调用链解释 state、messages、callbacks/config、stream、prompt / policy 是怎么流动的
5. 标出哪些点是稳定机制，哪些只是当前实现，哪些是已知限制
6. 收束到维护动作：这个问题应该修哪层，最小验证闭环是什么

这套章法的目标不是让所有章节形式一致，而是保证读者每次都能快速定位相同类型的信息。

## 哪些章节是受控例外

以下章节会偏离标准骨架，但偏离是受控的：

- [第2章：仓库地图与包边界](./part1-foundations/02-repo-map-and-package-boundaries.md) 更像跨仓系统地图，重点是建立边界与文件入口，而不是展开单一案例
- [第16章：像维护者一样阅读 examples](./part4-maintenance-and-extension/16-reading-the-examples-like-a-maintainer.md) 以样例反推系统行为，重点是构造阅读入口，而不是重复通用模板
- 各附录更偏检索和执行清单，例如测试矩阵、传播速查表、排障手册，它们服务于维护动作，不承担完整叙述职责

这些例外仍然遵守同一个总目标：帮助维护者分清 ownership、传播路径、观测点和修复落层。

## 推荐的两种读法

### 读法一：顺着系统展开

适合第一次完整阅读：

1. 先读本前言，接受标签体系和阅读契约
2. 读 Part 1，建立三层边界和 assembly root
3. 读 Part 2，理解状态、上下文和执行机制
4. 读 Part 3，专门处理传播、可见性和观测
5. 读 Part 4，把理解收束到扩展、测试、排障和维护工作流

### 读法二：按维护问题跳读

适合已经在改代码或排问题：

- 改边界判断或装配策略时，从 Part 1 开始
- 查 state、subagent、tool runtime 时，从 Part 2 开始
- 查 callback、config、stream、visibility 时，从 Part 3 开始
- 做扩展、补测试、做升级回归或排障时，从 Part 4 和附录开始

## 建议的并行阅读材料

建议至少同时打开以下文件，与教程交叉阅读：

### 装配与边界

- `deepagents/libs/deepagents/deepagents/graph.py`
- `langchain/libs/langchain_v1/langchain/agents/factory.py`

### Subagent / Tool Runtime

- `deepagents/libs/deepagents/deepagents/middleware/subagents.py`
- `langgraph/libs/prebuilt/langgraph/prebuilt/tool_node.py`
- `langchain/libs/core/langchain_core/tools/base.py`

### Callback / Config / Prompt

- `langchain/libs/core/langchain_core/runnables/config.py`
- `langchain/libs/core/langchain_core/callbacks/manager.py`
- `langchain/libs/core/langchain_core/language_models/chat_models.py`

### Streaming / Subgraphs / Checkpoints

- `langgraph/libs/langgraph/langgraph/pregel/main.py`
- `langgraph/libs/langgraph/langgraph/pregel/_messages.py`
- `langgraph/libs/langgraph/langgraph/constants.py`

## 范围声明

本教程不以以下内容为重点：

- 外部产品文档或 CLI 使用说明
- LangSmith UI 教学
- 某个 example 的业务逻辑细节
- 与三层边界无关的泛化 agent 概念复述

本教程重点解释的是：

- 三层栈如何协作
- Deep Agents 在其中增加了哪些本地策略
- 这些策略靠哪些 code path、测试和观测点被约束

如果某一章已经把问题带到上游 primitive 或 runtime contract，那一章就应把你送到上游源码，而不是继续让你在本地仓库里兜圈子。
