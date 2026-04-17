# Deep Agents / LangGraph / LangChain Internal 教程

## 项目简介

这份教程不再把 `deepagents` 当成一个孤立仓库来读，而是把它放回它真实依赖的三层栈里：

- `LangChain / langchain-core`：提供模型、tool、runnable、callback manager、config 传播这些基础抽象
- `LangGraph`：提供 state graph、Pregel runtime、subgraph、checkpoint、streaming、`ToolRuntime` 这些运行时机制
- `Deep Agents`：在前两层之上装配出一个默认 harness，把 filesystem、todo、skills、subagent、permissions、profiles 等能力组织成维护者可复用的 agent 内核

所以这不是“Deep Agents 怎么用”的快速入门，而是“当三层栈一起工作时，责任边界在哪里、出问题该看哪层、改能力该落哪层”的 maintainer 教程。

---

## 三层图例

后续章节会反复使用这三个 ownership 标签：

| 标签 | 这层负责什么 | 典型源码位置 |
|------|--------------|--------------|
| `LangChain` | 模型调用、tool 执行、callback manager、`RunnableConfig`、agent middleware hook surface | `langchain/libs/core/langchain_core/`、`langchain/libs/langchain_v1/langchain/agents/` |
| `LangGraph` | graph 编排、state/reducer、subgraph、checkpoint、stream mode、`Runtime` / `ToolRuntime` | `langgraph/libs/langgraph/langgraph/`、`langgraph/libs/prebuilt/` |
| `Deep Agents` | 默认 middleware 栈、`create_deep_agent()` 装配、backend/profile 策略、subagent policy、permissions | `deepagents/libs/deepagents/deepagents/` |

如果一个行为解释不清，先问自己一句：它到底属于哪一层。

---

## 目标读者

- 需要维护 `deepagents` SDK 的工程师
- 需要追查 Deep Agents 与 LangGraph / LangChain 边界问题的框架开发者
- 需要扩展 subagent、streaming、callback、backend、profile 能力的贡献者
- 需要把“这次 bug 应该修在上游还是本地 harness”说清楚的内部维护者

默认你已经具备：

- Python 代码阅读能力
- agent / tool calling / state graph 的基础概念
- 愿意同时打开三个仓库对照阅读，而不是只盯住 `deepagents/`

---

## 章节导航

### 开始之前

- [前言：如何使用本教程](./00-preface.md)

### 第一部分：基础认知

| 章节 | 标题 | 重点 |
|------|------|------|
| 第1章 | [这一栈到底在构建什么](./part1-foundations/01-what-deepagents-builds.md) | 先把 LangChain、LangGraph、Deep Agents 三层分别看清 |
| 第2章 | [仓库地图与包边界](./part1-foundations/02-repo-map-and-package-boundaries.md) | 系统梳理三个仓库各自的架构、模块组织、交互链、hook surface、streaming / callback / subagent 边界 |
| 第3章 | [create_deep_agent 作为装配根](./part1-foundations/03-create-deep-agent-as-assembly-root.md) | 说明 Deep Agents 是如何把上游 primitive 组装成 harness 的 |

### 第二部分：核心运行时

| 章节 | 标题 | 重点 |
|------|------|------|
| 第4章 | [Filesystem 与状态模型](./part2-core-runtime/04-filesystem-and-state-model.md) | filesystem tool surface、`files` state、backend、`execute` 与大结果落盘机制如何一起工作 |
| 第5章 | [Subagents、拦截边界与上下文隔离](./part2-core-runtime/05-subagents-and-context-isolation.md) | `task` handoff、compiled subagent、streaming 可见性、callback 传播 |
| 第6章 | [Memory、Skills、Prompt Layering 与 Config 传播](./part2-core-runtime/06-memory-skills-and-system-prompt-layering.md) | memory 如何加载/缓存/注入 prompt，skills 如何按需暴露，以及 callback/config 在三层之间如何流动 |
| 第7章 | [Summarization、Streaming、Permissions 与安全边界](./part2-core-runtime/07-summarization-permissions-and-safety-boundaries.md) | compaction、可见性、权限收口、哪些能力是策略层而不是上游保证 |

### 第三部分：可扩展性

| 章节 | 标题 | 重点 |
|------|------|------|
| 第8章 | [Backend 协议与存储策略](./part3-extensibility/08-backend-protocol-and-storage-strategy.md) | `BackendProtocol` 为什么是 Deep Agents 的 adapter 层 |
| 第9章 | [Provider Profiles、模型路由与 Middleware Surface](./part3-extensibility/09-provider-profiles-and-model-routing.md) | 上游 provider 集成已经提供什么，Deep Agents 又额外改了什么 |
| 第10章 | [如何测试一个 Harness](./part3-extensibility/10-testing-the-harness.md) | 哪些假设应信任上游，哪些必须在本地测试里钉死 |

### 第四部分：维护工作流

| 章节 | 标题 | 重点 |
|------|------|------|
| 第11章 | [像维护者一样阅读 examples](./part4-production-patterns/11-reading-the-examples-like-a-maintainer.md) | 用 example 反推三层职责，而不是把它当 black box demo |
| 第12章 | [如何安全地新增一种能力](./part4-production-patterns/12-how-to-add-a-new-capability-safely.md) | 先判断改动落层，再补 contract、测试、集成和升级验证 |

### 附录

| 附录 | 标题 | 内容 |
|------|------|------|
| 附录 A | [代码阅读检查表](./appendix/code-reading-checklist.md) | 跨三个仓库的最短源码阅读路径 |
| 附录 B | [测试矩阵](./appendix/test-matrix.md) | 改动不同边界时的最小验证集 |
| 附录 C | [Examples 索引与阅读顺序](./appendix/examples-index.md) | 每个 example 最适合回答什么问题、该怎样向下追源码 |
| 附录 D | [传播与可见性速查表](./appendix/propagation-and-visibility-cheatsheet.md) | prompt、config/callback、state、messages、custom 五条线如何区分 |
| 附录 E | [Troubleshooting Playbook](./appendix/troubleshooting-playbook.md) | callback、streaming、permissions、upgrade 边界问题的排障入口 |

---

## 这套教程如何形成一个系统

- [第1章](./part1-foundations/01-what-deepagents-builds.md) 到 [第3章](./part1-foundations/03-create-deep-agent-as-assembly-root.md) 先把三层 ownership、跨仓调用链和 assembly root 讲清，负责建立系统地图。
- [第4章](./part2-core-runtime/04-filesystem-and-state-model.md) 到第7章负责 runtime case studies，把 filesystem、subagent、memory/config、streaming/permissions 这些高频边界逐个拆开。
- [第8章](./part3-extensibility/08-backend-protocol-and-storage-strategy.md) 到第10章负责扩展与验证，回答“新能力该落哪层、怎样测、怎样避免把 harness 改坏”。
- [第11章](./part4-production-patterns/11-reading-the-examples-like-a-maintainer.md) 和第12章负责维护工作流，把 examples 反推成源码入口，再把新增能力的工作流收束成维护者动作。
- 换句话说，这套教程不是按仓库切开写，而是按“先系统地图，再运行时，再扩展面，最后维护动作”来组织。

---

## 推荐阅读路径

### 路径 1：先抓总边界

1. 读 [第1章](./part1-foundations/01-what-deepagents-builds.md)
2. 读 [第2章](./part1-foundations/02-repo-map-and-package-boundaries.md)
3. 读 [第3章](./part1-foundations/03-create-deep-agent-as-assembly-root.md)

### 路径 2：专看 subagent / callback / streaming

1. 读 [第5章](./part2-core-runtime/05-subagents-and-context-isolation.md)
2. 再读 [第6章](./part2-core-runtime/06-memory-skills-and-system-prompt-layering.md)
3. 再读 [第7章](./part2-core-runtime/07-summarization-permissions-and-safety-boundaries.md)

### 路径 3：准备做改动

1. 读 [第8章](./part3-extensibility/08-backend-protocol-and-storage-strategy.md)
2. 读 [第9章](./part3-extensibility/09-provider-profiles-and-model-routing.md)
3. 读 [第10章](./part3-extensibility/10-testing-the-harness.md)
4. 读 [第12章](./part4-production-patterns/12-how-to-add-a-new-capability-safely.md)

### 路径 4：从 Examples 反推三层栈

1. 读 [第11章](./part4-production-patterns/11-reading-the-examples-like-a-maintainer.md)
2. 再读 [附录 C](./appendix/examples-index.md)
3. 按你要查的问题回跳到对应章节

### 路径 5：排 callback / stream / token visibility

1. 读 [第5章](./part2-core-runtime/05-subagents-and-context-isolation.md)
2. 再读 [第6章](./part2-core-runtime/06-memory-skills-and-system-prompt-layering.md)
3. 再读 [第7章](./part2-core-runtime/07-summarization-permissions-and-safety-boundaries.md)
4. 对照 [附录 D](./appendix/propagation-and-visibility-cheatsheet.md)
5. 遇到具体症状时查 [附录 E](./appendix/troubleshooting-playbook.md)

### 路径 6：做上游升级 / 边界 Bug 修复

1. 先读 [第2章](./part1-foundations/02-repo-map-and-package-boundaries.md)
2. 再读 [第6章](./part2-core-runtime/06-memory-skills-and-system-prompt-layering.md)
3. 再读 [第7章](./part2-core-runtime/07-summarization-permissions-and-safety-boundaries.md)
4. 再读 [第10章](./part3-extensibility/10-testing-the-harness.md)
5. 最后对照 [附录 C](./appendix/examples-index.md) 和 [附录 E](./appendix/troubleshooting-playbook.md)

---

## 这份教程会反复回答的三个问题

### 1. 这个行为是谁提供的

例如：

- `CallbackManager.configure()` 是 LangChain
- `StreamMessagesHandler` 和 `subgraphs=True` 是 LangGraph
- `CompiledSubAgent` 不继承顶层 `interrupt_on` 是 Deep Agents 装配策略

### 2. 这个行为是如何传播的

例如：

- `RunnableConfig`、tags、metadata、callbacks 是如何通过 tool / model / subgraph 继续向下传
- token 流为什么会出现在外层 stream consumer，而不是“被主 agent 直接截获”
- 什么只是 UI 不可见，什么才是真正没有进入 state / callback / stream

### 3. 这个问题应该修在哪层

例如：

- 如果是 `nostream` tag 识别逻辑，优先看 LangGraph
- 如果是 `BaseTool.run` 对 callbacks/config 的拼装，优先看 `langchain_core`
- 如果是默认 subagent middleware / permissions / profile 继承规则，优先看 Deep Agents

---

## 与三个仓库的关系

- `deepagents/` 是本教程的 assembly layer 主角
- `langgraph/` 是本教程的 runtime layer 主角
- `langchain/` 是本教程的 primitive / middleware / callback layer 主角

阅读本教程时，建议至少同时打开这些文件：

- `deepagents/libs/deepagents/deepagents/graph.py`
- `deepagents/libs/deepagents/deepagents/middleware/subagents.py`
- `langgraph/libs/langgraph/langgraph/pregel/main.py`
- `langgraph/libs/langgraph/langgraph/pregel/_messages.py`
- `langchain/libs/core/langchain_core/runnables/config.py`
- `langchain/libs/core/langchain_core/tools/base.py`
- `langchain/libs/core/langchain_core/language_models/chat_models.py`
- `langchain/libs/langchain_v1/langchain/agents/middleware/types.py`
