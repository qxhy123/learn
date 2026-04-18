# 第16章：像维护者一样阅读 Examples

## 本章回答什么

- 为什么 examples 在维护工作流里应该被当成“证据入口”，而不是“官方 contract 目录”
- 读 `deepagents/examples/` 时，怎样快速区分 example 本地约定、Deep Agents 装配约定、上游 runtime / primitive 约定
- 哪些样本最适合回答装配、部署、outer loop、评测优化这些不同问题
- 什么时候该继续追源码，什么时候该停止从 example 推理，回到 Part 3 或附录
- 第 13 到 15 章判断完边界、provider 与测试之后，为什么这一章是维护工作流后半段的“证据校准”步骤

## 在整套系统中的位置

- 这一部分默认假设你已经读过 Part 1 和 Part 2。
- 如果当前问题和传播、可见性、callback tree 有关，先回看 Part 3。
- 横向主题：`Maintenance`、`Examples as evidence`、`Code-reading workflow`
- 前置章节：[第13章：Backend 协议、存储介质与执行边界](./13-backend-protocol-and-storage-strategy.md)、[第14章：Provider Profiles、模型解析与 Middleware Surface](./14-provider-profiles-and-model-routing.md)、[第15章：如何测试一个三层栈 Harness](./15-testing-the-harness.md)
- 配套索引：[附录 C：Examples 索引与阅读顺序](../appendix/examples-index.md)、[附录 E：Troubleshooting Playbook](../appendix/troubleshooting-playbook.md)
- 后续章节：[第17章：如何安全地新增一种跨三层能力](./17-how-to-add-a-new-capability-safely.md)

第 13 到 15 章先教你判断问题属于哪层、哪些默认策略会影响行为、怎样用测试钉住回归。到了这里，维护工作流进入后半段：你要开始把 example 当成证据，去验证“这个行为到底是谁装出来的、谁暴露出来的、谁只是顺带演示了它”。

## 静态结构

这一章保留 example-index 的写法，但这是一个受控例外：顶层仍按 Part 4 的共享合同组织，内部才用样本矩阵展开。原因很简单，examples 不是一组平行概念，而是一组读源码入口。

先把 `deepagents/examples/` 分成三类：

| 类别 | 代表目录 | 最适合回答什么 |
| --- | --- | --- |
| harness 装配样本 | `deep_research`、`content-builder-agent`、`text-to-sql-agent` | `create_deep_agent()` 被怎样参数化，memory / skills / tools / subagents / backend 怎样组合 |
| runtime / deployment 样本 | `async-subagent-server`、`deploy-*`、`nvidia_deep_agent` | LangGraph server、远端 thread/run、MCP、sandbox、部署配置怎样接入 |
| outer loop / meta-harness 样本 | `ralph_mode`、`better-harness` | 哪些能力该放在 graph 外层，eval 驱动优化如何围住 harness 本身 |

读例子前先把三类标签写在旁边：

- `example 本地约定`
  例如 `subagents.yaml` loader、FastAPI server、REPL、eval config、README 驱动流程。
- `Deep Agents 装配约定`
  例如 `create_deep_agent()`、memory / skills / permissions / backend / profile。
- `上游 runtime / primitive 约定`
  例如 callback manager、`RunnableConfig`、`stream_mode`、checkpoint、tool runtime。

如果这三个标签还没分开，就不要急着把 example 里的 wiring 当成结论。

## 运行时链路

把 example 当证据来读，建议固定走这条维护者链路：

### 1. 先找装配入口，不要先看 README 文案

优先定位这些地方：

- `create_deep_agent()`
- `langgraph.json`
- `deepagents.toml`
- `run_non_interactive(...)`
- `SQLDatabaseToolkit(...)`
- server / supervisor 的入口函数

这些位置真正决定了“样本把哪一层能力接到了哪一层”。

### 2. 再标出 example 自己新增了什么

你要先判断哪些只是样本私有层：

- `subagents.yaml` loader
- FastAPI server
- eval runner
- CLI REPL
- 业务 prompt
- 研究型 patch / scorecard / keep-or-discard 逻辑

这些内容常常解释“这个 example 为什么这么工作”，但不自动解释“框架为什么必须这样工作”。

### 3. 然后追到 Deep Agents 装配层

对大多数样本，下一站都还是：

- `deepagents/libs/deepagents/deepagents/graph.py`
- `deepagents/libs/deepagents/deepagents/middleware/subagents.py`
- `deepagents/libs/deepagents/deepagents/middleware/async_subagents.py`
- `deepagents/libs/deepagents/deepagents/middleware/memory.py`
- `deepagents/libs/deepagents/deepagents/middleware/skills.py`
- `deepagents/libs/deepagents/deepagents/backends/filesystem.py`

这一步的目的不是“把所有实现看完”，而是确认 example 传进去的参数最终收敛成了什么 harness contract。

### 4. 最后再去上游确认执行语义

如果你查的是 callback、stream、tool runtime、checkpoint、remote thread/run 生命周期，这些都不是 example 自己定义的。最终要回到上游：

- `langchain/libs/core/langchain_core/tools/base.py`
- `langchain/libs/core/langchain_core/language_models/chat_models.py`
- `langchain/libs/core/langchain_core/runnables/config.py`
- `langgraph/libs/langgraph/langgraph/pregel/main.py`
- `langgraph/libs/langgraph/langgraph/pregel/_messages.py`

### 5. 六个主样本分别回答什么

| Example | 真正最值得看的地方 | 维护者最该追的下一站 |
| --- | --- | --- |
| `deep_research` | specialized harness，不是“Deep Agents 内建 research mode” | `graph.py`、`middleware/subagents.py` |
| `async-subagent-server` | remote async task 协议，不是“多一个 `async def`” | `middleware/async_subagents.py`、server 的 `/threads` `/runs` |
| `content-builder-agent` | 文件化 harness；最适合分辨 filesystem primitive 与 example helper | `memory.py`、`skills.py`、`filesystem.py` |
| `text-to-sql-agent` | LangChain toolkit 接入 Deep Agents harness 的标准样本 | `SQLDatabaseToolkit`、`graph.py` |
| `ralph_mode` | fresh context outer loop，不是 graph 内部循环教程 | 先确认 CLI boundary，再回 `graph.py` |
| `better-harness` | harness 自己成为优化对象的研究样本 | `core.py`、`agent.py`、`runners.py`、`patching.py` |

## 传播 / 可见性 / 拦截点

examples 最容易误导维护者的地方，不是代码本身，而是你会不知不觉把“看到了某种现象”和“知道它的 owner layer”混成一句。

### 哪些问题不要继续靠 example 推理

如果你已经进入下面这些问题，就该先回系统章节，而不是继续在 example 目录里找“类似写法”：

- callback tree 怎样接起来
- token 为什么出现在外层流里
- `subgraphs=True` 时哪些事件会被消费者看到
- `nostream` 过滤了什么，没过滤什么
- parent / child state、结果折返、阶段事件分别沿哪条线走

统一回跳规则如下：

- streaming 的说明统一回看第9章到第12章：[第9章](../part3-propagation/09-propagation-overview-and-four-lanes.md)、[第10章](../part3-propagation/10-callbacks-config-and-callback-manager.md)、[第11章](../part3-propagation/11-streaming-visibility-and-selective-exposure.md)、[第12章](../part3-propagation/12-subagent-propagation-matrix-and-maintainer-recipes.md)
- subagent + callback 的混合说明统一回看第10章与第12章：[第10章](../part3-propagation/10-callbacks-config-and-callback-manager.md)、[第12章](../part3-propagation/12-subagent-propagation-matrix-and-maintainer-recipes.md)
- 可见性速查表回跳统一回看第11章 + 附录 D：[第11章](../part3-propagation/11-streaming-visibility-and-selective-exposure.md)、[附录 D](../appendix/propagation-and-visibility-cheatsheet.md)

### 例子能提供什么“传播证据”

例子依然有价值，但它提供的是“哪条线值得追”的证据：

- `deep_research` 能证明 `task` tool、research prompt 与 subagent wiring 是怎样拼起来的
- `async-subagent-server` 能证明远端任务状态是 live 查询而不是会话复述
- `content-builder-agent` 能证明 memory / skills / filesystem 这三条装配线怎样在一个样本里同时出现
- `ralph_mode` 能证明 outer loop 可以在 graph 外，而不是 graph 内必有循环节点

但它们不能替代 Part 3 去定义传播 contract。

## 扩展接口

维护者最常做的，不是“新增一个 example”，而是把 example 当成路由表，判断下一步该读哪一层、修哪一层。下面这张表就是最实用的扩展接口。

| 你在查什么 | 先看哪个 example | 再追哪里 |
| --- | --- | --- |
| memory / skills / filesystem 到底怎样组合 | `content-builder-agent` | `graph.py`、`memory.py`、`skills.py`、`filesystem.py` |
| 同步 subagent handoff 与结果回传 | `deep_research` | `subagents.py`、`pregel/main.py` |
| 远端 async task 生命周期 | `async-subagent-server` | `async_subagents.py`、server 的 `/threads` `/runs` 处理 |
| 上游 toolkit 怎样接入 harness | `text-to-sql-agent` | `SQLDatabaseToolkit`、`tools/base.py`、`graph.py` |
| outer loop 是否该进 graph | `ralph_mode` | 先确认 CLI boundary，再回 `graph.py` |
| eval 驱动优化 harness 本身 | `better-harness` | `core.py`、`agent.py`、`runners.py`、`patching.py` |

次级样本则更适合按专题补充，而不是作为维护工作流第一站：

- `deploy-content-writer`、`deploy-coding-agent`、`deploy-mcp-docs-agent`
  更适合看 `deepagents.toml`、`AGENTS.md`、MCP、sandbox、部署打包接缝。
- `nvidia_deep_agent`
  更像高密度综合样本，适合在前面几类已经读熟后再看。
- `downloading_agents`
  更适合解释“agent 为什么本质上可以是一个文件夹”，不适合学习 runtime internals。

## 常见问题与排障入口

- “这个 example 跑通了，为什么我还不能把它当 SDK 保证”：因为 example 展示的是组合证据，不是稳定公共 contract；先回 [第3章：Create Deep Agent 作为 Assembly Root](../part1-foundations/03-create-deep-agent-as-assembly-root.md) 和本章的三类标签。
- “我在 example 里看到了流输出，所以是不是已经知道 streaming 规则”：不是；规则定义回到第9章到第12章，不在样本目录里。
- “为什么 `subagents.yaml` / `deepagents.toml` / `langgraph.json` 看起来都像配置，但意义完全不同”：因为它们分别落在 example helper、部署包装、上游 runtime 接缝，不属于同一层。
- “我想知道 bug 该修在 example、Deep Agents 还是上游”：先用 [附录 E：Troubleshooting Playbook](../appendix/troubleshooting-playbook.md) 按症状分层，再决定是否回 [第13章](./13-backend-protocol-and-storage-strategy.md)、[第14章](./14-provider-profiles-and-model-routing.md)、[第15章](./15-testing-the-harness.md)。
- “我只想快速知道某个问题先看哪个目录”：先查 [附录 C：Examples 索引与阅读顺序](../appendix/examples-index.md)，它比本章更适合做检索。

最容易踩的坑有四类：

- 把 example 里的 helper、YAML、脚本、REPL、README 文案当成 SDK 保证。
- 只读 example，不继续追 `deepagents` 装配层和上游 runtime / primitive。
- 看到 `langgraph.json`、`deepagents.toml`、CLI 脚本，就误以为它们描述的是同一层。
- 把产品化 outer loop 或研究型工作流强行抽回 `graph.py`。

## 本章结论

- 谁提供：examples 提供的是维护者的证据入口；真正的 contract 仍由 `LangChain`、`LangGraph` 和 `Deep Agents` 各自提供。
- 如何传播：先从 example 找到装配入口，再追到 `deepagents` 装配层，最后把 callback、streaming、visibility 之类的问题回收到 Part 3 的传播章节。
- 修在哪层：样本私有 helper 修在 example，本地默认装配修在 Deep Agents，上游 runtime / primitive 语义问题回到 LangGraph 或 LangChain。
