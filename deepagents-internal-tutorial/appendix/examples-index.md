# 附录 C：Examples 索引与阅读顺序

这一页把 `deepagents/examples/` 当成“源码阅读入口表”，不是“照抄模板集合”。

如果你只想知道某个问题先看哪个目录，这一页比第16章更适合快速检索。

---

## 核心样本

这六个目录，是本教程默认优先阅读的主样本。

| Example | 类型 | 主入口 | 最值得回答的问题 | 读完后下一站 |
|---------|------|--------|------------------|--------------|
| `deep_research` | specialized harness | `agent.py` | research orchestrator + subagent wiring 到底在哪层 | `deepagents/graph.py`、`middleware/subagents.py` |
| `async-subagent-server` | remote async task | `supervisor.py`、`server.py` | 远端 task id、thread/run、update/cancel 是怎么接进主 agent 的 | `middleware/async_subagents.py` |
| `content-builder-agent` | filesystem harness | `content_writer.py` | memory / skills / files / subagents 怎样一起工作 | `memory.py`、`skills.py`、`filesystem.py` |
| `text-to-sql-agent` | toolkit integration | `agent.py` | LangChain toolkit 怎样嵌进 Deep Agents harness | `SQLDatabaseToolkit`、`graph.py` |
| `ralph_mode` | outer loop | `ralph_mode.py` | fresh context loop 为什么不在 graph 内部 | 先确认 CLI boundary，再回 `graph.py` |
| `better-harness` | meta-harness / eval | `README.md`、`examples/deepagents_example.toml` | harness 本身如何成为优化对象 | `better_harness/core.py`、`agent.py`、`runners.py` |

---

## 六个主样本的最短阅读顺序

### `deep_research`

1. `deepagents/examples/deep_research/agent.py`
2. `deepagents/examples/deep_research/research_agent/prompts.py`
3. `deepagents/examples/deep_research/research_agent/tools.py`
4. `deepagents/libs/deepagents/deepagents/graph.py`
5. `deepagents/libs/deepagents/deepagents/middleware/subagents.py`

适合在你要查这些问题时作为第一站：

- `task` tool 怎样引出 subagent
- research workflow 是 prompt 还是 runtime
- tool / model / stream 行为到底落哪层

### `async-subagent-server`

1. `deepagents/examples/async-subagent-server/supervisor.py`
2. `deepagents/libs/deepagents/deepagents/middleware/async_subagents.py`
3. `deepagents/examples/async-subagent-server/server.py`

适合在你要查这些问题时作为第一站：

- remote async task 怎样开始、轮询、更新、取消
- 为什么 task status 不能靠 conversation history 复述
- 本地 supervisor 与远端 server 的责任边界是什么

### `content-builder-agent`

1. `deepagents/examples/content-builder-agent/content_writer.py`
2. `deepagents/examples/content-builder-agent/AGENTS.md`
3. `deepagents/examples/content-builder-agent/skills/*/SKILL.md`
4. `deepagents/examples/content-builder-agent/subagents.yaml`
5. `deepagents/libs/deepagents/deepagents/middleware/memory.py`
6. `deepagents/libs/deepagents/deepagents/middleware/skills.py`
7. `deepagents/libs/deepagents/deepagents/backends/filesystem.py`

适合在你要查这些问题时作为第一站：

- `AGENTS.md` / `SKILL.md` 怎样参与系统提示词装配
- `FilesystemBackend` 到底承担了什么
- 哪些文件约定是 Deep Agents 原生的，哪些只是 example helper

### `text-to-sql-agent`

1. `deepagents/examples/text-to-sql-agent/agent.py`
2. `deepagents/examples/text-to-sql-agent/AGENTS.md`
3. `deepagents/examples/text-to-sql-agent/skills/*/SKILL.md`
4. `langchain_community.agent_toolkits.SQLDatabaseToolkit`
5. `deepagents/libs/deepagents/deepagents/graph.py`

适合在你要查这些问题时作为第一站：

- 现成 LangChain toolkit 怎样挂到 Deep Agents
- 哪层负责数据库语义，哪层负责 harness 语义
- toolkit tool 的 callback/config 为什么还能继续传播

### `ralph_mode`

1. `deepagents/examples/ralph_mode/ralph_mode.py`
2. 标出外部边界：`deepagents_cli.non_interactive.run_non_interactive(...)`
3. 再回看 `deepagents/libs/deepagents/deepagents/graph.py`

适合在你要查这些问题时作为第一站：

- fresh context loop 是否应该进图
- 文件系统记忆与会话记忆怎样分工
- 哪些模式天然属于 CLI / product layer

### `better-harness`

1. `deepagents/examples/better-harness/README.md`
2. `deepagents/examples/better-harness/examples/deepagents_example.toml`
3. `deepagents/examples/better-harness/better_harness/core.py`
4. `deepagents/examples/better-harness/better_harness/agent.py`
5. `deepagents/examples/better-harness/better_harness/patching.py`
6. `deepagents/examples/better-harness/better_harness/runners.py`

适合在你要查这些问题时作为第一站：

- prompt/tool/skill/middleware 哪些是可编辑 surface
- 外层 eval loop 如何包住 inner harness
- baseline / candidate / keep-discard 机制怎样组织

---

## 次级样本

这些目录更像“专题补充件”。

| Example | 最值得看什么 | 不适合直接回答什么 |
|---------|--------------|--------------------|
| `deploy-content-writer` | `deepagents.toml`、`AGENTS.md`、per-user memory 样式 | callback / stream 内核 |
| `deploy-coding-agent` | deployment 配置、sandbox、MCP、技能打包 | subagent runtime 细节 |
| `deploy-mcp-docs-agent` | docs-first agent 的 MCP 接缝 | compiled graph 行为 |
| `nvidia_deep_agent` | 多模型路由、sandbox backend、skills 上传到 sandbox | 最小化基础阅读路径 |
| `downloading_agents` | “agent 就是文件夹”的分发模型 | 内部执行链与 callback tree |

---

## 如果你在查某类问题，先看哪个目录

| 你在查什么 | 推荐先看 |
|------------|----------|
| subagent handoff / tool result 汇总 | `deep_research` |
| remote async task 生命周期 | `async-subagent-server` |
| memory / skills / filesystem 如何一起装配 | `content-builder-agent` |
| Deep Agents 如何消费 LangChain toolkit | `text-to-sql-agent` |
| outer loop 是否应该放进 graph | `ralph_mode` |
| eval 驱动优化 harness 自身 | `better-harness` |
| 部署配置、MCP、sandbox 包装 | `deploy-*` |
| 多模型 + GPU sandbox 综合样本 | `nvidia_deep_agent` |
| agent artifact 分发 | `downloading_agents` |

---

## 按横向主题查 example

| 横向主题 | 推荐 example | 为什么先看它 |
|----------|--------------|--------------|
| `Assembly` | `deep_research`、`text-to-sql-agent` | 最容易看清 `create_deep_agent()` 如何把上游 primitive 和本地策略装成一个 harness |
| `Context` | `content-builder-agent` | memory、skills、filesystem、`AGENTS.md` / `SKILL.md` 的装配关系最集中 |
| `Execution` | `deep_research`、`async-subagent-server` | 一个代表本地 `task` handoff，一个代表远端 async handoff |
| `Propagation` | `deep_research`、`async-subagent-server` | 最适合对照 callback/config、stream、结果折返三条不同可见面 |
| `Extension` | `text-to-sql-agent`、`better-harness` | 一个看 toolkit 接缝，一个看 harness 本身如何作为扩展对象 |
| `Operations` | `better-harness`、`ralph_mode`、`deploy-*` | 一个偏评估和回归，一个偏外循环，一个偏部署与运行环境 |

---

## 阅读时要始终标注的三件事

读任何 example，都建议先在旁边记下这三类标签：

- `example 本地约定`
  例如 `subagents.yaml` loader、FastAPI server、REPL、eval config。

- `Deep Agents 装配约定`
  例如 `create_deep_agent()`、memory / skills / permissions / backend / profile。

- `上游 runtime / primitive 约定`
  例如 callback manager、`RunnableConfig`、`stream_mode`、checkpoint、tool runtime。

只要这三个标签没分清，你就很难判断 bug 应该修在哪层。

---

## 本页小结

- 先用这一页定位 example，再去第16章看详细拆解。
- 六个主样本优先级最高，其余目录更适合按专题补充。
- examples 的最大价值，不是提供“标准答案”，而是提供“最短追源码入口”。
