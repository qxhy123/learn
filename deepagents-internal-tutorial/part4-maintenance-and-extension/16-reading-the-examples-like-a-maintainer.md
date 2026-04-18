# 第11章：像维护者一样阅读 Examples

## 学习目标

学完本章，你应该能回答：

1. 为什么 example 只能当“边界样本”，不能直接当 SDK contract
2. 读 `deepagents/examples/` 时，应该如何一路追到 `deepagents`、`langgraph`、`langchain`
3. 哪些 example 主要回答装配问题，哪些主要回答部署问题，哪些主要回答 outer loop / eval 问题
4. `deep_research`、`async-subagent-server`、`content-builder-agent`、`text-to-sql-agent`、`ralph_mode`、`better-harness` 各自最值得维护者看的入口在哪里

---

## 问题是什么

很多人读 example 的方式是：

- 看 README
- 跑通 demo
- 把能工作的 wiring 直接当成“官方架构”

这对使用者够了，但对维护者不够。

维护者更该问的是：

- 这个目录到底在复用哪一层能力
- 哪些行为是 example 本地 helper，哪些是 Deep Agents 默认装配，哪些是 LangGraph / LangChain 上游语义
- 如果这里出 bug，第一站应该追哪个文件

所以本章的重点不是“怎么运行 examples”，而是“怎么用 examples 反推三层栈的责任边界”。

---

## 为什么这一章故意长得不一样

这一章是 example-lens / index 型特例章节，所以它不会像 [第2章](../part1-foundations/02-repo-map-and-package-boundaries.md) 或 [第3章](../part1-foundations/03-create-deep-agent-as-assembly-root.md) 那样先给一整套通用分层模板。
它的职责是把样本目录和源码入口一一钉住，再把你引回系统章节和 [附录 C](../appendix/examples-index.md)。

读 example 时如果你发现自己开始争论 ownership、assembly root、streaming visibility、callback propagation 这些通用问题，就说明应该先回跳到系统章节，再回来继续读样本。

---

## 先把 Examples 分三类

| 类别 | 代表目录 | 这类样本最适合回答什么问题 |
|------|----------|----------------------------|
| harness 装配样本 | `deep_research`、`content-builder-agent`、`text-to-sql-agent` | `create_deep_agent()` 被怎样参数化，memory / skills / tools / subagents / backend 怎样组合 |
| runtime / deployment 样本 | `async-subagent-server`、`deploy-*`、`nvidia_deep_agent` | LangGraph server、远端 thread/run、MCP、sandbox、部署配置怎样接入 |
| outer loop / meta-harness 样本 | `ralph_mode`、`better-harness` | 哪些能力该放在 graph 外层，eval 驱动优化如何围住 harness 本身 |

如果你一开始就把这三类混在一起看，很容易得出错误结论：

- 把部署目录里的 `deepagents.toml` 当成 runtime 内核
- 把 example 里的 YAML loader 当成 SDK contract
- 把 CLI outer loop 当成 graph 内节点

---

## 维护者阅读总流程

读任何一个 example，建议固定走这四步：

1. 先找入口文件。
   先定位 `create_deep_agent()`、`langgraph.json`、`deepagents.toml`、`run_non_interactive(...)`、`SQLDatabaseToolkit(...)` 这种真正决定 wiring 的地方。

2. 再标出 example 自己新增了什么。
   例如 `subagents.yaml` loader、FastAPI server、eval runner、本地 REPL、产品化 prompt。这些 often 是 example 私有层，不是框架默认层。

3. 再追到 `deepagents` 装配层。
   重点看 `deepagents/libs/deepagents/deepagents/graph.py` 和对应 middleware / backend 文件，确认这些参数最终怎样被装配成 graph。

4. 最后去上游确认执行语义。
   例如 tool callback、token stream、`RunnableConfig`、`stream_mode`、`ToolRuntime`、checkpoint、remote thread/run 生命周期，这些都不是 example 自己定义的。

---

## 六个核心 Example

下面这六个目录，是本教程最值得当 maintainer 样本反复读的 examples。

### 1. `deep_research`

#### 这个 example 真正展示什么

它展示的不是“Deep Agents 内建 research mode”，而是：

- 如何用额外 prompt 把默认 harness 专门化成 research orchestrator
- 如何用 `task` 工具把 research subagent 接进主线程
- 如何把 example 自己的搜索工具和反思工具注入进去

也就是说，它是“specialized harness”样本，不是“SDK 核心功能清单”。

#### 三层落点

- `LangChain`
  `init_chat_model(...)`、`@tool`、`InjectedToolArg`、model/tool callback 都在这层。
- `LangGraph`
  `langgraph.json`、server/studio 接入、`messages`/`updates` stream 面在这层。
- `Deep Agents`
  `create_deep_agent()`、`SubAgentMiddleware`、默认文件与 todo harness、prompt layering 在这层。

#### 追源码顺序

1. 看 `deepagents/examples/deep_research/agent.py`
2. 看 `deepagents/examples/deep_research/research_agent/prompts.py`
3. 看 `deepagents/examples/deep_research/research_agent/tools.py`
4. 追到 `deepagents/libs/deepagents/deepagents/graph.py`
5. 再看 `deepagents/libs/deepagents/deepagents/middleware/subagents.py`
6. 如果你在查 streaming / UI 可见性，再看 `langgraph/libs/langgraph/langgraph/pregel/main.py` 和 `langgraph/libs/langgraph/langgraph/pregel/_messages.py`
7. 如果你在查 tool/model callback 事件，再看 `langchain/libs/core/langchain_core/tools/base.py` 和 `langchain/libs/core/langchain_core/language_models/chat_models.py`

#### 容易误判什么

- 误判 1：research workflow 是 Deep Agents 默认内核。
  其实这些 workflow prompt 基本都在 example 自己的 `prompts.py`。

- 误判 2：research subagent 的存在说明主 agent 会自动并行拆任务。
  实际是 example prompt 在驱动拆分，`task` 只是提供隔离执行面。

- 误判 3：Tavily + think tool 属于框架能力。
  它们只是 example 注入的普通工具。

### 2. `async-subagent-server`

#### 这个 example 真正展示什么

它展示的不是“subagent 的 async 版本”，而是：

- Deep Agents 如何把远端 Agent Protocol server 暴露成一组 async task tools
- 本地 supervisor 怎样通过 task id 跟远端 thread/run 生命周期交互
- `start` / `check` / `update` / `cancel` / `list` 这些动作如何映射到远端 server

它是“远端任务协议样本”，不是“Python 里多一个 `async def` 就完了”的样本。

#### 三层落点

- `LangChain`
  远端 server 里的 `_agent = create_deep_agent(...)` 依然靠 LangChain model/tool primitive 工作。
- `LangGraph`
  `MemorySaver`、thread id、LangGraph SDK client、remote runs/status 协议都属于这一层及其邻接生态。
- `Deep Agents`
  `AsyncSubAgentMiddleware` 把远端 server 暴露成 `start_async_task` / `check_async_task` / `update_async_task` / `cancel_async_task` / `list_async_tasks`。

#### 追源码顺序

1. 先看 `deepagents/examples/async-subagent-server/supervisor.py`
2. 再看 `deepagents/libs/deepagents/deepagents/middleware/async_subagents.py`
3. 再看 `deepagents/examples/async-subagent-server/server.py`
4. 如果你在查本地 supervisor 状态保存，再看 `langgraph.checkpoint.memory.MemorySaver` 的使用点
5. 如果你在查远端结果为什么最终回到父线程，再回看 `async_subagents.py` 里各 tool 返回的 `Command(update=...)`
6. 如果你在查远端 agent 内部 model/tool 语义，再按普通 Deep Agents 路径追回 `graph.py`、`tools/base.py`、`chat_models.py`

#### 容易误判什么

- 误判 1：async subagent 只是本地 subagent 的非阻塞包装。
  不对，它背后是远端 thread/run 协议和状态轮询。

- 误判 2：只要 conversation 里写着某个 task 正在 running，就可以直接复述。
  不对，源码里明确要求状态必须重新调用 tool 获取。

- 误判 3：远端 server 的安全边界由本地 permissions 自动接管。
  不对，远端 agent 要在自己的 server / tool / sandbox 层再做一遍。

### 3. `content-builder-agent`

#### 这个 example 真正展示什么

它是当前仓库里最好的“文件化 harness”阅读样本，能同时看到：

- `AGENTS.md` 作为 memory
- `skills/*/SKILL.md` 作为按需技能
- `FilesystemBackend` 作为工作目录与持久文件层
- 自定义工具与自定义 subagent 配置如何插进 `create_deep_agent()`

它非常适合维护者用来分辨“filesystem primitive”与“example helper”的边界。

#### 三层落点

- `LangChain`
  `@tool`、`AIMessage` / `ToolMessage`、工具调用协议在这层。
- `LangGraph`
  `agent.astream(..., stream_mode="values")` 的执行与流传播在这层。
- `Deep Agents`
  `MemoryMiddleware`、`SkillsMiddleware`、`FilesystemBackend`、`SubAgentMiddleware` 和默认文件工具在这层。

#### 追源码顺序

1. 看 `deepagents/examples/content-builder-agent/content_writer.py` 的 `create_content_writer()`
2. 再看 `deepagents/examples/content-builder-agent/AGENTS.md`
3. 再看 `deepagents/examples/content-builder-agent/skills/*/SKILL.md`
4. 再看 `deepagents/examples/content-builder-agent/subagents.yaml` 和同文件里的 `load_subagents()`
5. 追到 `deepagents/libs/deepagents/deepagents/graph.py`
6. 再看 `deepagents/libs/deepagents/deepagents/middleware/memory.py`、`deepagents/libs/deepagents/deepagents/middleware/skills.py`、`deepagents/libs/deepagents/deepagents/middleware/subagents.py`
7. 如果你在查文件读写根目录，再看 `deepagents/libs/deepagents/deepagents/backends/filesystem.py`
8. 如果你在查为什么 `values` 流里会看到整批消息，再看 `langgraph/libs/langgraph/langgraph/pregel/main.py`

#### 容易误判什么

- 误判 1：`subagents.yaml` 是 Deep Agents 原生配置格式。
  不是。源码里 `load_subagents()` 明确说明这只是 example 自己的 helper。

- 误判 2：skills 是一开始整包注入模型上下文。
  不是。它们是按需加载的 progressive disclosure。

- 误判 3：界面里看到的输出顺序就是 graph 内部节点顺序。
  不一定。这里的展示层还叠加了 `Rich` 的 live rendering。

### 4. `text-to-sql-agent`

#### 这个 example 真正展示什么

它最有价值的地方不是 SQL 本身，而是边界拆分：

- SQL toolkit 与数据库访问 primitive 来自 LangChain 生态
- memory / skills / filesystem / planning harness 来自 Deep Agents
- 运行时仍然落到 LangGraph 编译图上

所以它是“上游 toolkit 接入 Deep Agents harness”的标准样本。

#### 三层落点

- `LangChain`
  `SQLDatabaseToolkit`、`SQLDatabase`、SQL 工具集合都在这层。
- `LangGraph`
  `create_deep_agent()` 返回的其实是 LangGraph 编译结果，`invoke()` 仍在这层执行。
- `Deep Agents`
  memory / skills / filesystem / 默认 planning 与文件工具装配在这层。

#### 追源码顺序

1. 看 `deepagents/examples/text-to-sql-agent/agent.py`
2. 再看 `deepagents/examples/text-to-sql-agent/AGENTS.md` 与 `skills/*/SKILL.md`
3. 再追 `langchain_community.agent_toolkits.SQLDatabaseToolkit` 与 `langchain_community.utilities.SQLDatabase`
4. 然后回到 `deepagents/libs/deepagents/deepagents/graph.py`
5. 再看 `deepagents/libs/deepagents/deepagents/backends/filesystem.py`
6. 如果你在查 toolkit tools 为什么会继续带 callbacks/config，再看 `langchain/libs/core/langchain_core/tools/base.py`
7. 如果你在查 graph 执行与 state，再看 `langgraph/libs/langgraph/langgraph/pregel/main.py`

#### 容易误判什么

- 误判 1：SQL 能力属于 Deep Agents 内建工具集。
  不对，这里真正的数据库能力来自 LangChain Community toolkit。

- 误判 2：没有 subagent，就说明这个例子不涉及 context isolation。
  也不对，filesystem、skills、todo planning 仍然在控制上下文组织方式。

- 误判 3：只要 toolkit 能跑，Deep Agents 层就不用关心测试。
  错。真正容易回归的是 harness 和 toolkit 交界处。

### 5. `ralph_mode`

#### 这个 example 真正展示什么

它最重要的价值是把一个常见误会拆开：

- fresh context loop 不一定要做成 graph 内部循环节点
- 文件系统和 git 可以承担跨轮记忆
- 真正的 agent 执行发生在 outer loop 每轮重新调用时

所以 Ralph 读法的关键不是“图里怎么循环”，而是“为什么循环根本不在图里”。

#### 三层落点

- `LangChain`
  每轮真正 agent 调用时，底层 model/tool primitive 仍然是 LangChain。
- `LangGraph`
  每轮 fresh thread 进入的 compiled graph 仍然是 LangGraph runtime。
- `Deep Agents`
  这个目录本身更像 Deep Agents CLI 的外层使用模式，而不是库内 middleware 组合样本。

#### 追源码顺序

1. 看 `deepagents/examples/ralph_mode/ralph_mode.py`
2. 先标出边界点：`deepagents_cli.non_interactive.run_non_interactive(...)` 不在当前三仓源码里
3. 再回到本教程第 3 章对应的 `deepagents/libs/deepagents/deepagents/graph.py`
4. 如果你在查 streaming / checkpoint / thread 语义，再看 `langgraph/libs/langgraph/langgraph/pregel/main.py`
5. 如果你在查 tool/model 事件，再看 `langchain/libs/core/langchain_core/tools/base.py` 与 `langchain/libs/core/langchain_core/language_models/chat_models.py`

#### 容易误判什么

- 误判 1：Ralph 模式应该被抽回 `graph.py`。
  不一定。这个例子恰恰说明有些模式天然属于 CLI outer loop。

- 误判 2：fresh context 等于完全没有记忆。
  不对。文件系统和 git 依然在承担跨轮记忆。

- 误判 3：这个例子能直接证明 compiled graph 的 loop 语义。
  不能。它主要证明的是“外层编排”。

### 6. `better-harness`

#### 这个 example 真正展示什么

它不是普通业务 agent example，而是“用一个 Deep Agent 去优化另一个 harness”的研究样本。

对维护者最有启发的不是业务流程，而是这几个概念：

- editable surfaces
- proposer workspace
- baseline / candidate / keep-or-discard
- train / holdout / scorecard

也就是说，它展示的是“harness 自己成为优化对象”。

#### 三层落点

- `LangChain`
  外层 proposer agent 本身仍然调用 LangChain model/tool primitive。
- `LangGraph`
  外层 proposer 由 `create_deep_agent()` 装配后仍在 LangGraph runtime 上执行。
- `Deep Agents`
  这个例子把 `create_deep_agent()` 作为 outer optimizer 的内核，而不是直接业务 agent。

#### 追源码顺序

1. 先看 `deepagents/examples/better-harness/README.md`
2. 再看 `deepagents/examples/better-harness/examples/deepagents_example.toml`
3. 再看 `deepagents/examples/better-harness/better_harness/core.py`
4. 再看 `deepagents/examples/better-harness/better_harness/agent.py`
5. 再看 `deepagents/examples/better-harness/better_harness/patching.py`
6. 最后看 `deepagents/examples/better-harness/better_harness/runners.py`
7. 如果你想确认 proposer agent 自己是怎么装起来的，再回到 `deepagents/libs/deepagents/deepagents/graph.py`

#### 容易误判什么

- 误判 1：这是 Deep Agents 官方默认工作流。
  不是。它是研究 artifact。

- 误判 2：只有 prompt 值得当 surface。
  不对。这个例子明确把 tool、skill、middleware 实现、middleware 注册都当成 surface。

- 误判 3：holdout/private 可见性已经是严格安全隔离。
  README 已经说明这里更接近 research infrastructure，而不是强隔离沙箱。

---

## 次级 Example 该怎么用

下面这些目录也值得看，但更适合作为“补全局部主题”的样本，而不是第一个读的入口。

### `deploy-content-writer`、`deploy-coding-agent`、`deploy-mcp-docs-agent`

这三个目录最适合回答：

- `deepagents.toml` 怎么组织 deployment-facing 配置
- `AGENTS.md`、`skills/`、`mcp.json` 怎样变成一个可部署 agent 包
- sandbox / MCP / user memory 这种产品化接缝放在哪里

不适合直接拿来回答：

- callback manager 怎样传播
- subagent 内部 token 为什么可见
- `CompiledSubAgent` 为什么不继承某个 middleware

### `nvidia_deep_agent`

这个目录适合在你已经看完 `content-builder-agent` 和 `deep_research` 后再读，因为它把多个主题叠在一起：

- 多模型路由
- sandbox backend
- skills 上传到 sandbox
- 自我修正式 memory / skills 维护

它是“高密度综合样本”，不是第一站。

### `downloading_agents`

这个目录很适合拿来解释：

- agent 为什么本质上可以是一个文件夹
- `AGENTS.md + skills/` 为什么足够构成可分发 artifact

但它不适合拿来学习 runtime internals，因为这里几乎没有装配源码可追。

---

## 哪些模式值得抽回库里，哪些应该留在 Example

更像应该抽回库里的信号：

- 多个 example 都在重复相同的装配逻辑
- 这个逻辑与具体产品场景无关
- 它已经开始在多个目录里手写复制

更应该留在 example 的信号：

- 它主要是业务 prompt、产品 UX、部署脚本或 README 约定
- 它依赖某个场景专属工具或服务
- 抽回库里会扩大默认 contract，却没有足够测试和通用性支持

---

## 用 Example 反查问题的最快路径

如果你要查这些问题，第一站通常是：

| 你在查什么 | 先看哪个 example | 再追哪里 |
|------------|------------------|----------|
| memory / skills / filesystem 到底怎样组合 | `content-builder-agent` | `graph.py`、`memory.py`、`skills.py`、`filesystem.py` |
| 同步 subagent handoff 与结果回传 | `deep_research` | `subagents.py`、`pregel/main.py` |
| 远端 async task 生命周期 | `async-subagent-server` | `async_subagents.py`、server 里的 `/threads` `/runs` 处理 |
| 上游 toolkit 怎样接入 harness | `text-to-sql-agent` | `SQLDatabaseToolkit`、`tools/base.py`、`graph.py` |
| outer loop 是否该进 graph | `ralph_mode` | 先确认 CLI boundary，再回 `graph.py` |
| eval 驱动优化 harness 本身 | `better-harness` | `core.py`、`agent.py`、`runners.py`、`patching.py` |

---

## 容易踩什么坑

- 坑 1：把 example 目录中的 helper、YAML、脚本、REPL、README 文案当成 SDK 保证。

- 坑 2：只读 example，不继续追 `deepagents` 装配层和上游 runtime / primitive。

- 坑 3：看到 `langgraph.json`、`deepagents.toml`、CLI 脚本，就误以为它们描述的是同一层。

- 坑 4：把产品化外层模式强行抽回 `graph.py`。

---

## 本章小结

- examples 是“边界样本”，不是“合同文本”。
- 维护者读 example 的正确方式，是先找入口，再分清 example 私有层、Deep Agents 装配层、LangGraph runtime、LangChain primitive。
- 六个核心 example 分别回答了 specialized harness、remote async task、filesystem harness、toolkit 接入、outer loop、meta-harness 这六类问题。
