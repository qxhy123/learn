# 附录 A：代码阅读检查表

这一页的目标不是“把所有源码都列出来”，而是给维护者一个跨三仓的最短阅读路径。

---

## 最短阅读路径

### Step 1：先看 LangChain primitive

先打开：

- `langchain/libs/core/langchain_core/runnables/config.py`
- `langchain/libs/core/langchain_core/callbacks/manager.py`
- `langchain/libs/core/langchain_core/tools/base.py`
- `langchain/libs/core/langchain_core/language_models/chat_models.py`

先确认三件事：

- config 怎么合并
- callback manager 怎么构树
- model/tool 调用在什么位置触发 callback

### Step 2：再看 LangGraph runtime

再打开：

- `langgraph/libs/langgraph/langgraph/pregel/main.py`
- `langgraph/libs/langgraph/langgraph/pregel/_messages.py`
- `langgraph/libs/prebuilt/langgraph/prebuilt/tool_node.py`

重点确认：

- `stream_mode` / `subgraphs`
- `StreamMessagesHandler`
- `ToolRuntime`
- subgraph / checkpoint namespace

### Step 3：最后看 Deep Agents 装配层

再打开：

- `deepagents/libs/deepagents/deepagents/graph.py`
- `deepagents/libs/deepagents/deepagents/middleware/subagents.py`
- `deepagents/libs/deepagents/deepagents/middleware/filesystem.py`
- `deepagents/libs/deepagents/deepagents/middleware/skills.py`
- `deepagents/libs/deepagents/deepagents/middleware/memory.py`

这时再问：

- Deep Agents 改写了什么
- 它只是把上游组合起来，还是自己新增了本地策略

### Step 4：再看测试

优先看：

- `deepagents/libs/deepagents/tests/unit_tests/test_graph.py`
- `deepagents/libs/deepagents/tests/unit_tests/test_subagents.py`
- `deepagents/libs/deepagents/tests/integration_tests/test_subagent_middleware.py`

测试是确认“当前本地 contract 到底锁了什么”的最快方式。

---

## 读 subagent / streaming / callback 问题时必须同时开的文件

- `deepagents/.../middleware/subagents.py`
- `langgraph/.../pregel/main.py`
- `langgraph/.../pregel/_messages.py`
- `langchain_core/runnables/config.py`
- `langchain_core/callbacks/manager.py`
- `langchain_core/tools/base.py`

如果这六个文件没同时打开，通常会误判边界。

---

## 阅读时应特别留意的符号

- `RunnableConfig`
- `patch_config(...)`
- `get_child()`
- `ToolRuntime`
- `StreamMessagesHandler`
- `TAG_NOSTREAM`
- `Command(update=...)`
- `_EXCLUDED_STATE_KEYS`

---

## 初读时最容易犯的误判

- 误判 1：把 callback manager 当成 handler 列表，而不是 run tree。
- 误判 2：把 `subgraphs=True` 当成 Deep Agents 能力，而不是 LangGraph runtime 行为。
- 误判 3：把 `AGENTS.md` / `SKILL.md` 当成上游标准，而不是 Deep Agents 约定。
- 误判 4：把“token 不可见”与“结果不回传”混为一谈。

---

## 快速判断模板

如果你卡住了，按这个顺序问：

1. 这是 model/tool/config/callback primitive 问题吗
2. 这是 graph/subgraph/stream/checkpoint runtime 问题吗
3. 这是 Deep Agents 装配或本地策略问题吗

只有先过完这三问，后续 patch 才不容易改错层。
