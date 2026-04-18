# 附录 B：测试矩阵

这一页回答的是维护者最实际的问题：当改动跨越 Deep Agents / LangGraph / LangChain 时，哪些假设要信任上游，哪些必须在本地测试里重新钉死。

---

## 测试层级

### `deepagents` 本地 unit tests

用途：

- 锁住 middleware / backend / helper 的局部 contract
- 锁住本地装配策略

### `deepagents` integration tests

用途：

- 锁住组装后的 harness 行为
- 锁住 parent/child、tool/state、prompt/tool surface 的组合关系

### 上游源码与测试

用途：

- 确认你依赖的 LangGraph / LangChain 语义是不是本来就这样
- 避免把上游既有行为误写成 Deep Agents 本地 contract

---

## 按改动类型选择测试

| 你改了什么 | 最少要补/跑什么 |
|------------|------------------|
| `graph.py` 默认 middleware 顺序 | `test_graph.py`、相关 smoke tests、至少一个 integration test |
| subagent handoff / state bubbling | `test_subagents.py`、`test_async_subagents.py`、`integration_tests/test_subagent_middleware.py` |
| callback / config 传播相关结论 | 本地 `test_subagents.py` 对应 case，外加上游 `langchain_core` 源码核对 |
| Pregel state model / reducer / step boundary 相关结论 | 本地 state-native regression + Chapter 4 claims cross-check，外加 `langgraph/pregel/main.py`、`_loop.py` 核对 |
| Pregel execution path / runtime injection 相关结论 | 本地 runtime-path regression + Chapter 5 claims cross-check，外加 `langgraph/pregel/main.py`、`_runner.py` 核对 |
| streaming visibility / `nostream` / `subgraphs` 相关结论 | 本地 streaming 测试，外加 `langgraph/pregel/main.py`、`_messages.py` 核对 |
| result-return / parent-visible state / summary 折返相关结论 | 本地 subagent integration test + state / summary 相关断言，外加 `middleware/subagents.py` 与 LangGraph reducer 路径核对 |
| 文件工具 / backend | `test_file_system_tools.py`、`test_filesystem_middleware.py`、backend 对应单测 |
| memory / skills / private state 过滤 | `middleware/test_memory_middleware.py`、`middleware/test_skills_middleware.py` |
| provider profile / model routing | `test_models.py`、必要时加 smoke snapshot |
| 上游版本升级 | 重新跑与 callback/config/streaming/subagent 边界相关的本地 regression 集合 |

---

## 常见改动的最小验证集

### 改 `create_deep_agent()` 默认装配

- `pytest deepagents/libs/deepagents/tests/unit_tests/test_graph.py`
- `pytest deepagents/libs/deepagents/tests/unit_tests/smoke_tests/test_system_prompt.py`
- `pytest deepagents/libs/deepagents/tests/integration_tests/test_deepagents.py`

### 改 subagent 协议或边界

- `pytest deepagents/libs/deepagents/tests/unit_tests/test_subagents.py`
- `pytest deepagents/libs/deepagents/tests/unit_tests/test_async_subagents.py`
- `pytest deepagents/libs/deepagents/tests/integration_tests/test_subagent_middleware.py`

额外必须点名核对：

- `test_subagent_streaming_emits_messages_and_updates_from_subgraph`
- `test_config_passed_to_runnable_lambda_subagent`
- `test_context_passed_to_subagent_tool_runtime`
- `test_subagent_propagates_recursion_limit_to_tool_runtime`
- `test_subagent_propagates_callbacks_to_model_calls`

其中最后一个当前仍是 `xfail`，文档里不能把它写成已保证行为。

### 改 streaming / visibility 结论

至少同时核对：

- `deepagents/libs/deepagents/tests/unit_tests/test_subagents.py`
- `langgraph/libs/langgraph/langgraph/pregel/main.py`
- `langgraph/libs/langgraph/langgraph/pregel/_messages.py`

### 改 callback manager / config 传播结论

至少同时核对：

- `langchain/libs/core/langchain_core/runnables/config.py`
- `langchain/libs/core/langchain_core/callbacks/manager.py`
- `langchain/libs/core/langchain_core/tools/base.py`
- `langchain/libs/core/langchain_core/language_models/chat_models.py`

### 改 result-return / state bubbling / summary 结论

至少同时核对：

- `deepagents/libs/deepagents/tests/unit_tests/test_subagents.py`
- `deepagents/libs/deepagents/tests/integration_tests/test_subagent_middleware.py`
- `deepagents/libs/deepagents/deepagents/middleware/subagents.py`
- 相关 state reducer / summary 写回路径

---

## 维护者应主动补的测试类型

- 新 middleware 引入了新的 private state key
- 新 backend 改变了 tool 返回格式或 state 回传方式
- 新 subagent 类型改变了 config / context / callbacks 的传播
- 新 streaming 策略改变了“哪些事件对外可见”
- 新 profile 同时修改 prompt、tool description、excluded tools
- 上游升级后，本地依赖的 callback / stream / subgraph 假设可能漂移

---

## 依赖升级时的特别规则

如果你升级了 `langgraph` 或 `langchain`，不要只跑 happy path。

至少重新核对：

1. callback/config 是否仍按原路径传播
2. streaming visibility 是否仍按 `messages` / `updates` / `custom` 分开成立
3. result-return / state bubbling / summary 折返语义是否漂移
4. `ToolRuntime` 注入字段是否变化
5. Pregel state model 的 step boundary 解释是否仍然准确
6. Pregel execution path 的 `_defaults()` / loop / runner / runtime injection 解释是否仍然准确
7. Deep Agents 本地文档是否仍然准确

---

## 只跑 happy path 的风险

- middleware 顺序错了，功能还能跑，但 tool surface 或 permissions 已漂移
- `nostream` 还在工作，但只有 `messages` 被过滤，`updates` / 最终回传已变
- callback manager tree 变了，但只看最终结果根本发现不了
- compiled subagent 还能返回结果，但 parent/child 边界已经不再符合文档

---

## 本页的底线

只要改动触及：

- subagent
- callback/config
- streaming visibility
- middleware ordering

就必须把“本地测试 + 上游源码核对”一起做完，再宣称结论成立。
