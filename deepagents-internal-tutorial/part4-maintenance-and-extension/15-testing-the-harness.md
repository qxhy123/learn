# 第10章：如何测试一个三层栈 Harness

## 学习目标

学完本章，你应该能回答：

1. 哪些行为可以信任上游测试，哪些必须在 Deep Agents 本地钉死
2. unit、integration、smoke/snapshot 在这个项目里各自守什么
3. callback / streaming / subagent 这类边界问题该怎样写回归测试
4. 已知限制什么时候该用 `xfail` 诚实记录下来
5. 面对具体改动时，最小测试配方是什么

---

## 问题是什么

Deep Agents 不是一个“单层功能库”，而是一个建立在 `LangChain + LangGraph` 上的 harness。

所以测试它时，最重要的不是“测得多不多”，而是先回答：

- 这个行为是谁拥有的
- 本地是在复用上游 contract，还是在定义自己的 contract
- 失败时要证明的是 primitive 坏了、runtime 坏了，还是装配顺序漂了

如果这个问题不先分清，测试很快就会变成两种极端：

- 要么什么都在本地重复测一遍，成本极高
- 要么什么都寄希望于上游，结果本地装配回归没人守

---

## 哪一层负责什么

### `LangChain`

- tool / model primitive
- `RunnableConfig` merge / patch
- callback manager run tree
- agent middleware hook surface

### `LangGraph`

- graph 执行语义
- state reducer / checkpoint
- subgraph namespace
- `messages` / `updates` / `custom` streaming
- `ToolRuntime`

### `Deep Agents`

- 默认 middleware 栈与装配顺序
- backend/profile/subagent/permissions 等本地 policy
- parent-child state 过滤、`task` handoff、prompt layering

---

## 代码在哪里

优先看这些本地测试：

- `deepagents/libs/deepagents/tests/unit_tests/test_graph.py`
- `deepagents/libs/deepagents/tests/unit_tests/test_subagents.py`
- `deepagents/libs/deepagents/tests/unit_tests/test_models.py`
- `deepagents/libs/deepagents/tests/unit_tests/backends/`
- `deepagents/libs/deepagents/tests/integration_tests/test_deepagents.py`
- `deepagents/libs/deepagents/tests/integration_tests/test_subagent_middleware.py`

再对照这些上游实现文件：

- `langchain/libs/core/langchain_core/runnables/config.py`
- `langchain/libs/core/langchain_core/callbacks/manager.py`
- `langchain/libs/core/langchain_core/tools/base.py`
- `langgraph/libs/langgraph/langgraph/pregel/main.py`
- `langgraph/libs/langgraph/langgraph/pregel/_messages.py`

---

## 实现怎么工作

### 1. 测试前先做 ownership 分类

每次改动前，先问四个问题：

1. 这是本地 contract，还是上游 contract
2. 我是在改装配结果，还是改 primitive / runtime 行为
3. 这个变化会不会影响 prompt/tool surface
4. 这个变化会不会影响 subagent、callback、streaming 这种跨层边界

很多测试策略差异，其实都来自这四问。

### 2. unit tests 锁的是本地 contract，不是“整个 agent 能跑”

在这个项目里，unit tests 最适合守这些东西：

- backend 返回值与路径路由规则
- `_HarnessProfile` lookup / merge
- middleware 的 state update
- parent-child state 过滤
- `interrupt_on` 继承与覆盖策略

### 3. integration tests 锁的是边界组合，而不是单层 happy path

Deep Agents 最有价值的 integration test，通常都不是“主 agent 回了一句正确答案”，而是边界行为：

- `task` handoff 是否发生
- declarative 与 compiled subagent 的语义是否分叉正确
- `subgraphs=True` 时子图 streaming 是否可观察
- context / tags / recursion limit 是否继续进入 child runtime

### 4. smoke / snapshot 守的是模型可见面

对 harness 来说，prompt surface 和 tool surface 不是“文案层细节”，而是行为 contract。

所以 snapshot / smoke test 的价值非常高，因为它们会直接暴露：

- base system prompt 变了没有
- tool descriptions 变了没有
- 某个 middleware 是否真的进了最终模型可见面
- profile / permissions / skills / memory 是否悄悄改变了暴露面

### 5. callback / streaming 问题必须单独成类测试

你真正该测的是：

- tags 是否还能下传
- context 是否还能进入 child `ToolRuntime`
- `messages` / `updates` / `custom` 是否按预期发出
- `subgraphs=True` 与默认 root-only 可见性的差异
- `nostream` 只影响 `messages`，不影响最终 state/result 的事实

### 6. `xfail` 在这个项目里不是丢脸，而是边界文档

`test_subagent_propagates_callbacks_to_model_calls` 当前就是很好的例子：

- 测试明确写了想守的语义
- 现实里它还没有成立
- 于是用 `xfail` 把当前缺口诚实钉住

这比两种做法都更好：

- 假装它已经支持了
- 或者干脆不写测试，让维护者继续猜

---

## 一张实用的最小测试矩阵

| 你改了什么 | 最少应该补什么 |
|------------|----------------|
| backend contract / path routing | unit test + 依赖它的一个 integration |
| profile / model resolving | `test_models.py` 级 unit + 一个 prompt/tool surface smoke |
| middleware 顺序 / 注入规则 | unit + snapshot/smoke |
| subagent state / interrupt / return 面 | `test_subagents.py` 级 unit + 一个 integration |
| callback / streaming / `nostream` 相关 | integration 或边界回归测试，必要时保留 `xfail` |
| example / CLI outer loop | 至少跑一个对应 example 或非交互 smoke |

---

## testing cookbook

### 场景 1：我改了 `graph.py` 默认装配顺序

最小配方：

- 一个 unit test，直接断言工具或 middleware 暴露面
- 一个 smoke/snapshot，断言模型可见 prompt / tool surface 没意外漂移

### 场景 2：我改了 subagent ingress / egress state

最小配方：

- 一个 unit test 断言 `_EXCLUDED_STATE_KEYS` 相关行为
- 一个 integration 验证 parent 最终只收到压缩后的 `ToolMessage` / state update

### 场景 3：我改了 callback/config propagation

最小配方：

- 一个 regression test 锁 tags / context / recursion limit
- 如果当前行为尚未成立，就写 `xfail`，不要在文档里偷写成既成事实

### 场景 4：我改了 streaming 可见性

最小配方：

- 至少覆盖 `messages` + `updates`
- 最好同时覆盖 `subgraphs=False` 与 `subgraphs=True`
- 如果用了 `nostream`，再补一个“最终结果仍回 parent”的测试

### 场景 5：我改了 backend / sandbox 能力

最小配方：

- unit：路径、返回形状、错误对象
- integration：通过 `FilesystemMiddleware` 跑一条真实 tool 链
- execute：成功路径 + 禁止/失败路径

### 场景 6：我改了 profile / provider 适配

最小配方：

- `test_models.py` 级 unit
- 一个 prompt/tool surface smoke
- 如果有 provider 特殊 middleware，再补一个针对该 provider 的装配测试

---

## 三个具体证据点

### 证据 1：streaming 可见性确实要测 subgraphs

`test_subagent_streaming_emits_messages_and_updates_from_subgraph` 说明：

- `subgraphs=True`
- `stream_mode=["messages", "updates"]`

这类边界应该直接被 integration-like test 守住，而不是靠阅读实现推断。

### 证据 2：recursion limit / tags / context 这类传播面可以写成正向回归

当前已有 tests 能守这些点：

- `test_subagent_propagates_recursion_limit_to_tool_runtime`
- `test_config_passed_to_runnable_lambda_subagent`
- `test_context_passed_to_subagent_tool_runtime`

这类测试是很标准的“跨层但仍可稳定断言”的好测试。

### 证据 3：callbacks 传播缺口就该诚实保留为 `xfail`

`test_subagent_propagates_callbacks_to_model_calls` 当前仍是 `xfail`。

它最有价值的地方不是“红着”，而是：

- 它阻止文档把这个行为写成既成事实
- 它给未来修复留了准确的回归钉子

---

## 一张“先写什么测试”决策表

| 现象 | 先写哪类测试 |
|------|--------------|
| prompt/tool surface 漂移 | smoke / snapshot |
| child state 异常回传 | unit + integration |
| token 可见性异常 | streaming 边界测试 |
| child runtime 拿不到 tags/context | propagation regression |
| provider/profile 默认值错了 | model/profile unit |
| 某 example wiring 失效 | example smoke |

---

## 什么时候该修上游

### 更像上游问题

- `CallbackManager` / `patch_config()` 行为不符合预期
- `StreamMessagesHandler`、`subgraphs=True`、`ToolRuntime` 行为与文档不一致
- provider chat model 的 stream/callback 行为异常

### 更像 Deep Agents 本地问题

- 默认 middleware 顺序漂移
- profile merge / tool exclusion 失效
- `CompiledSubAgent` 与 declarative subagent 的本地边界处理错误
- child state 过滤和最终结果折返不合理

---

## 容易踩什么坑

- 坑 1：只写“最终答案对不对”的测试。
  这对 callback、streaming、subagent 边界几乎没有保护力。

- 坑 2：因为某行为属于上游，就完全不在本地写回归。
  Deep Agents 依赖这些 contract，本地仍需要关键边界测试。

- 坑 3：把 snapshot 更新当成机械动作。
  每次更新前都该先回答“为什么模型可见面现在就该变化”。

- 坑 4：明知有缺口却不写 `xfail`。
  这样团队会不断重复误判“是不是我用错了”。

---

## 本章小结

- Deep Agents 的测试重点不是重复实现上游测试，而是守住本地 harness contract 与跨层边界。
- unit 测本地语义，integration 测边界组合，smoke/snapshot 测模型可见面。
- callback、streaming、subagent 这类问题必须显式测试，不能只看最终文本输出。
- `xfail` 在这种三层栈项目里是有效的已知限制文档。
