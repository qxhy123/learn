# 第15章：如何测试一个三层栈 Harness

## 本章回答什么

- 哪些行为可以信任上游测试，哪些必须在 Deep Agents 本地钉死
- unit、integration、smoke/snapshot 在这个项目里各自守什么
- callback / streaming / subagent 这类跨层边界问题该怎样写成维护者能复用的回归测试
- 已知限制什么时候该用 `xfail` 诚实记录下来
- 面对具体改动时，最小测试配方应该怎么选

## 在整套系统中的位置

- 这一部分默认假设你已经读过 Part 1 和 Part 2。
- 如果当前问题和传播、可见性、callback tree 有关，先回看 Part 3。
- 横向主题：`Maintenance`、`Validation`、`Regression`
- 前置章节：[第13章：Backend 协议、存储介质与执行边界](./13-backend-protocol-and-storage-strategy.md)、[第14章：Provider Profiles、模型解析与 Middleware Surface](./14-provider-profiles-and-model-routing.md)
- 传播敏感的背景章节：[第9章：传播层总览与四条线](../part3-propagation/09-propagation-overview-and-four-lanes.md) 到 [第12章：Subagent 传播矩阵与维护者 recipes](../part3-propagation/12-subagent-propagation-matrix-and-maintainer-recipes.md)、[附录 D：传播与可见性速查表](../appendix/propagation-and-visibility-cheatsheet.md)

Part 4 的前两章先教你判边界，本章才处理“如何证明你真的修对了”。对一个三层栈 harness 来说，测试的核心不是多写 happy path，而是先把 ownership、传播线和模型可见面拆开，再决定该在本地钉哪类回归。

## 静态结构

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

先把测试对象静态分类：

| 测试层 | 在这个项目里主要守什么 | 容易被误当成什么 |
| --- | --- | --- |
| unit | backend contract、profile lookup、middleware state update、parent-child state 过滤、`interrupt_on` 继承与覆盖 | “把整个 agent 跑通” |
| integration | `task` handoff、subagent 语义分叉、`subgraphs=True` 下的边界组合、context / tags / recursion limit 传播 | “重复上游所有 happy path” |
| smoke / snapshot | prompt surface、tool surface、默认 middleware 暴露面、provider/profile 改动后的模型可见面 | “文案层细节，不值一测” |

如果一个测试问题涉及传播敏感行为，先按 Part 3 分流：

- streaming 的旧说明不要再回跳到零散旧章，统一回看第9章到第12章：[第9章](../part3-propagation/09-propagation-overview-and-four-lanes.md)、[第10章](../part3-propagation/10-callbacks-config-and-callback-manager.md)、[第11章](../part3-propagation/11-streaming-visibility-and-selective-exposure.md)、[第12章](../part3-propagation/12-subagent-propagation-matrix-and-maintainer-recipes.md)
- subagent + callback 的混合判断，优先回看第10章与第12章：[第10章](../part3-propagation/10-callbacks-config-and-callback-manager.md)、[第12章](../part3-propagation/12-subagent-propagation-matrix-and-maintainer-recipes.md)
- 可见性速查表的回跳，统一写成第11章 + 附录 D：[第11章](../part3-propagation/11-streaming-visibility-and-selective-exposure.md)、[附录 D](../appendix/propagation-and-visibility-cheatsheet.md)

## 运行时链路

### 1. 测试前先做 ownership 分类

每次改动前，先问四个问题：

1. 这是本地 contract，还是上游 contract
2. 我是在改装配结果，还是改 primitive / runtime 行为
3. 这个变化会不会影响 prompt/tool surface
4. 这个变化会不会影响 subagent、callback、streaming 这种跨层边界

很多测试策略差异，其实都来自这四问。

### 2. unit tests 锁的是本地 contract

在这个项目里，unit tests 最适合守这些东西：

- backend 返回值与路径路由规则
- `_HarnessProfile` lookup / merge
- middleware 的 state update
- parent-child state 过滤
- `interrupt_on` 继承与覆盖策略

### 3. integration tests 锁的是边界组合

Deep Agents 最有价值的 integration test，通常都不是“主 agent 回了一句正确答案”，而是：

- `task` handoff 是否发生
- declarative 与 compiled subagent 的语义是否分叉正确
- `subgraphs=True` 时子图 streaming 是否可观察
- context / tags / recursion limit 是否继续进入 child runtime

### 4. smoke / snapshot 守的是模型可见面

对 harness 来说，prompt surface 和 tool surface 不是“文案层细节”，而是行为 contract。它们会直接暴露：

- base system prompt 变了没有
- tool descriptions 变了没有
- 某个 middleware 是否真的进了最终模型可见面
- profile / permissions / skills / memory 是否悄悄改变了暴露面

### 5. propagation-sensitive 测试必须显式回到 Part 3

如果问题涉及 callback、streaming、subagent、visibility，不要再写成“某个测试里顺手一起看一下”。

更稳妥的定位是：

- streaming 相关说明统一回到第9章到第12章：[第9章](../part3-propagation/09-propagation-overview-and-four-lanes.md)、[第10章](../part3-propagation/10-callbacks-config-and-callback-manager.md)、[第11章](../part3-propagation/11-streaming-visibility-and-selective-exposure.md)、[第12章](../part3-propagation/12-subagent-propagation-matrix-and-maintainer-recipes.md)
- subagent + callback 的混合解释回到第10章与第12章：[第10章](../part3-propagation/10-callbacks-config-and-callback-manager.md)、[第12章](../part3-propagation/12-subagent-propagation-matrix-and-maintainer-recipes.md)
- 可见性速查回到第11章 + 附录 D：[第11章](../part3-propagation/11-streaming-visibility-and-selective-exposure.md)、[附录 D](../appendix/propagation-and-visibility-cheatsheet.md)

这样测试章节就不再重复承担传播理论，而是承担验证与回归职责。

### 6. `xfail` 在这里是边界文档

`test_subagent_propagates_callbacks_to_model_calls` 这类 case 的价值不在于“红着”，而在于：

- 测试明确写了想守的语义
- 现实里它还没有成立
- 于是用 `xfail` 把当前缺口诚实钉住

这比假装已经支持或干脆不写都更有维护价值。

## 传播 / 可见性 / 拦截点

测试章节要特别防止把“执行发生了”“系统观测到了”“流消费者看到了”“最终结果折返了”写成同一句。

### callback / config 传播

如果 child runtime 拿不到 tags、metadata、context、`recursion_limit`，优先把回归测试定位到 [第10章：Callbacks、Config 与 Callback Manager](../part3-propagation/10-callbacks-config-and-callback-manager.md) 解释的传播线，再决定是写 unit 还是 integration。

当前已有 tests 能守这些点：

- `test_subagent_propagates_recursion_limit_to_tool_runtime`
- `test_config_passed_to_runnable_lambda_subagent`
- `test_context_passed_to_subagent_tool_runtime`

### streaming / visibility

如果你要测 token、`messages`、`updates`、`nostream` 或 `subgraphs=True`，就不要再回跳旧的 streaming 章节说明，而是统一按第9章到第12章的框架来写：[第9章](../part3-propagation/09-propagation-overview-and-four-lanes.md)、[第10章](../part3-propagation/10-callbacks-config-and-callback-manager.md)、[第11章](../part3-propagation/11-streaming-visibility-and-selective-exposure.md)、[第12章](../part3-propagation/12-subagent-propagation-matrix-and-maintainer-recipes.md)。

`test_subagent_streaming_emits_messages_and_updates_from_subgraph` 这类测试最有价值的地方，就在于它直接守了：

- `subgraphs=True`
- `stream_mode=["messages", "updates"]`

而不是靠人读实现去猜。

### subagent 边界

如果一个问题混合了 subagent 类型与 callback 传播，不要在同一段话里把它写成“subagent 机制有问题”。更好的回跳是第10章与第12章：[第10章](../part3-propagation/10-callbacks-config-and-callback-manager.md) 解释 callback/config，[第12章](../part3-propagation/12-subagent-propagation-matrix-and-maintainer-recipes.md) 解释 `SubAgent`、`CompiledSubAgent`、`AsyncSubAgent` 的边界差异。

### 可见性速查

当你只是想快速确认“这个现象属于执行线、观测线、流输出线还是结果折返线”，不要在测试章节重复造一张简化表，直接回跳到第11章 + 附录 D：[第11章](../part3-propagation/11-streaming-visibility-and-selective-exposure.md)、[附录 D](../appendix/propagation-and-visibility-cheatsheet.md)。

## 扩展接口

### 一张实用的最小测试矩阵

| 你改了什么 | 最少应该补什么 |
| --- | --- |
| backend contract / path routing | unit test + 依赖它的一个 integration |
| profile / model resolving | `test_models.py` 级 unit + 一个 prompt/tool surface smoke |
| middleware 顺序 / 注入规则 | unit + snapshot/smoke |
| subagent state / interrupt / return 面 | `test_subagents.py` 级 unit + 一个 integration |
| callback / streaming / `nostream` 相关 | integration 或边界回归测试，必要时保留 `xfail` |
| example / CLI outer loop | 至少跑一个对应 example 或非交互 smoke |

### testing cookbook

#### 场景 1：我改了 `graph.py` 默认装配顺序

最小配方：

- 一个 unit test，直接断言工具或 middleware 暴露面
- 一个 smoke/snapshot，断言模型可见 prompt / tool surface 没意外漂移

#### 场景 2：我改了 subagent ingress / egress state

最小配方：

- 一个 unit test 断言 `_EXCLUDED_STATE_KEYS` 相关行为
- 一个 integration 验证 parent 最终只收到压缩后的 `ToolMessage` / state update

#### 场景 3：我改了 callback/config propagation

最小配方：

- 一个 regression test 锁 tags / context / recursion limit
- 如果当前行为尚未成立，就写 `xfail`，不要在文档里偷写成既成事实

#### 场景 4：我改了 streaming 可见性

最小配方：

- 至少覆盖 `messages` + `updates`
- 最好同时覆盖 `subgraphs=False` 与 `subgraphs=True`
- 如果用了 `nostream`，再补一个“最终结果仍回 parent”的测试

#### 场景 5：我改了 backend / sandbox 能力

最小配方：

- unit：路径、返回形状、错误对象
- integration：通过 `FilesystemMiddleware` 跑一条真实 tool 链
- execute：成功路径 + 禁止/失败路径

#### 场景 6：我改了 profile / provider 适配

最小配方：

- `test_models.py` 级 unit
- 一个 prompt/tool surface smoke
- 如果有 provider 特殊 middleware，再补一个针对该 provider 的装配测试

## 常见问题与排障入口

- “我只写了最终答案对不对的测试，为什么还是挡不住回归”：因为这几乎守不住 callback、streaming、subagent 边界。
- “某行为属于上游，所以我是不是完全不用在本地写测试”：不是。Deep Agents 依赖这些 contract，本地仍需要关键边界回归。
- “snapshot 更新是不是机械动作”：不是。每次更新前都该先回答为什么模型可见面现在就该变化。
- “明知有缺口却不写 `xfail` 行不行”：不行；这样只会让团队继续重复误判。
- “默认 middleware 顺序、profile merge、tool exclusion 失效了，该先看哪”：这更像本地 harness contract 问题。
- “`CallbackManager` / `patch_config()`、`StreamMessagesHandler`、`ToolRuntime` 行为与文档不一致，该先看哪”：这更像上游问题。

更像上游问题：

- `CallbackManager` / `patch_config()` 行为不符合预期
- `StreamMessagesHandler`、`subgraphs=True`、`ToolRuntime` 行为与文档不一致
- provider chat model 的 stream/callback 行为异常

更像 Deep Agents 本地问题：

- 默认 middleware 顺序漂移
- profile merge / tool exclusion 失效
- `CompiledSubAgent` 与 declarative subagent 的本地边界处理错误
- child state 过滤和最终结果折返不合理

## 本章结论

- 谁提供：上游 `LangChain` / `LangGraph` 提供 primitive、runtime 和观测面，`Deep Agents` 负责把它们装配成 harness contract，因此测试必须同时覆盖本地 contract 与关键跨层边界。
- 如何传播：先做 ownership 分类，再按 unit、integration、smoke/snapshot 分配验证；凡是传播敏感测试，一律回到第9章到第12章和附录 D 的框架去定位。
- 修在哪层：装配、profile、permissions、state return 这类问题修本地 harness；callback/config、streaming runtime、provider 原生行为问题则应明确区分是否属于上游。
