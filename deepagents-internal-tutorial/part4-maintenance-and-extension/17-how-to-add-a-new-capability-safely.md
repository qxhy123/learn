# 第17章：如何安全地新增一种跨三层能力

## 本章回答什么

- 新能力应该先落在哪一层，为什么“先判断 owner layer”比“先写代码”更重要
- 哪些 contract 必须先定义：模型可见面、state、streaming、interrupt / approval、结果折返
- 什么时候只改 Deep Agents，什么时候必须修 LangGraph / LangChain，什么时候应该留在 example / consumer 自己装配
- 为什么传播敏感的改动必须显式回到 Part 3，而不是在 Part 4 里顺手解释
- 作为维护工作流的收束章节，怎样把边界判断、examples 证据与测试验证串成一条安全改动路径

## 在整套系统中的位置

- 这一部分默认假设你已经读过 Part 1 和 Part 2。
- 如果当前问题和传播、可见性、callback tree 有关，先回看 Part 3。
- 横向主题：`Maintenance`、`Safe change`、`Capability design`
- 前置章节：[第13章：Backend 协议、存储介质与执行边界](./13-backend-protocol-and-storage-strategy.md)、[第14章：Provider Profiles、模型解析与 Middleware Surface](./14-provider-profiles-and-model-routing.md)、[第15章：如何测试一个三层栈 Harness](./15-testing-the-harness.md)、[第16章：像维护者一样阅读 Examples](./16-reading-the-examples-like-a-maintainer.md)
- 传播敏感背景：[第9章：传播层总览与四条线](../part3-propagation/09-propagation-overview-and-four-lanes.md) 到 [第12章：Subagent 传播矩阵与维护者 recipes](../part3-propagation/12-subagent-propagation-matrix-and-maintainer-recipes.md)

Part 4 到这里结束维护工作流的第二半：先在第 13 到 15 章判断边界并准备验证，再在第 16 章用 examples 校准证据，最后在本章把这些判断收束成一条安全落地路径。重点不是“如何加功能”，而是“如何避免把功能加在错误层、暴露错可见面、引入难回收的跨层回归”。

## 静态结构

做这类改动时，最常回看的文件通常是：

- `deepagents/libs/deepagents/deepagents/graph.py`
- `deepagents/libs/deepagents/deepagents/middleware/subagents.py`
- `deepagents/libs/deepagents/deepagents/middleware/filesystem.py`
- `deepagents/libs/deepagents/deepagents/middleware/async_subagents.py`
- `deepagents/libs/deepagents/deepagents/backends/`
- `langchain/libs/core/langchain_core/tools/base.py`
- `langchain/libs/core/langchain_core/runnables/config.py`
- `langgraph/libs/langgraph/langgraph/pregel/main.py`

先把三层职责静态拆开：

| 层 | 首先负责什么 | 维护时最容易误加成什么 |
| --- | --- | --- |
| `LangChain` | model/tool primitive、`RunnableConfig`、callback manager、provider 集成 | “Deep Agents 全局开关” |
| `LangGraph` | state graph、subgraph、checkpoint、`messages` / `updates` / `custom` streaming、`ToolRuntime` | “业务场景专用工作流” |
| `Deep Agents` | 默认 harness 装配、backend/profile/permissions/subagent policy、prompt layering、结果折返规则 | “上游 runtime 语义修补层” |

如果静态判断已经指向“这只是单个业务场景才需要的工作流”，那更可能属于 example / consumer 自己装配，而不是三层公共能力。

## 运行时链路

安全新增能力，建议固定按这条顺序推进。

### 1. 先做层次归属判断

先问自己四个问题：

- 这是 model/tool/callback/config primitive 问题吗
- 这是 graph runtime / subgraph / streaming / checkpoint 问题吗
- 这是默认 harness policy 问题吗
- 这是单个业务场景才需要的工作流吗

一个最实用的判断表：

| 需求 | 更适合哪层 |
| --- | --- |
| 新的 provider 默认参数或默认 tool exclusion | Deep Agents profile |
| 新的 tool 暴露面或默认 prompt policy | Deep Agents middleware / assembly |
| 新的存储或执行介质 | Deep Agents backend |
| token 可见性、`custom` 事件、subgraph streaming | LangGraph runtime / stream 配置 |
| callback tree / `RunnableConfig` merge 行为 | LangChain primitive |
| 单个场景专用 workflow | example / consumer 自己装配 |

### 2. 先定义 contract，再写实现

至少先写清楚五个面：

- 模型可见面：prompt / tools / descriptions 会怎么变
- state 面：新增哪些 key，如何 reducer，哪些是 private
- streaming 面：哪些事件会被外部流消费者看到
- interrupt / approval 面：谁能暂停谁，在哪一层暂停
- return 面：最终哪些结果能回到 parent / caller

这一步最容易被跳过，但跨三层能力一旦先写实现、后补 contract，最后往往就是各层各自猜语义。

### 3. 用最小层次实现，不要先动 `graph.py`

优先顺序通常应该是：

1. 能在现有上游 primitive / runtime 配置里实现，就不要改装配根。
2. 能放进单一 middleware / backend / profile，就不要改全局 assembly。
3. 只有当默认装配本身必须变化时，才去改 `create_deep_agent()` / `graph.py`。

这能最大化降低回归面，也让第 15 章里的测试矩阵更容易最小化。

### 4. 如果是 compiled subagent 的能力，优先在子图内部解决

例如你想要：

- 子代理内部自己的审批规则
- 子代理内部自己的 token 可见性策略
- 子代理内部自己的私有 planning state

通常都应该优先在 compiled runnable 自己内部加 middleware / node / stream 策略，而不是期待父图顶层开关自动伸进去。

### 5. 先补局部验证，再接默认装配

推荐顺序：

1. unit test 锁住局部 contract
2. integration test 锁住跨层边界
3. 再接入默认装配
4. 最后跑 smoke / snapshot / example 验证

这样失败时你更容易知道，是局部实现错了，还是 assembly 顺序把行为改坏了。

### 6. 改完后回头更新边界文档

对这套栈来说，文档不是收尾装饰，而是 contract 的一部分。尤其是涉及 callback 传播、streaming 可见性、compiled subagent 边界、permissions 与 execute capability 时，教程和测试都必须同步更新。

## 传播 / 可见性 / 拦截点

跨三层能力最危险的地方，不是“功能不能跑”，而是你把执行、观测、流输出、最终结果折返写成了一件事。

### 传播敏感改动必须先回 Part 3

如果你的新能力涉及下面这些问题，就不要在 Part 4 里继续自创解释：

- callback tree 怎样接起来
- `RunnableConfig`、tags、metadata、context 如何继续进入 child runtime
- `messages`、`updates`、`custom`、`subgraphs=True` 分别代表什么
- `nostream` 过滤的到底是 token、阶段事件，还是最终结果

统一回跳规则如下：

- streaming 的说明统一回看第9章到第12章：[第9章](../part3-propagation/09-propagation-overview-and-four-lanes.md)、[第10章](../part3-propagation/10-callbacks-config-and-callback-manager.md)、[第11章](../part3-propagation/11-streaming-visibility-and-selective-exposure.md)、[第12章](../part3-propagation/12-subagent-propagation-matrix-and-maintainer-recipes.md)
- subagent + callback 的混合说明统一回看第10章与第12章：[第10章](../part3-propagation/10-callbacks-config-and-callback-manager.md)、[第12章](../part3-propagation/12-subagent-propagation-matrix-and-maintainer-recipes.md)
- 可见性速查表回跳统一回看第11章 + 附录 D：[第11章](../part3-propagation/11-streaming-visibility-and-selective-exposure.md)、[附录 D](../appendix/propagation-and-visibility-cheatsheet.md)

### 五个最关键的拦截点

- 模型可见面：prompt、tools、descriptions 是否改变
- state 面：child private state 与 parent-visible state 是否分开
- 流输出面：外部 consumer 能看到哪些事件
- interrupt 面：审批与暂停到底发生在哪一层
- return 面：最终结果是 `ToolMessage`、`Command(update=...)`，还是别的 handoff 形式

如果这五个点里有任意一个没定义清楚，就不算“安全地新增能力”。

### 两个典型高风险例子

#### 例子 1：我想控制子代理内部哪些 token 对流消费者可见

这类需求不要先去找 Deep Agents 的“全局隐藏开关”。更合理的拆法是：

- token 是否进入 `messages` 流：LangGraph
- 是否对 root consumer 暴露子图事件：`subgraphs=True/False`
- 是否只发阶段信号：`custom` + `runtime.stream_writer(...)`
- 私有中间结果是否最终回到 parent：Deep Agents 的 return / state 边界

常见可行解法是：

- 私有模型调用打 `tags=["nostream"]`
- 私有草稿不写回 parent 可见 state
- 公开阶段用 `custom` 事件通知 UI

#### 例子 2：我想让 compiled subagent 内部工具也被父审批规则拦住

这通常不是父 `interrupt_on` 能自动做到的。因为声明式 `SubAgent` 会在构建子图时显式加对应 middleware，而 `CompiledSubAgent` 是直接复用现成 runnable。正确方向通常是：

- 在 compiled subagent 自己内部加 HITL / middleware
- 或者改回 declarative subagent 路径

而不是继续增强父图顶层开关，期待它透明穿透 child graph。

## 扩展接口

### profile / middleware / upstream 的选择表

| 需求 | 更适合哪层 |
| --- | --- |
| OpenAI 某模型默认 init kwargs | profile |
| 某 provider 默认禁用一个 built-in tool | profile |
| 按请求内容动态切快/慢模型 | 上层策略或 `wrap_model_call` |
| 给所有 agent 加一层 provider 专用 middleware | profile 的 `extra_middleware` |
| 修某 provider SDK 的参数支持 | 上游 `langchain` provider 集成 |
| 某个 example 的 research prompt | example / consumer 自己配置 |

### capability cookbook

#### 场景 1：你只想新增一个 provider / model 级默认行为

优先改 profile，不要把 provider 差异散落进 `graph.py`。

#### 场景 2：你想多暴露一个新的工具面

优先看 middleware 或装配层，而不是 backend。backend 解决的是运行介质，tool surface 解决的是模型可见面。

#### 场景 3：你想新增一种存储或执行介质

优先改 backend / sandbox contract，再决定是否需要让默认 harness 暴露这条能力线。

#### 场景 4：你想把一个 example 里验证过的模式抽回库里

先回 [第16章：像维护者一样阅读 Examples](./16-reading-the-examples-like-a-maintainer.md)，确认那到底是通用装配逻辑，还是 example 私有 helper / outer loop / 产品化脚本。只有多处重复、与具体场景无关、并且有最小测试矩阵可守的逻辑，才适合抽回库里。

## 常见问题与排障入口

- “我是不是应该先改 `graph.py`”：默认不是。装配根应该是最后动的地方，而不是默认入口。
- “这个能力已经能跑了，为什么还说不安全”：因为你可能还没定义哪些内容可见、可拦截、可回传。
- “为什么 profile、middleware、backend 看起来都能改”：能改不代表都该改；先用 owner layer 判断表收敛。
- “我只想修一个 streaming 现象，为什么这里反复让我回 Part 3”：因为传播 / 可见性 contract 的定义不在 Part 4，这里只负责维护流程，不负责重讲传播理论。
- “compiled subagent 为什么不像 declarative subagent 那样自动继承父规则”：因为两者装配路径不同；需要时应优先在 child runnable 内部解决。

更像上游问题：

- `patch_config()` / callback tree / `get_child()` 语义不对
- `messages` / `custom` / `subgraphs` streaming 与预期不一致
- provider 集成本身缺少需要的模型能力

更像 Deep Agents 本地问题：

- 默认 middleware / backend / profile / permissions policy 不合理
- declarative 与 compiled subagent 边界处理不一致
- prompt layering、memory、skills 的装配策略不合理

## 本章结论

- 谁提供：`LangChain` 提供 primitive 和 callback/config contract，`LangGraph` 提供 runtime、streaming 与 subgraph 执行语义，`Deep Agents` 把它们装配成默认 harness policy。
- 如何传播：先做 owner layer 判断，再定义模型可见面、state、streaming、interrupt、return 五个 contract；凡是传播敏感改动，一律回到 Part 3 的第9章到第12章和附录 D。
- 修在哪层：静态 provider / tool / backend / policy 问题修在最小本地层，runtime / callback / streaming 语义问题修在上游或显式按 Part 3 分流，业务专用 workflow 留在 example / consumer 自己装配。
