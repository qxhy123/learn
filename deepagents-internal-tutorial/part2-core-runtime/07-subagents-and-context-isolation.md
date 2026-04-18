# 第5章：Subagents、拦截边界与上下文隔离

## 学习目标

学完本章，你应该能回答：

1. `SubAgent`、`CompiledSubAgent`、`AsyncSubAgent` 的差别是什么
2. `task` 工具是如何被暴露出来的
3. 为什么 general-purpose subagent 默认存在
4. `CompiledSubAgent` 内部的 LLM / tool / interrupt / callbacks / stream 事件，哪些会被主 agent 看到，哪些不会
5. 如果不想让主 agent 干预 compiled subagent，应该怎么做

---

## 问题是什么

Deep Agents 之所以能在复杂任务里保持“深度”，另一个关键是它不要求所有工作都在主线程里做完。它允许主 agent 把一个独立任务切给 subagent，在隔离上下文里完成后再把结果拿回来。

这里真正重要的不是“能不能再起一个 agent”，而是：

- 谁来决定子代理类型
- 子代理拿到哪些上下文
- 子代理完成后把什么带回来
- 同步和异步版本如何区分
- 主 agent 到底能不能看见、暂停、改写子代理内部行为

---

## 哪一层负责什么

### `LangChain`

- `task` 最终仍沿着 tool / runnable / `RunnableConfig` / callback manager 链路执行
- callback/config 传播主要是上游 primitive 行为

### `LangGraph`

- subgraph namespace
- `subgraphs=True`
- `messages` / `updates` / `custom` 流
- `StreamMessagesHandler`、`TAG_NOSTREAM`

### `Deep Agents`

- declarative / compiled / async subagent 的装配分流
- parent state 过滤与 child result 折返
- general-purpose subagent 默认注入
- `interrupt_on` 的本地继承策略

---

## 代码在哪里

重点看：

- [`deepagents/libs/deepagents/deepagents/graph.py`](../../deepagents/libs/deepagents/deepagents/graph.py)
- [`deepagents/libs/deepagents/deepagents/middleware/subagents.py`](../../deepagents/libs/deepagents/deepagents/middleware/subagents.py)
- [`deepagents/libs/deepagents/deepagents/middleware/async_subagents.py`](../../deepagents/libs/deepagents/deepagents/middleware/async_subagents.py)
- [`deepagents/libs/deepagents/tests/unit_tests/test_subagents.py`](../../deepagents/libs/deepagents/tests/unit_tests/test_subagents.py)
- [`deepagents/libs/deepagents/tests/integration_tests/test_subagent_middleware.py`](../../deepagents/libs/deepagents/tests/integration_tests/test_subagent_middleware.py)

如果你要继续追到上游运行时实现，还要看本地环境里的这些包路径：

- `langgraph.pregel.main.Pregel.stream`
- `langgraph.pregel._messages.StreamMessagesHandler`
- `langgraph.constants.TAG_NOSTREAM`
- `langchain_core.runnables.config.ensure_config`
- `langchain_core.tools.base.BaseTool.run`
- `langchain_core.language_models.chat_models.BaseChatModel.invoke`
- `langchain_core.language_models.chat_models.BaseChatModel.stream`

---

## 实现怎么工作

### 1. 三种 subagent surface 有三条不同装配路径

#### `SubAgent`

声明式配置。你给出：

- `name`
- `description`
- `system_prompt`
- 可选 `tools` / `model` / `middleware` / `skills` / `permissions` / `interrupt_on`

`graph.py` 会先替你补齐默认 Deep Agents middleware 栈，然后 `SubAgentMiddleware._get_subagents()` 再调用 `create_agent()` 真正构出子图。

#### `CompiledSubAgent`

你提供一个现成 `runnable`。Deep Agents 不再替你构造内部 agent，只要求它的 state schema 里有 `messages`，这样父 agent 才能从最终消息里提取结果。

在实现上有两个关键点：

- `graph.py` 发现 spec 里有 `runnable` 时，直接把它放进 `inline_subagents`
- `SubAgentMiddleware._get_subagents()` 再次看到 `runnable` 时，直接把这个 runnable 原样塞进 task tool 用的 `subagent_graphs`

因此，`CompiledSubAgent` 的内部节点不会被主 agent 默认 middleware 栈重写。

#### `AsyncSubAgent`

它不是本地 runnable，而是远端异步任务接口。`graph.py` 会把这类 spec 交给 `AsyncSubAgentMiddleware`，而不是同步 `SubAgentMiddleware`。

### 2. `task` 工具是 parent-child handoff 的唯一正式入口

主 agent 真正调用的不是某个子代理对象，而是 `task` 工具。`TaskToolSchema` 要求至少提供：

- `description`
- `subagent_type`

这个设计很重要，因为它把 parent-child 协作抽象成“工具调用”，从而继续沿用 LangChain agent 的统一 tool call 语义。

`_build_task_tool()` 里的核心 handoff 逻辑是：

1. 用 `_validate_and_prepare_state()` 从 `runtime.state` 派生一个新的 `subagent_state`
2. 过滤 `_EXCLUDED_STATE_KEYS`
3. 把子任务描述作为新的 `HumanMessage` 放进子代理输入
4. 调用 `subagent.invoke(...)` 或 `subagent.ainvoke(...)`
5. 用 `_return_command_with_state_update()` 把结果压成一个回到主线程的 `ToolMessage`

这条路径解释了一个关键事实：

> 主 agent 看到的是一次 `task` 工具调用和一次任务结果回传，而不是子代理内部所有节点天然成为主图的一部分。

### 3. general-purpose subagent 默认存在

如果用户没有自己提供名为 `general-purpose` 的 subagent，`graph.py` 会自动插入一个。

它的作用不是“兜底文案”，而是保证主 agent 随时可以把一个复杂但不需要专业技能的任务丢进隔离上下文里做。这样即使没有专用子代理，Deep Agents 仍然具备 context isolation 能力。

### 4. parent state、child state、返回状态是三件不同的事

`subagents.py` 中显式定义了 `_EXCLUDED_STATE_KEYS`，过滤：

- `messages`
- `todos`
- `structured_response`
- `skills_metadata`
- `memory_contents`

这说明 Deep Agents 对状态透传是克制的。原因包括：

- 避免 parent 历史污染 child
- 避免没有明确定义 reducer 的 state 被错误回传
- 避免 parent 的 skills/memory 泄漏到 child，打破 child 自己的 prompt layering

返回给 child 的输入状态，不等于 child 运行中的完整状态；返回给 parent 的更新，也不等于 child 的完整最终状态。

### 5. 返回给 parent 的不是整个 child state，而是压缩后的回传面

subagent 完成后，parent 通常拿到的是最终消息，或者结构化输出序列化后的结果，而不是整个 child graph 的内部状态。

这是 deliberate design：主线程需要的是结果摘要，而不是把 child 的所有工具调用历史再塞回主上下文里。

`_return_command_with_state_update()` 的逻辑也说明了这一点：

- 必须有 `messages`
- `structured_response` 会被序列化成 `ToolMessage.content`
- `todos`、`structured_response`、`skills_metadata`、`memory_contents` 等不会冒泡回 parent

---

## 主 agent 能看到什么

这一节专门区分“可见性”和“拦截性”。

### 1. 主 agent 一定能看到 `task` 这次外层调用

因为 `task` 是主 agent 自己的工具调用。对主 agent 来说，delegate 发生时首先只是一次普通 tool call。

### 2. 主 agent 通常只拿到一个压缩后的任务结果

默认情况下，主 agent 在任务结束后收到的是：

- 一个 `ToolMessage`
- 加上少量允许冒泡的 state update

这就是为什么子代理可以做很多内部步骤，但主线程历史仍然保持干净。

### 3. `subgraphs=True` 能让外部观察者看到子图事件

现有 integration / unit tests 都覆盖了这一点：如果你用 `agent.stream(..., subgraphs=True)`，流消费者可以看到：

- 主 agent 的 model/tool 更新
- 子代理子图里的 model 更新
- 子代理返回给父图的 tool result

这是一种**观测能力**。它说明流式消费者能看见子图事件，但这不等于主 agent middleware 正在包裹子代理内部节点。

还要再区分一层：

- runtime 流里是否真的发出了这些事件
- 最终 CLI / UI 是否选择展示这些事件

Deep Agents CLI 常常会主动过滤非 root namespace，所以“runtime 可见”不等于“最终用户界面一定显示”。

### 4. compiled subagent 的内部 LLM 调用可以被观测，但不是被父 middleware 包裹

如果 compiled runnable 自己内部有 model node，那么在 `subgraphs=True` 流式消费下，外部可以看到这些 model updates。当前测试也覆盖了子图 message chunks 和 updates。

但代码路径仍然是：

- `task` 工具直接调用 compiled runnable
- compiled runnable 自己执行内部图
- 再把结果压回父图

所以，“看得到”不代表“主 agent 正在控制它”。

### 5. `messages` 流不是 Deep Agents 自己实现的，而是 LangGraph 在运行时挂出来的

这点非常容易误判。

当前代码里，Deep Agents 并没有自己维护一套“token 流分发器”。真正的路径是：

1. `create_deep_agent()` 返回的是 LangGraph compiled graph。
2. 外层调用 `agent.stream(..., stream_mode=["messages", ...])`。
3. LangGraph 在 `Pregel.stream()` 里检查 `stream_mode`。
4. 如果包含 `"messages"`，就把 `StreamMessagesHandler` 挂到当前 run manager 的 `inheritable_handlers` 上。
5. 后续 parent graph 和 subgraph 内部的 chat model 调用，只要走进这条 callback 链，就会把 token chunk 发到外层 stream consumer。

所以这里要严格区分：

- Deep Agents 负责把 subagent 挂进主图。
- LangGraph 负责把子图内部的 message/token 事件暴露到流式消费面。

### 6. `nostream` 不是 Deep Agents 自己定义的 tag

这也是一个边界问题。

`nostream` 来自上游 LangGraph，而不是 `deepagents` 仓库：

- 常量定义在 `langgraph.constants.TAG_NOSTREAM`
- 消费逻辑在 `langgraph.pregel._messages.StreamMessagesHandler.on_chat_model_start()`

这意味着：

- 你在 Deep Agents 里看到 `config={"tags": ["nostream"]}` 有效，不是因为 Deep Agents 特判了这个 tag
- 而是因为底层运行时在 `messages` 流处理器里识别了它

同类的还有一个上游 tag：`langsmith:hidden`。它更偏向 chain/node 级别的隐藏，而不是 chat model token 级别的隐藏。

### 7. 想控制“哪些 token 对流消费者可见”，当前可用的是分层控制，不是单一开关

当前实现里，没有发现 Deep Agents 自带的“按 subagent / 按 node / 按 token 类型”的细粒度策略层。现在真正可用的控制点有这些：

| 目标 | 主要控制点 | 外部流消费者看到什么 | 本质属于哪一层 |
|------|------------|----------------------|----------------|
| 完全隐藏子图内部流 | `subgraphs=False` | 只看到 root graph 的事件 | LangGraph stream 边界 |
| 只看步骤更新，不看 token | `stream_mode="updates"` | 看到 step/update，不看到逐 token chunk | LangGraph stream mode |
| 隐藏某次内部 LLM 调用的 token | 在该次模型调用上打 `config={"tags": ["nostream"]}` | 该次调用不会进入 `messages` 流 | LangGraph callback handler |
| 只暴露你允许看到的阶段信号 | `stream_mode=["updates", "custom"]` + `runtime.stream_writer(...)` | 看到你主动写出的 custom event | LangGraph runtime |
| 只在 UI 侧隐藏部分事件 | 消费端按 namespace / `lc_agent_name` / `langgraph_node` 过滤 | UI 不展示某些事件，但底层事件其实已经发出 | 消费端过滤 |

这里最重要的误区是：

> “对流消费者不可见”不等于“对父 agent 不可见”。

因为子代理结束后，`_return_command_with_state_update()` 仍然会把允许冒泡的 state update 和最终 `ToolMessage` 回给父图。

也就是说：

- `nostream` 只能阻止进入 `messages` 流
- 它不会自动阻止写回 parent state
- 如果 private intermediate data 最后被你留在 child result 里，它仍可能通过 `updates` 或最终状态暴露出去

### 8. 一个最实用的可见性设计：私有推理 + 公共事件

如果你的目标不是“彻底隐身”，而是“只让用户看到我允许公开的阶段信息”，最实用的是这套组合：

1. 私有规划节点的模型调用用 `tags=["nostream"]`
2. 不把规划草稿写回共享 `messages` 或共享 state
3. 用 `stream_mode="custom"` + `runtime.stream_writer(...)` 主动发一个脱敏后的阶段事件

这样你能做到：

- 内部 token 不外露
- UI 仍然知道子代理进行了规划、检索、汇总等阶段
- 父图最后只收到你明确允许回传的结果面

---

## 主 agent 能拦截什么

### 1. 父级 `interrupt_on` 首先作用于父图自己的工具

如果主 agent 顶层配置了 `interrupt_on`，它一定先作用于主图自己的工具面。`task` 本身就是其中一员。

因此：

- 如果你把 `task` 放进父级 `interrupt_on`，主 agent 可以在**启动子代理之前**暂停
- 这属于“拦截 delegation 行为”，不是进入子代理内部拦截

### 2. declarative `SubAgent` 可以继承父级 `interrupt_on`

这是因为 `graph.py` 会先把 `interrupt_on` 合并进 declarative spec，然后 `SubAgentMiddleware._get_subagents()` 在真正 `create_agent()` 时把 `HumanInTheLoopMiddleware` 加进子代理自己的 middleware 栈。

这里的关键是：

- 子代理之所以被“父级规则”拦到
- 不是因为父 middleware 伸进了 child graph
- 而是因为 child graph 在构建时就显式带上了相应 HITL middleware

### 3. `CompiledSubAgent` 不继承父级 `interrupt_on`

这点在 `graph.py` 的参数文档里写得很直接：`CompiledSubAgent` 不继承 top-level `interrupt_on`。

代码原因也很直接：

- compiled 分支是 use-as-is
- `SubAgentMiddleware._get_subagents()` 对有 `runnable` 的 spec 不会再追加 `HumanInTheLoopMiddleware`

所以，如果一个 compiled subagent 内部 node 调用了 LLM 或工具：

- 父级 `interrupt_on` 不会自动进入该内部图
- 父 agent 不会因为自己配置了 HITL，就自然拦住 compiled runnable 里的内部节点

### 4. 父级 callbacks 与父级 HITL 不是一回事

现有测试明确给出一个已知限制：父 config 里的 callbacks 当前**不会可靠传播**到 subagent model invocations，测试是 xfail。

这说明“拦截/回调/观测”在当前实现里是不同层次的问题，不能混为一谈。

---

## callbacks 和 callback manager 在这条链路里怎么工作

前面讲的是“能看到什么”和“能拦截什么”，这里再单独把 callback 链路拆开。

### 1. callback tree 主要不是 Deep Agents 自己搭的，而是上游 runnable/tool/runtime 在搭

可以把这条链路理解成：

1. 外层 runnable/tool 进入执行
2. 上游 runtime 创建当前 run 的 `run_manager`
3. `BaseTool.run()` 用 `run_manager.get_child()` 给子调用 patch 一份 child callbacks
4. `ensure_config()` 再把当前上下文中的 `callbacks`、`tags`、`metadata` 合并进后续 runnable config
5. `BaseChatModel.invoke()` / `.stream()` 把这些 config 字段继续传到 model run

这解释了为什么：

- tags 能进入 compiled subagent 的内部调用
- `recursion_limit` 能进入子代理工具运行时
- context 能通过 `ToolRuntime.context` 进入子代理工具

但要明确一点：Deep Agents 在 `task` handoff 里并没有显式把父级 config 原样传给 `subagent.invoke(...)`。很多 tags/context/config 之所以还能进入 child，依赖的是上游 `patch_config()`、`set_config_context()` 和 ambient propagation 机制。

### 2. 这条 callback/config 链解释的是“传播”，不是“继承父 middleware”

很多人会把它和 middleware 继承混在一起，但两者完全不同：

- middleware 继承讲的是 child graph 在装配时带没带某层逻辑
- callback/config 传播讲的是运行时调用时，config 有没有顺着 runnable stack 往下传

所以：

- `CompiledSubAgent` 不继承父级 HITL middleware
- 但它仍然可能因为共享 runtime config，而让部分 tags / callbacks / metadata 被下游看见

### 3. 当前真正有证据支持的传播面

现有测试已经覆盖到这些点：

- `test_config_passed_to_runnable_lambda_subagent`
  - 证明 tags 会进到 runnable config
- `test_context_passed_to_subagent_tool_runtime`
  - 证明 parent context 会进到 child tool runtime
- `test_subagent_propagates_recursion_limit_to_tool_runtime`
  - 证明部分 config 会和 child 自己的 config 合并，而不是被粗暴替换

### 4. 当前没有证据支持、反而有反例的传播面

`test_subagent_propagates_callbacks_to_model_calls` 当前是 `xfail`，明确写着：

- parent config 里的 callbacks 当前不会可靠传播到 subagent model invocations

这意味着教程里不能把“父 callbacks 会进入 compiled subagent 内部 LLM 调用”写成稳定能力。当前更准确的结论是：

- 有些 config 会传播
- callbacks 这件事目前仍有已知缺口

### 5. 为什么 callback manager 机制和 streaming 讨论要放在一起

因为 `messages` 流本质上就是建立在 callback handler 链上的。

也就是说，当你在问：

- “为什么子代理内部 token 能被外部看到？”
- “为什么 `nostream` 生效？”

你其实在问的是：

- 哪个 callback handler 被挂进去了
- 哪些 child model run 能走到这条 handler 链上
- 哪些 tags 在 handler 注册阶段被判定为应当忽略

所以 callback manager 是 streaming 可见性的机制基础，而不是独立话题。

---

## 如何让 `CompiledSubAgent` 按自己的规则运行

这是本章最实际的问题。

### 场景 1：你不想让主 agent 参与 compiled subagent 内部审批

默认就已经是这样。`CompiledSubAgent` 不会自动继承父级 `interrupt_on`。

### 场景 2：你想让 compiled subagent 有自己的审批规则

做法不是改父级 `interrupt_on`，而是：

- 在 compiled runnable 自己内部用 `create_agent(..., middleware=[...])` 或等价 graph wiring 配置 HITL
- 把审批逻辑作为 compiled runnable 的一部分

### 场景 3：你连 delegation 这一步也不想被父图暂停

那就不要在父级 `interrupt_on` 里拦 `task`。

### 场景 4：你不想让流式调用方看到子图内部节点

那就不要用 `subgraphs=True` 消费 `stream()`。这是观测面的控制，不是 middleware 继承面的控制。

### 场景 5：你想让子代理内部一部分 token 可见，另一部分不可见

当前没有发现 Deep Agents 内建的精细策略开关。实际可行的做法通常是：

- 对私有 LLM 调用打 `tags=["nostream"]`
- 对公开阶段保留正常 `messages` streaming
- 或者干脆关闭 raw token 暴露，只用 `custom` events 向外发你允许看到的阶段信号

这类需求最好在 compiled runnable 自己内部设计，而不是试图在父级 `create_deep_agent()` 上找一个全局开关。

---

## 三种 subagent 形态的边界矩阵

| 维度 | `SubAgent` | `CompiledSubAgent` | `AsyncSubAgent` |
|------|------------|--------------------|-----------------|
| 默认 middleware 栈是否自动补齐 | 是 | 否，use-as-is | 否，本地不构子图 |
| 是否自动挂到 `task` | 是 | 是 | 否，走 async task 工具集 |
| 是否继承父 `interrupt_on` | 是，默认继承，可覆盖 | 否 | 否 |
| 是否继承父 `skills` | 否；只有 general-purpose 默认继承，普通 custom `SubAgent` 需自己填 `skills` | 否，除非 runnable 自己带 | 否，远端自己决定 |
| 是否继承父 `permissions` | 是，可被子 spec 替换 | 否，除非 runnable 自己带 | 否，远端自己决定 |
| 是否继承父 `context` | 会进入子代理运行环境 | 会，现有测试覆盖 ToolRuntime.context | 取决于远端协议 |
| 是否传播部分 config / tags | 是 | 是，现有测试覆盖 tags / recursion_limit 合并 | 取决于远端协议 |
| 父 callbacks 是否可靠传播 | 当前未见明确保证 | 当前测试表明不会可靠传播，且为已知限制 | 取决于远端协议 |
| `subgraphs=True` 是否可观测内部节点 | 是 | 是 | 否，本地只能观测 async task 生命周期 |
| 返回给主 agent 的结果 | 过滤后的 state + `ToolMessage` | 同左 | async task handle / status，再由后续查询获取结果 |

---

## 结合代码看的两个典型案例

### 案例 1：为什么 declarative `SubAgent` 会被父级审批规则影响，而 compiled 不会

从代码路径看：

- declarative `SubAgent`
  - `graph.py` 先把 `interrupt_on` 合并进 spec
  - `_get_subagents()` 再 `create_agent(...)`，并追加 `HumanInTheLoopMiddleware`
- `CompiledSubAgent`
  - `graph.py` 和 `_get_subagents()` 都只把 `runnable` 原样透传
  - 不会补任何 HITL middleware

所以两者差异不是“行为偶然不同”，而是装配路径根本不同。

### 案例 2：内部 LLM 调用会不会被主 agent 拦截

如果你说的“拦截”是指：

- 父级 `interrupt_on`
- 父级 middleware

答案是：

- 对 declarative `SubAgent`，子代理内部工具调用可能被拦，因为 child graph 自己被构造成带 HITL 的 agent
- 对 `CompiledSubAgent`，父级不会自动拦截内部 LLM 或工具节点

如果你说的“拦截”其实是：

- 流式调用方在 `subgraphs=True` 下能看见子图事件

那是观测，不是拦截。

### 案例 3：为什么 `config={"tags": ["nostream"]}` 能隐藏子代理内部某次 token 流

这里真正发生的不是 Deep Agents 特判，而是：

- 该 tag 顺着 runnable config 传播到内部 chat model 调用
- `StreamMessagesHandler.on_chat_model_start()` 看到 tags 里有 `nostream`
- 这次 run 就不会被登记进 `messages` 流

因此你应该把它理解成：

- 一种 LangGraph 级别的 messages-stream suppression 机制
- 而不是 Deep Agents 特有的 subagent privacy API

### 案例 4：为什么“看不到 token”仍不代表 parent 不知道发生了什么

假设 compiled subagent 有两个 node：

- `plan_private`：内部规划，不希望对外暴露 token
- `answer_public`：最终回答，希望允许对外 streaming

如果你只做了：

- 在 `plan_private` 的模型调用上加 `tags=["nostream"]`

那么你得到的是：

- 外部看不到 `plan_private` 的 token
- 但如果 `plan_private` 把草稿写进 child state，并且最后又被保留在返回结果里，parent 仍然可能通过 state update 或最终消息间接看到它

所以真正安全的做法必须同时满足：

- 不把私有 token 流暴露给 `messages`
- 不把私有中间结果写回 parent 可见 state

---

## 现有测试如何证明这些边界

如果你想把本章结论逐条对回代码，优先看 [`test_subagents.py`](../../deepagents/libs/deepagents/tests/unit_tests/test_subagents.py) 里的这些测试：

- `test_subagent_inherits_interrupt_on_from_parent_agent`
  - 证明 declarative `SubAgent` 可以继承父 `interrupt_on`
- `test_subagent_interrupt_on_override_disables_parent_interrupt`
  - 证明 declarative `SubAgent` 可以覆盖父 HITL 设置
- `test_subagent_propagates_callbacks_to_model_calls`
  - 当前是 xfail，明确记录“父 callbacks 不会可靠传播到 subagent model calls”
- `test_subagent_streaming_emits_messages_and_updates_from_subgraph`
  - 证明 `subgraphs=True` 时外部流消费者可以看见子图 message 与 updates
- `test_config_passed_to_runnable_lambda_subagent`
  - 证明 runnable config 中的 tags 能传到 compiled subagent 调用面
- `test_context_passed_to_subagent_tool_runtime`
  - 证明 context 能进入子代理工具运行时
- `test_subagent_propagates_recursion_limit_to_tool_runtime`
  - 证明部分 config 和 tags 会传播并与子代理自身 config 合并

如果你想看 compiled 与 declarative 都经过 `task` 工具入口这一点，再看 [`test_subagent_middleware.py`](../../deepagents/libs/deepagents/tests/integration_tests/test_subagent_middleware.py) 中对自定义 runnable 和普通声明式 subagent 的对照。

---

## 为什么 subagent 是 Deep Agents 的核心，而不是附加功能

因为它解决的是上下文窗口的结构性问题：

- 大任务可以被切开
- 不同任务的工作记忆彼此隔离
- parent 只需要保留精炼结果

如果没有 subagent，Deep Agents 的 planning 和 filesystem 仍然有价值，但“深度任务分治”能力会显著下降。

---

## 容易踩什么坑

- 坑 1：把 subagent 当成普通 helper function。
  实际上它是通过 `task` tool 接入主 agent 决策回路的一等协作者。

- 坑 2：把“主 agent 能看到子图事件”误解成“主 middleware 正在包裹子图内部节点”。
  在 compiled 场景下，这两件事尤其要严格区分。

- 坑 3：默认让所有 parent state 透传给 child。
  这通常会制造 prompt layering 泄漏、state reducer 混乱和上下文膨胀。

- 坑 4：希望通过父级 `interrupt_on` 去控制 compiled runnable 内部审批。
  当前代码路径下，这种继承不会自动发生，应该把审批逻辑定义在 compiled runnable 自己内部。

- 坑 5：把 async subagent 理解为“同步 subagent 加 async/await”。
  实际上它代表的是不同的生命周期模型和远端运行边界。

- 坑 6：把 `nostream` 当成 Deep Agents 提供的私有性开关。
  它其实来自上游 LangGraph，而且只能控制 `messages` 流，不自动控制 state 回传。

- 坑 7：把 callback 传播和 middleware 继承混为一谈。
  前者是运行时 config/callback tree 的问题，后者是子图构建时有没有把某层逻辑装进去的问题。

---

## 本章小结

- Deep Agents 通过 `task` 工具把 delegation 统一纳入 tool-calling 语义。
- `CompiledSubAgent` 的关键特征是 use-as-is：主装配根负责挂接，不负责改写其内部图。
- “能看到子图事件”和“能拦截子图内部节点”是两回事；在 compiled 场景里尤其如此。
- `nostream` 来自 LangGraph，作用于 `messages` 流处理器，而不是 Deep Agents 私有 API。
- callback manager / config 传播解释了为什么部分 tags、context、config 能进入子代理运行时，但父 callbacks 到子代理 model call 仍有已知缺口。
- general-purpose subagent 是默认 harness 的组成部分，它确保隔离式委派始终可用。
