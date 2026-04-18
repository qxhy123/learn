# 第10章：Callbacks、Config 与 Callback Manager

## 本章回答什么

- `ensure_config()`、`patch_config()`、`set_config_context()` 各自负责什么，为什么 callback/config 传播要先从这里看
- callback manager 怎样把一次 run 组织成 parent-child tree，而不是简单 handler 列表
- `BaseTool.run()` 与 `BaseChatModel.stream()` 为什么是传播层里最关键的两个接入点
- 为什么“看到了 token”不能直接推出“父 tracing 完整”或“主 agent 拦截了内部调用”

## 在整套系统中的位置

- 横向主题：`Propagation`、`Observation`
- 前置章节：[第9章：传播层总览与四条线](./09-propagation-overview-and-four-lanes.md)、[第5章：Tools 作为 Runtime Surface](../part2-core-runtime/05-tools-as-runtime-surface.md)、[第7章：Subagents、任务交接与上下文隔离](../part2-core-runtime/07-subagents-and-context-isolation.md)、[附录 D：传播与可见性速查表](../appendix/propagation-and-visibility-cheatsheet.md)
- 后续章节：后面的 streaming 可见性与 subagent propagation matrix 都默认复用本章的 callback/config 判断框架

这一章只处理证据最硬的那一段：`RunnableConfig`、callback manager、tool/model lifecycle 怎样把 run tree 组织起来。它不把 Deep Agents 的本地 middleware 继承规则说成上游 callback contract，也不把某次当前实现观测到的效果写成稳定承诺。

## 静态结构

建议同时打开这些文件：

- `langchain/libs/core/langchain_core/runnables/config.py`
- `langchain/libs/core/langchain_core/callbacks/manager.py`
- `langchain/libs/core/langchain_core/tools/base.py`
- `langchain/libs/core/langchain_core/language_models/chat_models.py`
- `deepagents/libs/deepagents/deepagents/middleware/subagents.py`

先立三条必须显式标注稳定性的判断：

- [`LC`][`Stable mechanism`] `ensure_config()` 先确定本次 run 的 config。
- [`LC`][`Known limitation`] 父 callbacks 到 compiled subagent 内部 model call 目前不能写成稳定继承能力。
- [`DA`][`Current implementation`] 本地 middleware 继承规则不等于 callback tree 传播规则。

这一章里，“证据来自哪层”也必须分开：

| 主题 | 首先属于哪层 | 本章怎么使用它 |
| --- | --- | --- |
| `ensure_config()` / `patch_config()` / callback manager | `LangChain` | 解释 run context 与 run tree 如何建立 |
| `task`、`SubAgent`、`CompiledSubAgent` | `Deep Agents` | 说明本地 handoff 与 middleware 继承边界，不把它冒充成上游 callback 语义 |
| `messages` stream observer | `LangGraph` | 说明 token 为什么可能可见，但不把它写成 parent 拦截 |

## 运行时链路

### 1. `ensure_config()` 先把 run context 定出来

从 `langchain_core/runnables/config.py` 看，`ensure_config()` 的顺序很明确：

1. 先构造一份带默认值的空 `RunnableConfig`，包括 `tags`、`metadata`、`callbacks`、`recursion_limit`、`configurable`。
2. 如果当前上下文里已有 `var_child_runnable_config`，先把这份 ambient config 合进来。
3. 再把显式传入的 `config` 合进来。
4. 非 `CONFIG_KEYS` 的额外键被放进 `configurable`。
5. 某些 `configurable` 键还会被镜像到 `metadata`。

这意味着两个维护结论：

- [`LC`][`Stable mechanism`] 先看 `ensure_config()`，再讨论“下游为什么继承到了这些 tags / metadata / callbacks”。
- callback/config 传播不是“谁手动把参数一层层传下去”，而是“ambient child config + explicit config”共同定出本次 run context。

这里的 `context` 也要说清：

- 本章说的首先是 runnable 的 ambient config context，也就是 `set_config_context()` 写入、`ensure_config()` 读取的那条线。
- 它不是 prompt text。
- 它也不等于 Deep Agents 的本地 middleware 继承。

### 2. callback manager 如何形成 parent-child run tree

`CallbackManager.configure()` 负责先把 callbacks、tags、metadata 合成当前 run 的 manager；真正形成树形关系的是 `run_manager.get_child()`。

从 `langchain_core/callbacks/manager.py` 看，`get_child()` 的证据点很直接：

- child manager 会带上 `parent_run_id=self.run_id`
- child manager 继承 `inheritable_handlers`
- child manager 继承 `inheritable_tags`
- child manager 继承 `inheritable_metadata`
- 额外 tag 会作为非继承 tag 加到 child 上

所以 callback tree 的最小事实是：

- 这是一棵显式 parent-child run tree
- 继承的是 handler / tags / metadata 这一类 callback 观测信息
- 不是把父 run 原样复用成同一个 run

这也是为什么 `patch_config(config, callbacks=run_manager.get_child())` 会顺手清掉 `run_name` / `run_id`：child run 需要自己的 run identity，但仍挂在 parent tree 下面。

### 3. `BaseTool.run()` 和 `BaseChatModel.stream()` 如何接入 callback tree

`BaseTool.run()` 和 `BaseChatModel.stream()` 分别代表了两种最常见的观测接入口。

#### `BaseTool.run()`

从 `langchain_core/tools/base.py` 看，tool run 的关键顺序是：

1. `CallbackManager.configure(...)`
2. `on_tool_start(...)`
3. `child_config = patch_config(config, callbacks=run_manager.get_child())`
4. `set_config_context(child_config)`
5. 再真正执行 `_run`

这里最关键的事实不是“tool 有 callbacks”，而是：

- tool run 自己先成为一条 callback tree 上的 run
- tool 内部如果再触发嵌套 runnable，它们可以通过 ambient config context 读到 child callbacks

注意不要把它写过头：

- `BaseTool.run()` 显式传给 `_run` 的 config 参数未必就是 `child_config`
- 但它确实把 `child_config` 放进了 ambient context，因此嵌套 runnable 后续通过 `ensure_config()` 仍可能接到 child callbacks

#### `BaseChatModel.stream()`

从 `langchain_core/language_models/chat_models.py` 看，streaming path 的关键顺序是：

1. `ensure_config(config)`
2. `CallbackManager.configure(...)`
3. `on_chat_model_start(...)`
4. 每个 chunk 上 `on_llm_new_token(...)`
5. 结束时 `on_llm_end(...)`

所以 token 级 observability 不是旁路魔法，而是模型 run 通过 callback manager 发出事件之后，才可能再被上层观察机制消费。

这也是为什么“我看到了 token”首先说明：

- 这次 model run 进入了 callback/event path
- 外层还有某个观察者在消费这些事件

但它仍然不能单独证明 parent tracing 全量完整，或证明 Deep Agents 主层主动拦截了每个 token。

## 传播 / 可见性 / 拦截点

### 1. 哪些值通常继续传播：tags / metadata / recursion_limit / context

按上游语义，这些值最常见的传播路径如下：

- `tags`：`ensure_config()` 合并 ambient + explicit config，`get_child()` 继续继承 inheritable tags。
- `metadata`：同样通过 config 合并与 child manager 继承继续存在。
- `recursion_limit`：属于 `RunnableConfig` 的稳定键；若没有显式覆写，child config 会继续带着它。
- `context`：这里主要指 `set_config_context()` 建起来的 ambient runnable context；嵌套 runnable 后续用 `ensure_config()` 时可以继续读到它。

这些值之所以“通常继续传播”，不是因为 Deep Agents 自己维护了一套独立总线，而是因为 `LangChain` 已经把 config merge、callback tree、child context 这条线定义好了。

### 2. 哪些值不能直接推断：父 callbacks 到 compiled subagent model calls

这里必须收紧表述。

对 `CompiledSubAgent`，最稳妥的说法只能是：

- parent 的 `task` handoff 自己当然会进入 callback tree
- compiled runnable 内部如果继续使用 LangChain runnable / model path，确实可能通过 ambient child config 吃到 parent 派生出的 child callbacks
- 但这依赖 compiled runnable 的内部实现、是否覆写 config、是否替换 callbacks、是否仍沿标准 runnable path 执行

因此本章不能把下面这句话写成稳定承诺：

- “父 callbacks 会稳定继承到 compiled subagent 的内部 model call”

更安全的写法是：

- [`LC`][`Known limitation`] 父 callbacks 到 compiled subagent 内部 model call 目前不能写成稳定继承能力。

这不是回避问题，而是证据边界本来就到这里。当前实现里它可能发生，但 maintainer 教程不能把“当前常见效果”写成“长期 contract”。

### 3. 哪些结论必须标成 `Known limitation`

以下结论如果要写进教程，都必须显式标成 `Known limitation` 或 `Current implementation`，不能冒充稳定机制：

- `CompiledSubAgent` 内部 model/tool 调用与父 callbacks 的连通程度。
- “看到了 token”与“父 tracing 一定完整”的对应关系。
- 本地 middleware 继承规则与 callback tree 传播规则之间的对应关系。

最容易被误写错的两个点是：

- [`LC`][`Known limitation`] 父 callbacks 到 compiled subagent 内部 model call 目前不能写成稳定继承能力。
- [`DA`][`Current implementation`] 本地 middleware 继承规则不等于 callback tree 传播规则。

## 扩展接口

如果你真的要改 callback/config 相关行为，优先改这些入口：

- 改 tags / metadata / callbacks 的进入方式：查 `RunnableConfig` 与 `CallbackManager.configure()`。
- 改 child run 的树形组织方式：查 `get_child()` 与 `patch_config()`。
- 想让 compiled subagent 与父 run 更明确隔离：在 compiled runnable 自己那层显式覆写 config / callbacks，不要只靠顶层叙述。
- 想补可见性而不是改 contract：回到 `LangGraph` 的 stream observer，而不是把 callback tree 描述写大。

## 常见问题与排障入口

### 为什么我看到了 token，但父 tracing 不完整

因为“token 可见”与“parent tracing 完整”属于不同证据层。

你看到 token，通常只能先说明：

- 内部 model run 触发了 `on_llm_new_token(...)`
- 外层还有 observer 在消费这条事件线

但 tracing 是否完整，还要继续查：

- 这次调用是否真的挂在你期待的 callback tree 下
- callbacks / tags / metadata 是否被中途替换
- compiled runnable 是否覆写了 config 或绕开了标准 runnable path
- 外层看到的是 `messages` 可见性，还是 LangSmith / tracer 里的完整 parent-child 结构

### 为什么 `CompiledSubAgent` 的内部调用不该直接写成“被主 agent 拦截”

因为主 agent 稳定包住的，是 `task` 这次 handoff；不是一个名为“主 agent 拦截器”的统一内部总线。

更准确的说法应该是：

- `task` 工具调用进入了 callback tree
- compiled subagent 内部如果继续走 LangChain runnable / model path，相关事件可能沿 callback tree 与 stream observer 向外暴露
- 这属于 callback/config 传播加上 LangGraph 观测机制的结果，不是“主 agent 先接住每个内部 token 再决定是否转发”

如果文档直接写成“被主 agent 拦截”，会把三件不同的事混成一句：

- parent handoff tool 是否被包住
- callback tree 是否连着
- `messages` stream 是否对外可见

## 本章结论

- 谁提供：`LangChain` 的 `RunnableConfig`、callback manager、tool/model lifecycle 提供了本章讨论的稳定机制；`Deep Agents` 只在本地 handoff 与 middleware 继承边界上补自己的规则。
- 如何传播：先由 `ensure_config()` 定出本次 run context，再由 `CallbackManager.configure()` 与 `get_child()` 组织 parent-child run tree，最后由 tool/model lifecycle 把事件接进 callback path。
- 修在哪层：callback/config merge、run tree、child manager 语义优先修 `LangChain` 层；compiled subagent 是否隔离、哪些 middleware 继承则优先修 `Deep Agents` 本地装配层。
