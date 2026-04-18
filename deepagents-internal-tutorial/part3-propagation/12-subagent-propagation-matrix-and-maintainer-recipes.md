# 第12章：Subagent 传播矩阵与维护者 recipes

## 本章回答什么

- `SubAgent`、`CompiledSubAgent`、`AsyncSubAgent` 在执行线、观测线、流输出线、结果折返线上到底怎么分化
- 哪些结论可以写成当前代码支持的稳定判断，哪些只能写成 `Known limitation` 或“不要过度承诺”
- 维护者在排障时应该先修哪一层，而不是把所有现象都归因成“subagent 机制有问题”

## 在整套系统中的位置

- 横向主题：`Propagation`、`Visibility`、`Observation`
- 前置章节：[第9章：传播层总览与四条线](./09-propagation-overview-and-four-lanes.md)、[第10章：Callbacks、Config 与 Callback Manager](./10-callbacks-config-and-callback-manager.md)、[第11章：Streaming、Visibility 与 Selective Exposure](./11-streaming-visibility-and-selective-exposure.md)
- 相关背景：[第7章：Subagents、任务交接与上下文隔离](../part2-core-runtime/07-subagents-and-context-isolation.md)、[附录 D：传播与可见性速查表](../appendix/propagation-and-visibility-cheatsheet.md)

这一章不是再讲一次“怎么创建 subagent”。它的目标是给维护者一张 propagation matrix：当你看到 `SubAgent`、`CompiledSubAgent`、`AsyncSubAgent` 的行为不一样时，到底差在执行归属、callback/config 连通、外部流可见性，还是结果折返方式。

## Subagent propagation matrix

| 维度 | `SubAgent` | `CompiledSubAgent` | `AsyncSubAgent` |
| --- | --- | --- | --- |
| 内部执行归谁管 | `SubAgentMiddleware` 在本地用 `create_agent()` 重建一个子 agent，内部执行主要仍走本地 `LangChain` / `LangGraph` runnable 路径。 | 由传入的 `runnable` 自己决定；middleware 只负责把它接到 `task` 工具里并调用 `invoke()` / `ainvoke()`。 | 由远端 Agent Protocol / LangGraph server 执行；本地只负责 `start` / `check` / `update` / `cancel`。 |
| 父 middleware / approval 是否继承 | 不能笼统写成“父层全继承”；当前代码是对子 agent 重新装配 middleware，并只按该 spec 明确追加本地 middleware / `interrupt_on`。 | 不应写成继承；`runnable` 是 use-as-is，本地父 middleware 不会自动重建进内部图。 | 不继承本地父 middleware；远端图有它自己的 middleware、approval、server 侧策略。 |
| 父 callbacks 是否可靠进入内部 model call | 通常更接近标准 runnable path，但仍应沿 [第10章](./10-callbacks-config-and-callback-manager.md) 的 callback/config 证据来判断，不要写成“只要是 SubAgent 就 100% 保证”。 | 不能写成稳定承诺；父 callbacks 进入内部 model call 取决于该 `runnable` 是否继续沿标准 path、是否覆写 config/callbacks。 | 本地父 callbacks 不应被写成能可靠进入远端内部 model call；跨进程后应视为另一套观测域。 |
| 外层 stream consumer 默认能看到什么 | 如果子 agent 内部继续走标准本地 graph/model path，外层在开启相应 stream mode 且使用 `subgraphs=True` 时，可能看到子图 `messages` / `updates`。 | 当前测试证据表明本地 compiled subagent 在 `subgraphs=True` 时可向外暴露子图 `messages` / `updates`，但这描述的是当前实现可见性，不是“主 agent 拦截一切内部细节”。 | 默认看不到远端内部 raw token；本地通常只看到 task 启动、后续查询结果，以及写回本地 state 的 `ToolMessage` / `async_tasks` 更新。 |
| 最终结果如何折返 | 子 agent 结束后，`task` 工具把最后消息或 `structured_response` 压缩成 `ToolMessage`，并把允许返回的状态键折返给 parent。 | 同样通过 `task` 工具从 `messages[-1]` 或 `structured_response` 折返；`messages` 键是硬要求，否则会报错。 | 先把远端 `thread_id` / `run_id` 持久化到 `async_tasks`，之后由 `check_async_task` 读取远端 `values.messages` 并折返为本地 `ToolMessage`。 |
| 应该优先修哪层 | 先查本地 subagent 装配、middleware 边界、callback/config 传播，再查 stream surface。 | 先查传入 `runnable` 自己的 config/callbacks/graph 设计，再判断 `task` 包装层有没有过滤或折返问题。 | 先查远端 graph contract 与 Agent Protocol 生命周期，再查本地 `async_tasks` 状态管理和结果收集工具。 |

## 如何读这张矩阵

这张表刻意延续 [第9章：传播层总览与四条线](./09-propagation-overview-and-four-lanes.md) 的四线读法。

维护者最常犯的三种混写是：

- 把“谁负责内部执行”写成“谁对外可见”。
- 把“父 middleware 是否继承”写成“父 callbacks 必然可靠进入内部 model call”。
- 把“最终能折返”写成“外层 streaming 默认就能看到全过程”。

而从当前代码证据看，三类 subagent 的核心差别正好落在这三处：

- `SubAgent` 是本地重建型，边界主要在 middleware 装配与本地 runnable 传播。
- `CompiledSubAgent` 是 use-as-is 型，边界主要在传入 runnable 自己是否继续沿标准 callback/config path。
- `AsyncSubAgent` 是远端任务型，边界主要在远端执行域与本地结果收集域之间。

## 维护者 recipes

### 场景 1：我想让子代理内部规划不对外吐 token

优先处理流输出线，不要先改执行线。

更稳妥的做法是：

- 对本地 `SubAgent` / `CompiledSubAgent`，先判断外层到底消费的是 `messages` 还是 `updates`。
- 对不应公开的内部推理，避免把它设计成必须暴露给 `messages` consumer 的 surface；必要时配合 [第11章：Streaming、Visibility 与 Selective Exposure](./11-streaming-visibility-and-selective-exposure.md) 里的 `nostream` / selective exposure 思路。
- 如果仍需要让调用方感知进度，用 `custom` 事件或公开节点阶段信号，而不是泄露 raw token。

这件事不要写成：

- “主 agent 不知道子代理内部规划”

更准确的是：

- 子代理内部规划仍然沿执行线发生
- 是否进入系统内部观测线要看 callback/config
- 是否对外吐 token 取决于流输出线的暴露策略

### 场景 2：我只想向调用方暴露阶段事件，不暴露 raw token

这是最标准的“私有推理 + 公共事件”场景。

推荐顺序：

1. 把 raw token 留在私有推理面，不把它当成必须公开的 `messages` surface。
2. 对允许暴露的阶段，用 `custom` 事件或公开节点完成信号向外发射。
3. 最终再通过 `ToolMessage`、结构化结果或压缩后的 state update 折返给 parent。

对 `AsyncSubAgent` 尤其要注意：

- 本地根本不该把远端内部 token 当成默认可见 contract
- 它更适合通过任务状态、结果查询、最终折返来公开必要信息

### 场景 3：我想判断“父 callbacks 没进来”是设计边界还是 bug

先按类型分流，再回到 [第10章：Callbacks、Config 与 Callback Manager](./10-callbacks-config-and-callback-manager.md)。

判断顺序建议是：

1. 如果是 `SubAgent`，先看本地子 agent 是否确实沿标准 runnable path 创建与调用。
2. 如果是 `CompiledSubAgent`，先检查传入 `runnable` 是否自己覆写 config、替换 callbacks、绕开标准 model/tool lifecycle。
3. 如果是 `AsyncSubAgent`，默认先当作设计边界看，因为远端内部调用本来就不在本地 callback tree 的稳定承诺内。

只有在“本该在同一执行域里、也沿标准 path 走、却异常断开”的前提下，才更像 bug。否则很可能只是边界不同，不应写成传播失效。

### 场景 4：我想判断结果折返与流可见性的区别

最简单的判断口诀是：

- 流可见性看运行过程中 consumer 能看到什么
- 结果折返看运行结束后 parent 最终拿到了什么

放到三类 subagent 上：

- `SubAgent` / `CompiledSubAgent` 可能在运行中暴露 `messages` / `updates`，但 parent 真正稳定消费的仍是最后折返出来的 `ToolMessage` 或允许返回的状态键。
- `AsyncSubAgent` 运行中通常只有任务启动与状态查询这类间接可见性；真正结果要等后续检查远端 thread values 再折返。

所以“我最后拿到了结果”与“我实时看到了内部生成”是两个问题，排障时必须分开答。

## 常见问题与排障入口

- “为什么 `CompiledSubAgent` 有时能在 stream 里看到内部 token，有时又不行”：先看该 runnable 是否仍沿标准本地 graph/model path，再看是不是只观察到了当前实现上的可见性，而不是稳定 contract。
- “为什么 `AsyncSubAgent` 不给我远端内部 token”：因为它的主要 contract 是远端任务生命周期与结果查询，不是把远端内部流原样透传到本地 consumer。
- “为什么 `SubAgent` 的 middleware 表现和 `CompiledSubAgent` 不一样”：因为前者是本地重建，后者是 use-as-is，维护者首先要查的层本来就不同。
- “为什么最终 `ToolMessage` 正常，但 tracing 或 streaming 看起来不完整”：这是观测线、流输出线、结果折返线被混写了；先回到 [第9章：传播层总览与四条线](./09-propagation-overview-and-four-lanes.md)。

## 本章结论

- 谁提供：三类 subagent 共享 `task` / 结果折返这一层表面，但内部执行域、可见性与 callback/config 连通度并不相同。
- 如何传播：`SubAgent` 倾向本地重建传播，`CompiledSubAgent` 倾向 use-as-is 传播，`AsyncSubAgent` 倾向远端执行 + 本地收结果传播。
- 修在哪层：先按 subagent 类型找真正的边界，再分别落到本地装配层、传入 runnable 层或远端 Agent Protocol 生命周期层，不要把所有现象都归咎为“subagent 黑盒”。 
