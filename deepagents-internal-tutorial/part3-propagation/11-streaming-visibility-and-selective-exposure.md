# 第11章：Streaming、Visibility 与 Selective Exposure

## 本章回答什么

- 为什么 streaming/visibility 应该继续放在传播层里理解，而不是退化成“某个框架 API 怎么调”的用法说明
- `stream()` / `astream()`、`stream_mode`、`nostream`、consumer visibility、实际执行事实分别属于哪条线
- 为什么“外部流消费者看不到”不能被写成“主 agent 不知道”“系统没执行”“结果不会折返”
- 维护者要做 selective exposure 时，应该改流输出线、事件面还是结果折返面，而不是混改

## 在整套系统中的位置

- 横向主题：`Propagation`、`Visibility`、`Observation`
- 前置章节：[第9章：传播层总览与四条线](./09-propagation-overview-and-four-lanes.md)、[第10章：Callbacks、Config 与 Callback Manager](./10-callbacks-config-and-callback-manager.md)、[第7章：Subagents、任务交接与上下文隔离](../part2-core-runtime/07-subagents-and-context-isolation.md)
- 后续章节：[第12章：Subagent 传播矩阵与维护者 recipes](./12-subagent-propagation-matrix-and-maintainer-recipes.md)、[附录 D：传播与可见性速查表](../appendix/propagation-and-visibility-cheatsheet.md)

这一章故意不把 streaming 写成 Deep Agents 专属特性说明。它讨论的是更上层的传播问题：执行已经发生之后，哪些事件会进入观测线，哪些又会进一步暴露到流输出线，最后还有哪些结果会沿结果折返线回到 parent。只有先把这三件事拆开，`nostream` 与 selective visibility 才不会被误写成“禁止执行”。

## 静态结构

建议同时打开这些文件：

- `langgraph/libs/langgraph/langgraph/pregel/main.py`
- `langgraph/libs/langgraph/langgraph/pregel/_messages.py`
- `langchain/libs/core/langchain_core/language_models/chat_models.py`
- `langchain/libs/core/langchain_core/tools/base.py`
- `deepagents/libs/deepagents/deepagents/middleware/subagents.py`

静态上至少要先区分四个部件：

| 部件 | 首先属于哪条线 | 维护者最容易误解成什么 |
| --- | --- | --- |
| `stream()` / `astream()` 调用入口 | 流输出线 | “只要能 stream，就等于内部执行与结果传播都连上了” |
| `stream_mode` | 流输出线 | “切换 mode 就等于改了内部执行 contract” |
| `nostream` | 流输出线上的局部抑制点 | “关闭 streaming = 关闭模型调用或 callback” |
| `ToolMessage` / state update / summary | 结果折返线 | “最终能折返，所以中间一定对 consumer 可见” |

因此本章的主判断是：

- streaming 面回答的是“外层 consumer 此刻能看到什么”
- execution fact 回答的是“内部节点 / tool / model call 到底有没有发生”
- 结果折返面回答的是“运行结束后 parent 拿到了什么”
- 这三件事彼此相关，但绝不是同一层 contract

## 运行时链路

### `stream()` / `astream()` 为什么是 Pregel 暴露面，而不是另一套执行引擎

- `stream()` / `astream()` 只是把 Pregel 的运行过程暴露成一条可消费外流。
- 它们不重新定义 step、task、runner、checkpoint 或 result return。
- 真正的执行背景仍然回第4章和第5章。

### `messages` / `updates` / `custom` 分别挂在 Pregel 的哪一层

- `messages`：挂在 callback/message observer 看到的 message-token 级可见面。
- `updates`：挂在 Pregel state update 的增量可见面。
- `custom`：挂在 node / tool 主动通过 writer 发出的 side-channel 可见面。

### 1. `stream()` / `astream()` 与 `stream_mode`

`stream()` / `astream()` 只是把 Pregel 的运行过程暴露成一条可消费外流。它们不重新定义内部执行，也不单独创造结果折返机制。

更稳妥的读法是：

1. 执行线先决定 node / tool / model call 是否真的运行。
2. 观测线决定这些运行是否进入 callback / event path。
3. 流输出线再由 `stream()` / `astream()` 配合 `stream_mode` 决定外部 consumer 能消费哪一类实时事件。
4. 结果折返线最后决定 parent state、`ToolMessage`、summary 会留下什么。

因此：

- `stream()` / `astream()` 是 Pregel runtime 的暴露面，不是另一套执行引擎。
- `stream_mode` 是“选择暴露哪种事件切片”，不是“改变内部调用真实发生了什么”。
- 同一轮执行里，`messages` 可见、`updates` 可见、最终结果可折返，可能重合，也可能只发生其中一部分。

### 2. `messages` / `updates` / `custom` / `values` 分别暴露什么

四个常见 mode 最好直接对照四线模型理解：

| mode | 默认暴露什么 | 更接近哪条线 | 不该被误写成什么 |
| --- | --- | --- | --- |
| `messages` | message chunk、token、模型/工具消息事件 | 流输出线中的 token/message 面 | “完整 parent tracing”或“完整 child state” |
| `updates` | 节点完成后的增量 update、含 `messages` 的状态更新 | 流输出线中的 state 增量面 | “最终唯一结果面” |
| `custom` | 通过 `stream_writer` 主动发出的阶段事件 | 流输出线中的自定义暴露面 | “必须等同于原始 token” |
| `values` | 某一步或最终聚合出来的 state 视图 | 结果折返线附近的快照面 | “实时内部执行细节全量回放” |

维护者要特别记住：

- `messages` 最适合做 token/message 级可见性，但它看到的是事件，不是全部执行事实。
- `updates` 更像“某个节点已经提交了哪些增量状态”，它比 `messages` 更靠近 state，但仍不等于最终对 parent 的全部折返。
- `custom` 最适合 selective exposure，尤其适合公开阶段信号而不公开 raw token。
- `values` 更接近快照读取，不该被用来证明中间每一跳都对外公开。

### 3. `nostream` 作用在哪个处理环节

`nostream` 应该被写成流输出线上的抑制开关，而不是执行线上的禁用器。

最安全的说法是：

- `nostream` 作用在 token/message 暴露给外部流消费者之前的那一段处理环节。
- 它讨论的是“这次模型输出要不要继续进入 `messages` 这类 consumer-facing stream surface”。
- 它不自动否定内部 model run 的发生。
- 它也不直接取消 `updates`、`custom`、`values`、最终 `ToolMessage` 折返。

所以一个很典型但经常被写错的场景是：

- 子图内部确实执行了模型调用。
- callback/event path 里系统仍然可能知道这次调用发生了。
- 但外部 `messages` consumer 因为 `nostream` 看不到 raw token。
- 最终 parent 仍然可能拿到压缩过的 `ToolMessage` 或 state update。

## 传播 / 可见性 / 拦截点

### 1. 消费者不可见，不等于系统不可知

这是 streaming 章节里最容易写歪的一句。

“外部 consumer 不可见”通常只说明：

- 当前流输出线没有把那类事件暴露到你正在消费的 surface

它不能单独推出：

- 内部模型没有执行
- callback manager 没接到事件
- parent 没拿到最终结果
- 主 agent 对这段运行“一无所知”

更准确的写法是：

- consumer visibility 是面向某个观察者的暴露事实
- execution fact 是系统内部是否实际运行的事实
- observation fact 是 callback / run tree 是否接住了这次运行
- return fact 是最终有没有结果折返

这四个判断来自四条线，不能互相偷换。

### 2. selective visibility 的分层控制面

真正做 selective visibility 时，通常有三层控制面：

1. `messages` 暴露面：决定是否让外部消费者看到 token / message chunk。
2. `custom` / `updates` 事件面：决定是否公开阶段事件、节点完成信号、摘要化进度。
3. 结果折返面：决定 parent 最终只拿摘要、结构化结果还是完整消息。

这三层各自适合解决不同问题：

- 想隐藏 raw token：优先处理 `messages` 暴露面，必要时配合 `nostream`。
- 想保留进度感知：优先发 `custom` 事件，或者用公开节点经 `updates` 暴露阶段完成。
- 想缩窄 parent 最终拿到的内容：改 `ToolMessage` / state update / structured result 的返回面。

最差的做法是跨线修错层：

- 你想隐藏 token，却去改最终 `ToolMessage`。
- 你想改 parent 最终拿到什么，却只去调 `stream_mode`。
- 你想说明 consumer 看不到，就把文档写成“主 agent 不知道”。

### 3. 私有推理 + 公共事件

这是 propagation 视角下最实用的 selective exposure 组合：

- 私有推理留在系统内部运行，不把 raw token 暴露给外部 `messages` consumer。
- 公开面只发阶段事件、结果摘要或最终回答。
- 外部用户仍然能感知任务在推进，但看不到不该公开的细粒度内部生成过程。

它与四线模型的对应关系很直接：

- 执行线：内部推理照常发生。
- 观测线：必要时系统内部 tracing 仍可存在。
- 流输出线：只暴露 `custom` / 公开节点 / 允许暴露的 `updates`。
- 结果折返线：最后折返一个压缩后的公开结果。

### 一个最实用的 recipe：私有推理 + 公共事件
- 对私有 LLM 调用使用不向 `messages` 暴露的通道。
- 对允许暴露的阶段使用 `custom` 事件或公开回答节点。
- 如果只是不想把子图细节暴露给外部流消费者，不要把它写成“主 agent 完全不知道”。

## 扩展接口

想扩展 streaming/visibility，先确认自己改的是哪条线：

- 想新增对外阶段事件：优先看 `custom` 与 `stream_writer`。
- 想把节点完成暴露给调用方：优先看 `updates` 面而不是 raw token 面。
- 想改默认流消费者能看到的 token：优先看 `messages` 暴露与 `nostream` 边界。
- 想改 parent 最终只拿什么结果：优先看 `ToolMessage`、state reducer、summary contract。

与 [第10章：Callbacks、Config 与 Callback Manager](./10-callbacks-config-and-callback-manager.md) 的关系也要写清：

- callback/config 决定观测线怎么连
- streaming/visibility 决定已发生事件怎样向外暴露
- 观测线连着，通常有利于可见性成立，但“能追踪”与“对流消费者公开”仍不是一回事

## 常见问题与排障入口

- “我没在 `messages` 里看到 token，所以内部没执行”：先查执行线，再查是否有 `nostream` 或是否根本没开 `messages` mode。
- “我在 `updates` 里看到了完成事件，所以 parent 一定拿到全部内部状态”：不成立；`updates` 是流输出线上的增量面，不是完整结果折返承诺。
- “我只想让用户看到阶段提示，不想看到 raw token”：优先用 `custom` 事件或公开节点，不要先改 subagent return contract。
- “我看不到子图细节，所以主 agent 一定不知道”：先把 consumer visibility 和 system observability 分开，再回看 [第10章：Callbacks、Config 与 Callback Manager](./10-callbacks-config-and-callback-manager.md)。
- “我最终收到了 `ToolMessage`，那中间 token 应该也能拿到”：不成立；结果折返线与流输出线本来就是两条不同线。

## 本章结论

- 谁提供：streaming/visibility 不是单一框架故事，而是 `LangGraph` 的 stream surface、`LangChain` 的 callback/event 路径，以及 `Deep Agents` 的返回面共同组成的传播问题。
- 如何传播：先确认执行是否发生，再确认事件是否进入观测线，最后才判断它是否通过 `messages`、`updates`、`custom`、`values` 被对外暴露。
- 修在哪层：隐藏 token、公开阶段事件、限制最终折返分别属于不同控制面；不要再把 `nostream`、consumer visibility、执行事实写成同一件事。
