# 第7章：Summarization、Streaming、Permissions 与安全边界

## 学习目标

学完本章，你应该能回答：

1. summarization、streaming visibility、permissions 分别属于哪一层
2. `values`、`updates`、`messages`、`custom`、`checkpoints`、`tasks`、`debug` 各自在控制什么
3. 为什么“流里看不到”不等于“父图不知道”
4. 为什么 permissions 不是“整套系统的安全模型”

---

## 问题是什么

维护者最容易把“安全边界”说得过于宽泛：

- 觉得 permissions 能控制所有危险行为
- 觉得 `nostream` 能隐藏所有内部信息
- 觉得 summarization 只是长上下文时删一点旧消息

实际上这三块分属不同层：

- summarization 是 Deep Agents 的 harness policy
- streaming visibility 主要是 LangGraph runtime 语义
- permissions 是 Deep Agents 对部分 tool surface 的本地收口策略

这章真正要解决的是：

> 哪些东西只是“外层消费者看不见”，哪些东西是真的没有进入 stream、没有进入 state、也没有回到 parent。

---

## 哪一层负责什么

### `LangChain`

- 模型 / tool callback 事件本身由上游 primitive 触发
- `BaseChatModel.stream()` / `astream()` 触发 token callback

### `LangGraph`

- `stream_mode="values" / "updates" / "messages" / "custom" / "checkpoints" / "tasks" / "debug"`
- `subgraphs=True`
- `StreamMessagesHandler`
- `TAG_NOSTREAM`
- `Runtime.stream_writer` / `ToolRuntime.stream_writer`

### `Deep Agents`

- summarization middleware 的使用时机与默认策略
- permissions / filesystem rules 的本地 policy
- 哪些 child state 能回到 parent
- 主线程与 subagent 的结果折返方式

---

## 代码在哪里

建议同时打开：

- `deepagents/libs/deepagents/deepagents/graph.py`
- `deepagents/libs/deepagents/deepagents/middleware/summarization.py`
- `deepagents/libs/deepagents/deepagents/middleware/permissions.py`
- `deepagents/libs/deepagents/deepagents/middleware/subagents.py`
- `deepagents/libs/deepagents/tests/unit_tests/test_subagents.py`
- `langgraph/libs/langgraph/langgraph/pregel/main.py`
- `langgraph/libs/langgraph/langgraph/pregel/_messages.py`
- `langgraph/libs/langgraph/langgraph/constants.py`

---

## 实现怎么工作

### 1. summarization 是本地 compaction 策略

Deep Agents 把 summarization middleware 放进默认栈，是为了控制：

- 长线程下的上下文膨胀
- 大型 task / subagent 工作流的 token 负担
- 长期运行时“主线程还保留多少原始上下文”

这不是 LangGraph 自带的“自动压缩 state”机制，而是 harness policy。

### 2. streaming visibility 主要由 LangGraph 决定

在 `Pregel.stream()` / `astream()` 里：

- `stream_mode` 决定流的形状
- `messages` 模式会挂上 `StreamMessagesHandler`
- `custom` 模式会构造 `stream_writer`
- `subgraphs=True` 决定是否把子图 namespace 里的事件继续向外发

因此，Deep Agents 只是在使用上游 stream 面，而不是自己实现 token dispatcher。

### 3. `nostream` 是 LangGraph tag，不是 Deep Agents tag

`TAG_NOSTREAM` 定义在 `langgraph.constants`，`StreamMessagesHandler.on_chat_model_start()` 会检查这个 tag。

所以：

- 它只能阻止该次模型调用进入 `messages` 流
- 它不会自动阻止 `updates`
- 它也不会自动阻止最终 `ToolMessage` / state update 回到 parent

### 4. `custom` 流适合“私有推理 + 公共阶段信号”

如果目标是：

- 不暴露内部 token
- 但又想让流消费者知道阶段进度

最稳妥的做法通常是：

- 私有模型调用打 `tags=["nostream"]`
- 不把规划草稿写回共享 state
- 用 `runtime.stream_writer(...)` 主动发脱敏后的 custom event

### 5. permissions 是 Deep Agents 的本地 tool policy

`_PermissionMiddleware` 必须最后，是因为它要看到所有已注入工具。

但它的边界也很明确：

- 它主要约束本地 tool surface
- 它不是 LangGraph / LangChain 的通用安全模型
- 它不会神奇地约束你在 compiled subagent 里自己实现的任意逻辑

---

## 一张 stream mode 总表

| `stream_mode` | 主要看什么 | 典型用途 | 容易误判 |
|---------------|------------|----------|----------|
| `values` | 当前 values 视图 | 观察线程最终/阶段性状态 | 以为它会展示每个中间 node 的细节 |
| `updates` | node 或 task 的 update | 看 parent / subgraph 每一步返回了什么 | 以为它天然包含 token |
| `messages` | chat model token / message 事件 | 做 token 级 UI、调试 LLM 行为 | 以为它等于“所有内部执行都可见” |
| `custom` | `stream_writer` 主动发出的 side-channel 数据 | 进度条、脱敏阶段信号、内部里程碑 | 以为它会自动进 checkpoint |
| `checkpoints` | checkpoint 创建事件 | 调试持久化与恢复点 | 以为它包含原始 callback 流 |
| `tasks` | task 开始/结束/错误 | 看调度层 | 以为它能替代 `updates` |
| `debug` | 尽量多的调试事件 | 深度排障 | 以为它是稳定 UI contract |

对维护者最重要的一点是：

- `messages` 偏 token / callback 可见性
- `updates` 偏 state / result update 可见性
- `custom` 偏产品化 side channel

这三条线不要混写。

---

## 一张“谁能看到什么”矩阵

| 面 | 流消费者 | parent graph | checkpoint/state | 最终用户 UI |
|----|----------|--------------|------------------|-------------|
| `messages` token | 取决于 `stream_mode="messages"` 与 tag 过滤 | 不一定直接作为 parent state 保存 | 默认不会以 token 流形式保存 | 取决于 UI 是否展示 |
| `updates` | 取决于 `stream_mode="updates"` | parent 可通过 `Command(update=...)` 或 tool result 感知 | 可能通过 state update 间接体现 | 取决于 UI 是否展示该 namespace |
| `custom` 事件 | 取决于 `stream_mode="custom"` | 不是 parent state 的自动组成部分 | 默认不会进 checkpoint | 常作为产品层进度信号 |
| 最终 `ToolMessage` | 可以从 parent `tools` update 看见 | parent 一定能收到 | 会体现在最终 state/message history | UI 通常会展示 |
| child 私有 reasoning | 默认不应作为显式 state 暴露 | 不一定可见 | 不应作为公共结果持久化 | 通常不可见 |

这张表要解决一个非常具体的误判：

> “我没在流里看到”不等于“parent 根本不知道”。

---

## 为什么“看不到”不等于“父图不知道”

### 流消费者不可见

表示：

- 某些 token 没进入 `messages` 流
- 或者 UI 过滤掉了某些 namespace / tag / node
- 或者你根本没开对应 stream mode

### parent 仍可能知道

因为 child 结束后仍可能通过：

- `Command(update=...)`
- `ToolMessage`
- 非排除 state key

把结果折返给 parent。

这正是可见性与 state bubbling 必须分开讨论的原因。

---

## `nostream` 真正能做什么

### 能做的

- 阻止某次 chat model run 进入 `messages` 流
- 让“内部思考 token”不对 token 级消费者可见

### 做不到的

- 不会自动阻止 `updates`
- 不会自动阻止 `ToolMessage`
- 不会自动阻止 child 最终结果回到 parent
- 不会自动阻止 UI 展示 parent 最终摘要

### 维护者该怎么写教程

正确表述是：

> `nostream` 控制的是 message-stream observability，不是整套执行可见性。

---

## permissions 的真实边界

### 1. 它主要守的是文件工具面

当前 `_PermissionMiddleware` 的主战场是：

- `ls`
- `read_file`
- `write_file`
- `edit_file`
- `glob`
- `grep`

以及这些工具的 artifact / path 过滤。

### 2. 它不是 compiled subagent 的万能外部保险

如果 `CompiledSubAgent` 内部自己构了别的图、别的工具、别的外部调用逻辑：

- 顶层 `_PermissionMiddleware` 不会自动深入那层内部实现
- 除非子代理内部自己也有对应的权限策略

### 3. 它也不是 remote async subagent 的远端安全模型

`AsyncSubAgent` 更像远端 thread/run 协议入口。

这意味着：

- 本地 parent 可以控制“是否发起这次 delegation”
- 但远端内部工具权限、审批逻辑、执行环境权限，仍要在远端 agent 自己定义

### 4. execute 能力还要再分一层

即使你有 permissions rule，也不等于默认就安全了，因为：

- backend 是否支持 execute，是 backend contract 的问题
- execute 在什么环境里跑，是 sandbox / local shell / remote runtime 的问题

所以“有 permissions”与“有安全执行环境”必须拆开说。

---

## 反例表：哪些场景 permissions 不会自动替你兜底

| 场景 | 为什么顶层 permissions 不够 |
|------|-----------------------------|
| compiled subagent 内部自定义工具 | 顶层只守 parent tool surface，不会自动深入 child 自定义逻辑 |
| remote async subagent | 远端内部安全边界由远端 agent/runtime 决定 |
| backend 提供 execute 能力 | 是否真在 sandbox 中执行，属于 backend contract，不是 permission rule 本身 |
| 非文件类外部副作用工具 | 若没有被纳入对应 policy middleware，permissions 不会自动覆盖 |

---

## 一个实用模式：私有推理 + 公共阶段信号

这是当前最适合 maintainer 推荐给上层应用的一种可见性设计：

1. 内部规划或长推理模型调用打 `tags=["nostream"]`
2. 不把原始中间推理写回共享 state
3. 用 `stream_writer` 推阶段信号，例如：
   - `planning_started`
   - `draft_ready`
   - `verification_started`
4. 最终只把必要结果通过 `ToolMessage` / `Command(update=...)` 回传

这个模式能比较稳地满足：

- 对最终消费者隐藏中间 token
- 又不让产品层完全失去进度感知

---

## 什么时候该修上游

### 更像上游问题

- `subgraphs=True` 行为与文档/测试不一致
- `messages` / `custom` / `updates` 事件形状异常
- `nostream` tag 没有按预期过滤 `messages`
- `stream_writer` 行为与你理解的 LangGraph contract 不一致

### 更像 Deep Agents 本地问题

- 默认 summarization 策略不合适
- permissions rule 设计不合理
- child result 过滤不合理，导致私有数据不该回传却回传了
- parent 与 child 的结果折返面定义不够清晰

---

## 容易踩什么坑

- 坑 1：把 `nostream` 描述成“Deep Agents 提供的隐藏机制”。

- 坑 2：把 permissions 描述成“整套系统的安全边界”。
  它只是本地 harness 的一部分。

- 坑 3：把“UI 看不到”描述成“执行层完全不可见”。

- 坑 4：只测 `messages` 流，不测 `updates` / `custom` / 最终 state。
  这样会持续误判“看不到”与“没有返回结果”。

---

## 本章小结

- summarization 和 permissions 是 Deep Agents 本地策略。
- streaming visibility 主要由 LangGraph 决定。
- `nostream` 只能影响 `messages` 流，不会自动阻止 state / result 回传。
- permissions 主要守的是本地工具面，不是 compiled / remote / execute 全部场景的总安全模型。
- 维护者必须把 token 可见性、state 回传、UI 展示、真实安全边界这四件事分开看。
