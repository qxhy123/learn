# 附录 D：传播与可见性速查表

这一页只回答维护时最容易混掉的三件事：

- 某次内部调用到底沿哪条线传播
- 哪些内容会被外层流消费者看见
- 哪些内容只是执行了，但不会自动暴露

## 四条线速查

在继续看 selective visibility 之前，先分清回跳入口：

- 执行模型先回第4章。
- 执行路径先回第5章。
- callback/config 判断回第10章。
- 流可见性 / selective exposure 先回 [第11章：Streaming、Visibility 与 Selective Exposure](../part3-propagation/11-streaming-visibility-and-selective-exposure.md)，再用本页速查。

- 执行线
  先回答“它到底有没有跑起来”。第一站看 `BaseTool.run()`、`BaseChatModel.invoke()/stream()`、LangGraph node / subgraph step。
- 观测线
  先回答“callback tree / tracing 有没有接住它”。第一站看 `ensure_config()`、`patch_config()`、`CallbackManager.configure()`、`get_child()`。
- 流输出线
  先回答“外层 consumer 在运行中能看到什么”。第一站看 `stream_mode="messages" / "updates" / "custom"`、`StreamMessagesHandler`、`stream_writer`。
- 结果折返线
  先回答“运行完以后什么真正回到 parent / state / summary”。第一站看 `ToolMessage`、`Command(update=...)`、subagent return filtering、state reducer。

最容易错的是把这四条线拍成一句“传下去了”。维护时必须继续追问：

- 它是执行了，还是只是被观测到了
- 它是实时流出来了，还是只在结束后折返
- 它是 token 级可见，还是只有最终结果可见

## `nostream` 是什么，不是什么

`nostream` 不是 Deep Agents 代码里定义的 tag。

- 它来自 `langgraph/libs/langgraph/langgraph/constants.py`
- 公开常量名是 `TAG_NOSTREAM`
- `StreamMessagesHandler` 会检查这个 tag，决定当前模型 run 是否进入 `messages` 流

它是什么：

- 一个控制 `messages` token / message 可见性的 LangGraph 机制
- 一个对流输出线生效的过滤点

它不是什么：

- 不是 Deep Agents 自己发明的私有协议
- 不是阻止执行线继续运行的开关
- 不是阻止结果折返线回到 parent / state 的总闸门
- 不是 `custom` 事件或远端 async subagent 日志的统一屏蔽器

最短结论：

> `nostream` 控制的是 `messages` 可见性，不是整套系统的“主代理拦截开关”。

## selective visibility 调哪里

如果你想控制“哪些 token / 事件对流消费者可见，哪些不可见”，优先改的是流输出线，而不是假设主 agent 会统一拦截。

最常见的控制面如下：

| 目标 | 主要调点 | 作用在哪条线 | 结果 |
| --- | --- | --- | --- |
| 让内部 token 对外可见 | 外层订阅 `stream_mode="messages"`，需要时启用 `subgraphs=True`，内部调用不要打 `nostream` | 流输出线 | 外层 consumer 可见 token / message |
| 隐藏内部 token，但保留最终结果 | 内部模型调用使用 `tags=["nostream"]` 或等价配置 | 流输出线 | 外层 `messages` 看不到 token，但 `ToolMessage` / state update 仍可能回来 |
| 隐藏 token，只暴露阶段信号 | 内部模型调用用 `nostream`，同时在 node / tool 中用 `runtime.stream_writer(...)` 或 `ToolRuntime.stream_writer(...)` 发 `custom` 事件 | 流输出线 + 结果折返线 | 外层只看到你主动发出的阶段事件 |
| 尽量只保留最终聚合结果 | 不订阅 `messages`，必要时关闭 `subgraphs=True`，不要把内部 transcript 写回 parent-visible state | 流输出线 + 结果折返线 | 只保留较粗粒度结果面 |

### 一个最实用的 recipe：私有推理 + 公共事件

- 对私有 LLM 调用使用不向 `messages` 暴露的通道。
- 对允许暴露的阶段使用 `custom` 事件或公开回答节点。
- 如果只是不想把子图细节暴露给外部流消费者，不要把它写成“主 agent 完全不知道”。

```python
from langchain.tools import ToolRuntime


def run_private_step(runtime: ToolRuntime) -> str:
    runtime.stream_writer({"phase": "research_started"})

    # 这段 token 不希望被外层 messages consumer 看到
    result = model.with_config(tags=["nostream"]).invoke("do the internal work")

    runtime.stream_writer({"phase": "research_finished"})
    return result.content
```

## compiled subagent 内部调用该怎么判断

问题不要表述成“主 agent 会不会先拦截到它”。更准确的判断顺序是：

1. 先看执行线：这次 LLM 调用是不是 compiled subagent 自己内部 node / runnable 的真实模型调用。
2. 再看观测线：这次调用是不是还处在同一套 callback/config 传播链上，`ensure_config()` 能不能吃到 ambient child config。
3. 再看流输出线：外层是不是订阅了 `messages`，LangGraph 的 message stream observer 会不会观察到这次 model callback。
4. 最后看结果折返线：即使 token 不可见，child 的最终结果是否仍会经 `ToolMessage`、state update 或 summary 回到父级。

对 compiled subagent，最常见的真实链路是：

1. 父 agent 通过 `task` tool 发起 handoff。
2. `BaseTool.run()` 在工具执行前构造 child callback/config 语境。
3. compiled subagent 内部 runnable 继续运行，如果内部 node 又调用了 LangChain chat model，就会产生 model callback 事件。
4. LangGraph 的 `StreamMessagesHandler` 观察这些事件；如果没有被 `TAG_NOSTREAM` 过滤，外层 `messages` consumer 就可能看到 token。

所以“看到了 token”更接近下面这条链：

- `Deep Agents task tool -> LangChain callback / config 传播 -> LangGraph stream observer`

而不是：

- `主 agent 拿到全部 token -> 主 agent 再决定转不转发`

如果你想阻止外层看到 compiled subagent 内部 token，通常优先做这些事：

- 在 compiled subagent 内部那次模型调用上打 `nostream`
- 改成只发 `custom` 阶段信号，不直接暴露 token
- 调整外层订阅的 `stream_mode` / `subgraphs=True`
- 在 compiled subagent 自己的 runnable 里做更细粒度的 selective visibility 控制

## 一页判断题

### “我没看到 token，所以 parent 一定不知道这次内部调用存在过”

不成立。

- 你可能没开 `messages`
- 也可能打了 `nostream`
- 也可能只是没开 `subgraphs=True`

但结果折返线仍可能把 `ToolMessage`、state update、summary 写回父级。

### “我看到了 token，所以一定是 Deep Agents 主层拦截后再吐出来”

不成立。

优先怀疑：

- LangChain callback tree 仍然连着
- LangGraph 的 message stream observer 正在观察

### “我只要加 `nostream`，就把内部过程完全藏住了”

不成立。

`nostream` 只过滤 `messages`。它不自动屏蔽 state update、最终结果、`custom` 事件、远端 server 日志和安全边界。

### “我想做 selective visibility，所以应该先改 parent middleware”

通常不成立。

如果你真正要控制的是流消费者可见性，优先改内部模型调用、`stream_mode`、`stream_writer` 和 child runnable 的公开返回面。

## 本页小结

- 传播问题先拆成四条线：执行线、观测线、流输出线、结果折返线。
- `nostream` 是 LangGraph 的 `TAG_NOSTREAM`，不是 Deep Agents 私有 tag。
- compiled subagent 内部 token 是否外露，关键不在“主 agent 有没有统一拦截”，而在 callback/config 是否连着、外层订阅了什么流，以及内部调用有没有显式做 selective visibility 控制。
