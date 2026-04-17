# 附录 D：传播与可见性速查表

这一页专门解决一个常见混淆：

你在系统里看到的“有东西传下去了”，到底是：

- prompt text 传下去了
- `RunnableConfig` / callbacks 传下去了
- state update 传下去了
- token stream 被外层 consumer 看见了
- 还是 custom progress event 被主动发出来了

这五件事不是一回事。

---

## 先分清五条线

| 你看到的现象 | 主要 owner | 它怎么传播 | 第一站看哪里 | 不要误判成什么 |
|--------------|------------|------------|--------------|----------------|
| system prompt 里多了一段文字 | Deep Agents | middleware / profile / user prompt layering | `deepagents/graph.py`、`memory.py`、`skills.py` | callbacks/config 一定也跟着变了 |
| tags / metadata / callbacks 继续存在 | LangChain | `ensure_config()`、`patch_config()`、`CallbackManager.get_child()` | `langchain_core/runnables/config.py`、`langchain_core/callbacks/manager.py` | Deep Agents 自己维护了一套独立 parent-child 总线 |
| 子代理结果回到了父线程 | Deep Agents + LangGraph | `ToolMessage`、`Command(update=...)`、state reducer | `middleware/subagents.py`、`middleware/async_subagents.py` | token stream 一定对用户可见 |
| 外层 consumer 看到了 token | LangGraph + LangChain | model callback 被 `StreamMessagesHandler` 观察并转成 `messages` 流 | `langgraph/pregel/_messages.py`、`langchain_core/language_models/chat_models.py` | “主 agent 主动拦截并转发了每个 token” |
| 外层 consumer 看到了阶段事件 | LangGraph | `runtime.stream_writer(...)` / `ToolRuntime.stream_writer(...)` 主动发 `custom` 事件 | `langgraph/pregel/main.py` | 这些事件会自动进入 checkpoint 或 parent state |

---

## `nostream` 是什么

`nostream` 不是 Deep Agents 自己发明的 tag。

它来自：

- `langgraph/libs/langgraph/langgraph/constants.py`
- 常量名是 `TAG_NOSTREAM`

它控制的是：

- `messages` 流里的 message/token 可见性

它不自动控制的是：

- 子代理最终结果是否回父线程
- state update 是否继续发生
- `custom` 事件是否被发出
- 远端 async subagent server 自己的内部日志与安全策略

最短结论是：

> `nostream` 控制 observability，不控制全部执行后果。

---

## 我想控制“子代理内部 token 流哪些可见，哪些不可见”，该调什么

| 目标 | 建议设置 | 结果 | 注意事项 |
|------|----------|------|----------|
| 让子代理内部 token 对外层 consumer 可见 | 外层用 `stream_mode="messages"`，并在需要时启用 `subgraphs=True`；内部调用不要打 `nostream` | consumer 能看到 message/token 级事件 | 这是“可见性”，不是“父线程保存了完整 transcript” |
| 隐藏子代理内部 token，但保留最终结果 | 内部模型调用打 `tags=["nostream"]` 或等价配置；不要主动回放内部 transcript | 外层 `messages` consumer 看不到这段 token | 最终 `ToolMessage` / state update 仍可能回来 |
| 隐藏内部 token，但暴露脱敏阶段信号 | 内部模型调用用 `nostream`；再用 `runtime.stream_writer(...)` 或 `ToolRuntime.stream_writer(...)` 发 `custom` 事件 | 外层只看到你主动发出的阶段信号 | 记得 consumer 端必须打开 `stream_mode="custom"` |
| 尽量只暴露最终答案 | 不订阅 `messages`，避免发 `custom`，必要时关闭 `subgraphs=True`，并且不要把内部 transcript 写回 parent state | 外层只看到最终聚合结果或较粗粒度 update | 这不是严格保密边界；只是减少流可见性 |

---

## compiled subagent 内部 node 调用 LLM，会不会“被主 agent 拦截”

更准确的说法不是“主 agent 拦截”，而是：

- 如果这个内部 LLM 调用仍处在同一套 runnable / callback tree 之下
- 且外层有 `messages` stream observer 在观察
- 且这次调用没有被 `nostream` 之类的过滤掉

那么它的 token / message 事件就可能被 LangGraph 的 stream machinery 向外发出。

这不是“主 agent 先拿到 token 再决定要不要转发”的专门中间层。

更像是：

1. LangChain model 触发 callback 事件
2. LangGraph 的 message stream handler 观察到这些事件
3. 外层 consumer 订阅了对应 stream mode，于是看见它们

所以要阻止可见性，通常不是去“关掉主 agent 的拦截”，而是：

- 在子代理内部那次模型调用上打 `nostream`
- 或改成只发 `custom` 阶段信号
- 或减少外层订阅的 stream mode / namespace
- 或在 compiled subagent 自己的 runnable 内部做更细粒度控制

---

## 一个最实用的模式：私有推理 + 公共阶段信号

如果你想要的效果是：

- 子代理内部长推理或长搜索不要把 token 暴露给前端
- 但前端又要知道“现在进行到哪一步了”

推荐模式是：

1. 内部模型调用使用 `nostream`
2. 在 node 或 tool 里用 `stream_writer` 主动发阶段事件
3. 外层 consumer 只订阅 `custom` 或 `updates`

伪代码可以长这样：

```python
from langchain.tools import ToolRuntime


def run_private_step(runtime: ToolRuntime) -> str:
    runtime.stream_writer({"phase": "research_started"})

    # 这段 token 不希望被 messages consumer 看到
    result = model.with_config(tags=["nostream"]).invoke("do the internal work")

    runtime.stream_writer({"phase": "research_finished"})
    return result.content
```

如果你想看这个机制背后的源码，优先回看：

- 第 6 章关于 callback/config 传播
- 第 7 章关于 `nostream` 和 `custom` stream
- `langgraph/libs/langgraph/langgraph/pregel/main.py`
- `langgraph/libs/langgraph/langgraph/constants.py`

---

## 一页判断题

### “我没看到 token，所以 parent 一定不知道这次内部调用存在过”

不成立。

你可能只是：

- 没开 `messages` stream
- 打了 `nostream`
- 没开 `subgraphs=True`

但 parent 仍可能通过最终 `ToolMessage`、state update、summary 文件等方式知道结果。

### “我看到了 token，所以一定是 Deep Agents 主层拦截后再吐出来”

不成立。

优先怀疑：

- LangChain callback tree 仍然连着
- LangGraph 的 message stream observer 仍然在看

### “我想隐藏内部过程，只要加 `nostream` 就完了”

不成立。

`nostream` 只管 `messages` 可见性，不管 state、final result、remote server 自己的日志、安全策略。

---

## 本页小结

- 传播至少要分成 prompt、config/callback、state、messages、custom 五条线。
- `nostream` 是 LangGraph tag，不是 Deep Agents tag。
- 要控制 compiled subagent 内部 token 是否对流消费者可见，真正该改的是子代理内部调用方式和外层订阅方式，而不是假设“主 agent 有一个统一拦截开关”。
