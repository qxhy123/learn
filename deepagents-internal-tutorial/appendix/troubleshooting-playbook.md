# 附录 E：Troubleshooting Playbook

这一页不是概念教程，而是排障手册。

用法很简单：

1. 先按症状定位到最像的条目
2. 先判断 owner layer
3. 再去对应源码、测试、example 做最小复现

---

## 症状 1：compiled subagent 内部 token 出现在外层 stream 里

### 最可能的 owner layer

- 首先看 `LangGraph`
- 然后看 `LangChain`
- 最后才看 `Deep Agents`

### 常见原因

- 外层 consumer 开了 `stream_mode="messages"`
- 还开了 `subgraphs=True`
- 子代理内部模型调用没有打 `nostream`
- 那次模型调用仍在同一棵 callback tree 之下

### 先看哪里

- `langgraph/libs/langgraph/langgraph/pregel/_messages.py`
- `langgraph/libs/langgraph/langgraph/pregel/main.py`
- `langchain/libs/core/langchain_core/language_models/chat_models.py`
- `langchain/libs/core/langchain_core/tools/base.py`
- `deepagents/libs/deepagents/deepagents/middleware/subagents.py`

### 优先修法

- 在子代理内部那次模型调用上打 `nostream`
- 只把脱敏后的阶段事件改成 `custom`
- 如无必要，不要对外开启 `subgraphs=True`

### 不要做的误修

- 不要先去找“主 agent 有没有统一 token 拦截开关”
- 不要把“看到了 token”直接归因成 Deep Agents 主层 bug

---

## 症状 2：我加了 `nostream`，但最终结果还是回到了父线程

### 最可能的 owner layer

- `Deep Agents` + `LangGraph`

### 常见原因

- `nostream` 只过滤 `messages` 流
- 子代理仍然通过 `ToolMessage` 或 `Command(update=...)` 把结果回传
- 最终状态仍然被写回 parent-visible state

### 先看哪里

- `langgraph/libs/langgraph/langgraph/constants.py`
- `deepagents/libs/deepagents/deepagents/middleware/subagents.py`
- `deepagents/libs/deepagents/deepagents/middleware/async_subagents.py`

### 优先修法

- 如果只是不要 token，可保留现状
- 如果连最终结果也不想直接回父线程，就要改 child runnable 的返回设计，而不是只改 tag
- 如果只想暴露阶段信号，用 `custom` 事件替代原始 transcript

---

## 症状 3：父 callbacks / tracing 没有完整进入子代理内部 model 调用

### 最可能的 owner layer

- 首先看 `LangChain`
- 同时留意 `Deep Agents` 当前已知缺口

### 常见原因

- `BaseTool.run()` 的 child callback tree 传播不是你想象的那样
- `RunnableConfig` patch 发生在别的层
- 当前 Deep Agents 对某些 subagent model call 的 callback 传播并未完全钉死

### 先看哪里

- `langchain/libs/core/langchain_core/runnables/config.py`
- `langchain/libs/core/langchain_core/callbacks/manager.py`
- `langchain/libs/core/langchain_core/tools/base.py`
- 第 6 章中的证据矩阵
- `deepagents/libs/deepagents/tests/unit_tests/test_subagents.py`
- `deepagents/libs/deepagents/tests/integration_tests/test_subagent_middleware.py`

### 排障原则

- 先确认你看到的是 tags、metadata、callbacks 里的哪一种
- 不要把“parent callbacks 一定全量透传”当既成事实
- 优先参考现有 `xfail` 和集成测试，而不是靠直觉写文档

---

## 症状 4：`custom` 事件没有出现在流里

### 最可能的 owner layer

- `LangGraph`

### 常见原因

- 外层没开 `stream_mode="custom"`
- 代码里根本没调用 `runtime.stream_writer(...)` 或 `ToolRuntime.stream_writer(...)`
- 你以为 `updates` 或 `messages` 会自动包含 `custom` 事件

### 先看哪里

- `langgraph/libs/langgraph/langgraph/pregel/main.py`
- 触发该事件的 node / tool 源码

### 优先修法

- 先确认 consumer 订阅了 `custom`
- 再确认发事件的代码路径真的执行到了
- 如果想持久化，别指望 `custom` 自动进入 checkpoint，要自己决定是否同时写 state / file

---

## 症状 5：permissions 没挡住 compiled subagent 或 remote async subagent

### 最可能的 owner layer

- `Deep Agents` 本地 policy 设计

### 常见原因

- Deep Agents 的 permissions 主要守默认文件工具面
- compiled subagent 自己的 graph / tool / HTTP / sandbox 调用不一定经过这层
- remote async subagent 在远端服务上运行，本地 policy 不会自动替你兜底

### 先看哪里

- 第 7 章的 permissions 边界分析
- `deepagents/libs/deepagents/deepagents/middleware/subagents.py`
- `deepagents/libs/deepagents/deepagents/middleware/async_subagents.py`
- 远端 server 或 child graph 自己的 tool / backend 实现

### 优先修法

- 如果是 compiled subagent，去 child runnable 内部加约束
- 如果是 remote async subagent，去远端 server / sandbox / tool 层加约束
- 不要把本地 permissions 当成全局安全壳

---

## 症状 6：memory / skills 在子代理里看起来泄漏了，或者没有按预期隔离

### 最可能的 owner layer

- `Deep Agents`

### 常见原因

- 你把 prompt inheritance、state inheritance、skills/memory reloading 混成了一件事
- child 读取的是自己的 memory / skills source，而不是父级 state
- `skills_metadata` / `memory_contents` 会被特殊过滤

### 先看哪里

- `deepagents/libs/deepagents/deepagents/middleware/subagents.py`
- `deepagents/libs/deepagents/deepagents/middleware/memory.py`
- `deepagents/libs/deepagents/deepagents/middleware/skills.py`
- 第 6 章关于 `_EXCLUDED_STATE_KEYS` 和过滤边界的说明

### 优先修法

- 先问自己：你要隔离的是 prompt、state，还是 callback/config
- 再确认 child 的 sources 是否就是你想要的文件源
- 不要把 example 的 loader 逻辑误写成框架默认逻辑

---

## 症状 7：远端 async task 状态看起来不对，或者 user 追问时答复过期

### 最可能的 owner layer

- `Deep Agents` async subagent workflow

### 常见原因

- 你在复述旧的 tool result，而不是重新查询 live status
- update 会在同一 thread 上重启 run，但 task_id 保持不变
- 你把“会话里的上一次状态”当成当前真相

### 先看哪里

- `deepagents/examples/async-subagent-server/supervisor.py`
- `deepagents/libs/deepagents/deepagents/middleware/async_subagents.py`
- `deepagents/examples/async-subagent-server/server.py`

### 优先修法

- 查单个任务时用 `check_async_task`
- 查多个任务时用 `list_async_tasks`
- 别在 agent prompt 里允许自己复述历史状态

---

## 症状 8：升级上游后，不知道 bug 该修在 Deep Agents、LangGraph 还是 LangChain

### 最小分层判断法

先问这三个问题：

1. 是 tool/model primitive 的 callback/config 语义变了？
   先看 `LangChain`。

2. 是 `stream_mode`、subgraph、checkpoint、runtime context 的行为变了？
   先看 `LangGraph`。

3. 是默认 middleware 栈、permissions、profile、backend adapter、subagent 装配规则变了？
   先看 `Deep Agents`。

### 升级前最小检查清单

- 对照第 2 章重新确认边界归属
- 对照第 6、7 章确认传播线和可见性线有没有混淆
- 对照第 10 章补最小 regression test
- 用附录 C 找一个最像的 example 做现实样本
- 能复现到上游 primitive / runtime 层，就不要先在本地 harness 打补丁

---

## 本页小结

- 先按症状选条目，再按 owner layer 追源码。
- callback/config、stream visibility、state return、permissions 不是一条线，别混修。
- 真要做边界 bug 修复，先把复现缩到最小，再决定是修 LangChain、LangGraph，还是 Deep Agents。
