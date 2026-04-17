# 第12章：如何安全地新增一种跨三层能力

## 学习目标

学完本章，你应该能回答：

1. 新能力应该先落在哪一层
2. 为什么 visibility、interrupt、callback、state return 面必须先定义 contract
3. 什么时候只改 Deep Agents，什么时候必须修 LangGraph / LangChain
4. 如何在不污染 `graph.py` 的前提下把能力安全落地

---

## 问题是什么

维护这套栈时，最危险的不是“不会写功能”，而是把能力加在了错误的层。

典型症状包括：

- 明明是 streaming/runtime 问题，却硬往 Deep Agents 里找全局开关
- 明明是 provider 适配问题，却把逻辑塞进通用 middleware
- 明明是 compiled subagent 自己的内部规则，却试图靠父 `interrupt_on` 控制

所以新增能力之前，第一步永远不是写代码，而是做层次归属判断。

---

## 哪一层负责什么

### `LangChain`

- model/tool primitive
- `RunnableConfig` / callback manager
- agent middleware hook surface
- provider 模型集成

### `LangGraph`

- state graph、subgraph、checkpoint
- `messages` / `updates` / `custom` streaming
- `Runtime` / `ToolRuntime`
- graph 执行与 thread 级状态语义

### `Deep Agents`

- 默认 harness 装配
- backend/profile/permissions/subagent policy
- prompt layering、memory/skills conventions
- parent-child handoff 与结果折返规则

---

## 推荐工作流

### 1. 先做层次归属判断

先问自己这四个问题：

- 这是 model/tool/callback/config primitive 问题吗
- 这是 graph runtime / subgraph / streaming / checkpoint 问题吗
- 这是默认 harness policy 问题吗
- 这是单个业务场景才需要的工作流吗

一个简单判断表：

| 需求 | 更适合哪层 |
|------|------------|
| 新的 provider 默认参数或默认 tool exclusion | Deep Agents profile |
| 新的 tool 暴露面或默认 prompt policy | Deep Agents middleware / assembly |
| 新的存储或执行介质 | Deep Agents backend |
| token 可见性、`custom` 事件、subgraph streaming | LangGraph runtime/stream 配置 |
| callback tree / `RunnableConfig` merge 行为 | LangChain primitive |
| 单个场景专用 workflow | example / consumer 自己装配 |

### 2. 先定义 contract，再写实现

至少先写清楚五个面：

- 模型可见面：prompt / tools / descriptions 会怎么变
- state 面：新增哪些 key，如何 reducer，哪些是 private
- streaming 面：哪些事件会被外部流消费者看到
- interrupt / approval 面：谁能暂停谁，在哪一层暂停
- return 面：最终哪些结果能回到 parent / caller

这一步尤其重要，因为三层栈里最危险的 bug，往往不是“功能不能跑”，而是 contract 没定义，结果各层各自猜。

### 3. 用最小层次实现，不要先动 `graph.py`

优先顺序通常应该是：

1. 能在现有上游 primitive / runtime 配置里实现，就不要改装配根
2. 能放进单一 middleware / backend / profile，就不要改全局 assembly
3. 只有当默认装配本身必须变化时，才去改 `create_deep_agent()` / `graph.py`

这能最大化降低回归面。

### 4. 如果是 compiled subagent 的能力，优先在子图内部解决

这是一个很容易踩坑的点。

例如你想要：

- 子代理内部自己的审批规则
- 子代理内部自己的 token 可见性策略
- 子代理内部自己的私有 planning state

那通常应该：

- 在 compiled runnable 自己内部加 middleware / node / stream 策略
- 而不是希望父图顶层开关自动伸进去

第 5 章已经说明：

- `CompiledSubAgent` 是 use-as-is
- 父级 `interrupt_on` 不会自动进入它内部
- `nostream` 也只是 LangGraph 对某次模型调用的 `messages` 流抑制

### 5. 先补局部验证，再接默认装配

一个推荐顺序：

1. 先写 unit test 锁住局部 contract
2. 再写 integration test 锁住跨层边界
3. 再接入默认装配
4. 最后跑 smoke / snapshot / example 验证

这样失败时你更容易知道：

- 是局部实现错了
- 还是 assembly 顺序把行为改坏了

### 6. 改完后回头更新“已知边界说明”

对这套栈来说，文档不是收尾装饰，而是 contract 的一部分。

尤其是涉及这些场景时：

- callback 传播
- streaming 可见性
- compiled subagent 边界
- permissions 与 execute capability

如果行为有变化，教程和测试都要同步更新；如果行为仍有限制，最好保留 `xfail` 或显式 caveat。

---

## 两个具体案例

### 案例 1：我想控制子代理内部哪些 token 对流消费者可见

这类需求不要先去找 Deep Agents 的“全局隐藏开关”。

更合理的做法是先分层：

- token 是否进入 `messages` 流：LangGraph
- 是否对 root consumer 暴露子图事件：`subgraphs=True/False`
- 是否只发阶段信号：`custom` + `runtime.stream_writer(...)`
- 私有中间结果是否最终回到 parent：Deep Agents 的 return/state 边界

一个常见可行解法是：

- 私有模型调用打 `tags=["nostream"]`
- 私有草稿不写回 parent 可见 state
- 公开阶段用 `custom` 事件通知 UI

### 案例 2：我想让 compiled subagent 内部工具也被父审批规则拦住

这通常不是父 `interrupt_on` 能自动做到的。

因为：

- declarative `SubAgent` 会在构建子图时显式加对应 middleware
- `CompiledSubAgent` 则是直接复用现成 runnable

所以正确方向通常是：

- 在 compiled subagent 自己内部加 HITL / middleware
- 或者改回 declarative subagent 路径

而不是继续增强父图顶层开关，期待它透明穿透 child graph。

---

## 高风险区域

下面这些地方改动前要格外保守：

### `graph.py` 的默认装配顺序

这里一旦动错，坏的通常不是单个功能，而是整套默认 harness 行为。

### subagent 的 ingress / egress state 过滤

这里直接影响：

- private state 泄漏
- parent-child 污染
- structured output 与 messages 的边界

### streaming 与 result return 的混淆

“流里看不到”和“parent 永远不知道”不是一回事。

### profile 合并与 provider 差异

exact-model override 很容易无意间抹掉 provider 默认项。

### backend execute 能力与 permissions 的耦合

permissions 只能约束模型看到的工具，不能凭空替代运行介质能力模型。

---

## 什么时候该修上游

### 更像上游问题

- `patch_config()` / callback tree / `get_child()` 语义不对
- `messages` / `custom` / `subgraphs` streaming 与预期不一致
- provider 集成本身缺少需要的模型能力

### 更像 Deep Agents 本地问题

- 默认 middleware / backend / profile / permissions policy 不合理
- declarative 与 compiled subagent 边界处理不一致
- prompt layering / memory / skills 的装配策略不合理

---

## 容易踩什么坑

- 坑 1：一有需求就先改 `graph.py`。
  装配根应该是最后动的地方，而不是默认入口。

- 坑 2：把 runtime 可见性问题误写成 Deep Agents 全局策略问题。

- 坑 3：把 compiled subagent 也当成会自动继承父图所有规则。

- 坑 4：只定义“功能能做什么”，不定义“哪些内容可见、可拦截、可回传”。

- 坑 5：改完代码不更新边界文档和测试。

---

## 本章小结

- 新能力落地前，先判断 ownership，再定义 contract，再决定实现层。
- `LangChain` 管 primitive，`LangGraph` 管 runtime，`Deep Agents` 管默认 harness policy。
- 对 compiled subagent、callback、streaming 这类边界问题，最危险的做法就是在错误层硬加全局开关。
- 安全的改法通常是：最小层实现、局部测试先行、默认装配最后接入。
