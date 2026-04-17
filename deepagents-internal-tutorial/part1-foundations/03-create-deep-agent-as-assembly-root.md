# 第3章：create_deep_agent 作为装配根

## 学习目标

学完本章，你应该能回答：

1. `create_deep_agent()` 与上游 `create_agent()` 的关系是什么
2. 为什么它是 assembly root，而不是新的 runtime 实现
3. 哪些行为是继承上游，哪些行为是 Deep Agents 本地策略

---

## 问题是什么

维护者看到 `create_deep_agent()` 时，常见的两个错误理解是：

- 它只是一个 convenience wrapper
- 它重写了完整的 agent 执行模型

更准确的理解介于两者之间：

> `create_deep_agent()` 不是薄薄一层语法糖，但它也不是新的 runtime；它是把上游 agent/runtime primitive 约束成一套默认 harness 的 assembly root。

---

## 哪一层负责什么

### `LangChain`

- `create_agent()` 负责产出 agent graph
- middleware hook surface 决定哪些本地中间件可以插进去
- model/tool primitive 决定最终执行形状

### `LangGraph`

- compiled graph 的执行、state、checkpoint、streaming 由它负责
- `create_agent()` 返回的本质上仍是 compiled graph

### `Deep Agents`

- 选择默认 middleware 顺序
- 决定 general-purpose subagent 是否自动注入
- 决定 permissions、profiles、skills、memory、subagent defaults 如何接入

---

## 代码在哪里

建议同时打开：

- `deepagents/libs/deepagents/deepagents/graph.py`
- `deepagents/libs/deepagents/deepagents/middleware/subagents.py`
- `langchain/libs/langchain_v1/langchain/agents/factory.py`

---

## 实现怎么工作

### 1. 入口先做 model/profile 归一化

在 `graph.py` 里，`create_deep_agent()` 先解决：

- `model=None` 时的默认模型
- `resolve_model()` 后的标准模型对象
- `_harness_profile_for_model()` 给出的 provider/model 级本地策略

这一步还不是执行，只是在决定：

- tool 描述要不要改写
- extra middleware 要不要注入
- 某些工具要不要默认排除

### 2. 它先构建 general-purpose subagent

`graph.py` 里先单独拼出 general-purpose subagent middleware 栈，再把它作为默认 spec 注入。这说明两个事实：

- general-purpose 不是事后补丁，而是 harness 设计的一部分
- general-purpose 与主 agent 共用很多默认策略，但也有自己的局部栈

### 3. declarative subagent 会被补全，compiled subagent 不会

对 declarative `SubAgent`：

- Deep Agents 会补模型、工具、middleware、permissions、`interrupt_on`

对 `CompiledSubAgent`：

- 直接 use-as-is
- 不继承顶层 `interrupt_on`
- 不自动套上顶层默认 middleware

这正是 compiled subagent 边界讨论的根源。

### 4. 主 agent middleware 顺序是本地 contract

`graph.py` 明确拼出了主 agent 的默认顺序。这个顺序不是美观问题，而是行为 contract：

- 哪些工具先注入
- 哪些 prompt 先改写
- memory 为什么在 provider extra middleware 之后
- permissions 为什么必须最后

这里最重要的判断标准不是“能不能跑”，而是“行为是否仍然和既有测试、prompt、tool surface 一致”。

### 5. 最后仍然是上游 `create_agent()` 在产出 compiled graph

`create_deep_agent()` 的最后一步不是自己写执行器，而是调用上游 `create_agent()`。因此：

- 真正的 graph 执行语义仍来自 LangGraph / LangChain
- Deep Agents 主要决定的是装配结果，而不是底层执行循环

---

## `create_agent()` 和 `create_deep_agent()` 的区别

| 问题 | `create_agent()` | `create_deep_agent()` |
|------|------------------|-----------------------|
| 角色 | 上游通用 agent factory | Deep Agents 的 harness assembly root |
| 是否自动注入 filesystem/todo/subagent 等能力 | 否 | 是 |
| 是否自动处理 Deep Agents profile / backend / permissions | 否 | 是 |
| 最终返回值 | compiled agent graph | compiled agent graph |
| 底层 runtime 语义来自哪里 | LangChain + LangGraph | 仍然是 LangChain + LangGraph |

---

## 如何把后续章节挂回装配根

如果你后面在 [第4章](../part2-core-runtime/04-filesystem-and-state-model.md) 查 filesystem / backend 问题，或者在 [第8章](../part3-extensibility/08-backend-protocol-and-storage-strategy.md) 查扩展策略，最后都应该回到这里确认：
这些能力究竟是在哪个 middleware、backend、profile 位置被接进 `create_deep_agent()` 的。

如果你在 [第11章](../part4-production-patterns/11-reading-the-examples-like-a-maintainer.md) 里通过 example 追到某个 wiring，也应该回跳本章确认：
那个 wiring 究竟是在复用默认 harness，还是 example 自己又包了一层本地装配。

所以这章的作用不是再讲一遍 runtime，而是给后续所有 case study 和扩展章节提供一个固定回挂点：
先回到 assembly root，再判断问题属于上游 primitive，还是属于 Deep Agents 的默认装配策略。

---

## 为什么 middleware 顺序不能随便动

### `SubAgentMiddleware` 不只是多一个工具

它还会改 system prompt，并把可用 subagent 类型暴露给主模型。

### provider extra middleware 不能随便前后挪

因为它可能影响 prompt cache、tool surface、model-specific behavior。

### `_PermissionMiddleware` 必须最后

否则它看不到前面 middleware 新加进来的工具。

---

## 什么时候该修上游，什么时候该修本地

### 更像上游问题

- `create_agent()` 产出的 graph 本身行为变了
- middleware hook surface 不再按预期组合
- callback / stream / state reducer 语义漂移

### 更像 Deep Agents 本地问题

- 默认 middleware 顺序不合适
- general-purpose subagent 默认值不合适
- profile / permissions / backend adapter 策略不合适

---

## 容易踩什么坑

- 坑 1：把 compiled subagent 的所有行为都看成是主 agent 默认栈的一部分。
  实际上它在装配期就被明确标记为 use-as-is。

- 坑 2：看到 `create_deep_agent()` 最后返回 compiled graph，就以为前面装配逻辑不重要。
  恰恰相反，Deep Agents 的主要价值就在这些前置装配决策。

- 坑 3：只把 `graph.py` 当作“配置收集函数”。
  它实际上定义了 Deep Agents 最核心的本地 contract。

---

## 本章小结

- `create_deep_agent()` 是 assembly root。
- 它做的核心工作是装配，不是重写 runtime。
- 执行语义仍大量继承上游。
- 真正属于 Deep Agents 的核心设计，是默认 middleware、subagent、profile、permissions、backend 的组合方式。
