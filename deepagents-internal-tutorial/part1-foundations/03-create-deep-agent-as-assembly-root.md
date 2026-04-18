# 第3章：create_deep_agent 作为装配根

## 本章回答什么

- `create_deep_agent()` 与上游 `create_agent()` 的关系是什么
- 为什么它是 Deep Agents 的 assembly root，而不是新的 runtime 实现
- 哪些行为继承上游 contract，哪些行为属于 Deep Agents 的本地装配策略与默认顺序

## 在整套系统中的位置

- 横向主题：`Assembly`, `Execution`
- 前置章节：[README](../README.md)、[前言：如何使用本教程](../00-preface.md)、[第1章：这一栈到底在构建什么](./01-what-deepagents-builds.md)、[第2章：仓库地图与包边界](./02-repo-map-and-package-boundaries.md)
- 后续章节：[第4章：Filesystem 与状态模型](../part2-core-runtime/04-filesystem-and-state-model.md)、[第7章：Subagents、拦截边界与上下文隔离](../part2-core-runtime/07-subagents-and-context-isolation.md)、[第13章：Backend 协议与存储策略](../part4-maintenance-and-extension/13-backend-protocol-and-storage-strategy.md)、[第16章：像维护者一样阅读 examples](../part4-maintenance-and-extension/16-reading-the-examples-like-a-maintainer.md)

## 静态结构

这一章的任务是把 Deep Agents 的本地 contract 收口到一个明确入口：`create_deep_agent()`。后续看到 filesystem、subagent、permissions、profiles、memory 等能力时，都应该先追问它们是在哪个装配点接进去的。

### 代码在哪里

建议同时打开：

- `deepagents/libs/deepagents/deepagents/graph.py`
- `deepagents/libs/deepagents/deepagents/middleware/subagents.py`
- `langchain/libs/langchain_v1/langchain/agents/factory.py`

### `create_agent()` 和 `create_deep_agent()` 的区别

| 问题 | `create_agent()` | `create_deep_agent()` |
|------|------------------|-----------------------|
| 角色 | 上游通用 agent factory | Deep Agents 的 harness assembly root |
| 是否自动注入 filesystem / todo / subagent 等能力 | 否 | 是 |
| 是否自动处理 Deep Agents profile / backend / permissions | 否 | 是 |
| 最终返回值 | compiled agent graph | compiled agent graph |
| 底层 runtime 语义来自哪里 | LangChain + LangGraph | 仍然是 LangChain + LangGraph |

## 运行时链路

### 1. 入口先做 model / profile 归一化

在 `graph.py` 里，`create_deep_agent()` 先解决：

- `model=None` 时的默认模型
- `resolve_model()` 后的标准模型对象
- `_harness_profile_for_model()` 给出的 provider / model 级本地策略

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

## 传播 / 可见性 / 拦截点

### 为什么 middleware 顺序不能随便动

#### `SubAgentMiddleware` 不只是多一个工具

它还会改 system prompt，并把可用 subagent 类型暴露给主模型。

#### provider extra middleware 不能随便前后挪

因为它可能影响 prompt cache、tool surface、model-specific behavior。

#### `_PermissionMiddleware` 必须最后

否则它看不到前面 middleware 新加进来的工具。

### 后续章节为什么都要回挂 assembly root

如果你后面在 [第4章](../part2-core-runtime/04-filesystem-and-state-model.md) 查 filesystem / backend 问题，或者在 [第13章](../part4-maintenance-and-extension/13-backend-protocol-and-storage-strategy.md) 查扩展策略，最后都应该回到这里确认：这些能力究竟是在哪个 middleware、backend、profile 位置被接进 `create_deep_agent()` 的。

如果你在 [第16章](../part4-maintenance-and-extension/16-reading-the-examples-like-a-maintainer.md) 里通过 example 追到某个 wiring，也应该回跳本章确认：那个 wiring 究竟是在复用默认 harness，还是 example 自己又包了一层本地装配。

所以这章的作用不是再讲一遍 runtime，而是给后续所有 case study 和扩展章节提供一个固定回挂点：先回到 assembly root，再判断问题属于上游 primitive，还是属于 Deep Agents 的默认装配策略。

## 扩展接口

对维护者真正重要的装配入口包括：

- `middleware=`：在主 agent 栈中间插入本地策略
- `subagents=`：可选 declarative、compiled、async 三种形态
- `backend=`：替换 filesystem / execute / state 的底层实现
- `permissions=`：统一给主 agent 与 declarative subagent 收口 tool 权限
- profile / skills / memory：调整 prompt、工具排除、provider-specific middleware 和长期行为策略

### 什么时候该修上游，什么时候该修本地

#### 更像上游问题

- `create_agent()` 产出的 graph 本身行为变了
- middleware hook surface 不再按预期组合
- callback / stream / state reducer 语义漂移

#### 更像 Deep Agents 本地问题

- 默认 middleware 顺序不合适
- general-purpose subagent 默认值不合适
- profile / permissions / backend adapter 策略不合适

## 常见问题与排障入口

- 坑 1：把 compiled subagent 的所有行为都看成是主 agent 默认栈的一部分。实际上它在装配期就被明确标记为 use-as-is。
- 坑 2：看到 `create_deep_agent()` 最后返回 compiled graph，就以为前面装配逻辑不重要。恰恰相反，Deep Agents 的主要价值就在这些前置装配决策。
- 坑 3：只把 `graph.py` 当作“配置收集函数”。它实际上定义了 Deep Agents 最核心的本地 contract。

排障时可以先这样分层：

- 看默认顺序、默认 subagent、permissions/profile 注入：先查 `deepagents/graph.py`
- 看 subagent state 过滤、`task` tool、compiled subagent use-as-is 边界：再查 `middleware/subagents.py`
- 看 graph 执行语义、streaming、checkpoint：转去 LangGraph
- 看 callbacks / config / middleware hook 组合：转去 LangChain

## 本章结论

- 谁提供：`create_deep_agent()` 提供的是 Deep Agents 的 assembly contract；真正的 graph factory 和 runtime 仍由 LangChain 与 LangGraph 提供。
- 如何传播：装配先做 model / profile 归一化，再生成主 agent 与 subagent 的 middleware 栈，最后把配置交给上游 `create_agent()` 进入 compiled graph 执行。
- 修在哪层：只要问题是默认 middleware、subagent、permissions、backend、profile 的组合方式，就优先修 `deepagents/graph.py`；一旦涉及 graph 语义或 callback/config 传播，就回上游。
