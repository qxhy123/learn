# 第14章：Provider Profiles、模型解析与 Middleware Surface

## 本章回答什么

- `resolve_model()` 和 `_HarnessProfile` 分别负责什么，为什么这一步是维护工作流里的 provider/model 适配层
- provider 级 profile、exact-model 级 profile、动态模型路由之间的边界是什么
- `extra_middleware` 为什么落在 LangChain agent middleware surface 上，而不是 Deep Agents 自创协议
- 什么时候该改 profile，什么时候该改 middleware，什么时候该修上游 provider 集成
- 为什么 profile 应该被视为内部维护面，而不是稳定公共 API

## 在整套系统中的位置

- 这一部分默认假设你已经读过 Part 1 和 Part 2。
- 如果当前问题和传播、可见性、callback tree 有关，先回看 Part 3。
- 横向主题：`Maintenance`、`Provider adaptation`、`Middleware surface`
- 前置章节：[第3章：Create Deep Agent 作为 Assembly Root](../part1-foundations/03-create-deep-agent-as-assembly-root.md)、[第5章：Tools 作为 Runtime Surface](../part2-core-runtime/05-tools-as-runtime-surface.md)、[第13章：Backend 协议、存储介质与执行边界](./13-backend-protocol-and-storage-strategy.md)
- 后续章节：[第15章：如何测试一个三层栈 Harness](./15-testing-the-harness.md)

在维护工作流里，确认完“问题是不是 backend / 存储介质边界”之后，下一步通常就是看 provider/model 适配是不是把差异收敛对了。本章处理的正是这一层：哪些默认行为属于 profile，哪些应该通过 middleware surface 暴露，哪些则根本应该回到上游 provider 集成去修。

## 静态结构

建议同时打开这些文件：

- `deepagents/libs/deepagents/deepagents/_models.py`
- `deepagents/libs/deepagents/deepagents/profiles/_harness_profiles.py`
- `deepagents/libs/deepagents/deepagents/profiles/_openai.py`
- `deepagents/libs/deepagents/deepagents/profiles/_openrouter.py`
- `deepagents/libs/deepagents/deepagents/profiles/__init__.py`
- `deepagents/libs/deepagents/tests/unit_tests/test_models.py`
- `langchain/libs/langchain_v1/langchain/agents/middleware/types.py`
- `langchain/libs/langchain_v1/langchain/agents/factory.py`

先把职责边界静态拆开：

| 层 | 首先负责什么 | 维护时最容易误写成什么 |
| --- | --- | --- |
| `LangChain` | `init_chat_model(...)`、`BaseChatModel`、agent middleware hook surface、agent factory 装配 | “profile 就是 provider SDK” |
| `LangGraph` | 编译并执行已经选好的 model + middleware + tools | “图层决定 provider 默认策略” |
| `Deep Agents` | `resolve_model()`、`_HarnessProfile`、provider/exact-model profile 合并、默认 middleware 装配 | “一个通用动态模型路由系统” |

profile 可改的面也要静态列清：

- `init_kwargs`
- `pre_init`
- `init_kwargs_factory`
- `base_system_prompt`
- `system_prompt_suffix`
- `tool_description_overrides`
- `excluded_tools`
- `extra_middleware`

所以 `_HarnessProfile` 不只是 provider 参数表，而是 provider/model 级 harness customization registry。

## 运行时链路

### 1. `resolve_model()` 先做 harness 适配，再交给上游 provider 集成

`resolve_model()` 的流程很克制：

1. 如果传进来已经是 `BaseChatModel`，直接返回
2. 如果是字符串 spec，先查 `_HarnessProfile`
3. 执行 `pre_init`
4. 合并 `init_kwargs` 与 `init_kwargs_factory`
5. 最后才调用 `init_chat_model(...)`

这说明 Deep Agents 没有重写 provider SDK。它做的是实例化前的 harness 适配，而不是替代上游 provider integration。

### 2. profile lookup 不是二选一，而是 provider base + exact override

`_get_harness_profile(spec)` 的查找顺序是：

1. 先找精确的 `provider:model`
2. 再找 provider 级 key
3. 两者都存在时，先用 provider profile 作为 base，再叠 exact-model override

这个合并行为非常关键，因为 exact-model tweak 不应该把 provider 默认行为一起抹掉。

### 3. `extra_middleware` 的真实落点在 LangChain agent middleware surface

Deep Agents 可以在 profile 里注册 `extra_middleware`，但这些 middleware 的执行语义不是本地私有协议。真正的 hook surface 仍来自上游：

- `before_agent`
- `before_model`
- `wrap_model_call`
- `wrap_tool_call`
- `after_model`
- `after_agent`

`langchain_v1/langchain/agents/factory.py` 决定这些 hook 如何链接进最终 agent 图。Deep Agents 决定的是“默认挂哪些 middleware”，不是“重新定义 middleware 语义”。

### 4. middleware 合并是按类型替换，不是机械 append

`_merge_middleware()` 的策略是：

- override 中同类型 middleware 替换 base 中同类型实例，并保留原位置
- 新类型才追加到末尾

这让 profile override 更像“精确改默认值”，而不是“再叠一层重复中间件”。

### 5. profile 解决的是 provider/model 差异，不是业务场景路由

更适合 profile 的问题：

- 某 provider 必须带默认 init kwargs
- 某 exact model 要排除某个 built-in tool
- 某 provider 默认要追加一个 prompt suffix
- 某 provider 要在默认栈里多挂一个 middleware

不适合 profile 的问题：

- 某个业务场景才需要的 prompt
- 根据当前请求动态切模型
- 按 token budget / latency 做运行时选模
- 某个 example 私有的 workflow 规则

### 6. profile 的作用范围会扩散到 subagent 装配

profile 不只影响主 agent，还会影响：

- 主 agent
- auto-injected 的 `general-purpose` subagent
- 声明式 `SubAgent`

但对 `CompiledSubAgent` 必须更谨慎：

- 如果 compiled runnable 是你在外部先构好的，内部 model/middleware 是否受 profile 影响，取决于构它时有没有走同样装配路径
- 它不像声明式 subagent 那样会自动再吃一遍父级 profile 策略

## 传播 / 可见性 / 拦截点

provider/profile 问题很容易和 propagation 问题混写。对维护者来说，最好按三条线区分：

### profile 改的是默认装配，不是 callback contract

- `init_kwargs`、prompt suffix、tool exclusions、`extra_middleware` 会改变最终装配结果
- 但 callback/config/run tree 的传播语义仍来自上游

如果症状是 tags、callbacks、stream visibility 不对，先回看 [第10章：Callbacks、Config 与 Callback Manager](../part3-propagation/10-callbacks-config-and-callback-manager.md) 和 [第11章：Streaming、Visibility 与 Selective Exposure](../part3-propagation/11-streaming-visibility-and-selective-exposure.md)，不要先在 profile 里硬塞补丁。

### middleware surface 会改变模型可见面

profile 中的这些项会直接改变模型可见 surface：

- `base_system_prompt`
- `system_prompt_suffix`
- `tool_description_overrides`
- `excluded_tools`
- `extra_middleware`

这也是为什么 provider/profile 变更最终必须落到 [第15章：如何测试一个三层栈 Harness](./15-testing-the-harness.md) 里的 prompt/tool surface smoke 或 snapshot 去守。

### `CompiledSubAgent` 是最容易误判的拦截点

profile 对声明式 subagent 和 auto-injected subagent 的影响较直接；对 `CompiledSubAgent` 则不能默认成立。若问题只出现在 compiled 场景，更像是传入 runnable 自己的装配路径不同，而不是 profile registry 全局失效。

## 扩展接口

### 两个 provider case study

#### Case 1：`openai` profile 解决的是 provider 默认 API 面

`profiles/_openai.py` 当前注册的是 provider 级 profile，并默认注入 `init_kwargs={"use_responses_api": True}`。这类差异应该集中在 profile，而不是散落在 `graph.py` 里到处写 “如果是 OpenAI 就加这个参数”。

#### Case 2：`openrouter` profile 解决的是 preflight + attribution kwargs

`profiles/_openrouter.py` 现在做了两件很典型的 profile 工作：

- `pre_init` 里检查 `langchain-openrouter` 版本下限
- `init_kwargs_factory` 动态构造 attribution kwargs，并尊重环境变量覆盖

这也说明 profile 不只是静态字典，它还能表达 provider 级 runtime preflight 与默认 kwargs factory。

### profile / middleware / upstream 的选择表

| 需求 | 更适合哪层 |
| --- | --- |
| OpenAI 某模型默认 init kwargs | profile |
| 某 provider 默认禁用一个 built-in tool | profile |
| 按请求内容动态切快/慢模型 | 上层策略或 `wrap_model_call` |
| 给所有 agent 加一层 provider 专用 middleware | profile 的 `extra_middleware` |
| 修某 provider SDK 的参数支持 | 上游 `langchain` provider 集成 |
| 某个 example 的 research prompt | example / consumer 自己配置 |

### profile cookbook

#### 场景 1：你只想改一个 provider 的默认初始化参数

优先改 provider 级 profile。

#### 场景 2：你只想改某个精确模型的例外行为

优先加 exact-model profile，不要复制整份 provider base 配置。

#### 场景 3：你想按请求动态切模型

优先放到更上层的 runtime strategy 或 `wrap_model_call`，不要把它硬塞进 profile。

#### 场景 4：你想给某 provider 默认挂一层 agent middleware

可以放进 `extra_middleware`，但要明确：Deep Agents 只负责挂上去，真正执行语义仍由 LangChain agent factory 决定。

## 常见问题与排障入口

- “为什么某 provider 的默认参数没生效”：先查 `resolve_model()` 是否命中了正确 profile，再查 `pre_init` / `init_kwargs_factory` 是否被覆盖。
- “为什么 exact-model profile 把 provider 默认行为弄丢了”：检查 provider base + exact override 的合并是否还在，而不是先怀疑全部 registry 失效。
- “为什么 profile 变了，但 callback 或 streaming 表现没按想象变化”：profile 不是 callback tree 或 stream surface 的根定义，先回看第10章到第12章。
- “为什么某个 `CompiledSubAgent` 没吃到 profile 里的 middleware”：先看它是不是在外部提前构好、绕开了默认装配路径。
- “为什么这个能力不该放进 profile”：如果它是业务场景路由、动态选模或 example 私有策略，就更可能属于上层策略而不是 profile。

更像上游问题：

- `init_chat_model(...)` 对某 provider 的行为不符合预期
- provider chat model 本身缺少你需要的原生能力
- `AgentMiddleware` hook surface 或 agent factory 行为和文档不一致

更像 Deep Agents 本地问题：

- provider/default model 的 profile 缺失或配置错误
- `extra_middleware`、`excluded_tools`、prompt suffix 这类 harness policy 不合理
- exact-model profile 覆盖 provider profile 时丢了默认行为

还要明确一条内部 API 警示：

- `deepagents.profiles.__init__`
- `deepagents.profiles._openai`
- `deepagents.profiles._openrouter`

这些都应被当成 internal API subject to change without deprecation。维护教程可以教你怎么读、怎么改，但不该把它包装成对外稳定承诺。

## 本章结论

- 谁提供：`LangChain` 提供 provider 实例化接口和 middleware hook surface，`Deep Agents` 用 `resolve_model()` 与 `_HarnessProfile` 把 provider/model 差异收敛成默认 harness 适配。
- 如何传播：provider 或 exact-model profile 先合并成最终装配，再把 init kwargs、prompt、tool exclusions 和 `extra_middleware` 扩散到主 agent 与声明式 subagent；callback/config 传播仍走上游链路。
- 修在哪层：静态 provider/model 差异修 profile，模型可见的 middleware / prompt surface 修装配层，provider SDK 或 hook 语义问题回到上游集成。
