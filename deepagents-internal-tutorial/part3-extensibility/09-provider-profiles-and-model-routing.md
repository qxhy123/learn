# 第9章：Provider Profiles、模型解析与 Middleware Surface

## 学习目标

学完本章，你应该能回答：

1. `resolve_model()` 和 `_HarnessProfile` 分别负责什么
2. provider 级 profile、exact-model 级 profile、动态模型路由之间有什么边界
3. `extra_middleware` 为什么本质上依赖的是 LangChain agent middleware surface
4. 什么时候该改 profile，什么时候该改 middleware，什么时候该修上游 provider 集成
5. 为什么 profile 该被当成内部维护面，而不是稳定的公共 API

---

## 问题是什么

Deep Agents 对外希望保持一个统一入口，但底层 provider 永远不统一：

- 初始化参数不一样
- 默认 API surface 不一样
- 某些 provider 需要 prompt caching、特定 header、特定 tool policy
- 某些 exact model 又会在 provider 默认行为之上再有一层例外

如果这些差异直接散在 `graph.py` 里，维护很快会失控。于是 Deep Agents 把“provider/model 级 harness 差异”集中进了 profiles。

但这里还有一个常见误判：

> profile 不是通用模型路由框架，它更像 provider/model 适配层。

---

## 哪一层负责什么

### `LangChain`

- `init_chat_model(...)` 负责 provider 适配与模型实例化
- `BaseChatModel` 定义统一模型接口
- `langchain_v1/agents/middleware` 定义 agent middleware hook surface
- `factory.py` 负责把 `before_model`、`wrap_model_call`、`wrap_tool_call` 等 hook 组装进 agent

### `LangGraph`

- 执行编译后的 graph
- 不负责 provider 级默认策略
- 只消费“已经选好的 model + middleware + tool graph”

### `Deep Agents`

- `resolve_model()` 的 profile-aware 初始化
- `_HarnessProfile` 的 prompt/tool/middleware/provider policy
- provider 与 exact-model profile 的合并
- 把 profile 影响扩散到主 agent、默认 subagent、声明式 subagent

---

## 代码在哪里

建议同时打开：

- `deepagents/libs/deepagents/deepagents/_models.py`
- `deepagents/libs/deepagents/deepagents/profiles/_harness_profiles.py`
- `deepagents/libs/deepagents/deepagents/profiles/_openai.py`
- `deepagents/libs/deepagents/deepagents/profiles/_openrouter.py`
- `deepagents/libs/deepagents/deepagents/profiles/__init__.py`
- `deepagents/libs/deepagents/tests/unit_tests/test_models.py`
- `langchain/libs/langchain_v1/langchain/agents/middleware/types.py`
- `langchain/libs/langchain_v1/langchain/agents/factory.py`

---

## 实现怎么工作

### 1. `resolve_model()` 先做 harness 适配，再交给上游 provider 集成

`resolve_model()` 的流程非常克制：

1. 如果传进来已经是 `BaseChatModel`，直接返回
2. 如果是字符串 spec，先查 `_HarnessProfile`
3. 执行 `pre_init`
4. 合并 `init_kwargs` 与 `init_kwargs_factory`
5. 最后才调用 `init_chat_model(...)`

这说明：

- Deep Agents 没有重写 provider SDK
- 它只是把 provider/model 特有的 harness 适配放在实例化之前

### 2. profile lookup 不是二选一，而是 provider base + exact override

`_get_harness_profile(spec)` 的查找顺序是：

1. 先找精确的 `provider:model`
2. 再找 provider 级 key
3. 两者都存在时，先用 provider profile 作为 base，再叠 exact-model override

这个合并行为非常关键，因为 exact-model tweak 不应该把 provider 默认行为一起抹掉。

### 3. `_HarnessProfile` 改的不只是 init kwargs

profile 目前能改的面包括：

- `init_kwargs`
- `pre_init`
- `init_kwargs_factory`
- `base_system_prompt`
- `system_prompt_suffix`
- `tool_description_overrides`
- `excluded_tools`
- `extra_middleware`

所以它不是单纯的“provider 参数表”，而是：

> provider/model 级 harness customization registry

### 4. `extra_middleware` 的真实落点在 LangChain agent middleware surface

Deep Agents 可以在 profile 里注册 `extra_middleware`，但这些 middleware 并不是 Deep Agents 自创协议。

真正的 hook surface 仍然来自上游：

- `before_agent`
- `before_model`
- `wrap_model_call`
- `wrap_tool_call`
- `after_model`
- `after_agent`

`langchain_v1/langchain/agents/factory.py` 会把这些 hook 链接进最终 agent 图。

所以这里的层次关系应该写成：

- Deep Agents 决定“哪些 middleware 默认挂上去”
- LangChain 决定“这些 middleware 在 agent execution 里怎样生效”

### 5. middleware 合并是按类型替换，不是机械 append

`_merge_middleware()` 的策略是：

- override 中同类型 middleware 替换 base 中同类型实例，并保留原位置
- 新类型才追加到末尾

这让 profile override 更像“精确改默认值”，而不是“再套一层重复中间件”。

### 6. profile 解决的是 provider/model 差异，不是业务场景路由

更适合 profile 的东西：

- 某 provider 必须带默认 init kwargs
- 某 exact model 要排除某个 built-in tool
- 某 provider 默认要追加一个 prompt suffix
- 某 provider 要在全局默认栈里多挂一个 middleware

不适合 profile 的东西：

- 某个具体业务场景才需要的指令
- 根据当前用户请求动态切模型
- 按 token budget / latency 在运行时选模型
- 某个 example 私有的 workflow 规则

### 7. profile 的作用范围要按 subagent 形态区分

profile 不只影响主 agent。

它会影响：

- 主 agent
- auto-injected 的 `general-purpose` subagent
- 声明式 `SubAgent`

但对 `CompiledSubAgent`，结论要更谨慎：

- 如果 compiled runnable 是你自己在外部先构好的，内部 model/middleware 是否受 profile 影响，取决于你构它时有没有走同样的装配路径
- 它不是像声明式 subagent 那样自动再吃一遍父级 profile 策略

---

## 两个 provider case study

### Case 1：`openai` profile 解决的是 provider 默认 API 面

`profiles/_openai.py` 当前注册的是：

- `openai` provider profile
- 默认 `init_kwargs={"use_responses_api": True}`

这个 case 的重点不是“OpenAI 有什么功能”，而是：

- provider 默认行为差异被收敛到了 profile
- `graph.py` 不需要到处散落 “如果是 OpenAI 就加这个参数”

### Case 2：`openrouter` profile 解决的是运行前检查 + attribution kwargs

`profiles/_openrouter.py` 现在做了两件典型的 profile 工作：

- `pre_init` 里检查 `langchain-openrouter` 版本下限
- `init_kwargs_factory` 动态构造 attribution kwargs，并尊重环境变量覆盖

这个 case 非常适合说明：

- profile 不只是静态字典
- 它也可以表达 provider 级的 runtime preflight 与默认 kwargs factory

---

## 内部 API 警示

维护教程里必须明确写出这一点：

- `deepagents.profiles.__init__`
- `deepagents.profiles._openai`
- `deepagents.profiles._openrouter`

这些文件都明确标了：

> internal API subject to change without deprecation

所以对维护者的正确表述应该是：

- profile 是可维护、可扩展的内部面
- 但不应被包装成对外稳定承诺的公共 API

---

## profile / middleware / upstream 的选择表

| 需求 | 更适合哪层 |
|------|------------|
| OpenAI 某模型默认 init kwargs | profile |
| 某 provider 默认禁用一个 built-in tool | profile |
| 按请求内容动态切快/慢模型 | 上层策略或 `wrap_model_call` |
| 给所有 agent 加一层 provider 专用 middleware | profile 的 `extra_middleware` |
| 修某 provider SDK 的参数支持 | 上游 `langchain` provider 集成 |
| 某个 example 的 research prompt | example / consumer 自己配置 |

---

## profile cookbook

### 场景 1：你只想改一个 provider 的默认初始化参数

优先改 provider 级 profile。

### 场景 2：你只想改某个精确模型的例外行为

优先加 exact-model profile，不要复制整份 provider base 配置。

### 场景 3：你想按请求动态切模型

优先放在更上层的 runtime strategy 或 `wrap_model_call`，不要把它硬塞进 profile。

### 场景 4：你想给某 provider 默认挂一层 agent middleware

可以放进 `extra_middleware`，但要明确：

- Deep Agents 只负责挂上去
- 真正的执行语义仍由 LangChain agent factory 决定

---

## 什么时候该修上游

### 更像上游问题

- `init_chat_model(...)` 对某 provider 的行为不符合预期
- provider chat model 本身缺少你需要的原生能力
- `AgentMiddleware` hook surface 或 agent factory 行为和文档不一致

### 更像 Deep Agents 本地问题

- provider/default model 的 profile 缺失或配置错误
- `extra_middleware`、`excluded_tools`、prompt suffix 这种 harness policy 不合理
- exact-model profile 覆盖 provider profile 时丢了默认行为

---

## 容易踩什么坑

- 坑 1：把业务 prompt 定制塞进 profile。
  profile 该表达的是 provider/model 差异，不是产品策略。

- 坑 2：把 profile 当成通用模型路由系统。
  它更偏“静态适配”，不是“运行时路由控制面”。

- 坑 3：忘了 `extra_middleware` 的真实执行语义仍取决于 LangChain agent factory。

- 坑 4：新增 exact-model profile 时没检查 provider base 是否还被继承。

- 坑 5：把 `CompiledSubAgent` 也当成会自动吃到同样 profile 装配。
  compiled 场景必须看它是怎样构出来的。

---

## 本章小结

- `resolve_model()` 是 profile-aware 的模型解析入口，但最终 provider 实例化仍由上游 `init_chat_model()` 完成。
- `_HarnessProfile` 是 provider/model 级 harness 适配中心，能改 prompt、tool、middleware，不只是 init kwargs。
- `extra_middleware` 的执行面建立在 LangChain agent middleware surface 之上。
- profile 适合静态 provider/model 差异；动态模型选择通常不该塞进 profile。
- 维护文档应该把 profile 视为内部维护面，而不是稳定公共 API。
