# 第6章：Memory、Skills、Prompt Layering 与 Config 传播

## 本章回答什么

- `AGENTS.md`、`SKILL.md`、base system prompt、run config 分别属于哪种上下文注入面
- memory 与 skills 具体是怎么从 backend 装进当前 run 的
- `memory_contents`、`skills_metadata` 为什么是 private state，以及它们为什么不会随便泄漏给子代理
- `memory=[...]` 与 `/memories/...` 为什么相关但不是同一个概念
- 发现“prompt 里有 / prompt 里没有 / 子代理没继承 / 同线程没刷新”时应该先查哪一层

## 在整套系统中的位置

- 横向主题：`Context Injection`、`Prompt Layering`
- 前置章节：[README](../README.md)、[前言：如何使用本教程](../00-preface.md)、[第3章：create_deep_agent 作为装配根](../part1-foundations/03-create-deep-agent-as-assembly-root.md)、[第4章：Filesystem 与状态模型](./04-filesystem-and-state-model.md)、[第5章：Tools 作为 Runtime Surface](./05-tools-as-runtime-surface.md)
- 后续章节：[第7章：Subagents、任务交接与上下文隔离](./07-subagents-and-context-isolation.md)、[第13章：Backend 协议与存储策略](../part4-maintenance-and-extension/13-backend-protocol-and-storage-strategy.md)

这一章只讨论“哪些上下文材料会被装进当前 run”，不再把 callback tree、token streaming、外层观测矩阵一起揉进来。这里的核心是注入面，不是传播理论。

## 静态结构

建议同时打开这些实现文件：

- `deepagents/libs/deepagents/deepagents/graph.py`
- `deepagents/libs/deepagents/deepagents/middleware/memory.py`
- `deepagents/libs/deepagents/deepagents/middleware/skills.py`
- `deepagents/libs/deepagents/deepagents/middleware/subagents.py`

### 四种上下文注入面

| 注入面 | 由谁提供 | 本章关心的事实 |
| --- | --- | --- |
| base / user system prompt | `graph.py` | 定义当前 agent 的基础行为说明 |
| memory | `MemoryMiddleware` | 把若干 `AGENTS.md` source 读成 always-on prompt material |
| skills | `SkillsMiddleware` | 把 `SKILL.md` 元数据列进 prompt，完整技能文件按需再读 |
| run config / runtime context | `create_agent().with_config(...)`、`invoke(..., config=...)`、`Runtime` / `ToolRuntime` | 提供 run-local metadata、recursion limit、上下文对象；这和 prompt 文本不是一回事 |

### `AGENTS.md` 与 `SKILL.md` 的分工

| 文件 | 更接近什么 | 默认注入方式 |
| --- | --- | --- |
| `AGENTS.md` | always-on 行为说明、长期上下文 | 通过 `memory=[...]` 进入 `MemoryMiddleware`，再拼进 system prompt |
| `SKILL.md` | 按任务触发的工作流模块 | 通过 `skills=[...]` 让 `SkillsMiddleware` 列出技能目录与元数据，完整内容需要再读文件 |

因此它们都属于“上下文材料”，但不是同一层 contract：

- `AGENTS.md` 适合持续生效
- `SKILL.md` 适合 progressive disclosure

### 两个 private state key

这一章必须单独记住两个 key：

- `memory_contents`
- `skills_metadata`

它们都进入 state，但都被标记为 `PrivateStateAttr`。这意味着：

- 它们是系统内部持有的上下文材料
- 它们默认不应暴露成用户结果面
- 它们也不应在 parent-child handoff 时无约束泄漏

## 运行时链路

### 1. 装配顺序决定 prompt layering

`create_deep_agent()` 在 `graph.py` 中先拼基础 prompt，再按 middleware 顺序决定哪些材料会附加到当前 model call。

当前与本章最相关的顺序是：

1. 用户传入的 `system_prompt`
2. Deep Agents base prompt / profile suffix
3. `SkillsMiddleware` 注入技能目录与技能元数据摘要
4. 其他 middleware 改写
5. `MemoryMiddleware` 在较靠后位置追加 `<agent_memory>...</agent_memory>`
6. `create_agent(...).with_config(...)` 再给 run 加上默认 `recursion_limit` 与 `metadata`

这里顺序不是美观问题，而是行为 contract。尤其是 memory 放在 provider-specific middleware 之后，是为了不让 memory 更新破坏前缀缓存的可复用部分。

### 2. memory 的加载链路是 backend -> private state -> prompt

当你传入 `memory=[...]` 时，`graph.py` 会在主 agent middleware 栈中追加 `MemoryMiddleware`。之后链路是：

1. `before_agent()` / `abefore_agent()` 检查 state 是否已有 `memory_contents`
2. 若没有，则调用 backend 的 `download_files(...)` / `adownload_files(...)`
3. 按 `sources` 顺序读取若干 `AGENTS.md`
4. 缺失文件跳过，其他内容写入 `memory_contents`
5. `wrap_model_call()` / `awrap_model_call()` 把它们格式化成 `<agent_memory>` 片段并追加到 system prompt

维护时要抓住三点：

- memory source 的顺序就是 prompt 中的呈现顺序
- `memory_contents` 一旦已在当前线程状态中存在，同线程后续 turn 默认不会自动重载
- `MemoryMiddleware` 不直接碰本地磁盘，它只认 backend

### 3. skills 的加载链路是 backend -> metadata -> prompt 入口

当你传入 `skills=[...]` 时，`graph.py` 会把 `SkillsMiddleware` 接到主 agent 和 general-purpose subagent；声明式 subagent 只有在自己声明了 `skills` 时才会额外接入。

链路是：

1. `before_agent()` / `abefore_agent()` 扫描每个 skills source
2. 找出其中的 `SKILL.md`
3. 解析 YAML frontmatter，得到 `SkillMetadata`
4. 不同 source 中同名 skill 采用“后者覆盖前者”
5. `wrap_model_call()` 把 skills 目录位置和技能摘要写进当前 system prompt

这里的关键不是“把所有 skill 内容都塞进 prompt”，而是：

- 先把技能索引暴露给模型
- 完整 `SKILL.md` 由模型在需要时再通过文件工具读取

### 4. `/memories/...` 与 `memory=[...]` 不同层

这两个概念必须分开写：

- `memory=[...]` 是 `MemoryMiddleware` 的 source 列表，决定哪些文件会自动进入 always-on memory prompt
- `/memories/...` 只是某些 backend 或 `CompositeBackend` 路由里的长期路径约定

所以：

- 一个文件放在 `/memories/...` 下，不代表它会自动进入 prompt
- 一个 memory source 也不要求必须叫 `/memories/...`

### 5. backend 决定 memory / skills 的真实来源

这一章里，“上下文从哪里来”不是由 middleware 自己决定，而是由 backend 决定：

| backend | memory / skills 典型来源 |
| --- | --- |
| `StateBackend` | `invoke({"files": {...}})` 传入的 thread-scoped 文件 |
| `FilesystemBackend` | 宿主机文件系统中的真实 `AGENTS.md` / `SKILL.md` |
| `CompositeBackend` | 由路径前缀路由后的多介质来源，例如 `/memories/` 与 `/skills/` 分别落在不同后端 |

这也是为什么“memory 明明配了却没加载”通常要先查 backend 路由，而不是先查 prompt 拼接代码。

### 6. config 是 run-context 注入面，不是 prompt 文本

`create_agent(...).with_config(...)` 会给 Deep Agents 默认加上：

- `recursion_limit`
- `metadata`

同时，调用方还可以在 invoke/stream 时再传入 `config`。这些值进入的是 run context，不是自动进入 prompt 的自然语言文本。

因此维护时要分清：

- prompt 文本属于“模型看见什么说明”
- run config 属于“当前执行期挂了什么运行上下文”

## 传播 / 可见性 / 拦截点

这一节只保留“上下文是否被注入、是否被过滤”的最小判断。

### 1. `memory_contents` 与 `skills_metadata` 是私有上下文，不是公开结果面

从 state 角度看，这两个 key 真实存在；但它们被设计成：

- 不应出现在普通 final result 顶层
- 不应默认沿 parent -> child handoff 继续透传
- 不应在 child -> parent 回传时重新冒泡

`subagents.py` 里的 `_EXCLUDED_STATE_KEYS` 还会把它们显式挡在 parent-child ingress / egress 之外。

### 2. “文件存在”与“自动注入”是两回事

- 文件位于 `/memories/...`，只说明它可能是长期存储约定的一部分
- 文件位于某个 skills source 下并且有 `SKILL.md`，只说明它可能被扫描成 skill metadata
- 只有进入 `memory=[...]` 或 `skills=[...]` 的 source，才会变成当前 run 的自动上下文

### 3. 子代理不自动继承主线程的私有上下文

主 agent 装了 memory / skills，不等于所有子代理都天然共享同一份私有上下文。对于 maintainer，这是设计边界，不是缺陷：

- parent 的 `memory_contents` / `skills_metadata` 不应直接泄漏给 child
- child 要有自己的 skills/memory，就应该显式装配自己的 source 或 middleware

更完整的 parent-child 隔离边界，继续看 [第7章：Subagents、任务交接与上下文隔离](./07-subagents-and-context-isolation.md)。

### 4. 这里不展开 callback tree 与 stream 可见性

如果你现在关心的是传播、stream consumer 可见性、或者 callback tree 的形状，而不是本章的运行时职责，请跳到 Part 3。

## 扩展接口

### 1. 调整 always-on memory

- 用 `memory=[...]` 明确指定哪些 `AGENTS.md` 会自动进入 prompt
- 需要长期持久化时，让这些路径落到合适 backend，而不是只靠路径命名

### 2. 调整 skills 发现面

- 用 `skills=[...]` 指定技能源目录
- 需要分层覆盖时，让不同 source 按 base -> user -> project 的顺序排列
- 声明式 subagent 若需要独立技能集，应在该 subagent 自己的 `skills` 字段里声明

### 3. 调整 prompt layering

- 顶层 `system_prompt` 负责你自己的业务说明
- `MemoryMiddleware` 与 `SkillsMiddleware` 负责把文件化上下文追加进去
- 若只是改工具理论、tool description、tool surface，不应回到本章处理，而应先看 [第5章](./05-tools-as-runtime-surface.md)

### 4. 调整 run context

- 需要 run-local metadata、tags、recursion limit 时，走 `config`
- 需要线程外部注入的结构化上下文时，走 `context_schema` / runtime context
- 不要把这些运行时字段误写成“prompt 层已经天然知道”

## 常见问题与排障入口

- `AGENTS.md` 明明存在，为什么 prompt 没带上：先确认是否传了 `memory=[...]`，再确认 backend 能否从该路径 `download_files(...)`。
- 改了 memory 文件，为什么同一线程里没立即刷新：先查 `memory_contents` 是否已经在 state 中缓存；当前实现默认不会同线程自动重载。
- 技能目录里有很多文件，为什么 prompt 里只看到摘要：因为 `SkillsMiddleware` 注入的是 `skills_metadata` 列表，完整 `SKILL.md` 需要再读。
- 同名 skill 为什么被后面的 source 覆盖：这是 `SkillsMiddleware` 明确采用的“last one wins” 规则。
- `/memories/foo.md` 为什么没有 automatically 进入上下文：路径命名本身不触发注入，只有 `memory=[...]` 才会。
- 子代理为什么看不到主线程的 memory / skills：这是 private state 过滤结果；若要深入查 parent-child handoff，转去 [第7章](./07-subagents-and-context-isolation.md)。
- 某个值明明在 config 里，为什么 prompt 里没有自然语言体现：因为 config 是运行时上下文，不是 prompt 片段。

## 本章结论

- 谁提供：`graph.py` 提供基础 prompt 与默认 run config，`MemoryMiddleware` 提供 always-on memory 注入，`SkillsMiddleware` 提供 skills 索引注入，backend 提供这些上下文文件的真实来源。
- 如何传播：`AGENTS.md` 通过 `memory=[...]` 读入 `memory_contents` 再拼进 prompt；`SKILL.md` 通过 `skills=[...]` 扫描成 `skills_metadata` 再列进 prompt；run config 则经由 `with_config()` 和调用期 `config` 进入运行上下文。
- 修在哪层：memory source、skills source、private state 过滤、prompt layering 顺序优先修 Deep Agents 本地 middleware / `graph.py`；文件找不到或落错介质优先修 backend；涉及 parent-child 上下文隔离时转查 [第7章](./07-subagents-and-context-isolation.md)。
