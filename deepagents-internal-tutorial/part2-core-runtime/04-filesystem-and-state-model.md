# 第4章：Filesystem 与状态模型

## 本章回答什么

- Deep Agents 的 filesystem 默认到底由谁提供，为什么它默认不是宿主机磁盘
- `FilesystemState.files`、backend、`ToolRuntime` 分别承担什么职责
- `StateBackend`、`FilesystemBackend`、`CompositeBackend` 的语义边界是什么
- 文件内容什么时候落在 graph state，什么时候落在 backend
- 出现“文件没写进去 / 下一步看不到 / `execute` 消失”时该先查哪一层

## 在整套系统中的位置

- 横向主题：`State`、`Storage`、`Runtime Carrier`
- 前置章节：[README](../README.md)、[前言：如何使用本教程](../00-preface.md)、[第3章：create_deep_agent 作为装配根](../part1-foundations/03-create-deep-agent-as-assembly-root.md)、[第5章：Tools 作为 Runtime Surface](./05-tools-as-runtime-surface.md)
- 后续章节：[第6章：Memory、Skills、Prompt Layering 与 Config 传播](./06-memory-skills-and-system-prompt-layering.md)、[第7章：Subagents、拦截边界与上下文隔离](./07-subagents-and-context-isolation.md)、[第13章：Backend 协议与存储策略](../part4-maintenance-and-extension/13-backend-protocol-and-storage-strategy.md)

这一章只回答 filesystem 作为运行载体时的事实：文件能力挂在哪里、状态落在哪里、backend 怎么决定真实介质。通用的 tool-surface 理论已经收口到 [第5章](./05-tools-as-runtime-surface.md)，这里不再重复展开。

## 静态结构

这一章建议同时打开这些实现文件，但把它们按“状态面”和“介质面”来读，而不是按工具名字来读：

- `deepagents/libs/deepagents/deepagents/middleware/filesystem.py`
- `deepagents/libs/deepagents/deepagents/backends/state.py`
- `deepagents/libs/deepagents/deepagents/backends/filesystem.py`
- `deepagents/libs/deepagents/deepagents/backends/composite.py`
- `deepagents/libs/deepagents/deepagents/backends/protocol.py`
- `deepagents/libs/deepagents/deepagents/graph.py`

### 四个静态部件

| 部件 | 本章关心的职责 |
| --- | --- |
| `FilesystemMiddleware` | 把 filesystem 作为默认 capability 装进 agent，并在运行时解析 backend、过滤 `execute` |
| `FilesystemState.files` | graph 内部的文件状态 channel；只有走 state-native 路径时，它才是文件内容的 canonical surface |
| backend (`StateBackend` / `FilesystemBackend` / `CompositeBackend`) | 决定文件真实落到哪种介质，以及读写、搜索、执行能力的语义 |
| `ToolRuntime` | 文件工具在执行期拿到 `state`、`context`、`config` 的载体；backend factory 也通过它解析当前运行上下文 |

### `FilesystemState.files` 是什么

`FilesystemState` 在 `filesystem.py` 里定义了：

- `files: Annotated[..., _file_data_reducer]`

这说明 `files` 不是普通字典字段，而是带 reducer 的 graph channel。对维护者最重要的含义有两条：

1. 它适合承载 thread-scoped、checkpoint-aware 的工作区文件。
2. 它只描述“graph state 里的文件视图”，不自动等价于所有 backend 的真实存储。

### backend 与 graph state 不是一回事

| 问题 | 应先看哪一层 |
| --- | --- |
| 当前线程里有哪些文件快照 | `FilesystemState.files` |
| 文件最终写到哪里 | backend |
| 为什么这次能不能 `execute` | backend 是否满足 `SandboxBackendProtocol` |
| 为什么同一份路径既像统一 filesystem，又落到多种介质 | `CompositeBackend` |

### 什么算 graph state，什么算 backend

- 当 backend 是 `StateBackend` 时，文件内容本身就存放在 `files` channel 里，graph state 是 canonical source of truth。
- 当 backend 是 `FilesystemBackend` 时，文件内容落在宿主机文件系统；graph state 仍然存在，但不再是文件字节的唯一真源。
- 当 backend 是 `CompositeBackend` 时，不同路径前缀可以分别落到 state、host filesystem 或其他后端；“统一 filesystem 视图”来自路由层，而不是来自单一状态容器。

## 运行时链路

### 1. `create_deep_agent()` 默认把 filesystem 装进去

`graph.py` 默认会给主 agent 和 general-purpose subagent 注入 `FilesystemMiddleware`，并在未显式传参时把 backend 设为 `StateBackend()`。

因此默认结论是：

- filesystem 是默认能力，不是可选插件
- 默认文件介质是 graph state，不是宿主机磁盘

### 2. `ToolRuntime` 是文件工具的运行时载体

filesystem 工具真正执行时，并不是手工传一堆环境参数，而是由 LangGraph 注入 `ToolRuntime`。在这一章里，`ToolRuntime` 的意义很具体：

- tool wrapper 通过它读取 `state`
- backend factory 通过它决定当前应该返回哪个 backend 实例
- 同一次调用里的 `config`、`context`、`tool_call_id` 也都从这里进入工具

这就是为什么“文件工具怎么知道当前线程状态”首先要看 `ToolRuntime`，而不是先猜 middleware 自己维护了额外全局变量。

### 3. 默认链路：`StateBackend` 把文件写回 `files` channel

默认 backend 下，一条典型写路径是：

1. 模型发出 `read_file` / `write_file` / `edit_file` / `glob` / `grep`
2. `FilesystemMiddleware` 暴露出来的工具被 `ToolNode` 调用
3. 工具通过 `ToolRuntime` 解析到 `StateBackend`
4. `StateBackend` 通过 `CONFIG_KEY_READ` 读取当前 `files` 快照
5. 写操作通过 `CONFIG_KEY_SEND` 把增量更新排队进 `files` channel
6. `_file_data_reducer` 在 node boundary 合并这些更新

对维护者最关键的运行时语义是：

- 同一步里读取看到的是一致快照
- 写入不会在同一步内“瞬时改写全局状态”
- 文件会随着 thread / checkpoint 生命周期一起保存

### 4. `FilesystemBackend` 才是真实宿主机文件系统

只有显式提供 `FilesystemBackend(...)` 时，filesystem 才真正落到宿主机磁盘。

这里要抓住两个容易写错的点：

1. `root_dir` 在默认 `virtual_mode=False` 下主要影响相对路径解析，不是硬隔离边界。
2. `virtual_mode=True` 提供的是虚拟路径语义和路径逃逸防护，不等于 sandbox。

所以“给了 `root_dir` 就等于把 agent 关进这个目录”不是本章应当接受的说法。

### 5. `CompositeBackend` 提供统一视图，不提供单一介质

`CompositeBackend` 按路径前缀路由 backend。它的价值不在于“多 backend 并存”，而在于让 agent 仍然看到一个统一的 filesystem 视图。

典型分工是：

- 默认工作区走 `StateBackend`
- `/memories/` 走长期存储 backend
- `/artifacts/` 走独立产物 backend

因此 `/` 下看到的目录列表，可能只是多个介质拼出来的一个虚拟入口。

### 6. `execute` 仍然由 backend 能力决定

虽然 `FilesystemMiddleware` 会创建 `execute` 这个工具，但它只在 backend 具备 `SandboxBackendProtocol` 时才算真正可用。运行时拦截点是 `wrap_model_call()`：

- 有 `execute` 工具定义，不等于这次请求里一定保留它
- backend 不支持执行时，middleware 会把它从本次工具列表里过滤掉

因此“为什么这次没有 `execute`”是 backend 能力问题，不是 `files` state 问题。

## 传播 / 可见性 / 拦截点

这一章只保留和 runtime state 直接相关的可见性判断。

### 1. `files` 的可见性取决于 backend 语义

- `StateBackend`：文件内容对后续 step 与 checkpoint 可见，因为它本来就在 `files` channel 里。
- `FilesystemBackend`：文件内容的真源在宿主机文件系统，排障时应先查磁盘路径解析，而不是先查 reducer。
- `CompositeBackend`：先判断这条路径被路由到了哪个 backend，再判断它该不该出现在 state 或长期存储里。

### 2. 运行时拦截点只有两个最重要

- `FilesystemMiddleware.wrap_model_call()`：决定本次模型请求看见哪些 filesystem 工具，尤其是 `execute`
- backend 实现：决定读写、搜索、执行到底落到哪种介质

### 3. 不要把本章扩写成 callback / stream 理论

如果你现在关心的是传播、stream consumer 可见性、或者 callback tree 的形状，而不是本章的运行时职责，请跳到 Part 3。

## 扩展接口

围绕本章主题，真正稳定的扩展入口只有这些：

### 1. 换 backend

- 需要 thread-scoped 工作区：用 `StateBackend`
- 需要真实文件系统：用 `FilesystemBackend`
- 需要按路径拆多种介质：用 `CompositeBackend`

### 2. 用 backend factory 接运行时

`FilesystemMiddleware`、`MemoryMiddleware` 等都允许通过 runtime-aware factory 解析 backend。这里的关键不是“少写一层配置”，而是让 backend 决策能读到 `ToolRuntime` 当前上下文。

### 3. 预填 state-native 文件

如果你故意选择 `StateBackend`，预填文件的正确入口是：

- `agent.invoke({"messages": [...], "files": {...}})`

这属于 graph state 初始化，而不是宿主机文件预置。

### 4. 扩路径路由，而不是扩散状态语义

想把 `/memories/`、`/artifacts/`、普通工作区拆开时，优先扩 `CompositeBackend.routes`。不要试图靠 prompt 文案或 tool 描述去模拟“不同介质”的事实。

## 常见问题与排障入口

- 文件明明写成功，下一次调用却看不到：先确认是不是 `StateBackend`，再确认你看的是否是同一 thread / node boundary 之后的状态。
- 默认 agent 为什么没有读到本地磁盘文件：因为默认 backend 是 `StateBackend()`，除非你显式传了 `FilesystemBackend`。
- `root_dir` 明明设了，为什么还能访问目录外路径：先看 `virtual_mode`，默认 `virtual_mode=False` 并不提供安全边界。
- `/memories/...` 为什么没有长期保存：先看 backend 路由；只有对应路径真的被路由到长期 backend 时，它才不是 thread-local state。
- `execute` 为什么有时出现有时消失：查 `FilesystemMiddleware.wrap_model_call()` 和 backend 是否满足 `SandboxBackendProtocol`。
- 想排查通用 tool 描述、tool schema、tool-return surface：转去 [第5章：Tools 作为 Runtime Surface](./05-tools-as-runtime-surface.md)。

## 本章结论

- 谁提供：`FilesystemMiddleware` 提供默认 filesystem capability，`ToolRuntime` 提供执行期载体，backend 提供真实读写介质，`FilesystemState.files` 提供 graph 内部文件状态面。
- 如何传播：默认情况下文件经由 `ToolRuntime -> StateBackend -> files channel -> reducer` 进入线程状态；换成 `FilesystemBackend` 或 `CompositeBackend` 后，真实内容则由 backend 持有。
- 修在哪层：看不到文件、写入时序、checkpoint 语义先修 `StateBackend` / `FilesystemState.files`；路径解析与宿主机读写先修 `FilesystemBackend`；多介质路由先修 `CompositeBackend`；`execute` 可见性先修 `FilesystemMiddleware` 的 backend 能力判断。
