# 第13章：Backend 协议、存储介质与执行边界

## 本章回答什么

- `BackendProtocol` 在三层栈里真正抽象的是什么，为什么它是维护工作流的第一个边界判断
- `StateBackend` 为什么是 graph-native storage adapter，而不是“临时文件插件”
- `CompositeBackend`、`SandboxBackendProtocol`、permissions 分别收口哪种能力边界
- 一个问题应该修在 backend、middleware，还是上游 runtime / tool 层
- 新增 backend 时，最小实现和最小验证集应该长什么样

## 在整套系统中的位置

- 这一部分默认假设你已经读过 Part 1 和 Part 2。
- 如果当前问题和传播、可见性、callback tree 有关，先回看 Part 3。
- 横向主题：`Maintenance`、`Storage`、`Execution boundary`
- 前置章节：[第4章：Filesystem 与 State Model](../part2-core-runtime/04-filesystem-and-state-model.md)、[第5章：Tools 作为 Runtime Surface](../part2-core-runtime/05-tools-as-runtime-surface.md)、[第8章：Summarization、Permissions 与 Safety Boundaries](../part2-core-runtime/08-summarization-permissions-and-safety-boundaries.md)
- 后续章节：[第14章：Provider Profiles、模型解析与 Middleware Surface](./14-provider-profiles-and-model-routing.md)、[第15章：如何测试一个三层栈 Harness](./15-testing-the-harness.md)

Part 4 进入维护工作流后，第一步不是立刻跑测试，而是先确认问题到底出在什么介质、什么边界、什么 contract 上。本章就是这一步的入口：先把 backend 当成运行介质 adapter 来看，再决定后面是该去改 provider 适配、测试回归，还是直接回到上游 runtime。

## 静态结构

建议同时打开这些文件：

- `deepagents/libs/deepagents/deepagents/backends/protocol.py`
- `deepagents/libs/deepagents/deepagents/backends/state.py`
- `deepagents/libs/deepagents/deepagents/backends/store.py`
- `deepagents/libs/deepagents/deepagents/backends/composite.py`
- `deepagents/libs/deepagents/deepagents/backends/filesystem.py`
- `deepagents/libs/deepagents/deepagents/backends/local_shell.py`
- `deepagents/libs/deepagents/deepagents/middleware/filesystem.py`
- `deepagents/libs/deepagents/tests/unit_tests/backends/`
- `langchain/libs/core/langchain_core/tools/base.py`
- `langgraph/libs/langgraph/langgraph/config.py`

先把三层职责拆开看：

| 层 | 这里首先拥有的东西 | 维护时最容易误判成什么 |
| --- | --- | --- |
| `LangChain` | tool primitive、schema、`BaseTool.run()` / `arun()`、callback/config 传播 | “backend 自己决定 tool lifecycle” |
| `LangGraph` | `files` state channel、checkpoint 语义、`CONFIG_KEY_READ` / `CONFIG_KEY_SEND`、`ToolRuntime` | “写文件就是写本地磁盘” |
| `Deep Agents` | `BackendProtocol`、`SandboxBackendProtocol`、`StateBackend`、`CompositeBackend`、`FilesystemMiddleware`、permissions 收口 | “多挂几个文件工具而已” |

静态上还要先记住 backend capability matrix：

| Backend | 主要介质 | 持久化语义 | 典型用途 | 常见风险 |
| --- | --- | --- | --- | --- |
| `StateBackend` | LangGraph state channel | thread / checkpoint | 默认工作区、短生命周期文件 | 把它误当成真实磁盘，忽略 step boundary |
| `StoreBackend` | `BaseStore` namespace | 长期存储 | memories / cache / artifacts | 把 store 语义误当成工作目录 |
| `FilesystemBackend` | 本地磁盘 | 机器级持久化 | content builder、真实工作目录 | 不是 sandbox；路径与权限要单独收口 |
| `CompositeBackend` | 多 backend 组合 | 取决于 route | 混合工作区、分区数据 | route 设计混乱导致视图不一致 |
| `LocalShellBackend` | 宿主机 shell | 宿主机副作用 | 本地执行实验 | 不是安全边界 |

## 运行时链路

### 1. `BackendProtocol` 抽象的是运行介质，不是“文件 API 长什么样”

`protocol.py` 统一的不只是 `read` / `write` / `ls` 这些名字，还包括：

- 返回对象形状
- recoverable error 的表达方式
- 搜索、编辑、上传/下载这类会直接进入模型工具面的结果
- 可选执行能力与普通文件能力之间的边界

它的维护价值在于：

> 上层 middleware 不必知道底层到底是 graph state、真实磁盘、远端沙箱，还是它们的组合。

所以 backend 不是另一个存储 SDK，而是 Deep Agents 把运行介质接到 tool contract 上的 adapter 层。

### 2. `StateBackend` 通过 LangGraph runtime 进入 checkpoint 语义

`StateBackend` 最值得看的不是“读写文件 API 长什么样”，而是它怎样进入图执行上下文：

- `_read_files()` 通过 `get_config()` 取当前图执行上下文
- 再用 `CONFIG_KEY_READ` 读取 `files` channel
- `_send_files_update()` 通过 `CONFIG_KEY_SEND` 把部分更新排进当前 step 的 channel write 队列

这直接带出两个维护判断：

- 它的持久化范围默认是 thread / checkpoint 语义，不是机器磁盘语义
- 同一个 step 里读到的是一致快照，写入通常要到 node boundary 后才会生效

所以 `StateBackend()` 的核心不是“省得配路径”，而是“默认把文件状态绑到 graph state 生命周期上”。

### 3. backend 本身不是模型可见面，`FilesystemMiddleware` 才是

模型不会直接调用 `StateBackend.read_file()`。真实链路是：

1. `Deep Agents` 在 `FilesystemMiddleware` 中注入读写/搜索/编辑等工具
2. 这些工具沿着 `LangChain` 的 `BaseTool.run()` / `arun()` 执行
3. 工具内部再调用 backend
4. backend 最后通过 `LangGraph` runtime 读写 state 或外部介质

这也是维护时最常用的分流点：

- callback tree、tags、metadata、`ToolRuntime.context` 不对，先看 tool/runtime 链路
- 路径视图、存储语义、route 行为不对，再看 backend contract

### 4. `CompositeBackend` 维护的是虚拟路径视图

`CompositeBackend` 的本质不是“多个 backend 简单代理”，而是 path routing：

- 按最长前缀匹配 route
- 传给内部 backend 前先剥掉 route prefix
- 返回给上层前再把内部路径映射回外部视图
- 在根目录 `"/"` 下把虚拟 route 目录聚合成统一列表

这说明它维护的是一层虚拟工作区。典型组合是：

- 默认文件放 graph state
- `/memories/` 放长期存储
- `/artifacts/` 放另一种介质

### 5. 执行能力是更窄的一层 contract

不是每个 backend 都支持 `execute`。更准确的拆法是：

- `BackendProtocol` 解决通用文件与搜索 contract
- `SandboxBackendProtocol` 才额外描述执行能力

这也是为什么 permissions 和 execution 要分开理解：

- permissions 决定模型能否调用某些工具
- backend / sandbox contract 决定该介质有没有这个能力

## 传播 / 可见性 / 拦截点

维护 backend 时，最容易把“存储问题”和“传播问题”混成一句。更稳妥的判断是：

- backend 决定文件状态落在哪个介质、以什么 checkpoint / route 语义存在
- middleware 决定模型看到怎样的工具面
- callback/config/run tree 仍由上游 tool/runtime 传播线负责

几个关键拦截点必须分开：

### `files` channel 的可见性

- `StateBackend` 看到的是 LangGraph `files` channel 的快照与写队列
- 同一 step 内的读写可见性由 graph step boundary 决定，不由 backend 自己决定

### 工具暴露面的可见性

- 模型是否能调用读写/搜索/编辑工具，取决于 `FilesystemMiddleware` 和 permissions
- “模型看得到某个工具”不等于“每个 backend 都必须支持这个能力”

### 执行能力的拦截点

- execute 是否存在，先看是否实现 `SandboxBackendProtocol`
- execute 是否允许模型调用，再看 permissions / policy

### 传播问题不要在 backend 层误修

如果症状是：

- callback tree 不连
- tags / metadata 丢失
- `ToolRuntime.context` 不对

优先回看 [第10章：Callbacks、Config 与 Callback Manager](../part3-propagation/10-callbacks-config-and-callback-manager.md) 到 [第12章：Subagent 传播矩阵与维护者 recipes](../part3-propagation/12-subagent-propagation-matrix-and-maintainer-recipes.md)，不要先在 backend 里补私有传播逻辑。

## 扩展接口

### 决策树：应该改 backend、middleware，还是上游

| 你要改的东西 | 优先落点 |
| --- | --- |
| 真实存储介质 / 持久化策略 | backend |
| 虚拟路径路由 | `CompositeBackend` |
| 模型能看到哪些文件工具 | `FilesystemMiddleware` / 装配层 |
| tool callback / child config 传播 | 上游 `langchain_core` |
| `ToolRuntime` 注入与 graph step 边界 | 上游 `langgraph` |
| execute 运行环境安全 | sandbox/backend contract |
| 文件工具是否允许被调用 | permissions / policy |

### backend cookbook

#### 场景 1：我只是想换存储介质

优先改 backend，而不是 middleware。你想保留的是上层工具 contract 与 handoff 语义，真正变化的是工具背后的运行介质。

#### 场景 2：我想加一个新的 route 目录

优先看 `CompositeBackend`。更稳的做法是：

1. 设计虚拟外部路径
2. 选内部 backend
3. 定 route prefix
4. 验证 path remap 后，外部视图仍统一

#### 场景 3：我想多暴露一个新工具

通常先看 middleware，而不是 backend。backend 解决的是介质，tool surface 解决的是模型可见面。

#### 场景 4：我想让 execute 更安全

先拆清两层：

- 限制谁能点 execute：permissions / policy 层
- 真正让执行环境更安全：sandbox/backend contract 层

### 一个最小 backend skeleton

```python
from deepagents.backends.protocol import BackendProtocol


class MyBackend(BackendProtocol):
    def ls(self, path: str):
        ...

    def read_file(self, path: str):
        ...

    def write_file(self, path: str, content: bytes | str):
        ...

    def edit_file(self, path: str, old: str, new: str):
        ...

    def glob(self, pattern: str):
        ...

    def grep(self, query: str, path: str | None = None):
        ...
```

如果你还需要执行能力，再单独考虑是否实现 `SandboxBackendProtocol`，不要默认把 execute 混进普通 backend。

### 新增 backend 的最小验证集

- unit：路径正常化、`ls/read/write/edit/glob/grep` 返回形状、错误对象 / recoverable error 形状、route remap（如果有）
- integration：至少通过 `FilesystemMiddleware` 跑一条真实工具链；如果有 execute，至少验证一次成功路径和一次拒绝/错误路径
- policy：和 `_PermissionMiddleware` 组合后能否正确裁剪结果；在 `CompositeBackend` 下是否把 route 外的数据错误暴露给默认视图
- contract：如果实现了 `SandboxBackendProtocol`，应对照标准 sandbox tests 的能力形状

## 常见问题与排障入口

- “为什么写进 `StateBackend` 的内容不像真实文件系统那样立刻可见”：先查 `files` channel 的 step boundary 与 checkpoint 语义，不要先怀疑路径实现。
- “为什么模型能看到某个文件工具，但当前 backend 跑不通”：先分清是 tool surface 暴露过度，还是 backend 根本没实现该能力。
- “为什么本地磁盘 backend 看起来像安全边界”：它不是；`FilesystemBackend` 和 `LocalShellBackend` 都不能自动等同于 sandbox。
- “为什么 callback/config 行为像是丢了”：这更像上游 `BaseTool.run()`、`ToolRuntime`、callback/config 传播问题，而不是 backend contract 问题。
- “为什么 permissions 和 route 组合后出现数据泄露风险”：同时检查 `_PermissionMiddleware` 与 `CompositeBackend` 的默认视图，不要只看其中一层。

更像上游问题的症状：

- `BaseTool.run()` 的 config patch 行为和预期不一致
- `ToolRuntime` 没把 state/context/config 注进工具
- checkpoint / state channel 的一致性和文档不符

更像 Deep Agents 本地问题的症状：

- `StateBackend` / `CompositeBackend` 的 contract 设计不合理
- 默认文件工具暴露过多或过少
- permissions 与 backend 路由规则组合后出现策略漏洞

## 本章结论

- 谁提供：`LangChain` 提供 tool lifecycle，`LangGraph` 提供 state / checkpoint / runtime 语义，`Deep Agents` 用 backend 协议和 middleware 把运行介质接进 harness。
- 如何传播：文件状态先沿具体 backend 落到 graph state、store、磁盘或组合视图，再通过 `FilesystemMiddleware` 进入模型可见的工具面；callback/config 传播仍在上游链路里发生。
- 修在哪层：介质与路径问题修 backend，模型暴露面修 middleware / permissions，callback/config 与 step boundary 问题回到 `LangChain` / `LangGraph`。
