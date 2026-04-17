# 第8章：Backend 协议、存储介质与执行边界

## 学习目标

学完本章，你应该能回答：

1. `BackendProtocol` 在三层栈里真正抽象的是什么
2. `StateBackend` 为什么是 graph-native adapter，而不是普通文件插件
3. `CompositeBackend`、`SandboxBackendProtocol`、permissions 分别解决什么问题
4. 新需求应该落在 backend、middleware，还是上游 runtime / tool 层
5. 如果你要新增一个 backend，最小实现和最小验证集应该是什么

---

## 问题是什么

第一次看 Deep Agents 的文件系统能力，很容易把它理解成“给 agent 多挂了几个读写文件的工具”。

但真实情况更复杂。这里至少叠了三层：

- `LangChain` 负责 tool primitive、schema、callback/config 传播
- `LangGraph` 负责 state channel、checkpoint、step 边界和 `ToolRuntime`
- `Deep Agents` 才负责把这些能力装成 backend + middleware + permissions 的默认 harness

所以 backend 不是“另一个存储 SDK”，而是 Deep Agents 用来把运行介质接到上游工具协议上的 adapter 层。

---

## 哪一层负责什么

### `LangChain`

- `BaseTool.run()` / `arun()` 负责 tool call 的生命周期
- `patch_config(...callbacks=run_manager.get_child())` 与 `set_config_context(...)` 负责 child run config
- tool 参数 schema、异常到 `ToolMessage` 的转换规则首先属于这层

### `LangGraph`

- `files` 这类 state channel 的生命周期
- `CONFIG_KEY_READ` / `CONFIG_KEY_SEND` 这种图内读写入口
- checkpoint 后的 thread-scoped 持久化语义
- `ToolRuntime.state`、`ToolRuntime.context`、`ToolRuntime.stream_writer`

### `Deep Agents`

- `BackendProtocol` / `SandboxBackendProtocol`
- `StateBackend`、`StoreBackend`、`CompositeBackend`、本地文件系统 backend
- `FilesystemMiddleware` 如何把 backend 暴露给模型
- `_PermissionMiddleware` 如何对部分 tool surface 做收口

---

## 代码在哪里

建议同时打开：

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

---

## 实现怎么工作

### 1. `BackendProtocol` 抽象的是“运行介质”，不是“文件 API 长得像什么”

`protocol.py` 里定义的不只是 `read` / `write` / `ls`。

它还统一了：

- 返回对象形状
- recoverable error 的表达方式
- 文件上传/下载、搜索、编辑这类模型会直接消费的结果
- 可选的执行能力边界

这层设计的真正价值是：

> 上层 middleware 不必知道底层到底是 graph state、真实磁盘、远端沙箱，还是它们的组合。

### 2. `StateBackend` 不是 mock backend，而是默认的 graph-native backend

`StateBackend` 当前实现最值得看的不是文件读写逻辑本身，而是它怎样进入 LangGraph runtime：

- `_read_files()` 通过 `get_config()` 取当前图执行上下文
- 再用 `CONFIG_KEY_READ` 读取 `files` channel
- `_send_files_update()` 通过 `CONFIG_KEY_SEND` 把部分更新排进当前 step 的 channel write 队列

这意味着两个关键事实：

- 它的持久化范围默认是 thread/checkpoint 语义，不是机器磁盘语义
- 同一个 step 里读到的是一致快照，写入要到 node boundary 后才会生效

所以 `StateBackend()` 的核心不是“省得配路径”，而是“默认把文件状态跟 graph state 生命周期绑在一起”。

### 3. backend 本身不是模型可见面，middleware 才是

模型不会直接调用 `StateBackend.read()`。

真正的链路是：

1. `Deep Agents` 在 `FilesystemMiddleware` 中注入读写/搜索/编辑等工具
2. 这些工具沿着 `LangChain` 的 `BaseTool.run()` / `arun()` 执行
3. 工具内部再调用 backend
4. backend 再通过 `LangGraph` runtime 读写 state 或外部介质

所以你如果遇到：

- callback tree 不对
- tags / metadata 丢了
- tool runtime 里的 context 不对

不要先怀疑 backend。那通常先看上游 tool/runtime 链路。

### 4. `CompositeBackend` 的本质是 path routing，不是简单代理

`CompositeBackend` 做了几件很维护者导向的事：

- 按最长前缀匹配 route
- 把外部路径剥掉 route prefix 再传给内部 backend
- 把内部 backend 返回的路径重新映射回外部视图
- 在根目录 `"/"` 下把虚拟 route 目录聚合回统一列表

这说明它不是“多个 backend 放一起”这么简单，而是在维护一层虚拟文件视图。

典型用法是：

- 默认文件放 graph state
- `/memories/` 放长期存储
- `/artifacts/` 放另一种介质

### 5. 执行能力是更窄的一层 contract

并不是每个 backend 都支持 `execute`。

更准确的理解是：

- `BackendProtocol` 解决通用文件与搜索 contract
- `SandboxBackendProtocol` 才额外描述执行能力

这也是为什么 permissions 和 execution 要分开理解：

- permissions 决定模型能否调用某些工具
- backend / sandbox contract 决定该运行介质有没有这个能力

---

## 一张 backend capability matrix

| Backend | 主要介质 | 持久化语义 | 典型用途 | 常见风险 |
|---------|----------|------------|----------|----------|
| `StateBackend` | LangGraph state channel | thread / checkpoint | 默认工作区、短生命周期文件 | 误当成真实磁盘；忽略 step boundary |
| `StoreBackend` | `BaseStore` namespace | 长期存储 | memories / cache / artifacts | 把 store 语义误当成工作目录 |
| `FilesystemBackend` | 本地磁盘 | 机器级持久化 | content builder、真实工作目录 | 不是 sandbox；路径与权限要小心 |
| `CompositeBackend` | 多 backend 组合 | 取决于 route | 混合工作区、分区数据 | route 设计混乱导致路径视图不一致 |
| `LocalShellBackend` | 宿主机 shell | 宿主机副作用 | 本地执行实验 | 不是安全边界 |

---

## backend cookbook

### 场景 1：我只是想换存储介质

优先改 backend，而不是 middleware。

因为你想保留的是：

- 上层工具 contract
- 模型可见的工具名和描述
- parent-child handoff 语义

你真正想换的是“这些工具背后的介质”。

### 场景 2：我想加一个新的 route 目录

优先看 `CompositeBackend`，不是直接在 prompt 里教模型“以后把文件写到某个目录”。

更稳的做法是：

1. 设计虚拟外部路径
2. 选内部 backend
3. 定 route prefix
4. 验证 path remap 后，外部视图仍统一

### 场景 3：我想多暴露一个新工具

通常先看 middleware，而不是 backend。

因为 backend 解决的是介质，tool surface 解决的是模型可见面。

### 场景 4：我想让 execute 更安全

先分清你在改哪一层：

- 只是限制谁能点 execute：permissions / policy 层
- 真正让执行环境更安全：sandbox/backend contract 层

---

## 一个最小 backend skeleton

下面这个 skeleton 不是完整实现，而是维护者判断“我是不是该写 backend”的最小模板：

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

如果你还需要执行能力，再单独考虑是否要实现 `SandboxBackendProtocol`，而不是默认把 execute 混进普通 backend。

---

## 决策树：改 backend、middleware，还是上游

| 你要改的东西 | 优先落点 |
|--------------|----------|
| 真实存储介质 / 持久化策略 | backend |
| 虚拟路径路由 | `CompositeBackend` |
| 模型能看到哪些文件工具 | `FilesystemMiddleware` / 装配层 |
| tool callback / child config 传播 | 上游 `langchain_core` |
| `ToolRuntime` 注入与 graph step 边界 | 上游 `langgraph` |
| execute 运行环境安全 | sandbox/backend contract |
| 文件工具是否允许被调用 | permissions / policy |

---

## 新增 backend 的最小验证集

### unit

- path 正常化
- `ls/read/write/edit/glob/grep` 返回形状
- 错误对象 / recoverable error 形状
- route remap（如果有）

### integration

- 至少通过 `FilesystemMiddleware` 跑一条真实工具链
- 如果有 execute，至少验证一次成功路径和一次拒绝/错误路径

### policy

- 和 `_PermissionMiddleware` 组合后是否还能正确裁剪文件结果
- 在 `CompositeBackend` 下是否会错误地把 route 外的数据暴露给默认视图

### contract

- 如果实现了 `SandboxBackendProtocol`，考虑对照标准 sandbox tests 的形状

---

## 什么时候该修上游

### 更像上游问题

- `BaseTool.run()` 的 config patch 行为和预期不一致
- `ToolRuntime` 没把 state/context/config 注进工具
- checkpoint / state channel 的一致性和文档不符

### 更像 Deep Agents 本地问题

- `StateBackend` / `CompositeBackend` 的 contract 设计不合理
- 默认文件工具暴露过多或过少
- permissions 与 backend 路由规则组合后出现策略漏洞

---

## 容易踩什么坑

- 坑 1：把 backend 当成“存储插件”。
  实际上它承担的是运行介质 adapter 角色。

- 坑 2：在 middleware 里偷偷依赖某个 backend 的私有细节。
  这样其他 backend 就无法复用同一套工具 surface。

- 坑 3：把 `StateBackend` 看成临时 mock。
  它恰恰是默认设计中心，而不是测试替身。

- 坑 4：因为模型能看到某个工具，就默认所有 backend 都该支持它。
  execution capability 必须和 file capability 分开设计。

---

## 本章小结

- backend 是 Deep Agents 的运行介质 adapter，不是独立 runtime。
- `StateBackend` 默认站在 LangGraph state / checkpoint 之上工作。
- `CompositeBackend` 提供的是统一虚拟路径视图，而不只是多 backend 拼接。
- backend 决定“能力在哪个介质上发生”，middleware 决定“模型看见怎样的工具面”。
