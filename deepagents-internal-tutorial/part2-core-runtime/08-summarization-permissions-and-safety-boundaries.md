# 第8章：Summarization、Permissions 与安全边界

## 本章回答什么

- summarization 在 Deep Agents 里承担的是哪种 compaction 策略，而不是什么“通用传播层”
- permissions 为什么首先是 tool policy，与真正执行环境安全边界是什么关系
- 为什么“流里看不见”不等于“系统安全了”，也不等于“结果不会回传”
- `CompiledSubAgent` 与 `AsyncSubAgent` 为什么天然是 permissions 的边界外侧
- 维护者在设计安全边界时，应该把本地权限规则、远端执行环境、结果回传面分别放在哪层修

## 在整套系统中的位置

- 横向主题：`Compaction`、`Safety Boundary`
- 前置章节：[README](../README.md)、[前言：如何使用本教程](../00-preface.md)、[第4章：Filesystem 与状态模型](./04-filesystem-and-state-model.md)、[第5章：Tools 作为 Runtime Surface](./05-tools-as-runtime-surface.md)、[第7章：Subagents、任务交接与上下文隔离](./07-subagents-and-context-isolation.md)
- 后续章节：Part 3 会单独讨论传播、callbacks、streaming、可见性与 recipes

这一章只讨论两件事：系统怎样压缩上下文，系统又在哪些本地工具面上设置权限与安全边界。stream mode 总表、callback tree 细节、可见性矩阵不再放在这里。

## 静态结构

建议同时打开这些实现文件：

- `deepagents/libs/deepagents/deepagents/graph.py`
- `deepagents/libs/deepagents/deepagents/middleware/summarization.py`
- `deepagents/libs/deepagents/deepagents/middleware/permissions.py`
- `deepagents/libs/deepagents/deepagents/middleware/subagents.py`
- `deepagents/libs/deepagents/tests/unit_tests/middleware/test_summarization_factory.py`
- `deepagents/libs/deepagents/tests/unit_tests/middleware/test_summarization_middleware.py`
- `deepagents/libs/deepagents/tests/unit_tests/test_permissions.py`

### summarization 与 permissions 的静态分工

| 主题 | 主要实现位置 | 本章关心的边界 |
| --- | --- | --- |
| summarization | `middleware/summarization.py` | 什么时候对长上下文做本地压缩，压缩后保留什么运行时价值 |
| permissions | `middleware/permissions.py` | 哪些本地工具调用需要经过 policy 收口，哪些行为不在它的自动防护面内 |
| child return filtering | `middleware/subagents.py` | 安全边界不只看“能不能调用工具”，还要看结果会不会折返到 parent |

### 两条最容易混写的线

| 线 | 正确理解 |
| --- | --- |
| 可见性线 | 某些 token、事件、更新是否被外部消费者看到 |
| 安全边界线 | 某次工具调用、文件访问、远端执行是否真的受当前 policy 约束 |

维护时必须把它们分开。可见性降低不等于安全边界就成立。

## 运行时链路

### 1. summarization 是本地 compaction 策略

Deep Agents 把 summarization 当成 harness 里的上下文压缩器，用来处理：

- 长线程导致的消息膨胀
- 多轮执行后的上下文占用
- 需要保留工作结论、但不必保留全部原始消息明细的场景

因此它的职责是：

- 在合适节点把对后续推理仍有价值的信息压成更短的表示
- 降低主线程或长任务继续运行时的上下文负担

它的职责不是：

- 定义 callback/stream 的传播规则
- 代替权限策略决定谁能调用危险工具
- 自动成为“私有推理隔离层”

### 2. permissions 是本地 tool policy

`_PermissionMiddleware` 的核心任务，是在当前 harness 暴露出来的工具面上加一层规则判断。实际主战场通常是文件类工具，例如：

- `ls`
- `read_file`
- `write_file`
- `edit_file`
- `glob`
- `grep`

这里要抓住两个事实：

- permissions 管的是“当前工具面允许什么调用”
- permissions 不是执行环境本身

也就是说，policy 决定“能不能尝试做”，而 sandbox / backend / remote runtime 决定“实际上在哪个环境里做、还能做出什么副作用”。

### 3. 安全边界还要再经过 return surface

即使某个工具调用本身被限制住了，维护者仍然要看另一条线：

- 哪些中间结果会进入共享 state
- 哪些结果会通过 `ToolMessage` 或 update 回到 parent

这就是为什么第7章和本章要连着看：

- 第7章负责说明 child 结果怎样被压缩回传
- 本章负责说明哪些本地 policy 和 compaction 策略在这之前/之后生效

### 4. compiled 与 async 是 permissions 的天然边界外侧

`CompiledSubAgent` 与 `AsyncSubAgent` 都提醒我们：顶层 permissions 不是万能外壳。

- `CompiledSubAgent`：顶层不会自动深入 runnable 内部再套一层同样的 middleware 栈
- `AsyncSubAgent`：本地顶层最多约束“是否发起 delegation”，远端工具权限和执行环境必须由远端系统定义

因此，看到“顶层配了 permissions”时，维护者应该立刻追问：

- 这次执行到底发生在本地图工具面、compiled runnable 内部，还是远端 async runtime？

### 5. safety 不等于 visibility suppression

如果你只是让某些内部输出不对外可见，那只是缩小观测面，不等于建立了真正的安全边界。

真正的边界至少要分别检查：

- 工具调用是否被 policy 收口
- 执行环境是否可信、是否有 sandbox
- 中间结果是否写入了共享 state
- 最终结果是否仍会回传给 parent 或用户

## 传播 / 可见性 / 拦截点

这一节只保留和安全边界直接相关的判断，不再承担 streaming 教程。

### 1. summarization 影响的是保留量，不是传播协议

摘要压缩会改变后续模型调用看到的上下文量，但它本身不是“消息传播开关”。它回答的是：

- 哪些旧信息还值得保留
- 以多短的形式保留

而不是：

- callback tree 怎么长
- stream consumer 为什么能或不能看到某个 token

### 2. permissions 拦截的是受管工具面

当某个文件工具、受管工具或 policy 覆盖的调用进入执行时，`_PermissionMiddleware` 可以成为拦截点。但它只对自己真正包裹到的工具面负责。

### 3. “不可见”不等于“不会回传”

维护者最容易误判的是这一点：

- 某个中间过程对外部观察者不可见
- 不代表它没有进入 child state
- 也不代表它不会在结束时被压缩成结果回到 parent

如果你现在关心的是传播、stream consumer 可见性、或者 callback tree 的形状，而不是本章的运行时职责，请跳到 Part 3。

## 扩展接口

### 1. 调整 summarization 策略

- 改 `middleware/summarization.py` 中的摘要触发时机、保留字段、摘要格式
- 先明确你是在优化上下文成本，还是在修结果保真度
- 不要把“我想减少可见性”直接翻译成“去改 summarization”

### 2. 调整 permissions policy

- 改 `middleware/permissions.py` 中对本地工具面的 allow/deny 规则
- 明确规则作用的是哪些工具、哪些路径、哪些 artifact
- 若需求指向的是执行环境隔离，而不是工具级 allow/deny，就不该只改 permissions

### 3. 给 compiled 子代理补本地安全规则

- 在 compiled runnable 自己那层补 middleware / guardrail / sandbox 约束
- 不要依赖顶层 permissions 自动穿透 child runnable

### 4. 给 async 子代理定义远端边界

- 在远端 agent/runtime 上定义自己的审批、权限、执行环境与结果回传规则
- 本地顶层章节只负责 delegation 入口与结果接回，不负责远端内部安全

## 常见问题与排障入口

- 上下文太长，为什么不是去改 permissions：因为这是 compaction 问题，先看 summarization，而不是安全策略。
- 顶层 permissions 已经配了，为什么 compiled 子代理里还能做危险操作：因为顶层 policy 没有自动深入 runnable 内部；应修 compiled runnable 自己的装配。
- async 子代理为什么还能在远端访问本地看不到的能力：因为远端执行环境是另一层系统；要检查远端 agent/runtime 的权限模型。
- 已经把中间过程“藏起来”了，为什么 parent 还是知道结果：因为结果可能仍通过 `ToolMessage` 或 state update 回传；继续查第7章的 return surface。
- 某些文件工具为什么被挡住了：先查 `_PermissionMiddleware` 的规则，再查工具是否真的由当前 harness 注入。
- 改了 summarization 后结果变短但信息丢失：先查摘要触发时机和保留字段，而不是先怀疑 permissions。

## 本章结论

- 谁提供：`summarization.py` 提供本地上下文压缩策略，`permissions.py` 提供本地工具面的 policy 收口，child 返回过滤逻辑与真实结果面还要结合 `subagents.py` 一起看。
- 如何传播：summarization 改变的是后续调用可用的上下文表示，permissions 约束的是受管工具调用是否被允许；真正的结果暴露还会经过 state 写入与 parent return surface。
- 修在哪层：上下文膨胀与摘要保真度问题修 summarization；本地文件/工具访问规则修 permissions；compiled/async 子代理的内部或远端安全边界修各自 runnable / runtime，而不是假设顶层 policy 自动兜底。
