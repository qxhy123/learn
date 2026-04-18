# Pregel 运行时专题深化设计

- 日期：2026-04-18
- 状态：已完成设计确认，待进入实施规划
- 适用范围：`deepagents-internal-tutorial/` 目录内的教程文档

## 1. 背景

当前教程已经能多次把读者引到 LangGraph Pregel 相关源码入口，例如：

- `StateGraph.compile()`
- `Pregel.stream()` / `astream()`
- `_defaults()`
- `SyncPregelLoop` / `AsyncPregelLoop`
- `StreamMessagesHandler`
- checkpoint / channel / reducer / `ToolRuntime`

但这些内容更多是“定位式引用”，还没有形成一条独立、系统、可维护的运行时主线。结果是：

1. Pregel 被频繁提到，但没有稳定主章。
2. 第 2 章、第 9 章、第 11 章分别承担了一部分运行时解释，职责边界不够清楚。
3. 第 4 章和第 5 章当前更偏 filesystem / tools 专题，尚未成为 LangGraph runtime 的核心教学章节。
4. 维护者能知道“应该去看 Pregel”，但很难直接回答：
   - 一次 superstep 到底发生了什么。
   - 某次写入为什么不是立刻对全局可见。
   - `ToolRuntime` 在哪个执行阶段注入。
   - `stream()` / `astream()` 为什么只是 Pregel 的暴露面，而不是另一套执行引擎。

用户希望在不新增 Part、也不把教程拆成全新 Pregel 专题的前提下，把 Pregel 运行时内容讲得更深入、更系统。

## 2. 设计目标

### 2.1 主目标

1. 把 Pregel 从“被频繁引用的后台机制”提升为 Part 2 的运行时主线。
2. 让第 4 章成为 Pregel 执行模型主章。
3. 让第 5 章成为 Pregel 主执行路径主章。
4. 让第 2 章只保留三层边界与源码入口，不再承载详细 Pregel 执行教学。
5. 让第 9 章和第 11 章只承接 Pregel 对传播与流可见性的影响，而不重复解释执行主线。
6. 让维护者可以通过教程稳定回答 Pregel 的执行模型、主路径、持久化边界和流输出边界。

### 2.2 非目标

1. 不新增独立 Part 或单开“Pregel Part”。
2. 不把教程改成 LangGraph API 手册。
3. 不逐文件平铺 Pregel 源码实现细节。
4. 不修改 `deepagents/`、`langgraph/`、`langchain/` 源码仓库。
5. 不改变现有四 Part 总结构。

## 3. 已确认的设计决策

本次设计已经通过对话确认了以下选择：

1. 范围选择：做一整套 Pregel 专题深化，但主轴先抓执行模型和主执行路径。
2. 结构选择：不新增结构，直接在现有章节内增强。
3. 主承载章选择：Pregel 主线放在第 4 章和第 5 章；第 2 章只保留边界与入口图。
4. 叙事方式选择：采用混合方式。
   - 第 4 章先讲执行模型。
   - 第 5 章再讲代码路径和 runtime / tool 接缝。

## 4. 方案比较与推荐

### 4.1 方案 A：轻量增强

做法：

- 只在现有 Pregel 提及处补充少量解释。
- 主要扩写第 4、5、9、11 章的局部段落。

优点：

- 改动小。
- 风险低。

缺点：

- Pregel 仍然只是“被引用的背景机制”。
- 不足以解决“缺少主章”的系统性问题。

### 4.2 方案 B：重源码路径

做法：

- 把第 4、5 章改成几乎完全沿源码调用链讲解。

优点：

- 排源码路径很直接。

缺点：

- 抽象模型容易被代码细节淹掉。
- 更像源码导读，不像教程。

### 4.3 方案 C：双章分工的系统化增强

做法：

- 第 4 章承载 Pregel 执行模型。
- 第 5 章承载 Pregel 主执行路径。
- 第 2 章回收到边界与入口。
- 第 9、11 章只承接 Pregel 对传播与流可见性的影响。

优点：

- 模型和代码路径都有稳定主章。
- 不破坏现有结构。
- 能自然衔接 propagation、streaming、subagent 等现有章节。

缺点：

- 需要跨多章同步重写职责和交叉引用。

### 4.4 推荐方案

采用方案 C。

原因：

- 它最符合用户已确认的 `不加新结构 + 第4/5章承载主线 + 模型与代码路径分章` 组合。
- 它能把 Pregel 教学从“局部增强”升级成“现有结构内的真正主线”。

## 5. 章节职责重分配

### 5.1 第 2 章：仓库地图与包边界

职责：

- 保持三层边界图和源码入口地图。
- 回答 Pregel 在 LangGraph 层中的 ownership 和入口位置。
- 明确 `StateGraph.compile()` 不是执行本体。

不再承担：

- `_defaults()` 详细过程。
- loop / runner 详细职责。
- `Pregel.stream()` / `astream()` 主路径讲解。

交叉引用策略：

- Pregel 执行模型统一回跳第 4 章。
- Pregel 主执行路径统一回跳第 5 章。

### 5.2 第 4 章：Filesystem 与状态模型

职责：

- 变为 Pregel 执行模型主章。
- 用 `files` channel 解释 Pregel state model。

保留：

- filesystem / backend 相关内容。

重排原则：

- 先 runtime state model。
- 再 filesystem 作为具体例子。

### 5.3 第 5 章：Tools 作为 Runtime Surface

职责：

- 变为 Pregel 主执行路径主章。
- 解释 compile 之后如何进入真正执行。

保留：

- tool runtime、`ToolRuntime`、`task`、result return surface。

重排原则：

- 先 Pregel 主路径。
- 再把 tools / runtime injection 放回主路径里解释。

### 5.4 第 9 章：传播层总览与四条线

职责：

- 保留传播四线框架。
- 明确四条线与 Pregel runtime 的对应关系。

不再承担：

- Pregel 执行主线教学。

### 5.5 第 11 章：Streaming、Visibility 与 Selective Exposure

职责：

- 聚焦 `stream_mode`、`messages` / `updates` / `custom`、`subgraphs`、`nostream`。
- 解释这些流输出面如何挂在 Pregel runtime 上。

不再承担：

- Pregel 执行模型本身。

### 5.6 附录

- 附录 D 增加对第 4 / 5 章的明确回跳。
- 附录 B 增加 Pregel state model / execution path 相关验证线。

## 6. 第 4 章设计

第 4 章改造后的核心目标是回答：

> Pregel 是怎么“想”的，也就是 channel、task、writes、reducer、barrier、checkpoint 这些对象如何组成执行模型。

### 6.1 推荐内部结构

1. 为什么这一章先讲 Pregel state model
2. Pregel 的最小执行对象：channel、task、pending writes、reducer
3. superstep 与 barrier：什么时候状态才进入下一轮可见面
4. checkpoint 记录什么，不记录什么
5. `files` channel 为什么是理解 Pregel state model 的最佳例子
6. `StateBackend`、`FilesystemBackend`、`CompositeBackend` 分别接在哪一层
7. 维护者最容易误判的四种“状态已经提交”

### 6.2 本章要解决的核心误判

1. 把 state 当成普通 dict，而不是 reducer 驱动的 channel 集合。
2. 把某次 tool / node 返回误认为“已经对全局立刻可见”。
3. 把 callback / stream 事件误认为 checkpoint 语义。
4. 把 consumer 当前可见结果误认为下一 step 已提交状态。

### 6.3 与现有 filesystem 内容的关系

本章不会删除 filesystem 主题，但要改变其位置：

- filesystem 不再是主角。
- 它变成解释 Pregel state model 的最佳案例。

换句话说：

- 这一章不再是“先讲文件，再顺带提到 runtime”。
- 而是“先讲 Pregel state model，再用 `files` channel 和 backend 路径把抽象模型落地”。

## 7. 第 5 章设计

第 5 章改造后的核心目标是回答：

> Pregel 是怎么“跑”的，也就是从 compile 之后如何进入 `_defaults()`、loop、runner、runtime injection、output / stream。

### 7.1 推荐内部结构

1. 为什么 tool runtime 必须放回 Pregel 主路径里理解
2. 从 `StateGraph.compile()` 到 Pregel：compile 固化了什么
3. `Pregel._defaults()` 在运行前装配了什么
4. `SyncPregelLoop` / `AsyncPregelLoop` 如何推进 step
5. `PregelRunner` 如何把 node、tool、subgraph 变成可执行 task
6. `Runtime` / `ToolRuntime` 是在哪个阶段注入的
7. tool output、state update、stream output 分别从哪条路径出来
8. Deep Agents 的 tool / subagent / backend 装配插在 Pregel 路径的哪一段

### 7.2 本章要解决的核心误判

1. 把 `StateGraph.compile()` 当成执行本体。
2. 把 `stream()` / `astream()` 当成独立执行引擎。
3. 把 `ToolRuntime` 当成 Deep Agents 自己的上下文系统。
4. 把 output surface、stream surface、result return surface 混成一条线。

### 7.3 与现有 tools 内容的关系

本章不会删除 `Tools as Runtime Surface` 主题，但会改变解释顺序：

- tools 不再孤立存在。
- tools 会被放进 `PregelRunner` 与 `ToolRuntime` 的执行路径中理解。

## 8. 第 9 章与第 11 章的承接策略

### 8.1 第 9 章

新增桥接内容：

- 四条线与 Pregel runtime 的对应关系。

明确回跳：

- 执行线的 step / task / runner 背景回第 4 / 5 章。
- 观测线主要回第 10 章。
- 流输出线主要回第 11 章。
- 结果折返线要同时回看第 4 / 5 / 7 / 12 章。

### 8.2 第 11 章

新增桥接内容：

- `stream()` / `astream()` 为什么是 Pregel 暴露面，而不是另一套执行引擎。
- `messages` / `updates` / `custom` 分别挂在 Pregel 的哪一层。

明确限制：

- 第 11 章不再重复解释 Pregel 执行模型本身。

## 9. 内容迁移与删减规则

### 9.1 从第 2 章迁出的内容

- `_defaults()` 细节
- `SyncPregelLoop` / `AsyncPregelLoop` 详细职责
- `Pregel.stream()` / `astream()` 详细路径
- 过长的 checkpoint / runtime 注入展开

### 9.2 从第 9 章迁出的内容

- 对执行面本身的主线解释

### 9.3 从第 11 章收紧的内容

- 保留流输出面与可见性。
- 收紧对执行路径本体的解释。

## 10. 交叉引用规则

为避免再次出现“Pregel 到处都提一点，但没有主章”的问题，本次改造要求统一交叉引用策略。

### 10.1 优先回第 4 / 5 章的主题

- graph runtime
- superstep / barrier
- reducer / pending writes
- checkpoint step boundary
- subgraph 执行背景
- `Runtime` / `ToolRuntime`
- state update 与 result 折返的执行背景

### 10.2 优先回第 10 章的主题

- callback / config
- `ensure_config()`
- `patch_config()`
- callback tree / run tree

### 10.3 优先回第 11 章和附录 D 的主题

- `messages` / `updates` / `custom`
- `nostream`
- selective visibility
- stream consumer 可见性

### 10.4 优先回第 7 / 12 章的主题

- `task` handoff
- subagent result return
- parent-child return surface
- subagent 类型差异

## 11. 维护风险与改写边界

### 11.1 风险

1. 第 4 章和第 5 章可能变得过重。
2. 第 2 章、第 9 章、第 11 章可能残留旧职责，导致重复。
3. 如果只补 Pregel 术语、不改章节分工，系统性问题不会真正解决。

### 11.2 控制策略

1. 第 4 章只承载执行模型，不平铺完整代码路径。
2. 第 5 章只承载主路径，不重复讲抽象模型。
3. 第 2、9、11 章只保留边界、传播、流输出方面的承接内容。
4. 通过交叉引用规则强制形成单一主章。

## 12. 测试与验收标准

### 12.1 结构验收

1. 第 4 章能单独回答 Pregel 执行模型。
2. 第 5 章能单独回答 Pregel 主执行路径。
3. 第 2、9、11 章不再重复承载 Pregel 主线。

### 12.2 内容验收

教程必须明确区分：

1. Pregel state / writes / reducer
2. LangChain callback / config / run tree
3. LangGraph stream surface
4. Deep Agents 的 tool / subagent / backend 装配

教程必须明确解释：

1. 为什么 `StateGraph.compile()` 不是执行本体。
2. 为什么 checkpoint 记录 channel snapshot / pending writes，而不是 callback 流。
3. 为什么 `stream()` / `astream()` 是 Pregel 暴露面，而不是另一套执行引擎。
4. 为什么 `ToolRuntime` / `Runtime` 属于 Pregel 执行路径注入，而不是 Deep Agents 私有上下文系统。

### 12.3 可追源码验收

读者按教程回跳，应能稳定落到这些入口：

- `langgraph/.../pregel/main.py`
- `langgraph/.../pregel/_loop.py`
- `langgraph/.../pregel/_runner.py`
- `langgraph/.../pregel/_messages.py`

### 12.4 维护者视角验收

读者至少能独立回答以下五个问题：

1. 一次 Pregel superstep 到底发生了什么。
2. 某次 tool / node 写入为什么不是立刻对下一处读取全局可见。
3. `ToolRuntime` 是在哪个执行阶段注入的。
4. 为什么能看到 token 流，但这不等于 checkpoint 或最终 state 已提交。
5. 某个问题应先修 Pregel runtime、LangChain callback/config，还是 Deep Agents 装配层。

### 12.5 文档级验证

1. 跑相对链接检查。
2. 跑旧职责残留扫描。
3. 人工抽查：
   - 第 2 章
   - 第 4 章
   - 第 5 章
   - 第 9 章
   - 第 11 章
   - 附录 D

## 13. 进入实施规划前的边界

这份 spec 只定义：

- 章节职责
- 重写范围
- 内容迁移规则
- 交叉引用规则
- 验收标准

它不直接展开到逐段改写顺序、批次划分、验证命令和提交拆分。那些内容应在下一步 implementation plan 中单独细化。
