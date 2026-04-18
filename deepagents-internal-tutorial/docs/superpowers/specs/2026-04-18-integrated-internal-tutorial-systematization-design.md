# DeepAgents / LangGraph / LangChain Internal 教程系统化改造设计

- 日期：2026-04-18
- 状态：已完成设计确认，待进入实施规划
- 适用范围：`deepagents-internal-tutorial/` 目录内的教程文档

## 1. 背景

当前教程已经具备较强的内容密度，尤其在运行时、subagent、memory、filesystem、streaming 边界等主题上，已经能回答不少维护者问题。但它仍然存在四类系统化不足：

1. 章节承接不够稳定，读起来像一组强单章，而不是一套有教学顺序的课程。
2. 缺少统一分类法，同类概念虽然都有涉及，但没有被压到同一分析框架下。
3. 同类问题分散在不同章节，维护者按问题定位时需要跨章拼接。
4. 深度分布不均衡，少数章节承载了过多机制，导致局部很强、整体不稳。

用户希望把它重构为一套混合型 internal 教程：

- 既能顺着学习。
- 也能按问题快速定位。
- 默认优先服务维护者和扩展者，同时兼顾首次进入三层栈的工程师与已有 API 使用者。

## 2. 设计目标

这次改造的目标不是“补更多内容”，而是把现有内容组织成一个稳定的系统。

### 2.1 主目标

1. 把教程改造成“双导航”结构：
   - 纵向是课程式学习路径。
   - 横向是按机制和问题检索的维护者索引。
2. 为整套教程建立统一分析框架，让每章都回答同一组核心问题。
3. 把 callbacks、callback manager、streaming、visibility、subagent 传播边界从散点内容提升为一条独立主线。
4. 让 README 从“导航页”升级为“系统总图 + 读法入口 + 问题索引”。
5. 让附录承担检索和排障面板职责，而不是零散补充材料。

### 2.2 非目标

1. 不修改 `deepagents/`、`langgraph/`、`langchain/` 源码仓库。
2. 不把教程扩成百科全书，不追求覆盖所有上游细节。
3. 不引入新的产品功能设计；本次只重构教程结构、话语体系、索引和关键机制说明。
4. 不用“完全重写”取代渐进改造；优先复用现有强章节，在骨架层面重组。

## 3. 目标读者与优先级

### 3.1 主读者

- 需要改源码、排运行时问题、补扩展能力的维护者和扩展者。

### 3.2 次读者

- 第一次系统接触 `deepagents / langgraph / langchain` 三层栈的工程师。
- 已经会用公开 API，但希望吃透内部机制的人。

### 3.3 读者优先级策略

教程默认从维护者视角组织：

1. 先能判断行为属于哪一层。
2. 再能判断机制如何传播。
3. 再能判断问题应该修在哪层。

新读者和 API 使用者通过纵向主线获得逐步理解，但整套教程的结构中心仍然是维护者定位、扩展和排障。

## 4. 核心设计原则

### 4.1 双导航是主结构，不是附加说明

教程必须同时提供两套正交导航：

1. 纵向主线：适合顺着学的课程路径。
2. 横向主线：适合按机制和问题检索的主题索引。

这两条导航都必须在首页显式出现，并贯穿每章开头和结尾，而不是只在 README 中出现一次。

### 4.2 每章都必须进入同一分析坐标系

所有章节统一回答以下六个问题：

1. 这层能力由谁提供。
2. 它在运行时如何传播。
3. 哪些状态是显式的，哪些是隐式的。
4. 外部可见面在哪里。
5. 可拦截点和可扩展点在哪里。
6. 出问题时先查哪一层。

### 4.3 传播层必须独立成主线

callbacks、callback manager、streaming、token visibility、subagent 折返不是四个互不相干的话题，而是同一个传播层问题的不同投影。教程必须显式拆出传播层，而不是把这部分内容分散在 subagent、memory、permissions 等章节里。

### 4.4 机制事实与维护建议分开写

每章都要区分：

- 机制事实：代码当前如何工作。
- 维护建议：维护者应该如何修改、排障或扩展。

### 4.5 来源与稳定性要标记出来

重要结论要标注两类标签：

- 来源标签：`LC`（LangChain）、`LG`（LangGraph）、`DA`（Deep Agents）
- 稳定性标签：`Stable mechanism`、`Current implementation`、`Known limitation`、`Test-backed behavior`

这样可以避免把 LangGraph 或 LangChain 层的机制误记为 Deep Agents 私有能力，也能避免把当前实现细节写成稳定契约。

## 5. 统一教程骨架

### 5.1 纵向课程主线

纵向主线回答“如果我要系统学内部机制，应该按什么顺序建立心智模型”。

建议顺序为：

1. 系统总览与边界
2. 组装入口与运行时状态
3. 执行机制与委派
4. 传播、可见性与观测
5. 扩展、测试与维护工作流

### 5.2 横向机制索引

横向主线回答“我现在遇到某类问题，应该去哪查”。

全书统一挂接到六个主题：

1. `Assembly`
2. `Context`
3. `Execution`
4. `Propagation`
5. `Extension`
6. `Operations`

每章必须明确声明自己属于哪一个或哪几个主题。

### 5.3 全书统一的维护者三问

整套教程反复回答三个问题：

1. 这个行为是谁提供的。
2. 这个行为是如何传播的。
3. 这个问题应该修在哪层。

每章末尾都必须以这三个问题收束。

## 6. 目录级重组方案

本次改造保留“四个大 part”的总体形式，但重定义各部分职责。

### 6.1 Part 0：如何使用这套教程

由 `README.md` 与 `00-preface.md` 共同承担，职责包括：

1. 给出三层系统总图。
2. 解释双导航：顺着学与按问题查。
3. 给出统一分析框架。
4. 给出读者分流方式。
5. 给出高频维护任务的入口索引。

### 6.2 Part 1：系统边界与组装根

目标是建立全系统静态图，回答“这三层栈共同构成了什么，入口在哪里，边界怎么划”。

建议保留并强化：

1. `01-what-deepagents-builds.md`
2. `02-repo-map-and-package-boundaries.md`
3. `03-create-deep-agent-as-assembly-root.md`

### 6.3 Part 2：运行时状态与执行机制

目标是解释系统带着什么状态运行，以及这些状态如何被执行链消费。

建议组织为两段连续主线：

1. 状态面
   - filesystem
   - state model
   - memory
   - skills
   - prompt layering
   - permissions
2. 执行面
   - tools
   - graph/node execution
   - subagents
   - context isolation

这里必须补强一章专讲 tools，把 tool execution 从“扩展点”提升为核心运行时机制。

### 6.4 Part 3：传播、可见性与观测

这是本次系统化改造的重点新增主线。

这一部分专门回答：

1. 内部事件如何向外传播。
2. callback manager 和 config tree 如何影响观测面。
3. stream consumer 到底能看到什么。
4. 哪些信息只是消费者不可见，哪些信息是真的没有进入 callback、stream 或折返结果。
5. compiled subagent、declarative subagent、async subagent 在传播语义上分别如何表现。

### 6.5 Part 4：维护、排障与安全扩展

目标是把 examples、testing、safe extension、排障组织成维护者工作流，而不是松散的收尾章节。

顺序应强调：

1. 先从 examples 反推机制。
2. 再从测试矩阵和排障剧本定位问题。
3. 最后落到扩展新能力的安全修改路径。

## 7. 传播层统一模型

教程必须为传播相关问题建立一张统一模型图，本文称之为“传播层四分图”。

### 7.1 四条线

1. 执行线
   - 谁真正执行了 node、tool、model、subgraph。
2. 观测线
   - callback manager / run manager 是否接到了这一调用。
3. 流输出线
   - `stream()` / `astream()` 最终向消费者发出了什么。
4. 结果折返线
   - 子图或子代理执行完成后，哪些 state update、message、summary、tool result 折返到 parent。

### 7.2 设计意义

任何“主 agent 有没有拦截到子代理内部调用”这一类问题，都必须被拆成四个独立判断：

1. 调用是否真的执行了。
2. 父级 callback tree 是否可靠观测到。
3. 外部 stream consumer 是否看得到。
4. 最终结果是否折返回 parent。

### 7.3 典型场景

以 compiled subagent 内部 node 调 LLM 为例：

1. 执行线：调用发生在 subagent 自己的内部图中。
2. 观测线：是否进入父 callbacks，取决于 callback/config 传播能力及当前已知限制。
3. 流输出线：是否暴露给外层消费者，取决于 stream mode、subgraphs 配置、消息流处理链和可见性控制方式。
4. 结果折返线：即使原始 token 不可见，summary、message、update 仍可能回到 parent。

教程在处理 subagent、callback、streaming、visibility 主题时，必须统一使用这张四分图，而不是混用“看见”“拦截”“知道”“接管”这些模糊词。

## 8. 章节模板规范

每章尽量使用统一模板，只允许少量受控例外。

### 8.1 固定结构

1. 本章回答什么
2. 在整套系统中的位置
3. 静态结构
4. 运行时链路
5. 传播 / 可见性 / 拦截点
6. 扩展接口
7. 常见问题与排障入口
8. 本章结论

### 8.2 写法要求

1. 每章只保留一条主链路，其他分支作为例外或变体处理。
2. 正常路径与受控例外分开写。
3. 机制事实与维护建议分开写。
4. 跨章跳转使用固定文案，避免随意引用。
5. 章节结尾固定用“谁提供 / 如何传播 / 修在哪层”收束。

## 9. 具体内容改造要求

### 9.1 README 的职责升级

`README.md` 必须从导航页升级为系统入口，至少包含：

1. 三层栈与六个横向主题的一张总图。
2. 顺着学与按问题查的双导航。
3. 统一分析框架。
4. 高频问题索引。
5. 维护者任务入口索引。

### 9.2 Tools 升格为核心机制

教程必须有一个足够强的 tools 机制主章，解释：

1. tool execution 在三层栈中的位置。
2. tools 与 runtime state、memory、filesystem、subagent 的关系。
3. tools 与 callbacks、streaming、config 传播的关系。
4. tools 的官方扩展面、半官方扩展面与测试要求。

### 9.3 Subagent 章节拆负载

当前 subagent 相关内容过于集中。改造后，subagent 章节主要负责：

1. subagent 的三种形态及边界。
2. execution 与 context isolation。
3. middleware 继承与装配策略。

callbacks、streaming、visibility 的机制解释应抽走到传播主线中，再由 subagent 章节回链。

### 9.4 Callbacks / Streaming / Visibility 成体系组织

传播主线至少要显式覆盖：

1. callback manager 的组装与 child run 传播。
2. `RunnableConfig`、tags、metadata、callbacks 的下传。
3. LangGraph stream mode 的形状差异。
4. selective visibility 的控制面。
5. “消费者不可见”与“系统不可知”的区别。
6. 当前测试已暴露的已知限制。

### 9.5 Appendix 变成检索面板

附录除了保留现有速查表外，还应承担：

1. 问题到章节的快速跳转。
2. 问题到源码文件与测试的快速跳转。
3. 问题到三层责任边界的快速定位。

## 10. 建议的最终信息架构

以下是建议的目标形态。实施时可以保留部分现有文件名，但内容职责需要对齐。

### 10.1 Part 0

1. `README.md`：总图、双导航、问题索引、读者入口
2. `00-preface.md`：阅读约定、来源标签、稳定性标签、受控例外写法

### 10.2 Part 1：系统边界与组装根

1. 三层栈共同构成什么系统
2. 三个仓库的架构、模块边界、交互面
3. `create_deep_agent()` 如何成为 assembly root

### 10.3 Part 2：运行时状态与执行机制

1. filesystem 与 state model
2. memory、skills、prompt layering
3. tools 机制与执行表面
4. subagents、execution model、context isolation
5. summarization、permissions、safety boundaries

### 10.4 Part 3：传播、可见性与观测

1. 传播层总览与四分图
2. callback / callback manager / config propagation
3. streaming / visibility / selective exposure
4. subagent 传播矩阵与维护者 recipes

### 10.5 Part 4：维护、排障与安全扩展

1. backend protocol 与 storage strategy
2. provider profiles 与 model routing
3. testing the harness
4. reading the examples like a maintainer
5. how to add a new capability safely

### 10.6 Appendix

1. code reading checklist
2. examples index
3. propagation and visibility cheatsheet
4. test matrix
5. troubleshooting playbook

## 11. 图表要求

本次改造至少要新增或重绘两张全书级图表：

1. 模块交互关系图
   - 以 Deep Agents assembly root 为入口，串到 LangGraph runtime，再串到 LangChain primitives / callback layer。
2. 传播层四分图
   - 把执行线、观测线、流输出线、结果折返线画出来，并用 subagent / compiled subagent 场景标注。

这两张图必须成为全书共用地图，而不是只在某一章局部出现。

## 12. 验收标准

当改造完成时，至少满足以下条件：

1. 首页能够同时支持“顺着学”和“按问题查”。
2. 每章都明确声明所属横向主题与上下游依赖。
3. 每章都使用统一模板，或显式声明受控例外。
4. 教程中存在独立的传播主线，而不是把 callbacks / streaming / visibility 分散在多个主题里。
5. tools 被当作核心运行时机制讲清楚。
6. 重要结论带有来源标签和稳定性标签。
7. 附录能支持维护者从症状快速跳到章节、源码和测试。
8. 教程不需要修改上游源码仓库即可完成本轮系统化重构。

## 13. 风险与取舍

### 13.1 主要风险

1. 只做导航增强而不重构章节职责，最后仍然会像“加强版目录页”。
2. 传播层不单独成线，subagent 章节会继续超载。
3. 继续把工具、callback、stream、state return 混写，会让维护者误判边界。
4. 章法不统一，即使新增内容，整体仍像若干强章节拼盘。

### 13.2 取舍结论

本次优先做“最小但结构性”的改造：

1. 保留四大 part。
2. 重做首页骨架。
3. 统一章节模板。
4. 抽出传播主线。
5. 升格 tools。
6. 把附录改造成维护者检索面板。

不以“全面扩写”作为第一目标。

## 14. 后续规划边界

本设计文档只定义系统化改造目标、结构原则、章节职责和验收标准。

后续实施规划需要继续明确：

1. 每个现有文件的具体改写范围。
2. 是否新增文件、如何重编号、如何保留旧链接。
3. 改造顺序与分批提交策略。
4. 每个章节需要补哪些图、表、案例和测试引用。

这些内容应在后续 implementation plan 中完成，而不是继续堆在设计文档里。
