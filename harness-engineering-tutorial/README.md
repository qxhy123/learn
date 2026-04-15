# Harness Engineering 教程：从零到高阶

## 项目简介

**Harness Engineering** 是 2026 年兴起的 AI 工程学科，其核心洞见是：一个真正可靠的 AI Agent 系统，模型本身只是其中一半——另一半是包裹在模型外面的控制层，即 **Harness**。这层控制基础设施涵盖评估、测试、编排、可观测性、安全护栏等方方面面，决定了 AI 系统在生产环境中的可靠性、可维护性和可演进性。用一个公式概括：**Agent = Model + Harness**。

本教程从零开始，系统地覆盖 Harness Engineering 的完整知识体系。你将从核心心智模型出发，依次学习评估 Harness、测试与 CI Harness、编排 Harness，最终掌握生产运维与前沿技术。教程融合了 Martin Fowler 的 Guides & Sensors 分类、OpenAI 的百万行代码实验、Anthropic 的多 Agent Harness 架构等代表性工作的关键洞见。

**本教程的独特之处**：每章都包含可运行的代码示例和「动手实验」，不仅讲"怎么搭 Harness"，更讲"为什么这样设计 Harness"——从评估流水线的统计学直觉到编排架构的工程权衡，帮助读者建立对 AI 系统控制层的系统性理解。

---

## 目标受众

- 正在构建 AI Agent 系统、需要系统化地提升可靠性和可维护性的 **AI 应用工程师**
- 负责 AI 系统评估、测试和质量保障的 **AI 质量工程师 / QA 工程师**
- 设计和维护 AI 系统生产基础设施的 **平台工程师 / MLOps 工程师**
- 希望理解 AI 系统工程化全貌、从研究走向工程落地的 **ML 研究者**
- 对 AI Agent 编排、可观测性和安全合规感兴趣的 **技术管理者 / 架构师**

---

## 章节导航目录

### 开始之前

- [前言：如何使用本教程](./00-preface.md)

### 第一部分：基础概念

| 章节 | 标题 | 主要内容 | 实验重点 |
|------|------|----------|----------|
| 第1章 | [Harness Engineering 的起源与动机](./part1-foundations/01-origins-and-motivation.md) | AI 系统从"Demo 能跑"到"生产可靠"的鸿沟、Harness 概念的历史演变、代表性失败案例分析 | 分析一个无 Harness 的 Agent 系统的失败模式 |
| 第2章 | [核心心智模型：Agent = Model + Harness](./part1-foundations/02-agent-equals-model-plus-harness.md) | 核心公式的推导、Model 与 Harness 的职责边界、Harness 的五层架构（评估/测试/编排/可观测/安全） | 拆解一个开源 Agent 项目的 Harness 成分 |
| 第3章 | [Guides 与 Sensors：前馈控制与反馈控制](./part1-foundations/03-guides-and-sensors.md) | Martin Fowler 的分类框架、Guides（系统提示/工具定义/Few-shot）与 Sensors（评估器/断言/监控）、前馈-反馈闭环设计 | 为一个简单 Agent 分别实现 Guide 和 Sensor |
| 第4章 | [Harness Engineering 与相邻学科](./part1-foundations/04-adjacent-disciplines.md) | 与 MLOps、传统软件测试、控制论、可靠性工程的关系与区别、Harness Engineering 的独特价值 | 绘制 Harness Engineering 与相邻学科的知识地图 |
| 第5章 | [Harness 工程师的技能图谱与工具链](./part1-foundations/05-skill-map-and-toolchain.md) | 核心技能（评估设计/统计分析/编排架构/可观测性）、工具链概览、学习路径建议 | 搭建本教程所需的完整开发环境 |

### 第二部分：评估 Harness

| 章节 | 标题 | 主要内容 | 实验重点 |
|------|------|----------|----------|
| 第6章 | [评估 Harness 基础](./part2-evaluation-harness/06-evaluation-harness-basics.md) | 评估的核心概念（指标/基准/数据集）、离线评估 vs 在线评估、评估流水线的基本架构 | 构建一个最小评估流水线并运行首次评估 |
| 第7章 | [lm-evaluation-harness 架构剖析](./part2-evaluation-harness/07-lm-eval-harness-architecture.md) | EleutherAI lm-eval-harness 的设计理念、Task/Model/Filter/Metric 四大组件、YAML 配置系统 | 用 lm-eval-harness 评估一个模型并分析结果 |
| 第8章 | [构建自定义评估 Harness](./part2-evaluation-harness/08-custom-evaluation-harness.md) | 自定义 Task 开发、评估指标设计（精确匹配/模糊匹配/LLM-as-Judge）、统计显著性检验 | 为领域特定任务开发完整的评估 Harness |
| 第9章 | [RAG 管线评估 Harness](./part2-evaluation-harness/09-rag-evaluation-harness.md) | RAG 评估的特殊挑战、RAGAS 框架、检索质量与生成质量的分离评估、端到端评估设计 | 构建一个 RAG 管线的多维评估 Harness |
| 第10章 | [Agent 系统评估 Harness](./part2-evaluation-harness/10-agent-evaluation-harness.md) | Agent 评估的难点（多步/非确定性/工具调用）、轨迹评估、任务完成率评估、SWE-bench 式 Harness 设计 | 为一个多步 Agent 构建评估 Harness |

### 第三部分：测试与 CI Harness

| 章节 | 标题 | 主要内容 | 实验重点 |
|------|------|----------|----------|
| 第11章 | [AI 系统的测试 Harness 模式](./part3-test-and-ci-harness/11-test-harness-patterns.md) | 测试 Harness 与传统单元测试的区别、模拟（Mock）策略、快照测试、契约测试在 AI 系统中的应用 | 为 LLM 调用链编写测试 Harness |
| 第12章 | [测试驱动微调（TDF）Harness](./part3-test-and-ci-harness/12-test-driven-fine-tuning.md) | TDF 理念（先写评估再微调）、评估集设计、过拟合检测、微调质量门禁 | 用 TDF 方法完成一次小规模微调 |
| 第13章 | [CI/CD 集成与质量门禁](./part3-test-and-ci-harness/13-ci-cd-quality-gates.md) | GitHub Actions 中的评估流水线、质量门禁设计、评估结果的版本化与可视化、成本控制 | 配置一个包含 AI 评估的 CI 流水线 |
| 第14章 | [非确定性系统的回归测试](./part3-test-and-ci-harness/14-nondeterministic-regression.md) | 非确定性输出的统计回归方法、分布级断言、置信区间、A/B 评估框架 | 实现一个基于统计的回归测试 Harness |

### 第四部分：编排 Harness

| 章节 | 标题 | 主要内容 | 实验重点 |
|------|------|----------|----------|
| 第15章 | [单 Agent Harness 设计](./part4-orchestration-harness/15-single-agent-harness.md) | 单 Agent 的 Harness 骨架（输入验证/Prompt 管理/工具注册/输出解析/重试策略）、状态机模式 | 从零构建一个结构清晰的单 Agent Harness |
| 第16章 | [OpenAI Harness 蓝图](./part4-orchestration-harness/16-openai-harness-blueprint.md) | OpenAI 百万行代码实验的 Harness 架构、Agents SDK 的设计理念、Handoff 模式、Guardrails 集成 | 用 OpenAI Agents SDK 实现一个多步任务 Harness |
| 第17章 | [多 Agent Harness 架构](./part4-orchestration-harness/17-multi-agent-harness.md) | Anthropic 多 Agent 架构模式（Planner/Generator/Evaluator）、GAN 式反馈循环、Sub-agent 协调与冲突解决 | 实现一个多 Agent 协作 Harness |
| 第18章 | [长时运行 Agent Harness](./part4-orchestration-harness/18-long-running-agent-harness.md) | 长任务的特殊挑战（超时/检查点/恢复/人类介入）、持久化执行引擎、异步 Harness 模式 | 构建一个支持检查点和恢复的长时运行 Harness |
| 第19章 | [RAG 生产系统 Harness](./part4-orchestration-harness/19-rag-production-harness.md) | RAG 系统的完整 Harness（索引管线/检索链/生成链/缓存/回退策略）、混合检索编排、查询路由 | 搭建一个生产级 RAG Harness |

### 第五部分：生产运维与前沿

| 章节 | 标题 | 主要内容 | 实验重点 |
|------|------|----------|----------|
| 第20章 | [可观测性 Harness](./part5-production-and-frontier/20-observability-harness.md) | AI 系统可观测性的三大支柱（Trace/Metric/Log）、Langfuse/Arize 集成、Token 用量与成本追踪、异常检测 | 为 Agent 系统接入完整的可观测性 Harness |
| 第21章 | [部署 Harness 与流水线](./part5-production-and-frontier/21-deployment-harness.md) | 模型与 Harness 的协同部署、蓝绿部署/金丝雀发布、特性标志、回滚策略、基础设施即代码 | 实现一个 Harness 感知的部署流水线 |
| 第22章 | [安全、护栏与合规 Harness](./part5-production-and-frontier/22-safety-guardrails-compliance.md) | 输入/输出护栏、Prompt 注入防护、内容过滤、PII 检测、审计日志、合规框架（EU AI Act） | 为 Agent 系统构建多层安全 Harness |
| 第23章 | [熵管理与持续改进](./part5-production-and-frontier/23-entropy-management.md) | AI 系统的"熵增"问题、模型漂移检测、Prompt 漂移、持续评估闭环、Harness 的版本演进 | 构建一个检测和应对模型漂移的 Harness |
| 第24章 | [Harness Engineering 的未来](./part5-production-and-frontier/24-future-of-harness-engineering.md) | 自适应 Harness、自生成测试、Agent 自我改进闭环、Harness 标准化趋势、开放问题与研究方向 | 设计一个自适应 Harness 的原型 |

### 附录

| 附录 | 标题 | 内容 |
|------|------|------|
| 附录 A | [工具与框架速查](./appendix/tools-reference.md) | 评估、编排、可观测性、CI/CD 工具的快速参考 |
| 附录 B | [设计模式速查表](./appendix/design-patterns.md) | 15 个 Harness 设计模式的速查表 |
| 附录 C | [练习参考答案](./appendix/answers.md) | 全部 24 章练习题的答案提示 |

---

## 学习路径建议

### 路径一：快速入门（4 小时）

适合想快速了解 Harness Engineering 全貌的读者。

> 第1章（起源）→ 第2章（核心公式）→ 第3章（Guides & Sensors）→ 第6章（评估基础）→ 第15章（单 Agent Harness）→ 第20章（可观测性）

### 路径二：评估与质量专精（8 小时）

适合 QA 工程师或需要为 AI 系统建立质量保障体系的读者。

> 第1-3章（基础概念）→ 第6-10章（评估 Harness 全部）→ 第11章（测试模式）→ 第13章（CI/CD）→ 第14章（回归测试）→ 附录 A

### 路径三：全栈 Harness 工程师（20+ 小时）

适合希望系统掌握 Harness Engineering 完整知识体系的读者。

> 按顺序阅读全部 24 章，完成每章的动手实验和练习题。建议每天 1-2 章，两周完成。

---

## 前置要求

- **Python 编程**：能够读写 Python 代码，熟悉 async/await、类型注解
- **LLM 基础**：了解大语言模型的基本工作原理（Prompt、Token、Temperature 等概念）
- **API 使用经验**：使用过至少一种 LLM API（OpenAI、Anthropic、本地模型等）
- **命令行基础**：熟悉终端操作、Git 基本使用
- **（可选）Docker**：部分实验涉及容器化部署，但非必需

---

## 如何使用本教程

1. **按顺序阅读，但可以跳读**：章节之间有递进关系，但每章也尽量自包含。如果你对某一领域已有经验，可以直接跳到感兴趣的部分
2. **动手实验是核心**：每章末尾的「动手实验」是理解内容的关键。仅阅读文字会遗漏很多工程直觉——请务必动手跑代码
3. **先理解"为什么"，再学"怎么做"**：每章都从问题和动机出发。理解了设计动机，具体 API 的使用自然水到渠成
4. **反复对照实际项目**：学习过程中不断将教程内容与你自己的 AI 项目对照，思考"我的系统中这层 Harness 在哪里？缺了什么？"
5. **善用附录**：附录 A 是工具速查，附录 B 是设计模式速查——它们是你在实际工作中快速查阅的参考手册

---

## 教程特色

- **体系完整**：从评估、测试、编排到可观测性、安全，覆盖 Harness Engineering 的完整生命周期
- **公式驱动**：围绕 Agent = Model + Harness 这一核心公式展开，所有内容都有清晰的定位
- **工程导向**：不是学术综述，而是面向一线工程师的实践指南。每章都有可运行的代码和实验
- **融合前沿**：整合了 Martin Fowler、OpenAI、Anthropic 等 2025-2026 年的最新实践和思想
- **中文优先**：用中文把概念讲透，用英文保留术语精确性，兼顾可读性与专业性

---

## 与仓库其他教程的关系

本教程与仓库中的其他教程形成互补：

- **[vLLM 教程](../vllm-tutorial/README.md)**：vLLM 是模型推理层的核心基础设施，Harness Engineering 则关注包裹在推理层之上的控制层。两者结合，覆盖从模型部署到系统可靠性的完整链路
- **[TDD 教程](../tdd-tutorial/README.md)**：TDD 是 Harness Engineering 中测试 Harness 的方法论基础。本教程第三部分（测试与 CI Harness）在 TDD 基础上，进一步讨论 AI 系统特有的非确定性测试挑战
- **[Vibe Coding 教程](../vibe-coding-tutorial/README.md)**：Vibe Coding 关注如何与 AI 高效协作编程，而 Harness Engineering 关注如何让 AI 系统本身变得可靠。当你用 Vibe Coding 的方式构建 AI 系统时，Harness Engineering 提供了质量保障的方法论
