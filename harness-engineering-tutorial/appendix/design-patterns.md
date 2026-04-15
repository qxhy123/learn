# 附录 B：Harness 设计模式速查表

本附录汇总了 Harness Engineering 中常见的 15 个设计模式。每个模式都经过实际生产验证，可以直接应用到你的 AI 系统中。

> **如何使用**：根据你遇到的问题场景，在表中找到对应的模式，然后跳转到详细讲解的章节深入学习。

---

## 模式总览

| # | 模式名 | 问题场景 | 解决方案 | 适用层 | 复杂度 |
|---|--------|----------|----------|--------|--------|
| 1 | **Evaluation Pipeline** | 需要系统化地评估 AI 系统质量，但评估逻辑散落在各处 | 将评估抽象为标准化的流水线：数据加载 → 推理执行 → 指标计算 → 结果聚合，每个阶段可独立替换和扩展 | 评估 | ★★☆☆☆ |
| 2 | **LLM-as-Judge** | 传统指标（BLEU/ROUGE）无法捕捉生成质量的语义维度 | 用一个强力 LLM 作为评审员，通过结构化的评分 Prompt 和多维度评分表来评估目标 LLM 的输出，配合人工校准确保一致性 | 评估 | ★★★☆☆ |
| 3 | **Statistical Regression Guard** | LLM 输出的非确定性导致传统的精确匹配回归测试无法使用 | 基于多次采样的分布级断言：对同一输入运行 N 次，计算指标的置信区间，当置信区间超出基线范围时触发告警 | 测试 | ★★★☆☆ |
| 4 | **Snapshot + Semantic Diff** | 需要检测 LLM 输出是否发生了有意义的变化，但逐字对比噪声太大 | 保存输出的语义快照（嵌入向量），用余弦相似度而非字符串相等来判断回归，设置语义距离阈值区分"噪声变化"和"实质回归" | 测试 | ★★★☆☆ |
| 5 | **Quality Gate Chain** | CI/CD 中需要在多个维度上把关 AI 系统质量，但单一门禁不够 | 设计多级质量门禁链：快速检查（格式/长度/安全）→ 采样评估（核心指标抽检）→ 全量评估（发布前完整评估），逐级递进，平衡速度和覆盖度 | 测试 | ★★☆☆☆ |
| 6 | **Guide-Sensor Loop** | Agent 行为不稳定，需要同时使用前馈控制和反馈控制 | 将 Harness 拆分为 Guide（前馈：系统提示、工具定义、Few-shot）和 Sensor（反馈：输出验证、评分、监控），形成闭环：Sensor 检测到问题 → 动态调整 Guide → 重新执行 | 编排 | ★★★☆☆ |
| 7 | **Handoff Protocol** | 多 Agent 系统中，Agent 之间的控制权转移缺乏规范，导致上下文丢失或任务重复 | 定义标准化的 Handoff 协议：发起方封装上下文摘要和任务描述，接收方确认能力匹配后接管，全程通过 Harness 记录 Trace 和状态 | 编排 | ★★★★☆ |
| 8 | **Router-Specialist** | 不同类型的请求需要不同的处理策略，单一 Agent 难以覆盖所有场景 | 设计一个轻量级路由 Agent 负责意图分类和任务分发，多个专家 Agent 各自处理特定类型的任务，路由决策可以是规则、分类器或 LLM | 编排 | ★★★☆☆ |
| 9 | **Checkpoint-Resume** | 长时运行的 Agent 任务可能因超时、错误或外部中断而失败，从头重来代价太高 | 在 Harness 中实现检查点机制：每完成一个关键步骤就持久化当前状态，失败时从最近的检查点恢复执行，而非从头开始 | 编排 | ★★★★☆ |
| 10 | **Fallback Cascade** | AI 系统依赖的外部服务（LLM API、向量数据库等）可能不可用或响应异常 | 设计多级回退策略：主模型 → 备用模型 → 缓存响应 → 降级响应。每级回退都有明确的触发条件和质量预期，Harness 记录回退事件 | 编排/运维 | ★★★☆☆ |
| 11 | **Trace-Through Observability** | AI 系统的调用链复杂（用户请求 → 路由 → 检索 → 生成 → 后处理），出了问题不知道在哪个环节 | 在 Harness 的每个组件中注入 Trace 点，用唯一的 Trace ID 串联完整调用链，记录每个环节的输入/输出/延迟/Token 数/成本 | 运维 | ★★★☆☆ |
| 12 | **Layered Guardrail** | 安全防护需要覆盖多个维度（Prompt 注入、有害内容、PII 泄露），单层过滤不够 | 设计多层护栏：输入层（Prompt 注入检测、输入清洗）→ 处理层（工具调用权限控制）→ 输出层（内容过滤、PII 脱敏），每层独立配置、独立监控 | 运维 | ★★★★☆ |
| 13 | **Drift Detector** | 模型性能随时间悄然退化（数据分布变化、Prompt 失效），但没有及时发现的机制 | 持续运行评估 Harness：定时对生产流量采样评估，将评估结果与基线对比，当指标超出容忍带时触发告警和自动化响应（通知、回滚、重新校准） | 运维 | ★★★★☆ |
| 14 | **Config-Driven Harness** | Harness 的行为需要频繁调整（换模型、改 Prompt、调参数），但每次都要改代码和重新部署 | 将 Harness 的可变部分抽取到配置中（YAML/JSON）：模型选择、Prompt 模板、重试策略、护栏规则、评估阈值等，支持热加载和特性标志 | 全部 | ★★☆☆☆ |
| 15 | **Harness-as-Code** | 团队协作中，Harness 的配置和演进缺乏版本控制和可审计性 | 将完整的 Harness 定义（评估配置、编排拓扑、护栏规则、部署策略）作为代码管理：Git 版本控制、PR 审查、自动化测试、环境隔离，与应用代码同等对待 | 全部 | ★★★☆☆ |

---

## 复杂度说明

| 复杂度 | 含义 |
|--------|------|
| ★☆☆☆☆ | 简单，几小时内可实现 |
| ★★☆☆☆ | 基础，一天内可实现，需要少量设计 |
| ★★★☆☆ | 中等，需要几天时间，涉及多个组件的协调 |
| ★★★★☆ | 较高，需要一周以上，涉及分布式状态管理或复杂的错误处理 |
| ★★★★★ | 高，需要团队协作和持续迭代 |

---

## 模式组合建议

在实际项目中，这些模式通常组合使用。以下是三个常见的组合：

### 组合一：最小可用 Harness

适合 MVP 或早期原型阶段。

> **Evaluation Pipeline** + **Guide-Sensor Loop** + **Config-Driven Harness**

先建立基本的评估能力，用 Guide-Sensor 闭环稳定 Agent 行为，通过配置化加速迭代。

### 组合二：生产级 Agent Harness

适合即将上线或已上线的 Agent 系统。

> **Router-Specialist** + **Handoff Protocol** + **Fallback Cascade** + **Trace-Through Observability** + **Layered Guardrail** + **Quality Gate Chain**

完整的编排、可观测性和安全防护，确保生产环境的可靠性。

### 组合三：持续演进 Harness

适合需要长期运营和持续改进的系统。

> 在组合二基础上 + **Drift Detector** + **Statistical Regression Guard** + **Harness-as-Code**

增加漂移检测和统计回归，用代码化管理确保 Harness 本身的可维护性。

---

## 相关章节索引

| 模式 | 详细讲解章节 |
|------|------------|
| Evaluation Pipeline | [第6章](../part2-evaluation-harness/06-evaluation-harness-basics.md) |
| LLM-as-Judge | [第8章](../part2-evaluation-harness/08-custom-evaluation-harness.md) |
| Statistical Regression Guard | [第14章](../part3-test-and-ci-harness/14-nondeterministic-regression.md) |
| Snapshot + Semantic Diff | [第14章](../part3-test-and-ci-harness/14-nondeterministic-regression.md) |
| Quality Gate Chain | [第13章](../part3-test-and-ci-harness/13-ci-cd-quality-gates.md) |
| Guide-Sensor Loop | [第3章](../part1-foundations/03-guides-and-sensors.md)、[第15章](../part4-orchestration-harness/15-single-agent-harness.md) |
| Handoff Protocol | [第16章](../part4-orchestration-harness/16-openai-harness-blueprint.md)、[第17章](../part4-orchestration-harness/17-multi-agent-harness.md) |
| Router-Specialist | [第17章](../part4-orchestration-harness/17-multi-agent-harness.md) |
| Checkpoint-Resume | [第18章](../part4-orchestration-harness/18-long-running-agent-harness.md) |
| Fallback Cascade | [第15章](../part4-orchestration-harness/15-single-agent-harness.md)、[第19章](../part4-orchestration-harness/19-rag-production-harness.md) |
| Trace-Through Observability | [第20章](../part5-production-and-frontier/20-observability-harness.md) |
| Layered Guardrail | [第22章](../part5-production-and-frontier/22-safety-guardrails-compliance.md) |
| Drift Detector | [第23章](../part5-production-and-frontier/23-entropy-management.md) |
| Config-Driven Harness | [第5章](../part1-foundations/05-skill-map-and-toolchain.md)、[第21章](../part5-production-and-frontier/21-deployment-harness.md) |
| Harness-as-Code | [第21章](../part5-production-and-frontier/21-deployment-harness.md) |
