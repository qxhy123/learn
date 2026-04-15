# 附录 A：工具与框架速查

本附录汇总了 Harness Engineering 实践中常用的工具和框架，按功能分类整理。每个工具列出定位、适用场景和官方链接，便于在实际项目中快速选型。

> **注意**：AI 工程工具链演进极快，以下信息截至 2026 年 4 月。使用前请查阅官方文档确认最新版本和 API。

---

## 评估框架

用于系统化评估 LLM、RAG 管线和 Agent 系统的质量。

| 名称 | 定位 | 适用场景 | 官方链接 |
|------|------|----------|----------|
| **lm-evaluation-harness** | EleutherAI 开源的标准化 LLM 评估框架 | 模型基准测试、多任务评估、自定义 Task 开发 | https://github.com/EleutherAI/lm-evaluation-harness |
| **DeepEval** | AI 系统的单元测试框架，集成 pytest | AI 应用的自动化测试、CI 集成、LLM-as-Judge | https://github.com/confident-ai/deepeval |
| **RAGAS** | 专注于 RAG 管线的评估框架 | RAG 检索质量评估、生成忠实度评估、端到端评估 | https://github.com/explodinggradients/ragas |
| **Promptfoo** | Prompt 和 LLM 输出的评估与对比工具 | Prompt 迭代评估、A/B 比较、红队测试 | https://github.com/promptfoo/promptfoo |
| **Inspect AI** | Anthropic 的 AI 系统评估框架 | Agent 评估、多步任务评估、安全评估 | https://github.com/UKGovernmentBEIS/inspect_ai |
| **OpenAI Evals** | OpenAI 开源的评估框架 | 模型能力评估、自定义评估集 | https://github.com/openai/evals |
| **Braintrust** | 商业评估与可观测性平台 | 企业级评估管理、评估数据集版本化、团队协作 | https://www.braintrust.dev/ |
| **Galileo** | AI 质量智能平台 | 幻觉检测、RAG 质量监控、生产评估 | https://www.rungalileo.io/ |

### 选型建议

- **模型基准测试**：lm-evaluation-harness 是事实标准，生态最完整
- **应用级测试**：DeepEval 与 pytest 集成最好，适合 CI/CD
- **RAG 评估**：RAGAS 专注 RAG，指标体系最成熟
- **Prompt 迭代**：Promptfoo 上手最快，适合快速对比

---

## 编排框架

用于构建 Agent 工作流、多 Agent 协作和复杂编排逻辑。

| 名称 | 定位 | 适用场景 | 官方链接 |
|------|------|----------|----------|
| **LangGraph** | LangChain 生态的有状态 Agent 编排框架 | 复杂 Agent 工作流、有状态对话、人类介入循环 | https://github.com/langchain-ai/langgraph |
| **OpenAI Agents SDK** | OpenAI 官方的 Agent 构建框架 | Agent 编排、Handoff 模式、Guardrails 集成 | https://github.com/openai/openai-agents-python |
| **AutoGen** | Microsoft 的多 Agent 对话框架 | 多 Agent 协作、角色扮演、群聊编排 | https://github.com/microsoft/autogen |
| **CrewAI** | 基于角色的多 Agent 协作框架 | 团队式 Agent 编排、任务分配、顺序/并行执行 | https://github.com/crewAIInc/crewAI |
| **Claude Agent SDK** | Anthropic 的 Agent 构建框架 | Claude 模型的 Agent 编排、工具使用 | https://github.com/anthropics/claude-code-sdk |
| **Semantic Kernel** | Microsoft 的 AI 编排 SDK | 企业级 AI 集成、插件系统、多模型编排 | https://github.com/microsoft/semantic-kernel |
| **DSPy** | 声明式 LLM 编程框架 | Prompt 优化、模块化 LLM 管线、自动调优 | https://github.com/stanfordnlp/dspy |
| **Haystack** | 端到端 NLP/LLM 管线框架 | RAG 管线构建、文档处理、组件化编排 | https://github.com/deepset-ai/haystack |

### 选型建议

- **有状态工作流**：LangGraph 最灵活，支持复杂的状态图
- **多 Agent 对话**：AutoGen 的对话模式最成熟
- **快速原型**：CrewAI 上手最快，适合简单的多 Agent 场景
- **企业集成**：Semantic Kernel 的企业级特性最完善

---

## 可观测性平台

用于追踪、监控和调试 AI 系统的运行状态。

| 名称 | 定位 | 适用场景 | 官方链接 |
|------|------|----------|----------|
| **Langfuse** | 开源的 LLM 可观测性平台 | Trace 追踪、Prompt 管理、评估集成、成本分析 | https://github.com/langfuse/langfuse |
| **Arize Phoenix** | 开源的 AI 可观测性工具 | Trace 可视化、嵌入分析、LLM 评估 | https://github.com/Arize-ai/phoenix |
| **Arize AI** | 商业 ML/AI 可观测性平台 | 生产监控、漂移检测、性能分析 | https://arize.com/ |
| **LangSmith** | LangChain 生态的可观测性平台 | LangChain/LangGraph 应用的 Trace、调试、评估 | https://smith.langchain.com/ |
| **Weights & Biases (W&B)** | 实验追踪与模型监控平台 | 训练实验管理、模型注册、生产监控 | https://wandb.ai/ |
| **Helicone** | 开源的 LLM API 代理与分析平台 | API 调用日志、成本追踪、速率限制、缓存 | https://github.com/Helicone/helicone |
| **OpenTelemetry + Traceloop** | 基于 OTel 标准的 AI 追踪 | 标准化 Trace 采集、与现有可观测性栈集成 | https://github.com/traceloop/openllmetry |

### 选型建议

- **开源优先**：Langfuse 功能最全面，社区最活跃
- **LangChain 生态**：LangSmith 集成最紧密
- **已有 OTel 基础设施**：Traceloop/OpenLLMetry 可无缝接入
- **企业级需求**：Arize AI 的漂移检测和告警最成熟

---

## CI/CD 与质量门禁工具

用于将评估和测试 Harness 集成到持续集成 / 持续部署流水线中。

| 名称 | 定位 | 适用场景 | 官方链接 |
|------|------|----------|----------|
| **GitHub Actions** | GitHub 原生的 CI/CD 平台 | 评估流水线自动化、PR 评估报告、定时评估 | https://github.com/features/actions |
| **Harness.io** | 商业 CI/CD 与 DevOps 平台 | 企业级部署流水线、特性标志、混沌工程 | https://harness.io/ |
| **GitLab CI/CD** | GitLab 内置的 CI/CD 系统 | 评估流水线、制品管理、环境管理 | https://docs.gitlab.com/ee/ci/ |
| **CircleCI** | 云原生 CI/CD 平台 | 高性能并行构建、GPU Runner 支持 | https://circleci.com/ |
| **DVC (Data Version Control)** | 数据和模型版本控制 | 评估数据集版本化、模型版本化、评估结果追踪 | https://github.com/iterative/dvc |
| **MLflow** | 开源 ML 生命周期管理平台 | 实验追踪、模型注册、部署管理 | https://github.com/mlflow/mlflow |
| **CML (Continuous ML)** | ML 项目的持续集成工具 | PR 中自动生成评估报告、可视化对比 | https://github.com/iterative/cml |

### 选型建议

- **标准 CI/CD**：GitHub Actions 最通用，社区 Action 生态最丰富
- **数据版本化**：DVC 是事实标准
- **评估结果管理**：MLflow 的实验追踪适合管理评估历史
- **PR 评估报告**：CML 可以在 PR 中自动生成评估对比图表

---

## 安全与护栏工具

用于为 AI 系统构建输入/输出安全护栏。

| 名称 | 定位 | 适用场景 | 官方链接 |
|------|------|----------|----------|
| **Guardrails AI** | 开源的 AI 输出验证框架 | 输出格式验证、内容过滤、结构化输出保障 | https://github.com/guardrails-ai/guardrails |
| **NeMo Guardrails** | NVIDIA 的对话安全框架 | 对话流控制、主题限制、事实性检查 | https://github.com/NVIDIA/NeMo-Guardrails |
| **LLM Guard** | 开源的 LLM 安全工具包 | Prompt 注入检测、PII 检测、有害内容过滤 | https://github.com/protectai/llm-guard |
| **Rebuff** | Prompt 注入检测框架 | 多层 Prompt 注入防护、启发式 + ML 检测 | https://github.com/protectai/rebuff |
| **Microsoft Presidio** | 开源的 PII 检测与匿名化工具 | 个人信息识别、数据脱敏、合规处理 | https://github.com/microsoft/presidio |

### 选型建议

- **通用护栏**：Guardrails AI 最灵活，验证器生态最丰富
- **对话安全**：NeMo Guardrails 的 Colang 语言适合复杂对话策略
- **PII 处理**：Presidio 最成熟，支持多语言
- **Prompt 注入**：LLM Guard 覆盖最全面

---

## 相关教程章节索引

| 工具类别 | 重点讲解章节 |
|----------|------------|
| 评估框架 | [第6章](../part2-evaluation-harness/06-evaluation-harness-basics.md)、[第7章](../part2-evaluation-harness/07-lm-eval-harness-architecture.md)、[第8章](../part2-evaluation-harness/08-custom-evaluation-harness.md) |
| 编排框架 | [第15章](../part4-orchestration-harness/15-single-agent-harness.md)、[第16章](../part4-orchestration-harness/16-openai-harness-blueprint.md)、[第17章](../part4-orchestration-harness/17-multi-agent-harness.md) |
| 可观测性平台 | [第20章](../part5-production-and-frontier/20-observability-harness.md) |
| CI/CD 工具 | [第13章](../part3-test-and-ci-harness/13-ci-cd-quality-gates.md)、[第21章](../part5-production-and-frontier/21-deployment-harness.md) |
| 安全与护栏 | [第22章](../part5-production-and-frontier/22-safety-guardrails-compliance.md) |
