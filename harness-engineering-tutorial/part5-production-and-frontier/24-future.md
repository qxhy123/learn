# 第24章：Harness Engineering 的未来

> 模型会迭代，框架会更替，但约束、评估和编排的需求永远存在。Harness Engineering 不是某个特定模型时代的产物——它是让 AI 在生产环境中可靠工作的永恒工程学科。

---

## 学习目标

学完本章，你将能够：

1. 理解 Harness 作为"模型漂移解药"的长期价值
2. 预判编排、评估、可观测性的平台收敛趋势
3. 评估 Managed Harnesses as a Service 的成熟度
4. 理解为什么 Harness 比任何特定模型都长寿
5. 探索 AutoHarness 的可能性与局限

---

## 24.1 Harness 作为长时任务中"模型漂移"的解药

### 问题：模型不断变，系统不能断

```
模型生命周期：

2024 ──── GPT-4, Claude 3 ────────────────────────
2025 ──── Claude 3.5, GPT-4o, Claude 4 ──────────
2026 ──── Claude 4+, GPT-5, 开源追赶 ─────────────
2027 ──── ??? ────────────────────────────────────

你的系统需要在模型更迭中保持稳定
唯一不变的是：约束、评估、编排
              └──── Harness ────┘
```

### 为什么 Harness 是解药而不是补丁

```
没有 Harness 的世界：
  模型更新 → 行为改变 → 发现质量下降 → 紧急修 prompt → 下次还会
  （被动、滞后、每次都从头开始）

有 Harness 的世界：
  模型更新
    │
    ▼
  自动运行 eval suite → 检测行为差异 → 自动标记回归
    │                                      │
    ▼                                      ▼
  回归轻微 → 自动调节参数              回归严重 → 阻止切换 + 告警
  （主动、实时、系统化应对）
```

### 模型无关的 Harness 设计原则

```python
class ModelAgnosticHarness:
    """模型无关的 Harness 设计"""

    def __init__(self):
        # 这些组件不依赖任何特定模型
        self.eval_suite = None          # 评估套件：检查输出属性
        self.guardrails = None          # 护栏：输入/输出过滤
        self.orchestrator = None        # 编排器：多 agent 协调
        self.observability = None       # 可观测性：tracing + metrics
        self.artifact_manager = None    # Artifact：跨会话状态

    # 只有这一层是模型相关的
    def set_model_adapter(self, adapter):
        """设置模型适配器——唯一的模型相关组件"""
        self.model_adapter = adapter

    # 核心 Harness 逻辑完全模型无关
    def execute(self, task: dict) -> dict:
        """执行任务：Harness 逻辑不关心底层是哪个模型"""
        # 1. 输入护栏
        cleaned_input = self.guardrails.check_input(task)

        # 2. 编排（多 agent 或单 agent）
        raw_output = self.orchestrator.run(cleaned_input, self.model_adapter)

        # 3. 输出护栏
        safe_output = self.guardrails.check_output(raw_output)

        # 4. 评估
        quality = self.eval_suite.evaluate(task, safe_output)

        # 5. 记录
        self.observability.log(task, safe_output, quality)

        return {"output": safe_output, "quality": quality}
```

### 模型切换成本对比

| 方面 | 没有 Harness | 有 Harness |
|------|-------------|-----------|
| 切换模型需要修改 | 所有 prompt + 参数 + 测试 | 只改适配器层 |
| 切换后验证 | 手动测试，不确定覆盖率 | 自动运行 eval suite |
| 发现回归时间 | 天-周（用户反馈） | 分钟（自动检测） |
| 回滚能力 | 手动、不可靠 | 一键回滚，有 artifact 保护 |

---

## 24.2 编排、评估、可观测性向统一平台收敛

### 当前的碎片化

```
2025 年的 AI 工程工具链：

编排:        LangChain / LlamaIndex / CrewAI / Autogen
评估:        DeepEval / Ragas / Braintrust / HELM
可观测性:    Langfuse / Arize / Galileo / Weights & Biases
部署:        Modal / Replicate / Together AI
护栏:        Guardrails AI / NeMo Guardrails / Lakera
Prompt 管理: PromptLayer / Humanloop / Agenta

问题：每个环节用不同工具，数据不互通，维护成本高
```

### 收敛趋势

```
2026-2027 年收敛方向：

┌─────────────────────────────────────────────────────┐
│              Unified AI Engineering Platform          │
│                                                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐    │
│  │ Orchestrate│  │  Evaluate  │  │  Observe   │    │
│  │ 编排       │  │  评估      │  │  可观测    │    │
│  │            │  │            │  │            │    │
│  │ Multi-agent│  │ Eval suite │  │ Tracing    │    │
│  │ Tool use   │  │ Regression │  │ Metrics    │    │
│  │ RAG        │  │ A/B test   │  │ Alerts     │    │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘    │
│        │               │               │            │
│        └───────────────┬┘───────────────┘            │
│                        │                             │
│                 ┌──────▼──────┐                      │
│                 │ Shared Data │                      │
│                 │    Layer    │                      │
│                 │ 统一数据层  │                      │
│                 └─────────────┘                      │
│                                                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐    │
│  │  Deploy    │  │  Guard     │  │  Manage    │    │
│  │  部署      │  │  护栏      │  │  Prompt    │    │
│  └────────────┘  └────────────┘  └────────────┘    │
└─────────────────────────────────────────────────────┘
```

### 为什么收敛不可避免

```python
CONVERGENCE_DRIVERS = {
    "data_gravity": {
        "explanation": "评估需要 trace 数据，可观测性需要评估结果，编排需要评估反馈",
        "consequence": "数据天然想要集中",
    },
    "workflow_coupling": {
        "explanation": "编排的结果必须被评估，评估的结果必须被观测，观测的告警必须回到编排",
        "consequence": "工具间接口成本高于平台内集成",
    },
    "developer_fatigue": {
        "explanation": "维护 6+ 个工具的配置、认证、版本、兼容性",
        "consequence": "团队倾向于选择"够用的一体化"而非"最优的碎片化"",
    },
    "vendor_expansion": {
        "explanation": "每个工具厂商都在向相邻功能扩展",
        "consequence": "LangSmith 加了评估，Langfuse 加了 prompt 管理",
    },
}
```

---

## 24.3 Managed Harnesses as a Service

### 从"自己造轮子"到"声明式配置"

```
                    自建 Harness          Managed Harness
                    (2024-2025)           (2026-2027)

开发工作量            数周-数月             数小时-数天
运维成本              团队自己承担          平台承担
可靠性                取决于团队能力        平台 SLA
定制化                完全定制              配置 + 插件
模型切换              自己改代码            改配置
扩展性                自己做                平台自动
```

### Managed Harness 架构

```yaml
# 声明式 Managed Harness 配置
apiVersion: harness.ai/v2
kind: ManagedHarness
metadata:
  name: customer-support-bot
  version: "3.2.0"

spec:
  # 任务定义
  task:
    type: conversational
    domain: customer-support
    languages: [zh-CN, en-US]

  # 模型配置（可以随时切换）
  models:
    primary:
      provider: anthropic
      model: claude-sonnet-4-20250514
    fallback:
      provider: anthropic
      model: claude-haiku-4-20250514
    cascade:
      enabled: true
      quality_threshold: 0.8

  # 编排
  orchestration:
    type: single-agent  # 或 multi-agent
    max_turns: 20
    context_strategy: compaction

  # RAG
  rag:
    enabled: true
    sources:
      - type: vector_db
        provider: pinecone
        index: support-docs
    reranking: true
    top_k: 5

  # 评估
  evaluation:
    continuous: true
    interval: 5m
    suite:
      - name: quality-check
        threshold: 0.8
      - name: safety-check
        threshold: 0.99
    regression:
      baseline: last-release
      tolerance: 0.05

  # 护栏
  guardrails:
    input:
      - prompt_injection_detection
      - topic_boundary: [customer-support, returns, billing]
      - pii_warning
    output:
      - pii_redaction
      - hallucination_detection
      - response_format_check

  # 可观测性
  observability:
    tracing: opentelemetry
    dashboard: grafana
    alerts:
      - condition: quality_score < 0.7
        channel: slack
        severity: high

  # 成本控制
  budget:
    daily_limit: $50
    per_request_limit: $0.50
    alert_at: 80%
```

---

## 24.4 Harness 比任何特定模型世代都长寿

### 持久性层次

```
                  持久性（年）
                  ↑
                  │
  Harness 设计模式│ ████████████████████████  >10年
                  │
    评估方法论    │ ██████████████████████    8-10年
                  │
    编排框架      │ ████████████████          5-8年
                  │
    Prompt 技术   │ ████████████              3-5年
                  │
    特定模型      │ ████████                  1-3年
                  │
    模型版本      │ ████                      3-12月
                  │
                  └─────────────────────────→
```

### 为什么约束比能力长寿

```python
# 这些约束在 2023 年成立，2026 年依然成立，2030 年大概率也成立

ETERNAL_CONSTRAINTS = [
    "输出需要被验证",              # 不信任单一来源
    "错误需要被检测和恢复",        # 容错
    "成本需要被控制",              # 经济约束
    "安全边界需要被强制执行",      # 法规和伦理
    "行为需要可观测和可审计",      # 问责
    "系统需要在模型更迭中存活",    # 持续运营
    "质量需要持续监控",            # 对抗熵增
]

# 这些技术在 2023 年成立，2026 年已经不适用
TRANSIENT_TECHNIQUES = [
    "Chain of Thought prompting",  # 某些模型不需要了
    "特定的 JSON mode 语法",       # API 语法变了
    "特定的 function calling 格式",# 每个提供商不一样
    "特定 token 限制的 workaround",# 窗口变大了
]

# 结论：投资 Harness（约束层）比投资 Prompt 技巧更长久
```

### 跨模型世代的投资回报

| 投资方向 | 初始成本 | 模型切换时的返工 | 长期价值 |
|---------|---------|-----------------|---------|
| 精心调优的 Prompt | 低 | 高（几乎全部返工） | 低 |
| 评估数据集 | 中 | 零（完全复用） | 极高 |
| Harness 编排逻辑 | 高 | 低（换适配器） | 极高 |
| 护栏规则 | 中 | 低（规则不变） | 高 |
| 可观测性基础设施 | 高 | 零（模型无关） | 极高 |

---

## 24.5 AutoHarness：自动合成 Harness

### 愿景

```
当前：人工设计 Harness
  人类分析任务需求 → 人类选择编排模式 → 人类编写评估 → ...

未来：AutoHarness
  描述任务需求 → AI 自动合成 Harness 配置 → 人类审核和调优
```

### AutoHarness 原型

```python
class AutoHarness:
    """自动 Harness 合成器（概念原型）"""

    def __init__(self, meta_llm):
        """
        meta_llm: 用来生成 Harness 配置的元 LLM
        （用 AI 来配置 AI 的 Harness）
        """
        self.meta_llm = meta_llm

    def synthesize(self, task_description: str) -> dict:
        """从任务描述自动生成 Harness 配置"""
        # Step 1: 分析任务特征
        features = self._analyze_task(task_description)

        # Step 2: 选择编排模式
        orchestration = self._select_orchestration(features)

        # Step 3: 生成评估方案
        evaluation = self._generate_eval_plan(features)

        # Step 4: 配置护栏
        guardrails = self._configure_guardrails(features)

        # Step 5: 组装完整配置
        config = {
            "task_features": features,
            "orchestration": orchestration,
            "evaluation": evaluation,
            "guardrails": guardrails,
            "observability": self._default_observability(),
        }

        return config

    def _analyze_task(self, description: str) -> dict:
        """分析任务特征"""
        prompt = f"""分析以下 AI 任务的特征：

任务描述：{description}

请输出 JSON，包含：
- complexity: low/medium/high
- requires_verification: true/false
- multi_step: true/false
- safety_sensitive: true/false
- output_type: text/code/structured_data
- domain: 领域描述
- error_cost: low/medium/high/critical
"""
        response = self.meta_llm.generate(prompt)
        import json
        return json.loads(response)

    def _select_orchestration(self, features: dict) -> dict:
        """根据任务特征选择编排模式"""
        if features.get("complexity") == "low" and not features.get("multi_step"):
            return {
                "type": "single_agent",
                "model": "haiku",
                "reason": "简单任务，单 agent 足够",
            }
        elif features.get("requires_verification"):
            return {
                "type": "three_agent",
                "agents": ["planner", "generator", "evaluator"],
                "model": "sonnet",
                "reason": "需要验证，使用 Planner+Generator+Evaluator",
            }
        else:
            return {
                "type": "dual_agent",
                "agents": ["generator", "reviewer"],
                "model": "sonnet",
                "reason": "中等复杂度，生成+审查",
            }

    def _generate_eval_plan(self, features: dict) -> dict:
        """生成评估方案"""
        eval_plan = {
            "metrics": ["quality_score"],
            "threshold": 0.8,
        }

        if features.get("output_type") == "code":
            eval_plan["metrics"].extend(["syntax_check", "test_pass_rate"])
        if features.get("safety_sensitive"):
            eval_plan["metrics"].extend(["safety_score", "bias_check"])
            eval_plan["threshold"] = 0.9
        if features.get("output_type") == "structured_data":
            eval_plan["metrics"].append("format_validity")

        return eval_plan

    def _configure_guardrails(self, features: dict) -> dict:
        """配置护栏"""
        guardrails = {
            "input": ["prompt_injection_detection", "length_limit"],
            "output": ["format_check"],
        }

        if features.get("safety_sensitive"):
            guardrails["input"].extend(["toxicity_filter", "topic_boundary"])
            guardrails["output"].extend(["pii_redaction", "safety_filter"])

        if features.get("error_cost") in ("high", "critical"):
            guardrails["output"].append("human_approval_gate")

        return guardrails

    def _default_observability(self) -> dict:
        return {
            "tracing": "opentelemetry",
            "metrics": ["latency", "token_count", "quality_score"],
            "alerting": {"quality_drop": 0.1, "error_rate": 0.05},
        }
```

### AutoHarness 的局限

```
AutoHarness 能做的：                  AutoHarness 做不好的：
✓ 基于任务特征推荐编排模式            ✗ 领域特定的约束发现
✓ 生成标准化的评估配置                ✗ 微妙的安全边界定义
✓ 配置通用护栏                        ✗ 基于失败经验的改进
✓ 设置可观测性基础设施                ✗ 成本-质量权衡的精细调优
✓ 提供起步配置                        ✗ 替代人类的判断和审核

结论：AutoHarness 是好的起点，但 Harness Engineering 仍需要人类专家
```

---

## 24.6 开放问题与推荐学习路径

### 开放问题：Harness Engineering 的归属

```
问题：Harness Engineering 会成为独立学科还是回归软件工程？

观点 A: 独立学科                    观点 B: 回归软件工程
┌───────────────────────┐          ┌───────────────────────┐
│ - AI 系统有独特挑战    │          │ - 核心思想（测试、    │
│   (非确定性、漂移)    │          │   约束、监控）来自    │
│ - 需要专门工具链      │          │   传统 SE              │
│ - 评估方法论完全不同  │          │ - 最终会被吸收为      │
│ - 需要专门培训的      │          │   "AI 时代的软件      │
│   工程师              │          │   工程实践"            │
└───────────────────────┘          └───────────────────────┘

现实答案：短期独立（因为需要专门关注），长期融合（因为所有软件都会有 AI）
```

### 未解决的核心难题

| 难题 | 当前状态 | 关键挑战 |
|------|---------|---------|
| 评估的评估 | 用 LLM 评估 LLM，谁评估评估者？ | 无限递归问题 |
| 非确定性测试 | 属性检查 + 统计，但不完备 | 什么时候"足够"测试了？ |
| 成本天花板 | Harness 增加了 2-5x 成本 | 有没有零成本的约束方式？ |
| 通用 vs 专用 | 通用 Harness 质量有限 | 定制化的自动化 |
| 实时安全 | 事后检测 vs 事前阻止 | 延迟-安全权衡 |
| Agent 自主性 | 太自由则失控，太约束则无用 | 动态边界的平衡 |

### 学完本教程后的推荐学习路径

```
                    ┌────────────────────────┐
                    │   你现在的位置：        │
                    │   Harness Engineering  │
                    │   从零到高阶 完成！     │
                    └───────────┬────────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
                ▼               ▼               ▼
        ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
        │ 深入方向 A   │ │ 深入方向 B   │ │ 深入方向 C   │
        │ Agent 架构   │ │ 评估工程     │ │ 生产运维     │
        │              │ │              │ │              │
        │ - Anthropic  │ │ - DeepEval   │ │ - OTel for  │
        │   Agent SDK  │ │   深度使用   │ │   LLM       │
        │ - CrewAI /   │ │ - 评估集构建 │ │ - K8s AI    │
        │   Autogen    │ │   方法论     │ │   Workloads │
        │ - Tool use   │ │ - Human eval │ │ - 成本优化  │
        │   最佳实践   │ │   设计       │ │   高级技术  │
        └──────────────┘ └──────────────┘ └──────────────┘
                │               │               │
                ▼               ▼               ▼
        ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
        │ 高级方向 A   │ │ 高级方向 B   │ │ 高级方向 C   │
        │ 多模态 Agent │ │ 对抗鲁棒性   │ │ AI 安全      │
        │              │ │              │ │              │
        │ - Vision +   │ │ - Red        │ │ - NIST AI    │
        │   Code agent │ │   teaming    │ │   Standards  │
        │ - Audio      │ │ - Adversarial│ │ - EU AI Act  │
        │   processing │ │   testing    │ │   实施指南   │
        │ - 端到端     │ │ - Jailbreak  │ │ - 企业 AI    │
        │   系统设计   │ │   防御       │ │   治理框架   │
        └──────────────┘ └──────────────┘ └──────────────┘
```

### 推荐阅读资源

```
论文：
1. Anthropic - "Building Effective Agents" (2024)
2. Google - "Agents" white paper (2025)
3. NIST - "AI Agent Standards Initiative" (2026.02)

工具文档：
1. Anthropic Claude Agent SDK
2. DeepEval 评估框架
3. OpenTelemetry LLM Semantic Conventions
4. Langfuse 可观测性平台

实践社区：
1. AI Engineering World's Fair
2. LLM Ops 社区
3. Harness Engineering 讨论组（新兴）
```

---

## 本章小结

| 概念 | 核心要点 |
|------|----------|
| 模型漂移解药 | Harness 让系统在模型更迭中保持稳定 |
| 平台收敛 | 编排 + 评估 + 可观测性将合并为统一平台 |
| Managed Harness | 从自建走向声明式配置 + 平台托管 |
| 长寿性 | 约束和评估比 prompt 技巧和模型版本长寿得多 |
| AutoHarness | AI 生成 Harness 配置，好的起点但不能替代专家 |
| 学科归属 | 短期独立发展，长期融入软件工程 |

---

## 动手实验

### 实验 1：AutoHarness 原型

**目标**：实现一个能从任务描述自动生成 Harness 配置的系统。

```python
# 步骤：
# 1. 实现 AutoHarness 类
# 2. 输入："构建一个内部知识库问答系统"
# 3. 验证生成的配置包含：RAG 组件、评估方案、护栏
# 4. 输入："构建一个代码审查 agent"
# 5. 验证生成的配置使用三 agent 架构
# 6. 对比两个配置的差异
```

**验收标准**：
- 不同任务生成不同的配置
- 安全敏感任务有额外的护栏
- 配置格式正确且可执行

### 实验 2：模型无关性验证

**目标**：验证同一个 Harness 能在不同模型上运行。

**步骤**：
1. 实现 ModelAgnosticHarness
2. 创建两个模型适配器（如 Haiku 和 Sonnet）
3. 用同一个任务分别运行
4. 验证 Harness 逻辑（护栏、评估）在两个模型上都正常工作

### 实验 3：Harness 投资回报分析

**目标**：量化 Harness 工程投资在多次模型切换后的回报。

```python
# 模拟 3 次模型切换
# 没有 Harness：每次 40h 返工
# 有 Harness：首次 80h 建设 + 每次 8h 适配

def simulate_roi(switches: int, harness_build_hours: int = 80,
                  per_switch_adapt: int = 8,
                  no_harness_per_switch: int = 40,
                  hourly_rate: int = 75) -> dict:
    with_harness = (harness_build_hours + switches * per_switch_adapt) * hourly_rate
    without_harness = switches * no_harness_per_switch * hourly_rate
    return {
        "with_harness": with_harness,
        "without_harness": without_harness,
        "break_even_at_switch": 3,  # 计算实际值
    }
```

---

## 练习题

### 基础题

1. **概念题**：为什么说 "Harness 比 Prompt 技巧长寿"？举出 3 个从 2023 到 2026 年失效的 Prompt 技巧和 3 个依然有效的 Harness 设计原则。

2. **趋势题**：解释"编排、评估、可观测性平台收敛"的 4 个驱动力。

3. **计算题**：一个团队建设 Harness 花了 120 小时。之后每次模型切换只需 10 小时适配。如果没有 Harness，每次切换需要 50 小时。在第几次模型切换后 Harness 投资开始回本？

### 实践题

4. **AutoHarness 评估**：用 AutoHarness 生成 3 种不同任务的配置，人工评审生成质量。记录哪些部分是准确的，哪些需要人工修正。这个数据本身就是改进 AutoHarness 的训练数据。

5. **模型切换演练**：设计一个完整的"模型切换演练"方案。包括：切换前检查清单、切换中的 eval suite 运行、切换后的监控计划。

### 思考题

6. **自动化的悖论**：如果 AI 最终强大到可以自己编写完美的 Harness（AutoHarness 达到 100% 准确），那 Harness Engineering 作为一个职业是否会消失？讨论"自动化工具的可信度"和"人类判断的不可替代性"。

7. **终极问题**：Harness Engineering 的本质是什么？它是一种"对 AI 的不信任"，还是一种"对 AI 的赋能"？如果 AI 变得完全可靠，还需要 Harness 吗？请从"没有绝对可靠的系统"这个工程哲学角度讨论。

---

## 全教程总结

恭喜你完成了 **Harness Engineering 从零到高阶** 全部 24 章的学习！

### 回顾：从第 1 章到第 24 章

```
Part 1: 基础         Part 2: 评估          Part 3: 测试与 CI
理解 Harness 是什么   学会评估 AI 输出       自动化评估流水线

Part 4: 编排         Part 5: 生产与前沿
多 agent + 长时运行   安全、可观测性、未来
+ RAG 生产系统       + 持续改进
```

### 核心理念一句话

**Harness Engineering 的本质是：用确定性的工程方法约束和引导非确定性的 AI 系统。**

这个理念今天成立，在可预见的未来也将继续成立。模型会变，框架会变，但让 AI 可靠工作的需求不会变。
