# 第21章：部署 Harness 与流水线

> 传统 CI/CD 的世界观是确定性的：同样的代码 + 同样的配置 = 同样的行为。AI 系统彻底打破了这个假设——同样的 prompt + 同样的模型 ≠ 同样的输出。当你的部署对象从"代码"变成"行为"，整个流水线需要重新设计。

---

## 学习目标

学完本章，你将能够：

1. 理解传统 CI/CD 为什么对非确定性系统不适用
2. 掌握 AI 部署多层栈的结构
3. 设计 agent 作为一等流水线步骤的 CI/CD
4. 实现持续评估（Continuous Evaluation）机制
5. 应用 Canary 部署和灰度发布到 AI 系统

---

## 21.1 传统 CI/CD 为什么对非确定性系统崩溃

### 确定性假设的崩塌

```
传统软件：                         AI 系统：
Code v1.2 → Build → Test → Deploy  Prompt v3 + Model v2 + RAG index v7
  │                                    │
  ▼                                    ▼
确定性输出                           非确定性输出
  │                                    │
  ▼                                    ▼
测试通过 = 生产安全                   测试通过 ≠ 生产安全
                                    （因为同样的 prompt 可能给出不同的回答）
```

### 传统 CI/CD 的盲区

| CI/CD 假设 | 在 AI 系统中的现实 | 后果 |
|-----------|-------------------|------|
| 代码变更 = 行为变更 | 模型更新无代码变更也改行为 | 漏检回归 |
| 测试通过 = 安全部署 | 10 次测试通过不代表第 11 次也对 | 假安全感 |
| 回滚 = 恢复 | Prompt 回滚但模型已更新 | 回滚无效 |
| 构建是幂等的 | 相同输入不同输出 | 构建不可重现 |
| 依赖是固定的 | 模型 API 随时可能调整行为 | 隐式依赖 |

### 需要什么样的新流水线

```
传统 CI/CD:
  git push → build → unit test → integration test → deploy

AI CI/CD:
  git push ──────────────────────────────────────────→
       │                                              │
       ▼                                              │
  build + lint ───→ eval suite ───→ comparison ──→ deploy
       │                │               │             │
       │           ┌────┴────┐    ┌────┴────┐         │
       │           │ prompt  │    │ vs prev │    canary│
       │           │ eval    │    │ version │    ──────│
       │           │ RAG eval│    │ quality │    shadow│
       │           │ safety  │    │ diff    │    ──────│
       │           └─────────┘    └─────────┘    full  │
       │                                              │
       └──── continuous eval (sensors) ←──────────────┘
```

---

## 21.2 AI 部署多层栈

### 六层部署栈

AI 应用不是一个单一的"二进制"，而是一个多层栈——每一层都可以独立变更：

```
┌─────────────────────────────────────────────┐
│  Layer 6: Agents / Orchestration            │  ← agent 逻辑
│  Harness 配置、agent 拓扑、工具权限           │
├─────────────────────────────────────────────┤
│  Layer 5: Guardrails / Safety               │  ← 安全层
│  输入/输出过滤规则、话题边界                  │
├─────────────────────────────────────────────┤
│  Layer 4: RAG / Context Engine              │  ← 检索层
│  Embedding 模型、向量索引、reranking 配置     │
├─────────────────────────────────────────────┤
│  Layer 3: Prompts / Templates               │  ← Prompt 层
│  System prompts、few-shot examples           │
├─────────────────────────────────────────────┤
│  Layer 2: Inference Engine                  │  ← 推理层
│  模型选择、温度、采样参数                     │
├─────────────────────────────────────────────┤
│  Layer 1: Application Code                  │  ← 应用层
│  API 路由、数据处理、业务逻辑                 │
└─────────────────────────────────────────────┘
```

### 各层的变更频率和风险

```python
DEPLOYMENT_LAYERS = {
    "application_code": {
        "change_frequency": "weekly",
        "risk_level": "medium",
        "rollback_ease": "easy",
        "test_strategy": "unit + integration tests",
    },
    "inference_engine": {
        "change_frequency": "monthly",
        "risk_level": "high",
        "rollback_ease": "easy",  # 切回旧模型即可
        "test_strategy": "eval suite + shadow deployment",
    },
    "prompts": {
        "change_frequency": "daily",
        "risk_level": "high",
        "rollback_ease": "easy",
        "test_strategy": "eval suite + A/B testing",
    },
    "rag_context": {
        "change_frequency": "daily",
        "risk_level": "medium",
        "rollback_ease": "medium",  # 需要重建索引
        "test_strategy": "retrieval eval + end-to-end",
    },
    "guardrails": {
        "change_frequency": "weekly",
        "risk_level": "high",
        "rollback_ease": "easy",
        "test_strategy": "adversarial test suite",
    },
    "agents": {
        "change_frequency": "weekly",
        "risk_level": "critical",
        "rollback_ease": "hard",  # 状态依赖复杂
        "test_strategy": "full pipeline eval + canary",
    },
}
```

---

## 21.3 Agent 作为一等流水线步骤

### Harness.io 方法论

Harness.io 在 2025-2026 年推出了将 AI agent 集成到 CI/CD 流水线中的方法——agent 不再是"被部署的对象"，而是"参与部署的主体"：

```yaml
# harness-pipeline.yaml
pipeline:
  name: ai-app-deployment
  stages:
    - stage:
        name: Code Quality
        type: CI
        steps:
          - step:
              type: AIAgent
              name: Code Review Agent
              spec:
                agent: code-reviewer
                model: claude-sonnet-4-20250514
                task: "审查 PR 中的代码变更"
                pass_criteria:
                  - "无安全漏洞"
                  - "测试覆盖率 > 80%"

    - stage:
        name: Eval Suite
        type: Custom
        steps:
          - step:
              type: AIEval
              name: Prompt Evaluation
              spec:
                eval_suite: prompt-regression-v2
                baseline: last-release
                threshold:
                  quality_score: 0.85
                  regression_tolerance: 0.05

          - step:
              type: AIEval
              name: Safety Evaluation
              spec:
                eval_suite: safety-adversarial-v1
                threshold:
                  pass_rate: 0.99

    - stage:
        name: Canary Deploy
        type: Deployment
        spec:
          strategy: canary
          phases:
            - traffic_percent: 5
              duration: 30m
              success_criteria:
                error_rate: < 1%
                quality_score: > 0.8
            - traffic_percent: 25
              duration: 2h
              success_criteria:
                error_rate: < 1%
                quality_score: > 0.8
            - traffic_percent: 100
```

### 从仓库扫描自动创建流水线

```python
class PipelineAutoGenerator:
    """扫描仓库自动生成部署流水线"""

    def __init__(self, repo_path: str):
        self.repo_path = repo_path

    def scan_and_generate(self) -> dict:
        """扫描仓库结构，推断需要的流水线步骤"""
        detected = {
            "has_prompts": self._detect_prompts(),
            "has_rag": self._detect_rag(),
            "has_agents": self._detect_agents(),
            "has_guardrails": self._detect_guardrails(),
            "languages": self._detect_languages(),
        }

        pipeline = self._build_pipeline(detected)
        return pipeline

    def _detect_prompts(self) -> bool:
        """检测是否有 prompt 模板"""
        import glob
        patterns = [
            f"{self.repo_path}/**/prompts/**",
            f"{self.repo_path}/**/*prompt*",
            f"{self.repo_path}/**/*system_message*",
        ]
        for pattern in patterns:
            if glob.glob(pattern, recursive=True):
                return True
        return False

    def _detect_rag(self) -> bool:
        """检测是否有 RAG 组件"""
        rag_indicators = [
            "vector", "embedding", "retriev", "chromadb",
            "pinecone", "weaviate", "qdrant",
        ]
        # 简化：检查依赖文件
        return self._check_dependencies(rag_indicators)

    def _detect_agents(self) -> bool:
        """检测是否有 agent 组件"""
        agent_indicators = ["agent", "tool_use", "function_calling"]
        return self._check_dependencies(agent_indicators)

    def _detect_guardrails(self) -> bool:
        """检测是否有 guardrail 组件"""
        guardrail_indicators = ["guardrail", "safety", "content_filter"]
        return self._check_dependencies(guardrail_indicators)

    def _detect_languages(self) -> list[str]:
        """检测编程语言"""
        import os
        extensions = set()
        for root, dirs, files in os.walk(self.repo_path):
            for f in files:
                _, ext = os.path.splitext(f)
                if ext in (".py", ".js", ".ts", ".go", ".rs"):
                    extensions.add(ext)
        return list(extensions)

    def _check_dependencies(self, indicators: list[str]) -> bool:
        """检查依赖文件中是否包含指标关键词"""
        import os
        dep_files = ["requirements.txt", "pyproject.toml", "package.json"]
        for dep_file in dep_files:
            path = os.path.join(self.repo_path, dep_file)
            if os.path.exists(path):
                with open(path) as f:
                    content = f.read().lower()
                for indicator in indicators:
                    if indicator in content:
                        return True
        return False

    def _build_pipeline(self, detected: dict) -> dict:
        """根据检测结果构建流水线"""
        stages = [{"name": "Code Quality", "steps": ["lint", "type_check", "unit_test"]}]

        if detected["has_prompts"]:
            stages.append({
                "name": "Prompt Eval",
                "steps": ["prompt_regression", "quality_check"],
            })

        if detected["has_rag"]:
            stages.append({
                "name": "RAG Eval",
                "steps": ["retrieval_quality", "end_to_end_qa"],
            })

        if detected["has_guardrails"]:
            stages.append({
                "name": "Safety Eval",
                "steps": ["adversarial_test", "toxicity_check"],
            })

        if detected["has_agents"]:
            stages.append({
                "name": "Agent Eval",
                "steps": ["agent_task_completion", "tool_use_safety"],
            })

        stages.append({
            "name": "Deploy",
            "steps": ["canary_5%", "canary_25%", "full_rollout"],
        })

        return {"stages": stages, "detected_features": detected}
```

---

## 21.4 持续评估：Sensors 在变更生命周期之外运行

### 问题：部署成功 ≠ 持续正确

```
传统思维：                         AI 现实：
部署成功 → 监控错误率 → 没报警     部署成功 → 监控错误率 → 没报警
         → 一切正常                           → 但质量在悄悄下降

      ┌─────────────────────────────────────────→ 时间
      │
质量  │ ████████████████                ← 传统指标看不到
 1.0  │ ████████████████████
      │ ████████████████████████
 0.8  │ ████████████████████████████
      │ ████████████████████████████████
 0.6  │ ████████████████████████████████████  ← 质量已经不可接受
      │
      └─────────────────────────────────────
        部署    1天     1周     1月
```

### Continuous Evaluation 架构

```python
import time
import threading
from datetime import datetime

class ContinuousEvaluationSensor:
    """持续评估 Sensor：独立于部署周期运行"""

    def __init__(self, eval_suite, alert_manager, interval_seconds: int = 300):
        self.eval_suite = eval_suite
        self.alert_manager = alert_manager
        self.interval = interval_seconds
        self.running = False
        self.history: list[dict] = []

    def start(self):
        """启动持续评估"""
        self.running = True
        self._run_loop()

    def stop(self):
        """停止持续评估"""
        self.running = False

    def _run_loop(self):
        """评估主循环"""
        while self.running:
            result = self._run_evaluation()
            self.history.append(result)

            # 检查是否需要报警
            if result["score"] < result["threshold"]:
                self.alert_manager.alert(
                    severity="HIGH",
                    message=f"质量分数 {result['score']:.2f} 低于阈值 {result['threshold']:.2f}",
                    details=result,
                )

            # 检查趋势
            if len(self.history) >= 5:
                trend = self._detect_trend()
                if trend["declining"]:
                    self.alert_manager.alert(
                        severity="MEDIUM",
                        message=f"质量呈下降趋势：{trend['slope']:.4f}/次",
                        details=trend,
                    )

            time.sleep(self.interval)

    def _run_evaluation(self) -> dict:
        """运行一次评估"""
        # 从生产流量中采样
        samples = self.eval_suite.sample_production_traffic(n=10)

        scores = []
        for sample in samples:
            score = self.eval_suite.evaluate(sample)
            scores.append(score)

        return {
            "timestamp": datetime.now().isoformat(),
            "score": sum(scores) / len(scores) if scores else 0,
            "threshold": 0.8,
            "sample_count": len(samples),
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
        }

    def _detect_trend(self) -> dict:
        """检测质量趋势"""
        recent = self.history[-5:]
        scores = [r["score"] for r in recent]
        # 简单线性回归斜率
        n = len(scores)
        x_mean = (n - 1) / 2
        y_mean = sum(scores) / n
        numerator = sum((i - x_mean) * (s - y_mean) for i, s in enumerate(scores))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        slope = numerator / max(denominator, 1e-8)

        return {
            "declining": slope < -0.01,
            "slope": slope,
            "recent_scores": scores,
        }
```

---

## 21.5 Canary 部署和灰度发布

### AI 系统的 Canary 部署

```
Phase 1: Shadow（0% 真实流量）
┌──────────┐     ┌──────────┐
│ 旧版本   │────→│ 用户     │
│ (v1)     │     └──────────┘
└──────────┘
┌──────────┐     ┌──────────┐
│ 新版本   │────→│ 只记录   │  ← 不影响用户，只比较
│ (v2)     │     │ 不返回   │
└──────────┘     └──────────┘

Phase 2: Canary（5% 真实流量）
┌──────────┐     ┌──────────┐
│ v1 (95%) │────→│ 95% 用户 │
└──────────┘     └──────────┘
┌──────────┐     ┌──────────┐
│ v2 (5%)  │────→│ 5% 用户  │  ← 真实流量，监控质量
└──────────┘     └──────────┘

Phase 3: 逐步扩大
  5% → 25% → 50% → 100%
  每步至少观察 30 分钟
  任何阶段质量下降 → 自动回滚
```

### Canary 部署控制器

```python
from enum import Enum

class DeployPhase(Enum):
    SHADOW = "shadow"
    CANARY_5 = "canary_5"
    CANARY_25 = "canary_25"
    CANARY_50 = "canary_50"
    FULL = "full"
    ROLLED_BACK = "rolled_back"

class CanaryController:
    """AI 系统 Canary 部署控制器"""

    def __init__(self, evaluator, min_observation_minutes: int = 30):
        self.evaluator = evaluator
        self.min_observation = min_observation_minutes
        self.current_phase = DeployPhase.SHADOW
        self.phase_results: dict[str, dict] = {}

    def advance_phase(self) -> dict:
        """推进到下一阶段"""
        phases = list(DeployPhase)
        current_idx = phases.index(self.current_phase)

        if current_idx >= len(phases) - 2:  # 已经是 FULL
            return {"status": "already_at_full"}

        # 评估当前阶段
        eval_result = self._evaluate_current_phase()

        if eval_result["pass"]:
            next_phase = phases[current_idx + 1]
            self.current_phase = next_phase
            return {
                "status": "advanced",
                "from": phases[current_idx].value,
                "to": next_phase.value,
                "eval_result": eval_result,
            }
        else:
            # 不通过 → 回滚
            self.current_phase = DeployPhase.ROLLED_BACK
            return {
                "status": "rolled_back",
                "reason": eval_result["reason"],
                "eval_result": eval_result,
            }

    def _evaluate_current_phase(self) -> dict:
        """评估当前阶段是否可以推进"""
        metrics = self.evaluator.get_current_metrics()

        checks = {
            "error_rate_ok": metrics.get("error_rate", 1.0) < 0.01,
            "quality_ok": metrics.get("quality_score", 0) > 0.8,
            "latency_ok": metrics.get("p99_latency_ms", 10000) < 5000,
            "no_safety_issues": metrics.get("safety_violations", 1) == 0,
        }

        all_pass = all(checks.values())
        failed = [k for k, v in checks.items() if not v]

        return {
            "pass": all_pass,
            "checks": checks,
            "reason": f"未通过检查: {', '.join(failed)}" if not all_pass else "全部通过",
        }

    def get_traffic_split(self) -> dict:
        """获取当前流量分配"""
        splits = {
            DeployPhase.SHADOW: {"v1": 100, "v2": 0, "shadow": True},
            DeployPhase.CANARY_5: {"v1": 95, "v2": 5},
            DeployPhase.CANARY_25: {"v1": 75, "v2": 25},
            DeployPhase.CANARY_50: {"v1": 50, "v2": 50},
            DeployPhase.FULL: {"v1": 0, "v2": 100},
            DeployPhase.ROLLED_BACK: {"v1": 100, "v2": 0},
        }
        return {
            "phase": self.current_phase.value,
            "split": splits[self.current_phase],
        }
```

---

## 21.6 完整部署流水线设计

### 端到端流水线

```python
class AIDeploymentPipeline:
    """AI 应用完整部署流水线"""

    def __init__(self, config: dict):
        self.config = config
        self.stages = []

    def run(self, change: dict) -> dict:
        """执行完整部署流水线"""
        results = []

        # Stage 1: 代码质量
        result = self._stage_code_quality(change)
        results.append(result)
        if not result["pass"]:
            return {"status": "failed", "stage": "code_quality", "results": results}

        # Stage 2: 评估套件
        result = self._stage_eval_suite(change)
        results.append(result)
        if not result["pass"]:
            return {"status": "failed", "stage": "eval_suite", "results": results}

        # Stage 3: 对比分析
        result = self._stage_comparison(change)
        results.append(result)
        if not result["pass"]:
            return {"status": "failed", "stage": "comparison", "results": results}

        # Stage 4: Canary 部署
        result = self._stage_canary_deploy(change)
        results.append(result)
        if not result["pass"]:
            return {"status": "rolled_back", "stage": "canary", "results": results}

        # Stage 5: 启动持续评估
        self._start_continuous_eval()

        return {"status": "deployed", "results": results}

    def _stage_code_quality(self, change: dict) -> dict:
        """Stage 1: 代码质量检查"""
        checks = {
            "lint": True,  # 运行 linter
            "type_check": True,  # 类型检查
            "unit_tests": True,  # 单元测试
            "security_scan": True,  # 安全扫描
        }
        return {"stage": "code_quality", "pass": all(checks.values()), "checks": checks}

    def _stage_eval_suite(self, change: dict) -> dict:
        """Stage 2: AI 评估套件"""
        evals = {
            "prompt_quality": 0.88,
            "retrieval_quality": 0.85,
            "safety_score": 0.99,
            "hallucination_rate": 0.03,
        }
        thresholds = {
            "prompt_quality": 0.80,
            "retrieval_quality": 0.80,
            "safety_score": 0.95,
            "hallucination_rate": 0.05,  # 越低越好
        }
        pass_check = all(
            evals[k] >= thresholds[k] if k != "hallucination_rate"
            else evals[k] <= thresholds[k]
            for k in evals
        )
        return {"stage": "eval_suite", "pass": pass_check, "scores": evals}

    def _stage_comparison(self, change: dict) -> dict:
        """Stage 3: 与前一版本对比"""
        return {
            "stage": "comparison",
            "pass": True,
            "regression_detected": False,
            "improvement": "+2.3% quality",
        }

    def _stage_canary_deploy(self, change: dict) -> dict:
        """Stage 4: Canary 部署"""
        return {"stage": "canary", "pass": True, "final_phase": "full"}

    def _start_continuous_eval(self):
        """Stage 5: 启动持续评估"""
        pass  # 启动后台 sensor
```

---

## 本章小结

| 概念 | 核心要点 |
|------|----------|
| 确定性崩塌 | 同代码 ≠ 同行为，传统 CI/CD 假设不成立 |
| 六层部署栈 | 应用代码 → 推理引擎 → Prompt → RAG → Guardrails → Agent |
| Agent 流水线步骤 | Agent 不只被部署，也参与部署过程 |
| 仓库扫描 | 自动检测 prompt/RAG/agent 组件，生成流水线 |
| 持续评估 | Sensor 在部署后持续运行，检测质量退化 |
| Canary 部署 | Shadow → 5% → 25% → 50% → 100%，逐步验证 |

---

## 动手实验

### 实验 1：设计一个 AI 应用的多阶段部署流水线

**目标**：为一个包含 RAG + Agent 的客服系统设计完整部署流水线。

```python
# 步骤：
# 1. 定义 6 层部署栈的具体组件
# 2. 为每层设计变更检测和评估方法
# 3. 实现 AIDeploymentPipeline
# 4. 模拟一次 prompt 变更的完整部署流程
# 5. 模拟一次评估失败的回滚流程
```

**验收标准**：
- 流水线包含至少 4 个阶段
- prompt 变更触发 eval suite
- 评估不通过时能阻止部署

### 实验 2：持续评估 Sensor

**目标**：实现一个 ContinuousEvaluationSensor，能检测质量趋势并报警。

**步骤**：
1. 模拟生产流量（分数逐渐下降）
2. 启动 sensor 运行 5 个周期
3. 验证 sensor 检测到下降趋势
4. 验证 alert 被正确触发

### 实验 3：Canary 部署控制器

**目标**：实现 CanaryController 的完整生命周期。

**步骤**：
1. 从 SHADOW 开始
2. 模拟每个阶段的评估通过/失败
3. 成功路径：SHADOW → CANARY_5 → CANARY_25 → FULL
4. 失败路径：SHADOW → CANARY_5 → ROLLED_BACK

---

## 练习题

### 基础题

1. **概念题**：列举传统 CI/CD 的 3 个核心假设，以及每个假设在 AI 系统中是如何被打破的。

2. **分层题**：AI 部署六层栈中，哪一层的变更频率最高？哪一层的变更风险最大？为什么？

3. **Canary 计算**：一个系统日均 10 万请求。Canary 5% 阶段持续 30 分钟。这 30 分钟内大约有多少请求会走新版本？这个样本量足够做出统计可靠的判断吗？

### 实践题

4. **流水线设计**：为一个 LLM 驱动的代码生成工具设计部署流水线。这个工具的特殊之处在于输出是代码（可以被编译和测试），请利用这一点设计更强的验证。

5. **回滚策略**：设计一个 AI 系统的回滚策略，需要考虑：prompt 版本、模型版本、RAG 索引版本三者不同步的情况。

### 思考题

6. **持续评估的成本**：如果持续评估每 5 分钟用 LLM-as-Judge 评估 10 个样本，每个样本评估消耗 2000 tokens，模型价格 $0.01/1K tokens。一天的持续评估成本是多少？如何优化？

7. **Shadow 部署的局限**：Shadow 部署（新版本并行运行但不返回给用户）看似安全，但对 AI 系统有什么局限？提示：考虑"用户反馈循环"和"交互式对话"场景。
