# 第23章：熵管理与持续改进

> 物理学中的熵增原理告诉我们：一个孤立系统会自发地趋向混乱。AI agent 系统也不例外——在没有持续维护的情况下，输出质量会逐渐退化。Harness Engineering 的核心职责之一就是**对抗熵增**。

---

## 学习目标

学完本章，你将能够：

1. 理解 agent 输出退化（熵增）的根本原因
2. 检测无代码变更时的模型漂移
3. 将 harness 遥测数据反馈到模型改进循环
4. 构建和维护回归测试数据集
5. 设计 harness 持续改进循环

---

## 23.1 熵问题：Agent 输出为什么会逐渐退化

### 什么是 Agent 熵增

```
                    初始部署        3个月后         6个月后
                        │               │               │
系统质量                ▼               ▼               ▼

  1.0  ●
       │ ●
  0.9  │   ●
       │     ●
  0.8  │       ● ● ●
       │               ● ●
  0.7  │                    ●
       │                      ● ●
  0.6  │                           ●
       │                             ●  ← "怎么突然不好用了？"
  0.5  │
       └─────────────────────────────────→ 时间
```

### 退化的六个根因

```
┌──────────────────────────────────────────────────────┐
│            Agent 熵增的六个根因                        │
├──────────────┬───────────────────────────────────────┤
│ 根因         │ 说明                                   │
├──────────────┼───────────────────────────────────────┤
│ 1. 模型漂移  │ 提供商静默更新模型，行为改变             │
│              │ 没有代码变更也会导致质量变化              │
├──────────────┼───────────────────────────────────────┤
│ 2. 数据漂移  │ 用户查询分布变化，超出训练分布            │
│              │ 新话题、新术语、新场景不断出现             │
├──────────────┼───────────────────────────────────────┤
│ 3. 知识过时  │ RAG 中的文档过时                        │
│              │ 模型知识截止日期之后的信息                │
├──────────────┼───────────────────────────────────────┤
│ 4. 上下文累积│ 长会话中错误的上下文逐步累积              │
│              │ 早期的小错误被后续步骤放大                │
├──────────────┼───────────────────────────────────────┤
│ 5. Prompt 腐│ Prompt 与模型版本不匹配                  │
│    化        │ 为旧模型优化的 prompt 在新模型上不最优    │
├──────────────┼───────────────────────────────────────┤
│ 6. 护栏滞后  │ 新的攻击模式出现但护栏未更新              │
│              │ 新的偏差模式未被检测到                    │
└──────────────┴───────────────────────────────────────┘
```

### 量化熵增

```python
from datetime import datetime, timedelta

class EntropyMetrics:
    """Agent 熵增度量"""

    def __init__(self, quality_history: list[dict]):
        """
        quality_history = [
            {"date": "2026-01-01", "score": 0.92},
            {"date": "2026-01-15", "score": 0.89},
            ...
        ]
        """
        self.history = sorted(quality_history, key=lambda x: x["date"])

    def entropy_rate(self) -> float:
        """计算熵增速率（质量下降速度）"""
        if len(self.history) < 2:
            return 0.0

        scores = [h["score"] for h in self.history]
        n = len(scores)
        x_mean = (n - 1) / 2
        y_mean = sum(scores) / n
        numerator = sum((i - x_mean) * (s - y_mean) for i, s in enumerate(scores))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        slope = numerator / max(denominator, 1e-8)

        return slope  # 负值表示退化

    def time_to_threshold(self, threshold: float = 0.7) -> int | None:
        """预测多少天后质量会降到阈值以下"""
        rate = self.entropy_rate()
        if rate >= 0:
            return None  # 没有退化趋势

        current_score = self.history[-1]["score"]
        if current_score <= threshold:
            return 0  # 已经低于阈值

        days_per_period = 14  # 假设每 14 天一个数据点
        periods_to_threshold = (current_score - threshold) / abs(rate)
        return int(periods_to_threshold * days_per_period)

    def detect_sudden_drop(self, window: int = 3, threshold: float = 0.1) -> list[dict]:
        """检测突然质量下降"""
        drops = []
        scores = [h["score"] for h in self.history]

        for i in range(window, len(scores)):
            window_avg = sum(scores[i - window:i]) / window
            if window_avg - scores[i] > threshold:
                drops.append({
                    "date": self.history[i]["date"],
                    "score": scores[i],
                    "window_avg": window_avg,
                    "drop": window_avg - scores[i],
                })

        return drops

    def report(self) -> dict:
        """生成熵增报告"""
        rate = self.entropy_rate()
        ttl = self.time_to_threshold()
        drops = self.detect_sudden_drop()

        return {
            "current_score": self.history[-1]["score"] if self.history else None,
            "entropy_rate": rate,
            "interpretation": "退化中" if rate < -0.01 else "稳定" if abs(rate) <= 0.01 else "改善中",
            "time_to_critical": f"{ttl} 天" if ttl else "无退化趋势",
            "sudden_drops": len(drops),
            "recommendation": self._recommend(rate, ttl, drops),
        }

    def _recommend(self, rate, ttl, drops) -> str:
        if drops:
            return "检测到突然质量下降，建议立即调查最近的模型或数据变更"
        if ttl and ttl < 30:
            return f"预计 {ttl} 天后质量将低于阈值，建议立即启动 harness 改进"
        if rate < -0.02:
            return "退化速度较快，建议检查模型版本和数据分布变化"
        if rate < 0:
            return "轻微退化，建议增加监控频率"
        return "系统运行正常"
```

---

## 23.2 检测无代码变更时的模型漂移

### 模型漂移的隐蔽性

```
你的代码仓库：                    模型提供商：
┌────────────────┐               ┌────────────────┐
│ git log        │               │ 悄悄更新模型   │
│ 最近无提交     │               │ claude-sonnet   │
│ 代码没变       │               │ v20250501 →    │
│                │               │ v20250601      │
│ "什么都没改"   │               │                │
│                │               │ 行为有细微变化 │
└────────────────┘               └────────────────┘
         │                                │
         └────────── 但输出质量变了 ───────┘
```

### 漂移检测系统

```python
class ModelDriftDetector:
    """模型漂移检测器"""

    def __init__(self, probe_suite: list[dict], llm_client):
        """
        probe_suite: 一组固定的探针请求和期望输出特征
        [
            {
                "id": "probe_001",
                "input": "1+1等于几？",
                "expected_properties": {
                    "contains": ["2"],
                    "max_length": 50,
                    "tone": "factual",
                },
            },
            ...
        ]
        """
        self.probes = probe_suite
        self.llm = llm_client
        self.baseline: dict[str, dict] = {}

    def capture_baseline(self, runs: int = 5) -> dict:
        """采集基线（部署时运行一次）"""
        for probe in self.probes:
            responses = []
            for _ in range(runs):
                response = self.llm.generate([
                    {"role": "user", "content": probe["input"]},
                ])
                responses.append(response)

            self.baseline[probe["id"]] = {
                "responses": responses,
                "avg_length": sum(len(r) for r in responses) / len(responses),
                "consistency": self._measure_consistency(responses),
            }

        return self.baseline

    def check_drift(self, runs: int = 5) -> dict:
        """运行漂移检测"""
        if not self.baseline:
            return {"error": "请先运行 capture_baseline()"}

        drifts = []
        for probe in self.probes:
            baseline = self.baseline.get(probe["id"])
            if not baseline:
                continue

            # 获取当前响应
            current_responses = []
            for _ in range(runs):
                response = self.llm.generate([
                    {"role": "user", "content": probe["input"]},
                ])
                current_responses.append(response)

            # 对比
            drift = self._compare(probe["id"], baseline, current_responses, probe)
            if drift["drifted"]:
                drifts.append(drift)

        return {
            "total_probes": len(self.probes),
            "drifted_probes": len(drifts),
            "drift_ratio": len(drifts) / max(len(self.probes), 1),
            "details": drifts,
            "verdict": "DRIFT_DETECTED" if drifts else "STABLE",
        }

    def _compare(
        self, probe_id: str, baseline: dict,
        current: list[str], probe: dict
    ) -> dict:
        """对比基线和当前结果"""
        current_avg_len = sum(len(r) for r in current) / len(current)
        baseline_avg_len = baseline["avg_length"]

        # 长度漂移
        length_drift = abs(current_avg_len - baseline_avg_len) / max(baseline_avg_len, 1)

        # 属性检查
        property_failures = 0
        expected = probe.get("expected_properties", {})
        for response in current:
            if "contains" in expected:
                for keyword in expected["contains"]:
                    if keyword not in response:
                        property_failures += 1
            if "max_length" in expected:
                if len(response) > expected["max_length"]:
                    property_failures += 1

        drifted = length_drift > 0.3 or property_failures > len(current) * 0.5

        return {
            "probe_id": probe_id,
            "drifted": drifted,
            "length_drift": length_drift,
            "property_failures": property_failures,
            "baseline_avg_length": baseline_avg_len,
            "current_avg_length": current_avg_len,
        }

    def _measure_consistency(self, responses: list[str]) -> float:
        """测量多次响应的一致性"""
        if len(responses) < 2:
            return 1.0
        # 简化：用长度方差作为代理
        lengths = [len(r) for r in responses]
        mean_len = sum(lengths) / len(lengths)
        variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
        # 归一化到 0-1
        return max(0, 1 - variance / max(mean_len ** 2, 1))
```

---

## 23.3 把 Harness 遥测反馈到改进循环

### 遥测驱动的改进

```
┌──────────────────────────────────────────────────────────┐
│                    遥测反馈循环                            │
│                                                          │
│  生产运行                                                 │
│    │                                                     │
│    ▼                                                     │
│  收集遥测 ──→ 聚类分析 ──→ 根因定位 ──→ Harness 修复     │
│    │              │              │              │         │
│    │         "输出格式错误    "system prompt   "添加格式   │
│    │          占失败的40%"    缺少格式约束"    验证规则"   │
│    │                                             │       │
│    └──────────────── 重新部署 ←───────────────────┘       │
│                                                          │
│  关键原则：修 Harness，不修输出                            │
└──────────────────────────────────────────────────────────┘
```

### 遥测分析器

```python
class TelemetryAnalyzer:
    """遥测分析器：从失败模式中提取 Harness 改进建议"""

    def __init__(self, trace_store):
        self.traces = trace_store

    def analyze_failures(
        self, time_window_hours: int = 168  # 一周
    ) -> dict:
        """分析最近的失败模式"""
        failures = self.traces.query(
            time_window=time_window_hours,
            filter={"quality_score": {"$lt": 0.7}},
        )

        # 聚类失败
        clusters = self._cluster_failures(failures)

        # 为每个聚类生成修复建议
        improvements = []
        for cluster in clusters:
            improvement = self._suggest_improvement(cluster)
            improvements.append(improvement)

        return {
            "total_failures": len(failures),
            "clusters": clusters,
            "improvements": improvements,
            "priority_order": sorted(
                improvements, key=lambda x: x["impact"], reverse=True
            ),
        }

    def _cluster_failures(self, failures: list[dict]) -> list[dict]:
        """聚类失败模式"""
        patterns = {}
        for f in failures:
            # 提取失败特征
            key = self._extract_failure_signature(f)
            if key not in patterns:
                patterns[key] = {"signature": key, "count": 0, "examples": []}
            patterns[key]["count"] += 1
            if len(patterns[key]["examples"]) < 3:
                patterns[key]["examples"].append(f)

        return sorted(patterns.values(), key=lambda x: x["count"], reverse=True)

    def _extract_failure_signature(self, failure: dict) -> str:
        """提取失败签名"""
        # 简化：基于失败类型分类
        if "format" in str(failure.get("error", "")).lower():
            return "format_error"
        elif "hallucination" in str(failure.get("checks", {})):
            return "hallucination"
        elif "safety" in str(failure.get("checks", {})):
            return "safety_violation"
        elif "timeout" in str(failure.get("error", "")).lower():
            return "timeout"
        return "unknown"

    def _suggest_improvement(self, cluster: dict) -> dict:
        """为一个失败聚类生成改进建议"""
        suggestions = {
            "format_error": {
                "harness_fix": "在 system prompt 中添加更严格的格式约束",
                "sensor_fix": "添加输出格式验证 sensor",
                "type": "prompt + guardrail",
            },
            "hallucination": {
                "harness_fix": "增强 RAG 检索质量，添加引用要求",
                "sensor_fix": "添加 MetaQA 幻觉检测 sensor",
                "type": "rag + evaluation",
            },
            "safety_violation": {
                "harness_fix": "更新安全护栏规则",
                "sensor_fix": "增加对抗性测试用例",
                "type": "guardrail",
            },
            "timeout": {
                "harness_fix": "优化 prompt 减少输出长度",
                "sensor_fix": "添加超时预警",
                "type": "performance",
            },
        }
        sig = cluster["signature"]
        suggestion = suggestions.get(sig, {
            "harness_fix": "需要人工分析",
            "sensor_fix": "添加通用质量 sensor",
            "type": "manual",
        })

        return {
            "failure_pattern": sig,
            "occurrence": cluster["count"],
            "impact": cluster["count"],  # 简化：用出现次数作为影响力
            **suggestion,
        }
```

### 闭环修复原则

```
┌────────────────────────────────────────────────────────┐
│                    闭环修复原则                          │
│                                                        │
│  错误 → 不要修输出                                      │
│       → 溯源到缺失的约束                                │
│       → 修 Harness                                     │
│                                                        │
│  例子：                                                 │
│                                                        │
│  ✗ 错误做法：                                           │
│    "输出格式错了" → 人工修正输出 → 下次还会错             │
│                                                        │
│  ✓ 正确做法：                                           │
│    "输出格式错了"                                       │
│    → 为什么错？→ system prompt 没有严格的格式要求         │
│    → 修什么？ → 在 system prompt 中添加 JSON schema      │
│    → 加什么？ → 添加输出格式验证 guardrail               │
│    → 结果：   → 同类错误不会再出现                       │
└────────────────────────────────────────────────────────┘
```

---

## 23.4 回归测试数据集的构建和维护

### 为什么需要专门的 AI 回归测试集

```
传统回归测试:
  固定输入 → 固定期望输出 → assertEqual()

AI 回归测试:
  固定输入 → 非确定性输出 → 属性检查()
  
  不检查"答案是什么"，而检查"答案有什么属性"：
  - 是否包含关键信息？
  - 格式是否正确？
  - 是否引用了正确的来源？
  - 是否违反了安全规则？
```

### 回归测试集构建

```python
from dataclasses import field

@dataclass
class RegressionTestCase:
    """回归测试用例"""
    id: str
    category: str           # "format" | "quality" | "safety" | "factual"
    input_messages: list[dict]
    expected_properties: dict  # 期望的输出属性
    source: str             # "production_failure" | "adversarial" | "manual"
    created_date: str
    last_validated: str | None = None
    is_active: bool = True

class RegressionTestSuite:
    """回归测试套件"""

    def __init__(self):
        self.test_cases: list[RegressionTestCase] = []

    def add_from_production_failure(
        self,
        failure: dict,
        expected_fix: dict,
    ) -> str:
        """从生产失败中创建回归测试"""
        test_case = RegressionTestCase(
            id=f"reg_{len(self.test_cases):04d}",
            category=failure.get("category", "quality"),
            input_messages=failure["input_messages"],
            expected_properties=expected_fix,
            source="production_failure",
            created_date=datetime.now().isoformat(),
        )
        self.test_cases.append(test_case)
        return test_case.id

    def add_adversarial(
        self,
        attack_input: list[dict],
        expected_behavior: dict,
    ) -> str:
        """添加对抗性测试用例"""
        test_case = RegressionTestCase(
            id=f"adv_{len(self.test_cases):04d}",
            category="safety",
            input_messages=attack_input,
            expected_properties=expected_behavior,
            source="adversarial",
            created_date=datetime.now().isoformat(),
        )
        self.test_cases.append(test_case)
        return test_case.id

    def run(self, llm_client) -> dict:
        """运行全部回归测试"""
        results = {"passed": 0, "failed": 0, "errors": 0, "details": []}

        for tc in self.test_cases:
            if not tc.is_active:
                continue

            try:
                response = llm_client.generate(tc.input_messages)
                passed = self._check_properties(response, tc.expected_properties)

                if passed:
                    results["passed"] += 1
                else:
                    results["failed"] += 1

                results["details"].append({
                    "id": tc.id,
                    "category": tc.category,
                    "passed": passed,
                    "response_preview": response[:200],
                })

            except Exception as e:
                results["errors"] += 1
                results["details"].append({
                    "id": tc.id,
                    "error": str(e),
                })

        total = results["passed"] + results["failed"] + results["errors"]
        results["pass_rate"] = results["passed"] / max(total, 1)
        return results

    def _check_properties(self, response: str, expected: dict) -> bool:
        """检查输出属性"""
        for prop, value in expected.items():
            if prop == "contains" and not all(v in response for v in value):
                return False
            if prop == "not_contains" and any(v in response for v in value):
                return False
            if prop == "max_length" and len(response) > value:
                return False
            if prop == "min_length" and len(response) < value:
                return False
            if prop == "format" and value == "json":
                try:
                    import json
                    json.loads(response)
                except json.JSONDecodeError:
                    return False
        return True

    def maintain(self) -> dict:
        """维护测试集：清理过时用例"""
        stats = {"total": len(self.test_cases), "deactivated": 0, "stale": 0}

        for tc in self.test_cases:
            # 超过 90 天未验证的标记为 stale
            if tc.last_validated:
                validated_date = datetime.fromisoformat(tc.last_validated)
                if (datetime.now() - validated_date).days > 90:
                    stats["stale"] += 1

        return stats
```

---

## 23.5 Harness 改进循环：每个复发问题变成新的 Guide 或 Sensor

### 改进循环模型

```
          ┌─────────────┐
          │ 问题出现     │
          └──────┬──────┘
                 │
          ┌──────▼──────┐      第一次：
          │ 手动修复     │      可以接受
          └──────┬──────┘
                 │
          ┌──────▼──────┐      第二次：
          │ 同类问题     │      必须系统化修复
          │ 再次出现     │
          └──────┬──────┘
                 │
      ┌──────────┼──────────┐
      ▼          ▼          ▼
┌──────────┐┌──────────┐┌──────────┐
│ 新增     ││ 新增     ││ 更新     │
│ Guide    ││ Sensor   ││ 回归测试 │
│ (prompt  ││ (自动    ││          │
│ 中的约束)││  检测)   ││          │
└──────────┘└──────────┘└──────────┘
      │          │          │
      └──────────┼──────────┘
                 │
          ┌──────▼──────┐
          │ 同类问题     │
          │ 不再出现     │  ← 闭环
          └─────────────┘
```

### 改进循环实现

```python
class HarnessImprovementLoop:
    """Harness 持续改进循环"""

    def __init__(self, telemetry_analyzer, test_suite, harness_config):
        self.analyzer = telemetry_analyzer
        self.test_suite = test_suite
        self.config = harness_config
        self.improvement_log: list[dict] = []

    def run_cycle(self) -> dict:
        """运行一个改进周期"""
        # Step 1: 分析遥测
        analysis = self.analyzer.analyze_failures()

        # Step 2: 识别复发问题
        recurring = self._identify_recurring(analysis)

        # Step 3: 为每个复发问题生成改进
        improvements = []
        for problem in recurring:
            improvement = self._generate_improvement(problem)
            improvements.append(improvement)

        # Step 4: 应用改进
        applied = []
        for imp in improvements:
            result = self._apply_improvement(imp)
            applied.append(result)

        # Step 5: 验证改进
        verification = self.test_suite.run(None)  # 需要 LLM client

        cycle_result = {
            "analyzed_failures": analysis["total_failures"],
            "recurring_problems": len(recurring),
            "improvements_applied": len(applied),
            "verification_pass_rate": verification.get("pass_rate", 0),
        }

        self.improvement_log.append({
            "timestamp": datetime.now().isoformat(),
            **cycle_result,
        })

        return cycle_result

    def _identify_recurring(self, analysis: dict) -> list[dict]:
        """识别复发问题"""
        recurring = []
        for cluster in analysis.get("clusters", []):
            # 检查是否在之前的改进日志中出现过
            signature = cluster["signature"]
            past_occurrences = sum(
                1 for log in self.improvement_log
                if any(
                    imp.get("pattern") == signature
                    for imp in log.get("improvements", [])
                )
            )
            if cluster["count"] > 5 or past_occurrences > 0:
                recurring.append({
                    **cluster,
                    "recurrence_count": past_occurrences + 1,
                })
        return recurring

    def _generate_improvement(self, problem: dict) -> dict:
        """生成改进方案"""
        sig = problem["signature"]

        # 根据复发次数升级改进力度
        recurrence = problem.get("recurrence_count", 1)

        if recurrence == 1:
            # 第一次复发：添加 prompt guide
            return {
                "type": "guide",
                "pattern": sig,
                "action": f"在 system prompt 中添加针对 '{sig}' 的约束",
                "severity": "LOW",
            }
        elif recurrence == 2:
            # 第二次复发：添加 sensor
            return {
                "type": "sensor",
                "pattern": sig,
                "action": f"添加自动检测 '{sig}' 的 sensor",
                "severity": "MEDIUM",
            }
        else:
            # 三次以上：架构级改变
            return {
                "type": "architecture",
                "pattern": sig,
                "action": f"需要重新设计 '{sig}' 相关的 harness 组件",
                "severity": "HIGH",
            }

    def _apply_improvement(self, improvement: dict) -> dict:
        """应用改进"""
        # 1. 更新 harness 配置
        # 2. 添加回归测试
        test_id = self.test_suite.add_from_production_failure(
            failure={"category": improvement["pattern"], "input_messages": []},
            expected_fix={"not_contains": [improvement["pattern"]]},
        )

        return {
            "improvement": improvement,
            "regression_test_id": test_id,
            "applied": True,
        }

    def get_improvement_history(self) -> list[dict]:
        """获取改进历史"""
        return self.improvement_log
```

---

## 23.6 实战：设计一个 Harness 持续改进流程

### 完整流程图

```
                 ┌──────────────────┐
                 │  Weekly Cadence  │
                 │  (每周一次)       │
                 └────────┬─────────┘
                          │
              ┌───────────▼───────────┐
              │  Step 1: 收集遥测     │
              │  - 上周的 trace 数据  │
              │  - 质量分数趋势       │
              │  - 用户反馈           │
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │  Step 2: 聚类分析     │
              │  - 失败模式聚类       │
              │  - Top 5 问题排序     │
              │  - 对比上周           │
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │  Step 3: 根因分析     │
              │  - 每个 Top 问题      │
              │  → 追溯到缺失约束     │
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │  Step 4: 修复         │
              │  - 更新 prompt/guide  │
              │  - 添加 sensor        │
              │  - 添加回归测试       │
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │  Step 5: 验证         │
              │  - 运行回归测试套件   │
              │  - 对比修复前后       │
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │  Step 6: 部署         │
              │  - Canary 部署        │
              │  - 监控改进效果       │
              └───────────────────────┘
```

---

## 本章小结

| 概念 | 核心要点 |
|------|----------|
| 熵增问题 | 无维护的 agent 系统质量必然退化 |
| 六个根因 | 模型漂移、数据漂移、知识过时、上下文累积、Prompt 腐化、护栏滞后 |
| 漂移检测 | 固定探针 + 属性检查，定期运行 |
| 闭环原则 | 修 Harness 不修输出 |
| 回归测试 | 属性检查而非精确匹配 |
| 改进循环 | 复发 1 次 → Guide，2 次 → Sensor，3 次 → 架构重设计 |

---

## 动手实验

### 实验 1：设计一个 Harness 持续改进流程

**目标**：实现从遥测到改进的完整闭环。

```python
# 步骤：
# 1. 创建 TelemetryAnalyzer，输入 50 条模拟失败 trace
# 2. 聚类分析，识别 Top 3 失败模式
# 3. 为每个模式生成改进建议
# 4. 创建对应的回归测试
# 5. 模拟改进后重新运行测试，验证通过率提升
```

**验收标准**：
- 能正确聚类出至少 3 种失败模式
- 每种模式有对应的改进建议
- 回归测试覆盖所有已知失败模式

### 实验 2：模型漂移探针

**目标**：为一个 QA 系统设计和实现漂移探针。

**步骤**：
1. 设计 10 个探针问题（覆盖不同类型）
2. 采集基线（运行 5 次取均值）
3. 模拟模型漂移（修改 LLM 参数）
4. 运行漂移检测
5. 验证能正确检测到行为变化

### 实验 3：熵增仪表盘

**目标**：实现 EntropyMetrics 并可视化质量趋势。

```python
# 准备 30 个数据点的质量历史（模拟逐渐退化）
history = [
    {"date": f"2026-{m:02d}-01", "score": 0.95 - i * 0.008 + random.gauss(0, 0.02)}
    for i, m in enumerate(range(1, 13))
    for _ in range(3)  # 每月 3 个数据点
][:30]

metrics = EntropyMetrics(history)
report = metrics.report()
print(report)
```

---

## 练习题

### 基础题

1. **概念题**：列举 Agent 熵增的六个根因，并为每个根因给出一个具体的检测方法。

2. **计算题**：一个系统的质量分数每两周下降 0.02。当前分数 0.88，阈值 0.70。多少周后需要干预？

3. **设计题**：设计 5 个模型漂移探针问题，要求覆盖：事实问答、格式遵循、安全边界、推理能力、语言风格。

### 实践题

4. **闭环修复**：一个 RAG 系统的幻觉率从 5% 上升到 15%。设计完整的闭环修复流程：从检测 → 根因分析 → Harness 修复 → 验证。

5. **回归测试维护**：设计一个回归测试套件的维护策略，包括：新增测试的来源、过时测试的清理规则、测试集规模控制（不超过 200 个）。

### 思考题

6. **Harness 自身的熵**：Harness 是用来对抗 agent 熵增的，但 Harness 本身也会随时间腐化（规则过时、sensor 误报增多）。如何管理"Harness 的 Harness"？是否需要二阶改进循环？

7. **改进的边际收益**：每次改进循环都能减少一部分错误，但随着时间推移，容易修的问题都修完了，剩下的都是"长尾"难题。如何判断继续改进是否值得？什么时候应该接受"足够好"？
