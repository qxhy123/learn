# 第13章：CI/CD 集成与质量门禁

> 如果评估只在本地跑，它就不存在。真正的质量保障发生在 CI 管线中——每个 PR 都自动跑 prompt 评估，任何 metric regression 都阻止合并。本章将把前面构建的评估 harness 接入 CI/CD 管线，让质量门禁成为自动化的"不可绕过的关卡"。

---

## 学习目标

学完本章，你将能够：

1. 在 GitHub Actions 中配置 LLM 评估管线
2. 定义 metric regression 门禁并触发构建失败
3. 设计 LLM Readiness Harness 的多维度就绪分数
4. 使用 OpenTelemetry 为评估管线做插桩
5. 生成并发布详细的评估报告到 PR 评论

---

## 13.1 在每个 PR 运行 Prompt 评估

### 问题：质量保障的最后一公里

开发者改了 prompt，本地试了几个例子觉得"看起来不错"，提了 PR。但：

- 谁验证了新 prompt 在 200 个边界用例上的表现？
- 谁检查了改动是否让某个场景的准确率从 95% 掉到了 80%？
- 谁能保证这个改动不会让成本翻倍？

答案：**CI 管线自动跑评估**。

### 架构总览

```
Developer pushes PR
        │
        ▼
┌──────────────────────────────────────────┐
│           GitHub Actions Workflow          │
│                                          │
│  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │ Unit Test │  │ Eval Run │  │ Cost   │ │
│  │ (mock)    │  │ (real)   │  │ Check  │ │
│  │  ~30s     │  │  ~5min   │  │ ~10s   │ │
│  └─────┬────┘  └─────┬────┘  └───┬────┘ │
│        │             │            │      │
│        ▼             ▼            ▼      │
│  ┌──────────────────────────────────────┐│
│  │         Quality Gate (门禁)          ││
│  │  - All unit tests pass              ││
│  │  - No metric regression > 5%        ││
│  │  - Cost per call < $0.05            ││
│  │  - Latency p95 < 3s                ││
│  └──────────────────────────────────────┘│
│        │                                 │
│    PASS│FAIL                             │
│        ▼                                 │
│  ┌──────────────┐                        │
│  │ PR Comment    │ ← 评估报告            │
│  │ with Report   │                       │
│  └──────────────┘                        │
└──────────────────────────────────────────┘
```

### GitHub Actions Workflow

```yaml
# .github/workflows/llm-eval.yml
name: LLM Evaluation

on:
  pull_request:
    paths:
      - 'prompts/**'
      - 'src/agents/**'
      - 'src/tools/**'
      - 'eval/**'

env:
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  EVAL_BUDGET_USD: "2.00"

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements-test.txt
      - run: pytest tests/unit -v --tb=short

  llm-eval:
    runs-on: ubuntu-latest
    needs: unit-tests  # 先过 unit test 再花钱跑评估
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      
      - name: Install dependencies
        run: pip install -r requirements-eval.txt
      
      - name: Run evaluation suite
        id: eval
        run: |
          python eval/run_eval.py \
            --config eval/config.yaml \
            --output eval-results.json \
            --budget $EVAL_BUDGET_USD \
            --compare-baseline eval/baseline.json
      
      - name: Check quality gate
        id: gate
        run: |
          python eval/quality_gate.py \
            --results eval-results.json \
            --baseline eval/baseline.json \
            --max-regression 0.05 \
            --output gate-result.json
      
      - name: Post evaluation report
        if: always()
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const results = JSON.parse(fs.readFileSync('eval-results.json'));
            const gate = JSON.parse(fs.readFileSync('gate-result.json'));
            
            const status = gate.passed ? '✅ PASSED' : '❌ FAILED';
            let body = `## LLM Evaluation Report ${status}\n\n`;
            body += `| Metric | Baseline | Current | Delta | Status |\n`;
            body += `|--------|----------|---------|-------|--------|\n`;
            
            for (const [name, data] of Object.entries(results.metrics)) {
              const delta = (data.current - data.baseline).toFixed(4);
              const icon = data.regression ? '🔴' : '🟢';
              body += `| ${name} | ${data.baseline} | ${data.current} | ${delta} | ${icon} |\n`;
            }
            
            body += `\n**Cost**: $${results.total_cost.toFixed(4)}`;
            body += ` | **Latency p95**: ${results.latency_p95}ms`;
            body += ` | **Eval cases**: ${results.total_cases}`;
            
            await github.rest.issues.createComment({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: context.issue.number,
              body: body
            });
      
      - name: Fail if gate not passed
        if: steps.gate.outcome == 'failure'
        run: exit 1
      
      - name: Upload eval artifacts
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: eval-results
          path: |
            eval-results.json
            gate-result.json
```

---

## 13.2 Metric Regression 触发构建失败

### 什么算 Regression

不是"分数变了"就是 regression，而是"分数显著变差了"：

| 场景 | Baseline | Current | 判定 |
|------|----------|---------|------|
| 准确率 0.95 → 0.94 | 0.95 | 0.94 | 可能噪声，不 block |
| 准确率 0.95 → 0.88 | 0.95 | 0.88 | 明确 regression，block |
| 延迟 200ms → 350ms | 200ms | 350ms | 性能 regression，block |
| 成本 $0.02 → $0.03 | $0.02 | $0.03 | 成本增长 50%，warn |

### Quality Gate 实现

```python
import json
import sys
from dataclasses import dataclass
from pathlib import Path

@dataclass
class GateRule:
    metric: str
    max_regression: float       # 允许的最大下降比例（0.05 = 5%）
    absolute_minimum: float     # 绝对下限
    direction: str = "higher_is_better"  # 或 "lower_is_better"
    severity: str = "error"     # error = 阻止合并, warning = 只告警

class QualityGate:
    """CI 质量门禁"""
    
    def __init__(self, rules: list[GateRule]):
        self.rules = rules
    
    def check(self, baseline: dict, current: dict) -> dict:
        """检查当前评估结果是否通过门禁"""
        results = {}
        all_passed = True
        
        for rule in self.rules:
            b_val = baseline.get(rule.metric, 0)
            c_val = current.get(rule.metric, 0)
            
            if rule.direction == "higher_is_better":
                regression = (b_val - c_val) / b_val if b_val != 0 else 0
                below_minimum = c_val < rule.absolute_minimum
            else:  # lower_is_better (如延迟、成本)
                regression = (c_val - b_val) / b_val if b_val != 0 else 0
                below_minimum = c_val > rule.absolute_minimum
            
            passed = regression <= rule.max_regression and not below_minimum
            
            if not passed and rule.severity == "error":
                all_passed = False
            
            results[rule.metric] = {
                "baseline": b_val,
                "current": c_val,
                "regression": round(regression, 4),
                "max_allowed": rule.max_regression,
                "below_minimum": below_minimum,
                "passed": passed,
                "severity": rule.severity,
            }
        
        return {
            "passed": all_passed,
            "details": results,
            "summary": self._summary(results),
        }
    
    def _summary(self, results: dict) -> str:
        failures = [
            f"  - {k}: {v['regression']*100:.1f}% regression (max {v['max_allowed']*100:.1f}%)"
            for k, v in results.items() if not v["passed"]
        ]
        if not failures:
            return "All metrics within acceptable range"
        return "Quality gate FAILED:\n" + "\n".join(failures)


# 示例配置
DEFAULT_RULES = [
    GateRule("accuracy",       max_regression=0.05, absolute_minimum=0.85),
    GateRule("format_compliance", max_regression=0.03, absolute_minimum=0.90),
    GateRule("factual_score",  max_regression=0.05, absolute_minimum=0.90),
    GateRule("latency_p95_ms", max_regression=0.20, absolute_minimum=5000,
             direction="lower_is_better"),
    GateRule("cost_per_call",  max_regression=0.30, absolute_minimum=0.10,
             direction="lower_is_better", severity="warning"),
]
```

### Baseline 管理

```python
class BaselineManager:
    """Baseline 的版本管理"""
    
    def __init__(self, baseline_path: str = "eval/baseline.json"):
        self.path = Path(baseline_path)
    
    def load(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {}
    
    def update(self, new_results: dict, reason: str):
        """更新 baseline（应该是有意识的决策，不是自动的）"""
        current = self.load()
        history_entry = {
            "previous": current.get("metrics", {}),
            "reason": reason,
        }
        
        new_baseline = {
            "metrics": new_results,
            "updated_by": reason,
            "history": current.get("history", []) + [history_entry],
        }
        self.path.write_text(json.dumps(new_baseline, indent=2))
        return new_baseline
    
    def should_update(self, current: dict, improvement_threshold: float = 0.02) -> bool:
        """当前结果显著优于 baseline 时建议更新"""
        baseline = self.load().get("metrics", {})
        improvements = 0
        for key in baseline:
            if key in current:
                if current[key] > baseline[key] * (1 + improvement_threshold):
                    improvements += 1
        return improvements > len(baseline) / 2  # 超过半数指标改善
```

---

## 13.3 LLM Readiness Harness：多维度就绪分数

### 不是一个数字，而是一个向量

"模型准备好了吗？"不应该用 yes/no 回答，而是一个多维度的就绪向量：

```
Readiness Vector = [
    accuracy:    0.93  ✓  (threshold: 0.90)
    safety:      0.98  ✓  (threshold: 0.95)
    latency:     0.85  ✓  (threshold: 0.80)
    cost:        0.72  ✗  (threshold: 0.80)  ← 成本超标
    robustness:  0.91  ✓  (threshold: 0.85)
]
Ready: NO (cost dimension failed)
```

### Readiness Harness

```python
from typing import Callable

class ReadinessHarness:
    """多维度就绪评估 harness"""
    
    def __init__(self):
        self.dimensions: dict[str, dict] = {}
    
    def add_dimension(
        self,
        name: str,
        evaluator: Callable[[], float],
        threshold: float,
        weight: float = 1.0,
        blocking: bool = True,
    ):
        """添加一个就绪维度"""
        self.dimensions[name] = {
            "evaluator": evaluator,
            "threshold": threshold,
            "weight": weight,
            "blocking": blocking,
        }
    
    def evaluate(self) -> dict:
        """运行完整的就绪评估"""
        results = {}
        weighted_sum = 0.0
        weight_total = 0.0
        all_blocking_passed = True
        
        for name, dim in self.dimensions.items():
            score = dim["evaluator"]()
            passed = score >= dim["threshold"]
            
            if dim["blocking"] and not passed:
                all_blocking_passed = False
            
            weighted_sum += score * dim["weight"]
            weight_total += dim["weight"]
            
            results[name] = {
                "score": round(score, 4),
                "threshold": dim["threshold"],
                "passed": passed,
                "blocking": dim["blocking"],
                "weight": dim["weight"],
            }
        
        composite = round(weighted_sum / weight_total, 4) if weight_total else 0
        
        return {
            "dimensions": results,
            "composite_score": composite,
            "ready": all_blocking_passed,
            "summary": self._format_summary(results, all_blocking_passed),
        }
    
    def _format_summary(self, results, ready):
        lines = []
        for name, r in results.items():
            status = "PASS" if r["passed"] else "FAIL"
            block = " [BLOCKING]" if r["blocking"] and not r["passed"] else ""
            lines.append(
                f"  {name}: {r['score']:.2f} / {r['threshold']:.2f} → {status}{block}"
            )
        header = "READY" if ready else "NOT READY"
        return f"[{header}]\n" + "\n".join(lines)
```

### 成本-效用前沿分析

```python
class CostUtilityFrontier:
    """成本-效用前沿：找到最优的模型配置"""
    
    def __init__(self):
        self.configurations: list[dict] = []
    
    def add_config(self, name: str, cost: float, utility: float, **metadata):
        self.configurations.append({
            "name": name,
            "cost": cost,
            "utility": utility,
            **metadata,
        })
    
    def find_pareto_frontier(self) -> list[dict]:
        """找到 Pareto 前沿上的配置"""
        sorted_configs = sorted(self.configurations, key=lambda x: x["cost"])
        frontier = []
        max_utility = -1
        
        for config in sorted_configs:
            if config["utility"] > max_utility:
                frontier.append(config)
                max_utility = config["utility"]
        
        return frontier
    
    def recommend(self, max_cost: float) -> dict | None:
        """在成本约束下推荐最优配置"""
        frontier = self.find_pareto_frontier()
        candidates = [c for c in frontier if c["cost"] <= max_cost]
        if not candidates:
            return None
        return max(candidates, key=lambda x: x["utility"])
    
    def visualize_ascii(self) -> str:
        """ASCII 可视化成本-效用前沿"""
        frontier = self.find_pareto_frontier()
        all_configs = self.configurations
        
        max_cost = max(c["cost"] for c in all_configs)
        max_util = max(c["utility"] for c in all_configs)
        
        width, height = 50, 20
        grid = [[" " for _ in range(width)] for _ in range(height)]
        
        frontier_names = {c["name"] for c in frontier}
        
        for config in all_configs:
            x = int((config["cost"] / max_cost) * (width - 1))
            y = height - 1 - int((config["utility"] / max_util) * (height - 1))
            x = min(x, width - 1)
            y = max(0, min(y, height - 1))
            
            if config["name"] in frontier_names:
                grid[y][x] = "★"
            else:
                grid[y][x] = "·"
        
        lines = ["Utility ↑"]
        for row in grid:
            lines.append("│" + "".join(row))
        lines.append("└" + "─" * width + "→ Cost")
        lines.append("")
        lines.append("★ = Pareto frontier  · = dominated")
        
        return "\n".join(lines)
```

---

## 13.4 OpenTelemetry 为评估管线做插桩

### 为什么评估管线需要可观测性

评估管线本身也是软件——它也会出 bug、变慢、产生错误结果：

```
问题："昨天的评估跑了 25 分钟，今天跑了 50 分钟，为什么？"
答案：没有 tracing，你只能猜。
```

### 插桩设计

```python
from contextlib import contextmanager
import time

# 简化的 tracing（概念演示，实际使用 opentelemetry SDK）
class EvalTracer:
    """评估管线的追踪器"""
    
    def __init__(self, service_name: str = "eval-pipeline"):
        self.service_name = service_name
        self.spans: list[dict] = []
        self._active_span_stack: list[dict] = []
    
    @contextmanager
    def span(self, name: str, attributes: dict | None = None):
        """创建一个追踪 span"""
        span_data = {
            "name": name,
            "service": self.service_name,
            "start_time": time.time(),
            "attributes": attributes or {},
            "children": [],
            "status": "ok",
        }
        
        # 如果有父 span，添加为子 span
        if self._active_span_stack:
            self._active_span_stack[-1]["children"].append(span_data)
        else:
            self.spans.append(span_data)
        
        self._active_span_stack.append(span_data)
        try:
            yield span_data
        except Exception as e:
            span_data["status"] = "error"
            span_data["error"] = str(e)
            raise
        finally:
            span_data["end_time"] = time.time()
            span_data["duration_ms"] = round(
                (span_data["end_time"] - span_data["start_time"]) * 1000, 2
            )
            self._active_span_stack.pop()
    
    def record_metric(self, name: str, value: float, unit: str = ""):
        """记录一个指标"""
        if self._active_span_stack:
            span = self._active_span_stack[-1]
            span["attributes"][f"metric.{name}"] = value
            span["attributes"][f"metric.{name}.unit"] = unit


# 使用示例
tracer = EvalTracer(service_name="llm-eval-ci")

def run_eval_with_tracing(eval_harness, model_fn, cases):
    with tracer.span("eval_run", {"total_cases": len(cases)}):
        results = []
        
        for i, case in enumerate(cases):
            with tracer.span(f"eval_case_{i}", {
                "category": case.category,
                "difficulty": case.difficulty,
            }) as span:
                # LLM 调用
                with tracer.span("llm_call") as llm_span:
                    output = model_fn(case.input_text)
                    tracer.record_metric("output_tokens", len(output.split()))
                
                # 评分
                with tracer.span("scoring"):
                    scores = eval_harness._score_single(output, case)
                    for name, score in scores.items():
                        tracer.record_metric(name, score)
                
                results.append({"output": output, "scores": scores})
        
        # 聚合
        with tracer.span("aggregation"):
            final = eval_harness._aggregate(results)
    
    return final
```

### Trace 输出示例

```
eval_run (12453ms) [total_cases=50]
├── eval_case_0 (234ms) [category=legal, difficulty=normal]
│   ├── llm_call (198ms) [metric.output_tokens=45]
│   └── scoring (12ms) [metric.accuracy=0.92, metric.format=1.0]
├── eval_case_1 (312ms) [category=medical, difficulty=hard]
│   ├── llm_call (287ms) [metric.output_tokens=89]
│   └── scoring (8ms) [metric.accuracy=0.85, metric.format=1.0]
├── ...
└── aggregation (5ms)
```

---

## 13.5 从 CI 发布详细评估报告

### 报告生成器

```python
class EvalReportGenerator:
    """生成 CI 评估报告"""
    
    def __init__(self, results: dict, baseline: dict, gate_result: dict):
        self.results = results
        self.baseline = baseline
        self.gate = gate_result
    
    def to_markdown(self) -> str:
        """生成 Markdown 格式的报告"""
        status = "PASSED" if self.gate["passed"] else "FAILED"
        
        lines = [
            f"## LLM Evaluation Report: {status}",
            "",
            "### Metrics Summary",
            "",
            "| Metric | Baseline | Current | Delta | Threshold | Status |",
            "|--------|----------|---------|-------|-----------|--------|",
        ]
        
        for name, detail in self.gate.get("details", {}).items():
            b = detail["baseline"]
            c = detail["current"]
            delta = c - b
            delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
            status_icon = "PASS" if detail["passed"] else "FAIL"
            lines.append(
                f"| {name} | {b:.4f} | {c:.4f} | {delta_str} | "
                f"{detail.get('max_allowed', 'N/A')} | {status_icon} |"
            )
        
        # 成本和延迟摘要
        lines.extend([
            "",
            "### Resource Usage",
            "",
            f"- **Total cost**: ${self.results.get('total_cost', 0):.4f}",
            f"- **Latency p50**: {self.results.get('latency_p50', 'N/A')}ms",
            f"- **Latency p95**: {self.results.get('latency_p95', 'N/A')}ms",
            f"- **Total eval cases**: {self.results.get('total_cases', 0)}",
            f"- **Total LLM calls**: {self.results.get('total_llm_calls', 0)}",
        ])
        
        # 失败用例详情
        failures = self.results.get("failures", [])
        if failures:
            lines.extend([
                "",
                "### Failed Cases (top 5)",
                "",
            ])
            for f in failures[:5]:
                lines.extend([
                    f"<details><summary>{f.get('category', 'unknown')}: "
                    f"{f.get('input', '')[:80]}...</summary>",
                    "",
                    f"**Expected**: {f.get('expected', 'N/A')[:200]}",
                    "",
                    f"**Got**: {f.get('actual', 'N/A')[:200]}",
                    "",
                    f"**Scores**: {f.get('scores', {})}",
                    "",
                    "</details>",
                    "",
                ])
        
        return "\n".join(lines)
    
    def to_json(self) -> str:
        """生成 JSON 格式的报告（用于机器消费）"""
        return json.dumps({
            "status": "passed" if self.gate["passed"] else "failed",
            "metrics": self.gate.get("details", {}),
            "resource_usage": {
                "cost_usd": self.results.get("total_cost", 0),
                "latency_p50_ms": self.results.get("latency_p50"),
                "latency_p95_ms": self.results.get("latency_p95"),
            },
            "failure_count": len(self.results.get("failures", [])),
        }, indent=2)
```

---

## 13.6 完整的评估入口脚本

```python
#!/usr/bin/env python3
"""eval/run_eval.py — CI 评估管线入口"""

import argparse
import json
import sys
import time
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run LLM evaluation")
    parser.add_argument("--config", required=True, help="Eval config YAML")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--budget", type=float, default=2.0, help="Max budget USD")
    parser.add_argument("--compare-baseline", help="Baseline JSON to compare against")
    args = parser.parse_args()
    
    # 1. 加载配置和评估用例
    config = load_config(args.config)
    cases = load_eval_cases(config["cases_path"])
    
    # 2. 初始化模型客户端（带成本控制）
    client = create_eval_client(
        model=config["model"],
        budget_usd=args.budget,
    )
    
    # 3. 运行评估
    start = time.time()
    results = run_evaluation(client, cases, config["metrics"])
    elapsed = time.time() - start
    
    results["wall_time_seconds"] = round(elapsed, 2)
    
    # 4. 保存结果
    Path(args.output).write_text(json.dumps(results, indent=2))
    
    # 5. 如果有 baseline，进行对比
    if args.compare_baseline:
        baseline = json.loads(Path(args.compare_baseline).read_text())
        gate = QualityGate(DEFAULT_RULES)
        gate_result = gate.check(
            baseline=baseline.get("metrics", {}),
            current=results.get("metrics", {}),
        )
        
        gate_path = args.output.replace(".json", "-gate.json")
        Path(gate_path).write_text(json.dumps(gate_result, indent=2))
        
        if not gate_result["passed"]:
            print(f"QUALITY GATE FAILED:\n{gate_result['summary']}", file=sys.stderr)
            sys.exit(1)
    
    print(f"Evaluation complete: {len(cases)} cases in {elapsed:.1f}s")
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
```

---

## 本章小结

| 主题 | 关键洞察 |
|------|---------|
| CI 中的 LLM 评估 | 每个 PR 自动跑评估；质量保障不能依赖人工 |
| Metric regression | 不是"变了"就 fail，而是"显著变差"才 fail |
| Quality gate | 多维度门禁，区分 blocking 和 warning |
| Readiness harness | 就绪状态是向量不是标量；Pareto 前沿找最优配置 |
| OpenTelemetry | 评估管线本身也需要可观测性 |
| 报告发布 | 评估报告直接评论到 PR，让 reviewer 有数据基础 |

---

## 动手实验

### 实验 1：写一个 GitHub Actions Workflow

**目标**：创建一个完整的 `.github/workflows/llm-eval.yml`

要求：
1. 只在 `prompts/` 或 `src/agents/` 下的文件改动时触发
2. 先跑 mock unit test，通过后再跑真实评估
3. 设置 $2.00 的预算上限
4. 评估结果发布为 PR 评论
5. metric regression 超过 5% 触发构建失败

### 实验 2：实现 Quality Gate 配置化

**目标**：扩展 `QualityGate`，支持从 YAML 配置文件加载规则

```yaml
# eval/gate-config.yaml
rules:
  - metric: accuracy
    max_regression: 0.05
    absolute_minimum: 0.85
    direction: higher_is_better
    severity: error
  - metric: cost_per_call
    max_regression: 0.30
    absolute_minimum: 0.10
    direction: lower_is_better
    severity: warning
```

### 实验 3：构建评估结果的趋势看板

**目标**：收集多次 CI 评估结果，生成趋势报告

```python
# 读取 eval_history/ 目录下的历史结果
# 生成每个 metric 的趋势图（ASCII 或 matplotlib）
# 检测是否有持续下降趋势
# 输出告警
```

---

## 练习题

### 基础题

1. 解释为什么 LLM 评估应该在 unit test 之后运行，而不是并行运行。从成本和效率角度论证。

2. 一个 metric 从 0.92 变成 0.89，regression 为 3.3%。在 max_regression=5% 的配置下，这个 PR 应该通过还是失败？如果连续 3 个 PR 都有类似的 3% 下降呢？

3. 列出至少 5 种应该纳入 LLM Readiness Harness 的维度，并为每个维度定义合理的阈值。

### 实践题

4. 实现一个 `CIBudgetManager`，它能：(a) 在 CI 开始时分配预算，(b) 每次 LLM 调用后扣减，(c) 预算耗尽时优雅降级（跑更少的用例而非直接失败）。

5. 你的评估管线在 CI 中偶尔因为 API rate limit 而失败。设计一个重试策略，包括退避算法和部分结果恢复。

### 思考题

6. "在 CI 中跑 LLM 评估意味着每个 PR 都要花钱。对于一个活跃的开源项目（每天 50 个 PR），如何平衡质量保障和成本？"

7. 评估管线本身的正确性如何保证？如果评估管线有 bug（比如评分函数写错了），它会产生什么后果？如何防范？
