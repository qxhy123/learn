# 第6章：评估 Harness 基础

> "你无法改进你无法测量的东西。" 这不是管理学鸡汤，而是工程铁律。当你部署一个 AI 系统后，用户告诉你"回答不太对"——你怎么定位问题？是 prompt 写得差、模型能力不足、还是检索召回率低？没有评估 harness，你就是在黑暗中调参。本章建立评估的第一性原理：什么是评估 harness，如何设计指标体系，以及如何从零搭建一个最小可用的评估框架。

---

## 学习目标

学完本章，你将能够：

1. 定义什么是评估 harness，区分它与"写几个 assert"的本质差别
2. 区分模型评估（model evaluation）和系统评估（system evaluation）的适用场景
3. 设计一套核心指标体系，涵盖准确性、忠实性、相关性、连贯性和公平性
4. 运用确定性基线和统计置信区间来保证评估结果可复现
5. 独立实现一个最小评估 harness 框架，完成从数据加载到报告生成的完整流程

---

## 6.1 什么是评估 Harness

### 6.1.1 从手动测试到系统化评估

大多数团队的 AI 评估起步于这样的场景：

```python
# 这不是评估 harness，这是 ad-hoc 测试
response = call_llm("法国的首都是哪里？")
assert "巴黎" in response  # 脆弱、不可扩展、没有指标
```

这种方式有三个根本缺陷：

| 缺陷 | 后果 |
|------|------|
| 不可复现 | 换个人跑结果不同（随机种子、API 版本、环境差异） |
| 不可比较 | 无法回答"新版本比旧版本好了多少" |
| 不可扩展 | 10 个测试用例能手写，1000 个不行 |

### 6.1.2 评估 Harness 的严格定义

**评估 Harness** 是一套受控、可复现的测量基础设施，它将以下要素标准化：

```
评估 Harness = 黄金数据集 + 指标定义 + 执行引擎 + 报告系统
```

关键属性：

- **受控（Controlled）**：相同输入 + 相同配置 = 相同输出（或统计等价的输出）
- **可复现（Reproducible）**：任何人在任何时间跑，结果一致
- **自动化（Automated）**：不需要人工逐条检查（但可以有人工审查环节）
- **版本化（Versioned）**：数据集、指标、配置都有版本号

### 6.1.3 评估 Harness 的核心组件

```
┌─────────────────────────────────────────────┐
│              评估 Harness 架构               │
├──────────┬──────────┬──────────┬────────────┤
│ 黄金数据集 │ 指标引擎  │ 执行引擎  │ 报告系统    │
│ (Dataset) │ (Metrics)│ (Runner) │ (Reporter) │
├──────────┼──────────┼──────────┼────────────┤
│ 输入样本  │ 准确率   │ 批量调用  │ 分数汇总   │
│ 期望输出  │ 忠实性   │ 并发控制  │ 对比视图   │
│ 元数据    │ 相关性   │ 重试逻辑  │ 回归检测   │
│ 版本标签  │ 自定义   │ 缓存     │ 可视化     │
└──────────┴──────────┴──────────┴────────────┘
```

---

## 6.2 模型评估 vs 系统评估

这是评估工程中最重要的二分法。混淆两者是绝大多数团队犯的第一个错误。

### 6.2.1 模型评估（Model Evaluation）

**目标**：测量模型本身的能力边界。

```yaml
# 典型模型评估配置
evaluation_type: model
target: gpt-4o-mini
benchmarks:
  - name: MMLU
    description: 多领域知识问答
    metric: accuracy
  - name: HumanEval
    description: 代码生成
    metric: pass@k
  - name: GSM8K
    description: 数学推理
    metric: exact_match
settings:
  temperature: 0
  max_tokens: 1024
  num_samples: 1
```

**特点**：

| 维度 | 模型评估 |
|------|---------|
| 输入 | 标准化 prompt（通常是学术 benchmark） |
| 变量 | 只有模型本身 |
| 指标 | accuracy、perplexity、pass@k |
| 目的 | 模型选型、能力基准线 |
| 频率 | 模型切换时 |

### 6.2.2 系统评估（System Evaluation）

**目标**：测量整个产品管线的端到端质量。

```yaml
# 典型系统评估配置
evaluation_type: system
target: customer-support-bot-v2.3
pipeline:
  - retriever: hybrid_search
  - reranker: cross_encoder
  - generator: gpt-4o
  - guardrails: content_filter_v3
test_scenarios:
  - name: 退款查询
    input: "我三天前买的耳机坏了，怎么退款？"
    expected_behavior:
      - 引用退款政策
      - 提供具体步骤
      - 不编造不存在的政策
metrics:
  - faithfulness  # 是否忠于检索到的文档
  - latency_p95   # 95 分位延迟
  - cost_per_query # 单次查询成本
```

**特点**：

| 维度 | 系统评估 |
|------|---------|
| 输入 | 真实用户场景 |
| 变量 | prompt + 检索 + 模型 + 后处理 + 护栏 |
| 指标 | faithfulness、latency、cost、用户满意度 |
| 目的 | 产品质量门禁、回归检测 |
| 频率 | 每次部署前 |

### 6.2.3 两者的关系

```
模型评估 → "这个模型能做什么"
系统评估 → "这个产品做得怎么样"

模型评估好 ≠ 系统评估好
（好模型 + 差 prompt + 差检索 = 差产品）

模型评估差 ≠ 系统评估差
（弱模型 + 好 prompt + 好检索 + 好后处理 = 还行的产品）
```

---

## 6.3 核心指标体系

### 6.3.1 指标分类框架

```
                    AI 评估指标体系
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
       正确性          质量性          安全性
    (Correctness)   (Quality)      (Safety)
          │              │              │
    ┌─────┼─────┐  ┌─────┼─────┐  ┌────┼────┐
    ▼     ▼     ▼  ▼     ▼     ▼  ▼    ▼    ▼
  准确率 精确  F1 忠实性 相关性 连贯性 偏见 毒性 泄露
```

### 6.3.2 核心指标详解

| 指标 | 英文 | 定义 | 适用场景 | 计算方式 |
|------|------|------|---------|---------|
| 准确率 | Accuracy | 正确回答占比 | 分类、问答 | correct / total |
| 忠实性 | Faithfulness | 回答是否忠于给定上下文 | RAG | LLM-as-Judge 或 NLI |
| 相关性 | Relevance | 回答是否切题 | 所有生成任务 | LLM-as-Judge |
| 连贯性 | Coherence | 回答是否逻辑通顺 | 长文本生成 | LLM-as-Judge |
| 偏见 | Bias | 是否存在系统性偏差 | 所有任务 | 对比测试 |
| 毒性 | Toxicity | 是否包含有害内容 | 面向用户的系统 | 分类器 |

### 6.3.3 指标实现示例

```python
from dataclasses import dataclass
from typing import Callable

@dataclass
class Metric:
    """评估指标的统一抽象"""
    name: str
    compute_fn: Callable
    higher_is_better: bool = True
    description: str = ""

def exact_match(prediction: str, reference: str) -> float:
    """精确匹配：最简单但最严格的指标"""
    return 1.0 if prediction.strip() == reference.strip() else 0.0

def contains_match(prediction: str, reference: str) -> float:
    """包含匹配：宽松版本"""
    return 1.0 if reference.strip().lower() in prediction.strip().lower() else 0.0

def f1_token_level(prediction: str, reference: str) -> float:
    """Token 级别 F1：适合开放式回答"""
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())

    if not ref_tokens:
        return 1.0 if not pred_tokens else 0.0

    common = pred_tokens & ref_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)

# 注册指标
METRICS = {
    "exact_match": Metric("exact_match", exact_match),
    "contains": Metric("contains", contains_match),
    "f1": Metric("f1", f1_token_level),
}
```

---

## 6.4 确定性基线与统计置信

### 6.4.1 确定性基线

评估的第一条铁律：**消除不必要的随机性**。

```python
# 确定性配置
DETERMINISTIC_CONFIG = {
    "temperature": 0,       # 消除采样随机性
    "top_p": 1.0,           # 不截断分布
    "seed": 42,             # 固定随机种子（如果 API 支持）
    "max_tokens": 1024,     # 固定最大长度
}
```

即使设置了 `temperature=0`，仍然可能有不确定性来源：

| 来源 | 原因 | 缓解措施 |
|------|------|---------|
| 浮点运算 | GPU 并行计算的非结合性 | 多次运行取均值 |
| API 版本 | 提供商静默更新模型 | 锁定模型版本（如 `gpt-4o-2024-08-06`） |
| 批处理顺序 | 某些框架对顺序敏感 | 固定数据顺序 |
| 上下文长度 | 截断策略影响结果 | 记录实际 token 数 |

### 6.4.2 Bootstrap 置信区间

当你报告"准确率 85%"时，这个数字有多可靠？Bootstrap 置信区间给出答案。

```python
import numpy as np

def bootstrap_ci(
    scores: list[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42
) -> tuple[float, float, float]:
    """
    计算 Bootstrap 置信区间

    Returns:
        (mean, lower_bound, upper_bound)
    """
    rng = np.random.RandomState(seed)
    scores_array = np.array(scores)
    n = len(scores_array)

    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(scores_array, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    bootstrap_means = sorted(bootstrap_means)
    alpha = 1 - confidence
    lower = bootstrap_means[int(alpha / 2 * n_bootstrap)]
    upper = bootstrap_means[int((1 - alpha / 2) * n_bootstrap)]

    return float(np.mean(scores_array)), float(lower), float(upper)

# 示例
scores = [1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]
mean, lower, upper = bootstrap_ci(scores)
print(f"准确率: {mean:.1%}, 95% CI: [{lower:.1%}, {upper:.1%}]")
# 输出: 准确率: 75.0%, 95% CI: [55.0%, 90.0%]
```

**关键洞察**：20 个样本的 75% 准确率，置信区间宽达 35 个百分点。这意味着你至少需要几百个样本才能得到有意义的评估结论。

### 6.4.3 样本量与置信度的关系

| 样本量 | 95% CI 宽度（准确率约75%时） | 是否可用于决策 |
|--------|----------------------------|--------------|
| 20 | ~35% | 否，噪声太大 |
| 100 | ~17% | 勉强，用于快速验证 |
| 500 | ~7% | 是，适合大多数场景 |
| 2000 | ~4% | 是，适合严格基准 |

---

## 6.5 评估流程：从数据到报告

### 6.5.1 标准评估流程

```
黄金数据集 → 指标定义 → 自动评估 → 结果分析 → 人工审查
    │           │          │          │          │
    ▼           ▼          ▼          ▼          ▼
  版本化     可配置     批量执行   统计汇总    抽样验证
  不可变     可组合     可并发     可视化      校准反馈
```

### 6.5.2 黄金数据集的要求

```python
@dataclass
class EvalSample:
    """评估样本的标准结构"""
    id: str                      # 唯一标识
    input: str                   # 输入
    expected_output: str         # 期望输出（可选）
    metadata: dict               # 元数据（类别、难度、来源等）

@dataclass
class EvalDataset:
    """评估数据集"""
    name: str
    version: str                 # 语义化版本号
    samples: list[EvalSample]
    created_at: str
    description: str

    def freeze_hash(self) -> str:
        """数据集指纹：检测意外修改"""
        import hashlib, json
        content = json.dumps(
            [s.__dict__ for s in self.samples],
            sort_keys=True, ensure_ascii=False
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]
```

### 6.5.3 完整评估流程代码

```python
import json
import time
from pathlib import Path

class MinimalEvalHarness:
    """最小评估 harness 框架"""

    def __init__(self, model_fn, metrics: list[str], output_dir: str = "./eval_results"):
        self.model_fn = model_fn            # Callable: str -> str
        self.metrics = {m: METRICS[m] for m in metrics}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_dataset(self, path: str) -> EvalDataset:
        """加载黄金数据集"""
        with open(path) as f:
            data = json.load(f)
        samples = [EvalSample(**s) for s in data["samples"]]
        return EvalDataset(
            name=data["name"],
            version=data["version"],
            samples=samples,
            created_at=data["created_at"],
            description=data["description"],
        )

    def run(self, dataset: EvalDataset) -> dict:
        """执行评估"""
        results = []
        start_time = time.time()

        for sample in dataset.samples:
            # 1. 调用模型
            prediction = self.model_fn(sample.input)

            # 2. 计算每个指标
            scores = {}
            for name, metric in self.metrics.items():
                score = metric.compute_fn(prediction, sample.expected_output)
                scores[name] = score

            results.append({
                "id": sample.id,
                "input": sample.input,
                "expected": sample.expected_output,
                "prediction": prediction,
                "scores": scores,
            })

        elapsed = time.time() - start_time

        # 3. 汇总统计
        summary = self._compute_summary(results, elapsed, dataset)
        return summary

    def _compute_summary(self, results: list[dict], elapsed: float, dataset) -> dict:
        """计算汇总统计"""
        metric_scores = {name: [] for name in self.metrics}
        for r in results:
            for name, score in r["scores"].items():
                metric_scores[name].append(score)

        summary = {
            "dataset": dataset.name,
            "dataset_version": dataset.version,
            "dataset_hash": dataset.freeze_hash(),
            "num_samples": len(results),
            "elapsed_seconds": round(elapsed, 2),
            "metrics": {},
            "details": results,
        }

        for name, scores in metric_scores.items():
            mean, ci_lower, ci_upper = bootstrap_ci(scores)
            summary["metrics"][name] = {
                "mean": round(mean, 4),
                "ci_95_lower": round(ci_lower, 4),
                "ci_95_upper": round(ci_upper, 4),
            }

        return summary

    def save_report(self, summary: dict, run_id: str) -> Path:
        """保存评估报告"""
        report_path = self.output_dir / f"eval_{run_id}.json"
        with open(report_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return report_path

    def print_report(self, summary: dict):
        """打印简洁报告"""
        print(f"\n{'='*60}")
        print(f"评估报告: {summary['dataset']} v{summary['dataset_version']}")
        print(f"样本数: {summary['num_samples']}, 耗时: {summary['elapsed_seconds']}s")
        print(f"{'='*60}")
        for name, stats in summary["metrics"].items():
            print(f"  {name:>15}: {stats['mean']:.2%} "
                  f"[{stats['ci_95_lower']:.2%}, {stats['ci_95_upper']:.2%}]")
        print(f"{'='*60}\n")
```

---

## 6.6 工程原则："不能评估就不能改进"

### 6.6.1 评估驱动开发（Evaluation-Driven Development）

类比 TDD（测试驱动开发），评估驱动开发要求：

```
1. 先定义评估标准（写评估 harness）
2. 再构建 AI 系统
3. 运行评估，看到失败
4. 改进系统
5. 再运行评估，验证改进
6. 回到第 4 步
```

### 6.6.2 评估的反模式

| 反模式 | 后果 | 正确做法 |
|--------|------|---------|
| 在训练集上评估 | 虚假的高分 | 严格分离训练/评估数据 |
| 只看平均分 | 忽略分布和边缘情况 | 看分位数和分类别分数 |
| 评估集太小 | 置信区间过宽 | 至少 200-500 样本 |
| 手动挑选评估样本 | 选择偏差 | 随机抽样 + 分层抽样 |
| 不记录评估配置 | 不可复现 | 完整记录所有参数 |
| 评估跑一次就行 | 忽略随机性 | 多次运行取均值 |

### 6.6.3 评估基础设施的成熟度模型

| 等级 | 特征 | 典型团队 |
|------|------|---------|
| L0 | 无评估，靠直觉 | 原型阶段 |
| L1 | 手动测试几个例子 | 早期创业公司 |
| L2 | 有评估脚本，手动触发 | 成长期团队 |
| L3 | 自动化评估，CI 集成 | 成熟团队 |
| L4 | 持续评估，自动回归检测，A/B 测试 | 平台级团队 |

---

## 本章小结

| 概念 | 要点 |
|------|------|
| 评估 Harness | 受控、可复现的测量基础设施，包含数据集、指标、执行引擎、报告系统 |
| 模型评估 vs 系统评估 | 模型评估测能力边界，系统评估测产品质量；两者独立且都需要 |
| 核心指标 | 正确性（accuracy/F1）、质量性（faithfulness/relevance/coherence）、安全性（bias/toxicity） |
| 确定性基线 | temperature=0 + 固定种子 + 锁定模型版本 = 可复现 |
| Bootstrap 置信区间 | 量化评估结果的可靠性，样本量 < 200 时谨慎下结论 |
| 评估流程 | 黄金数据集 → 指标定义 → 自动评估 → 结果分析 → 人工审查 |
| 评估驱动开发 | 先定义评估标准再构建系统，类比 TDD |

---

## 动手实验

### 实验 1：构建最小评估 Harness

**目标**：用上文的 `MinimalEvalHarness` 类评估一个模拟模型。

**步骤**：

1. 创建黄金数据集文件 `eval_dataset.json`：

```json
{
  "name": "capital_cities",
  "version": "1.0.0",
  "created_at": "2026-04-15",
  "description": "世界主要城市首都问答",
  "samples": [
    {"id": "001", "input": "法国的首都是哪里？", "expected_output": "巴黎", "metadata": {"difficulty": "easy"}},
    {"id": "002", "input": "日本的首都是哪里？", "expected_output": "东京", "metadata": {"difficulty": "easy"}},
    {"id": "003", "input": "巴西的首都是哪里？", "expected_output": "巴西利亚", "metadata": {"difficulty": "medium"}},
    {"id": "004", "input": "澳大利亚的首都是哪里？", "expected_output": "堪培拉", "metadata": {"difficulty": "medium"}},
    {"id": "005", "input": "缅甸的首都是哪里？", "expected_output": "内比都", "metadata": {"difficulty": "hard"}}
  ]
}
```

2. 实现一个模拟模型和评估运行：

```python
def mock_model(question: str) -> str:
    """模拟模型：故意让部分回答错误以展示评估效果"""
    answers = {
        "法国的首都是哪里？": "法国的首都是巴黎。",
        "日本的首都是哪里？": "东京",
        "巴西的首都是哪里？": "巴西的首都是里约热内卢。",  # 故意错误
        "澳大利亚的首都是哪里？": "堪培拉是澳大利亚的首都。",
        "缅甸的首都是哪里？": "仰光",  # 故意错误
    }
    return answers.get(question, "我不知道")

harness = MinimalEvalHarness(
    model_fn=mock_model,
    metrics=["exact_match", "contains", "f1"]
)
dataset = harness.load_dataset("eval_dataset.json")
summary = harness.run(dataset)
harness.print_report(summary)
harness.save_report(summary, run_id="run_001")
```

3. 观察三个指标的差异——exact_match 最严格，contains 最宽松，f1 居中。

### 实验 2：评估结果对比与回归检测

**目标**：对比两个模型版本的评估结果，实现简单的回归检测。

```python
def compare_runs(run_a: dict, run_b: dict, threshold: float = 0.05) -> dict:
    """
    对比两次评估结果，检测回归

    Args:
        threshold: 可接受的下降幅度（如 5%）
    """
    comparison = {"regressions": [], "improvements": [], "stable": []}

    for metric_name in run_a["metrics"]:
        score_a = run_a["metrics"][metric_name]["mean"]
        score_b = run_b["metrics"][metric_name]["mean"]
        delta = score_b - score_a

        result = {
            "metric": metric_name,
            "before": score_a,
            "after": score_b,
            "delta": round(delta, 4),
        }

        if delta < -threshold:
            comparison["regressions"].append(result)
        elif delta > threshold:
            comparison["improvements"].append(result)
        else:
            comparison["stable"].append(result)

    comparison["has_regression"] = len(comparison["regressions"]) > 0
    return comparison

# 使用方式
# result = compare_runs(summary_v1, summary_v2)
# if result["has_regression"]:
#     print("警告：检测到回归！")
#     for r in result["regressions"]:
#         print(f"  {r['metric']}: {r['before']:.2%} → {r['after']:.2%} ({r['delta']:+.2%})")
```

---

## 练习题

### 基础题

1. **概念辨析**：用你自己的话解释模型评估和系统评估的区别。给出一个场景，其中模型评估分数高但系统评估分数低。

2. **指标计算**：给定以下预测和参考，手动计算 exact_match、contains 和 token-level F1：
   - 预测："北京是中国的首都城市"
   - 参考："北京"

3. **置信区间**：你的评估集有 50 个样本，准确率为 80%。这个结果可靠吗？你需要多少样本才能将 95% 置信区间缩小到 5% 以内？

### 实践题

1. **扩展 Harness**：为 `MinimalEvalHarness` 添加以下功能：
   - 支持按 `metadata` 中的类别分组统计（如按难度分组）
   - 支持并发调用模型（使用 `asyncio` 或 `concurrent.futures`）
   - 输出 Markdown 格式的报告

2. **多指标决策**：设计一个评估门禁（eval gate），当以下任一条件满足时阻止部署：
   - 任何指标的置信区间下界低于阈值
   - 任何指标相对上一版本回归超过 5%
   - 评估样本覆盖率不足（某些类别缺失）

### 思考题

1. 如果你的 AI 系统每天处理 10 万次查询，你如何设计评估数据集以确保它对真实流量有代表性？考虑长尾分布和新出现的查询模式。

2. "Goodhart's Law"（当一个指标变成目标，它就不再是好指标）如何影响 AI 评估？在实践中如何缓解这个问题？
