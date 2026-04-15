# 第9章：RAG 管线评估 Harness

> RAG（Retrieval-Augmented Generation）是 2024-2026 年最普遍的 AI 系统架构模式。但 RAG 也是最容易出错的——它有两个独立的失败点：检索可能找错文档，生成可能编造内容。更糟糕的是，这两个失败模式会互相掩盖：用户看到的是"回答不对"，但你需要评估 harness 才能定位到底是哪一层出了问题。本章系统讲解如何为 RAG 管线构建专用的评估 harness。

---

## 学习目标

学完本章，你将能够：

1. 分析 RAG 管线的双失败点问题，精确定位是检索还是生成出了问题
2. 计算关键检索指标：Precision@k、Recall@k、MRR 和 NDCG
3. 区分 faithfulness（忠实性）和 factual correctness（事实正确性），理解为什么两者都需要
4. 评估 RAG 系统的端到端性能，包括延迟和成本
5. 在 DeepEval、RAGAS、Arize Phoenix 等主流框架之间做出合理选择

---

## 9.1 RAG 的双失败点问题

### 9.1.1 RAG 管线的两层结构

```
用户查询 → [检索层] → 相关文档 → [生成层] → 最终回答
              │                      │
         可能失败点 1            可能失败点 2
         检索质量差              生成质量差
         (wrong docs)           (hallucination)
```

### 9.1.2 四种结果组合

| 检索 | 生成 | 结果 | 可检测性 |
|------|------|------|---------|
| 正确 | 正确 | 用户满意 | 无需检测 |
| 正确 | 错误 | 有正确文档但回答错误（幻觉） | 通过 faithfulness 指标检测 |
| 错误 | 正确 | 找错文档但碰巧答对（危险的假阳性） | 通过检索指标检测 |
| 错误 | 错误 | 找错文档且回答错误 | 两层指标都会报警 |

**关键洞察**：第三种组合最危险——它在端到端评估中看起来正确，但实际上系统是脆弱的。只有分层评估才能发现这类问题。

### 9.1.3 为什么必须分层评估

```python
def diagnose_rag_failure(
    query: str,
    retrieved_docs: list[str],
    gold_docs: list[str],
    generated_answer: str,
    gold_answer: str
) -> dict:
    """RAG 失败诊断器"""
    retrieval_ok = any(
        gold in retrieved for gold in gold_docs for retrieved in retrieved_docs
    )
    generation_ok = gold_answer.lower() in generated_answer.lower()

    if retrieval_ok and generation_ok:
        diagnosis = "PASS"
    elif retrieval_ok and not generation_ok:
        diagnosis = "GENERATION_FAILURE: 检索正确但生成错误（可能是幻觉或理解错误）"
    elif not retrieval_ok and generation_ok:
        diagnosis = "LUCKY_HIT: 检索失败但答案碰巧正确（系统不可靠）"
    else:
        diagnosis = "FULL_FAILURE: 检索和生成都失败"

    return {
        "query": query,
        "retrieval_ok": retrieval_ok,
        "generation_ok": generation_ok,
        "diagnosis": diagnosis,
    }
```

---

## 9.2 检索指标

### 9.2.1 指标总览

| 指标 | 全称 | 关注点 | 适用场景 |
|------|------|--------|---------|
| Precision@k | Precision at k | top-k 中有多少是相关的 | 用户只看前几条结果 |
| Recall@k | Recall at k | 相关文档有多少被召回 | 不能遗漏关键信息 |
| MRR | Mean Reciprocal Rank | 第一个相关结果的排名 | 用户期望第一条就对 |
| NDCG | Normalized Discounted Cumulative Gain | 排序质量（考虑相关度分级） | 多级相关性 |

### 9.2.2 指标实现

```python
import numpy as np

def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """
    Precision@k：top-k 结果中相关文档的比例

    Args:
        retrieved: 检索返回的文档 ID 列表（已按相关性排序）
        relevant: 相关文档 ID 集合（黄金标准）
        k: 截断位置
    """
    top_k = retrieved[:k]
    relevant_in_top_k = sum(1 for doc in top_k if doc in relevant)
    return relevant_in_top_k / k

def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """
    Recall@k：相关文档被召回的比例
    """
    if not relevant:
        return 1.0  # 没有相关文档时定义为 1
    top_k = retrieved[:k]
    recalled = sum(1 for doc in top_k if doc in relevant)
    return recalled / len(relevant)

def mean_reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """
    MRR：第一个相关结果的排名的倒数
    """
    for i, doc in enumerate(retrieved):
        if doc in relevant:
            return 1.0 / (i + 1)
    return 0.0  # 没有相关结果

def ndcg_at_k(retrieved: list[str], relevance_scores: dict[str, int], k: int) -> float:
    """
    NDCG@k：归一化折损累积增益

    Args:
        relevance_scores: {doc_id: relevance_score}（0=不相关, 1=部分相关, 2=高度相关）
    """
    # DCG
    dcg = 0.0
    for i, doc in enumerate(retrieved[:k]):
        rel = relevance_scores.get(doc, 0)
        dcg += rel / np.log2(i + 2)  # i+2 因为 log2(1) = 0

    # IDCG（理想排序的 DCG）
    ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_scores))

    return dcg / idcg if idcg > 0 else 0.0
```

### 9.2.3 检索评估示例

```python
# 示例：评估一次检索结果
retrieved_docs = ["doc_3", "doc_1", "doc_7", "doc_2", "doc_9"]
relevant_docs = {"doc_1", "doc_2", "doc_5"}

print(f"Precision@3: {precision_at_k(retrieved_docs, relevant_docs, 3):.2f}")  # 1/3 = 0.33
print(f"Recall@3:    {recall_at_k(retrieved_docs, relevant_docs, 3):.2f}")     # 1/3 = 0.33
print(f"Recall@5:    {recall_at_k(retrieved_docs, relevant_docs, 5):.2f}")     # 2/3 = 0.67
print(f"MRR:         {mean_reciprocal_rank(retrieved_docs, relevant_docs):.2f}") # 1/2 = 0.50

relevance_map = {"doc_1": 2, "doc_2": 2, "doc_3": 1, "doc_5": 2, "doc_7": 0, "doc_9": 0}
print(f"NDCG@5:      {ndcg_at_k(retrieved_docs, relevance_map, 5):.2f}")
```

---

## 9.3 生成指标

### 9.3.1 Faithfulness vs Factual Correctness

这是 RAG 评估中最容易混淆的两个概念：

| 维度 | Faithfulness（忠实性） | Factual Correctness（事实正确性） |
|------|----------------------|-------------------------------|
| 定义 | 回答是否忠于检索到的文档 | 回答是否与客观事实一致 |
| 参考基准 | 检索到的上下文 | 世界知识/黄金答案 |
| 检测目标 | 幻觉（编造上下文中没有的内容） | 错误（回答与事实不符） |
| 重要性 | 避免编造 | 避免误导 |

```
问题：公司的退款政策是什么？
检索到的文档：「30天内可退款，需提供发票。」

回答A："30天内可退款，需提供发票。"
  → faithfulness: 高 ✓  factual_correctness: 高 ✓

回答B："30天内可退款，需提供发票。超过30天可以换货。"
  → faithfulness: 低 ✗（"换货"是编造的）  factual_correctness: 部分正确

回答C："7天内可退款。"
  → faithfulness: 低 ✗（7天与文档不符）  factual_correctness: 低 ✗
```

### 9.3.2 Faithfulness 评估实现

```python
FAITHFULNESS_PROMPT = """
请评估以下回答对给定上下文的忠实程度。

**任务**：
1. 从回答中提取所有事实性声明（factual claims）
2. 对每个声明判断是否能从上下文中找到支撑
3. 计算忠实声明的比例

**上下文**：
{context}

**回答**：
{answer}

**请按以下 JSON 格式输出**：
{{
  "claims": [
    {{"claim": "声明内容", "supported": true/false, "evidence": "支撑或反驳的证据"}}
  ],
  "faithfulness_score": <0.0-1.0>,
  "reasoning": "整体分析"
}}
"""

class FaithfulnessEvaluator:
    """忠实性评估器"""

    def __init__(self, judge_model: str = "gpt-4o"):
        self.client = openai.OpenAI()
        self.judge_model = judge_model

    def evaluate(self, context: str, answer: str) -> dict:
        prompt = FAITHFULNESS_PROMPT.format(context=context, answer=answer)

        response = self.client.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)
        return result

    def batch_evaluate(self, samples: list[dict]) -> dict:
        """批量评估并汇总"""
        scores = []
        details = []
        for s in samples:
            result = self.evaluate(
                context=s["context"],
                answer=s["answer"]
            )
            scores.append(result.get("faithfulness_score", 0))
            details.append({"id": s.get("id"), **result})

        return {
            "mean_faithfulness": np.mean(scores),
            "std": np.std(scores),
            "min": np.min(scores),
            "max": np.max(scores),
            "details": details,
        }
```

### 9.3.3 Answer Relevance 评估

```python
ANSWER_RELEVANCE_PROMPT = """
请评估以下回答对问题的相关性。

**问题**：{question}
**回答**：{answer}

**评分标准**：
- 1.0：完全回答了问题的所有方面
- 0.75：回答了主要方面，有少量遗漏
- 0.5：部分相关，但遗漏了重要内容
- 0.25：勉强相关，大部分内容离题
- 0.0：完全不相关

**请按以下 JSON 格式输出**：
{{"score": <0.0-1.0>, "reasoning": "分析"}}
"""
```

---

## 9.4 端到端指标

### 9.4.1 超越准确性：生产环境关心什么

| 指标 | 为什么重要 | 目标值（参考） |
|------|-----------|-------------|
| 端到端延迟（P50/P95） | 用户体验 | P50 < 2s, P95 < 5s |
| 单查询成本 | 可持续性 | < $0.01 |
| 检索延迟 | 瓶颈定位 | < 200ms |
| 生成延迟 | 瓶颈定位 | < 3s |
| Token 使用量 | 成本控制 | context < 4k tokens |
| 管线版本 | 可追溯性 | 每次评估记录完整版本 |

### 9.4.2 端到端评估数据结构

```python
@dataclass
class RAGEvalResult:
    """RAG 单次查询的完整评估结果"""
    query_id: str
    query: str

    # 检索层
    retrieved_doc_ids: list[str]
    gold_doc_ids: list[str]
    precision_at_k: float
    recall_at_k: float
    mrr: float
    retrieval_latency_ms: float

    # 生成层
    generated_answer: str
    gold_answer: str
    faithfulness: float
    answer_relevance: float
    generation_latency_ms: float

    # 端到端
    total_latency_ms: float
    total_tokens: int
    estimated_cost_usd: float

    # 元数据
    pipeline_version: str
    timestamp: str

class RAGEvalHarness:
    """RAG 管线评估 Harness"""

    def __init__(self, rag_pipeline, config: dict):
        self.pipeline = rag_pipeline
        self.config = config
        self.faithfulness_eval = FaithfulnessEvaluator()

    def evaluate_single(self, sample: dict) -> RAGEvalResult:
        """评估单个查询"""
        import time

        # 1. 检索
        t0 = time.time()
        retrieved = self.pipeline.retrieve(sample["query"], k=self.config["k"])
        retrieval_latency = (time.time() - t0) * 1000

        retrieved_ids = [doc["id"] for doc in retrieved]
        gold_ids = set(sample["gold_doc_ids"])

        # 2. 生成
        t1 = time.time()
        context = "\n".join(doc["content"] for doc in retrieved)
        answer, token_usage = self.pipeline.generate(sample["query"], context)
        generation_latency = (time.time() - t1) * 1000

        # 3. 计算指标
        k = self.config["k"]
        faith_result = self.faithfulness_eval.evaluate(context=context, answer=answer)

        return RAGEvalResult(
            query_id=sample["id"],
            query=sample["query"],
            retrieved_doc_ids=retrieved_ids,
            gold_doc_ids=sample["gold_doc_ids"],
            precision_at_k=precision_at_k(retrieved_ids, gold_ids, k),
            recall_at_k=recall_at_k(retrieved_ids, gold_ids, k),
            mrr=mean_reciprocal_rank(retrieved_ids, gold_ids),
            retrieval_latency_ms=retrieval_latency,
            generated_answer=answer,
            gold_answer=sample.get("gold_answer", ""),
            faithfulness=faith_result.get("faithfulness_score", 0),
            answer_relevance=0.0,  # 类似方式计算
            generation_latency_ms=generation_latency,
            total_latency_ms=retrieval_latency + generation_latency,
            total_tokens=token_usage,
            estimated_cost_usd=token_usage * 0.000003,  # 按模型定价
            pipeline_version=self.config["version"],
            timestamp=datetime.now().isoformat(),
        )

    def run(self, dataset: list[dict]) -> dict:
        """运行完整评估"""
        results = [self.evaluate_single(s) for s in dataset]

        # 汇总
        summary = {
            "pipeline_version": self.config["version"],
            "num_queries": len(results),
            "retrieval": {
                "mean_precision_at_k": np.mean([r.precision_at_k for r in results]),
                "mean_recall_at_k": np.mean([r.recall_at_k for r in results]),
                "mean_mrr": np.mean([r.mrr for r in results]),
                "mean_latency_ms": np.mean([r.retrieval_latency_ms for r in results]),
            },
            "generation": {
                "mean_faithfulness": np.mean([r.faithfulness for r in results]),
                "mean_relevance": np.mean([r.answer_relevance for r in results]),
                "mean_latency_ms": np.mean([r.generation_latency_ms for r in results]),
            },
            "end_to_end": {
                "mean_latency_ms": np.mean([r.total_latency_ms for r in results]),
                "p95_latency_ms": np.percentile([r.total_latency_ms for r in results], 95),
                "mean_cost_usd": np.mean([r.estimated_cost_usd for r in results]),
                "total_cost_usd": sum(r.estimated_cost_usd for r in results),
            },
        }
        return summary
```

---

## 9.5 框架对比

### 9.5.1 主流 RAG 评估框架

| 框架 | 定位 | 核心优势 | 局限 |
|------|------|---------|------|
| **DeepEval** | 全功能评估框架 | 指标丰富、Pytest 集成、CI 友好 | 部分指标依赖 GPT-4 |
| **RAGAS** | RAG 专用评估 | 学术基础扎实、指标设计严谨 | 生态较小 |
| **Arize Phoenix** | 可观测性 + 评估 | 可视化强、trace 集成 | 重量级 |
| **Maxim AI** | 企业级评估平台 | SaaS 托管、团队协作 | 非开源 |

### 9.5.2 DeepEval 示例

```python
from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
from deepeval.test_case import LLMTestCase

# 构造测试用例
test_case = LLMTestCase(
    input="公司的年假政策是什么？",
    actual_output="员工入职满一年后享有 10 天年假，满五年后 15 天。",
    expected_output="入职满一年享有10天年假，满五年15天年假。",
    retrieval_context=[
        "根据公司员工手册第 5.2 条，员工入职满一年后享有 10 天带薪年假。"
        "入职满五年后，年假增加至 15 天。年假需提前两周申请。"
    ],
)

# 定义指标
metrics = [
    FaithfulnessMetric(threshold=0.7, model="gpt-4o"),
    AnswerRelevancyMetric(threshold=0.7, model="gpt-4o"),
    ContextualPrecisionMetric(threshold=0.7, model="gpt-4o"),
    ContextualRecallMetric(threshold=0.7, model="gpt-4o"),
]

# 运行评估
results = evaluate([test_case], metrics)
```

### 9.5.3 RAGAS 示例

```python
from ragas import evaluate as ragas_evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

# 准备数据（RAGAS 使用 HF Dataset 格式）
data = {
    "question": ["公司的年假政策是什么？"],
    "answer": ["员工入职满一年后享有 10 天年假，满五年后 15 天。"],
    "contexts": [[
        "根据公司员工手册第 5.2 条，员工入职满一年后享有 10 天带薪年假。"
        "入职满五年后，年假增加至 15 天。年假需提前两周申请。"
    ]],
    "ground_truth": ["入职满一年享有10天年假，满五年15天年假。"],
}
dataset = Dataset.from_dict(data)

# 运行评估
result = ragas_evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)
print(result)
# {'faithfulness': 0.95, 'answer_relevancy': 0.88, ...}
```

### 9.5.4 框架选择决策树

```
你的需求是什么？
├── 快速验证 RAG 质量 → RAGAS（轻量、聚焦）
├── CI/CD 集成 → DeepEval（Pytest 集成、门禁）
├── 生产可观测性 → Arize Phoenix（trace + 评估）
└── 企业级团队协作 → Maxim AI（SaaS、仪表盘）
```

---

## 9.6 从单脚本到 CI 集成

### 9.6.1 演进路径

| 阶段 | 工具 | 触发方式 | 耗时 |
|------|------|---------|------|
| V1 | Python 脚本 | 手动运行 | 分钟级 |
| V2 | Pytest + DeepEval | 本地 `pytest` | 分钟级 |
| V3 | GitHub Actions | PR 自动触发 | 5-15分钟 |
| V4 | 定时 + 事件驱动 | 每日 + PR + 模型更新 | 持续 |

### 9.6.2 Pytest 集成示例

```python
# tests/test_rag_eval.py
import pytest
from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

@pytest.fixture
def rag_pipeline():
    """加载 RAG 管线"""
    from my_app.rag import RAGPipeline
    return RAGPipeline(config="eval_config.yaml")

@pytest.fixture
def eval_dataset():
    """加载评估数据集"""
    import json
    with open("tests/data/rag_eval_v1.2.json") as f:
        return json.load(f)

def test_faithfulness_above_threshold(rag_pipeline, eval_dataset):
    """忠实性评估门禁：均分不低于 0.8"""
    faithfulness = FaithfulnessMetric(threshold=0.8, model="gpt-4o")

    for sample in eval_dataset["samples"][:20]:  # CI 中用子集
        result = rag_pipeline.query(sample["question"])
        test_case = LLMTestCase(
            input=sample["question"],
            actual_output=result["answer"],
            retrieval_context=result["contexts"],
        )
        assert_test(test_case, [faithfulness])

def test_retrieval_recall(rag_pipeline, eval_dataset):
    """检索召回率门禁：Recall@5 不低于 0.7"""
    recalls = []
    for sample in eval_dataset["samples"]:
        results = rag_pipeline.retrieve(sample["question"], k=5)
        retrieved_ids = [r["id"] for r in results]
        gold_ids = set(sample["gold_doc_ids"])
        recalls.append(recall_at_k(retrieved_ids, gold_ids, 5))

    mean_recall = sum(recalls) / len(recalls)
    assert mean_recall >= 0.7, f"Recall@5 = {mean_recall:.2f} < 0.7 阈值"
```

### 9.6.3 GitHub Actions 配置

```yaml
# .github/workflows/rag-eval.yml
name: RAG Evaluation
on:
  pull_request:
    paths:
      - 'src/rag/**'
      - 'prompts/**'
      - 'config/rag_config.yaml'

jobs:
  rag-eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -r requirements-eval.txt

      - name: Run RAG evaluation
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: pytest tests/test_rag_eval.py -v --tb=short

      - name: Upload evaluation report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: rag-eval-report
          path: eval_results/
```

---

## 本章小结

| 概念 | 要点 |
|------|------|
| 双失败点 | RAG 有检索和生成两个独立失败点，必须分层评估 |
| 检索指标 | Precision@k（准确性）、Recall@k（完整性）、MRR（排名）、NDCG（排序质量） |
| Faithfulness vs Factual | faithfulness 看是否忠于上下文，factual correctness 看是否符合事实 |
| 端到端指标 | 延迟（P50/P95）、单查询成本、token 使用量、管线版本 |
| 框架选择 | RAGAS（轻量验证）、DeepEval（CI 集成）、Phoenix（可观测性）、Maxim（企业级） |
| CI 集成 | Pytest + DeepEval 门禁 → GitHub Actions 自动触发 |

---

## 动手实验

### 实验 1：实现 RAG 检索指标计算器

**目标**：实现并验证四个核心检索指标。

```python
# 测试数据
test_cases = [
    {
        "query": "Python GIL 是什么？",
        "retrieved": ["doc_a", "doc_c", "doc_b", "doc_e", "doc_d"],
        "relevant": {"doc_a", "doc_b", "doc_d"},
        "relevance_scores": {"doc_a": 2, "doc_b": 2, "doc_c": 0, "doc_d": 1, "doc_e": 0},
    },
    {
        "query": "如何使用 Docker？",
        "retrieved": ["doc_x", "doc_y", "doc_z", "doc_w", "doc_v"],
        "relevant": {"doc_y", "doc_w"},
        "relevance_scores": {"doc_x": 0, "doc_y": 2, "doc_z": 0, "doc_w": 2, "doc_v": 1},
    },
]

# 计算并打印每个查询的指标
for tc in test_cases:
    print(f"\nQuery: {tc['query']}")
    print(f"  Precision@3: {precision_at_k(tc['retrieved'], tc['relevant'], 3):.3f}")
    print(f"  Recall@3:    {recall_at_k(tc['retrieved'], tc['relevant'], 3):.3f}")
    print(f"  Recall@5:    {recall_at_k(tc['retrieved'], tc['relevant'], 5):.3f}")
    print(f"  MRR:         {mean_reciprocal_rank(tc['retrieved'], tc['relevant']):.3f}")
    print(f"  NDCG@5:      {ndcg_at_k(tc['retrieved'], tc['relevance_scores'], 5):.3f}")
```

**验证**：手动计算第一个 test case 的 Precision@3，确认你的实现正确。

### 实验 2：为简单 RAG 管线编写完整评估脚本

**目标**：整合检索指标和生成指标，输出完整的 RAG 评估报告。

```python
import json
import time

def evaluate_simple_rag(
    rag_fn,                    # Callable: query -> {"docs": [...], "answer": str}
    eval_dataset: list[dict],  # [{"query": str, "gold_docs": [str], "gold_answer": str}]
    k: int = 5
) -> dict:
    """简化版 RAG 评估脚本"""
    retrieval_metrics = {"precision": [], "recall": [], "mrr": []}
    generation_metrics = {"contains_match": []}
    latencies = []

    for sample in eval_dataset:
        t0 = time.time()
        result = rag_fn(sample["query"])
        latency = (time.time() - t0) * 1000

        # 检索指标
        retrieved_ids = [d["id"] for d in result["docs"]]
        gold_ids = set(sample["gold_doc_ids"])

        retrieval_metrics["precision"].append(precision_at_k(retrieved_ids, gold_ids, k))
        retrieval_metrics["recall"].append(recall_at_k(retrieved_ids, gold_ids, k))
        retrieval_metrics["mrr"].append(mean_reciprocal_rank(retrieved_ids, gold_ids))

        # 生成指标（简化版：包含匹配）
        answer_correct = sample["gold_answer"].lower() in result["answer"].lower()
        generation_metrics["contains_match"].append(1.0 if answer_correct else 0.0)

        latencies.append(latency)

    # 汇总报告
    report = {
        "num_queries": len(eval_dataset),
        "retrieval": {
            name: {"mean": np.mean(scores), "std": np.std(scores)}
            for name, scores in retrieval_metrics.items()
        },
        "generation": {
            name: {"mean": np.mean(scores), "std": np.std(scores)}
            for name, scores in generation_metrics.items()
        },
        "latency": {
            "mean_ms": np.mean(latencies),
            "p95_ms": np.percentile(latencies, 95),
        },
    }

    # 打印报告
    print("\n" + "=" * 50)
    print("RAG 评估报告")
    print("=" * 50)
    print(f"查询数: {report['num_queries']}")
    print("\n-- 检索层 --")
    for name, stats in report["retrieval"].items():
        print(f"  {name:>12}: {stats['mean']:.3f} (std={stats['std']:.3f})")
    print("\n-- 生成层 --")
    for name, stats in report["generation"].items():
        print(f"  {name:>12}: {stats['mean']:.3f} (std={stats['std']:.3f})")
    print(f"\n-- 延迟 --")
    print(f"  {'mean':>12}: {report['latency']['mean_ms']:.0f}ms")
    print(f"  {'p95':>12}: {report['latency']['p95_ms']:.0f}ms")
    print("=" * 50)

    return report
```

---

## 练习题

### 基础题

1. **指标辨析**：解释 Precision@5=0.6 和 Recall@5=0.6 在实际含义上的区别。给出一个场景，你更关心 Precision；再给一个更关心 Recall 的场景。

2. **Faithfulness vs Factual**：以下 RAG 回答的 faithfulness 和 factual correctness 分别如何？
   - 检索到的文档："地球绕太阳一周需要 365.25 天。"
   - 生成的回答："地球绕太阳一周需要 365 天，即一年。同时，月球绕地球一周需要约 27 天。"

3. **双失败点诊断**：描述一个具体场景，其中 RAG 系统的检索正确但生成错误。用户看到的症状是什么？你如何通过评估指标定位问题？

### 实践题

1. **框架对比实验**：选择 DeepEval 或 RAGAS 中的一个，对同一组 RAG 测试数据运行评估。记录安装过程、API 调用方式和输出格式。与本章实现的自定义评估脚本对比：框架带来了哪些便利？有哪些限制？

2. **检索优化实验**：用你的检索指标计算器，对比以下两种检索策略的指标差异：
   - 策略 A：纯语义检索（embedding cosine similarity）
   - 策略 B：混合检索（语义 + BM25 关键词匹配）
   设计至少 10 个查询的评估数据集，记录两种策略在 Precision@3、Recall@5、MRR 上的差异。

### 思考题

1. RAG 评估依赖黄金标注（gold documents、gold answers）。但在实际产品中，文档库不断更新，黄金标注很快过时。你如何设计一个"自适应评估"策略，在文档库变化后自动更新或标记需要重新标注的评估样本？

2. 如果你的 RAG 系统处理多语言查询（中文、英文、日文），评估 harness 需要做哪些调整？考虑指标定义、LLM-as-Judge 的语言能力、以及数据集构建。
