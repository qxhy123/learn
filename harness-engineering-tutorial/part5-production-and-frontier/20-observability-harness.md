# 第20章：可观测性 Harness

> "没有评估的 tracing 只是昂贵的日志。"——在 LLM 应用中，你记录了每一次 API 调用、每一个 prompt 版本、每一毫秒的延迟，但如果不能回答"这次回答好不好"，那你只是在高价收集垃圾数据。

---

## 学习目标

学完本章，你将能够：

1. 理解为什么传统可观测性不足以应对 LLM 系统
2. 使用 OpenTelemetry 为 LLM 应用做插桩
3. 实现 trace-to-prompt-version linking
4. 部署 embedding 语义漂移检测
5. 理解实时幻觉检测方法（MetaQA、CLAP）

---

## 20.1 为什么"没有评估的 Tracing 只是昂贵的日志"

### 传统可观测性 vs LLM 可观测性

```
传统 Web 服务：                    LLM 应用：
┌────────────────────┐             ┌────────────────────┐
│ 请求进来           │             │ 请求进来           │
│ → 处理逻辑确定     │             │ → 处理逻辑非确定   │
│ → 结果正确或报错   │             │ → 结果"看起来对"   │
│ → HTTP 200/500     │             │ → HTTP 200 但内容  │
│                    │             │   可能完全错误      │
│ 监控：延迟、错误率  │             │ 监控：延迟、错误率  │
│ → 够了             │             │ → 远远不够          │
└────────────────────┘             └────────────────────┘
```

### LLM 可观测性的三层金字塔

```
            ╱╲
           ╱  ╲         第 3 层：评估（最有价值）
          ╱ 评估╲        回答质量如何？幻觉了吗？
         ╱──────╲
        ╱        ╲       第 2 层：语义追踪
       ╱ 语义追踪 ╲      prompt 版本、检索质量、模型行为
      ╱────────────╲
     ╱              ╲    第 1 层：基础指标
    ╱  基础指标      ╲   延迟、token 数、成本、错误率
   ╱──────────────────╲
```

大多数团队停留在第 1 层——这就是"昂贵的日志"问题。

### 需要追踪什么

| 层级 | 指标 | 传统监控有吗 | LLM 特有 |
|------|------|-------------|----------|
| 基础 | 延迟 (p50/p95/p99) | 有 | |
| 基础 | Token 消耗 / 成本 | | 是 |
| 基础 | 错误率 / 限流 | 有 | |
| 语义 | Prompt 版本 | | 是 |
| 语义 | 检索文档 relevance | | 是 |
| 语义 | Embedding 漂移 | | 是 |
| 评估 | 回答质量分数 | | 是 |
| 评估 | 幻觉检测 | | 是 |
| 评估 | 用户满意度 | 部分 | |

---

## 20.2 OpenTelemetry 为 LLM 应用做插桩

### 为什么选 OpenTelemetry

OpenTelemetry（OTel）是厂商中立的可观测性标准。2026 年，它已经成为 LLM 可观测性的事实标准：

```
OTel 的 LLM 语义约定（Semantic Conventions）：

gen_ai.system = "anthropic"          # 提供商
gen_ai.request.model = "claude-sonnet-4-20250514"  # 模型
gen_ai.request.max_tokens = 4096      # 最大 token
gen_ai.response.finish_reason = "stop" # 结束原因
gen_ai.usage.input_tokens = 1500      # 输入 token
gen_ai.usage.output_tokens = 800      # 输出 token
```

### 插桩实现

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
import time

# 初始化 OTel
resource = Resource.create({
    "service.name": "rag-service",
    "service.version": "1.2.0",
    "deployment.environment": "production",
})

provider = TracerProvider(resource=resource)
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="http://otel-collector:4317"))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("rag-service")


class InstrumentedLLMClient:
    """带 OTel 插桩的 LLM 客户端"""

    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    def generate(self, messages: list[dict], **kwargs) -> dict:
        with tracer.start_as_current_span("llm.generate") as span:
            # 记录请求属性
            span.set_attribute("gen_ai.system", "anthropic")
            span.set_attribute("gen_ai.request.model", self.model)
            span.set_attribute("gen_ai.request.max_tokens", kwargs.get("max_tokens", 4096))
            span.set_attribute("gen_ai.request.temperature", kwargs.get("temperature", 0.7))

            # 记录 prompt 版本（关键！）
            prompt_hash = self._hash_prompt(messages)
            span.set_attribute("harness.prompt_version", prompt_hash)

            start_time = time.time()
            try:
                response = self.client.messages.create(
                    model=self.model,
                    messages=messages,
                    **kwargs,
                )

                # 记录响应属性
                span.set_attribute("gen_ai.response.finish_reason", response.stop_reason)
                span.set_attribute("gen_ai.usage.input_tokens", response.usage.input_tokens)
                span.set_attribute("gen_ai.usage.output_tokens", response.usage.output_tokens)
                span.set_attribute("harness.latency_ms", (time.time() - start_time) * 1000)

                return {
                    "content": response.content[0].text,
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    },
                    "trace_id": format(span.get_span_context().trace_id, "032x"),
                }

            except Exception as e:
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                raise

    def _hash_prompt(self, messages: list[dict]) -> str:
        """生成 prompt 版本哈希"""
        import hashlib
        # 只哈希 system prompt（用户消息每次不同）
        system_msgs = [m["content"] for m in messages if m["role"] == "system"]
        content = "||".join(system_msgs)
        return hashlib.sha256(content.encode()).hexdigest()[:12]
```

### 检索阶段插桩

```python
class InstrumentedRetriever:
    """带插桩的检索器"""

    def __init__(self, retriever):
        self.retriever = retriever

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        with tracer.start_as_current_span("retrieval.search") as span:
            span.set_attribute("retrieval.query_length", len(query))
            span.set_attribute("retrieval.top_k", top_k)

            start_time = time.time()
            results = self.retriever.search(query, top_k=top_k)

            span.set_attribute("retrieval.result_count", len(results))
            span.set_attribute("retrieval.latency_ms", (time.time() - start_time) * 1000)

            # 记录 top result 的相似度分数
            if results:
                span.set_attribute("retrieval.top_score", results[0].get("score", 0))
                span.set_attribute("retrieval.min_score", results[-1].get("score", 0))

            return results
```

---

## 20.3 Trace-to-Prompt-Version Linking（Langfuse 模式）

### 问题：哪个 Prompt 版本导致了回归？

```
时间线：
─────────────────────────────────────────→ t

v1.0 prompt     v1.1 prompt      v1.2 prompt
  部署            部署              部署
   │               │                │
   ▼               ▼                ▼
质量: 0.85       质量: 0.90       质量: 0.72 ← 回归！

没有 linking → 不知道是 v1.2 导致了回归
有 linking   → 立即定位到 v1.2 的变更
```

### 实现 Prompt 版本管理

```python
import json
from datetime import datetime
from pathlib import Path

class PromptVersionManager:
    """Prompt 版本管理器：关联 trace 和 prompt 版本"""

    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.versions: dict[str, dict] = {}

    def register_prompt(
        self,
        name: str,
        template: str,
        metadata: dict | None = None,
    ) -> str:
        """注册新的 prompt 版本"""
        import hashlib
        version_hash = hashlib.sha256(template.encode()).hexdigest()[:12]
        version_id = f"{name}@{version_hash}"

        self.versions[version_id] = {
            "name": name,
            "template": template,
            "version_hash": version_hash,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        # 持久化
        path = self.storage_dir / f"{version_id}.json"
        with open(path, "w") as f:
            json.dump(self.versions[version_id], f, ensure_ascii=False, indent=2)

        return version_id

    def get_prompt(self, version_id: str) -> dict:
        """获取指定版本的 prompt"""
        if version_id in self.versions:
            return self.versions[version_id]
        path = self.storage_dir / f"{version_id}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        raise ValueError(f"版本 {version_id} 不存在")

    def compare_versions(self, v1: str, v2: str) -> dict:
        """对比两个版本的差异"""
        p1 = self.get_prompt(v1)
        p2 = self.get_prompt(v2)

        import difflib
        diff = list(difflib.unified_diff(
            p1["template"].splitlines(),
            p2["template"].splitlines(),
            fromfile=v1, tofile=v2, lineterm="",
        ))
        return {
            "v1": v1,
            "v2": v2,
            "diff": "\n".join(diff),
            "v1_created": p1["created_at"],
            "v2_created": p2["created_at"],
        }


class TracePromptLinker:
    """将 trace 与 prompt 版本关联"""

    def __init__(self, prompt_manager: PromptVersionManager):
        self.prompt_manager = prompt_manager
        self.trace_links: list[dict] = []

    def link(self, trace_id: str, prompt_version_id: str, quality_score: float):
        """记录 trace ↔ prompt version ↔ quality 的关联"""
        self.trace_links.append({
            "trace_id": trace_id,
            "prompt_version": prompt_version_id,
            "quality_score": quality_score,
            "timestamp": datetime.now().isoformat(),
        })

    def get_version_quality(self, version_id: str) -> dict:
        """获取某个 prompt 版本的质量统计"""
        scores = [
            link["quality_score"]
            for link in self.trace_links
            if link["prompt_version"] == version_id
        ]
        if not scores:
            return {"version": version_id, "count": 0}

        return {
            "version": version_id,
            "count": len(scores),
            "mean": sum(scores) / len(scores),
            "min": min(scores),
            "max": max(scores),
            "p50": sorted(scores)[len(scores) // 2],
        }

    def detect_regression(self, threshold: float = 0.1) -> list[dict]:
        """检测 prompt 版本之间的质量回归"""
        # 按版本分组
        by_version = {}
        for link in self.trace_links:
            v = link["prompt_version"]
            if v not in by_version:
                by_version[v] = []
            by_version[v].append(link["quality_score"])

        # 按时间排序版本
        versions = sorted(
            by_version.keys(),
            key=lambda v: self.prompt_manager.get_prompt(v)["created_at"],
        )

        regressions = []
        for i in range(1, len(versions)):
            prev_mean = sum(by_version[versions[i-1]]) / len(by_version[versions[i-1]])
            curr_mean = sum(by_version[versions[i]]) / len(by_version[versions[i]])
            if prev_mean - curr_mean > threshold:
                regressions.append({
                    "from_version": versions[i-1],
                    "to_version": versions[i],
                    "quality_drop": prev_mean - curr_mean,
                    "prev_mean": prev_mean,
                    "curr_mean": curr_mean,
                })

        return regressions
```

---

## 20.4 Embedding 语义漂移检测

### 什么是语义漂移

当 embedding 模型更新、或数据分布变化时，相同查询的 embedding 会悄然改变——**检索结果不同了，但没有任何代码变更**：

```
时间 T1（模型 v1）:
  query "如何重置密码" → embedding_v1 → 检索到"密码重置指南"  ✓

时间 T2（模型 v2 或数据漂移）:
  query "如何重置密码" → embedding_v2 → 检索到"密码安全策略"  ✗
```

### 漂移检测实现

```python
import numpy as np
from typing import Optional

class EmbeddingDriftDetector:
    """Embedding 语义漂移检测器"""

    def __init__(self, reference_embeddings: dict[str, list[float]]):
        """
        reference_embeddings: 基准 embedding（在系统部署时采集）
        {"query_1": [0.1, 0.2, ...], "query_2": [...]}
        """
        self.reference = reference_embeddings

    def check_drift(
        self,
        current_embeddings: dict[str, list[float]],
        threshold: float = 0.05,
    ) -> dict:
        """检查当前 embedding 是否偏离基准"""
        drifts = []
        for query, ref_emb in self.reference.items():
            if query not in current_embeddings:
                continue
            curr_emb = current_embeddings[query]

            # 计算余弦距离
            distance = 1 - self._cosine_similarity(ref_emb, curr_emb)

            if distance > threshold:
                drifts.append({
                    "query": query,
                    "cosine_distance": distance,
                    "severity": "HIGH" if distance > 0.15 else "MEDIUM",
                })

        total_queries = len(self.reference)
        drifted_count = len(drifts)

        return {
            "drifted": drifted_count > 0,
            "drift_ratio": drifted_count / max(total_queries, 1),
            "details": sorted(drifts, key=lambda d: d["cosine_distance"], reverse=True),
            "recommendation": self._recommend(drifted_count, total_queries),
        }

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        a_np, b_np = np.array(a), np.array(b)
        return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np) + 1e-8))

    def _recommend(self, drifted: int, total: int) -> str:
        ratio = drifted / max(total, 1)
        if ratio > 0.3:
            return "严重漂移：建议重新构建向量索引"
        elif ratio > 0.1:
            return "中度漂移：建议增量更新受影响文档的 embedding"
        elif ratio > 0:
            return "轻微漂移：继续监控"
        return "无漂移"


class DriftMonitorScheduler:
    """定期运行漂移检测"""

    def __init__(self, detector: EmbeddingDriftDetector, embedder, alert_callback=None):
        self.detector = detector
        self.embedder = embedder
        self.alert_callback = alert_callback
        self.probe_queries = list(detector.reference.keys())

    def run_check(self) -> dict:
        """执行一次漂移检查"""
        current = {}
        for query in self.probe_queries:
            current[query] = self.embedder.embed(query)

        result = self.detector.check_drift(current)

        if result["drifted"] and self.alert_callback:
            self.alert_callback(result)

        return result
```

---

## 20.5 实时幻觉检测：MetaQA 与 CLAP

### MetaQA：变形 Prompt 突变检测

MetaQA 的思路是：**如果模型真的"知道"答案，那换一种方式问应该得到一致的答案**。

```
原始问题: "法国首都是什么？"
变形1:     "哪个城市是法国的首都？"
变形2:     "France 的 capital city 是？"
变形3:     "巴黎是_____的首都"（反向验证）

如果 4 个回答一致 → 高置信度
如果 4 个回答不一致 → 可能是幻觉
```

```python
class MetaQADetector:
    """MetaQA 幻觉检测器：通过 prompt 变形检测一致性"""

    def __init__(self, llm_client, mutator_llm=None):
        self.llm = llm_client
        self.mutator = mutator_llm or llm_client

    def detect(self, question: str, original_answer: str, num_mutations: int = 3) -> dict:
        """检测回答是否可能是幻觉"""
        # Step 1: 生成变形问题
        mutations = self._generate_mutations(question, num_mutations)

        # Step 2: 对每个变形问题获取回答
        answers = [original_answer]
        for mutation in mutations:
            answer = self.llm.generate([{
                "role": "user",
                "content": mutation,
            }])
            answers.append(answer)

        # Step 3: 检查一致性
        consistency = self._check_consistency(answers)

        return {
            "is_hallucination": consistency < 0.6,
            "consistency_score": consistency,
            "original_answer": original_answer,
            "mutation_answers": list(zip(mutations, answers[1:])),
            "verdict": self._verdict(consistency),
        }

    def _generate_mutations(self, question: str, n: int) -> list[str]:
        """生成问题的变形版本"""
        prompt = f"""请将以下问题改写成 {n} 个不同的表达方式，保持语义不变：

原始问题：{question}

只输出改写后的问题，每行一个。"""
        response = self.mutator.generate([{"role": "user", "content": prompt}])
        return [line.strip() for line in response.strip().split("\n") if line.strip()][:n]

    def _check_consistency(self, answers: list[str]) -> float:
        """检查多个回答的一致性"""
        if len(answers) < 2:
            return 1.0

        # 简化：使用关键词重叠度
        base_keywords = set(answers[0])
        overlaps = []
        for answer in answers[1:]:
            answer_keywords = set(answer)
            overlap = len(base_keywords & answer_keywords) / max(len(base_keywords | answer_keywords), 1)
            overlaps.append(overlap)

        return sum(overlaps) / len(overlaps)

    def _verdict(self, consistency: float) -> str:
        if consistency >= 0.8:
            return "HIGH_CONFIDENCE（一致性高，可能不是幻觉）"
        elif consistency >= 0.6:
            return "MEDIUM_CONFIDENCE（需要人工验证）"
        else:
            return "LOW_CONFIDENCE（一致性低，可能是幻觉）"
```

### CLAP：注意力探测

CLAP（Confidence via Language model Attention Probing）通过分析模型的注意力分布来检测不确定性：

```
高置信度回答：                    低置信度（可能幻觉）：
注意力集中在文档证据上             注意力分散在无关 token 上

  [文档A] ████████░░             [文档A] ██░░░░░░░░
  [文档B] ███░░░░░░░             [文档B] ██░░░░░░░░
  [query] ██░░░░░░░░             [query] █████░░░░░
  [其他]  █░░░░░░░░░             [其他]  ████████░░
                                         ↑
                                  注意力落在非证据位置
```

```python
class CLAPDetector:
    """CLAP 风格的置信度检测"""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def estimate_confidence(
        self,
        answer: str,
        source_docs: list[str],
        attention_weights: list[float] | None = None,
    ) -> dict:
        """
        估计回答置信度。
        在没有真实 attention weights 时，使用代理指标。
        """
        # 代理指标 1: 答案中引用源文档的程度
        citation_score = self._citation_coverage(answer, source_docs)

        # 代理指标 2: 答案的具体性（具体比模糊好）
        specificity_score = self._specificity(answer)

        # 代理指标 3: 避免不确定表达
        certainty_score = self._certainty(answer)

        confidence = (citation_score * 0.5 + specificity_score * 0.3 + certainty_score * 0.2)

        return {
            "confidence": confidence,
            "is_reliable": confidence >= self.threshold,
            "breakdown": {
                "citation_coverage": citation_score,
                "specificity": specificity_score,
                "certainty": certainty_score,
            },
        }

    def _citation_coverage(self, answer: str, docs: list[str]) -> float:
        """答案内容在源文档中的覆盖率"""
        if not docs:
            return 0.0
        answer_chars = set(answer)
        doc_chars = set()
        for doc in docs:
            doc_chars.update(set(doc))
        overlap = len(answer_chars & doc_chars) / max(len(answer_chars), 1)
        return min(overlap, 1.0)

    def _specificity(self, answer: str) -> float:
        """具体性评分"""
        vague_markers = ["一些", "某些", "大概", "之类的", "等等", "可能"]
        specific_markers = ["具体来说", "例如", "根据", "数据显示", "在...中"]
        vague_count = sum(1 for m in vague_markers if m in answer)
        specific_count = sum(1 for m in specific_markers if m in answer)
        return min(1.0, (specific_count + 1) / (vague_count + specific_count + 1))

    def _certainty(self, answer: str) -> float:
        """确定性评分"""
        uncertain = ["不确定", "不太清楚", "可能是", "也许", "我猜"]
        count = sum(1 for u in uncertain if u in answer)
        return max(0.0, 1.0 - count * 0.25)
```

---

## 20.6 平台对比与闭环系统

### 可观测性平台对比

| 平台 | 核心强项 | Prompt 版本管理 | 幻觉检测 | 价格模型 |
|------|---------|---------------|---------|---------|
| Arize AI | Embedding 漂移、数据质量 | 有 | LLM-as-judge | 按数据量 |
| Langfuse | Trace-to-prompt linking | 原生支持 | 需集成 | 开源+托管 |
| Confident AI | DeepEval 评估集成 | 有 | 内置多种 | 按评估量 |
| Galileo | 实时监控、幻觉分数 | 有 | 内置 Luna | 企业订阅 |
| Maxim AI | 全链路 agent 追踪 | 有 | 内置 | 按 trace 量 |

### 闭环系统：从 Trace 到改进

```
                    ┌────────────┐
                    │ 生产流量   │
                    └─────┬──────┘
                          │
                    ┌─────▼──────┐
                    │ Tracing +  │
              ┌─────│ Evaluation │──────┐
              │     └────────────┘      │
              │                         │
       ┌──────▼──────┐          ┌──────▼──────┐
       │ Failure     │          │ Success     │
       │ Clustering  │          │ Patterns    │
       │ 聚类失败模式 │          │ 成功模式   │
       └──────┬──────┘          └──────┬──────┘
              │                         │
       ┌──────▼──────┐          ┌──────▼──────┐
       │ Root Cause  │          │ Golden      │
       │ Analysis    │          │ Examples    │
       │ 根因分析    │          │ 提取        │
       └──────┬──────┘          └──────┬──────┘
              │                         │
              └────────┬────────────────┘
                       │
                ┌──────▼──────┐
                │ Harness     │
                │ Update      │
                │ 更新 harness │
                └─────────────┘
```

```python
class ClosedLoopSystem:
    """闭环可观测系统：trace → 分析 → 改进"""

    def __init__(self, trace_store, evaluator, harness_updater):
        self.traces = trace_store
        self.evaluator = evaluator
        self.updater = harness_updater

    def analyze_failures(self, time_window_hours: int = 24) -> dict:
        """分析最近的失败模式"""
        # 获取最近的失败 trace
        failed_traces = self.traces.query(
            time_window=time_window_hours,
            filter={"quality_score": {"$lt": 0.6}},
        )

        # 聚类失败模式
        clusters = self._cluster_failures(failed_traces)

        # 为每个聚类生成修复建议
        recommendations = []
        for cluster in clusters:
            rec = self._generate_fix(cluster)
            recommendations.append(rec)

        return {
            "total_failures": len(failed_traces),
            "clusters": len(clusters),
            "recommendations": recommendations,
        }

    def _cluster_failures(self, traces: list[dict]) -> list[dict]:
        """聚类失败模式（简化版）"""
        clusters = {}
        for t in traces:
            # 简化：按错误类型分组
            error_type = t.get("error_type", "unknown")
            if error_type not in clusters:
                clusters[error_type] = []
            clusters[error_type].append(t)

        return [
            {"type": k, "count": len(v), "examples": v[:3]}
            for k, v in clusters.items()
        ]

    def _generate_fix(self, cluster: dict) -> dict:
        """为一个失败聚类生成修复建议"""
        return {
            "cluster_type": cluster["type"],
            "occurrence": cluster["count"],
            "suggestion": f"为 '{cluster['type']}' 类型错误添加新的 guardrail",
            "priority": "HIGH" if cluster["count"] > 10 else "MEDIUM",
        }
```

---

## 本章小结

| 概念 | 核心要点 |
|------|----------|
| 昂贵的日志 | 没有评估的 tracing 只是成本高的日志收集 |
| 三层金字塔 | 基础指标 → 语义追踪 → 评估 |
| OpenTelemetry | 厂商中立标准，LLM 语义约定覆盖模型/token/prompt |
| Prompt Version Linking | 将 trace 关联到 prompt 版本，定位回归 |
| Embedding 漂移 | 模型更新/数据变化导致检索结果悄然偏移 |
| MetaQA | 变形 prompt 检测回答一致性 |
| CLAP | 注意力分布分析推测置信度 |
| 闭环系统 | trace failures → cluster → fix harness |

---

## 动手实验

### 实验 1：为 LLM 调用添加 OpenTelemetry Tracing

**目标**：为一个简单的 LLM 调用添加完整的 OTel 插桩。

```python
# 步骤：
# 1. 安装依赖：pip install opentelemetry-api opentelemetry-sdk
# 2. 配置 TracerProvider（可使用 ConsoleSpanExporter 在终端查看）
# 3. 实现 InstrumentedLLMClient
# 4. 发起 3 次调用，观察 trace 输出
# 5. 验证 trace 包含：model, tokens, latency, prompt_version
```

**验收标准**：
- 每次调用产生一个 span
- span 包含所有 `gen_ai.*` 属性
- prompt 版本哈希可用于关联

### 实验 2：Prompt 版本回归检测

**目标**：模拟 prompt 版本变更导致的质量回归。

**步骤**：
1. 注册 prompt v1（质量好）和 v2（质量差）
2. 用 v1 生成 10 个回答，用 v2 生成 10 个回答
3. 记录 trace + quality_score
4. 调用 `detect_regression()` 检测回归
5. 验证能正确定位到 v2 是问题版本

### 实验 3：MetaQA 幻觉检测

**目标**：用 MetaQA 方法检测 LLM 回答中的幻觉。

**步骤**：
1. 准备 5 个有明确答案的问题和 5 个容易触发幻觉的问题
2. 对每个问题运行 MetaQA 检测
3. 比较一致性分数，验证幻觉问题的一致性更低

---

## 练习题

### 基础题

1. **概念题**：解释"昂贵的日志"问题。为什么仅仅记录 latency 和 token count 对 LLM 应用不够？

2. **插桩题**：列出一个 RAG 系统中需要插桩的 5 个关键点，以及每个点应该记录哪些属性。

3. **对比题**：MetaQA 和 CLAP 两种幻觉检测方法各有什么优缺点？

### 实践题

4. **漂移检测设计**：为一个客服问答系统设计 embedding 漂移检测方案，包括：探针查询选择策略、检测频率、报警阈值、修复流程。

5. **闭环实现**：设计一个闭环系统，当 MetaQA 检测到幻觉率超过 10% 时，自动触发 prompt 优化。描述完整的数据流和决策逻辑。

### 思考题

6. **观测者效应**：为 LLM 调用添加评估（如用另一个 LLM 打分）本身会增加成本和延迟。如何在"观测更多"和"开销更少"之间取平衡？采样率应该设多少？

7. **评估的评估**：如果我们用 LLM-as-Judge 来评估回答质量，那谁来评估 Judge 的质量？这个无限递归问题有实际的解决方案吗？
