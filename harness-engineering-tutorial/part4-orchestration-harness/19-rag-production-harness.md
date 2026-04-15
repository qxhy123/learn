# 第19章：RAG 生产系统 Harness

> RAG 在 2024 年是"检索增强生成"，到 2026 年已经演化为完整的 **Context Engine**——不仅检索文档，还要组装上下文、验证输出、路由查询。把 RAG 做到生产级，靠的不是更好的 embedding 模型，而是更好的 Harness。

---

## 学习目标

学完本章，你将能够：

1. 理解 RAG 从"检索增强"到 "Context Engine" 的 2026 演进
2. 设计检索 harness：embedding 管线、向量库、reranking
3. 设计生成 harness：prompt 组装、guardrails、输出验证
4. 实现级联模式降低成本
5. 利用 Golden Examples 实现动态 few-shot 注入

---

## 19.1 RAG 作为 "Context Engine"（2026 演进）

### 从 RAG 到 Context Engine

```
2023 RAG:
  query → embed → 向量搜索 → top-k docs → 塞进 prompt → LLM 回答

2026 Context Engine:
  query → 意图分类 → 路由 → 多源检索 → rerank → 压缩 →
  prompt 组装 → guardrails → LLM 回答 → 输出验证 → 引用检查
```

### 架构对比

```
2023 简单 RAG                    2026 Context Engine

┌───────┐                        ┌────────────┐
│ Query │                        │   Query    │
└───┬───┘                        └─────┬──────┘
    │                                  │
    ▼                            ┌─────▼──────┐
┌───────┐                        │ Intent     │
│Embed +│                        │ Router     │
│Search │                        └──┬──┬──┬───┘
└───┬───┘                           │  │  │
    │                          ┌────┘  │  └────┐
    ▼                          ▼       ▼       ▼
┌───────┐                  ┌──────┐┌──────┐┌──────┐
│  LLM  │                  │Vector││Graph ││ SQL  │
└───┬───┘                  │  DB  ││  DB  ││  DB  │
    │                      └──┬───┘└──┬───┘└──┬───┘
    ▼                         └───┬───┘───┬───┘
┌───────┐                        ▼        │
│Answer │                  ┌──────────┐   │
└───────┘                  │ Reranker │   │
                           └────┬─────┘   │
                                ▼         │
                           ┌──────────┐   │
                           │ Context  │←──┘
                           │ Assembler│
                           └────┬─────┘
                                ▼
                           ┌──────────┐
                           │Guardrails│
                           └────┬─────┘
                                ▼
                           ┌──────────┐
                           │   LLM    │
                           └────┬─────┘
                                ▼
                           ┌──────────┐
                           │ Output   │
                           │Validator │
                           └────┬─────┘
                                ▼
                           ┌──────────┐
                           │ Answer + │
                           │Citations │
                           └──────────┘
```

### Context Engine 的 Harness 层

| 阶段 | Harness 职责 | 关键指标 |
|------|-------------|----------|
| 检索 | 管理 embedding、索引更新、多源融合 | Recall@K, Latency |
| 重排 | 二次排序、去重、多样性保证 | NDCG, MRR |
| 组装 | 上下文窗口分配、压缩、结构化 | Context utilization |
| 生成 | Prompt 模板、guardrails、温度控制 | Answer quality |
| 验证 | 引用检查、幻觉检测、一致性校验 | Faithfulness score |

---

## 19.2 检索 Harness：Embedding 管线、向量库管理、Reranking

### Embedding 管线

```python
from dataclasses import dataclass
from typing import Optional
import hashlib

@dataclass
class Document:
    id: str
    content: str
    metadata: dict
    embedding: Optional[list[float]] = None
    chunk_id: Optional[str] = None

class EmbeddingPipeline:
    """Embedding 管线 Harness"""

    def __init__(self, embedder, vector_store, chunk_size: int = 512):
        self.embedder = embedder
        self.vector_store = vector_store
        self.chunk_size = chunk_size

    def ingest(self, documents: list[Document]) -> dict:
        """完整的文档摄入流水线"""
        stats = {"total": len(documents), "chunked": 0, "embedded": 0, "stored": 0}

        # Step 1: 分块
        chunks = []
        for doc in documents:
            doc_chunks = self._chunk_document(doc)
            chunks.extend(doc_chunks)
            stats["chunked"] += len(doc_chunks)

        # Step 2: 去重（基于内容哈希）
        unique_chunks = self._deduplicate(chunks)

        # Step 3: Embedding
        for chunk in unique_chunks:
            chunk.embedding = self.embedder.embed(chunk.content)
            stats["embedded"] += 1

        # Step 4: 存入向量库
        self.vector_store.upsert(unique_chunks)
        stats["stored"] = len(unique_chunks)

        return stats

    def _chunk_document(self, doc: Document) -> list[Document]:
        """智能分块：按段落边界切分，保留上下文重叠"""
        paragraphs = doc.content.split("\n\n")
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) > self.chunk_size:
                if current_chunk:
                    chunk_id = hashlib.md5(current_chunk.encode()).hexdigest()[:12]
                    chunks.append(Document(
                        id=f"{doc.id}_{chunk_id}",
                        content=current_chunk,
                        metadata={**doc.metadata, "parent_id": doc.id},
                        chunk_id=chunk_id,
                    ))
                current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para

        if current_chunk:
            chunk_id = hashlib.md5(current_chunk.encode()).hexdigest()[:12]
            chunks.append(Document(
                id=f"{doc.id}_{chunk_id}",
                content=current_chunk,
                metadata={**doc.metadata, "parent_id": doc.id},
                chunk_id=chunk_id,
            ))
        return chunks

    def _deduplicate(self, chunks: list[Document]) -> list[Document]:
        """基于内容哈希去重"""
        seen = set()
        unique = []
        for chunk in chunks:
            h = hashlib.md5(chunk.content.encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                unique.append(chunk)
        return unique
```

### Reranking Harness

```python
class RerankingHarness:
    """重排序 Harness：二次排序提升精度"""

    def __init__(self, reranker, diversity_weight: float = 0.3):
        self.reranker = reranker
        self.diversity_weight = diversity_weight

    def rerank(
        self,
        query: str,
        candidates: list[Document],
        top_k: int = 5,
    ) -> list[Document]:
        """重排序 + 多样性保证"""
        # Step 1: 用 cross-encoder 精排
        scores = self.reranker.score(query, [c.content for c in candidates])
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        # Step 2: MMR（Maximal Marginal Relevance）多样性
        selected = []
        remaining = [s for s in scored]

        while len(selected) < top_k and remaining:
            if not selected:
                selected.append(remaining.pop(0))
                continue

            # 计算 MMR 分数
            best_idx, best_mmr = 0, -float("inf")
            for i, (doc, score) in enumerate(remaining):
                similarity_to_selected = max(
                    self._cosine_sim(doc.embedding, s[0].embedding)
                    for s in selected
                )
                mmr = (
                    (1 - self.diversity_weight) * score
                    - self.diversity_weight * similarity_to_selected
                )
                if mmr > best_mmr:
                    best_idx, best_mmr = i, mmr

            selected.append(remaining.pop(best_idx))

        return [doc for doc, _ in selected]

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        """余弦相似度"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x**2 for x in a) ** 0.5
        norm_b = sum(x**2 for x in b) ** 0.5
        return dot / (norm_a * norm_b + 1e-8)
```

---

## 19.3 生成 Harness：Prompt 组装、Guardrails、输出验证

### Prompt 组装器

```python
class PromptAssembler:
    """Prompt 组装 Harness：智能分配上下文窗口"""

    def __init__(self, max_context_tokens: int = 8000):
        self.max_context_tokens = max_context_tokens

    def assemble(
        self,
        query: str,
        retrieved_docs: list[Document],
        system_prompt: str,
        few_shot_examples: list[dict] | None = None,
    ) -> list[dict]:
        """组装完整 prompt"""
        messages = []

        # 1. System prompt（固定开销）
        messages.append({"role": "system", "content": system_prompt})

        # 2. 预算分配
        budget = self.max_context_tokens
        budget -= self._estimate_tokens(system_prompt)
        budget -= self._estimate_tokens(query)
        budget -= 500  # 留给输出

        # 3. Few-shot examples（如果有）
        if few_shot_examples:
            example_budget = min(budget // 3, 2000)
            examples_text = self._format_examples(
                few_shot_examples, example_budget
            )
            messages.append({
                "role": "system",
                "content": f"参考示例：\n{examples_text}",
            })
            budget -= self._estimate_tokens(examples_text)

        # 4. Retrieved context（占用剩余预算）
        context_text = self._format_context(retrieved_docs, budget)
        messages.append({
            "role": "user",
            "content": f"参考资料：\n{context_text}\n\n问题：{query}",
        })

        return messages

    def _estimate_tokens(self, text: str) -> int:
        """粗略估算 token 数"""
        return len(text) // 3  # 中文大约 1.5 字/token

    def _format_context(
        self, docs: list[Document], budget: int
    ) -> str:
        """格式化检索上下文，在预算内尽量多放"""
        context_parts = []
        used_tokens = 0

        for i, doc in enumerate(docs):
            doc_text = f"[文档 {i+1}] {doc.content}"
            doc_tokens = self._estimate_tokens(doc_text)
            if used_tokens + doc_tokens > budget:
                break
            context_parts.append(doc_text)
            used_tokens += doc_tokens

        return "\n\n".join(context_parts)

    def _format_examples(
        self, examples: list[dict], budget: int
    ) -> str:
        """格式化 few-shot 示例"""
        parts = []
        used = 0
        for ex in examples:
            text = f"Q: {ex['question']}\nA: {ex['answer']}"
            tokens = self._estimate_tokens(text)
            if used + tokens > budget:
                break
            parts.append(text)
            used += tokens
        return "\n---\n".join(parts)
```

### 输出验证 Harness

```python
class OutputValidator:
    """输出验证 Harness"""

    def __init__(self, llm_client=None):
        self.llm = llm_client
        self.checks = [
            self._check_citation,
            self._check_hallucination_signals,
            self._check_completeness,
            self._check_safety,
        ]

    def validate(
        self,
        query: str,
        answer: str,
        source_docs: list[Document],
    ) -> dict:
        """运行全部验证检查"""
        results = {}
        for check in self.checks:
            name = check.__name__.replace("_check_", "")
            results[name] = check(query, answer, source_docs)

        # 综合判定
        all_pass = all(r["pass"] for r in results.values())
        return {
            "pass": all_pass,
            "checks": results,
            "confidence": sum(
                r.get("score", 1.0) for r in results.values()
            ) / len(results),
        }

    def _check_citation(
        self, query: str, answer: str, docs: list[Document]
    ) -> dict:
        """检查答案是否有来源支持"""
        # 简单实现：检查答案中的关键声明是否能在文档中找到
        answer_sentences = [s.strip() for s in answer.split("。") if s.strip()]
        supported = 0
        for sentence in answer_sentences:
            for doc in docs:
                # 简化：检查关键词重叠
                overlap = len(
                    set(sentence) & set(doc.content)
                ) / max(len(set(sentence)), 1)
                if overlap > 0.3:
                    supported += 1
                    break

        ratio = supported / max(len(answer_sentences), 1)
        return {"pass": ratio > 0.6, "score": ratio, "detail": f"{supported}/{len(answer_sentences)} 句有来源"}

    def _check_hallucination_signals(
        self, query: str, answer: str, docs: list[Document]
    ) -> dict:
        """检测幻觉信号"""
        signals = {
            "hedge_words": ["可能", "大概", "也许", "据说"],
            "overconfidence": ["绝对", "一定", "毫无疑问", "100%"],
            "fabrication": ["根据最新研究表明", "众所周知"],
        }
        found = []
        for category, words in signals.items():
            for word in words:
                if word in answer:
                    found.append(f"{category}: '{word}'")

        return {
            "pass": len(found) < 3,
            "score": max(0, 1 - len(found) * 0.2),
            "detail": found if found else "无幻觉信号",
        }

    def _check_completeness(
        self, query: str, answer: str, docs: list[Document]
    ) -> dict:
        """检查答案是否完整回答了问题"""
        # 简化：检查答案长度是否合理
        is_complete = len(answer) > 50
        return {"pass": is_complete, "score": min(len(answer) / 200, 1.0)}

    def _check_safety(
        self, query: str, answer: str, docs: list[Document]
    ) -> dict:
        """安全检查"""
        unsafe_patterns = ["密码是", "信用卡号", "身份证号"]
        found = [p for p in unsafe_patterns if p in answer]
        return {"pass": len(found) == 0, "score": 1.0 if not found else 0.0}
```

---

## 19.4 级联模式：便宜模型先跑，升级到贵模型

### 为什么需要级联

```
每个查询都用 Opus？
→ 80% 的查询用 Haiku 就够了
→ 白白浪费 20x 的成本

级联策略：
┌───────┐   简单查询(60%)   ┌───────┐
│       │──────────────────→│ Haiku │→ 回答
│       │                   └───────┘
│ Query │   中等查询(30%)   ┌────────┐
│Router │──────────────────→│ Sonnet │→ 回答
│       │                   └────────┘
│       │   复杂查询(10%)   ┌───────┐
│       │──────────────────→│ Opus  │→ 回答
└───────┘                   └───────┘

平均成本 = 0.6 × $0.001 + 0.3 × $0.01 + 0.1 × $0.05
         = $0.0086（比全部用 Opus 的 $0.05 便宜 83%）
```

### 级联实现

```python
class CascadeRouter:
    """级联路由器：根据查询复杂度选择模型"""

    def __init__(self, models: dict[str, dict]):
        """
        models = {
            "haiku": {"client": haiku_client, "cost_per_1k": 0.001},
            "sonnet": {"client": sonnet_client, "cost_per_1k": 0.01},
            "opus": {"client": opus_client, "cost_per_1k": 0.05},
        }
        """
        self.models = models

    def route(self, query: str, context: list[Document]) -> str:
        """决定使用哪个模型"""
        complexity = self._assess_complexity(query, context)

        if complexity < 0.3:
            return "haiku"
        elif complexity < 0.7:
            return "sonnet"
        else:
            return "opus"

    def execute_with_cascade(
        self,
        query: str,
        context: list[Document],
        quality_threshold: float = 0.7,
    ) -> dict:
        """级联执行：低级模型不够好时自动升级"""
        model_order = ["haiku", "sonnet", "opus"]

        for model_name in model_order:
            model = self.models[model_name]
            answer = model["client"].generate(query, context)

            # 快速质量评估
            quality = self._quick_quality_check(query, answer, context)

            if quality >= quality_threshold:
                return {
                    "answer": answer,
                    "model_used": model_name,
                    "quality_score": quality,
                    "cascaded": model_name != model_order[0],
                }

        # 最贵的模型也用了，返回最后结果
        return {
            "answer": answer,
            "model_used": "opus",
            "quality_score": quality,
            "cascaded": True,
        }

    def _assess_complexity(
        self, query: str, context: list[Document]
    ) -> float:
        """评估查询复杂度 0-1"""
        score = 0.0
        # 长查询更复杂
        if len(query) > 100:
            score += 0.2
        # 多个问号表示多个子问题
        score += min(query.count("？") * 0.15, 0.3)
        # 上下文多表示需要综合推理
        if len(context) > 5:
            score += 0.2
        # 包含对比/分析关键词
        complex_keywords = ["对比", "分析", "为什么", "区别", "优劣"]
        for kw in complex_keywords:
            if kw in query:
                score += 0.1
        return min(score, 1.0)

    def _quick_quality_check(
        self, query: str, answer: str, context: list[Document]
    ) -> float:
        """快速质量检查（不用 LLM，纯规则）"""
        score = 0.5
        # 答案太短扣分
        if len(answer) < 30:
            score -= 0.3
        # 答案包含"我不确定"之类扣分
        uncertainty = ["不确定", "不知道", "无法判断"]
        for u in uncertainty:
            if u in answer:
                score -= 0.2
        # 答案与上下文有关联加分
        for doc in context:
            overlap = len(set(answer) & set(doc.content)) / max(len(set(answer)), 1)
            if overlap > 0.2:
                score += 0.1
                break
        return max(0.0, min(1.0, score))
```

---

## 19.5 Golden Examples via Vector DB：动态 Few-Shot 注入

### 理念

Few-shot examples 不应该硬编码在 prompt 里——它们应该**从向量库中动态检索**，找到与当前查询最相似的历史问答对：

```
传统 few-shot:                    动态 few-shot:
┌─────────────────┐               ┌─────────────────┐
│ 固定 3 个示例    │               │ query → embed   │
│ - 示例1         │               │      ↓          │
│ - 示例2         │               │ Golden Examples │
│ - 示例3         │               │ Vector DB 检索  │
│                 │               │      ↓          │
│ 对所有查询      │               │ 最相似的 3 个   │
│ 示例都一样      │               │ 历史问答对      │
└─────────────────┘               └─────────────────┘
```

### 实现

```python
class GoldenExampleStore:
    """Golden Examples 管理：维护和检索高质量问答对"""

    def __init__(self, vector_store, embedder):
        self.vector_store = vector_store
        self.embedder = embedder
        self.collection = "golden_examples"

    def add_example(
        self,
        question: str,
        answer: str,
        metadata: dict | None = None,
    ) -> str:
        """添加一个 golden example"""
        embedding = self.embedder.embed(question)
        doc_id = hashlib.md5(question.encode()).hexdigest()[:12]

        self.vector_store.upsert([{
            "id": doc_id,
            "embedding": embedding,
            "content": question,
            "metadata": {
                "question": question,
                "answer": answer,
                "quality_score": metadata.get("quality_score", 1.0) if metadata else 1.0,
                "domain": metadata.get("domain", "general") if metadata else "general",
                "created_at": datetime.now().isoformat(),
                **(metadata or {}),
            },
        }])
        return doc_id

    def retrieve_examples(
        self,
        query: str,
        top_k: int = 3,
        min_quality: float = 0.8,
    ) -> list[dict]:
        """检索最相似的 golden examples"""
        query_embedding = self.embedder.embed(query)
        results = self.vector_store.search(
            query_embedding, top_k=top_k * 2,  # 多检索一些以便过滤
            collection=self.collection,
        )

        # 过滤低质量
        filtered = [
            r for r in results
            if r["metadata"].get("quality_score", 0) >= min_quality
        ]

        return [{
            "question": r["metadata"]["question"],
            "answer": r["metadata"]["answer"],
            "similarity": r["score"],
        } for r in filtered[:top_k]]

    def build_few_shot_prompt(
        self, query: str, top_k: int = 3
    ) -> str:
        """构建动态 few-shot prompt"""
        examples = self.retrieve_examples(query, top_k)
        if not examples:
            return ""

        prompt = "以下是类似问题的参考答案：\n\n"
        for i, ex in enumerate(examples, 1):
            prompt += f"示例 {i}:\n"
            prompt += f"问：{ex['question']}\n"
            prompt += f"答：{ex['answer']}\n\n"

        return prompt
```

---

## 19.6 路由模式：把查询导向专业化子系统

### 路由架构

```
                    ┌─────────────┐
                    │ Query Input │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Router    │
                    │  (分类器)   │
                    └──┬──┬──┬───┘
                       │  │  │
          ┌────────────┘  │  └────────────┐
          ▼               ▼               ▼
   ┌────────────┐  ┌────────────┐  ┌────────────┐
   │ 技术文档   │  │  产品FAQ   │  │  财务报表  │
   │ 子系统     │  │  子系统    │  │  子系统    │
   │            │  │            │  │            │
   │ 专业embedding│ │ 轻量检索   │  │ SQL + 表格 │
   │ + reranker  │ │ + 模板回答  │  │ + 计算引擎 │
   └──────┬─────┘  └──────┬─────┘  └──────┬─────┘
          │               │               │
          └───────────────┼───────────────┘
                          ▼
                   ┌────────────┐
                   │   Answer   │
                   │ Aggregator │
                   └────────────┘
```

### 路由器实现

```python
class QueryRouter:
    """查询路由器：将查询导向最合适的子系统"""

    def __init__(self, classifier, subsystems: dict):
        """
        subsystems = {
            "technical": TechnicalRAG(),
            "faq": FAQSystem(),
            "financial": FinancialSystem(),
        }
        """
        self.classifier = classifier
        self.subsystems = subsystems

    def route(self, query: str) -> dict:
        """路由查询到对应子系统"""
        # Step 1: 分类
        classification = self.classifier.classify(query)
        target = classification["category"]
        confidence = classification["confidence"]

        # Step 2: 低置信度时多路查询
        if confidence < 0.6:
            return self._multi_route(query, classification["top_categories"])

        # Step 3: 高置信度时单路查询
        if target not in self.subsystems:
            target = "general"

        subsystem = self.subsystems[target]
        answer = subsystem.query(query)

        return {
            "answer": answer,
            "routed_to": target,
            "confidence": confidence,
        }

    def _multi_route(
        self, query: str, candidates: list[str]
    ) -> dict:
        """低置信度：查询多个子系统，合并结果"""
        answers = {}
        for cat in candidates[:2]:  # 最多查 2 个
            if cat in self.subsystems:
                answers[cat] = self.subsystems[cat].query(query)

        # 合并答案
        merged = self._merge_answers(query, answers)
        return {
            "answer": merged,
            "routed_to": candidates,
            "confidence": "multi-route",
        }

    def _merge_answers(
        self, query: str, answers: dict[str, str]
    ) -> str:
        """合并多个子系统的答案"""
        if len(answers) == 1:
            return list(answers.values())[0]

        parts = []
        for source, answer in answers.items():
            parts.append(f"[来源: {source}] {answer}")
        return "\n\n".join(parts)
```

---

## 本章小结

| 概念 | 核心要点 |
|------|----------|
| Context Engine | 2026 RAG 不只是检索，是完整的上下文引擎 |
| Embedding 管线 | 分块 → 去重 → embedding → 向量库 |
| Reranking | Cross-encoder 精排 + MMR 多样性 |
| Prompt 组装 | 预算分配：system > examples > context > query |
| 输出验证 | 引用检查 + 幻觉检测 + 完整性 + 安全性 |
| 级联模式 | Haiku → Sonnet → Opus，成本降 83% |
| Golden Examples | 向量库存储高质量问答对，动态 few-shot |
| 路由模式 | 按查询意图导向专业子系统 |

---

## 动手实验

### 实验 1：构建一个带检索 + 生成 + 验证的 RAG Harness

**目标**：端到端实现 Context Engine。

```python
# 实验步骤：
# 1. 准备 10 篇文档（可用 Wikipedia 摘要）
# 2. 实现 EmbeddingPipeline 进行摄入
# 3. 实现检索 + reranking
# 4. 实现 PromptAssembler 组装上下文
# 5. 调用 LLM 生成答案
# 6. 用 OutputValidator 验证答案

class RAGHarness:
    def __init__(self):
        self.pipeline = EmbeddingPipeline(embedder, vector_store)
        self.reranker = RerankingHarness(reranker)
        self.assembler = PromptAssembler()
        self.validator = OutputValidator()

    def query(self, question: str) -> dict:
        candidates = self.vector_store.search(question, top_k=20)
        reranked = self.reranker.rerank(question, candidates, top_k=5)
        messages = self.assembler.assemble(question, reranked, SYSTEM_PROMPT)
        answer = self.llm.generate(messages)
        validation = self.validator.validate(question, answer, reranked)
        return {"answer": answer, "validation": validation}
```

**验收标准**：
- 检索返回相关文档
- 答案通过至少 3/4 项验证检查
- 全流程延迟 < 5 秒

### 实验 2：级联成本优化

**目标**：对比全部用 Opus vs 级联策略的成本差异。

**步骤**：
1. 准备 50 个查询（简单 30 + 中等 15 + 复杂 5）
2. 用 CascadeRouter 处理所有查询
3. 记录每个查询使用的模型和质量分数
4. 计算总成本对比

### 实验 3：Golden Examples 动态注入

**目标**：比较固定 few-shot vs 动态 few-shot 的答案质量。

**步骤**：
1. 构建 30 个 golden examples 存入向量库
2. 准备 10 个测试查询
3. 方案 A：固定 3 个 few-shot examples
4. 方案 B：动态检索最相似的 3 个 examples
5. 对比两组答案的质量分数

---

## 练习题

### 基础题

1. **概念题**：解释 "Context Engine" 与传统 RAG 的 3 个主要区别。

2. **计算题**：上下文窗口 8000 tokens，system prompt 占 1000，query 占 200，输出预留 500。如果每个文档块约 300 tokens，最多能放多少个文档块？

3. **设计题**：为一个"公司内部知识库"设计 reranking 策略。哪些信号应该用来排序？

### 实践题

4. **级联优化**：一个系统 70% 查询是"什么是X"类简单问题，20% 是"对比X和Y"，10% 是"为什么X导致Y"。设计级联策略和路由规则。

5. **Golden Examples 维护**：Golden Examples 会过时（产品更新后旧答案不再正确）。设计一个过期检测和更新机制。

### 思考题

6. **Reranking 悖论**：如果 reranker 比 embedding search 更准确，为什么不直接用 reranker 做全量搜索？从计算复杂度角度分析。

7. **Context Window 经济学**：上下文窗口越大是否越好？讨论"塞更多文档"vs"塞更精准的文档"的权衡，以及对成本和质量的影响。
