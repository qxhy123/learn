# 第7章：lm-evaluation-harness 架构剖析

> EleutherAI 的 lm-evaluation-harness 是评估框架领域的"Linux"——它不是最易用的，但它是最通用、最可扩展、社区贡献最活跃的。超过 400 个 benchmark、支持十余种模型后端、被 Hugging Face Open LLM Leaderboard 直接采用。理解它的架构，等于理解评估 harness 的设计范式。本章将逐层拆解这个项目的核心设计，并带你在自定义 benchmark 上完成一次完整评估。

---

## 学习目标

学完本章，你将能够：

1. 解释 lm-evaluation-harness 为什么能成为评估框架的事实标准
2. 描述 Registry 模式在评估框架中的作用，以及延迟加载的好处
3. 编写 YAML 格式的声明式任务配置（ConfigurableTask）
4. 理解 Filter Chains 如何对模型原始输出进行后处理
5. 为自定义 benchmark 编写完整的 YAML 配置并用命令行运行评估

---

## 7.1 为什么 lm-eval-harness 是标杆

### 7.1.1 历史定位

```
2021: EleutherAI 发布 lm-evaluation-harness（当时叫 lm-eval）
      目标：为 GPT-Neo 系列模型提供标准化评估
2022: 成为 Hugging Face Open LLM Leaderboard 的后端引擎
2023: 支持的 benchmark 超过 200 个
2024: v0.4.x 大重构，引入 ConfigurableTask 和 YAML 配置
2025: 成为事实标准，几乎所有开源模型发布都引用它的结果
```

### 7.1.2 核心优势

| 优势 | 具体表现 |
|------|---------|
| 标准化 | 相同任务 + 相同配置 = 可比较的结果 |
| 覆盖广 | 400+ benchmarks（MMLU、ARC、HellaSwag、GSM8K 等） |
| 后端多 | HuggingFace、vLLM、OpenAI、Anthropic、GGUF 等 |
| 可扩展 | 通过 Registry 模式，任何人可以注册新 task/model/filter |
| 社区活跃 | 2000+ GitHub stars，频繁更新 |

### 7.1.3 何时用、何时不用

| 场景 | 是否用 lm-eval-harness | 理由 |
|------|----------------------|------|
| 模型选型对比 | 是 | 标准 benchmark 直接可用 |
| 学术论文复现 | 是 | 社区共识的评估方式 |
| 产品级系统评估 | 否 | 不支持 RAG/Agent 管线 |
| 内部领域评估 | 看情况 | 可以写自定义 task，但可能过重 |
| 快速 demo | 否 | 启动成本较高 |

---

## 7.2 Registry 模式：通过字符串注册一切

### 7.2.1 什么是 Registry 模式

Registry 模式是 lm-eval-harness 的架构基石。核心思想：**一切组件都通过字符串名称注册和查找，实现延迟加载和松耦合**。

```python
# lm-eval-harness 的 Registry 核心思想（简化）
class Registry:
    """全局注册表：组件名 → 组件类/工厂"""
    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """装饰器：注册组件"""
        def decorator(component_cls):
            cls._registry[name] = component_cls
            return component_cls
        return decorator

    @classmethod
    def get(cls, name: str):
        """通过名称查找组件（延迟加载）"""
        if name not in cls._registry:
            raise ValueError(f"Unknown component: {name}")
        return cls._registry[name]
```

### 7.2.2 三大 Registry

lm-eval-harness 有三个核心 Registry：

```
┌──────────────────────────────────────────────┐
│               Registry 架构                   │
├──────────────┬───────────────┬────────────────┤
│  ModelRegistry │ TaskRegistry  │ FilterRegistry │
├──────────────┼───────────────┼────────────────┤
│ "hf"         │ "mmlu"        │ "take_first"   │
│ "vllm"       │ "arc_easy"    │ "regex"        │
│ "openai"     │ "gsm8k"       │ "lowercase"    │
│ "anthropic"  │ "hellaswag"   │ "strip"        │
│ "gguf"       │ "custom_task" │ "custom_fn"    │
└──────────────┴───────────────┴────────────────┘
```

### 7.2.3 延迟加载的好处

```python
# 不用 Registry：启动时加载所有 400+ tasks
import all_tasks  # 慢！占内存！

# 用 Registry：只在需要时加载
task = TaskRegistry.get("mmlu")  # 只加载 mmlu
```

**实际影响**：
- 启动时间从数十秒降到 < 1 秒
- 内存占用减少 90%+
- 新增 task 不影响已有代码（开放/封闭原则）

### 7.2.4 Registry 模式在实际代码中的体现

```python
# 在 lm-eval-harness 中注册一个模型后端（简化示意）
@ModelRegistry.register("vllm")
class VLLMModel(BaseLM):
    def __init__(self, model_name: str, **kwargs):
        self.model = load_vllm_model(model_name)

    def generate(self, prompts: list[str], **kwargs) -> list[str]:
        return self.model.generate(prompts, **kwargs)

    def loglikelihood(self, requests: list) -> list[float]:
        return self.model.compute_loglikelihoods(requests)

# 使用时通过字符串查找
model = ModelRegistry.get("vllm")(model_name="meta-llama/Llama-3-8B")
```

---

## 7.3 声明式 YAML 任务配置

### 7.3.1 ConfigurableTask：从代码到配置

在 v0.4.x 之前，添加新 task 需要写 Python 类。重构后，绝大多数 task 可以用纯 YAML 定义：

```yaml
# 一个完整的 YAML 任务配置示例
task: my_qa_benchmark
dataset_path: json
dataset_kwargs:
  data_files: ./data/my_qa.jsonl
output_type: generate_until
doc_to_text: "问题: {{question}}\n答案:"
doc_to_target: "{{answer}}"
generation_kwargs:
  max_gen_toks: 128
  temperature: 0
  until:
    - "\n"
    - "问题:"
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
  - metric: f1
    aggregation: mean
    higher_is_better: true
num_fewshot: 3
```

### 7.3.2 YAML 配置关键字段

| 字段 | 作用 | 示例 |
|------|------|------|
| `task` | 任务名（注册到 TaskRegistry） | `my_qa_benchmark` |
| `dataset_path` | 数据来源（HF dataset 或本地） | `json`, `csv`, `huggingface/dataset_name` |
| `output_type` | 评估方式 | `generate_until`, `loglikelihood`, `multiple_choice` |
| `doc_to_text` | 输入模板（Jinja2） | `"问题: {{question}}\n答案:"` |
| `doc_to_target` | 目标输出模板 | `"{{answer}}"` |
| `metric_list` | 评估指标列表 | `exact_match`, `perplexity`, `bleu` |
| `num_fewshot` | few-shot 示例数 | `0`, `3`, `5` |
| `generation_kwargs` | 生成参数 | `max_gen_toks`, `temperature` |

### 7.3.3 output_type 详解

这是任务配置中最重要的选择——它决定了如何与模型交互：

| output_type | 模型交互方式 | 典型 benchmark | 指标 |
|-------------|------------|---------------|------|
| `generate_until` | 生成文本直到停止符 | GSM8K、HumanEval | exact_match、BLEU |
| `loglikelihood` | 计算候选答案的对数概率 | HellaSwag、PIQA | accuracy |
| `multiple_choice` | loglikelihood 的特化形式 | MMLU、ARC | accuracy |

```yaml
# loglikelihood 示例（多选题）
task: my_multiple_choice
output_type: multiple_choice
doc_to_text: "{{question}}\n"
doc_to_choice:
  - "{{choice_a}}"
  - "{{choice_b}}"
  - "{{choice_c}}"
  - "{{choice_d}}"
doc_to_target: "{{answer_index}}"
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
  - metric: acc_norm
    aggregation: mean
    higher_is_better: true
```

---

## 7.4 统一模型接口

### 7.4.1 BaseLM 抽象

lm-eval-harness 通过统一接口支持多种模型后端。任何模型只需实现两个核心方法：

```python
class BaseLM:
    """所有模型后端的基类（简化）"""

    def generate_until(
        self, requests: list[dict]
    ) -> list[str]:
        """
        生成文本直到满足停止条件
        用于 generate_until 类型的任务
        """
        raise NotImplementedError

    def loglikelihood(
        self, requests: list[tuple[str, str]]
    ) -> list[tuple[float, bool]]:
        """
        计算 (context, continuation) 对的对数概率
        用于 loglikelihood 和 multiple_choice 类型的任务
        Returns: [(log_prob, is_greedy), ...]
        """
        raise NotImplementedError
```

### 7.4.2 已支持的模型后端

| 后端 | 注册名 | 适用场景 | 安装要求 |
|------|--------|---------|---------|
| HuggingFace | `hf` | 本地 GPU 推理 | `transformers` |
| vLLM | `vllm` | 高吞吐本地推理 | `vllm` |
| OpenAI API | `openai` | GPT 系列 | `openai` |
| Anthropic | `anthropic` | Claude 系列 | `anthropic` |
| GGUF | `gguf` | CPU/低端 GPU | `llama-cpp-python` |
| TGI | `textsynth` | Text Generation Inference | HTTP client |
| Local Completions | `local-completions` | 自定义 API | HTTP client |

### 7.4.3 命令行中的模型指定

```bash
# HuggingFace 模型（本地 GPU）
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct \
    --tasks mmlu \
    --batch_size 8

# vLLM 后端（高吞吐）
lm_eval --model vllm \
    --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,tensor_parallel_size=2 \
    --tasks mmlu \
    --batch_size auto

# OpenAI API
lm_eval --model openai-completions \
    --model_args model=gpt-4o-mini \
    --tasks mmlu \
    --batch_size 4
```

---

## 7.5 Filter Chains：输出后处理

### 7.5.1 为什么需要 Filter

模型的原始输出往往不能直接用于指标计算。例如：

```
问题：2 + 3 = ?
模型输出："Let me think... 2 plus 3 equals 5. So the answer is 5."
期望输出："5"
```

如果直接做 exact_match，结果为 0。Filter 的作用是从原始输出中提取可评估的部分。

### 7.5.2 内置 Filter 列表

| Filter 名 | 作用 | 示例 |
|-----------|------|------|
| `take_first` | 取第一个生成结果 | 多个候选取第一个 |
| `regex` | 正则提取 | 从长文本提取数字 |
| `strip` | 去除首尾空白 | `" 42 "` → `"42"` |
| `lowercase` | 转小写 | `"Paris"` → `"paris"` |
| `map` | 映射转换 | `"A"` → `0` |
| `take_first_k` | 取前 k 个 token | 截断长输出 |

### 7.5.3 Filter Chain 配置

Filter 可以链式组合，按顺序执行：

```yaml
task: math_word_problems
output_type: generate_until
doc_to_text: "问题: {{question}}\n逐步推理后给出最终答案。\n"
doc_to_target: "{{answer}}"
generation_kwargs:
  max_gen_toks: 512
  temperature: 0
  until:
    - "\n\n"
filter_list:
  - name: "get_answer"
    filter:
      - function: "regex"
        regex_pattern: "(?:答案|answer)(?:是|:|：|=)\\s*(.+?)(?:\\.|。|$)"
        group_select: 1
      - function: "strip"
      - function: "take_first"
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
```

**处理流程**：

```
模型原始输出: "先分析题目条件...所以答案是 42。"
    │
    ▼  regex: 提取 "答案是" 后面的内容
    "42。"
    │
    ▼  strip: 去除空白
    "42。"
    │
    ▼  take_first: 取第一个结果
    "42。"
    │
    ▼  exact_match vs "42"
    结果: 0（因为多了"。"）
```

这个例子也说明了 filter 设计需要仔细——你可能需要更精确的正则来处理标点。

### 7.5.4 自定义 Filter

```python
from lm_eval.api.filter import Filter

@FilterRegistry.register("extract_number")
class ExtractNumberFilter(Filter):
    """提取输出中的最后一个数字"""

    def apply(self, resps: list[str], docs: list[dict]) -> list[str]:
        import re
        results = []
        for resp in resps:
            numbers = re.findall(r'-?\d+\.?\d*', resp)
            results.append(numbers[-1] if numbers else "")
        return results
```

---

## 7.6 Template 继承与扩展

### 7.6.1 Template 继承机制

许多 benchmark 有多个子任务（如 MMLU 有 57 个学科）。逐个配置太冗余。Template 继承解决这个问题：

```yaml
# _default.yaml - 基础模板
group: my_benchmark
output_type: multiple_choice
doc_to_text: "{{question}}\n"
doc_to_choice: ["{{A}}", "{{B}}", "{{C}}", "{{D}}"]
doc_to_target: "{{answer}}"
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
num_fewshot: 5

---
# math_subset.yaml - 继承并覆盖
include: _default.yaml
task: my_benchmark_math
dataset_path: my_org/my_benchmark
dataset_name: math_subset
num_fewshot: 3  # 覆盖：数学题用更少的 few-shot

---
# history_subset.yaml - 继承并覆盖
include: _default.yaml
task: my_benchmark_history
dataset_path: my_org/my_benchmark
dataset_name: history_subset
# num_fewshot 继承默认值 5
```

### 7.6.2 Task Group

可以将多个 task 组织成一个 group，一起运行：

```yaml
# my_benchmark.yaml
group: my_benchmark
task:
  - my_benchmark_math
  - my_benchmark_history
  - my_benchmark_science
aggregate_metric_list:
  - metric: acc
    aggregation: mean
    weight_by_size: true  # 按子任务样本数加权
```

```bash
# 运行整个 group
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct \
    --tasks my_benchmark \
    --output_path results/
```

### 7.6.3 自定义扩展点汇总

| 扩展点 | 方式 | 复杂度 |
|--------|------|--------|
| 新 Task | YAML 配置 | 低 |
| 新 Metric | Python 函数 + 注册 | 中 |
| 新 Filter | Python 类 + 注册 | 中 |
| 新 Model Backend | 继承 BaseLM + 注册 | 高 |
| Task Group | YAML 配置 | 低 |

---

## 本章小结

| 概念 | 要点 |
|------|------|
| lm-eval-harness 定位 | 开源评估框架的事实标准，400+ benchmarks，10+ 模型后端 |
| Registry 模式 | 一切通过字符串注册和查找，延迟加载，松耦合 |
| ConfigurableTask | 声明式 YAML 任务配置，消除大部分 Python 模板代码 |
| output_type | 三种模式：generate_until、loglikelihood、multiple_choice |
| 统一模型接口 | BaseLM 抽象，两个核心方法：generate_until 和 loglikelihood |
| Filter Chains | 链式后处理，从原始输出提取可评估内容（regex、strip 等） |
| Template 继承 | include 关键字实现配置复用，减少重复 |
| Task Group | 多个 task 组织成 group，支持加权聚合 |

---

## 动手实验

### 实验 1：用 lm-eval-harness 评估模型

**目标**：安装 lm-eval-harness，在标准 benchmark 上评估一个小模型。

**步骤**：

1. 安装：

```bash
pip install lm-eval[vllm]
# 或者从源码安装（获取最新 task）
# git clone https://github.com/EleutherAI/lm-evaluation-harness.git
# cd lm-evaluation-harness && pip install -e ".[vllm]"
```

2. 运行标准 benchmark：

```bash
# 用较小的模型和 benchmark 快速验证
lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen2.5-0.5B-Instruct,dtype=float16 \
    --tasks hellaswag,arc_easy \
    --batch_size 16 \
    --output_path ./eval_results/ \
    --log_samples
```

3. 查看结果：

```bash
# 结果保存在 ./eval_results/ 下的 JSON 文件
cat ./eval_results/results_*.json | python -m json.tool
```

4. 关注输出中的关键字段：

```json
{
  "results": {
    "hellaswag": {
      "acc": 0.3245,
      "acc_norm": 0.4012,
      "acc_stderr": 0.0048
    },
    "arc_easy": {
      "acc": 0.5821,
      "acc_norm": 0.5534,
      "acc_stderr": 0.0101
    }
  }
}
```

### 实验 2：创建自定义 Benchmark

**目标**：为中文问答场景编写 YAML 配置，用 lm-eval-harness 运行。

1. 准备数据文件 `my_chinese_qa.jsonl`：

```jsonl
{"question": "中国最长的河流是什么？", "answer": "长江"}
{"question": "光的速度大约是多少？", "answer": "每秒30万公里"}
{"question": "水的化学式是什么？", "answer": "H2O"}
{"question": "地球距离太阳大约多远？", "answer": "1.5亿公里"}
{"question": "人体最大的器官是什么？", "answer": "皮肤"}
```

2. 编写任务配置 `my_chinese_qa.yaml`：

```yaml
task: my_chinese_qa
dataset_path: json
dataset_kwargs:
  data_files: ./my_chinese_qa.jsonl
output_type: generate_until
doc_to_text: "请用简短的中文回答以下问题。\n\n问题：{{question}}\n答案："
doc_to_target: "{{answer}}"
generation_kwargs:
  max_gen_toks: 64
  temperature: 0
  until:
    - "\n"
    - "问题："
filter_list:
  - name: "clean_answer"
    filter:
      - function: "strip"
      - function: "take_first"
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
num_fewshot: 0
metadata:
  version: 1.0
  description: "中文常识问答自定义 benchmark"
```

3. 运行自定义任务：

```bash
lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen2.5-0.5B-Instruct,dtype=float16 \
    --tasks my_chinese_qa \
    --include_path ./ \
    --batch_size 4 \
    --output_path ./custom_results/ \
    --log_samples
```

4. 分析：查看每个样本的预测结果（`--log_samples` 会记录），找出哪些问题答错了，思考 filter 是否需要调整。

---

## 练习题

### 基础题

1. **Registry 理解**：解释为什么 lm-eval-harness 使用 Registry 模式而不是直接 import。如果有 400 个 task 但你只想跑 1 个，Registry 带来的性能收益是什么？

2. **output_type 选择**：对于以下场景，应该选择哪种 output_type？
   - (a) 给模型一个数学题，让它算出答案
   - (b) 给模型一个句子，让它从 4 个选项中选最合理的续写
   - (c) 测量模型对不同文本的 perplexity

3. **Filter 设计**：模型对数学题的回答格式不固定：有时是 `"答案是42"`，有时是 `"The answer is 42."`，有时是 `"42"`。请设计一个 filter chain（用 YAML 格式）来稳健地提取数字答案。

### 实践题

1. **多模型对比**：用 lm-eval-harness 在同一个 benchmark（如 ARC Easy）上评估两个模型，对比结果。提交包含两个模型分数的对比表格。

2. **自定义 Filter**：实现一个 Python 自定义 filter，功能是：从模型输出的 JSON 字符串中提取某个指定 key 的值。将它注册到 FilterRegistry 并在 YAML 中使用。

### 思考题

1. lm-eval-harness 主要面向模型评估。如果你要将它的设计理念（Registry、YAML 配置、Filter Chain）应用到系统评估（如 RAG 管线评估），你会如何修改架构？哪些组件可以复用，哪些需要重新设计？

2. 学术 benchmark（如 MMLU）和产品级评估之间存在"benchmark gap"。一个模型在 MMLU 上 90 分，但在你的产品中表现平平。这个 gap 的根本原因是什么？如何缩小它？
