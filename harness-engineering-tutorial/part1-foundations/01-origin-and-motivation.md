# 第1章：Harness Engineering 的起源与动机

> 当 88% 的 AI agent 项目无法上线时，问题不在模型——而在于模型周围缺少的一切。本章追溯 Harness Engineering 从萌芽到成型的关键历程，帮助你理解这门学科为什么在 2026 年成为 AI 工程的核心。

---

## 学习目标

学完本章，你将能够：

1. 解释为什么单靠大语言模型无法构建可靠的 AI agent
2. 描述模型商品化趋势及其对工程实践的影响
3. 梳理从 prompt engineering 到 context engineering 再到 harness engineering 的演进脉络
4. 列举 Harness Engineering 发展中的四个关键里程碑
5. 通过实验直观感受 harness 对 agent 可靠性的决定性作用

---

## 1.1 模型本身为什么不够

### 一个令人不安的数据

2025 年底，多家研究机构的调研报告指出：企业 AI agent 项目中，约 **88%** 无法从原型阶段推进到生产环境。这个数字令人震惊——大语言模型在基准测试上不断刷新纪录，为什么真实项目却频频失败？

### Demo 陷阱

问题的根源在于 **"demo 到 production 的鸿沟"**。一个典型的 AI agent demo 看起来像这样：

```python
# demo.py —— 看起来很美好
response = llm.chat("帮我分析这段代码的 bug 并修复")
print(response)
```

在演示环境中，这段代码可能表现完美。但一旦面对真实世界：

```
┌─────────────────────────────────────────────────┐
│              Demo vs Production                  │
├──────────────────┬──────────────────────────────┤
│   Demo 环境      │    Production 环境            │
├──────────────────┼──────────────────────────────┤
│ 输入可控         │ 输入千变万化                  │
│ 单次调用         │ 需要重试和降级策略             │
│ 忽略错误         │ 必须处理超时、限流、幻觉       │
│ 无需验证         │ 输出必须经过校验               │
│ 人工观察         │ 需要自动化监控                 │
│ 成本不敏感       │ 每次调用都是真金白银            │
│ 无安全顾虑       │ prompt injection 是真实威胁    │
└──────────────────┴──────────────────────────────┘
```

### 模型的五个固有局限

无论模型多强大，它始终面临以下固有局限：

1. **非确定性输出**：相同的 prompt，两次调用可能返回完全不同的结果
2. **幻觉问题**：模型会自信地编造不存在的 API、错误的事实
3. **上下文窗口有限**：即使是百万 token 的窗口，面对大型代码库也捉襟见肘
4. **无状态性**：模型不记得上一次交互，每次调用都是全新开始
5. **无法自我验证**：模型无法可靠地判断自己的输出是否正确

这些不是"等下一代模型就能解决"的问题——它们是语言模型这一技术路线的结构性特征。

---

## 1.2 模型商品化：竞争力的迁移

### 能力趋同

2025-2026 年，一个显著趋势是头部模型的能力快速趋同：

```
模型能力趋同示意图（2024-2026）

能力
 ▲
 │         ╭── Claude Opus
 │        ╱ ╭── GPT-5
 │       ╱ ╱ ╭── Gemini Ultra
 │      ╱ ╱ ╱
 │     ╱ ╱ ╱     ← 差距越来越小
 │    ╱ ╱ ╱
 │   ╱ ╱ ╱
 │  ╱ ╱ ╱
 │ ╱ ╱ ╱
 │╱ ╱ ╱
 ┼─────────────────────────► 时间
 2024     2025     2026
```

Claude、GPT、Gemini 在大多数任务上的表现差距已经缩小到工程误差范围内。这意味着什么？

### 竞争力迁移

当模型本身不再是差异化因素时，**竞争力从"选择哪个模型"迁移到"如何使用模型"**。

| 竞争力来源 | 2023 年 | 2024 年 | 2026 年 |
|------------|---------|---------|---------|
| 模型选择 | 核心 | 重要 | 次要 |
| Prompt 技巧 | 重要 | 核心 | 基础 |
| 上下文工程 | 萌芽 | 重要 | 核心 |
| Harness 工程 | 无 | 萌芽 | 核心 |

这与计算机硬件的历史如出一辙：当 CPU 性能差距缩小后，操作系统和软件生态成为竞争焦点。同理，当模型趋同后，围绕模型的工程体系成为决定 AI 产品成败的关键。

---

## 1.3 三代演进：从 Prompt 到 Context 到 Harness

### 第一代：Prompt Engineering（2023-2024）

Prompt Engineering 关注的是"如何写好一段指令"：

```python
# Prompt Engineering 时代
prompt = """你是一个资深 Python 开发者。
请按照以下要求分析代码：
1. 找出所有 bug
2. 按严重程度排序
3. 给出修复建议

代码：
{code}
"""
response = llm.chat(prompt)
```

它的核心假设是：**只要 prompt 写得足够好，模型就能给出正确答案。**

这个假设在简单任务上成立，但随着任务复杂度增加迅速失效。

### 第二代：Context Engineering（2025）

Context Engineering 的洞察是：模型的表现取决于它看到的上下文质量。

```python
# Context Engineering 时代
context = retrieve_relevant_docs(query)      # 检索相关文档
examples = find_similar_cases(query)          # 找到相似案例
schema = load_output_schema("bug_report")     # 加载输出格式

prompt = build_prompt(
    system=system_instruction,
    context=context,
    examples=examples,
    schema=schema,
    user_query=query
)
response = llm.chat(prompt)
```

Context Engineering 把注意力从"怎么写 prompt"转移到"怎么组装上下文"。这是重要的进步，但它仍然只关注 **输入端**——模型调用之前发生的事情。

### 第三代：Harness Engineering（2026）

Harness Engineering 的核心主张是：**可靠的 AI agent 需要一个完整的运行时系统，不只是好的输入，还需要输出验证、错误恢复、状态管理、可观测性。**

```python
# Harness Engineering 时代
class CodeAnalysisHarness:
    def __init__(self, model, config):
        self.model = model
        self.retry_policy = config.retry_policy
        self.validators = config.validators
        self.sensors = config.sensors
        self.logger = config.logger

    def analyze(self, code: str) -> BugReport:
        context = self.build_context(code)        # Context Engineering
        prompt = self.build_prompt(context)         # Prompt Engineering

        for attempt in range(self.retry_policy.max_retries):
            response = self.model.call(prompt)      # 模型调用
            report = self.parse(response)           # 结构化解析

            if self.validate(report):               # 输出验证
                self.logger.log_success(attempt)    # 可观测性
                return report

            prompt = self.refine(prompt, report)    # 反馈修正
            self.logger.log_retry(attempt, report)

        return self.fallback(code)                  # 降级策略
```

三代演进的关系不是替代，而是包含：

```
┌─────────────────────────────────────┐
│  Harness Engineering                │
│  ┌─────────────────────────────┐    │
│  │  Context Engineering        │    │
│  │  ┌─────────────────────┐    │    │
│  │  │ Prompt Engineering  │    │    │
│  │  └─────────────────────┘    │    │
│  └─────────────────────────────┘    │
│  + 验证 + 重试 + 监控 + 编排 + ... │
└─────────────────────────────────────┘
```

---

## 1.4 关键里程碑

### 里程碑一览

```
时间线：Harness Engineering 的诞生

2025.Q3          2026.02          2026.03          2026.04
   │                │                │                │
   ▼                ▼                ▼                ▼
 Hashimoto        OpenAI           Anthropic        Fowler
 首提 Harness     百万行代码       多 Agent &       Guides/
 公式             实验报告         长时运行架构      Sensors
```

### Mitchell Hashimoto：命名与公式化（2025 Q3）

Mitchell Hashimoto——HashiCorp 的联合创始人，以 Terraform 和 Vagrant 闻名——在深度参与 AI 编码工具开发后，提出了那个简洁有力的公式：

> **Agent = Model + Harness**

他指出，大多数人把注意力放在等号右边的第一项（Model），但真正决定 agent 能否工作的是第二项（Harness）。这个命名精准地捕捉到了一个关键洞察：模型需要被"驾驭"（harness 的字面含义），就像马力需要缰绳。

### OpenAI 百万行代码实验（2026.02）

OpenAI 在 2026 年 2 月发表了一项里程碑式的实验报告，展示了 AI agent 在百万行级别代码库上执行复杂工程任务的能力。这项研究的关键发现不在模型本身，而在于：

- **Harness 设计比模型选择更影响成功率**
- 合理的工具调用编排将任务完成率从 34% 提升到 78%
- 结构化的验证反馈循环是处理大规模代码库的关键

### Anthropic 多 Agent 架构（2026.03）

Anthropic 发表了两篇重要的架构文档：

1. **多 Agent Harness 架构**：如何将复杂任务分解给多个专门化 agent，通过 harness 协调它们的工作
2. **长时运行 Agent Harness**：如何构建能够持续运行数小时甚至数天的 agent 系统，包括检查点、恢复和渐进式输出

### Martin Fowler 的分类框架（2026.04）

Martin Fowler 从控制论的角度提出了 harness 组件的分类方法：

- **Guides（前馈控制）**：在模型执行前注入的约束和指导
- **Sensors（反馈控制）**：在模型执行后进行的检测和校正

这个分类框架为 Harness Engineering 提供了严谨的理论基础，我们将在第 3 章详细展开。

---

## 1.5 Harness Engineering 的定义

综合以上演进，我们可以给出 Harness Engineering 的工作定义：

> **Harness Engineering** 是设计、构建和运维围绕大语言模型的运行时系统的工程学科。它关注如何通过前馈控制（Guides）和反馈控制（Sensors）使 AI agent 在真实生产环境中可靠、可控、可观测地运行。

核心关注点包括：

| 关注领域 | 关键问题 |
|----------|---------|
| 可靠性 | 如何让 agent 在面对不确定性时依然稳定工作？ |
| 可控性 | 如何确保 agent 的行为在预期范围内？ |
| 可观测性 | 如何知道 agent 在做什么、做得怎么样？ |
| 可扩展性 | 如何编排多个 agent 协作完成复杂任务？ |
| 经济性 | 如何在保证质量的前提下控制 token 消耗？ |
| 安全性 | 如何防御 prompt injection 等攻击？ |

---

## 本章小结

| 概念 | 要点 |
|------|------|
| 88% 失败率 | AI agent 项目的瓶颈不在模型，在模型周围的工程系统 |
| 模型商品化 | Claude/GPT/Gemini 能力趋同，竞争力迁移到工程层 |
| 三代演进 | Prompt → Context → Harness，层层包含而非替代 |
| 核心公式 | Agent = Model + Harness（Hashimoto, 2025） |
| Guides/Sensors | Fowler 的前馈/反馈控制分类（2026.04） |
| 学科定义 | 设计、构建和运维围绕 LLM 的运行时系统的工程学科 |

---

## 动手实验

### 实验 1：无 Harness 的 Agent

下面这段代码模拟了一个没有 harness 的 agent——直接调用模型并信任其输出：

```python
"""
实验 1：无 Harness 的 Agent
目标：观察直接调用模型的不可靠性
"""
import random
import json

# 模拟一个有 30% 概率出错的 LLM
def unreliable_llm(prompt: str) -> str:
    """模拟模型的非确定性：相同输入可能返回错误格式、幻觉或正确结果"""
    roll = random.random()
    if roll < 0.15:
        return "抱歉，我无法处理这个请求。"  # 拒绝回答
    elif roll < 0.30:
        return '{"bugs": [{"line": 999, "desc": "不存在的 bug"}]}'  # 幻觉
    else:
        return '{"bugs": [{"line": 42, "severity": "high", "desc": "未处理的 None 值"}]}'

# 无 harness 的 agent
def naive_agent(code: str) -> dict:
    response = unreliable_llm(f"分析代码 bug：{code}")
    return json.loads(response)  # 可能 JSON 解析失败！

# 运行 20 次，观察成功率
successes = 0
failures = 0
for i in range(20):
    try:
        result = naive_agent("def foo(x): return x.strip()")
        if result.get("bugs") and result["bugs"][0].get("severity"):
            successes += 1
        else:
            failures += 1
    except Exception as e:
        failures += 1
        print(f"  运行 {i+1}: 失败 - {e}")

print(f"\n无 Harness: {successes}/20 成功 ({successes/20*100:.0f}%)")
```

### 实验 2：有 Harness 的 Agent

现在给同一个 agent 加上最基本的 harness：

```python
"""
实验 2：有 Harness 的 Agent
目标：观察 harness（重试 + 验证）如何提升可靠性
"""
import random
import json

def unreliable_llm(prompt: str) -> str:
    """与实验 1 完全相同的不可靠 LLM"""
    roll = random.random()
    if roll < 0.15:
        return "抱歉，我无法处理这个请求。"
    elif roll < 0.30:
        return '{"bugs": [{"line": 999, "desc": "不存在的 bug"}]}'
    else:
        return '{"bugs": [{"line": 42, "severity": "high", "desc": "未处理的 None 值"}]}'

# 验证函数（Sensor）
def validate_bug_report(report: dict) -> bool:
    """检查输出格式是否正确、内容是否合理"""
    if not isinstance(report.get("bugs"), list):
        return False
    for bug in report["bugs"]:
        if "severity" not in bug:
            return False  # 必须包含严重程度
        if bug.get("line", 0) > 500:
            return False  # 我们的文件不到 500 行，这是幻觉
    return True

# 有 harness 的 agent
def harnessed_agent(code: str, max_retries: int = 5) -> dict:
    for attempt in range(max_retries):
        try:
            response = unreliable_llm(f"分析代码 bug：{code}")
            result = json.loads(response)

            if validate_bug_report(result):
                return result  # 验证通过，返回结果

        except json.JSONDecodeError:
            pass  # JSON 解析失败，重试

    # 所有重试用完，返回安全的降级结果
    return {"bugs": [], "note": "分析未能完成，请人工检查"}

# 运行 20 次，对比成功率
successes = 0
for i in range(20):
    result = harnessed_agent("def foo(x): return x.strip()")
    if result.get("bugs") and len(result["bugs"]) > 0:
        if result["bugs"][0].get("severity"):
            successes += 1

print(f"有 Harness: {successes}/20 成功 ({successes/20*100:.0f}%)")
print("（对比实验 1 的无 Harness 结果）")
```

运行两个实验多次，你会看到：无 harness 的成功率通常在 60-75%，有 harness 的成功率通常在 90-100%。这就是 harness 的价值——它不改变模型，而是在模型周围构建可靠性。

---

## 练习题

### 基础题

1. 用自己的话解释 "Agent = Model + Harness" 这个公式的含义。
2. 列举模型的五个固有局限中，你认为对生产环境影响最大的两个，并解释原因。
3. Prompt Engineering、Context Engineering 和 Harness Engineering 三者是什么关系？画出它们的包含关系。

### 实践题

4. 修改实验 2 的代码，添加一个 `logging` 模块来记录每次重试的原因（JSON 解析失败 / 验证不通过 / 幻觉检测）。统计 100 次运行中各类失败的比例。
5. 在实验 2 的基础上，添加一个"指数退避"重试策略（第 n 次重试等待 2^n 秒），并思考在真实 API 调用场景下这为什么重要。

### 思考题

6. "等模型足够强大，就不需要 harness 了"——你同意这个观点吗？用操作系统的类比来论证你的立场。
7. 如果你是一家创业公司的 CTO，你会把工程资源优先投入到选择更好的模型，还是构建更好的 harness？在什么条件下你的答案会改变？
