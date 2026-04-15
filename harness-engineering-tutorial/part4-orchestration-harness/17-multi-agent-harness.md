# 第17章：多 Agent Harness 架构

> 一个人写代码，另一个人审代码——这不是新鲜事。当 AI agent 也遵循同样的分工原则时，系统质量会发生质变。多 Agent Harness 的本质是把"生成"与"评估"拆开，让它们互相制衡。

---

## 学习目标

学完本章，你将能够：

1. 判断何时应该从单 agent 升级到多 agent 架构
2. 实现 Anthropic 三 agent 架构（Planner + Generator + Evaluator）
3. 理解 GAN 式反馈循环在 agent 系统中的作用
4. 调节 evaluator 怀疑度与 generator 自信度之间的平衡
5. 设计专业化 agent 的通信与冲突解决协议

---

## 17.1 何时从单 Agent 转向多 Agent

### 单 Agent 的天花板

单 agent 系统存在一个根本矛盾：**同一个 agent 既生成又自评，等于让学生批改自己的试卷**。

```
单 Agent 的问题矩阵：

┌──────────────────┬────────────────────┬────────────────────┐
│      问题        │       症状          │       后果         │
├──────────────────┼────────────────────┼────────────────────┤
│ 自我偏见         │ "我生成的代码看起来  │ 错误被忽略          │
│                  │  对我来说很好"       │                    │
├──────────────────┼────────────────────┼────────────────────┤
│ 角色混淆         │ 既要创造又要批评     │ 两方面都做不好      │
├──────────────────┼────────────────────┼────────────────────┤
│ 上下文污染       │ 生成过程的中间思考   │ 影响评估的客观性    │
│                  │ 残留在评估阶段       │                    │
├──────────────────┼────────────────────┼────────────────────┤
│ 优化方向冲突     │ 同时优化创造性和     │ 模型找不到帕累托    │
│                  │ 正确性              │ 最优解              │
└──────────────────┴────────────────────┴────────────────────┘
```

### 升级判断标准

```python
def should_use_multi_agent(task_profile: dict) -> bool:
    """判断是否需要多 agent 架构"""
    signals = [
        task_profile["requires_verification"],       # 需要独立验证
        task_profile["error_cost"] > "medium",       # 错误代价高
        task_profile["output_length"] > 500,         # 输出较长
        task_profile["domain_count"] > 1,            # 涉及多领域
        task_profile["iteration_needed"],             # 需要迭代改进
    ]
    # 满足 3 个以上条件就应该考虑多 agent
    return sum(signals) >= 3
```

### 单 Agent vs 多 Agent 投入产出

| 维度 | 单 Agent | 多 Agent |
|------|----------|----------|
| 开发复杂度 | 低 | 中-高 |
| Token 消耗 | 1x | 2-5x |
| 输出质量 | 可接受 | 显著提升 |
| 错误检出率 | ~40% | ~85% |
| 适用场景 | 简单任务 | 关键任务 |
| 可维护性 | 简单 | 需要协调逻辑 |

---

## 17.2 Anthropic 三 Agent 架构：Planner + Generator + Evaluator

### 架构全景

这是 Anthropic 在 2025-2026 年推荐的核心多 agent 模式：

```
           ┌─────────────────────────────────────────┐
           │             Orchestrator                 │
           │   (调度器 / Harness 控制层)               │
           └──────────┬──────────┬──────────┬────────┘
                      │          │          │
                      ▼          ▼          ▼
              ┌──────────┐ ┌──────────┐ ┌──────────┐
              │ Planner  │ │Generator │ │Evaluator │
              │  计划者   │ │  生成者   │ │  评估者   │
              │          │ │          │ │          │
              │ 分解任务  │ │ 执行生成  │ │ 验证质量  │
              │ 制定策略  │ │ 产出代码  │ │ 给出评分  │
              │ 分配资源  │ │ 写文档    │ │ 发现缺陷  │
              └──────────┘ └──────────┘ └──────────┘
                  │              │              │
                  └──────────────┴──────────────┘
                         反馈循环 ↺
```

### 各角色职责

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

class AgentRole(Enum):
    PLANNER = "planner"
    GENERATOR = "generator"
    EVALUATOR = "evaluator"

@dataclass
class AgentConfig:
    role: AgentRole
    model: str
    system_prompt: str
    temperature: float
    max_tokens: int

# 三 agent 配置模板
PLANNER_CONFIG = AgentConfig(
    role=AgentRole.PLANNER,
    model="claude-sonnet-4-20250514",
    system_prompt="""你是一个任务规划专家。你的职责是：
    1. 将复杂任务分解为可执行步骤
    2. 为每个步骤定义明确的验收标准
    3. 识别步骤间的依赖关系
    4. 预判可能的风险和失败模式
    不要尝试执行任务，只做规划。""",
    temperature=0.3,  # 低温度 → 更严谨的规划
    max_tokens=2000,
)

GENERATOR_CONFIG = AgentConfig(
    role=AgentRole.GENERATOR,
    model="claude-sonnet-4-20250514",
    system_prompt="""你是一个代码生成专家。你的职责是：
    1. 严格按照计划执行
    2. 生成高质量、可运行的代码
    3. 添加必要的注释和文档
    4. 不跳过计划中的任何步骤
    只做生成，不做自我评估。""",
    temperature=0.4,  # 适中温度 → 平衡创造性和一致性
    max_tokens=4000,
)

EVALUATOR_CONFIG = AgentConfig(
    role=AgentRole.EVALUATOR,
    model="claude-sonnet-4-20250514",
    system_prompt="""你是一个严格的代码审查专家。你的职责是：
    1. 逐条检查是否满足验收标准
    2. 查找逻辑错误、边界条件、安全漏洞
    3. 给出 PASS / FAIL / NEEDS_REVISION 评判
    4. FAIL 时必须说明具体原因和修改建议
    保持怀疑态度，不要被"看起来正确"所欺骗。""",
    temperature=0.1,  # 极低温度 → 最大确定性
    max_tokens=2000,
)
```

### 信息流动

```
用户请求
    │
    ▼
┌─────────┐    plan.json     ┌──────────┐   code + docs   ┌──────────┐
│ Planner │ ───────────────→ │Generator │ ──────────────→ │Evaluator │
└─────────┘                  └──────────┘                  └──────────┘
                                  ▲                             │
                                  │      evaluation_report      │
                                  └─────────────────────────────┘
                                       (如果 FAIL → 重新生成)
```

---

## 17.3 GAN 式反馈循环：分离生成与评估

### 灵感来源

GAN（Generative Adversarial Network）的核心思想是：**生成器和判别器在对抗中共同进化**。在多 agent 系统中，我们借鉴这个思路：

```
传统 GAN:                      Agent GAN:
Generator → fake image          Generator → code/text
     ↕                               ↕
Discriminator → real/fake       Evaluator → pass/fail + feedback
```

### 反馈循环实现

```python
import json
from typing import Any

class GANFeedbackLoop:
    """GAN 式反馈循环：Generator 和 Evaluator 在迭代中相互提升"""

    def __init__(
        self,
        generator,      # LLM client for generation
        evaluator,       # LLM client for evaluation
        max_iterations: int = 3,
        pass_threshold: float = 0.8,
    ):
        self.generator = generator
        self.evaluator = evaluator
        self.max_iterations = max_iterations
        self.pass_threshold = pass_threshold
        self.history: list[dict] = []

    def run(self, plan: dict) -> dict:
        """执行 GAN 式迭代"""
        feedback = None

        for iteration in range(self.max_iterations):
            # --- Generator 阶段 ---
            gen_prompt = self._build_generator_prompt(plan, feedback)
            output = self.generator.generate(gen_prompt)

            # --- Evaluator 阶段 ---
            eval_prompt = self._build_evaluator_prompt(
                plan, output, iteration
            )
            evaluation = self.evaluator.evaluate(eval_prompt)

            # 记录历史
            self.history.append({
                "iteration": iteration,
                "output_preview": output[:200],
                "score": evaluation["score"],
                "verdict": evaluation["verdict"],
            })

            # 通过则退出
            if evaluation["score"] >= self.pass_threshold:
                return {
                    "output": output,
                    "iterations": iteration + 1,
                    "final_score": evaluation["score"],
                    "history": self.history,
                }

            # 未通过 → 把评估反馈给 generator
            feedback = evaluation["feedback"]

        # 达到最大迭代次数
        return {
            "output": output,
            "iterations": self.max_iterations,
            "final_score": evaluation["score"],
            "history": self.history,
            "warning": "达到最大迭代次数，输出可能不完美",
        }

    def _build_generator_prompt(
        self, plan: dict, feedback: str | None
    ) -> str:
        prompt = f"按照以下计划生成代码：\n{json.dumps(plan, ensure_ascii=False)}"
        if feedback:
            prompt += f"\n\n上一轮评审反馈（请据此改进）：\n{feedback}"
        return prompt

    def _build_evaluator_prompt(
        self, plan: dict, output: str, iteration: int
    ) -> str:
        return f"""评审以下输出（第 {iteration+1} 轮）：

验收标准：
{json.dumps(plan.get('acceptance_criteria', []), ensure_ascii=False)}

待评审内容：
{output}

请返回 JSON：
{{"score": 0.0-1.0, "verdict": "PASS|FAIL|NEEDS_REVISION", "feedback": "..."}}
"""
```

### 迭代质量提升曲线

```
质量分数
1.0 ┤                              ●─────── PASS 阈值通过
    │                         ●
0.8 ┤─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─●─ ─ ─ ─ ─ ─ PASS 阈值
    │                   ●
0.6 ┤              ●
    │         ●
0.4 ┤    ●
    │ ●
0.2 ┤
    └───┬───┬───┬───┬───┬───┬──→ 迭代次数
        1   2   3   4   5   6

典型模式：前 2-3 轮提升最大，之后收敛
```

---

## 17.4 调节 Evaluator 怀疑度 vs Generator 自夸

### 失衡的代价

这是多 agent 系统中最微妙的调参问题：

```
Evaluator 太宽松          Evaluator 太严格
    │                          │
    ▼                          ▼
低质量输出也 PASS          高质量输出也 FAIL
→ 系统形同虚设             → 无限循环烧 Token
→ 错误溜进生产环境         → 成本爆炸
→ 用户信任崩塌             → 任务永远完不成
```

### 校准策略

```python
@dataclass
class CalibrationConfig:
    """Evaluator 校准配置"""

    # 怀疑度：0.0 = 完全信任, 1.0 = 极度怀疑
    skepticism: float = 0.6

    # 各维度权重
    weights: dict = field(default_factory=lambda: {
        "correctness": 0.35,    # 正确性
        "completeness": 0.25,   # 完整性
        "code_quality": 0.20,   # 代码质量
        "security": 0.20,       # 安全性
    })

    # 分数映射：根据怀疑度调整阈值
    @property
    def pass_threshold(self) -> float:
        # 怀疑度 0.6 → 阈值 0.8
        return 0.5 + self.skepticism * 0.5

    # 最大重试次数：怀疑度越高允许越多重试
    @property
    def max_retries(self) -> int:
        return max(2, int(self.skepticism * 5))


def build_evaluator_system_prompt(config: CalibrationConfig) -> str:
    """根据校准配置生成 evaluator 的 system prompt"""
    skepticism_desc = {
        (0.0, 0.3): "你是一个友善的审查者，主要看大方向是否正确",
        (0.3, 0.6): "你是一个认真的审查者，会检查常见错误",
        (0.6, 0.8): "你是一个严格的审查者，会仔细检查每个细节",
        (0.8, 1.0): "你是一个极其严格的审查者，假设代码有隐藏的 bug",
    }
    for (lo, hi), desc in skepticism_desc.items():
        if lo <= config.skepticism < hi:
            personality = desc
            break
    else:
        personality = skepticism_desc[(0.6, 0.8)]

    return f"""{personality}

评分权重：
- 正确性: {config.weights['correctness']:.0%}
- 完整性: {config.weights['completeness']:.0%}
- 代码质量: {config.weights['code_quality']:.0%}
- 安全性: {config.weights['security']:.0%}

通过阈值: {config.pass_threshold:.2f}
"""
```

### Generator 反校准

Generator 也会"学到"如何应对严格 evaluator——这就是 **generator 自夸**（self-promotion）问题：

```python
# 反自夸策略：在 generator prompt 中加入反制措施
ANTI_SELF_PROMOTION = """
重要规则：
1. 不要在代码注释中声称"经过全面测试"——evaluator 会验证
2. 不要生成看起来正确但实际跳过边界检查的代码
3. 不要用 try/except: pass 来隐藏错误
4. 如果某个需求你无法完全实现，请明确标注 TODO 而不是假装完成
"""
```

---

## 17.5 专业化 Agent：Testing、QA、Cleanup

### 超越三角色

在实际系统中，三 agent 架构可以扩展为多个专业化 agent：

```
                    ┌──────────┐
                    │ Planner  │
                    └────┬─────┘
                         │ plan
            ┌────────────┼────────────┐
            ▼            ▼            ▼
      ┌──────────┐ ┌──────────┐ ┌──────────┐
      │Generator │ │ Testing  │ │ Cleanup  │
      │ (编码)   │ │ Agent    │ │ Agent    │
      │          │ │ (测试)   │ │ (清理)   │
      └────┬─────┘ └────┬─────┘ └────┬─────┘
           │             │            │
           ▼             ▼            ▼
      ┌──────────┐ ┌──────────┐ ┌──────────┐
      │   QA     │ │ Security │ │ Docs     │
      │ Agent    │ │ Agent    │ │ Agent    │
      │ (质检)   │ │ (安全)   │ │ (文档)   │
      └────┬─────┘ └────┬─────┘ └────┬─────┘
           └─────────────┼────────────┘
                         ▼
                    ┌──────────┐
                    │Evaluator │
                    │ (终审)   │
                    └──────────┘
```

### 专业化 Agent 配置表

| Agent | 职责 | 模型选择 | Temperature | 关键 Prompt 指令 |
|-------|------|----------|-------------|------------------|
| Testing Agent | 生成测试用例 | Sonnet | 0.4 | "覆盖正常/异常/边界" |
| QA Agent | 集成验证 | Sonnet | 0.2 | "模拟用户操作流" |
| Cleanup Agent | 代码清理 | Haiku | 0.1 | "移除 dead code，统一风格" |
| Security Agent | 安全审查 | Opus | 0.1 | "检查 OWASP Top 10" |
| Docs Agent | 生成文档 | Sonnet | 0.5 | "面向使用者，不面向开发者" |

### 专业化 Agent 实现模式

```python
class SpecializedAgent:
    """专业化 agent 基类"""

    def __init__(self, name: str, config: AgentConfig):
        self.name = name
        self.config = config
        self.input_schema: dict = {}   # 期望的输入格式
        self.output_schema: dict = {}  # 承诺的输出格式

    def validate_input(self, data: dict) -> bool:
        """验证输入是否符合 schema"""
        # 使用 jsonschema 或 pydantic 验证
        return True  # 简化

    def execute(self, data: dict) -> dict:
        """执行专业任务并返回结构化结果"""
        if not self.validate_input(data):
            return {"error": "输入不符合 schema"}

        result = self._call_llm(data)
        return self._format_output(result)

    def _call_llm(self, data: dict) -> str:
        """调用 LLM（子类可覆盖）"""
        raise NotImplementedError

    def _format_output(self, raw: str) -> dict:
        """格式化输出（子类可覆盖）"""
        raise NotImplementedError


class TestingAgent(SpecializedAgent):
    """测试生成 Agent"""

    def __init__(self):
        super().__init__("testing", GENERATOR_CONFIG)
        self.input_schema = {
            "required": ["source_code", "language", "framework"],
        }
        self.output_schema = {
            "required": ["test_code", "coverage_targets", "test_count"],
        }

    def _format_output(self, raw: str) -> dict:
        return {
            "test_code": raw,
            "coverage_targets": self._extract_coverage(raw),
            "test_count": raw.count("def test_"),
        }

    def _extract_coverage(self, code: str) -> list[str]:
        """从生成的测试代码中提取覆盖目标"""
        targets = []
        for line in code.split("\n"):
            if line.strip().startswith("def test_"):
                targets.append(line.strip())
        return targets
```

---

## 17.6 Sub-Agent 协调：通信、输出合并、冲突解决

### 通信模式

```
模式 A: 星型（Hub-and-Spoke）        模式 B: 流水线（Pipeline）
                                     
      Agent1                         Agent1 → Agent2 → Agent3
        ↑                                 │          │
  Agent4←Hub→Agent2                       ▼          ▼
        ↓                             artifact1  artifact2
      Agent3                         

模式 C: 黑板（Blackboard）
                                     
  ┌─────────────────────────┐        
  │     共享黑板 (Artifact)  │        
  │  ┌───┐ ┌───┐ ┌───┐     │        
  │  │ A │ │ B │ │ C │     │        
  │  └───┘ └───┘ └───┘     │        
  └─────────────────────────┘        
  Agent1 写 A                        
  Agent2 读 A，写 B                   
  Agent3 读 A+B，写 C                 
```

### 输出合并策略

```python
class OutputMerger:
    """多 agent 输出合并器"""

    @staticmethod
    def merge_sequential(outputs: list[dict]) -> dict:
        """流水线模式：后面 agent 的输出覆盖前面的"""
        merged = {}
        for output in outputs:
            merged.update(output)
        return merged

    @staticmethod
    def merge_by_domain(outputs: dict[str, dict]) -> dict:
        """领域合并：每个 agent 负责自己的领域"""
        return {
            "code": outputs.get("generator", {}).get("code", ""),
            "tests": outputs.get("testing", {}).get("test_code", ""),
            "docs": outputs.get("docs", {}).get("documentation", ""),
            "security_report": outputs.get("security", {}).get("report", ""),
        }

    @staticmethod
    def merge_with_conflict_resolution(
        outputs: list[dict],
        resolver: "ConflictResolver",
    ) -> dict:
        """冲突感知合并"""
        merged = {}
        conflicts = []
        for output in outputs:
            for key, value in output.items():
                if key in merged and merged[key] != value:
                    conflicts.append({
                        "key": key,
                        "existing": merged[key],
                        "incoming": value,
                    })
                else:
                    merged[key] = value

        # 解决冲突
        for conflict in conflicts:
            resolved = resolver.resolve(conflict)
            merged[conflict["key"]] = resolved
        return merged


class ConflictResolver:
    """冲突解决器"""

    def __init__(self, strategy: str = "evaluator_decides"):
        self.strategy = strategy

    def resolve(self, conflict: dict) -> Any:
        if self.strategy == "evaluator_decides":
            # 交给 evaluator agent 裁决
            return self._ask_evaluator(conflict)
        elif self.strategy == "newest_wins":
            return conflict["incoming"]
        elif self.strategy == "priority_based":
            return self._priority_resolution(conflict)
        return conflict["existing"]

    def _ask_evaluator(self, conflict: dict) -> Any:
        """让 evaluator 裁决冲突"""
        # 调用 evaluator LLM，传入两个版本
        prompt = f"""两个 agent 对 '{conflict["key"]}' 产生了冲突：
版本A: {conflict['existing']}
版本B: {conflict['incoming']}
请选择更优版本并说明原因。"""
        # ... 调用 LLM
        return conflict["existing"]  # 简化

    def _priority_resolution(self, conflict: dict) -> Any:
        """基于优先级解决"""
        return conflict["existing"]  # 简化
```

---

## 本章小结

| 概念 | 核心要点 |
|------|----------|
| 升级时机 | 需验证 + 高错误成本 + 多领域 → 多 agent |
| 三 Agent 架构 | Planner 分解、Generator 执行、Evaluator 评审 |
| GAN 式循环 | 生成与评估分离，迭代提升质量 |
| 怀疑度校准 | 太松则放行错误，太严则烧钱，需动态调节 |
| 专业化 Agent | Testing / QA / Cleanup / Security / Docs |
| 通信模式 | 星型、流水线、黑板——按任务选择 |
| 冲突解决 | Evaluator 裁决 / 最新优先 / 优先级规则 |

---

## 动手实验

### 实验 1：实现 Planner + Generator + Evaluator 三 Agent 系统

**目标**：构建一个三 agent 系统来完成"实现一个带输入验证的用户注册 API"。

```python
# 实验框架（使用 mock LLM 或真实 API）
class ThreeAgentSystem:
    def __init__(self):
        self.planner = PlannerAgent()
        self.generator = GeneratorAgent()
        self.evaluator = EvaluatorAgent()

    def execute(self, task: str) -> dict:
        # Step 1: Planner 分解任务
        plan = self.planner.plan(task)
        print(f"计划：{len(plan['steps'])} 个步骤")

        # Step 2-3: Generator-Evaluator 循环
        loop = GANFeedbackLoop(
            self.generator, self.evaluator,
            max_iterations=3, pass_threshold=0.8,
        )
        result = loop.run(plan)

        print(f"迭代 {result['iterations']} 轮，最终分数 {result['final_score']}")
        return result

# 运行
system = ThreeAgentSystem()
result = system.execute("实现一个带输入验证的用户注册 REST API")
```

**验收标准**：
- Planner 产出至少 3 个步骤
- Generator-Evaluator 至少迭代 2 轮
- 最终分数 >= 0.8

### 实验 2：怀疑度校准实验

**目标**：用同一个任务在不同怀疑度下运行，观察迭代次数和最终质量。

```python
results = {}
for skepticism in [0.2, 0.5, 0.8]:
    config = CalibrationConfig(skepticism=skepticism)
    system = ThreeAgentSystem(evaluator_config=config)
    result = system.execute("实现一个 LRU 缓存")
    results[skepticism] = {
        "iterations": result["iterations"],
        "score": result["final_score"],
        "total_tokens": result.get("total_tokens", 0),
    }

# 对比分析
for s, r in results.items():
    print(f"怀疑度={s}: 迭代={r['iterations']}, 分数={r['score']}")
```

### 实验 3：专业化 Agent 流水线

**目标**：构建 Generator → Testing Agent → QA Agent → Evaluator 流水线。

**步骤**：
1. Generator 生成一个 Python 函数
2. Testing Agent 为该函数生成测试
3. QA Agent 执行测试并报告覆盖率
4. Evaluator 基于所有结果给出终审

---

## 练习题

### 基础题

1. **概念题**：解释 Planner、Generator、Evaluator 三个角色的职责边界。为什么不能让一个 agent 同时承担多个角色？

2. **配置题**：一个电商系统的订单处理 agent，正确性权重应该设为多少？为什么安全性权重不能低于 0.15？

3. **模式选择**：星型通信 vs 流水线通信，各自适合什么场景？列举各两个例子。

### 实践题

4. **冲突解决**：Generator 生成了一段使用递归的代码，Security Agent 认为可能栈溢出，Cleanup Agent 认为代码简洁。设计一个冲突解决流程。

5. **成本优化**：一个三 agent 系统平均迭代 4 次才能通过，每次消耗约 5000 tokens。请设计一个策略将平均迭代次数降到 2 次以内，同时不降低质量。

### 思考题

6. **Agent 串通问题**：如果 Generator 和 Evaluator 使用同一个底层模型，是否会出现"串通"（collusion）现象——即 Evaluator 对 Generator 的输出系统性偏好？如何检测和防范？

7. **N Agent 问题**：随着专业化 agent 数量增加（>5），协调成本会如何增长？是否存在一个最优的 agent 数量？请从通信复杂度角度分析。
