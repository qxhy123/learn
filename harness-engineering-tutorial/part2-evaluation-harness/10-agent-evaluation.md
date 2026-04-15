# 第10章：Agent 系统评估 Harness

> 评估一个聊天机器人是否回答正确，难度等级是"中等"。评估一个 Agent 系统是否正确执行了多步任务，难度等级是"噩梦"。Agent 会调用工具、做决策、处理错误、甚至自我修正——它的行为空间是组合爆炸的。只看最终结果不够，因为同一个正确结果可以来自正确的路径或错误的路径。本章深入 Agent 评估的核心挑战：如何评估 trajectory（行为轨迹），如何用 recorded tapes 实现可复现测试，以及如何用混沌工程发现 Agent 的脆弱点。

---

## 学习目标

学完本章，你将能够：

1. 解释为什么 Agent 评估比单轮 LLM 评估根本性地更难
2. 设计 trajectory 评估方案，将行为轨迹作为正确性的主要评估单元
3. 使用 recorded tool tapes 实现确定性、可复现的 Agent 测试
4. 构建端到端场景测试和步级评估
5. 运用混沌工程（故障注入）测试 Agent 的鲁棒性和停机安全性

---

## 10.1 为什么 Agent 评估根本性地更难

### 10.1.1 单轮评估 vs Agent 评估

| 维度 | 单轮 LLM 评估 | Agent 系统评估 |
|------|--------------|---------------|
| 交互次数 | 1 次 | N 次（不确定） |
| 行为空间 | 有限（一次回答） | 组合爆炸（工具调用序列） |
| 正确性定义 | 输出匹配参考答案 | 最终结果正确 + 路径合理 |
| 外部依赖 | 无或极少 | 多个工具/API |
| 确定性 | 容易控制 | 极难控制（工具返回值变化） |
| 评估成本 | 低（一次 API 调用） | 高（多次调用 + 工具调用） |
| 失败模式 | 回答错误 | 无限循环、工具误用、错误传播、过度调用 |

### 10.1.2 Agent 的独特失败模式

```
传统 LLM 失败：                  Agent 独有失败：
├── 回答不准确                   ├── 无限循环（反复调用同一工具）
├── 幻觉                        ├── 工具序列错误（先写后读）
├── 拒绝回答                    ├── 工具参数构造错误
└── 偏见                        ├── 错误传播（A步骤的错误导致后续全错）
                                ├── 过度调用（能一步解决的用了十步）
                                ├── 遗漏步骤（跳过关键验证）
                                ├── 死锁（等待永远不会来的响应）
                                └── 成本爆炸（单次任务消耗过多 token）
```

### 10.1.3 评估复杂度的数学直觉

```
单轮评估：
  行为空间 ≈ |vocabulary|^max_tokens
  但评估只看最终输出 → 维度可控

Agent 评估：
  行为空间 ≈ (|tools| × |parameters|)^max_steps
  每一步的决策影响后续所有步骤 → 维度爆炸

  例：5 个工具，每个 3 种参数组合，最多 10 步
  行为空间 = 15^10 ≈ 5.7 × 10^11
```

---

## 10.2 Trajectory 是正确性的主要单元

### 10.2.1 什么是 Trajectory

Trajectory（行为轨迹）是 Agent 从接收任务到完成任务的完整行为序列：

```python
from dataclasses import dataclass, field
from enum import Enum

class StepType(Enum):
    THINK = "think"             # 思考/推理
    TOOL_CALL = "tool_call"     # 调用工具
    TOOL_RESULT = "tool_result" # 工具返回结果
    ANSWER = "answer"           # 最终回答
    ERROR = "error"             # 错误发生

@dataclass
class AgentStep:
    """Agent 行为轨迹中的一步"""
    step_index: int
    step_type: StepType
    content: str                # 思考内容或最终回答
    tool_name: str = ""         # 工具名称（如果是 tool_call）
    tool_args: dict = field(default_factory=dict)
    tool_result: str = ""       # 工具返回值
    timestamp_ms: float = 0     # 时间戳
    token_count: int = 0        # 本步消耗 token

@dataclass
class Trajectory:
    """完整的 Agent 行为轨迹"""
    task_id: str
    task_description: str
    steps: list[AgentStep]
    final_answer: str
    total_steps: int = 0
    total_tokens: int = 0
    total_time_ms: float = 0
    success: bool = False

    def tool_call_sequence(self) -> list[str]:
        """提取工具调用序列"""
        return [s.tool_name for s in self.steps if s.step_type == StepType.TOOL_CALL]

    def has_repeated_calls(self, threshold: int = 3) -> bool:
        """检测是否有重复调用（潜在无限循环）"""
        from collections import Counter
        calls = self.tool_call_sequence()
        counts = Counter(calls)
        return any(c >= threshold for c in counts.values())
```

### 10.2.2 为什么需要 Trajectory 评估

```
任务：查询北京今天的天气，并判断是否适合户外运动。

Trajectory A（好路径）：
  1. [THINK] 需要先查天气，再判断
  2. [TOOL_CALL] weather_api(city="北京")
  3. [TOOL_RESULT] {"temp": 25, "condition": "晴", "aqi": 35}
  4. [THINK] 25°C，晴天，AQI 35（优），适合户外
  5. [ANSWER] 北京今天25°C晴天，空气质量优，非常适合户外运动。
  → 最终答案: 正确 ✓  路径: 高效合理 ✓

Trajectory B（差路径，答案碰巧对）：
  1. [TOOL_CALL] search("北京天气")           # 用了搜索而不是天气API
  2. [TOOL_RESULT] "北京今天晴天..."
  3. [TOOL_CALL] search("北京空气质量")         # 又搜了一次
  4. [TOOL_RESULT] "AQI 35..."
  5. [TOOL_CALL] search("25度适合运动吗")       # 不必要的搜索
  6. [TOOL_RESULT] "25度适合..."
  7. [TOOL_CALL] weather_api(city="北京")       # 终于用了正确API
  8. [TOOL_RESULT] {"temp": 25, "condition": "晴", "aqi": 35}
  9. [ANSWER] 北京今天适合户外运动。
  → 最终答案: 正确 ✓  路径: 低效且冗余 ✗
```

### 10.2.3 Trajectory 评估维度

| 维度 | 评估内容 | 指标 |
|------|---------|------|
| 最终正确性 | 最终答案是否正确 | exact_match / LLM-Judge |
| 工具选择 | 是否选了正确的工具 | tool_precision, tool_recall |
| 调用顺序 | 工具调用顺序是否合理 | sequence_similarity |
| 步骤效率 | 是否有冗余步骤 | step_count / optimal_step_count |
| 参数正确性 | 工具参数是否正确 | param_accuracy |
| 错误处理 | 遇到错误时是否正确恢复 | recovery_rate |

---

## 10.3 Recorded Tool Tapes

### 10.3.1 测试 Agent 的根本困难

Agent 依赖外部工具（API、数据库、搜索引擎），这些外部依赖带来不确定性：

```
问题：Agent 测试失败了
原因A：Agent 逻辑有 bug          → 需要修复 Agent
原因B：天气 API 返回了不同数据    → 不是 Agent 的问题
原因C：网络超时                  → 不是 Agent 的问题
```

你无法区分是 Agent 的问题还是环境的问题——除非你控制环境。

### 10.3.2 Tool Tapes：Agent 世界的 HTTP Fixtures

Tool Tape 就是预录制的工具调用和返回值，类似于 HTTP test fixtures / VCR cassettes：

```python
import json
from pathlib import Path

@dataclass
class ToolTapeEntry:
    """单条工具调用记录"""
    tool_name: str
    args: dict
    result: str
    latency_ms: float = 0

class ToolTapeRecorder:
    """录制模式：调用真实工具并记录结果"""

    def __init__(self, real_tools: dict):
        self.real_tools = real_tools  # {tool_name: callable}
        self.tape: list[ToolTapeEntry] = []

    def call(self, tool_name: str, **kwargs) -> str:
        result = self.real_tools[tool_name](**kwargs)
        self.tape.append(ToolTapeEntry(
            tool_name=tool_name,
            args=kwargs,
            result=result,
        ))
        return result

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump([e.__dict__ for e in self.tape], f, indent=2, ensure_ascii=False)

class ToolTapePlayer:
    """回放模式：从录制文件返回预设结果"""

    def __init__(self, tape_path: str):
        with open(tape_path) as f:
            entries = json.load(f)
        self.tape = [ToolTapeEntry(**e) for e in entries]
        self.call_index = 0

    def call(self, tool_name: str, **kwargs) -> str:
        if self.call_index >= len(self.tape):
            raise RuntimeError(
                f"Tool tape exhausted at step {self.call_index}. "
                f"Agent made more tool calls than expected."
            )

        expected = self.tape[self.call_index]

        # 可选：严格匹配或宽松匹配
        if expected.tool_name != tool_name:
            raise AssertionError(
                f"Step {self.call_index}: expected tool '{expected.tool_name}', "
                f"got '{tool_name}'"
            )

        self.call_index += 1
        return expected.result

    def verify_complete(self):
        """验证所有录制的调用都被消费了"""
        if self.call_index < len(self.tape):
            remaining = len(self.tape) - self.call_index
            raise AssertionError(
                f"Tool tape has {remaining} unconsumed entries. "
                f"Agent made fewer tool calls than expected."
            )
```

### 10.3.3 录制-回放工作流

```
Phase 1: 录制（第一次手动验证时）
  Agent + 真实工具 → 记录所有工具调用和返回值 → 保存 tape 文件

Phase 2: 回放（CI/CD 中自动测试）
  Agent + Tape Player → 确定性重放 → 对比 trajectory

Phase 3: 更新（工具 API 变更时）
  重新录制 → 人工验证 → 替换 tape 文件
```

```python
# 使用示例
# 录制
recorder = ToolTapeRecorder(real_tools={
    "weather": lambda city: '{"temp": 25, "condition": "sunny"}',
    "search": lambda query: "search results...",
})
# ... 运行 agent，agent 通过 recorder.call() 调用工具
recorder.save("tapes/weather_task_001.json")

# 回放
player = ToolTapePlayer("tapes/weather_task_001.json")
# ... 运行 agent，agent 通过 player.call() 调用工具
player.verify_complete()
```

---

## 10.4 端到端场景测试

### 10.4.1 场景测试设计

```python
@dataclass
class AgentTestScenario:
    """Agent 端到端测试场景"""
    id: str
    name: str
    description: str
    task: str                              # 给 Agent 的任务描述
    expected_final_answer: str             # 期望的最终答案
    expected_tool_sequence: list[str]       # 期望的工具调用序列（可选）
    max_steps: int = 20                    # 最大允许步数
    max_tokens: int = 10000                # 最大允许 token
    timeout_seconds: int = 60              # 超时限制
    tape_path: str = ""                    # tool tape 路径
    tags: list[str] = field(default_factory=list)

# 场景示例
SCENARIOS = [
    AgentTestScenario(
        id="weather_001",
        name="简单天气查询",
        description="基础功能：查天气并给建议",
        task="查询上海今天的天气，告诉我是否需要带伞",
        expected_final_answer="不需要带伞",
        expected_tool_sequence=["weather_api"],
        max_steps=5,
        tape_path="tapes/weather_001.json",
        tags=["basic", "weather"],
    ),
    AgentTestScenario(
        id="research_001",
        name="多步研究任务",
        description="复杂功能：需要多次搜索和综合",
        task="比较 Python 和 Rust 的 2025 年 Stack Overflow 开发者调查结果",
        expected_final_answer="",  # 开放式，用 LLM-Judge 评估
        expected_tool_sequence=["search", "search", "read_page"],
        max_steps=15,
        tape_path="tapes/research_001.json",
        tags=["complex", "research"],
    ),
]
```

### 10.4.2 场景测试执行器

```python
class AgentTestRunner:
    """Agent 场景测试执行器"""

    def __init__(self, agent, judge_model: str = "gpt-4o"):
        self.agent = agent
        self.judge_model = judge_model

    def run_scenario(self, scenario: AgentTestScenario) -> dict:
        """执行单个测试场景"""
        import time

        # 加载 tool tape
        if scenario.tape_path:
            player = ToolTapePlayer(scenario.tape_path)
            self.agent.set_tool_backend(player)

        # 执行 agent
        start = time.time()
        trajectory = self.agent.run(
            task=scenario.task,
            max_steps=scenario.max_steps,
        )
        elapsed = time.time() - start

        # 评估
        results = {
            "scenario_id": scenario.id,
            "scenario_name": scenario.name,
            "elapsed_seconds": round(elapsed, 2),
            "total_steps": len(trajectory.steps),
            "total_tokens": trajectory.total_tokens,
            "checks": {},
        }

        # Check 1: 最终答案正确性
        if scenario.expected_final_answer:
            results["checks"]["answer_correct"] = self._check_answer(
                trajectory.final_answer, scenario.expected_final_answer
            )
        else:
            results["checks"]["answer_quality"] = self._judge_answer(
                scenario.task, trajectory.final_answer
            )

        # Check 2: 工具调用序列
        if scenario.expected_tool_sequence:
            actual_sequence = trajectory.tool_call_sequence()
            results["checks"]["tool_sequence"] = self._check_tool_sequence(
                actual_sequence, scenario.expected_tool_sequence
            )

        # Check 3: 效率
        results["checks"]["efficiency"] = {
            "steps_within_limit": len(trajectory.steps) <= scenario.max_steps,
            "tokens_within_limit": trajectory.total_tokens <= scenario.max_tokens,
            "time_within_limit": elapsed <= scenario.timeout_seconds,
        }

        # Check 4: 安全性
        results["checks"]["safety"] = {
            "no_infinite_loop": not trajectory.has_repeated_calls(threshold=5),
            "completed": trajectory.success,
        }

        return results

    def _check_answer(self, actual: str, expected: str) -> dict:
        """简单答案匹配"""
        contains = expected.lower() in actual.lower()
        return {"passed": contains, "actual": actual, "expected": expected}

    def _judge_answer(self, task: str, answer: str) -> dict:
        """用 LLM-as-Judge 评估开放式回答"""
        prompt = f"""评估以下 Agent 的回答质量。

任务：{task}
回答：{answer}

请按 1-5 分评分，并给出理由。
输出 JSON 格式：{{"score": <1-5>, "reasoning": "..."}}"""

        import openai
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        result["passed"] = result.get("score", 0) >= 3
        return result

    def _check_tool_sequence(
        self, actual: list[str], expected: list[str]
    ) -> dict:
        """检查工具调用序列"""
        # 严格匹配
        exact_match = actual == expected

        # 宽松匹配：expected 是 actual 的子序列
        def is_subsequence(sub, full):
            it = iter(full)
            return all(item in it for item in sub)

        subsequence_match = is_subsequence(expected, actual)

        return {
            "exact_match": exact_match,
            "subsequence_match": subsequence_match,
            "actual_sequence": actual,
            "expected_sequence": expected,
            "passed": subsequence_match,
        }
```

---

## 10.5 步级评估

### 10.5.1 评估每一步而不只是最终结果

```python
STEP_EVALUATION_RUBRIC = """
你是一个 Agent 行为轨迹评估专家。请逐步评估以下 Agent 的行为。

**任务**：{task}

**行为轨迹**：
{trajectory_text}

**请对每一步评估以下维度**：
1. 必要性（Necessity）：这一步是否有必要？有没有更直接的方式？
2. 正确性（Correctness）：工具选择、参数、推理是否正确？
3. 进展性（Progress）：这一步是否推进了任务完成？

**请按以下 JSON 格式输出**：
{{
  "step_evaluations": [
    {{
      "step_index": 0,
      "necessity": {{"score": <1-5>, "feedback": "..."}},
      "correctness": {{"score": <1-5>, "feedback": "..."}},
      "progress": {{"score": <1-5>, "feedback": "..."}}
    }}
  ],
  "overall": {{
    "plan_quality": <1-5>,
    "execution_efficiency": <1-5>,
    "error_handling": <1-5>,
    "overall_score": <1-5>,
    "feedback": "..."
  }}
}}
"""

def evaluate_trajectory_steps(
    task: str,
    trajectory: Trajectory,
    judge_model: str = "gpt-4o"
) -> dict:
    """对 Agent trajectory 进行步级评估"""
    # 格式化 trajectory
    trajectory_text = ""
    for step in trajectory.steps:
        if step.step_type == StepType.THINK:
            trajectory_text += f"[思考] {step.content}\n"
        elif step.step_type == StepType.TOOL_CALL:
            trajectory_text += f"[工具调用] {step.tool_name}({json.dumps(step.tool_args, ensure_ascii=False)})\n"
        elif step.step_type == StepType.TOOL_RESULT:
            trajectory_text += f"[工具结果] {step.tool_result[:200]}...\n"
        elif step.step_type == StepType.ANSWER:
            trajectory_text += f"[最终回答] {step.content}\n"

    prompt = STEP_EVALUATION_RUBRIC.format(
        task=task,
        trajectory_text=trajectory_text
    )

    import openai
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content)
```

### 10.5.2 工具调用正确性指标

```python
def tool_call_precision_recall(
    actual_calls: list[dict],      # [{"tool": str, "args": dict}]
    expected_calls: list[dict]     # [{"tool": str, "args": dict}]
) -> dict:
    """计算工具调用的精确率和召回率"""

    def normalize_call(call):
        return (call["tool"], json.dumps(call.get("args", {}), sort_keys=True))

    actual_set = set(normalize_call(c) for c in actual_calls)
    expected_set = set(normalize_call(c) for c in expected_calls)

    if not actual_set and not expected_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    true_positives = actual_set & expected_set
    precision = len(true_positives) / len(actual_set) if actual_set else 0.0
    recall = len(true_positives) / len(expected_set) if expected_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "true_positives": len(true_positives),
        "false_positives": len(actual_set - expected_set),
        "false_negatives": len(expected_set - actual_set),
    }
```

---

## 10.6 混沌工程：故障注入

### 10.6.1 为什么需要混沌工程

生产环境中，Agent 依赖的工具不会总是正常工作：

```
现实世界的 Agent 会遇到：
├── HTTP 500 错误
├── 超时（工具响应时间 > 30s）
├── 返回格式错误的 JSON
├── 返回空结果
├── 返回与预期完全不同的数据
└── 服务完全不可用
```

一个好的 Agent 应该能优雅地处理这些情况。混沌工程（Chaos Engineering）通过主动注入故障来测试 Agent 的鲁棒性。

### 10.6.2 故障注入器

```python
import random

class ChaosToolWrapper:
    """混沌工程工具包装器：在真实工具调用上注入故障"""

    def __init__(
        self,
        real_tool_fn,
        failure_rate: float = 0.2,
        failure_modes: list[str] = None,
        seed: int = 42,
    ):
        self.real_tool_fn = real_tool_fn
        self.failure_rate = failure_rate
        self.failure_modes = failure_modes or [
            "http_500",
            "timeout",
            "malformed_json",
            "empty_response",
        ]
        self.rng = random.Random(seed)
        self.call_log = []

    def __call__(self, **kwargs) -> str:
        """调用工具，以一定概率注入故障"""
        if self.rng.random() < self.failure_rate:
            failure = self.rng.choice(self.failure_modes)
            self.call_log.append({"args": kwargs, "failure": failure})
            return self._inject_failure(failure)

        result = self.real_tool_fn(**kwargs)
        self.call_log.append({"args": kwargs, "result": result, "failure": None})
        return result

    def _inject_failure(self, mode: str) -> str:
        if mode == "http_500":
            raise Exception("HTTP 500: Internal Server Error")
        elif mode == "timeout":
            import time
            time.sleep(0.1)  # 模拟短暂延迟
            raise TimeoutError("Tool call timed out after 30 seconds")
        elif mode == "malformed_json":
            return '{"result": "incomplete data, "missing_bracket'
        elif mode == "empty_response":
            return ""
        else:
            raise Exception(f"Unknown error: {mode}")

# 使用示例
def real_weather_api(city: str) -> str:
    return json.dumps({"temp": 25, "condition": "sunny"})

chaos_weather = ChaosToolWrapper(
    real_weather_api,
    failure_rate=0.3,
    failure_modes=["http_500", "timeout", "malformed_json"],
)

# 运行 Agent 多次，统计成功率
results = []
for i in range(50):
    chaos_weather.rng = random.Random(i)  # 不同种子
    try:
        # agent.run(tools={"weather": chaos_weather})
        results.append("success")
    except Exception as e:
        results.append(f"failure: {e}")
```

### 10.6.3 无限循环检测和停机安全

```python
class AgentSafetyGuard:
    """Agent 安全护栏：检测无限循环和资源滥用"""

    def __init__(
        self,
        max_steps: int = 50,
        max_tokens: int = 50000,
        max_time_seconds: int = 120,
        max_repeated_calls: int = 5,
        max_consecutive_errors: int = 3,
    ):
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.max_time_seconds = max_time_seconds
        self.max_repeated_calls = max_repeated_calls
        self.max_consecutive_errors = max_consecutive_errors

    def check(self, trajectory: Trajectory) -> dict:
        """检查 trajectory 是否触发安全规则"""
        violations = []

        # 1. 步数限制
        if len(trajectory.steps) >= self.max_steps:
            violations.append({
                "rule": "max_steps",
                "message": f"步数 {len(trajectory.steps)} >= {self.max_steps}",
                "severity": "critical",
            })

        # 2. Token 限制
        if trajectory.total_tokens >= self.max_tokens:
            violations.append({
                "rule": "max_tokens",
                "message": f"Token 使用 {trajectory.total_tokens} >= {self.max_tokens}",
                "severity": "critical",
            })

        # 3. 重复调用检测
        from collections import Counter
        tool_calls = trajectory.tool_call_sequence()
        call_counts = Counter(tool_calls)
        for tool, count in call_counts.items():
            if count >= self.max_repeated_calls:
                violations.append({
                    "rule": "repeated_calls",
                    "message": f"工具 '{tool}' 被调用 {count} 次 >= {self.max_repeated_calls}",
                    "severity": "warning",
                })

        # 4. 连续错误检测
        consecutive_errors = 0
        max_consecutive = 0
        for step in trajectory.steps:
            if step.step_type == StepType.ERROR:
                consecutive_errors += 1
                max_consecutive = max(max_consecutive, consecutive_errors)
            else:
                consecutive_errors = 0

        if max_consecutive >= self.max_consecutive_errors:
            violations.append({
                "rule": "consecutive_errors",
                "message": f"连续错误 {max_consecutive} 次 >= {self.max_consecutive_errors}",
                "severity": "critical",
            })

        return {
            "safe": len(violations) == 0,
            "violations": violations,
            "critical_count": sum(1 for v in violations if v["severity"] == "critical"),
            "warning_count": sum(1 for v in violations if v["severity"] == "warning"),
        }
```

---

## 10.7 Benchmark 生态

### 10.7.1 主要 Agent Benchmark

| Benchmark | 领域 | 评估方式 | 难度 |
|-----------|------|---------|------|
| **AgentBench** | 通用（代码、游戏、Web） | 端到端 + trajectory | 高 |
| **SWE-bench Verified** | 软件工程（修 bug） | 自动化测试通过率 | 极高 |
| **WebArena** | Web 浏览操作 | 任务完成率 | 高 |
| **GAIA** | 通用助手 | 最终答案正确性 | 中-高 |
| **ToolBench** | 工具调用 | 调用序列 + 结果 | 中 |
| **τ-bench** | 客服场景 | 用户满意度模拟 | 中 |

### 10.7.2 SWE-bench 的评估方式

SWE-bench 是目前最有影响力的 Agent benchmark 之一。它的评估逻辑值得学习：

```
任务：给定一个 GitHub issue，让 Agent 修改代码来解决 issue
评估：
  1. Agent 提交一个 patch（代码修改）
  2. 运行仓库的原有测试套件
  3. 之前失败的测试现在通过了 → 成功
  4. 之前通过的测试没有被破坏 → 无回归

关键设计：
  - 不用 LLM-Judge 评估代码质量
  - 完全用确定性测试（运行测试套件）
  - 同时检测正向（修复）和负向（回归）
```

### 10.7.3 选择 Benchmark 的原则

```
你的 Agent 做什么？
├── 写代码/修 bug → SWE-bench Verified
├── 浏览网页 → WebArena
├── 调用 API/工具 → ToolBench
├── 通用问答助手 → GAIA
├── 客服/对话 → τ-bench
└── 多领域混合 → AgentBench
```

---

## 本章小结

| 概念 | 要点 |
|------|------|
| 根本性难度 | Agent 行为空间是组合爆炸的，独有失败模式包括无限循环、工具误用、错误传播 |
| Trajectory 评估 | 行为轨迹是正确性的主要单元——正确答案 + 错误路径 ≠ 真正正确 |
| Tool Tapes | 录制-回放工具调用，实现确定性可复现的 Agent 测试 |
| 端到端场景测试 | 用 AgentTestScenario 定义完整测试场景，检查答案 + 工具序列 + 效率 + 安全 |
| 步级评估 | 对每一步评估必要性、正确性、进展性 |
| 混沌工程 | 注入故障（500 error、timeout、malformed JSON）测试鲁棒性 |
| 安全护栏 | 无限循环检测、步数限制、token 限制、连续错误检测 |
| Benchmark 生态 | SWE-bench（代码）、WebArena（Web）、AgentBench（通用） |

---

## 动手实验

### 实验 1：构建 Trajectory 评估系统

**目标**：为一个简单的工具调用 Agent 实现 trajectory 评估。

**步骤**：

1. 定义一个模拟 Agent 和工具集：

```python
import json

# 模拟工具
def calculator(expression: str) -> str:
    try:
        result = eval(expression)  # 简化示例，生产环境不要用 eval
        return json.dumps({"result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})

def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    conversions = {
        ("km", "miles"): 0.621371,
        ("miles", "km"): 1.60934,
        ("kg", "lbs"): 2.20462,
        ("lbs", "kg"): 0.453592,
        ("celsius", "fahrenheit"): lambda v: v * 9/5 + 32,
        ("fahrenheit", "celsius"): lambda v: (v - 32) * 5/9,
    }
    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        factor = conversions[key]
        if callable(factor):
            result = factor(value)
        else:
            result = value * factor
        return json.dumps({"result": round(result, 2), "unit": to_unit})
    return json.dumps({"error": f"Unsupported conversion: {from_unit} -> {to_unit}"})

TOOLS = {"calculator": calculator, "unit_converter": unit_converter}
```

2. 模拟两个不同质量的 trajectory：

```python
# 好的 trajectory
good_trajectory = Trajectory(
    task_id="math_001",
    task_description="将 100 公里转换为英里，然后计算以每小时 60 英里的速度需要多少分钟",
    steps=[
        AgentStep(0, StepType.THINK, "需要两步：先转换单位，再算时间"),
        AgentStep(1, StepType.TOOL_CALL, "", "unit_converter",
                  {"value": 100, "from_unit": "km", "to_unit": "miles"}),
        AgentStep(2, StepType.TOOL_RESULT, "", tool_result='{"result": 62.14, "unit": "miles"}'),
        AgentStep(3, StepType.TOOL_CALL, "", "calculator",
                  {"expression": "62.14 / 60 * 60"}),
        AgentStep(4, StepType.TOOL_RESULT, "", tool_result='{"result": 62.14}'),
        AgentStep(5, StepType.ANSWER, "100公里约等于62.14英里，以60英里/小时需要约62.14分钟"),
    ],
    final_answer="100公里约等于62.14英里，以60英里/小时需要约62.14分钟",
    total_steps=6,
    success=True,
)

# 差的 trajectory
bad_trajectory = Trajectory(
    task_id="math_001",
    task_description="将 100 公里转换为英里，然后计算以每小时 60 英里的速度需要多少分钟",
    steps=[
        AgentStep(0, StepType.TOOL_CALL, "", "calculator",
                  {"expression": "100 * 0.6"}),  # 错误的转换因子
        AgentStep(1, StepType.TOOL_RESULT, "", tool_result='{"result": 60}'),
        AgentStep(2, StepType.TOOL_CALL, "", "calculator",
                  {"expression": "100 / 1.6"}),  # 重新算了一次
        AgentStep(3, StepType.TOOL_RESULT, "", tool_result='{"result": 62.5}'),
        AgentStep(4, StepType.TOOL_CALL, "", "unit_converter",
                  {"value": 100, "from_unit": "km", "to_unit": "miles"}),  # 又用了工具
        AgentStep(5, StepType.TOOL_RESULT, "", tool_result='{"result": 62.14, "unit": "miles"}'),
        AgentStep(6, StepType.TOOL_CALL, "", "calculator",
                  {"expression": "62.14 / 60 * 60"}),
        AgentStep(7, StepType.TOOL_RESULT, "", tool_result='{"result": 62.14}'),
        AgentStep(8, StepType.ANSWER, "大约62分钟"),
    ],
    final_answer="大约62分钟",
    total_steps=9,
    success=True,
)
```

3. 对两个 trajectory 进行步级评估，对比分数差异。

### 实验 2：混沌工程故障注入测试

**目标**：用 ChaosToolWrapper 测试 Agent 对故障的处理能力。

```python
# 包装工具
chaos_calc = ChaosToolWrapper(
    calculator,
    failure_rate=0.3,
    failure_modes=["http_500", "malformed_json", "empty_response"],
)

chaos_converter = ChaosToolWrapper(
    unit_converter,
    failure_rate=0.2,
    failure_modes=["http_500", "timeout"],
)

# 运行 50 次，统计 Agent 在故障条件下的表现
success_count = 0
graceful_failure_count = 0  # 识别错误并优雅处理
crash_count = 0             # 崩溃
loop_count = 0              # 陷入循环

guard = AgentSafetyGuard(
    max_steps=20,
    max_repeated_calls=4,
    max_consecutive_errors=3,
)

# 每次用不同种子以获得不同的故障模式
for seed in range(50):
    chaos_calc.rng = random.Random(seed)
    chaos_converter.rng = random.Random(seed + 1000)

    # trajectory = agent.run(
    #     task="将100公里转换为英里",
    #     tools={"calculator": chaos_calc, "unit_converter": chaos_converter}
    # )
    # safety_check = guard.check(trajectory)
    # ... 分类结果

print(f"成功率: {success_count}/50")
print(f"优雅降级: {graceful_failure_count}/50")
print(f"崩溃: {crash_count}/50")
print(f"死循环: {loop_count}/50")
```

---

## 练习题

### 基础题

1. **Trajectory vs 最终答案**：解释为什么 Agent 评估中"正确的最终答案"不能作为唯一指标。给出两个具体例子：(a) 最终答案正确但 trajectory 差，(b) 最终答案错误但 trajectory 大部分正确。

2. **Tool Tape 设计**：你的 Agent 使用三个工具：搜索引擎、计算器、邮件发送。为"帮我搜索明天北京到上海的机票价格并发邮件通知我"这个任务设计一份 tool tape（JSON 格式）。

3. **安全护栏**：列出三种 Agent 可能陷入无限循环的场景，并为每种场景设计一个检测规则。

### 实践题

1. **完整评估管线**：整合本章的代码，构建一个可以：
   - 加载测试场景（JSON 格式）
   - 使用 tool tape 进行确定性测试
   - 评估 trajectory（步级评估 + 最终答案）
   - 输出包含通过/失败信息的测试报告
   的完整评估管线。

2. **混沌工程实验**：为一个你熟悉的 Agent 框架（如 LangGraph、CrewAI 或自定义 Agent），实现 ChaosToolWrapper，在 30% 故障率下运行 50 次测试，统计并分析 Agent 的容错能力。

### 思考题

1. SWE-bench 用确定性测试（运行代码测试套件）来评估 Agent。这种方法能否推广到其他 Agent 场景（如客服、研究助手）？如果不能，根本障碍是什么？

2. 随着 Agent 能力越来越强，评估 Agent 本身可能需要另一个更强的 Agent。这是否会导致"评估无穷递归"？你如何设计一个评估体系来避免这个问题？
