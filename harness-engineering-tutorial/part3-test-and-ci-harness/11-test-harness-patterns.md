# 第11章：AI 系统的测试 Harness 模式

> 软件工程有六十年的测试积累，但 AI 系统在第一秒就打破了最基本的假设：相同输入产生相同输出。本章将展示如何继承测试 harness 的工程遗产，同时彻底重构它以适应非确定性世界。

---

## 学习目标

学完本章，你将能够：

1. 解释测试 harness 的历史演变，以及 AI 系统为何需要新的测试范式
2. 为 LLM 调用设计完整的 mock 策略（沙箱执行、工具 mock、安全隔离）
3. 使用 seed 固定、temperature=0、输出固定等手段构建确定性测试
4. 设计执行步数限制机制防止测试成本失控
5. 在 AI 系统中正确划分 unit / integration / e2e 测试边界

---

## 11.1 软件工程遗产：测试 Harness 的前世今生

### 什么是测试 Harness

测试 harness 不是一个测试，而是**运行测试的基础设施**。它包括：

| 组件 | 传统软件 | AI 系统 |
|------|---------|---------|
| 测试运行器 | pytest / JUnit | 评估管线 + pytest |
| 断言机制 | `assertEqual(a, b)` | 语义相似度 / 属性检查 |
| 环境隔离 | Docker / virtualenv | 沙箱 + model mock |
| 数据管理 | fixtures / factories | 黄金数据集 + 动态生成 |
| 结果报告 | pass/fail | 多维度分数 + 分布统计 |

### 传统假设的崩塌

```
传统软件：f(x) = y       （确定性映射）
AI 系统：  f(x) ∈ Y       （概率分布中的采样）
```

这意味着：

- **`assertEqual` 失效**：同一个 prompt 两次调用可能返回不同文本
- **覆盖率无意义**：你不是在测试代码路径，而是在测试模型行为的分布
- **回归的定义改变**：不是"输出变了"，而是"输出质量的分布向下漂移了"

### Harness 作为适配层

```
┌─────────────────────────────────────────────────┐
│                Test Harness                      │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ Mock 层   │  │ 断言适配 │  │ 执行控制      │  │
│  │ - LLM    │  │ - 语义   │  │ - 步数限制    │  │
│  │ - Tools  │  │ - 属性   │  │ - 成本限制    │  │
│  │ - I/O    │  │ - 分布   │  │ - 超时控制    │  │
│  └──────────┘  └──────────┘  └───────────────┘  │
│  ┌──────────────────────────────────────────┐    │
│  │          环境隔离与安全沙箱               │    │
│  └──────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

---

## 11.2 Mock 一切：沙箱执行、工具 Mock、安全隔离

### 为什么 AI 测试需要 Mock 一切

AI agent 可以调用工具、执行代码、访问文件系统。测试中放任这些行为意味着：

1. **成本不可控**：每次测试都调用真实 LLM API
2. **安全风险**：agent 可能删除文件、发送请求
3. **不可复现**：网络延迟、API 版本变化导致测试不稳定

### 三层 Mock 架构

```python
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import Any

# ===== 第一层：模型 Mock =====
class MockLLM:
    """可预测的 LLM 替身"""
    
    def __init__(self, responses: dict[str, str] | None = None):
        self.responses = responses or {}
        self.call_log: list[dict] = []
        self._default = "This is a mock response."
    
    def complete(self, prompt: str, **kwargs) -> str:
        self.call_log.append({"prompt": prompt, **kwargs})
        # 按关键词匹配预设响应
        for pattern, response in self.responses.items():
            if pattern in prompt:
                return response
        return self._default
    
    def assert_called_with_prompt_containing(self, text: str):
        for call in self.call_log:
            if text in call["prompt"]:
                return True
        raise AssertionError(f"No call contained: {text}")


# ===== 第二层：工具 Mock =====
class MockToolbox:
    """沙箱化的工具执行环境"""
    
    def __init__(self):
        self.tools: dict[str, Any] = {}
        self.call_history: list[dict] = []
        self._blocked_tools: set[str] = set()
    
    def register(self, name: str, handler):
        self.tools[name] = handler
    
    def block(self, name: str):
        """禁止某个工具被调用（安全隔离）"""
        self._blocked_tools.add(name)
    
    def execute(self, name: str, **kwargs) -> Any:
        if name in self._blocked_tools:
            raise PermissionError(f"Tool '{name}' is blocked in test")
        self.call_history.append({"tool": name, **kwargs})
        if name in self.tools:
            return self.tools[name](**kwargs)
        return {"status": "mock_success"}


# ===== 第三层：安全沙箱 =====
@dataclass
class SandboxConfig:
    allow_network: bool = False
    allow_filesystem: bool = False
    allow_subprocess: bool = False
    max_memory_mb: int = 512
    max_time_seconds: int = 30
    blocked_tools: list[str] = field(default_factory=lambda: [
        "shell_exec", "file_delete", "http_request"
    ])
```

### 组合使用

```python
class AITestHarness:
    """AI 系统测试 harness 骨架"""
    
    def __init__(self, sandbox_config: SandboxConfig | None = None):
        self.llm = MockLLM()
        self.toolbox = MockToolbox()
        self.sandbox = sandbox_config or SandboxConfig()
        self._step_count = 0
        self._max_steps = 50
        self._cost_cents = 0.0
        self._max_cost_cents = 100.0
    
    def set_llm_responses(self, responses: dict[str, str]):
        self.llm = MockLLM(responses)
    
    def run_agent_step(self, agent_fn, input_data: dict) -> dict:
        """执行一步 agent 逻辑，带安全护栏"""
        self._step_count += 1
        if self._step_count > self._max_steps:
            raise RuntimeError(
                f"Agent exceeded {self._max_steps} steps — aborting"
            )
        return agent_fn(
            llm=self.llm,
            tools=self.toolbox,
            input_data=input_data,
        )
    
    def get_report(self) -> dict:
        return {
            "steps": self._step_count,
            "llm_calls": len(self.llm.call_log),
            "tool_calls": len(self.toolbox.call_history),
        }
```

---

## 11.3 确定性测试：驯服非确定性输出

### 确定性三件套

| 技术 | 原理 | 适用场景 | 局限 |
|------|------|---------|------|
| `seed` 固定 | 固定随机采样种子 | 本地开发调试 | 跨版本不保证一致 |
| `temperature=0` | 贪心解码，选概率最高 token | CI 中的回归检测 | 仍可能有微小变化 |
| Output pinning | 缓存预期输出，比较新输出 | 关键路径冻结 | 模型升级后需更新 pin |

### Seed + Temperature=0 实现

```python
import hashlib
import json
from pathlib import Path

class DeterministicTestClient:
    """尽可能确定性的 LLM 测试客户端"""
    
    def __init__(self, client, seed: int = 42):
        self.client = client
        self.seed = seed
    
    def complete(self, prompt: str, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=kwargs.get("model", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,       # 贪心解码
            seed=self.seed,      # 固定种子
            max_tokens=kwargs.get("max_tokens", 500),
        )
        return response.choices[0].message.content


class OutputPinner:
    """输出固定（pinning）管理器"""
    
    def __init__(self, pin_dir: str = ".test_pins"):
        self.pin_dir = Path(pin_dir)
        self.pin_dir.mkdir(exist_ok=True)
    
    def _key(self, test_name: str) -> Path:
        return self.pin_dir / f"{test_name}.json"
    
    def pin(self, test_name: str, output: str):
        """保存一个期望输出"""
        data = {"output": output, "hash": hashlib.sha256(output.encode()).hexdigest()}
        self._key(test_name).write_text(json.dumps(data, indent=2))
    
    def check(self, test_name: str, actual: str, tolerance: float = 0.0) -> bool:
        """比较实际输出与 pinned 输出"""
        pin_file = self._key(test_name)
        if not pin_file.exists():
            # 首次运行，自动 pin
            self.pin(test_name, actual)
            return True
        
        data = json.loads(pin_file.read_text())
        expected = data["output"]
        
        if tolerance == 0.0:
            return actual == expected
        
        # 带容差的比较（用 embedding 相似度）
        return self._semantic_similarity(actual, expected) >= (1.0 - tolerance)
    
    def _semantic_similarity(self, a: str, b: str) -> float:
        """简化版语义相似度（实际应使用 embedding）"""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)
```

### 确定性测试的分级策略

```
Level 0: 完全 mock（零 API 调用）
  └─ 适用：逻辑测试、prompt 拼装测试

Level 1: seed + temp=0（少量 API 调用）
  └─ 适用：关键路径回归

Level 2: output pinning + 容差（少量 API 调用）
  └─ 适用：自然语言输出的稳定性测试

Level 3: 统计测试（N 次调用 + 分布检验）
  └─ 适用：上线前的全面验证
```

---

## 11.4 执行控制：限制步数防止成本失控

### 问题：失控的 Agent 循环

一个没有约束的 agent 可能进入无限循环：

```
Agent: 我需要搜索文件 → 调用 search_tool
Agent: 结果不够，让我再搜索 → 调用 search_tool
Agent: 还是不够... → 调用 search_tool × ∞
```

每次循环都是一次 LLM 调用，都在烧钱。

### 多维度限制器

```python
import time
from dataclasses import dataclass

@dataclass
class ExecutionLimits:
    max_steps: int = 50
    max_llm_calls: int = 20
    max_tool_calls: int = 30
    max_cost_usd: float = 1.0
    max_wall_time_seconds: float = 120.0
    max_tokens_total: int = 100_000

class ExecutionGuard:
    """测试执行的多维度护栏"""
    
    def __init__(self, limits: ExecutionLimits):
        self.limits = limits
        self.steps = 0
        self.llm_calls = 0
        self.tool_calls = 0
        self.cost_usd = 0.0
        self.tokens_total = 0
        self.start_time = time.time()
    
    def check(self, event_type: str, **kwargs):
        """每次操作前检查是否超限"""
        self.steps += 1
        
        if event_type == "llm_call":
            self.llm_calls += 1
            tokens = kwargs.get("tokens", 0)
            self.tokens_total += tokens
            # 简化计费：$0.01 / 1K tokens
            self.cost_usd += tokens * 0.00001
        
        elif event_type == "tool_call":
            self.tool_calls += 1
        
        elapsed = time.time() - self.start_time
        
        violations = []
        if self.steps > self.limits.max_steps:
            violations.append(f"steps: {self.steps}/{self.limits.max_steps}")
        if self.llm_calls > self.limits.max_llm_calls:
            violations.append(f"llm_calls: {self.llm_calls}/{self.limits.max_llm_calls}")
        if self.tool_calls > self.limits.max_tool_calls:
            violations.append(f"tool_calls: {self.tool_calls}/{self.limits.max_tool_calls}")
        if self.cost_usd > self.limits.max_cost_usd:
            violations.append(f"cost: ${self.cost_usd:.2f}/${self.limits.max_cost_usd}")
        if elapsed > self.limits.max_wall_time_seconds:
            violations.append(f"time: {elapsed:.0f}s/{self.limits.max_wall_time_seconds}s")
        if self.tokens_total > self.limits.max_tokens_total:
            violations.append(f"tokens: {self.tokens_total}/{self.limits.max_tokens_total}")
        
        if violations:
            raise ResourceExhaustedError(
                f"Execution limits exceeded: {', '.join(violations)}"
            )
    
    def summary(self) -> dict:
        return {
            "steps": self.steps,
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "cost_usd": round(self.cost_usd, 4),
            "wall_time_s": round(time.time() - self.start_time, 2),
            "tokens_total": self.tokens_total,
        }

class ResourceExhaustedError(Exception):
    pass
```

---

## 11.5 测试分层：Unit / Integration / E2E 的含义变化

### 传统 vs AI 系统的测试金字塔

```
传统软件：                    AI 系统：
    /\                           /\
   /E2E\                        /E2E\  ← 全链路 agent 运行
  /------\                     /------\
 /Integr. \                   /Integr. \ ← prompt + tool + model
/----------\                 /----------\
/   Unit    \               / Unit+Mock  \ ← 纯逻辑 + mock LLM
--------------              ----------------
```

### 各层定义

| 层级 | 传统含义 | AI 系统含义 | Mock 什么 | 运行频率 |
|------|---------|------------|----------|---------|
| Unit | 测试单个函数 | 测试 prompt 组装、输出解析、工具调用逻辑 | Mock LLM、Mock 工具 | 每次提交 |
| Integration | 模块间交互 | Prompt + 真实/半真实 LLM + 工具 | Mock 外部服务 | 每个 PR |
| E2E | 完整用户流程 | 完整 agent 循环，真实 LLM | 最少 mock | 每日 / 发布前 |

### 各层示例

```python
import pytest

# ===== Unit 测试：零 API 调用 =====
class TestPromptAssembly:
    def test_system_prompt_includes_tools(self):
        tools = [{"name": "search", "description": "Search files"}]
        prompt = build_system_prompt(tools=tools)
        assert "search" in prompt
        assert "Search files" in prompt
    
    def test_output_parser_extracts_json(self):
        raw = 'Some thinking...\n```json\n{"action": "search"}\n```'
        parsed = parse_agent_output(raw)
        assert parsed["action"] == "search"
    
    def test_output_parser_handles_malformed(self):
        raw = "No JSON here at all"
        with pytest.raises(ParseError):
            parse_agent_output(raw)


# ===== Integration 测试：有限 API 调用 =====
class TestAgentToolIntegration:
    def test_agent_calls_correct_tool(self):
        harness = AITestHarness()
        harness.set_llm_responses({
            "search": '{"action": "search", "query": "test file"}',
        })
        harness.toolbox.register("search", lambda query: ["file1.py"])
        
        result = harness.run_agent_step(
            agent_fn=my_agent_step,
            input_data={"user_query": "Find test files"},
        )
        
        assert len(harness.toolbox.call_history) == 1
        assert harness.toolbox.call_history[0]["tool"] == "search"


# ===== E2E 测试：真实 LLM（标记为 slow） =====
@pytest.mark.slow
@pytest.mark.cost("~$0.05")
class TestFullAgentLoop:
    def test_agent_completes_task(self, real_llm_client):
        result = run_agent(
            client=real_llm_client,
            task="Summarize the file README.md",
            max_steps=10,
        )
        assert result.status == "completed"
        assert len(result.output) > 50
        assert result.steps_taken <= 10
```

---

## 11.6 综合：测试 Harness 配置模板

```python
# conftest.py — AI 项目的 pytest 配置
import pytest

def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", help="Run slow E2E tests")
    parser.addoption("--run-costly", action="store_true", help="Run tests that cost money")

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "cost: marks tests that cost money")

def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="need --run-slow to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

@pytest.fixture
def ai_harness():
    """标准 AI 测试 harness fixture"""
    harness = AITestHarness(
        sandbox_config=SandboxConfig(
            allow_network=False,
            allow_filesystem=False,
            max_time_seconds=30,
        )
    )
    yield harness
    # 测试后输出报告
    report = harness.get_report()
    print(f"\n[Harness Report] {report}")

@pytest.fixture
def deterministic_client():
    """确定性 LLM 客户端 fixture"""
    return DeterministicTestClient(seed=42)
```

---

## 本章小结

| 主题 | 关键洞察 |
|------|---------|
| 测试 harness 本质 | 运行测试的基础设施，不是测试本身 |
| Mock 三层架构 | 模型 mock → 工具 mock → 安全沙箱 |
| 确定性策略 | seed + temperature=0 + output pinning 三件套 |
| 执行控制 | 步数、调用次数、成本、时间的多维限制 |
| 测试分层 | Unit（纯 mock）→ Integration（半真实）→ E2E（全真实） |
| 成本管理 | 在 harness 层面强制成本上限，而非依赖开发者自觉 |

---

## 动手实验

### 实验 1：为 LLM 调用写确定性测试

**目标**：使用 mock + output pinning 为一个摘要任务写测试

```python
# 实验骨架
def summarize(llm, text: str) -> str:
    prompt = f"请用一句话总结以下内容:\n\n{text}"
    return llm.complete(prompt)

# TODO: 
# 1. 用 MockLLM 写 unit 测试验证 prompt 拼装
# 2. 用 OutputPinner 写一个 pinned 测试
# 3. 验证当 mock 返回空字符串时的错误处理
```

### 实验 2：构建完整的执行限制器

**目标**：扩展 `ExecutionGuard`，增加以下能力：

1. 按工具名称设置单独的调用上限（如 `search` 最多调用 5 次）
2. 实现"预算衰减"——随着步数增加，每步允许的 token 数递减
3. 写测试验证各种超限场景

### 实验 3：模糊测试 LLM 输出解析器

**目标**：用 hypothesis 库对 `parse_agent_output` 做 property-based testing

```python
from hypothesis import given, strategies as st

@given(st.text(min_size=0, max_size=10000))
def test_parser_never_crashes(random_text):
    """无论输入什么，解析器不应该抛出未处理的异常"""
    try:
        result = parse_agent_output(random_text)
        # 如果解析成功，结果应该是 dict
        assert isinstance(result, dict)
    except ParseError:
        pass  # 预期的错误类型是允许的
    # 不应该有其他异常类型
```

---

## 练习题

### 基础题

1. 解释为什么 `assertEqual("Hello, world!", llm.complete("Say hello"))` 是一个糟糕的测试。给出三种改进方案。

2. 在以下场景中，你应该使用哪个确定性级别（Level 0-3）？
   - (a) 测试 prompt 模板是否正确包含了用户名
   - (b) 验证模型在更新后仍然能正确分类情感
   - (c) 测试 agent 是否在遇到错误时调用了 fallback 工具

3. 列出三种 AI agent 测试中必须 mock 的外部依赖，并说明不 mock 的风险。

### 实践题

4. 设计一个 `CostTracker` 类，它能：(a) 按模型追踪 token 用量，(b) 按测试用例分配成本，(c) 生成成本报告。写出完整实现。

5. 你的 E2E 测试在本地通过但在 CI 中随机失败。列出你的排查步骤（至少5步），以及每步对应的修复策略。

### 思考题

6. "AI 系统的测试金字塔应该倒过来——E2E 测试比 Unit 测试更有价值。" 你同意吗？论证你的观点。

7. 如果一个 AI 系统的所有测试都用 mock LLM，那么这些测试到底在测什么？讨论 mock 测试的价值边界。
