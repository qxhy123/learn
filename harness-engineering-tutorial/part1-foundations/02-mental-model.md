# 第2章：核心心智模型：Agent = Model + Harness

> 一个公式可以浓缩一门学科的精华。本章深入拆解 "Agent = Model + Harness" 的每一个组成部分，帮助你建立坚实的心智模型——这是后续所有章节的基石。

---

## 学习目标

学完本章，你将能够：

1. 用操作系统类比精确解释 Model、Harness、Agent 和 Context Window 的角色
2. 描述 Harness 的三层内部结构及各层职责
3. 区分 Harness、Framework 和 Scaffold 三个易混淆概念
4. 实现一个包含核心组件的最小 Harness
5. 规划从手工 prompt 到工程化 harness 的渐进升级路径

---

## 2.1 基本等式与操作系统类比

### Agent = Model + Harness

这个等式看起来简单，实则蕴含深意。让我们拆开每一项：

- **Agent**：一个能自主完成任务的 AI 系统（对标应用程序 Application）
- **Model**：大语言模型，提供推理和生成能力（对标中央处理器 CPU）
- **Harness**：围绕模型的运行时系统，提供控制、协调和基础设施（对标操作系统 OS）

### 操作系统类比

这个类比值得深入展开：

```
┌─────────────────────────────────────────────────────────┐
│               计算机系统  ←→  AI Agent 系统               │
├─────────────────────────┬───────────────────────────────┤
│  Application (应用)     │  Agent (智能体)                │
│    ↕ 通过 OS 调度       │    ↕ 通过 Harness 编排         │
│  OS (操作系统)          │  Harness (驾驭系统)            │
│    ↕ 管理硬件资源       │    ↕ 管理模型调用              │
│  CPU (处理器)           │  Model (大语言模型)            │
│    ↕ 使用内存           │    ↕ 使用上下文                │
│  RAM (内存)             │  Context Window (上下文窗口)   │
└─────────────────────────┴───────────────────────────────┘
```

让我们逐层对照：

| 计算机组件 | AI 对应物 | 功能对比 |
|-----------|----------|---------|
| CPU | Model | 执行计算/推理。能力强大但需要指令流来驱动 |
| RAM | Context Window | 工作记忆。有大小限制，需要精心管理 |
| OS | Harness | 资源调度、进程管理、I/O 控制、错误处理 |
| Application | Agent | 面向用户的完整解决方案 |
| 文件系统 | 外部存储/RAG | 持久化存储，弥补内存不足 |
| 设备驱动 | Tool Adapters | 连接外部硬件/服务的接口 |

### 类比的启发

这个类比揭示了几个重要洞察：

**1. 没有人直接用 CPU 编程**

你不会写机器码直接操控 CPU——你通过操作系统来使用它。同理，直接裸调模型 API 就像直接写汇编，可以工作，但脆弱、低效且难以维护。

**2. OS 不替代 CPU，而是让 CPU 可用**

Harness 不替代模型，不改变模型的能力上限。它做的是让模型的能力被可靠地、可控地释放出来。

**3. 好的 OS 让差的硬件也能工作**

一个精心设计的 harness 可以让中等模型超越粗糙使用的顶级模型——就像 Linux 可以让廉价硬件跑出色的服务器性能。

---

## 2.2 Harness 的三层结构

一个完整的 Harness 可以分解为三层，每层有清晰的职责边界：

```
┌───────────────────────────────────────────┐
│          编排层 (Orchestration)            │  ← 任务分解、多步推理、agent 间协调
│  ┌───────────────────────────────────┐    │
│  │     运行时环境层 (Runtime)         │    │  ← 重试、验证、状态管理、可观测性
│  │  ┌───────────────────────────┐    │    │
│  │  │   模型接口层 (Interface)  │    │    │  ← API 调用、prompt 构建、解析响应
│  │  │                           │    │    │
│  │  │       [ M o d e l ]       │    │    │
│  │  └───────────────────────────┘    │    │
│  └───────────────────────────────────┘    │
└───────────────────────────────────────────┘
```

### 第一层：模型接口层（Model Interface）

最内层，负责与模型 API 的直接交互：

```python
class ModelInterface:
    """模型接口层：封装对 LLM 的所有直接调用"""

    def __init__(self, api_key: str, model: str = "claude-sonnet"):
        self.client = APIClient(api_key)
        self.model = model

    def build_prompt(self, system: str, messages: list, tools: list = None) -> dict:
        """组装请求体：system prompt + 消息历史 + 工具定义"""
        return {
            "model": self.model,
            "system": system,
            "messages": messages,
            "tools": tools or []
        }

    def call(self, request: dict) -> ModelResponse:
        """发送 API 调用，返回结构化响应"""
        raw = self.client.post("/messages", json=request)
        return ModelResponse.parse(raw)

    def parse_structured(self, response: ModelResponse, schema: dict) -> dict:
        """将自由文本响应解析为结构化数据"""
        # JSON 提取、schema 校验等
        ...
```

**职责边界**：仅关注"如何与模型对话"，不关心业务逻辑或错误恢复。

### 第二层：运行时环境层（Runtime Environment）

中间层，提供可靠性和可观测性基础设施：

```python
class RuntimeEnvironment:
    """运行时环境层：提供可靠性保障"""

    def __init__(self, model_interface: ModelInterface, config: RuntimeConfig):
        self.model = model_interface
        self.config = config
        self.logger = Logger(config.log_level)
        self.metrics = MetricsCollector()

    def execute_with_retry(self, request: dict, validators: list) -> Result:
        """带重试和验证的模型调用"""
        for attempt in range(self.config.max_retries):
            try:
                response = self.model.call(request)
                result = self.model.parse_structured(response, self.config.schema)

                # 运行所有验证器
                validation_errors = []
                for validator in validators:
                    error = validator.check(result)
                    if error:
                        validation_errors.append(error)

                if not validation_errors:
                    self.metrics.record_success(attempt)
                    return Result.ok(result)

                self.logger.warn(f"验证失败 (尝试 {attempt+1}): {validation_errors}")

            except APIError as e:
                self.logger.error(f"API 错误 (尝试 {attempt+1}): {e}")
                self.metrics.record_error(e)

        return Result.fail("超过最大重试次数")

    def manage_state(self, session_id: str) -> StateManager:
        """会话状态管理"""
        ...
```

**职责边界**：关注"如何可靠地使用模型"，不关心具体业务流程。

### 第三层：编排层（Orchestration）

最外层，负责复杂任务的分解和多步骤协调：

```python
class Orchestrator:
    """编排层：任务分解与多步推理"""

    def __init__(self, runtime: RuntimeEnvironment, tools: list):
        self.runtime = runtime
        self.tools = {t.name: t for t in tools}

    def run_agent_loop(self, task: str) -> FinalResult:
        """主 agent 循环：规划 → 执行 → 观察 → 调整"""
        plan = self.create_plan(task)

        for step in plan.steps:
            if step.requires_tool:
                tool_result = self.tools[step.tool_name].execute(step.args)
                self.runtime.manage_state(plan.session_id).update(tool_result)

            result = self.runtime.execute_with_retry(
                request=self.build_step_request(step),
                validators=step.validators
            )

            if result.is_fail():
                plan = self.replan(plan, step, result.error)

        return self.synthesize(plan)
```

**职责边界**：关注"如何完成复杂任务"，依赖下层提供的可靠性保障。

---

## 2.3 Harness vs Framework vs Scaffold

这三个术语在社区中经常被混用，让我们严格区分它们：

### 定义对比

```
┌──────────────────────────────────────────────────────────────┐
│                      概念边界图                               │
│                                                              │
│  Framework (框架)                                            │
│  ├── 通用的代码库/SDK                                        │
│  ├── 提供构建 agent 的积木                                   │
│  └── 例：LangGraph, CrewAI, Anthropic Agent SDK               │
│                                                              │
│  Scaffold (脚手架)                                           │
│  ├── 项目模板或起步代码                                      │
│  ├── 帮你快速搭建结构，然后你自己填充逻辑                      │
│  └── 例：cookiecutter 模板, create-agent-app                 │
│                                                              │
│  Harness (驾驭系统)                                          │
│  ├── 围绕特定模型/任务的完整运行时                            │
│  ├── 包含控制逻辑、验证规则、编排策略                          │
│  ├── 是你为你的 agent 量身定制的"操作系统"                    │
│  └── 可能用 Framework 构建，可能从 Scaffold 起步              │
└──────────────────────────────────────────────────────────────┘
```

### 对比表

| 维度 | Framework | Scaffold | Harness |
|------|-----------|----------|---------|
| 本质 | 库/SDK | 模板/起步代码 | 运行时系统 |
| 生命周期 | 在开发时引用 | 在项目初始化时使用 | 在运行时持续工作 |
| 定制程度 | 通用，配置化使用 | 起步通用，逐步定制 | 高度定制，任务特定 |
| 类比 | React | create-react-app | 你的 Web 应用 |
| 谁提供 | 开源社区/云厂商 | 开源社区 | 你的工程团队 |
| 关注焦点 | 怎么调用模型 | 怎么开始项目 | 怎么可靠地完成任务 |

### 关键洞察

一个常见的误解是"使用了 LangChain 就等于有了 harness"。这就像说"使用了 React 就等于有了 Web 应用"。Framework 是工具，Harness 是你用这些工具（或不用）构建的、面向你的具体场景的完整解决方案。

---

## 2.4 一个最小 Harness 的完整实现

理论够了，让我们写一个真正可运行的最小 Harness。它包含 Harness 的五个核心组件：

```
最小 Harness 的五个组件：

1. Model Call    ─── 调用模型
2. Retry Logic   ─── 失败时重试
3. Validation    ─── 验证输出
4. Logging       ─── 记录行为
5. Fallback      ─── 降级策略
```

```python
"""
minimal_harness.py —— 一个最小但完整的 Harness

运行方式：python minimal_harness.py
依赖：无（使用模拟 LLM，演示 harness 结构）
"""
import json
import time
import random
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable

# ─── 配置 ───────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("harness")

@dataclass
class HarnessConfig:
    max_retries: int = 3
    retry_delay_base: float = 1.0   # 指数退避基数（秒）
    timeout: float = 10.0

# ─── 模拟 LLM ──────────────────────────────────────────

def simulated_llm(prompt: str) -> str:
    """模拟一个不可靠的 LLM，用于演示 harness 的价值"""
    roll = random.random()
    if roll < 0.10:
        raise ConnectionError("API 连接超时")
    elif roll < 0.20:
        return "这不是有效的 JSON 输出"
    elif roll < 0.30:
        return json.dumps({"result": None})  # 空结果
    else:
        return json.dumps({
            "result": "Python 3.12",
            "confidence": round(random.uniform(0.75, 0.99), 2)
        })

# ─── 验证器 ─────────────────────────────────────────────

def validate_response(data: dict) -> Optional[str]:
    """验证模型输出的结构和内容。返回 None 表示通过，返回字符串表示错误原因。"""
    if not isinstance(data, dict):
        return "输出不是字典类型"
    if "result" not in data:
        return "缺少 'result' 字段"
    if data["result"] is None:
        return "'result' 为空"
    if "confidence" in data and data["confidence"] < 0.5:
        return f"置信度过低: {data['confidence']}"
    return None

# ─── Harness 核心 ───────────────────────────────────────

@dataclass
class HarnessMetrics:
    total_calls: int = 0
    successful_calls: int = 0
    retries: int = 0
    fallbacks: int = 0
    errors: dict = field(default_factory=dict)

class MinimalHarness:
    """最小但完整的 Harness 实现"""

    def __init__(
        self,
        llm_fn: Callable,
        validator: Callable,
        config: HarnessConfig = None
    ):
        self.llm_fn = llm_fn
        self.validator = validator
        self.config = config or HarnessConfig()
        self.metrics = HarnessMetrics()

    def run(self, prompt: str) -> dict:
        """执行一次带 harness 保护的模型调用"""
        self.metrics.total_calls += 1

        for attempt in range(self.config.max_retries):
            try:
                # 1. Model Call
                logger.info(f"调用模型 (尝试 {attempt + 1}/{self.config.max_retries})")
                raw_response = self.llm_fn(prompt)

                # 2. 解析
                try:
                    data = json.loads(raw_response)
                except json.JSONDecodeError:
                    logger.warning(f"JSON 解析失败: {raw_response[:50]}...")
                    self._record_error("json_parse_error")
                    continue  # → 重试

                # 3. Validation
                error = self.validator(data)
                if error:
                    logger.warning(f"验证失败: {error}")
                    self._record_error(f"validation: {error}")
                    continue  # → 重试

                # 成功!
                logger.info(f"成功 (尝试 {attempt + 1})")
                self.metrics.successful_calls += 1
                return {"status": "success", "data": data, "attempts": attempt + 1}

            except Exception as e:
                logger.error(f"异常: {type(e).__name__}: {e}")
                self._record_error(type(e).__name__)

            # 4. Retry with backoff
            if attempt < self.config.max_retries - 1:
                delay = self.config.retry_delay_base * (2 ** attempt)
                logger.info(f"等待 {delay:.1f}s 后重试...")
                self.metrics.retries += 1
                time.sleep(delay)

        # 5. Fallback
        logger.warning("所有重试用尽，执行降级策略")
        self.metrics.fallbacks += 1
        return {"status": "fallback", "data": None, "message": "请求未能成功，请稍后重试"}

    def _record_error(self, error_type: str):
        self.metrics.errors[error_type] = self.metrics.errors.get(error_type, 0) + 1

    def report(self) -> str:
        """输出运行统计"""
        m = self.metrics
        return (
            f"\n{'='*50}\n"
            f"Harness 运行统计\n"
            f"{'='*50}\n"
            f"总调用次数:   {m.total_calls}\n"
            f"成功次数:     {m.successful_calls}\n"
            f"重试次数:     {m.retries}\n"
            f"降级次数:     {m.fallbacks}\n"
            f"成功率:       {m.successful_calls/max(m.total_calls,1)*100:.1f}%\n"
            f"错误分布:     {json.dumps(m.errors, ensure_ascii=False, indent=2)}\n"
        )

# ─── 运行演示 ───────────────────────────────────────────

if __name__ == "__main__":
    harness = MinimalHarness(
        llm_fn=simulated_llm,
        validator=validate_response,
        config=HarnessConfig(max_retries=3, retry_delay_base=0.1)  # 快速演示
    )

    print("运行 30 次调用...\n")
    for i in range(30):
        result = harness.run(f"查询 #{i+1}: 最新的 Python 版本是什么？")
        if result["status"] == "success":
            print(f"  #{i+1} ✓ {result['data']} (尝试 {result['attempts']} 次)")
        else:
            print(f"  #{i+1} ✗ 降级: {result['message']}")

    print(harness.report())
```

---

## 2.5 从手工到工程化的渐进升级路径

你不需要从零开始构建一个完整的 harness。下面是一条实践中验证过的渐进升级路径：

### 五级成熟度模型

```
Level 0        Level 1        Level 2        Level 3        Level 4
裸调用         基础防护       结构化控制     全面运行时     生产级系统
   │              │              │              │              │
   ▼              ▼              ▼              ▼              ▼
直接调用       + 重试          + 验证          + 编排          + 多 Agent
模型 API       + 超时          + 结构化输出    + 状态管理      + 可观测性
               + try/catch     + 日志          + 工具集成      + A/B 测试
                                               + 降级策略      + 自动扩缩
```

### 各级别的代码对比

**Level 0 — 裸调用**

```python
response = llm.chat("翻译成英文：你好世界")
print(response)
```

**Level 1 — 基础防护**

```python
for attempt in range(3):
    try:
        response = llm.chat("翻译成英文：你好世界")
        print(response)
        break
    except Exception:
        time.sleep(2 ** attempt)
```

**Level 2 — 结构化控制**

```python
response = llm.chat(
    system="你是专业翻译。只输出 JSON: {\"translation\": \"...\"}",
    user="翻译成英文：你好世界"
)
result = json.loads(response)
assert "translation" in result
logger.info(f"翻译完成: {result}")
```

**Level 3 — 全面运行时**（第 2.4 节的最小 Harness）

**Level 4 — 生产级系统**（将在后续章节详细展开）

### 何时升级

| 信号 | 建议动作 |
|------|---------|
| 偶尔报错但手动重试能解决 | 升级到 Level 1 |
| 输出格式不稳定，需要人工检查 | 升级到 Level 2 |
| 任务涉及多步骤或工具调用 | 升级到 Level 3 |
| 需要 SLA、监控告警、多人使用 | 升级到 Level 4 |

---

## 2.6 为什么 Harness 不是可选的

最后，让我们回答一个常见质疑："我的 agent 跑得挺好的，为什么要加 harness？"

答案是：**在 demo 中跑得好和在生产中跑得好是两件完全不同的事情。**

```
可靠性需求的金字塔

         /\
        /  \      ← 99.9% 可靠性（生产 SLA）
       / 安 \        需要: 完整 harness + 监控 + 降级
      / 全 性 \
     /────────\
    / 可观测性  \   ← 95% 可靠性（内部工具）
   /────────────\      需要: 验证 + 日志 + 重试
  / 可控性       \
 /────────────────\  ← 80% 可靠性（原型）
/    基本功能      \    裸调用可能就够了
──────────────────────
```

Harness 不是锦上添花，它是从原型到生产的必经之路。如同你不会在生产环境中运行没有错误处理的代码，你也不应该在生产环境中运行没有 harness 的 agent。

---

## 本章小结

| 概念 | 要点 |
|------|------|
| 操作系统类比 | Model=CPU, Context Window=RAM, Harness=OS, Agent=Application |
| 三层结构 | 模型接口层 → 运行时环境层 → 编排层，职责清晰、逐层构建 |
| Harness vs Framework | Framework 是工具库，Harness 是你用工具构建的运行时系统 |
| 最小 Harness | 五个核心组件：Model Call + Retry + Validation + Logging + Fallback |
| 成熟度模型 | Level 0-4 渐进升级，按需投入，不要过度工程 |

---

## 动手实验

### 实验 1：构建最小可工作的 Harness

将 2.4 节的 `minimal_harness.py` 保存并运行：

```bash
python minimal_harness.py
```

观察输出中的重试模式和最终统计。然后修改 `HarnessConfig` 的参数：

- 把 `max_retries` 改为 1，观察成功率下降
- 把 `max_retries` 改为 10，观察成功率上升但耗时增加
- 思考：重试次数与成本/延迟的权衡点在哪里？

### 实验 2：为 Harness 添加超时机制

在 `MinimalHarness.run()` 方法中添加总超时控制：

```python
import signal

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Harness 总执行超时")

# 在 run() 方法开头添加：
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(int(self.config.timeout))

# 在 run() 方法结尾（return 前）添加：
signal.alarm(0)  # 取消计时器
```

测试：将 timeout 设为 2 秒，retry_delay_base 设为 1.0，max_retries 设为 5，观察超时触发的情况。

### 实验 3：添加自定义验证器

创建一个新的验证器，检查模型输出的"情感极性"：

```python
def validate_sentiment(data: dict) -> Optional[str]:
    """确保情感分析结果在合理范围内"""
    score = data.get("sentiment_score")
    if score is None:
        return "缺少 sentiment_score"
    if not (-1.0 <= score <= 1.0):
        return f"sentiment_score 超出范围 [-1, 1]: {score}"
    if data.get("label") not in ("positive", "negative", "neutral"):
        return f"无效的情感标签: {data.get('label')}"
    return None
```

将 `simulated_llm` 修改为输出情感分析结果，然后用新验证器运行 harness。

---

## 练习题

### 基础题

1. 画出 Agent = Model + Harness 的操作系统类比图，标注每个组件的对应关系。
2. Harness 的三层结构中，哪一层负责重试逻辑？哪一层负责任务分解？
3. 用一句话区分 Framework 和 Harness。

### 实践题

4. 给 `MinimalHarness` 添加一个 `circuit_breaker` 功能：如果连续 5 次调用都失败，暂停 30 秒后再尝试（熔断器模式）。
5. 实现一个 `CachingHarness`，它在调用模型之前先检查缓存。如果相同的 prompt 在过去 5 分钟内已有成功结果，直接返回缓存。

### 思考题

6. 在成熟度模型中，从 Level 2 到 Level 3 的跳跃最大。为什么编排层的引入会显著增加复杂度？这对团队意味着什么？
7. 如果模型的能力在未来 2 年提升 10 倍，Harness 的哪些组件会变得不重要？哪些反而会变得更重要？
