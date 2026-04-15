# 第5章：Harness 工程师的技能图谱与工具链

> 知道了 Harness Engineering 是什么，下一个问题自然是：我需要学什么、用什么工具？本章为你画出完整的技能图谱和工具链全景，并规划一条从 0 到 1 的学习路径。

---

## 学习目标

学完本章，你将能够：

1. 理解 Harness Engineer 的核心技能是系统工程和 DevOps，而非 ML 研究
2. 按照技能树规划自己的学习路径
3. 了解当前 Harness Engineering 生态中的主流工具
4. 为团队中的 Harness Engineer 角色撰写职位描述
5. 搭建一个可用的最小 harness 开发环境

---

## 5.1 核心技能定位：你不需要成为 ML 研究员

### 一个重要的心理建设

很多工程师在面对 AI 工程时有一种"冒名顶替综合征"：觉得自己不懂 Transformer 的注意力机制、没读过论文、不会训练模型，就不配做 AI 相关的工作。

**这是一个误解。**

Harness Engineer 的核心技能画像：

```
Harness Engineer 技能雷达图

                    系统工程
                       ★★★★★
                      ╱      ╲
              DevOps ★★★★★    ★★★★☆ API 设计
                    ╱              ╲
          可观测性 ★★★★☆            ★★★★☆ 测试方法
                    ╲              ╱
              安全  ★★★☆☆    ★★★☆☆ ML 基础
                      ╲      ╱
                    Prompt 设计
                       ★★★☆☆

  ★★★★★ = 必须精通
  ★★★★☆ = 需要熟练
  ★★★☆☆ = 了解即可
```

### 三个核心技能

1. **系统工程思维**：理解分布式系统、状态管理、错误处理、并发控制。这是构建可靠 harness 的基础——你在构建的是一个持续运行的系统，不是一个脚本。

2. **DevOps 实践**：CI/CD、容器化、基础设施即代码、监控告警。Harness 需要被版本管理、自动测试、可靠部署，这与传统软件没有区别。

3. **API 设计能力**：模型通过 API 调用，工具通过 API 集成，agent 通过 API 对外提供服务。你的每一天都在与 API 打交道。

---

## 5.2 技能树：渐进学习路径

### 六级技能树

```
Level 6: 安全与合规
   │  prompt injection 防御、数据脱敏、合规审计
   │
Level 5: 评估方法论
   │  eval 设计、统计验证、A/B 测试、基准测试
   │
Level 4: 可观测性
   │  追踪、指标、日志、异常检测、成本监控
   │
Level 3: 容器化与编排
   │  Docker、Kubernetes、服务网格、API 网关
   │
Level 2: CI/CD 与自动化
   │  GitHub Actions、评估流水线、自动化部署
   │
Level 1: 编程基础
   │  Python / TypeScript、API 调用、JSON 处理
   │
Level 0: 理论基础 ← 你在这里（完成 Part 1 后）
      Harness 概念、Guides/Sensors、控制矩阵
```

### 各级详细内容

| 级别 | 核心技能 | 典型交付物 | 学习时间估计 |
|------|---------|-----------|------------|
| Level 0 | Harness 理论 | 能解释核心概念 | 1 周 |
| Level 1 | Python + API | 能写最小 harness | 2-4 周 |
| Level 2 | CI/CD | 评估流水线能跑起来 | 2-3 周 |
| Level 3 | 容器 / K8s | Agent 能容器化部署 | 3-4 周 |
| Level 4 | 可观测性 | 有监控仪表盘和告警 | 2-3 周 |
| Level 5 | 评估方法 | 能设计统计显著的 eval | 3-4 周 |
| Level 6 | 安全合规 | 通过安全审计 | 持续学习 |

### 跳板角色

不同背景的工程师有不同的学习捷径：

```
你的背景              已有技能               需要补充
──────────          ────────             ────────────

后端工程师           Level 1-3 大部分       Level 4-6 + AI 特有知识
DevOps 工程师        Level 2-4 大部分       Level 1 AI 部分 + Level 5-6
数据工程师           Level 1-2 部分         Level 3-6 + 系统设计
ML 工程师            Level 5 部分           Level 1-4 系统工程技能
前端工程师           Level 1 部分           Level 2-6 几乎全部
```

---

## 5.3 工具链生态一览

### 评估工具

| 工具 | 用途 | 特点 | 适用阶段 |
|------|------|------|---------|
| Braintrust | Agent 评估平台 | 在线评估、数据集管理、对比分析 | 开发到生产 |
| Promptfoo | 本地评估框架 | 开源、CLI 驱动、CI/CD 友好 | 开发和 CI |
| LangSmith | 追踪 + 评估 | 与 LangChain 集成好 | 开发到生产 |
| Inspect AI | Anthropic 评估框架 | 专注安全和对齐评估 | 研究和开发 |
| 自建 eval | 定制评估逻辑 | 最灵活，维护成本高 | 所有阶段 |

### 编排工具

| 工具 | 用途 | 特点 | 适用场景 |
|------|------|------|---------|
| Claude Agent SDK | Agent 构建 | Anthropic 官方、原生工具调用 | Claude 生态 |
| LangGraph | 有状态 agent 编排 | 图定义工作流、检查点（v1.1+） | 复杂多步任务 |
| CrewAI | 多 agent 协作 | 独立框架（v1.14+）、角色定义 | 多 agent 场景 |
| Temporal | 长时运行编排 | 持久化工作流、故障恢复 | 生产级长任务 |
| 自建编排 | 定制编排逻辑 | 完全控制，复杂度高 | 特殊需求 |

### 可观测性工具

| 工具 | 用途 | 特点 | 适用场景 |
|------|------|------|---------|
| Langfuse | LLM 可观测性 | 开源、追踪、成本分析（v4+） | 开发到生产 |
| Phoenix (Arize) | AI 可观测性 | 追踪、评估、漂移检测 | 生产监控 |
| Helicone | API 代理+分析 | 零代码集成、缓存、限流 | 快速接入 |
| OpenTelemetry | 通用追踪标准 | 标准化、厂商中立 | 企业级集成 |
| Prometheus + Grafana | 指标和仪表盘 | 成熟、生态丰富 | 基础设施监控 |

### 部署工具

| 工具 | 用途 | 特点 | 适用场景 |
|------|------|------|---------|
| Docker | 容器化 | 标准化运行环境 | 所有项目 |
| Kubernetes | 容器编排 | 自动扩缩、滚动更新 | 生产部署 |
| Modal | 无服务器 GPU | 按需启动、Python 原生 | 快速原型和推理 |
| Fly.io | 全球边缘部署 | 低延迟、简单部署 | API 服务 |
| AWS Bedrock / GCP Vertex | 云原生 AI 服务 | 企业级、合规、集成 | 企业部署 |

---

## 5.4 从 0 到 1 的学习路径建议

### 第一个月：建立基础

```
Week 1: 理论（你正在做的事情）
  ├── 完成本教程 Part 1
  ├── 阅读 Fowler 的 Guides/Sensors 文章
  └── 阅读 Anthropic 的 agent 架构文档

Week 2: 最小 harness
  ├── 用 Python 实现第 2 章的 MinimalHarness
  ├── 接入真实的 LLM API（Claude 或 GPT）
  └── 为一个简单任务（如文本分类）运行端到端

Week 3: 添加评估
  ├── 为你的 harness 编写 5 个评估用例
  ├── 用 Promptfoo 或自建脚本运行评估
  └── 记录基线分数

Week 4: 可观测性
  ├── 添加结构化日志（JSON 格式）
  ├── 搭建简单的仪表盘（可以用 Streamlit）
  └── 追踪成功率、延迟、token 消耗
```

### 第二个月：工程化

```
Week 5-6: CI/CD
  ├── 将评估集成到 GitHub Actions
  ├── 每次 PR 自动运行 eval
  └── 建立"不低于基线"的合并门槛

Week 7-8: 容器化和部署
  ├── 将 agent 容器化
  ├── 部署到云环境
  └── 实现基本的健康检查和告警
```

### 第三个月：进阶

```
Week 9-10: 多步编排
  ├── 实现工具调用（tool use）
  ├── 构建多步推理 agent
  └── 添加状态管理

Week 11-12: 生产就绪
  ├── 实现 A/B 测试框架
  ├── 添加 prompt injection 防御
  └── 建立 on-call 流程
```

---

## 5.5 团队结构：Harness 工程师在组织中的定位

### 三种组织模式

```
模式 A：嵌入式                模式 B：独立团队            模式 C：平台团队
(适合早期/小团队)              (适合中期/中团队)           (适合成熟/大团队)

┌──────────────┐           ┌──────────────┐          ┌──────────────┐
│ 产品团队 A   │           │ 产品团队 A   │          │ 产品团队 A   │
│ ├── 后端     │           │ ├── 后端     │          │ ├── 后端     │
│ ├── 前端     │           │ └── 前端     │          │ └── 前端     │
│ └── Harness ◄┤           └──────────────┘          └──────┬───────┘
│              │                    ▲                        │
│ 产品团队 B   │           ┌───────┴────────┐        ┌──────▼───────┐
│ ├── 后端     │           │ Harness 团队   │        │ Harness      │
│ ├── 前端     │           │ ├── Eng. 1    │        │ Platform     │
│ └── Harness ◄┤           │ ├── Eng. 2    │        │ (内部平台)    │
└──────────────┘           │ └── Eng. 3    │        │ ├── SDK      │
                           └───────┬────────┘        │ ├── 评估服务  │
每个产品团队自带             服务所有产品团队          │ ├── 监控服务  │
Harness 工程师              集中的专业能力            │ └── 部署服务  │
                                                    └──────────────┘
                                                    提供自助式平台
                                                    产品团队自主使用
```

### 角色定义

一个典型的 Harness Engineer 职位描述：

```
职位：Harness Engineer (AI Agent 工程师)

核心职责：
- 设计和构建 AI agent 的运行时系统（harness）
- 实现前馈控制（Guides）和反馈控制（Sensors）
- 构建和维护评估流水线（eval pipeline）
- 搭建 agent 可观测性基础设施
- 优化 agent 的可靠性、延迟和成本

必备技能：
- 3+ 年 Python 或 TypeScript 开发经验
- 熟练使用 CI/CD 工具（GitHub Actions 等）
- 了解容器化和 Kubernetes
- 了解大语言模型的能力和局限
- 优秀的系统设计能力

加分项：
- DevOps 或 Platform Engineering 背景
- 有构建 agent/LLM 应用的经验
- 熟悉评估方法论和统计测试
- 了解 AI 安全（prompt injection 等）

不需要：
- ML 研究经验
- 论文发表经历
- 模型训练经验
```

### 与其他角色的协作

```
                        ┌──────────────┐
                        │ 产品经理     │
                        │ 定义需求     │
                        └──────┬───────┘
                               │
                    ┌──────────▼──────────┐
                    │   Harness Engineer  │
                    │   设计和构建系统     │
                    └──┬──────┬───────┬───┘
                       │      │       │
              ┌────────▼┐  ┌──▼────┐  ┌▼────────┐
              │ ML Eng.  │  │DevOps │  │QA Eng.  │
              │ 提供模型 │  │ 部署  │  │ 测试    │
              └──────────┘  └───────┘  └─────────┘
```

---

## 5.6 最小开发环境搭建

结合前几章所学，让我们搭建一个最小但可用的 harness 开发环境。

### 项目结构

```
my-first-harness/
├── harness/
│   ├── __init__.py
│   ├── core.py          # Harness 核心（第 2 章的最小实现）
│   ├── guides.py        # Guide 集合（第 3 章）
│   ├── sensors.py       # Sensor 集合（第 3 章）
│   └── config.py        # 配置管理
├── evals/
│   ├── __init__.py
│   ├── datasets/        # 评估数据集
│   │   └── basic.jsonl
│   └── run_eval.py      # 评估脚本
├── tests/
│   └── test_harness.py  # 单元测试
├── pyproject.toml
├── Dockerfile
└── .github/
    └── workflows/
        └── eval.yml     # CI 评估流水线
```

### 快速搭建脚本

```python
"""
setup_dev_env.py —— 搭建最小 harness 开发环境

运行方式：python setup_dev_env.py
"""
import os
import json

PROJECT_NAME = "my-first-harness"

# 目录结构
dirs = [
    f"{PROJECT_NAME}/harness",
    f"{PROJECT_NAME}/evals/datasets",
    f"{PROJECT_NAME}/tests",
    f"{PROJECT_NAME}/.github/workflows",
]

# 文件内容
files = {
    f"{PROJECT_NAME}/harness/__init__.py": "",

    f"{PROJECT_NAME}/harness/config.py": '''"""Harness 配置"""
from dataclasses import dataclass

@dataclass
class HarnessConfig:
    model: str = "claude-sonnet"
    max_retries: int = 3
    retry_delay_base: float = 1.0
    timeout: float = 30.0
    log_level: str = "INFO"
''',

    f"{PROJECT_NAME}/harness/guides.py": '''"""Guide 集合：前馈控制"""

SYSTEM_PROMPT = """你是一个专业的 AI 助手。
- 以 JSON 格式输出
- 如果不确定，说明不确定性
- 不要编造信息
"""

def build_prompt(system: str, user_message: str, examples: list = None) -> list:
    """组装完整的消息列表"""
    messages = []
    if examples:
        messages.extend(examples)
    messages.append({"role": "user", "content": user_message})
    return messages
''',

    f"{PROJECT_NAME}/harness/sensors.py": '''"""Sensor 集合：反馈控制"""
import json
from typing import Optional

def json_sensor(text: str) -> dict:
    """检查输出是否是合法 JSON"""
    try:
        data = json.loads(text)
        return {"passed": True, "data": data}
    except json.JSONDecodeError as e:
        return {"passed": False, "error": str(e)}

def length_sensor(text: str, min_len: int = 10, max_len: int = 5000) -> dict:
    """检查输出长度是否在合理范围内"""
    length = len(text)
    passed = min_len <= length <= max_len
    return {
        "passed": passed,
        "length": length,
        "error": None if passed else f"长度 {length} 不在 [{min_len}, {max_len}] 范围内"
    }

def safety_sensor(text: str) -> dict:
    """基础安全检查（实际项目中应更完善）"""
    dangerous_patterns = ["rm -rf", "DROP TABLE", "eval(", "<script>"]
    found = [p for p in dangerous_patterns if p.lower() in text.lower()]
    return {
        "passed": len(found) == 0,
        "violations": found
    }
''',

    f"{PROJECT_NAME}/harness/core.py": '''"""Harness 核心"""
import json
import time
import logging
from typing import Callable, Optional
from .config import HarnessConfig
from .sensors import json_sensor, length_sensor, safety_sensor

logger = logging.getLogger("harness")

class Harness:
    def __init__(self, llm_fn: Callable, config: HarnessConfig = None):
        self.llm_fn = llm_fn
        self.config = config or HarnessConfig()
        self.sensors = [json_sensor, length_sensor, safety_sensor]

    def run(self, messages: list, system: str = "") -> dict:
        """带保护的模型调用"""
        for attempt in range(self.config.max_retries):
            try:
                response = self.llm_fn(messages=messages, system=system)

                # 运行 sensors
                all_passed = True
                sensor_results = {}
                for sensor in self.sensors:
                    result = sensor(response)
                    sensor_results[sensor.__name__] = result
                    if not result["passed"]:
                        all_passed = False

                if all_passed:
                    return {"status": "success", "response": response, "attempts": attempt + 1}

                logger.warning(f"Sensor 检查未通过 (尝试 {attempt+1}): {sensor_results}")

            except Exception as e:
                logger.error(f"调用失败 (尝试 {attempt+1}): {e}")

            if attempt < self.config.max_retries - 1:
                time.sleep(self.config.retry_delay_base * (2 ** attempt))

        return {"status": "fallback", "response": None, "message": "所有重试用尽"}
''',

    f"{PROJECT_NAME}/evals/datasets/basic.jsonl": '\n'.join([
        json.dumps({"input": "1+1等于多少？", "expected_contains": "2"}, ensure_ascii=False),
        json.dumps({"input": "Python 的创造者是谁？", "expected_contains": "Guido"}, ensure_ascii=False),
        json.dumps({"input": "HTTP 200 状态码的含义？", "expected_contains": "成功"}, ensure_ascii=False),
    ]),

    f"{PROJECT_NAME}/evals/run_eval.py": '''"""简单的评估脚本"""
import json
import sys
sys.path.insert(0, "..")
from harness.core import Harness
from harness.config import HarnessConfig

def mock_llm(messages, system=""):
    """模拟 LLM（替换为真实 API 调用）"""
    return json.dumps({"answer": "这是一个模拟回答"})

def run_eval():
    harness = Harness(llm_fn=mock_llm)

    with open("datasets/basic.jsonl") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    passed = 0
    for case in cases:
        result = harness.run(
            messages=[{"role": "user", "content": case["input"]}]
        )
        if result["status"] == "success":
            if case["expected_contains"] in result["response"]:
                passed += 1
                print(f"  PASS: {case['input']}")
            else:
                print(f"  FAIL: {case['input']} (未包含期望内容)")
        else:
            print(f"  ERROR: {case['input']}")

    total = len(cases)
    print(f"\\n结果: {passed}/{total} ({passed/total*100:.0f}%)")

if __name__ == "__main__":
    run_eval()
''',

    f"{PROJECT_NAME}/tests/test_harness.py": '''"""Harness 单元测试"""
import json
from harness.sensors import json_sensor, length_sensor, safety_sensor

def test_json_sensor_valid():
    result = json_sensor(\'{"key": "value"}\')
    assert result["passed"] is True

def test_json_sensor_invalid():
    result = json_sensor("not json")
    assert result["passed"] is False

def test_length_sensor_valid():
    result = length_sensor("a" * 100, min_len=10, max_len=200)
    assert result["passed"] is True

def test_length_sensor_too_short():
    result = length_sensor("hi", min_len=10)
    assert result["passed"] is False

def test_safety_sensor_safe():
    result = safety_sensor("这是一段安全的文本")
    assert result["passed"] is True

def test_safety_sensor_unsafe():
    result = safety_sensor("执行 rm -rf / 来清理磁盘")
    assert result["passed"] is False

if __name__ == "__main__":
    tests = [
        test_json_sensor_valid,
        test_json_sensor_invalid,
        test_length_sensor_valid,
        test_length_sensor_too_short,
        test_safety_sensor_safe,
        test_safety_sensor_unsafe,
    ]
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
        except AssertionError as e:
            print(f"  FAIL: {test.__name__}: {e}")
    print(f"\\n{len(tests)} tests completed")
''',

    f"{PROJECT_NAME}/pyproject.toml": '''[project]
name = "my-first-harness"
version = "0.1.0"
description = "一个最小的 Harness Engineering 项目"
requires-python = ">=3.10"
dependencies = []

[project.optional-dependencies]
dev = ["pytest", "ruff"]
eval = ["anthropic"]
''',

    f"{PROJECT_NAME}/Dockerfile": '''FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install -e ".[dev]"
CMD ["python", "-m", "pytest", "tests/"]
''',

    f"{PROJECT_NAME}/.github/workflows/eval.yml": '''name: Eval Pipeline
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -v
      - run: cd evals && python run_eval.py
''',
}

# 创建目录和文件
def setup():
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"  目录: {d}/")

    for path, content in files.items():
        with open(path, "w") as f:
            f.write(content)
        print(f"  文件: {path}")

    print(f"\n开发环境创建完成！")
    print(f"\n快速开始：")
    print(f"  cd {PROJECT_NAME}")
    print(f"  pip install -e '.[dev]'")
    print(f"  python tests/test_harness.py")
    print(f"  cd evals && python run_eval.py")

if __name__ == "__main__":
    print("搭建 Harness 开发环境...\n")
    setup()
```

---

## 本章小结

| 概念 | 要点 |
|------|------|
| 核心技能 | 系统工程 + DevOps + API 设计，不是 ML 研究 |
| 技能树 | 6 级：理论 → 编程 → CI/CD → 容器 → 可观测性 → 评估 → 安全 |
| 工具生态 | 四大类：评估、编排、可观测性、部署，各有主流选择 |
| 学习路径 | 3 个月从零到可部署：第 1 月基础、第 2 月工程化、第 3 月进阶 |
| 团队定位 | 三种模式（嵌入式 / 独立团队 / 平台团队），按公司阶段选择 |
| 开发环境 | 5 个文件即可起步：core + guides + sensors + config + eval |

---

## 动手实验

### 实验 1：搭建最小开发环境

运行本章的 `setup_dev_env.py` 脚本：

```bash
python setup_dev_env.py
cd my-first-harness
python tests/test_harness.py
```

确认所有测试通过。然后：

1. 在 `sensors.py` 中添加一个新的 sensor（例如：检查输出是否包含特定关键词）
2. 在 `tests/test_harness.py` 中为新 sensor 添加测试
3. 运行测试确认通过

### 实验 2：接入真实 LLM API

将 `evals/run_eval.py` 中的 `mock_llm` 替换为真实的 API 调用：

```python
import anthropic

client = anthropic.Anthropic()  # 使用环境变量 ANTHROPIC_API_KEY

def real_llm(messages, system=""):
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=system,
        messages=messages
    )
    return response.content[0].text
```

运行评估并观察：
- 真实 LLM 的输出是否通过所有 sensor？
- 如果不通过，是 sensor 的问题还是 LLM 的问题？
- 调整 sensor 或 guide，让通过率达到 100%

### 实验 3：绘制你的个人技能图谱

```python
"""
实验 3：评估你当前的技能水平
对每项技能打分（0-5），识别需要优先提升的领域
"""

skills = {
    "Python 编程":       0,  # 你的评分
    "TypeScript":        0,
    "API 设计":          0,
    "CI/CD":             0,
    "Docker / K8s":      0,
    "监控和可观测性":     0,
    "Prompt 设计":       0,
    "LLM 基础知识":      0,
    "测试方法论":        0,
    "系统设计":          0,
    "安全意识":          0,
    "统计基础":          0,
}

print("你的 Harness Engineer 技能评估")
print("=" * 50)

for skill, score in skills.items():
    bar = "█" * score + "░" * (5 - score)
    print(f"  {skill:<20} [{bar}] {score}/5")

total = sum(skills.values())
max_total = len(skills) * 5
print(f"\n总分: {total}/{max_total} ({total/max_total*100:.0f}%)")

# 建议优先级
priority = sorted(skills.items(), key=lambda x: x[1])[:3]
print("\n建议优先提升：")
for skill, score in priority:
    print(f"  - {skill} (当前 {score}/5)")
```

---

## 练习题

### 基础题

1. Harness Engineer 最重要的三个核心技能是什么？为什么 ML 研究不在列表中？
2. 从技能树的 Level 0 到 Level 2，你需要掌握哪些具体技能？
3. 在可观测性工具中，LangFuse 和 OpenTelemetry 的主要区别是什么？

### 实践题

4. 为你的团队/公司写一份 Harness Engineer 的职位描述（JD）。根据你们的实际技术栈调整工具要求。
5. 使用本章的项目模板，构建一个完整的"文本情感分析"harness，包括：自定义 system prompt（Guide）、情感分数范围验证（Sensor）、5 个评估用例。

### 思考题

6. 如果你是一个有 5 年经验的后端工程师，你认为转型为 Harness Engineer 最大的挑战是什么？最大的优势呢？
7. 随着 Harness Engineering 工具链的成熟，你认为哪些手工工作会被工具化？哪些需要持续的人类判断？
