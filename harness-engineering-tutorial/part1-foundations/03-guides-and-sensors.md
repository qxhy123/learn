# 第3章：Guides 与 Sensors：前馈控制与反馈控制

> 你如何确保一个非确定性系统做正确的事？答案来自控制论：用前馈控制预防问题，用反馈控制检测并修正问题。本章深入 Martin Fowler 的分类框架，让你掌握 harness 设计中最重要的结构性思维工具。

---

## 学习目标

学完本章，你将能够：

1. 用控制论术语解释 Guides（前馈控制）和 Sensors（反馈控制）的本质区别
2. 列举并设计至少 5 种 Guide 和 5 种 Sensor
3. 区分计算型控制与推理型控制，并理解各自的适用场景
4. 运用控制矩阵分析现有 harness 的完整性
5. 为给定任务设计一套完整的 Guide + Sensor 组合方案

---

## 3.1 控制论视角：为什么需要两种控制

### 从工程控制论出发

控制论（Cybernetics）是研究系统如何通过信息和反馈进行自我调节的学科。任何控制系统都有两种基本策略：

```
前馈控制（Feedforward）           反馈控制（Feedback）
───────────────────────        ──────────────────────

    参考信号                         参考信号
       │                               │
       ▼                               ▼
  ┌──────────┐                    ┌──────────┐
  │ 控制器   │──→ 系统 ──→ 输出   │ 控制器   │──→ 系统 ──→ 输出
  └──────────┘                    └──────────┘         │
       ▲                               ▲              │
       │                               └──────────────┘
  已知的扰动                            测量输出
  (预测并补偿)                          (检测并修正)
```

- **前馈控制**：在系统运行之前，根据已知信息预先设定约束，防止错误发生
- **反馈控制**：在系统运行之后，检测输出是否符合预期，发现偏差后修正

### 对应到 AI Agent

在 AI agent 系统中：

| 控制论概念 | Harness 对应 | 时机 | 目标 |
|-----------|-------------|------|------|
| 前馈控制 | **Guides** | 模型调用之前 | 引导模型往正确方向走 |
| 反馈控制 | **Sensors** | 模型调用之后 | 检测输出是否正确 |

Martin Fowler 在 2026 年 4 月的文章中正式提出这个分类，将 harness 的各种组件归入这两个类别，为 Harness Engineering 提供了清晰的组织框架。

---

## 3.2 Guides：前馈控制的武器库

Guides 是你在模型执行前注入的所有约束和指导。它们的目标是**让模型一次就做对**。

### 六种核心 Guide

```
┌───────────────────────────────────────────────────┐
│                 Guides 全景图                      │
├───────────────────────────────────────────────────┤
│                                                   │
│  1. System Prompt         ← 角色定义和行为规范     │
│  2. Few-shot Examples     ← 用示例示范期望输出     │
│  3. Structured Output     ← 约束输出格式          │
│  4. AGENTS.md / Rules     ← 项目级规则文件         │
│  5. Architectural Rules   ← 代码/架构约束          │
│  6. Context Curation      ← 精心筛选的上下文       │
│                                                   │
└───────────────────────────────────────────────────┘
```

#### Guide 1：System Prompt

最基础的 Guide，定义模型的角色和行为边界：

```python
SYSTEM_PROMPT = """你是一个严格的代码审查助手。

## 行为规则
- 只关注代码质量问题，不讨论功能需求
- 每个问题必须引用具体的行号
- 严重程度分为 critical / warning / info 三级
- 如果代码没有问题，明确回复 "LGTM"
- 绝对不要编造不存在的问题

## 输出格式
必须使用 JSON 格式，schema 如下：
{
  "issues": [
    {"line": <int>, "severity": "<str>", "message": "<str>", "suggestion": "<str>"}
  ]
}
"""
```

#### Guide 2：Few-shot Examples

通过示例让模型理解你的期望：

```python
FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": "审查以下代码：\ndef add(a, b):\n    return a + b"
    },
    {
        "role": "assistant",
        "content": '{"issues": [{"line": 1, "severity": "warning", "message": "函数缺少类型注解", "suggestion": "def add(a: int, b: int) -> int:"}]}'
    },
    {
        "role": "user",
        "content": "审查以下代码：\ndef greet(name: str) -> str:\n    return f\"Hello, {name}!\""
    },
    {
        "role": "assistant",
        "content": '{"issues": []}'  # 示范"没问题"的情况
    }
]
```

#### Guide 3：Structured Output（结构化输出约束）

要求模型以特定格式输出，减少解析错误：

```python
# 方法一：JSON Schema 约束
response = client.messages.create(
    model="claude-sonnet",
    system=SYSTEM_PROMPT,
    messages=messages,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "code_review",
            "schema": {
                "type": "object",
                "properties": {
                    "issues": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "line": {"type": "integer"},
                                "severity": {"enum": ["critical", "warning", "info"]},
                                "message": {"type": "string"},
                                "suggestion": {"type": "string"}
                            },
                            "required": ["line", "severity", "message"]
                        }
                    }
                },
                "required": ["issues"]
            }
        }
    }
)
```

#### Guide 4：AGENTS.md / Rules 文件

项目级别的行为规则文件，自动注入到 system prompt：

```markdown
# AGENTS.md

## 代码风格
- 使用 black 格式化，行宽 88
- 所有函数必须有 docstring
- 使用 type hints

## 架构规则
- 不要直接导入内部模块，使用公共 API
- 数据库操作必须通过 repository 层
- 所有外部调用必须有超时设置

## 禁止事项
- 不要使用 eval() 或 exec()
- 不要在代码中硬编码密钥
- 不要忽略异常（bare except）
```

#### Guide 5：Architectural Rules（架构规则）

编码到 harness 中的硬性约束：

```python
ARCHITECTURE_RULES = {
    "max_file_changes": 5,         # 单次最多修改 5 个文件
    "forbidden_imports": ["os.system", "subprocess.call"],
    "required_patterns": ["try/except for I/O"],
    "max_function_length": 50,     # 函数最多 50 行
}
```

#### Guide 6：Context Curation（上下文精选）

不是把所有东西塞进上下文窗口，而是精心筛选最相关的信息：

```python
def curate_context(task: str, codebase: Codebase) -> str:
    """为特定任务精选上下文"""
    relevant_files = codebase.search(task, top_k=5)
    dependency_graph = codebase.get_dependencies(relevant_files)
    recent_changes = codebase.git_log(relevant_files, n=3)
    coding_standards = codebase.read("AGENTS.md")

    return assemble_context(
        files=relevant_files,
        deps=dependency_graph,
        history=recent_changes,
        standards=coding_standards
    )
```

---

## 3.3 Sensors：反馈控制的检测网

Sensors 是你在模型输出后运行的所有检测和校验。它们的目标是**发现错误并触发修正**。

### 六种核心 Sensor

```
┌───────────────────────────────────────────────────┐
│                 Sensors 全景图                     │
├───────────────────────────────────────────────────┤
│                                                   │
│  1. Linter / 静态分析    ← 代码语法和风格检查      │
│  2. Type Checker         ← 类型正确性验证          │
│  3. Test Runner          ← 运行测试验证行为        │
│  4. Schema Validator     ← 输出格式校验            │
│  5. LLM-as-Judge         ← 用另一个 LLM 评估质量  │
│  6. Human Review         ← 人类审核和反馈          │
│                                                   │
└───────────────────────────────────────────────────┘
```

#### Sensor 1：Linter / 静态分析

最快、最便宜的 Sensor——对代码生成任务几乎是必须的：

```python
import subprocess

def lint_sensor(code: str) -> dict:
    """对生成的代码运行 linter"""
    with open("/tmp/generated_code.py", "w") as f:
        f.write(code)

    result = subprocess.run(
        ["ruff", "check", "/tmp/generated_code.py", "--output-format", "json"],
        capture_output=True, text=True
    )

    issues = json.loads(result.stdout) if result.stdout else []
    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "fix_hint": "请修复以下 lint 错误并重新生成代码"
    }
```

#### Sensor 2：Type Checker

比 linter 更深层的语义检查：

```python
def type_check_sensor(code: str) -> dict:
    """对生成的代码运行类型检查"""
    with open("/tmp/generated_code.py", "w") as f:
        f.write(code)

    result = subprocess.run(
        ["mypy", "/tmp/generated_code.py", "--no-error-summary"],
        capture_output=True, text=True
    )

    errors = [line for line in result.stdout.split("\n") if "error:" in line]
    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "fix_hint": "请修复类型错误"
    }
```

#### Sensor 3：Test Runner

最可靠的 Sensor——如果测试通过，代码很可能正确：

```python
def test_runner_sensor(code: str, test_file: str) -> dict:
    """运行测试来验证生成的代码"""
    with open("/tmp/generated_code.py", "w") as f:
        f.write(code)

    result = subprocess.run(
        ["pytest", test_file, "-v", "--tb=short"],
        capture_output=True, text=True
    )

    return {
        "passed": result.returncode == 0,
        "output": result.stdout,
        "fix_hint": f"以下测试失败，请根据错误信息修复代码:\n{result.stdout}"
    }
```

#### Sensor 4：Schema Validator

验证结构化输出的格式：

```python
from jsonschema import validate, ValidationError

def schema_sensor(data: dict, schema: dict) -> dict:
    """验证输出是否符合 JSON Schema"""
    try:
        validate(instance=data, schema=schema)
        return {"passed": True}
    except ValidationError as e:
        return {
            "passed": False,
            "error": str(e.message),
            "path": list(e.path),
            "fix_hint": f"输出不符合 schema: {e.message}"
        }
```

#### Sensor 5：LLM-as-Judge

用另一个 LLM 来评估输出质量——当没有确定性验证方法时的选择：

```python
def llm_judge_sensor(original_task: str, output: str) -> dict:
    """用 LLM 评估另一个 LLM 的输出质量"""
    judge_prompt = f"""评估以下 AI 输出的质量。

原始任务：{original_task}

AI 输出：{output}

评估标准：
1. 正确性：输出是否准确回答了任务？(1-5)
2. 完整性：是否覆盖了所有要求？(1-5)
3. 安全性：是否存在有害内容？(pass/fail)

以 JSON 格式输出评估结果。"""

    judge_response = judge_llm.chat(judge_prompt)
    scores = json.loads(judge_response)

    passed = (
        scores["correctness"] >= 3
        and scores["completeness"] >= 3
        and scores["safety"] == "pass"
    )
    return {"passed": passed, "scores": scores}
```

#### Sensor 6：Human Review

最终的安全网——某些高风险场景必须有人类参与：

```python
def human_review_sensor(output: str, risk_level: str) -> dict:
    """根据风险等级决定是否需要人工审核"""
    if risk_level == "low":
        return {"passed": True, "review_required": False}

    # 高风险任务：暂停并请求人工审核
    review_id = submit_for_review(output)
    return {
        "passed": None,  # 待定
        "review_required": True,
        "review_id": review_id,
        "message": "已提交人工审核，等待结果..."
    }
```

---

## 3.4 计算型控制 vs 推理型控制

Fowler 的分类还有第二个维度：控制手段本身是**确定性的计算**还是**非确定性的推理**。

### 两种控制的本质区别

```
计算型控制 (Computational)         推理型控制 (Inferential)
─────────────────────────        ────────────────────────

  确定性                           非确定性
  基于规则                         基于判断
  CPU-bound                       GPU-bound
  快速、便宜                       慢速、昂贵
  100% 精确（在其范围内）          有概率出错

  例：                             例：
  - JSON Schema 校验               - LLM-as-Judge
  - 正则表达式匹配                  - 语义相似度检查
  - 单元测试                       - 意图分类
  - AST 分析                       - 内容安全评估
```

### 选择原则

> **能用计算型控制的地方，不要用推理型控制。**

这是 Harness Engineering 的一条核心原则。原因很简单：

1. 计算型控制更快、更便宜、更可靠
2. 推理型控制本身也可能出错（你用来检查的 LLM 也是非确定性的）
3. 计算型控制的结果是可重复的、可解释的

但推理型控制也是必不可少的——很多质量维度（如"回答是否有帮助""代码是否符合设计意图"）无法用确定性规则检查。

---

## 3.5 控制矩阵

将两个维度交叉，得到 Harness Engineering 的**控制矩阵**：

```
                 ┌──────────────────┬──────────────────┐
                 │   Computational  │   Inferential    │
                 │   (计算型)        │   (推理型)        │
    ┌────────────┼──────────────────┼──────────────────┤
    │ Guide      │ ① 结构化输出     │ ② Few-shot       │
    │ (前馈)     │    JSON Schema   │    Examples       │
    │            │    正则约束       │    System Prompt  │
    │            │    架构规则       │    AGENTS.md      │
    │            │    白名单/黑名单  │    角色设定        │
    ├────────────┼──────────────────┼──────────────────┤
    │ Sensor     │ ③ Linter         │ ④ LLM-as-Judge   │
    │ (反馈)     │    Type Checker  │    语义相似度     │
    │            │    Test Runner   │    意图验证       │
    │            │    Schema 校验   │    安全审查       │
    │            │    AST 分析      │    人工审核       │
    └────────────┴──────────────────┴──────────────────┘
```

### 四个象限的特点

| 象限 | 特点 | 优先级 |
|------|------|-------|
| ① Guide + Computational | 最可靠的预防手段，成本最低 | 最高 |
| ② Guide + Inferential | 灵活但不确定，依赖 prompt 质量 | 高 |
| ③ Sensor + Computational | 最可靠的检测手段，应最大化使用 | 高 |
| ④ Sensor + Inferential | 最灵活但最不可靠，成本也最高 | 中 |

### 设计原则

1. **先 ①，再 ③，然后 ②，最后 ④**：尽量用确定性手段解决问题
2. **Guide 优先于 Sensor**：预防比修正成本更低
3. **每个象限至少一个**：不要只依赖单一类型的控制
4. **推理型控制要有确定性兜底**：LLM-as-Judge 的结果要有计算型二次校验

---

## 3.6 为什么两种控制都不可或缺

### 只有 Guides 的问题

如果只依赖前馈控制：

```python
# 只有 Guides：精心设计 prompt 但不验证输出
response = llm.chat(
    system="你必须输出合法的 JSON...(详尽的 guide)",
    user=task
)
# 直接使用 response，不做任何检查
result = json.loads(response)  # 可能崩溃！
process(result)                # 可能处理错误数据！
```

问题：
- **无法确认 guide 是否生效**——模型可能忽略你的指令
- **对新类型的错误毫无防备**——你无法预见所有可能的失败模式
- **没有闭环**——错误会静默传播

### 只有 Sensors 的问题

如果只依赖反馈控制：

```python
# 只有 Sensors：不给指导，只靠检查和重试
for attempt in range(10):
    response = llm.chat(user=task)  # 没有 system prompt，没有 few-shot
    if passes_all_checks(response):
        break
# 可能重试 10 次才成功...或者 10 次都失败
```

问题：
- **浪费资源**——模型在黑暗中摸索，大量重试
- **重复犯同样的错误**——没有指导，模型不知道为什么失败
- **延迟高**——每次重试都是一次完整的 API 调用

### 正确的做法：两者结合

```python
# Guides + Sensors 协同工作
response = llm.chat(
    system=detailed_system_prompt,         # Guide: 角色和规则
    messages=few_shot_examples + [task],   # Guide: 示例
    response_format=json_schema            # Guide: 格式约束
)

# Sensor: 多层验证
parsed = json.loads(response)              # Sensor: 格式校验
validate(parsed, schema)                   # Sensor: Schema 校验
lint_result = run_linter(parsed["code"])   # Sensor: 静态分析
test_result = run_tests(parsed["code"])    # Sensor: 测试验证

if not all_passed:
    # 将 sensor 的反馈作为新的 guide 注入
    response = llm.chat(
        messages=[..., {"role": "user", "content": f"修复这些问题: {errors}"}]
    )
```

关键洞察：**Sensor 的输出可以转化为下一轮的 Guide**——这就形成了闭环控制。

---

## 本章小结

| 概念 | 要点 |
|------|------|
| Guides（前馈控制） | 模型调用前注入的约束：prompt、示例、schema、规则 |
| Sensors（反馈控制） | 模型调用后的检测：linter、测试、LLM 评估、人工审核 |
| 计算型控制 | 确定性、快速、便宜、可靠——优先使用 |
| 推理型控制 | 非确定性、灵活、昂贵——不可避免但需谨慎 |
| 控制矩阵 | {Guide, Sensor} x {Computational, Inferential} 四象限 |
| 设计原则 | 两种控制必须结合；Sensor 反馈可转化为下一轮 Guide |

---

## 动手实验

### 实验 1：为代码生成任务设计完整的 Guide 集

为以下任务设计尽可能完善的 Guide 集合：

> 任务：让 AI agent 生成一个 Python 函数，该函数接收一个 CSV 文件路径，返回按指定列排序后的 DataFrame。

```python
"""
实验 1：设计 Guide 集合
"""

# Guide 1: System Prompt
SYSTEM_PROMPT = """你是一个 Python 数据处理专家。
- 只使用 pandas 库
- 函数必须有完整的 type hints
- 必须处理文件不存在的情况
- 必须处理指定列不存在的情况
- 输出格式：纯 Python 代码，不要 markdown 标记
"""

# Guide 2: Few-shot Example
EXAMPLE_INPUT = "读取 users.csv 并按 age 列降序排序"
EXAMPLE_OUTPUT = '''import pandas as pd
from pathlib import Path

def sort_csv(file_path: str, sort_column: str, ascending: bool = True) -> pd.DataFrame:
    """读取 CSV 文件并按指定列排序。

    Args:
        file_path: CSV 文件路径
        sort_column: 排序列名
        ascending: 是否升序，默认 True

    Returns:
        排序后的 DataFrame

    Raises:
        FileNotFoundError: 文件不存在
        KeyError: 指定列不存在
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    df = pd.read_csv(file_path)

    if sort_column not in df.columns:
        raise KeyError(f"列 '{sort_column}' 不存在。可用列: {list(df.columns)}")

    return df.sort_values(by=sort_column, ascending=ascending)
'''

# Guide 3: 架构约束
CONSTRAINTS = {
    "allowed_imports": ["pandas", "pathlib"],
    "required_elements": ["type hints", "docstring", "error handling"],
    "forbidden_patterns": ["exec(", "eval(", "import os"],
    "max_lines": 40
}

print("Guide 设计完成。请思考：")
print("1. 这些 Guide 是否覆盖了所有可能的问题？")
print("2. 哪些问题只能通过 Sensor 来检测？")
```

### 实验 2：为同一任务设计完整的 Sensor 链

```python
"""
实验 2：设计 Sensor 链
"""
import subprocess
import json

def sensor_chain(generated_code: str) -> dict:
    """对生成的代码运行完整的 sensor 链"""
    results = {}

    # Sensor 1: 语法检查（Computational）
    try:
        compile(generated_code, "<generated>", "exec")
        results["syntax"] = {"passed": True}
    except SyntaxError as e:
        results["syntax"] = {"passed": False, "error": str(e)}
        return results  # 语法都不对，后续检查没意义

    # Sensor 2: 导入检查（Computational）
    forbidden = ["import os", "import subprocess", "eval(", "exec("]
    violations = [f for f in forbidden if f in generated_code]
    results["import_check"] = {
        "passed": len(violations) == 0,
        "violations": violations
    }

    # Sensor 3: 结构检查（Computational）
    has_type_hints = "->" in generated_code and ":" in generated_code
    has_docstring = '"""' in generated_code or "'''" in generated_code
    has_error_handling = "try" in generated_code or "raise" in generated_code
    results["structure"] = {
        "passed": all([has_type_hints, has_docstring, has_error_handling]),
        "type_hints": has_type_hints,
        "docstring": has_docstring,
        "error_handling": has_error_handling
    }

    # Sensor 4: 测试运行（Computational）
    test_code = '''
import tempfile, csv
# 创建测试 CSV
with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
    writer = csv.writer(f)
    writer.writerow(["name", "age", "score"])
    writer.writerow(["Alice", 30, 95])
    writer.writerow(["Bob", 25, 87])
    test_file = f.name

# 测试正常排序
result = sort_csv(test_file, "age")
assert list(result["age"]) == [25, 30], "排序结果不正确"

# 测试文件不存在
try:
    sort_csv("/nonexistent.csv", "age")
    assert False, "应该抛出 FileNotFoundError"
except FileNotFoundError:
    pass

# 测试列不存在
try:
    sort_csv(test_file, "nonexistent_column")
    assert False, "应该抛出 KeyError"
except KeyError:
    pass

print("所有测试通过!")
'''
    full_code = generated_code + "\n\n" + test_code
    with open("/tmp/test_generated.py", "w") as f:
        f.write(full_code)

    proc = subprocess.run(
        ["python", "/tmp/test_generated.py"],
        capture_output=True, text=True, timeout=10
    )
    results["tests"] = {
        "passed": proc.returncode == 0,
        "output": proc.stdout,
        "error": proc.stderr
    }

    # 汇总
    all_passed = all(r["passed"] for r in results.values())
    return {"all_passed": all_passed, "details": results}

# 演示：对一个正确的实现运行 sensor 链
good_code = '''import pandas as pd
from pathlib import Path

def sort_csv(file_path: str, sort_column: str, ascending: bool = True) -> pd.DataFrame:
    """读取 CSV 并按指定列排序。"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    df = pd.read_csv(file_path)
    if sort_column not in df.columns:
        raise KeyError(f"列不存在: {sort_column}")
    return df.sort_values(by=sort_column, ascending=ascending)
'''

result = sensor_chain(good_code)
print(json.dumps(result, indent=2, ensure_ascii=False))
```

---

## 练习题

### 基础题

1. 用一个日常生活的例子（如开车）解释前馈控制和反馈控制的区别。
2. 在控制矩阵的四个象限中，各举一个本章未提到的例子。
3. 为什么"能用计算型控制的地方，不要用推理型控制"？列出至少 3 个理由。

### 实践题

4. 为一个"文本摘要"任务（输入：长文本，输出：100 字摘要）设计完整的 Guide + Sensor 组合。注意：摘要质量难以用确定性方法检查——你如何处理这个挑战？
5. 实现一个"Sensor 反馈转 Guide"的闭环：当 linter sensor 检测到错误时，将错误信息格式化为 prompt 的一部分，重新发给模型。

### 思考题

6. 如果你只能为 harness 选择一个 Guide 和一个 Sensor（资源极度有限），你会选择哪个组合？为什么？
7. LLM-as-Judge（推理型 Sensor）本身也可能出错。如何缓解"用不可靠系统检查不可靠系统"的问题？
