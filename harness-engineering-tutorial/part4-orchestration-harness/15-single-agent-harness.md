# 第15章：单 Agent Harness 设计

> 一个裸露的 LLM API 调用不是 agent。Agent 是 LLM + 工具 + 记忆 + 约束的组合体。本章将展示如何设计单 agent harness——把模型能力组织在一个可靠、可测试、可控制的运行框架中。核心洞察：结构化约束不是限制器，而是乘法器。

---

## 学习目标

学完本章，你将能够：

1. 设计三层 agent harness 架构（模型接口、运行时环境、编排层）
2. 运用结构化约束作为能力乘法器
3. 实施 context engineering，让 guides 和 sensors 对 agent 可用
4. 编写有效的 AGENTS.md（目录而非百科全书）
5. 设计 retry、fallback、timeout 模式构建可靠的 agent 系统

---

## 15.1 三层 Harness 架构

### 裸 API 调用的问题

```python
# 这不是 agent，这是祈祷
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": user_input}],
)
print(response.choices[0].message.content)
```

没有工具调用、没有错误处理、没有成本控制、没有输出验证。一切都在"希望模型做对"。

### 三层架构

```
┌─────────────────────────────────────────────────┐
│          Layer 3: 编排层 (Orchestration)          │
│   ┌──────────────────────────────────────────┐  │
│   │  循环控制 | Retry/Fallback | 状态管理    │  │
│   │  输出路由 | 步数限制 | 成本预算          │  │
│   └──────────────────────────────────────────┘  │
├─────────────────────────────────────────────────┤
│          Layer 2: 运行时环境 (Runtime)           │
│   ┌──────────────────────────────────────────┐  │
│   │  工具注册 | 沙箱执行 | 权限控制          │  │
│   │  上下文管理 | 记忆存储 | 日志记录        │  │
│   └──────────────────────────────────────────┘  │
├─────────────────────────────────────────────────┤
│          Layer 1: 模型接口 (Model Interface)     │
│   ┌──────────────────────────────────────────┐  │
│   │  LLM 客户端 | Prompt 模板 | 输出解析     │  │
│   │  Token 计数 | 模型选择 | 参数配置        │  │
│   └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

### Layer 1：模型接口

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class ModelConfig:
    model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 4096
    seed: int | None = 42
    response_format: dict | None = None

class ModelInterface:
    """Layer 1: 模型接口封装"""
    
    def __init__(self, client, config: ModelConfig):
        self.client = client
        self.config = config
        self.total_tokens = 0
        self.call_count = 0
    
    def complete(
        self, 
        messages: list[dict],
        tools: list[dict] | None = None,
        **overrides,
    ) -> dict:
        """发送请求并返回结构化结果"""
        params = {
            "model": overrides.get("model", self.config.model),
            "messages": messages,
            "temperature": overrides.get("temperature", self.config.temperature),
            "max_tokens": overrides.get("max_tokens", self.config.max_tokens),
        }
        
        if self.config.seed is not None:
            params["seed"] = self.config.seed
        if self.config.response_format:
            params["response_format"] = self.config.response_format
        if tools:
            params["tools"] = tools
        
        response = self.client.chat.completions.create(**params)
        
        # 追踪 token 用量
        usage = response.usage
        self.total_tokens += usage.total_tokens
        self.call_count += 1
        
        choice = response.choices[0]
        return {
            "content": choice.message.content,
            "tool_calls": getattr(choice.message, "tool_calls", None),
            "finish_reason": choice.finish_reason,
            "tokens": {
                "prompt": usage.prompt_tokens,
                "completion": usage.completion_tokens,
                "total": usage.total_tokens,
            },
        }
```

### Layer 2：运行时环境

```python
import logging

class RuntimeEnvironment:
    """Layer 2: Agent 运行时环境"""
    
    def __init__(self):
        self.tools: dict[str, dict] = {}     # name → {handler, schema, permissions}
        self.context: dict[str, Any] = {}     # 共享上下文
        self.memory: list[dict] = []          # 对话/操作记忆
        self.logger = logging.getLogger("agent-runtime")
    
    def register_tool(
        self, 
        name: str, 
        handler, 
        schema: dict,
        requires_approval: bool = False,
    ):
        """注册一个工具"""
        self.tools[name] = {
            "handler": handler,
            "schema": schema,
            "requires_approval": requires_approval,
            "call_count": 0,
        }
    
    def execute_tool(self, name: str, arguments: dict) -> dict:
        """在沙箱中执行工具"""
        if name not in self.tools:
            return {"error": f"Unknown tool: {name}"}
        
        tool = self.tools[name]
        
        if tool["requires_approval"]:
            self.logger.warning(f"Tool '{name}' requires approval — skipping in auto mode")
            return {"error": f"Tool '{name}' requires human approval"}
        
        try:
            tool["call_count"] += 1
            result = tool["handler"](**arguments)
            self.logger.info(f"Tool '{name}' executed successfully")
            return {"result": result}
        except Exception as e:
            self.logger.error(f"Tool '{name}' failed: {e}")
            return {"error": str(e)}
    
    def get_tool_schemas(self) -> list[dict]:
        """获取所有工具的 schema（传给 LLM）"""
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool["schema"].get("description", ""),
                    "parameters": tool["schema"].get("parameters", {}),
                },
            }
            for name, tool in self.tools.items()
        ]
    
    def add_to_memory(self, entry: dict):
        self.memory.append(entry)
    
    def get_recent_memory(self, n: int = 10) -> list[dict]:
        return self.memory[-n:]
```

### Layer 3：编排层

```python
class OrchestrationLayer:
    """Layer 3: Agent 编排控制"""
    
    def __init__(
        self,
        model: ModelInterface,
        runtime: RuntimeEnvironment,
        max_steps: int = 20,
        max_cost_usd: float = 1.0,
    ):
        self.model = model
        self.runtime = runtime
        self.max_steps = max_steps
        self.max_cost_usd = max_cost_usd
        self.step = 0
    
    def run(self, system_prompt: str, user_input: str) -> dict:
        """运行完整的 agent 循环"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        
        while self.step < self.max_steps:
            self.step += 1
            
            # 1. 调用模型
            response = self.model.complete(
                messages=messages,
                tools=self.runtime.get_tool_schemas(),
            )
            
            # 2. 检查是否完成
            if response["finish_reason"] == "stop":
                return {
                    "status": "completed",
                    "output": response["content"],
                    "steps": self.step,
                    "tokens": self.model.total_tokens,
                }
            
            # 3. 处理工具调用
            if response["tool_calls"]:
                messages.append({
                    "role": "assistant",
                    "content": response["content"],
                    "tool_calls": [tc.model_dump() for tc in response["tool_calls"]],
                })
                
                for tool_call in response["tool_calls"]:
                    import json
                    args = json.loads(tool_call.function.arguments)
                    result = self.runtime.execute_tool(
                        tool_call.function.name, args
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result),
                    })
            
            # 4. 成本检查
            estimated_cost = self.model.total_tokens * 0.00001  # 简化
            if estimated_cost > self.max_cost_usd:
                return {
                    "status": "cost_limit",
                    "output": "Budget exceeded",
                    "steps": self.step,
                    "cost_usd": estimated_cost,
                }
        
        return {
            "status": "step_limit",
            "output": "Maximum steps reached",
            "steps": self.step,
        }
```

---

## 15.2 结构化约束作为乘法器

### 反直觉的洞察

大多数人认为：**约束 = 限制 = 模型能做的事变少了**。

现实恰好相反：**结构化约束 = 模型犯错的空间变小了 = 可靠性上升 = 能做的事变多了**。

```
无约束的 agent：
  理论能力：100%    实际可靠性：30%    有效能力：30%

结构化约束的 agent：
  理论能力：70%     实际可靠性：90%    有效能力：63%
```

### 约束类型矩阵

| 约束类型 | 机制 | 效果 | 示例 |
|---------|------|------|------|
| 输出格式约束 | JSON Schema / structured output | 消除解析错误 | `response_format: json_schema` |
| 工具约束 | 白名单 + schema 验证 | 防止非法操作 | 只允许 read 不允许 write |
| 步骤约束 | 最大步数限制 | 防止无限循环 | `max_steps=20` |
| 范围约束 | system prompt 中明确边界 | 聚焦任务 | "你只处理客服问题" |
| 依赖约束 | 代码层级分离 | 防止架构腐化 | Types → Config → Service |
| 审美约束 | 自定义 linter 规则 | 保持代码风格一致 | 命名规范、模式规范 |

### 实现：约束即配置

```python
@dataclass
class AgentConstraints:
    """Agent 的结构化约束配置"""
    
    # 输出约束
    output_format: str | None = None  # "json", "markdown", "text"
    output_schema: dict | None = None  # JSON Schema
    max_output_length: int = 5000
    
    # 行为约束
    max_steps: int = 20
    max_tool_calls_per_step: int = 3
    allowed_tools: list[str] | None = None  # None = 全部允许
    forbidden_tools: list[str] = field(default_factory=list)
    requires_reasoning: bool = False  # 要求 chain-of-thought
    
    # 安全约束
    max_cost_usd: float = 1.0
    max_tokens_per_call: int = 4096
    timeout_seconds: float = 120.0
    sandbox_mode: bool = True
    
    # 范围约束
    domain: str = ""  # 任务领域描述
    out_of_scope_response: str = "This question is outside my scope."
    
    def to_system_prompt_section(self) -> str:
        """将约束转化为 system prompt 的一部分"""
        lines = ["## Constraints"]
        
        if self.domain:
            lines.append(f"- You are a specialist in: {self.domain}")
            lines.append(f"- For out-of-scope questions, respond: {self.out_of_scope_response}")
        
        if self.output_format:
            lines.append(f"- Always respond in {self.output_format} format")
        
        if self.requires_reasoning:
            lines.append("- Always show your reasoning before giving a final answer")
        
        if self.allowed_tools:
            lines.append(f"- You may only use these tools: {', '.join(self.allowed_tools)}")
        
        lines.append(f"- Maximum {self.max_steps} steps per task")
        lines.append(f"- Maximum {self.max_output_length} characters per response")
        
        return "\n".join(lines)
    
    def validate_tool_call(self, tool_name: str) -> bool:
        """验证工具调用是否在约束内"""
        if tool_name in self.forbidden_tools:
            return False
        if self.allowed_tools is not None and tool_name not in self.allowed_tools:
            return False
        return True
```

---

## 15.3 Context Engineering：让 Guides 和 Sensors 对 Agent 可用

### Context Engineering 的核心

Agent 的能力上限由它的 context 决定——它只能用看到的信息做决策。Context engineering 就是精心设计 agent 在每一步看到什么。

```
┌──────────────────────────────────────────────┐
│              Agent Context                    │
│                                              │
│  ┌─────────────┐   ┌─────────────────────┐  │
│  │  Guides     │   │  Sensors            │  │
│  │  (静态指导)  │   │  (动态感知)          │  │
│  │             │   │                     │  │
│  │  - 规则     │   │  - 当前文件状态      │  │
│  │  - 风格指南  │   │  - 测试结果         │  │
│  │  - 架构约束  │   │  - Lint 输出        │  │
│  │  - 示例     │   │  - Git diff         │  │
│  └─────────────┘   └─────────────────────┘  │
│                                              │
│  ┌──────────────────────────────────────┐    │
│  │  Memory (积累)                       │    │
│  │  - 之前的决策和原因                    │    │
│  │  - 失败的尝试（不要重复）             │    │
│  │  - 用户偏好                          │    │
│  └──────────────────────────────────────┘    │
└──────────────────────────────────────────────┘
```

### Context 组装器

```python
class ContextAssembler:
    """为 agent 组装上下文"""
    
    def __init__(self):
        self.guides: list[dict] = []     # 静态指导
        self.sensors: list[dict] = []    # 动态感知
        self.max_context_tokens = 8000   # 上下文预算
    
    def add_guide(self, name: str, content: str, priority: int = 5):
        """添加静态指导（规则、风格、示例）"""
        self.guides.append({
            "name": name,
            "content": content,
            "priority": priority,
            "tokens": len(content.split()) * 1.3,  # 粗略估算
        })
    
    def add_sensor(self, name: str, reader_fn, priority: int = 5):
        """添加动态感知（运行时读取）"""
        self.sensors.append({
            "name": name,
            "reader_fn": reader_fn,
            "priority": priority,
        })
    
    def assemble(self) -> str:
        """组装完整的上下文"""
        sections = []
        token_budget = self.max_context_tokens
        
        # 收集所有内容
        items = []
        
        for guide in self.guides:
            items.append({
                "type": "guide",
                "name": guide["name"],
                "content": guide["content"],
                "priority": guide["priority"],
                "tokens": guide["tokens"],
            })
        
        for sensor in self.sensors:
            try:
                content = sensor["reader_fn"]()
                tokens = len(str(content).split()) * 1.3
                items.append({
                    "type": "sensor",
                    "name": sensor["name"],
                    "content": str(content),
                    "priority": sensor["priority"],
                    "tokens": tokens,
                })
            except Exception as e:
                items.append({
                    "type": "sensor",
                    "name": sensor["name"],
                    "content": f"[Sensor error: {e}]",
                    "priority": 0,
                    "tokens": 10,
                })
        
        # 按优先级排序，在预算内选择
        items.sort(key=lambda x: x["priority"], reverse=True)
        
        selected = []
        for item in items:
            if token_budget - item["tokens"] >= 0:
                selected.append(item)
                token_budget -= item["tokens"]
        
        # 组装
        for item in selected:
            icon = "GUIDE" if item["type"] == "guide" else "SENSOR"
            sections.append(f"## [{icon}] {item['name']}\n{item['content']}")
        
        return "\n\n---\n\n".join(sections)


# 使用示例
ctx = ContextAssembler()

# 静态 guides
ctx.add_guide("coding_style", """
- Use type hints for all function parameters
- Prefer dataclasses over plain dicts
- Every public function needs a docstring
""", priority=8)

ctx.add_guide("architecture", """
Dependency layers (strict, no upward imports):
  Types → Config → Repository → Service → Runtime → UI
""", priority=9)

# 动态 sensors
ctx.add_sensor("git_diff", lambda: run_command("git diff --stat"), priority=7)
ctx.add_sensor("test_status", lambda: run_command("pytest --tb=no -q"), priority=6)
ctx.add_sensor("lint_errors", lambda: run_command("ruff check . --select E"), priority=7)
```

---

## 15.4 AGENTS.md：目录而非百科全书

### 反模式：巨大的 AGENTS.md

```markdown
<!-- 反模式：1000 行的 AGENTS.md -->
# AGENTS.md

## 项目介绍
这个项目是一个... （200 行）

## 架构说明
我们使用了... （300 行）

## 编码规范
1. 变量命名...
2. 函数设计...
3. 错误处理...
（300 行）

## API 文档
### POST /api/users
...
（200 行）
```

问题：
- **超出 context 窗口**：agent 看不完
- **信噪比低**：99% 的内容与当前任务无关
- **维护成本高**：一改就过时

### 正确做法：AGENTS.md 是目录

```markdown
# AGENTS.md

## Quick Orientation
- Language: Python 3.12, strict typing
- Framework: FastAPI + SQLAlchemy
- Test: pytest, run with `make test`
- Lint: ruff + mypy, run with `make lint`

## Architecture
See `docs/architecture.md` for the layer diagram.
Dependency rule: Types → Config → Repo → Service → Runtime → UI

## Key Directories
- `src/models/`     — SQLAlchemy models (types layer)
- `src/services/`   — Business logic (service layer)
- `src/api/`        — FastAPI routes (runtime layer)
- `tests/`          — Mirror of src/ structure
- `docs/`           — Detailed docs (agent should read when needed)

## Common Tasks
- Add a new API endpoint: see `docs/guides/new-endpoint.md`
- Add a database migration: see `docs/guides/migration.md`
- Fix a failing test: run `pytest -x --tb=short` first

## Constraints
- Never import from a higher layer (e.g., services cannot import from api)
- All public functions must have type hints and docstrings
- Max function length: 30 lines
- Custom lint rules: see `.ruff.toml`
```

### AGENTS.md 设计原则

| 原则 | 说明 |
|------|------|
| **索引优先** | 指向详细文档，而非内联所有内容 |
| **任务导向** | 按"agent 想做什么"组织，而非按"项目有什么" |
| **可执行** | 给出具体的命令，而非描述性文字 |
| **短小精悍** | 控制在 50-100 行以内 |
| **持续更新** | 每次架构变更都同步更新 |

---

## 15.5 刚性架构模型：用结构化测试强制依赖层

### 什么是刚性架构

"刚性"不是贬义——它意味着**架构约束由代码强制执行**，而不是靠开发者的自觉。

```
层级依赖规则：
  Types → Config → Repository → Service → Runtime → UI
  
  ✓ Service 可以导入 Repository
  ✗ Repository 不能导入 Service（向上依赖）
  ✗ Types 不能导入任何其他层（最底层）
```

### 依赖检查器

```python
import ast
from pathlib import Path

# 层级定义（数字越小层级越低）
LAYER_MAP = {
    "types": 0,
    "config": 1,
    "repository": 2,
    "service": 3,
    "runtime": 4,
    "ui": 5,
}

class DependencyChecker:
    """检查代码是否违反层级依赖规则"""
    
    def __init__(self, src_dir: str, layer_map: dict = LAYER_MAP):
        self.src_dir = Path(src_dir)
        self.layer_map = layer_map
    
    def check_file(self, file_path: str) -> list[dict]:
        """检查单个文件的依赖违规"""
        path = Path(file_path)
        file_layer = self._get_layer(path)
        if file_layer is None:
            return []
        
        violations = []
        source = path.read_text()
        
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imported_module = self._get_import_module(node)
                if imported_module:
                    import_layer = self._module_to_layer(imported_module)
                    if import_layer is not None and import_layer > file_layer:
                        violations.append({
                            "file": str(path),
                            "line": node.lineno,
                            "file_layer": file_layer,
                            "import_layer": import_layer,
                            "imported": imported_module,
                            "message": (
                                f"Layer violation: {self._layer_name(file_layer)} "
                                f"imports from {self._layer_name(import_layer)}"
                            ),
                        })
        
        return violations
    
    def check_all(self) -> list[dict]:
        """检查整个项目"""
        all_violations = []
        for py_file in self.src_dir.rglob("*.py"):
            violations = self.check_file(str(py_file))
            all_violations.extend(violations)
        return all_violations
    
    def _get_layer(self, path: Path) -> int | None:
        for part in path.parts:
            for layer_name, level in self.layer_map.items():
                if layer_name in part.lower():
                    return level
        return None
    
    def _get_import_module(self, node) -> str | None:
        if isinstance(node, ast.ImportFrom) and node.module:
            return node.module
        if isinstance(node, ast.Import):
            return node.names[0].name if node.names else None
        return None
    
    def _module_to_layer(self, module: str) -> int | None:
        for layer_name, level in self.layer_map.items():
            if layer_name in module.lower():
                return level
        return None
    
    def _layer_name(self, level: int) -> str:
        for name, lvl in self.layer_map.items():
            if lvl == level:
                return name
        return f"level-{level}"
```

### 自定义 Linter + 修复指令注入

```python
class AgentLinter:
    """自定义 linter，输出可被 agent 直接使用的修复指令"""
    
    def __init__(self):
        self.rules: list[dict] = []
    
    def add_rule(self, name: str, check_fn, fix_instruction: str):
        """
        check_fn: (file_content, file_path) → list[violation]
        fix_instruction: agent 可读的修复指令模板
        """
        self.rules.append({
            "name": name,
            "check": check_fn,
            "fix": fix_instruction,
        })
    
    def lint(self, file_path: str) -> list[dict]:
        """运行所有规则，返回违规和修复指令"""
        content = Path(file_path).read_text()
        issues = []
        
        for rule in self.rules:
            violations = rule["check"](content, file_path)
            for v in violations:
                issues.append({
                    "rule": rule["name"],
                    "line": v.get("line", 0),
                    "message": v.get("message", ""),
                    "fix_instruction": rule["fix"].format(**v),
                })
        
        return issues
    
    def lint_to_agent_context(self, file_path: str) -> str:
        """生成可直接注入 agent context 的修复清单"""
        issues = self.lint(file_path)
        if not issues:
            return "No lint issues found."
        
        lines = ["## Lint Issues to Fix", ""]
        for i, issue in enumerate(issues, 1):
            lines.append(
                f"{i}. [{issue['rule']}] Line {issue['line']}: {issue['message']}"
            )
            lines.append(f"   Fix: {issue['fix_instruction']}")
            lines.append("")
        
        return "\n".join(lines)


# 使用示例
linter = AgentLinter()

linter.add_rule(
    "no-bare-except",
    check_fn=lambda content, path: [
        {"line": i+1, "message": "Bare except clause"}
        for i, line in enumerate(content.split("\n"))
        if "except:" in line and "except Exception" not in line
    ],
    fix_instruction="Change bare `except:` to `except Exception as e:` on line {line}",
)

linter.add_rule(
    "require-docstring",
    check_fn=lambda content, path: check_missing_docstrings(content),
    fix_instruction="Add a docstring to the function/class at line {line}",
)
```

---

## 15.6 Retry、Fallback、Timeout 模式

### 为什么 Agent 需要弹性模式

LLM API 会失败——rate limit、网络超时、模型返回垃圾。没有弹性模式的 agent 就像一辆没有安全气囊的汽车。

### 弹性模式实现

```python
import time
import random
from typing import TypeVar, Callable
from functools import wraps

T = TypeVar("T")

class RetryConfig:
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: tuple = (Exception,),
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions

def with_retry(config: RetryConfig):
    """Retry 装饰器，带指数退避"""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(config.max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e
                    if attempt < config.max_retries:
                        delay = min(
                            config.base_delay * (config.exponential_base ** attempt),
                            config.max_delay,
                        )
                        if config.jitter:
                            delay *= random.uniform(0.5, 1.5)
                        time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator


class FallbackChain:
    """Fallback 链：依次尝试多个策略"""
    
    def __init__(self):
        self.strategies: list[tuple[str, Callable]] = []
    
    def add(self, name: str, strategy: Callable):
        self.strategies.append((name, strategy))
        return self
    
    def execute(self, *args, **kwargs) -> dict:
        """依次尝试每个策略，直到成功"""
        errors = []
        
        for name, strategy in self.strategies:
            try:
                result = strategy(*args, **kwargs)
                return {
                    "status": "success",
                    "strategy": name,
                    "result": result,
                    "attempts": len(errors) + 1,
                }
            except Exception as e:
                errors.append({"strategy": name, "error": str(e)})
        
        return {
            "status": "all_failed",
            "errors": errors,
            "result": None,
        }


class TimeoutWrapper:
    """超时控制"""
    
    def __init__(self, timeout_seconds: float):
        self.timeout = timeout_seconds
    
    def run(self, fn: Callable, *args, **kwargs):
        """在超时内运行函数（使用 threading）"""
        import threading
        
        result = {"value": None, "error": None, "completed": False}
        
        def target():
            try:
                result["value"] = fn(*args, **kwargs)
                result["completed"] = True
            except Exception as e:
                result["error"] = e
        
        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout=self.timeout)
        
        if not result["completed"]:
            if result["error"]:
                raise result["error"]
            raise TimeoutError(
                f"Operation timed out after {self.timeout}s"
            )
        
        if result["error"]:
            raise result["error"]
        
        return result["value"]


# ===== 组合使用 =====
def build_resilient_agent_call(primary_model, fallback_model, cheap_model):
    """构建弹性 agent 调用"""
    
    chain = FallbackChain()
    
    # 策略 1：用主力模型
    @with_retry(RetryConfig(max_retries=2, base_delay=1.0))
    def try_primary(messages, tools):
        return primary_model.complete(messages, tools)
    
    # 策略 2：用备选模型
    @with_retry(RetryConfig(max_retries=1, base_delay=2.0))
    def try_fallback(messages, tools):
        return fallback_model.complete(messages, tools)
    
    # 策略 3：用廉价模型（降级服务）
    def try_cheap(messages, tools):
        return cheap_model.complete(messages, tools=None)  # 不给工具
    
    chain.add("primary", try_primary)
    chain.add("fallback", try_fallback)
    chain.add("degraded", try_cheap)
    
    return chain
```

---

## 本章小结

| 主题 | 关键洞察 |
|------|---------|
| 三层架构 | 模型接口 → 运行时环境 → 编排层，关注点分离 |
| 结构化约束 | 约束 = 乘法器；减少出错空间，提高有效能力 |
| Context engineering | Agent 的能力上限 = 它能看到什么 |
| AGENTS.md | 50-100 行的目录，指向详细文档，而非内联一切 |
| 刚性架构 | 用代码（linter / test）强制架构约束，而非靠自觉 |
| 弹性模式 | Retry + Fallback + Timeout 是生产 agent 的标配 |

---

## 动手实验

### 实验 1：构建一个包含 Retry + Validation + Fallback 的单 Agent Harness

**目标**：组合三层架构，构建一个能自动重试、验证输出、降级到备选模型的 agent

```python
# 要求：
# 1. Layer 1: 支持两个模型（primary + fallback）
# 2. Layer 2: 注册 3 个工具（search, read_file, write_file）
# 3. Layer 3: 
#    - 每步验证输出格式（必须是 JSON）
#    - 格式错误时自动 retry（最多 2 次）
#    - retry 失败时 fallback 到备选模型
#    - 总步数限制 15 步
# 4. 写测试验证每种失败路径
```

### 实验 2：设计 Context 组装策略

**目标**：为一个"代码重构 agent"设计 context 组装策略

1. 定义 3 个 guides（架构规则、编码风格、重构原则）
2. 定义 3 个 sensors（git diff、lint 输出、测试结果）
3. 设置 8000 token 预算
4. 实现优先级排序和预算分配

### 实验 3：写一个 AGENTS.md

**目标**：为一个真实项目写一个有效的 AGENTS.md

1. 控制在 80 行以内
2. 包含 Quick Orientation、Architecture、Key Directories、Common Tasks、Constraints
3. 所有详细内容通过链接指向 docs/ 目录
4. 让同事（或另一个 LLM）只读 AGENTS.md 就能开始贡献代码

---

## 练习题

### 基础题

1. 解释三层 harness 架构中每一层的职责。如果你把所有逻辑放在一层（"God class"），会产生什么问题？

2. 给出三个"结构化约束作为乘法器"的具体例子。对于每个例子，说明约束如何提高了有效能力。

3. AGENTS.md 应该包含什么，不应该包含什么？列出各 3 条。

### 实践题

4. 实现一个 `ContextBudgetAllocator`，它接收一组 guides 和 sensors，在给定 token 预算内选择最有价值的组合。考虑：(a) 优先级，(b) 与当前任务的相关性，(c) 时效性。

5. 为一个文件操作 agent 设计完整的约束配置（`AgentConstraints`），包括：只允许读写特定目录、禁止删除、每步最多修改 3 个文件、必须在修改前备份。

### 思考题

6. "结构化约束最终会被 AI 系统自己学会绕过。" 这种担忧是否合理？在 harness 层面如何防范？

7. Context engineering 的极限在哪里？当 context 窗口不够用时（信息量 >> 窗口大小），有哪些策略？各自的 trade-off 是什么？
