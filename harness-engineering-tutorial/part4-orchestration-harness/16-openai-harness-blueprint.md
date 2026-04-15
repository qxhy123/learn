# 第16章：OpenAI Harness 蓝图

> 2025年8月到2026年2月，OpenAI 内部团队用 AI agent 从零构建了一个百万行代码的生产系统——Codex 项目。全程零手写代码，1500+ 个 PR，开发速度达到人类团队的 10 倍。这不是 demo，这是工程实践。本章拆解这个标志性案例，提取 harness engineering 的核心蓝图。

---

## 学习目标

学完本章，你将能够：

1. 理解 OpenAI 百万行代码实验的关键决策和工程约束
2. 解释三大支柱（context engineering、架构约束、熵管理）如何协同工作
3. 设计分层依赖模型并用自动化手段强制执行
4. 实施"审美不变量"（taste invariants）通过自定义 lint 规则
5. 构建结构化 docs/ 目录作为 agent 的系统记录

---

## 16.1 百万行代码实验

### 背景

```
项目：Codex（OpenAI 内部工具平台）
时间：2025.08 - 2026.02（约 6 个月）
规模：~1,000,000 行代码
PR 数：1,500+
人类手写代码：0 行
开发速度：~10x vs 人类团队
Agent：基于 OpenAI 自有模型的 coding agent
```

### 为什么这个实验重要

这不是"让 GPT 帮我写个脚本"。这是用 AI agent 完成完整的软件工程流程：

```
需求 → 设计 → 编码 → 测试 → 代码审查 → 部署 → 维护
  ↑                                                ↑
  └──────── 全部由 AI agent 完成 ──────────────────┘
```

关键发现：**agent 能做到这些，不是因为模型足够强，而是因为 harness 足够好**。

### 人类的角色变化

```
传统开发：      人类写代码，工具辅助
Copilot 时代：  人类写代码，AI 补全
Agent 时代：    人类设计 harness，AI 写代码

┌─────────────────────────────────┐
│         人类的工作               │
│                                 │
│  ✗ 写代码                       │
│  ✓ 定义架构约束                  │
│  ✓ 设计依赖层级                  │
│  ✓ 编写 lint 规则               │
│  ✓ 构建评估 harness             │
│  ✓ 审查 agent 输出              │
│  ✓ 迭代 prompt 和约束           │
└─────────────────────────────────┘
```

---

## 16.2 三大支柱

### 支柱总览

```
┌─────────────────────────────────────────────────────────┐
│              OpenAI Harness 三大支柱                      │
│                                                         │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐  │
│  │   Context      │ │  Architecture │ │   Entropy     │  │
│  │   Engineering  │ │  Constraints  │ │   Management  │  │
│  │               │ │               │ │               │  │
│  │  让 agent 看到 │ │  让 agent 只能│ │  让 agent 的  │  │
│  │  正确的信息    │ │  做正确的事   │ │  输出可预测   │  │
│  └───────────────┘ └───────────────┘ └───────────────┘  │
│                                                         │
│  三者协同：context 提供方向，constraint 划定边界，        │
│           entropy management 确保一致性                   │
└─────────────────────────────────────────────────────────┘
```

### 支柱 1：Context Engineering

核心思想：**agent 的输出质量 ≤ 它 context 的质量**。

```python
class CodexContextStrategy:
    """Codex 项目的 context engineering 策略"""
    
    # 1. 分层文档结构
    DOCS_STRUCTURE = {
        "docs/architecture.md":     "系统架构总览和依赖规则",
        "docs/conventions.md":      "编码约定和命名规范",
        "docs/patterns.md":         "推荐的设计模式和反模式",
        "docs/api-contracts.md":    "API 接口契约",
        "docs/decisions/":          "架构决策记录（ADR）",
        "AGENTS.md":                "Agent 快速入口（50行目录）",
    }
    
    # 2. 渐进式 context 加载
    CONTEXT_LEVELS = {
        "level_0": ["AGENTS.md"],                          # 任何任务
        "level_1": ["docs/conventions.md"],                 # 写代码
        "level_2": ["docs/architecture.md"],                # 改架构
        "level_3": ["docs/decisions/"],                     # 重大决策
    }
    
    # 3. 任务-context 映射
    TASK_CONTEXT_MAP = {
        "add_endpoint":   ["docs/api-contracts.md", "docs/patterns.md"],
        "fix_bug":        ["docs/conventions.md"],
        "refactor":       ["docs/architecture.md", "docs/patterns.md"],
        "add_test":       ["docs/conventions.md"],
        "new_feature":    ["docs/architecture.md", "docs/api-contracts.md"],
    }
```

### 支柱 2：架构约束

核心思想：**约束不是限制，是 guardrail**。

在 Codex 项目中，人类不写代码，但人类定义了严格的架构规则。Agent 可以写任何代码，但必须遵守这些规则。如果违反，CI 会拒绝 PR。

### 支柱 3：熵管理

核心思想：**AI 输出天然是高熵的，harness 的任务是降低熵**。

```
高熵（无约束的 agent）：
  每个文件的风格不同
  命名不一致
  架构随意
  → 1000 行后无法维护

低熵（有 harness 的 agent）：
  统一风格
  一致命名
  严格分层
  → 100 万行仍然可维护
```

熵管理的工具：

| 工具 | 作用 | 实现 |
|------|------|------|
| Linter | 强制代码风格 | ruff / eslint + 自定义规则 |
| Type checker | 强制类型安全 | mypy / tsc --strict |
| 依赖检查 | 强制分层 | 自定义 import checker |
| 测试套件 | 强制行为正确 | pytest + 属性测试 |
| PR 模板 | 强制说明格式 | GitHub PR template |
| Commit hook | 强制提交规范 | pre-commit |

---

## 16.3 分层依赖模型

### 六层架构

```
Layer 5: UI            ← 用户交互、展示
Layer 4: Runtime       ← 应用启动、路由、中间件
Layer 3: Service       ← 业务逻辑
Layer 2: Repository    ← 数据访问
Layer 1: Config        ← 配置管理
Layer 0: Types         ← 类型定义、接口

依赖规则：
  ✓ 上层可以导入下层
  ✗ 下层不能导入上层
  ✗ 同层之间通过接口通信，不直接导入
```

### 为什么分层对 Agent 特别重要

Agent 没有人类的直觉——它不知道"把数据库查询放在 API handler 里是不对的"。但如果有分层规则：

```python
# agent 生成的代码
# 文件：src/api/users.py （Layer 4: Runtime）

from src.repository.user_repo import UserRepository  # ✓ Runtime → Repository
from src.service.user_service import UserService      # ✓ Runtime → Service

# 如果 agent 生成了这行：
from src.ui.dashboard import render_dashboard         # ✗ Runtime → UI（向上！）
# → CI 立刻失败，agent 收到错误信息，自动修正
```

### 分层依赖检查的完整实现

```python
from pathlib import Path
from dataclasses import dataclass
import ast

@dataclass
class LayerDefinition:
    name: str
    level: int
    path_patterns: list[str]  # 文件路径中的标识

class LayeredArchitectureChecker:
    """分层架构强制检查器"""
    
    def __init__(self, src_root: str, layers: list[LayerDefinition]):
        self.src_root = Path(src_root)
        self.layers = sorted(layers, key=lambda l: l.level)
        self._layer_map = {l.name: l for l in layers}
    
    def file_to_layer(self, file_path: str) -> LayerDefinition | None:
        """判断文件属于哪一层"""
        path = str(file_path)
        for layer in self.layers:
            for pattern in layer.path_patterns:
                if pattern in path:
                    return layer
        return None
    
    def check_file(self, file_path: str) -> list[dict]:
        """检查单个文件的依赖是否合规"""
        source_layer = self.file_to_layer(file_path)
        if source_layer is None:
            return []
        
        violations = []
        try:
            source = Path(file_path).read_text()
            tree = ast.parse(source)
        except (SyntaxError, FileNotFoundError):
            return []
        
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.ImportFrom) and node.module:
                module = node.module
            elif isinstance(node, ast.Import) and node.names:
                module = node.names[0].name
            
            if module is None:
                continue
            
            target_layer = self._module_to_layer(module)
            if target_layer and target_layer.level > source_layer.level:
                violations.append({
                    "file": file_path,
                    "line": node.lineno,
                    "source_layer": source_layer.name,
                    "source_level": source_layer.level,
                    "target_layer": target_layer.name,
                    "target_level": target_layer.level,
                    "imported_module": module,
                    "message": (
                        f"VIOLATION: {source_layer.name} (L{source_layer.level}) "
                        f"→ {target_layer.name} (L{target_layer.level}): "
                        f"upward dependency on '{module}'"
                    ),
                })
        
        return violations
    
    def check_all(self) -> dict:
        """检查整个项目"""
        all_violations = []
        files_checked = 0
        
        for py_file in self.src_root.rglob("*.py"):
            files_checked += 1
            violations = self.check_file(str(py_file))
            all_violations.extend(violations)
        
        return {
            "files_checked": files_checked,
            "violations": all_violations,
            "violation_count": len(all_violations),
            "clean": len(all_violations) == 0,
            "summary": self._summarize(all_violations),
        }
    
    def _module_to_layer(self, module: str) -> LayerDefinition | None:
        for layer in self.layers:
            for pattern in layer.path_patterns:
                if pattern.replace("/", ".") in module or pattern in module:
                    return layer
        return None
    
    def _summarize(self, violations):
        if not violations:
            return "All dependencies respect layer boundaries."
        
        by_type = {}
        for v in violations:
            key = f"{v['source_layer']} → {v['target_layer']}"
            by_type[key] = by_type.get(key, 0) + 1
        
        lines = ["Dependency violations found:"]
        for vtype, count in sorted(by_type.items(), key=lambda x: -x[1]):
            lines.append(f"  {vtype}: {count} violations")
        return "\n".join(lines)

    def visualize_layers(self) -> str:
        """ASCII 可视化层级结构"""
        lines = ["Layer Architecture:", ""]
        max_width = 40
        
        for layer in reversed(self.layers):
            bar = "=" * max_width
            lines.append(f"  L{layer.level}  [{bar}]")
            lines.append(f"       {layer.name}")
            lines.append(f"       patterns: {', '.join(layer.path_patterns)}")
            if layer.level > 0:
                lines.append(f"         ↓ (can import)")
        
        return "\n".join(lines)


# Codex 项目的层级定义
CODEX_LAYERS = [
    LayerDefinition("types",      0, ["src/types", "src/models"]),
    LayerDefinition("config",     1, ["src/config"]),
    LayerDefinition("repository", 2, ["src/repository", "src/repo"]),
    LayerDefinition("service",    3, ["src/service"]),
    LayerDefinition("runtime",    4, ["src/runtime", "src/api"]),
    LayerDefinition("ui",         5, ["src/ui", "src/frontend"]),
]
```

---

## 16.4 审美不变量：通过自定义 Lint 强制

### 什么是"审美不变量"

"审美"不是主观偏好，而是**一组可机器验证的代码质量标准**：

```
审美不变量 = 团队约定的"好代码应该长什么样" 的形式化表达

不是：    "代码应该写得优雅"       ← 主观、不可验证
而是：    "函数不超过 30 行"       ← 客观、可自动检查
         "所有 public 函数有 docstring"
         "import 顺序: stdlib → third-party → local"
         "类型注解覆盖率 > 95%"
```

### 为什么叫"不变量"

因为它们在整个代码库中必须**始终成立**——不是建议，是铁律。

### 自定义 Lint 规则实现

```python
class TasteInvariantChecker:
    """审美不变量检查器"""
    
    def __init__(self):
        self.invariants: list[dict] = []
    
    def add_invariant(
        self, 
        name: str, 
        check_fn, 
        description: str,
        severity: str = "error",  # error / warning
    ):
        self.invariants.append({
            "name": name,
            "check": check_fn,
            "description": description,
            "severity": severity,
        })
    
    def check_file(self, file_path: str) -> list[dict]:
        try:
            content = Path(file_path).read_text()
        except FileNotFoundError:
            return []
        
        issues = []
        for inv in self.invariants:
            violations = inv["check"](content, file_path)
            for v in violations:
                issues.append({
                    "invariant": inv["name"],
                    "description": inv["description"],
                    "severity": inv["severity"],
                    "file": file_path,
                    "line": v.get("line", 0),
                    "message": v.get("message", ""),
                })
        return issues
    
    def check_project(self, src_dir: str) -> dict:
        all_issues = []
        for py_file in Path(src_dir).rglob("*.py"):
            issues = self.check_file(str(py_file))
            all_issues.extend(issues)
        
        errors = [i for i in all_issues if i["severity"] == "error"]
        warnings = [i for i in all_issues if i["severity"] == "warning"]
        
        return {
            "total_issues": len(all_issues),
            "errors": len(errors),
            "warnings": len(warnings),
            "clean": len(errors) == 0,
            "issues": all_issues,
        }


# ===== Codex 项目的审美不变量 =====

def check_function_length(content: str, path: str, max_lines: int = 30):
    """函数体不超过 max_lines 行"""
    violations = []
    tree = ast.parse(content)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            length = node.end_lineno - node.lineno + 1
            if length > max_lines:
                violations.append({
                    "line": node.lineno,
                    "message": (
                        f"Function '{node.name}' is {length} lines "
                        f"(max {max_lines}). Split it."
                    ),
                })
    return violations

def check_docstrings(content: str, path: str):
    """所有 public 函数/类必须有 docstring"""
    violations = []
    tree = ast.parse(content)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name.startswith("_"):
                continue  # 跳过私有
            if not (node.body and isinstance(node.body[0], ast.Expr) 
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                violations.append({
                    "line": node.lineno,
                    "message": f"'{node.name}' missing docstring",
                })
    return violations

def check_no_star_imports(content: str, path: str):
    """禁止 from X import *"""
    violations = []
    for i, line in enumerate(content.split("\n"), 1):
        if "import *" in line and not line.strip().startswith("#"):
            violations.append({
                "line": i,
                "message": "Star import detected — use explicit imports",
            })
    return violations

def check_type_hints(content: str, path: str):
    """public 函数参数必须有类型注解"""
    violations = []
    tree = ast.parse(content)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_"):
                continue
            for arg in node.args.args:
                if arg.arg == "self" or arg.arg == "cls":
                    continue
                if arg.annotation is None:
                    violations.append({
                        "line": node.lineno,
                        "message": (
                            f"Parameter '{arg.arg}' in '{node.name}' "
                            f"missing type hint"
                        ),
                    })
    return violations


# 组装检查器
taste_checker = TasteInvariantChecker()
taste_checker.add_invariant(
    "max-function-length",
    check_function_length,
    "Functions must be ≤ 30 lines",
)
taste_checker.add_invariant(
    "require-docstrings",
    check_docstrings,
    "Public functions/classes must have docstrings",
)
taste_checker.add_invariant(
    "no-star-imports",
    check_no_star_imports,
    "Star imports are forbidden",
)
taste_checker.add_invariant(
    "require-type-hints",
    check_type_hints,
    "Public function parameters must have type hints",
)
```

---

## 16.5 结构化 docs/ 目录作为系统记录

### Docs 不是文档，是 Agent 的记忆

在 Codex 项目中，`docs/` 目录不是写给人看的文档——它是 **agent 的系统记忆**：

```
docs/
├── architecture.md          ← 系统架构（agent 做设计决策时读）
├── conventions.md           ← 编码约定（agent 写代码时读）
├── patterns.md              ← 设计模式（agent 选方案时读）
├── api-contracts.md         ← API 契约（agent 加端点时读）
├── glossary.md              ← 术语表（agent 命名时读）
├── decisions/               ← 架构决策记录
│   ├── 001-use-sqlalchemy.md
│   ├── 002-layer-architecture.md
│   └── 003-auth-strategy.md
└── guides/                  ← 操作指南
    ├── new-endpoint.md
    ├── new-model.md
    ├── migration.md
    └── debugging.md
```

### Docs 设计原则

| 原则 | 传统文档 | Agent 文档 |
|------|---------|-----------|
| 受众 | 人类开发者 | LLM agent（也对人类有用） |
| 格式 | 叙述性、有上下文 | 结构化、可直接执行 |
| 更新频率 | 经常过时 | 每个 PR 同步更新（CI 检查） |
| 长度 | 随意 | 有 token 预算意识 |
| 组织 | 按主题 | 按任务（agent 想做什么） |

### 架构决策记录（ADR）模板

```markdown
# ADR-{number}: {title}

## Status
Accepted | Deprecated | Superseded by ADR-{N}

## Context
What is the issue that we're seeing that is motivating this decision?

## Decision
What is the change that we're proposing and/or doing?

## Consequences
What becomes easier or more difficult to do because of this change?

## Agent Instructions
When an agent encounters code related to this decision:
- DO: {specific actions}
- DON'T: {specific anti-patterns}
- READ: {related docs}
```

### Docs 新鲜度检查器

```python
import subprocess
from datetime import datetime, timedelta

class DocsFreshnessChecker:
    """检查文档是否过时"""
    
    def __init__(self, docs_dir: str, max_age_days: int = 90):
        self.docs_dir = Path(docs_dir)
        self.max_age_days = max_age_days
    
    def check(self) -> dict:
        """检查所有文档的新鲜度"""
        results = []
        
        for md_file in self.docs_dir.rglob("*.md"):
            last_modified = self._git_last_modified(str(md_file))
            age_days = (datetime.now() - last_modified).days if last_modified else 999
            
            results.append({
                "file": str(md_file.relative_to(self.docs_dir)),
                "last_modified": last_modified.isoformat() if last_modified else "unknown",
                "age_days": age_days,
                "stale": age_days > self.max_age_days,
            })
        
        stale = [r for r in results if r["stale"]]
        
        return {
            "total_docs": len(results),
            "stale_count": len(stale),
            "stale_docs": stale,
            "freshness_score": round(
                1 - len(stale) / max(len(results), 1), 2
            ),
        }
    
    def _git_last_modified(self, file_path: str) -> datetime | None:
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--format=%aI", file_path],
                capture_output=True, text=True, cwd=str(self.docs_dir),
            )
            if result.stdout.strip():
                return datetime.fromisoformat(result.stdout.strip())
        except Exception:
            pass
        return None
```

---

## 16.6 成果与启示

### 数字说话

```
┌─────────────────────────────────────────────────┐
│             Codex 项目成果                        │
│                                                 │
│  代码量：    ~1,000,000 行                       │
│  PR 数：     1,500+                              │
│  开发周期：  6 个月                               │
│  人类手写代码：0 行                               │
│  开发速度：  ~10x vs 传统团队                     │
│                                                 │
│  关键指标：                                       │
│  - CI 通过率：>95%                               │
│  - 架构违规：持续为 0（被 lint 强制）             │
│  - 代码审查效率：人类只审查设计决策，不审查实现   │
│  - 维护成本：显著低于同规模手写项目               │
└─────────────────────────────────────────────────┘
```

### 为什么这是 Harness Engineering 的标志性实践

| 特征 | 说明 |
|------|------|
| **Harness 先行** | 先建约束和评估，再让 agent 写代码 |
| **约束即能力** | 严格的分层 + lint = agent 能写 100 万行不崩溃 |
| **文档即记忆** | docs/ 不是装饰，是 agent 的工作记忆 |
| **CI 即质量** | 每个 PR 都过完整检查，agent 的错误被即时拦截 |
| **人类角色转变** | 从"写代码的人"变成"设计 harness 的人" |

### 可迁移的经验

```
你不需要 OpenAI 的模型能力来复现这个方法论。
你需要的是：

1. 清晰的分层架构       ← 任何项目都能做
2. 自动化的约束检查      ← 写 lint 规则
3. 结构化的文档系统      ← 维护 docs/
4. CI 质量门禁          ← 配置 GitHub Actions
5. 评估优先的思维       ← 先定义"好"，再让 agent 做

这些工具是免费的。差距不在工具，在 discipline。
```

---

## 16.7 从蓝图到行动：实施路线图

### 30 天实施计划

```
Week 1: 基础
├── Day 1-2: 定义分层架构，写 AGENTS.md
├── Day 3-4: 实现依赖检查器，接入 pre-commit
└── Day 5-7: 建立 docs/ 结构，写 architecture.md

Week 2: 约束
├── Day 8-10: 实现审美不变量 lint 规则
├── Day 11-12: 配置 CI（lint + type check + dep check）
└── Day 13-14: 写第一批属性测试

Week 3: 评估
├── Day 15-17: 构建评估 harness 和黄金数据集
├── Day 18-19: 配置 CI 质量门禁
└── Day 20-21: 第一次全面评估 + 建立 baseline

Week 4: 迭代
├── Day 22-24: 用 agent 跑第一批真实任务
├── Day 25-27: 收集反馈，调整约束和 context
└── Day 28-30: 形成可重复的工作流
```

---

## 本章小结

| 主题 | 关键洞察 |
|------|---------|
| 百万行实验 | 证明了 harness engineering 能支撑大规模 AI 编码 |
| 三大支柱 | Context engineering + 架构约束 + 熵管理 协同工作 |
| 分层依赖 | Types → Config → Repo → Service → Runtime → UI，自动检查 |
| 审美不变量 | 把"代码品味"形式化为可机器验证的规则 |
| Docs 即记忆 | 结构化文档是 agent 的工作记忆，不是装饰 |
| 人类角色 | 从写代码转变为设计 harness、约束和评估 |

---

## 动手实验

### 实验 1：为一个小项目设计分层依赖模型和 Lint 规则

**目标**：选择一个真实的小项目（或创建一个示例项目），实施完整的 harness

```
步骤：
1. 定义 4-6 个层级（不需要和 Codex 完全一样）
2. 实现 LayeredArchitectureChecker
3. 实现至少 3 个审美不变量
4. 配置 pre-commit hook 自动检查
5. 故意制造一个违规，验证检查器能捕获
```

### 实验 2：构建 Agent 可消费的文档系统

**目标**：为一个项目建立结构化的 docs/ 目录

```
1. 写 AGENTS.md（50 行以内的目录）
2. 写 docs/architecture.md（包含层级图和依赖规则）
3. 写 docs/conventions.md（编码约定，agent 可直接遵循）
4. 写一个 ADR（架构决策记录）
5. 实现 DocsFreshnessChecker，验证所有文档都在 90 天内更新过
```

### 实验 3：模拟 Codex 工作流

**目标**：用 AI agent（Claude / GPT）完成一个小型开发任务，对比有 harness 和无 harness 的效果

```
任务：为一个 TODO 应用添加"标签"功能

A 组（无 harness）：直接让 agent "Add a tags feature to this TODO app"
B 组（有 harness）：
  1. 先更新 docs/architecture.md 描述新的数据模型
  2. 更新 AGENTS.md 添加任务指引
  3. 让 agent 在约束内实现
  4. 用 lint + dep check 验证

比较：代码质量、架构一致性、首次通过率
```

---

## 练习题

### 基础题

1. 列出 OpenAI Codex 项目三大支柱，并用一句话解释每个支柱的核心作用。

2. 为一个电商系统设计分层依赖模型。定义至少 5 个层级，并给出每层包含的模块示例。

3. 解释"审美不变量"和传统 lint 规则的区别。给出两个只有"审美不变量"能捕获的代码问题示例。

### 实践题

4. 实现一个完整的 `TasteInvariantChecker`，包含以下规则：
   - 函数不超过 30 行
   - 类不超过 200 行
   - 文件不超过 500 行
   - 嵌套深度不超过 4 层
   - 所有 TODO 注释必须包含 issue 号

5. 设计一个 `DocsConsistencyChecker`，它能检测 docs/ 中的文档是否与实际代码一致。例如：如果 architecture.md 说"我们使用 6 层架构"，但代码中只有 4 层目录，应该报告不一致。

### 思考题

6. "零手写代码"是一个值得追求的目标吗？在什么情况下，人类手写代码仍然比 agent 生成更优？讨论至少三种场景。

7. 如果 Codex 项目的约束规则本身有错误（比如某个 lint 规则过于严格，阻止了合理的代码模式），agent 会怎么表现？人类应该如何发现和修正这种"约束 bug"？
