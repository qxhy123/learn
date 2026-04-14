# 第4章：工具链配置

> "好的工具不会让你变聪明，但会让你的聪明不被浪费。"

---

## 4.1 Vibe Coding 工具栈概览

```
┌─────────────────────────────────────────────────────┐
│                  Vibe Coding 工具栈                   │
│                                                     │
│  ┌──────────────┐  ┌──────────────┐                │
│  │   AI 助手    │  │  版本控制    │                │
│  │  Claude/GPT  │  │    Git       │                │
│  └──────────────┘  └──────────────┘                │
│                                                     │
│  ┌──────────────┐  ┌──────────────┐                │
│  │  测试框架    │  │  代码质量    │                │
│  │  pytest      │  │  ruff/mypy   │                │
│  └──────────────┘  └──────────────┘                │
│                                                     │
│  ┌──────────────┐  ┌──────────────┐                │
│  │  DSL 解析    │  │  领域建模    │                │
│  │  lark-parser │  │  pydantic    │                │
│  └──────────────┘  └──────────────┘                │
└─────────────────────────────────────────────────────┘
```

---

## 4.2 Python 环境配置

### 推荐的项目结构

```
my-domain-project/
├── pyproject.toml           # 项目配置（取代 setup.py）
├── src/
│   └── myproject/
│       ├── __init__.py
│       ├── domain/          # 领域层（DDD 核心）
│       │   ├── __init__.py
│       │   ├── models.py    # 领域模型
│       │   ├── events.py    # 领域事件
│       │   └── services.py  # 领域服务
│       ├── dsl/             # DSL 层
│       │   ├── __init__.py
│       │   ├── internal.py  # 内部 DSL（流式接口）
│       │   └── external.py  # 外部 DSL（语法解析）
│       └── application/     # 应用层
│           ├── __init__.py
│           └── use_cases.py
├── tests/
│   ├── unit/               # 单元测试（TDD 驱动）
│   ├── integration/        # 集成测试
│   └── acceptance/         # 验收测试（用 DSL 写）
└── docs/
    └── ubiquitous-language.md  # DDD 统一语言词汇表
```

### pyproject.toml 配置

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "my-domain-project"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "pydantic>=2.0",       # 领域模型验证
    "lark>=1.1",           # 外部 DSL 解析
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov",          # 测试覆盖率
    "pytest-watch",        # 自动重跑测试（TDD 神器）
    "ruff",                # 快速 linter
    "mypy",                # 类型检查
    "hypothesis",          # 基于属性的测试
]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"

[tool.ruff]
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "I"]  # pycodestyle + pyflakes + isort

[tool.mypy]
python_version = "3.10"
strict = true
```

### 快速安装

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 安装依赖
pip install -e ".[dev]"

# 验证安装
pytest --version
python -c "import lark; print(lark.__version__)"
```

---

## 4.3 pytest 配置（TDD 必备）

### conftest.py 模板

```python
# tests/conftest.py
import pytest
from typing import Generator

# 共享的测试夹具
@pytest.fixture
def money_factory():
    """创建 Money 值对象的工厂"""
    def create(amount: float, currency: str = "CNY"):
        from myproject.domain.models import Money
        return Money(amount=amount, currency=currency)
    return create

@pytest.fixture
def customer_factory():
    """创建 Customer 聚合根的工厂"""
    def create(tier: str = "normal", customer_id: str = "test-customer-001"):
        from myproject.domain.models import Customer, CustomerTier, CustomerId
        return Customer(
            id=CustomerId(customer_id),
            tier=CustomerTier(tier)
        )
    return create

# 共享的领域事件收集器
@pytest.fixture
def event_collector():
    """收集测试中产生的领域事件"""
    events = []
    
    class Collector:
        def collect(self, event):
            events.append(event)
        
        def of_type(self, event_type):
            return [e for e in events if isinstance(e, event_type)]
        
        @property
        def all(self):
            return events.copy()
    
    return Collector()
```

### pytest-watch 自动测试（TDD 心流关键）

```bash
# 安装
pip install pytest-watch

# 启动自动监视
ptw -- -v

# 这会在你保存文件时自动重跑测试
# TDD 心流：改代码 → 自动跑测试 → 看红绿灯
```

### 测试文件命名约定（体现 TDD 意图）

```
tests/
├── unit/
│   ├── domain/
│   │   ├── test_order_placement.py      # 按业务场景命名
│   │   ├── test_order_confirmation.py
│   │   ├── test_pricing_rules.py
│   │   └── test_customer_registration.py
│   └── dsl/
│       ├── test_pricing_dsl.py
│       └── test_query_dsl.py
└── acceptance/
    └── test_order_workflow_acceptance.py  # 用 DSL 写的验收测试
```

---

## 4.4 AI 工具配置

### Claude Code / Cursor 配置

创建 `CLAUDE.md` 或 `.cursorrules` 文件，给 AI 提供领域上下文：

```markdown
# 项目上下文

## 领域概念（统一语言）
- **Customer**：在系统中注册的用户，有 NORMAL/VIP/PREMIUM 三个等级
- **Order**：客户下的订单，包含多个 OrderItem
- **Money**：金额值对象，必须包含 amount 和 currency
- **OrderItem**：订单中的商品行项目

## 编码约定
1. 所有领域对象使用 dataclass 或 pydantic BaseModel
2. 值对象必须是不可变的（frozen=True）
3. 聚合根方法返回领域事件，不直接修改外部状态
4. 先写测试，再写实现（TDD）

## 禁止事项
- 不在领域层使用 ORM 对象
- 不在领域层导入框架（Flask/Django/SQLAlchemy）
- 不使用全局状态

## 测试风格
- 测试名用 test_[场景]_[期望结果] 格式
- 每个测试只验证一件事
- 使用中文注释描述业务场景
```

### AI 提示词模板库

创建 `prompts/` 目录存放复用的提示词：

```markdown
# prompts/implement-domain-service.md

## 任务
实现领域服务 {ServiceName}

## 领域模型（已有）
```python
{粘贴相关领域对象}
```

## 测试规约（已写好）
```python
{粘贴测试代码}
```

## 实现要求
1. 通过所有测试
2. 遵循上面的编码约定
3. 方法返回类型必须明确标注
4. 不使用全局状态或副作用
5. 如果需要外部依赖，通过构造函数注入

## 输出格式
只输出代码，不需要解释。
```

---

## 4.5 Git 工作流配置

### TDD 友好的 Git 配置

```bash
# .gitconfig 别名
[alias]
    # 红灯提交（测试失败）
    red = commit -m "🔴 RED: "
    # 绿灯提交（测试通过）
    green = commit -m "🟢 GREEN: "
    # 重构提交
    refactor = commit -m "♻️ REFACTOR: "
    # 查看 TDD 循环历史
    tdd-log = log --oneline --grep="🔴\|🟢\|♻️"
```

### pre-commit 钩子（保证代码质量）

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.0  # ⚠️ 示例版本，请运行 pre-commit autoupdate 获取最新版本
    hooks:
      - id: ruff
        args: [--fix]
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.6.0  # ⚠️ 示例版本，请运行 pre-commit autoupdate 获取最新版本
    hooks:
      - id: mypy
        additional_dependencies: [pydantic]
  
  - repo: local
    hooks:
      - id: pytest-check
        name: pytest quick check
        entry: pytest tests/unit -x -q
        language: system
        pass_filenames: false
        always_run: true
```

```bash
# 安装 pre-commit
pip install pre-commit
pre-commit install
```

---

## 4.6 领域建模工具

### Pydantic v2 作为领域模型基础

```python
# src/myproject/domain/base.py
from pydantic import BaseModel, ConfigDict
from typing import ClassVar, List, Any

class ValueObject(BaseModel):
    """值对象基类：不可变，按值比较"""
    model_config = ConfigDict(frozen=True)

class Entity(BaseModel):
    """实体基类：可变，按 ID 比较"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    id: Any  # 子类定义具体类型
    
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.id == other.id
    
    def __hash__(self):
        return hash(self.id)

class AggregateRoot(Entity):
    """聚合根基类：持有领域事件"""
    # 使用 PrivateAttr 确保每个实例有独立的事件列表
    _events: List = PrivateAttr(default_factory=list)
    
    def _record_event(self, event: Any) -> None:
        self._events.append(event)
    
    def pull_events(self) -> List:
        """取出并清空领域事件（用于发布）"""
        events = self._events.copy()
        self._events.clear()
        return events
```

---

## 4.7 快速验证安装

创建一个端到端的验证脚本：

```python
# scripts/verify_setup.py
"""验证 Vibe Coding 工具栈安装正确"""

def test_domain_model():
    """验证 Pydantic 领域模型"""
    from pydantic import BaseModel, ConfigDict
    
    class Money(BaseModel):
        model_config = ConfigDict(frozen=True)
        amount: float
        currency: str
    
    m1 = Money(amount=100, currency="CNY")
    m2 = Money(amount=100, currency="CNY")
    assert m1 == m2, "值对象相等性验证失败"
    print("✅ Pydantic 领域模型正常")

def test_dsl_parser():
    """验证 Lark DSL 解析器"""
    from lark import Lark
    
    grammar = """
    expr: NUMBER "+" NUMBER
    NUMBER: /\\d+/
    %ignore " "
    """
    parser = Lark(grammar, start="expr")
    tree = parser.parse("1 + 2")
    assert tree is not None
    print("✅ Lark DSL 解析器正常")

def test_pytest():
    """验证 pytest"""
    import pytest
    print(f"✅ pytest {pytest.__version__} 正常")

if __name__ == "__main__":
    test_domain_model()
    test_dsl_parser()
    test_pytest()
    print("\n🎉 所有工具安装正确，可以开始 Vibe Coding！")
```

```bash
python scripts/verify_setup.py
```

---

## 4.8 IDE 推荐配置

### VS Code 扩展

```json
// .vscode/extensions.json
{
    "recommendations": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "charliermarsh.ruff",
        "littlefoxteam.vscode-python-test-adapter",
        "eamodio.gitlens"
    ]
}
```

### VS Code 设置

```json
// .vscode/settings.json
{
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests", "-v"],
    "editor.formatOnSave": true,
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff"
    },
    // TDD 关键设置：保存时自动运行相关测试
    "python.testing.autoTestDiscoverOnSaveEnabled": true
}
```

---

## 总结

本章配置的工具栈支持：
- **TDD**：pytest + pytest-watch 实现保存即测试的快速反馈循环
- **DDD**：Pydantic v2 提供不可变值对象和聚合根基类
- **DSL**：Lark-parser 支持外部 DSL；Python 本身支持内部 DSL
- **AI 协作**：CLAUDE.md 给 AI 注入领域上下文，提示词模板复用

---

**下一章**：[Red-Green-Refactor 深度实践](../part2-tdd-backbone/05-red-green-refactor-deep.md)
