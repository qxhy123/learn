# 第12章：Python魔法方法DSL

## 核心思维模型

> Python的魔法方法（dunder methods）是内部DSL的**秘密武器**：通过重载运算符和控制协议，让普通Python对象"变身"为DSL元素。关键是**克制**——只重载有明确领域含义的操作符，不要让代码变成谜题。

---

## 12.1 操作符重载构建表达式树

这是内部DSL中最强大的技术之一：让Python的比较/算术/逻辑运算符构建DSL的AST节点，而不是执行实际计算。

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass
class Expr:
    """所有DSL表达式的基类"""
    pass

@dataclass
class Column(Expr):
    """表示数据库列或数据字段"""
    name: str
    
    # ─── 比较运算符 ────────────────────────────────────────
    def __eq__(self, other) -> 'BinaryExpr':
        return BinaryExpr(self, "=", _wrap(other))
    
    def __ne__(self, other) -> 'BinaryExpr':
        return BinaryExpr(self, "!=", _wrap(other))
    
    def __gt__(self, other) -> 'BinaryExpr':
        return BinaryExpr(self, ">", _wrap(other))
    
    def __ge__(self, other) -> 'BinaryExpr':
        return BinaryExpr(self, ">=", _wrap(other))
    
    def __lt__(self, other) -> 'BinaryExpr':
        return BinaryExpr(self, "<", _wrap(other))
    
    def __le__(self, other) -> 'BinaryExpr':
        return BinaryExpr(self, "<=", _wrap(other))
    
    # ─── 字符串操作（利用in运算符模拟）────────────────────
    def like(self, pattern: str) -> 'BinaryExpr':
        return BinaryExpr(self, "LIKE", Literal(pattern))
    
    def in_(self, values: list) -> 'InExpr':
        return InExpr(self, [_wrap(v) for v in values])
    
    def between(self, low, high) -> 'BetweenExpr':
        return BetweenExpr(self, _wrap(low), _wrap(high))
    
    def is_null(self) -> 'IsNullExpr':
        return IsNullExpr(self)
    
    def is_not_null(self) -> 'IsNullExpr':
        return IsNullExpr(self, negated=True)
    
    # ─── 算术运算符（计算列） ──────────────────────────────
    def __add__(self, other) -> 'BinaryExpr':
        return BinaryExpr(self, "+", _wrap(other))
    
    def __mul__(self, other) -> 'BinaryExpr':
        return BinaryExpr(self, "*", _wrap(other))
    
    def __repr__(self):
        return f"Column({self.name!r})"

@dataclass
class Literal(Expr):
    value: Any
    
    def __repr__(self):
        if isinstance(self.value, str):
            return f"'{self.value}'"
        return str(self.value)

@dataclass
class BinaryExpr(Expr):
    left: Expr
    op: str
    right: Expr
    
    # 支持与其他表达式的逻辑组合
    def __and__(self, other: Expr) -> 'BinaryExpr':
        return BinaryExpr(self, "AND", other)
    
    def __or__(self, other: Expr) -> 'BinaryExpr':
        return BinaryExpr(self, "OR", other)
    
    def __invert__(self) -> 'NotExpr':
        return NotExpr(self)
    
    def to_sql(self) -> str:
        return f"({self.left.to_sql()} {self.op} {self.right.to_sql()})"

@dataclass
class InExpr(Expr):
    column: Expr
    values: list[Expr]
    
    def __and__(self, other): return BinaryExpr(self, "AND", other)
    def __or__(self, other): return BinaryExpr(self, "OR", other)

@dataclass
class BetweenExpr(Expr):
    column: Expr
    low: Expr
    high: Expr

@dataclass
class IsNullExpr(Expr):
    column: Expr
    negated: bool = False

@dataclass
class NotExpr(Expr):
    expr: Expr

def _wrap(value) -> Expr:
    """将Python字面量包装为Expr节点"""
    if isinstance(value, Expr):
        return value
    return Literal(value)

# 为每个Expr子类添加to_sql方法（或用访问者）
def expr_to_sql(expr: Expr) -> str:
    match expr:
        case Column(name=n):
            return n
        case Literal(value=v):
            return f"'{v}'" if isinstance(v, str) else str(v)
        case BinaryExpr(left=l, op=op, right=r):
            return f"({expr_to_sql(l)} {op} {expr_to_sql(r)})"
        case InExpr(column=c, values=vs):
            vals = ", ".join(expr_to_sql(v) for v in vs)
            return f"{expr_to_sql(c)} IN ({vals})"
        case BetweenExpr(column=c, low=lo, high=hi):
            return f"{expr_to_sql(c)} BETWEEN {expr_to_sql(lo)} AND {expr_to_sql(hi)}"
        case IsNullExpr(column=c, negated=False):
            return f"{expr_to_sql(c)} IS NULL"
        case IsNullExpr(column=c, negated=True):
            return f"{expr_to_sql(c)} IS NOT NULL"
        case NotExpr(expr=e):
            return f"NOT ({expr_to_sql(e)})"
        case _:
            raise ValueError(f"未知表达式类型: {type(expr)}")
```

### 使用效果

```python
# 定义列
age = Column("age")
status = Column("status")
name = Column("name")
email = Column("email")
score = Column("score")

# Python表达式 → DSL条件
cond1 = age > 18
cond2 = status == "active"
cond3 = (age > 18) & (status == "active")           # AND
cond4 = (age < 18) | (status == "inactive")         # OR
cond5 = ~(status == "banned")                        # NOT
cond6 = status.in_(["active", "pending"])
cond7 = age.between(18, 65)
cond8 = email.is_not_null()
cond9 = score.like("%excellent%")

# 转SQL
print(expr_to_sql(cond3))
# (age > 18) AND (status = 'active')
print(expr_to_sql(cond6))
# status IN ('active', 'pending')
print(expr_to_sql(cond7))
# age BETWEEN 18 AND 65
```

---

## 12.2 `__or__` 管道操作符DSL

Python的 `|` 运算符（位或）可以重载为管道操作符：

```python
from typing import TypeVar, Generic, Callable

T = TypeVar('T')
U = TypeVar('U')

class Pipeline(Generic[T]):
    """可管道化的数据处理容器"""
    
    def __init__(self, value: T):
        self._value = value
    
    def __or__(self, func: Callable[[T], U]) -> 'Pipeline[U]':
        """管道：将当前值传递给函数"""
        return Pipeline(func(self._value))
    
    def result(self) -> T:
        return self._value
    
    def __repr__(self):
        return f"Pipeline({self._value!r})"


class Transform:
    """DSL变换操作，支持管道"""
    
    def __init__(self, func: Callable):
        self._func = func
        self.__name__ = func.__name__
    
    def __call__(self, data):
        return self._func(data)
    
    def __ror__(self, other) -> 'Pipeline':
        """支持 data | transform 语法"""
        if isinstance(other, Pipeline):
            return other | self
        return Pipeline(self(other))


# 定义数据处理变换
@Transform
def filter_active(users):
    return [u for u in users if u.get("status") == "active"]

@Transform  
def sort_by_name(users):
    return sorted(users, key=lambda u: u["name"])

def select_fields(*fields):
    def transform(users):
        return [{f: u[f] for f in fields if f in u} for u in users]
    transform.__name__ = f"select({', '.join(fields)})"
    return Transform(transform)

def limit(n: int):
    def transform(data):
        return data[:n]
    transform.__name__ = f"limit({n})"
    return Transform(transform)


# 使用：Unix管道风格
users = [
    {"name": "Charlie", "status": "active", "email": "c@example.com"},
    {"name": "Alice", "status": "active", "email": "a@example.com"},
    {"name": "Bob", "status": "inactive", "email": "b@example.com"},
]

result = (
    Pipeline(users)
    | filter_active
    | sort_by_name
    | select_fields("name", "email")
    | limit(10)
).result()

# 或者更简洁（利用__ror__）：
result2 = users | filter_active | sort_by_name | select_fields("name") | limit(10)
```

---

## 12.3 装饰器DSL

装饰器是Python内部DSL最常用的机制，特别适合**注册和元数据模式**：

```python
from functools import wraps
from typing import Callable, TypeVar
import inspect

F = TypeVar('F', bound=Callable)

class CommandRegistry:
    """命令注册DSL"""
    
    def __init__(self):
        self._commands: dict[str, dict] = {}
    
    def command(
        self, 
        name: str = None,
        *,
        description: str = "",
        aliases: list[str] = None,
        requires_auth: bool = False,
    ):
        """
        命令注册装饰器
        
        使用：
            @registry.command("deploy", description="部署应用")
            def deploy_app(env: str, version: str = "latest"):
                ...
        """
        def decorator(func: F) -> F:
            cmd_name = name or func.__name__
            
            # 从函数签名提取参数信息
            sig = inspect.signature(func)
            params = {
                p_name: {
                    "type": p.annotation if p.annotation != inspect.Parameter.empty else Any,
                    "default": p.default if p.default != inspect.Parameter.empty else None,
                    "required": p.default == inspect.Parameter.empty,
                }
                for p_name, p in sig.parameters.items()
            }
            
            self._commands[cmd_name] = {
                "func": func,
                "description": description,
                "aliases": aliases or [],
                "requires_auth": requires_auth,
                "params": params,
            }
            
            # 同时注册别名
            for alias in (aliases or []):
                self._commands[alias] = self._commands[cmd_name]
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                if requires_auth and not is_authenticated():
                    raise PermissionError(f"命令 '{cmd_name}' 需要认证")
                return func(*args, **kwargs)
            
            return wrapper
        
        # 支持无括号调用：@registry.command（而非@registry.command()）
        if callable(name):
            func = name
            name = None
            return decorator(func)
        
        return decorator
    
    def execute(self, cmd: str, *args, **kwargs):
        if cmd not in self._commands:
            similar = [c for c in self._commands if cmd in c]
            hint = f"，是否想要: {similar[0]!r}" if similar else ""
            raise KeyError(f"未知命令: {cmd!r}{hint}")
        return self._commands[cmd]["func"](*args, **kwargs)
    
    def help(self) -> str:
        lines = ["可用命令：\n"]
        for name, info in self._commands.items():
            if name in [a for cmd in self._commands.values() for a in cmd["aliases"]]:
                continue  # 跳过别名
            aliases = f" ({', '.join(info['aliases'])})" if info['aliases'] else ""
            lines.append(f"  {name}{aliases}: {info['description']}")
        return "\n".join(lines)


# 定义命令注册中心
cli = CommandRegistry()

@cli.command("deploy", description="部署应用到指定环境", aliases=["dp"])
def deploy(env: str, version: str = "latest", dry_run: bool = False):
    print(f"部署 {version} 到 {env}{'（演练模式）' if dry_run else ''}")

@cli.command("rollback", description="回滚到上一个版本", requires_auth=True)
def rollback(env: str, steps: int = 1):
    print(f"回滚 {env} 最近 {steps} 个版本")

@cli.command  # 不带括号的简洁用法
def status():
    """查看当前部署状态"""
    print("系统运行正常")

# 使用
cli.execute("deploy", env="production", version="v2.1.0")
cli.execute("dp", env="staging")  # 使用别名
print(cli.help())
```

---

## 12.4 上下文管理器DSL

`with`语句天然适合表达"在...范围内"的DSL语义：

```python
from contextlib import contextmanager, ExitStack
from typing import Generator

class TransactionManager:
    """事务DSL：with语句管理数据库事务"""
    
    def __init__(self, db):
        self._db = db
        self._savepoints: list[str] = []
    
    def __enter__(self):
        self._db.begin()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self._db.commit()
        else:
            self._db.rollback()
        return False  # 不吞噬异常
    
    def savepoint(self, name: str):
        """嵌套事务支持"""
        return SavepointContext(self._db, name)


class SavepointContext:
    def __init__(self, db, name):
        self._db = db
        self._name = name
    
    def __enter__(self):
        self._db.execute(f"SAVEPOINT {self._name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self._db.execute(f"ROLLBACK TO SAVEPOINT {self._name}")
        else:
            self._db.execute(f"RELEASE SAVEPOINT {self._name}")
        return False


# 使用：事务作用域清晰
with TransactionManager(db) as tx:
    create_user("Alice")
    
    with tx.savepoint("order_creation"):
        create_order(user_id=1, amount=100)
        deduct_inventory(product_id=5, qty=1)
    
    send_welcome_email("Alice")  # 即使这里失败，整个事务仍然回滚


# ─── 测试夹具DSL ─────────────────────────────────────────────

class TestEnvironment:
    """测试环境管理DSL"""
    
    def __init__(self):
        self._resources: list = []
    
    @contextmanager
    def database(self, *, schema: str = "test") -> Generator:
        """提供临时测试数据库"""
        db = create_test_database(schema)
        self._resources.append(db)
        try:
            yield db
        finally:
            db.drop()
            self._resources.remove(db)
    
    @contextmanager
    def mock_service(self, service_name: str, responses: dict) -> Generator:
        """模拟外部服务"""
        mock = MockService(service_name, responses)
        mock.start()
        try:
            yield mock
        finally:
            mock.stop()
    
    @contextmanager
    def as_user(self, user_id: int) -> Generator:
        """在特定用户上下文中执行"""
        original = get_current_user()
        set_current_user(user_id)
        try:
            yield
        finally:
            set_current_user(original)


env = TestEnvironment()

# 测试用例中的DSL：
def test_user_creates_order():
    with env.database() as db, \
         env.mock_service("payment", {"charge": True}) as payment, \
         env.as_user(user_id=42):
        
        order = create_order(amount=100)
        assert order.status == "confirmed"
        assert payment.was_called_with("charge", amount=100)
```

---

## 12.5 `__getattr__` 动态DSL

```python
class DynamicQuery:
    """
    通过__getattr__动态创建查询条件：
    q.age_gt(18)    → WHERE age > 18
    q.status_eq("active") → WHERE status = 'active'
    """
    
    def __init__(self, table: str):
        self._table = table
        self._conditions: list = []
    
    def __getattr__(self, name: str):
        # 解析 field_op 格式
        op_map = {
            'eq': '=', 'ne': '!=', 'gt': '>', 'gte': '>=',
            'lt': '<', 'lte': '<=', 'like': 'LIKE', 'in': 'IN',
        }
        
        for suffix, op in op_map.items():
            if name.endswith(f'_{suffix}'):
                field = name[:-len(suffix)-1]
                def make_condition(f, o):
                    def condition_fn(value):
                        self._conditions.append((f, o, value))
                        return self
                    return condition_fn
                return make_condition(field, op)
        
        raise AttributeError(
            f"未知查询方法: {name!r}。"
            f"支持的格式: fieldname_{{eq|ne|gt|gte|lt|lte|like|in}}"
        )
    
    def to_sql(self) -> str:
        select = f"SELECT * FROM {self._table}"
        if not self._conditions:
            return select
        
        parts = []
        for field, op, value in self._conditions:
            if isinstance(value, str):
                parts.append(f"{field} {op} '{value}'")
            elif isinstance(value, list):
                vals = ", ".join(f"'{v}'" if isinstance(v, str) else str(v) for v in value)
                parts.append(f"{field} {op} ({vals})")
            else:
                parts.append(f"{field} {op} {value}")
        
        return f"{select}\nWHERE {' AND '.join(parts)}"


# 使用：
q = DynamicQuery("users")
q.age_gt(18).status_eq("active").name_like("Alice%")
print(q.to_sql())
# SELECT * FROM users
# WHERE age > 18 AND status = 'active' AND name LIKE 'Alice%'
```

---

## 12.6 魔法方法DSL的安全边界

**不应该重载的操作符**：

```python
# ❌ 不要重载 __bool__（会破坏if语句）
class Condition:
    def __bool__(self):
        ...  # 这会让 "if condition:" 行为异常

# ❌ 不要重载 __hash__（配合__eq__重载时会失去hashability）
class Column:
    def __eq__(self, other):
        return BinaryExpr(self, "=", other)
    # 必须同时处理：
    __hash__ = None  # 明确声明不可哈希，避免误用
    # 或者提供ID-based hash：
    # def __hash__(self): return id(self)

# ✅ 安全的重载
class Condition:
    def __and__(self, other): ...  # &
    def __or__(self, other): ...   # |
    def __invert__(self): ...      # ~
    def __repr__(self): ...        # repr()
```

---

## 小结

| 魔法方法 | DSL用途 | 示例 |
|---------|---------|------|
| `__eq__/__gt__等` | 构建比较表达式树 | `age > 18 → BinaryExpr` |
| `__and__/__or__/__invert__` | 逻辑组合表达式 | `(a > 1) & (b < 10)` |
| `__or__` (作为管道) | Unix风格数据管道 | `data \| filter \| sort` |
| `__getattr__` | 动态方法生成 | `q.age_gt(18)` |
| `__enter__/__exit__` | 作用域DSL | `with transaction():` |
| 装饰器 | 注册和元数据 | `@command("deploy")` |

---

**上一章**：[流畅接口设计](./11-fluent-interface.md)
**下一章**：[外部DSL全流程](../part4-external-dsl/13-external-dsl-pipeline.md)
