# 第10章：构建者模式DSL

## 核心思维模型

> 构建者模式是内部DSL中**最类型安全**的方式：它通过分步构造保证最终对象的完整性，利用Python的类型系统（或运行时检查）在`build()`时验证所有约束。

---

## 10.1 经典Builder vs DSL Builder

### 经典GoF Builder（过于冗长）

```python
# 传统四人帮构建者模式（Java风格）
builder = QueryBuilder()
builder.setTable("users")
builder.setCondition(Condition("age", ">", 18))
builder.setFields(["name", "email"])
builder.setLimit(10)
query = builder.build()
```

### DSL Builder（流畅、可读）

```python
# 现代DSL风格
query = (
    QueryBuilder("users")
    .where(age > 18)
    .select("name", "email")
    .limit(10)
    .build()
)
```

关键区别：DSL Builder的方法名是**领域术语**，方法调用是**自描述**的，读起来像需求文档。

---

## 10.2 不可变Builder的实现

**设计原则**：每次方法调用返回**新Builder实例**，而非修改当前实例。这使Builder线程安全，且支持分叉：

```python
from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Optional

@dataclass(frozen=True)  # frozen=True → 不可变
class QuerySpec:
    """最终构建的不可变查询规格"""
    table: str
    conditions: tuple = ()
    fields: tuple = ('*',)
    order_field: Optional[str] = None
    order_dir: str = "ASC"
    limit_count: Optional[int] = None
    offset_count: int = 0

class QueryBuilder:
    """不可变Builder：每次操作返回新实例"""
    
    def __init__(self, table: str):
        self._spec = QuerySpec(table=table)
    
    def _clone(self, **kwargs) -> 'QueryBuilder':
        """创建新Builder，spec使用更新后的值"""
        new_builder = QueryBuilder.__new__(QueryBuilder)
        new_builder._spec = replace(self._spec, **kwargs)
        return new_builder
    
    def where(self, condition) -> 'QueryBuilder':
        """添加WHERE条件（累加，与已有条件AND合并）"""
        new_conditions = self._spec.conditions + (condition,)
        return self._clone(conditions=new_conditions)
    
    def select(self, *fields: str) -> 'QueryBuilder':
        return self._clone(fields=fields)
    
    def order_by(self, field: str, direction: str = "ASC") -> 'QueryBuilder':
        if direction.upper() not in ("ASC", "DESC"):
            raise ValueError(f"排序方向必须是ASC或DESC，得到: {direction}")
        return self._clone(order_field=field, order_dir=direction.upper())
    
    def limit(self, n: int) -> 'QueryBuilder':
        if n <= 0:
            raise ValueError(f"LIMIT必须大于0，得到: {n}")
        return self._clone(limit_count=n)
    
    def offset(self, n: int) -> 'QueryBuilder':
        if n < 0:
            raise ValueError(f"OFFSET不能为负数，得到: {n}")
        return self._clone(offset_count=n)
    
    def build(self) -> QuerySpec:
        """验证并返回不可变QuerySpec"""
        if not self._spec.table:
            raise ValueError("必须指定表名")
        return self._spec
    
    def to_sql(self) -> str:
        """直接生成SQL字符串（不需要先build）"""
        spec = self._spec
        parts = [f"SELECT {', '.join(spec.fields)}"]
        parts.append(f"FROM {spec.table}")
        
        if spec.conditions:
            where_parts = [str(c) for c in spec.conditions]
            parts.append(f"WHERE {' AND '.join(where_parts)}")
        
        if spec.order_field:
            parts.append(f"ORDER BY {spec.order_field} {spec.order_dir}")
        
        if spec.limit_count:
            parts.append(f"LIMIT {spec.limit_count}")
        
        if spec.offset_count:
            parts.append(f"OFFSET {spec.offset_count}")
        
        return '\n'.join(parts)
```

### 不可变Builder的分叉特性

```python
# 基础查询
base_query = QueryBuilder("users").where(age > 18)

# 分叉：两个方向，互不影响
admin_query = base_query.where(role == "admin").select("*")
email_query = base_query.select("email").limit(100)

print(admin_query.to_sql())
# SELECT *
# FROM users
# WHERE age > 18 AND role = admin

print(email_query.to_sql())
# SELECT email
# FROM users
# WHERE age > 18
# LIMIT 100
```

---

## 10.3 类型安全的阶段性Builder

有些对象要求特定的构造顺序。可以用**类型系统强制构造阶段**：

```python
from typing import Protocol

# 阶段标记类（表示构造进度）
class WithTable(Protocol):
    def where(self, condition) -> 'WithConditionOrSelect': ...
    def select(self, *fields) -> 'WithSelect': ...

class WithConditionOrSelect(Protocol):
    def where(self, condition) -> 'WithConditionOrSelect': ...
    def select(self, *fields) -> 'WithSelect': ...
    def build(self) -> QuerySpec: ...

class WithSelect(Protocol):
    def order_by(self, field: str, direction: str = "ASC") -> 'WithSelect': ...
    def limit(self, n: int) -> 'WithSelect': ...
    def build(self) -> QuerySpec: ...

# 实现：在不同阶段返回不同类型
class TableStage:
    """第一阶段：只有表名"""
    def __init__(self, table: str):
        self._table = table
    
    def where(self, condition) -> 'ConditionStage':
        return ConditionStage(self._table, [condition])
    
    def select(self, *fields: str) -> 'SelectStage':
        return SelectStage(self._table, [], list(fields))

class ConditionStage:
    """中间阶段：有条件但未SELECT"""
    def __init__(self, table, conditions):
        self._table = table
        self._conditions = conditions
    
    def where(self, condition) -> 'ConditionStage':
        return ConditionStage(self._table, self._conditions + [condition])
    
    def select(self, *fields: str) -> 'SelectStage':
        return SelectStage(self._table, self._conditions, list(fields))
    
    # 注意：这个阶段没有 .build() 方法 → 强制必须有SELECT

class SelectStage:
    """最终阶段：可以build"""
    def __init__(self, table, conditions, fields):
        self._table = table
        self._conditions = conditions
        self._fields = fields
        self._order_field = None
        self._order_dir = "ASC"
        self._limit = None
    
    def order_by(self, field: str, direction: str = "ASC") -> 'SelectStage':
        new = SelectStage(self._table, self._conditions, self._fields)
        new._order_field = field
        new._order_dir = direction
        new._limit = self._limit
        return new
    
    def limit(self, n: int) -> 'SelectStage':
        new = SelectStage(self._table, self._conditions, self._fields)
        new._order_field = self._order_field
        new._order_dir = self._order_dir
        new._limit = n
        return new
    
    def build(self) -> QuerySpec:
        return QuerySpec(
            table=self._table,
            conditions=tuple(self._conditions),
            fields=tuple(self._fields) if self._fields else ('*',),
            order_field=self._order_field,
            order_dir=self._order_dir,
            limit_count=self._limit,
        )

# 使用：
def Q(table: str) -> TableStage:
    return TableStage(table)

# ✅ 正确：有SELECT才能build
query = Q("users").where(age > 18).select("name").build()

# ✅ 正确：可以直接SELECT
query2 = Q("orders").select("id", "amount").limit(10).build()

# ❌ 类型错误（Python运行时）：ConditionStage没有.build()
# query3 = Q("users").where(age > 18).build()  # AttributeError!
```

---

## 10.4 带验证的Builder

Builder的`build()`方法是执行所有验证的最佳时机：

```python
@dataclass
class EmailConfig:
    smtp_host: str
    smtp_port: int
    username: str
    password: str
    use_tls: bool = True
    timeout_seconds: int = 30

class EmailConfigBuilder:
    def __init__(self):
        self._host = None
        self._port = None
        self._username = None
        self._password = None
        self._use_tls = True
        self._timeout = 30
    
    def host(self, value: str) -> 'EmailConfigBuilder':
        self._host = value
        return self
    
    def port(self, value: int) -> 'EmailConfigBuilder':
        self._port = value
        return self
    
    def credentials(self, username: str, password: str) -> 'EmailConfigBuilder':
        self._username = username
        self._password = password
        return self
    
    def with_tls(self, enabled: bool = True) -> 'EmailConfigBuilder':
        self._use_tls = enabled
        return self
    
    def timeout(self, seconds: int) -> 'EmailConfigBuilder':
        self._timeout = seconds
        return self
    
    def build(self) -> EmailConfig:
        errors = []
        
        if not self._host:
            errors.append("smtp_host 未设置")
        
        if self._port is None:
            errors.append("smtp_port 未设置")
        elif not (1 <= self._port <= 65535):
            errors.append(f"smtp_port 必须在 1-65535 范围内，得到: {self._port}")
        
        if not self._username:
            errors.append("username 未设置")
        
        if not self._password:
            errors.append("password 未设置")
        
        if self._timeout <= 0:
            errors.append(f"timeout 必须大于0，得到: {self._timeout}")
        
        # 逻辑验证：TLS通常用443/587/465
        if self._use_tls and self._port and self._port == 80:
            errors.append("启用TLS时不应使用HTTP端口80")
        
        if errors:
            raise ValueError("EmailConfig配置错误:\n" + "\n".join(f"  - {e}" for e in errors))
        
        return EmailConfig(
            smtp_host=self._host,
            smtp_port=self._port,
            username=self._username,
            password=self._password,
            use_tls=self._use_tls,
            timeout_seconds=self._timeout,
        )

# 使用
email_config = (
    EmailConfigBuilder()
    .host("smtp.gmail.com")
    .port(587)
    .credentials("user@gmail.com", "app_password")
    .with_tls(True)
    .timeout(30)
    .build()
)
```

---

## 10.5 Builder与工厂方法结合

```python
class RequestBuilder:
    """HTTP请求Builder，工厂方法用于常见HTTP方法"""
    
    def __init__(self, method: str, url: str):
        self._method = method
        self._url = url
        self._headers: dict = {}
        self._params: dict = {}
        self._body = None
    
    # 工厂方法（静态快捷入口）
    @classmethod
    def get(cls, url: str) -> 'RequestBuilder':
        return cls("GET", url)
    
    @classmethod
    def post(cls, url: str) -> 'RequestBuilder':
        return cls("POST", url)
    
    @classmethod
    def put(cls, url: str) -> 'RequestBuilder':
        return cls("PUT", url)
    
    @classmethod
    def delete(cls, url: str) -> 'RequestBuilder':
        return cls("DELETE", url)
    
    # 配置方法
    def header(self, key: str, value: str) -> 'RequestBuilder':
        self._headers[key] = value
        return self
    
    def bearer_token(self, token: str) -> 'RequestBuilder':
        return self.header("Authorization", f"Bearer {token}")
    
    def json_body(self, data: dict) -> 'RequestBuilder':
        import json
        self._body = json.dumps(data)
        return self.header("Content-Type", "application/json")
    
    def param(self, key: str, value: str) -> 'RequestBuilder':
        self._params[key] = value
        return self
    
    def build(self):
        import urllib.parse
        url = self._url
        if self._params:
            url += "?" + urllib.parse.urlencode(self._params)
        return {
            "method": self._method,
            "url": url,
            "headers": self._headers,
            "body": self._body,
        }

# 使用：接近HTTP DSL
request = (
    RequestBuilder.post("https://api.example.com/users")
    .bearer_token("my_token_123")
    .json_body({"name": "Alice", "age": 25})
    .build()
)
```

---

## 10.6 Builder的测试策略

```python
import pytest

class TestQueryBuilder:
    def test_minimal_query(self):
        """最简用法"""
        q = Q("users").select("*").build()
        assert q.table == "users"
        assert q.fields == ('*',)
    
    def test_with_conditions(self):
        """多条件AND合并"""
        age = Column("age")
        status = Column("status")
        q = (Q("users")
             .where(age > 18)
             .where(status == "active")
             .select("name")
             .build())
        assert len(q.conditions) == 2
    
    def test_immutability(self):
        """不可变性：操作不影响原Builder"""
        base = Q("users").where(Column("age") > 18)
        q1 = base.select("name").build()
        q2 = base.select("email").build()
        assert q1.fields == ('name',)
        assert q2.fields == ('email',)
    
    def test_build_validates(self):
        """build()触发验证"""
        with pytest.raises(ValueError, match="LIMIT必须大于0"):
            Q("users").select("*").limit(-1).build()
    
    def test_fork_independence(self):
        """分叉独立性"""
        base = Q("orders").where(Column("amount") > 100)
        admin = base.where(Column("role") == "admin").select("*").build()
        public = base.select("id", "amount").build()
        
        assert len(admin.conditions) == 2
        assert len(public.conditions) == 1
```

---

## 小结

| 概念 | 要点 |
|------|------|
| 不可变Builder | 每次方法调用返回新实例，支持分叉，线程安全 |
| 阶段性Builder | 不同阶段返回不同类型，强制构造顺序 |
| 验证在build() | 所有约束检查集中在build()，快速失败 |
| 工厂方法 | 为常见用例提供快捷入口，减少模板代码 |

---

**上一章**：[内部DSL模式总览](./09-internal-dsl-patterns.md)
**下一章**：[流畅接口设计](./11-fluent-interface.md)
