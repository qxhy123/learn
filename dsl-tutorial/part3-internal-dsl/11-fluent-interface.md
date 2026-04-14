# 第11章：流畅接口设计

## 核心思维模型

> 流畅接口（Fluent Interface）的目标是让API读起来像**领域专家的自然表达**，而不是程序员的技术术语。关键不是"让所有方法返回self"，而是**用正确的动词、名词和介词构造出有意义的领域语句**。

---

## 11.1 流畅接口 vs 方法链

这两个概念经常被混淆，但有本质区别：

**方法链（Method Chaining）**：每个方法返回self，允许链式调用。只是一种技术手段。

```python
# 方法链（技术层面）
"hello world".strip().upper().split()  # Python字符串方法
```

**流畅接口（Fluent Interface）**：方法链 + **领域语义**，每个方法名构成自然语言句子。

```python
# 方法链 + 领域语义 = 流畅接口
invoice.for_client("Acme Corp")\
       .with_line_item("Web Design", hours=40, rate=150)\
       .with_line_item("Hosting", monthly=50, months=6)\
       .due_in_days(30)\
       .send_via_email()
```

读出来就是自然语言："为Acme Corp开发票，包含Web设计服务40小时每小时150元，托管服务每月50元共6个月，30天内付款，通过邮件发送。"

---

## 11.2 动词-名词设计法

流畅接口的命名遵循**动词-名词**结构，根据上下文切换：

```python
# 设置型方法（命令式）：动词 + 名词
.set_timeout(30)    # 差：set_前缀冗余
.timeout(30)        # 好：动词即名词

# 过渡型方法（介词）：模拟自然语言介词
.for_user("alice")
.with_permission("read")
.from_database("primary")
.via_email()

# 查询型方法（动词）：不修改状态，返回信息
.has_permission("write")   # 返回bool
.get_timeout()             # 返回值
.is_valid()                # 返回bool

# 终止型方法（执行操作）
.build()
.send()
.execute()
.save()
```

---

## 11.3 完整案例：权限规则DSL

```python
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Any

class Action(Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_MFA = "require_mfa"

@dataclass
class Rule:
    name: str
    subject_condition: Callable | None
    resource_condition: Callable | None
    action: Action
    reason: str = ""

class RuleBuilder:
    """权限规则流畅接口"""
    
    def __init__(self, name: str):
        self._name = name
        self._subject_conditions: list[Callable] = []
        self._resource_conditions: list[Callable] = []
        self._action: Action | None = None
        self._reason = ""
    
    # ─── 主语：谁 ──────────────────────────────────────────
    
    def for_users_with_role(self, role: str) -> 'RuleBuilder':
        self._subject_conditions.append(lambda user: user.role == role)
        return self
    
    def for_users_older_than(self, age: int) -> 'RuleBuilder':
        self._subject_conditions.append(lambda user: user.age > age)
        return self
    
    def for_authenticated_users(self) -> 'RuleBuilder':
        self._subject_conditions.append(lambda user: user.is_authenticated)
        return self
    
    def for_any_user(self) -> 'RuleBuilder':
        self._subject_conditions.append(lambda _: True)
        return self
    
    # ─── 宾语：什么资源 ────────────────────────────────────
    
    def accessing_resource(self, resource_type: str) -> 'RuleBuilder':
        self._resource_conditions.append(
            lambda r: r.type == resource_type
        )
        return self
    
    def in_organization(self, org_id: str) -> 'RuleBuilder':
        self._resource_conditions.append(
            lambda r: r.org_id == org_id
        )
        return self
    
    def owned_by_user(self) -> 'RuleBuilder':
        # 特殊条件：需要同时访问user和resource
        self._resource_conditions.append(
            lambda r, user=None: r.owner_id == (user.id if user else None)
        )
        return self
    
    # ─── 谓语：什么结果 ────────────────────────────────────
    
    def can_access(self) -> 'RuleBuilder':
        self._action = Action.ALLOW
        return self
    
    def is_denied(self) -> 'RuleBuilder':
        self._action = Action.DENY
        return self
    
    def must_verify_identity(self) -> 'RuleBuilder':
        self._action = Action.REQUIRE_MFA
        return self
    
    # ─── 补充说明 ──────────────────────────────────────────
    
    def because(self, reason: str) -> 'RuleBuilder':
        self._reason = reason
        return self
    
    # ─── 构建 ──────────────────────────────────────────────
    
    def build(self) -> Rule:
        if self._action is None:
            raise ValueError(f"规则 '{self._name}' 未定义动作（can_access/is_denied/must_verify_identity）")
        
        subject_cond = None
        if self._subject_conditions:
            conditions = self._subject_conditions
            subject_cond = lambda user: all(c(user) for c in conditions)
        
        resource_cond = None
        if self._resource_conditions:
            conditions = self._resource_conditions
            resource_cond = lambda res: all(c(res) for c in conditions)
        
        return Rule(self._name, subject_cond, resource_cond, self._action, self._reason)


def rule(name: str) -> RuleBuilder:
    """工厂函数，启动规则构建"""
    return RuleBuilder(name)


# 使用：读起来像自然语言
rules = [
    rule("管理员全访问")
        .for_users_with_role("admin")
        .can_access()
        .because("管理员有所有权限")
        .build(),
    
    rule("用户访问自己的资源")
        .for_authenticated_users()
        .accessing_resource("document")
        .owned_by_user()
        .can_access()
        .build(),
    
    rule("未成年人限制")
        .for_authenticated_users()
        .accessing_resource("adult_content")
        .is_denied()
        .because("内容分级限制")
        .build(),
    
    rule("高敏感操作要求MFA")
        .for_authenticated_users()
        .accessing_resource("payment")
        .must_verify_identity()
        .build(),
]
```

---

## 11.4 上下文感知的流畅接口

高级技巧：根据调用状态，同一个方法名在不同阶段有不同含义：

```python
class PipelineBuilder:
    """数据处理管道，上下文感知流畅接口"""
    
    def __init__(self):
        self._stages: list = []
        self._current_stage = None
    
    def stage(self, name: str) -> 'StageContext':
        """切换到新stage上下文"""
        context = StageContext(name, self)
        self._stages.append(context)
        return context
    
    def build(self):
        return Pipeline(self._stages)


class StageContext:
    """Stage上下文：在stage内调用的方法"""
    
    def __init__(self, name: str, pipeline: PipelineBuilder):
        self._name = name
        self._pipeline = pipeline
        self._steps: list = []
    
    def run(self, command: str) -> 'StageContext':
        """在当前stage内添加步骤"""
        self._steps.append(('run', command))
        return self
    
    def on_failure(self, action: str) -> 'StageContext':
        self._on_failure = action
        return self
    
    def then_stage(self, name: str) -> 'StageContext':
        """结束当前stage，创建新stage"""
        return self._pipeline.stage(name)
    
    def done(self) -> PipelineBuilder:
        """退出stage上下文，返回pipeline"""
        return self._pipeline


# 使用：上下文清晰
pipeline = (
    PipelineBuilder()
    .stage("build")
        .run("npm install")
        .run("npm run build")
        .on_failure("abort")
    .then_stage("test")
        .run("npm test")
        .run("npm run e2e")
    .then_stage("deploy")
        .run("kubectl apply -f k8s/")
    .done()
    .build()
)
```

---

## 11.5 与类型注解结合

Python 3.10+ 的类型注解让流畅接口更安全：

```python
from typing import Self  # Python 3.11+

class Config:
    def set_host(self, host: str) -> Self:
        self._host = host
        return self
    
    def set_port(self, port: int) -> Self:
        self._port = port
        return self

class SSLConfig(Config):
    def enable_ssl(self) -> Self:
        self._ssl = True
        return self

# 正确：子类方法链返回子类类型
ssl_config = SSLConfig().set_host("localhost").enable_ssl()
# ssl_config 类型推断为 SSLConfig，而非 Config
```

---

## 11.6 流畅接口的测试

流畅接口的测试要验证**链的每一步状态**：

```python
import pytest

class TestRuleBuilder:
    def test_basic_allow_rule(self):
        r = (rule("test")
             .for_users_with_role("admin")
             .can_access()
             .build())
        
        assert r.name == "test"
        assert r.action == Action.ALLOW
    
    def test_missing_action_raises(self):
        with pytest.raises(ValueError, match="未定义动作"):
            rule("incomplete").for_authenticated_users().build()
    
    def test_multiple_subject_conditions(self):
        """多个主语条件AND合并"""
        r = (rule("compound")
             .for_authenticated_users()
             .for_users_with_role("member")
             .can_access()
             .build())
        
        class MockUser:
            is_authenticated = True
            role = "member"
        
        class MockUserUnauthenticated:
            is_authenticated = False
            role = "member"
        
        assert r.subject_condition(MockUser()) is True
        assert r.subject_condition(MockUserUnauthenticated()) is False
    
    def test_reason_is_optional(self):
        r = rule("minimal").for_any_user().can_access().build()
        assert r.reason == ""
        
        r_with_reason = (rule("with_reason")
                        .for_any_user()
                        .can_access()
                        .because("公开资源")
                        .build())
        assert r_with_reason.reason == "公开资源"
```

---

## 11.7 常见陷阱与解决方案

### 陷阱一：方法顺序产生不同语义

```python
# 如果 .for_users_with_role 和 .accessing_resource 顺序不同，含义相同吗？
# 好的流畅接口：顺序无关性
rule("test").for_users_with_role("admin").accessing_resource("doc").can_access().build()
rule("test").accessing_resource("doc").for_users_with_role("admin").can_access().build()
# 两者应等价
```

### 陷阱二：方法链太长导致难以调试

```python
# 差：一行超长方法链，中间出错难定位
result = A().b().c().d().e().f().g().h().build()

# 好：分步赋值，便于调试
step1 = A().b().c()
step2 = step1.d().e()
result = step2.f().g().h().build()
```

### 陷阱三：终止方法（build/send）有副作用

```python
# 危险：build()有副作用（保存到数据库）
user = UserBuilder().name("Alice").build()  # 悄悄写数据库！

# 好：分离构造和持久化
user_data = UserBuilder().name("Alice").build()   # 纯构造，无副作用
user_service.save(user_data)                       # 显式持久化
```

---

## 小结

| 概念 | 要点 |
|------|------|
| 流畅接口 | 方法链 + 领域语义，目标是自然语言可读性 |
| 动词-名词设计 | 方法名用领域动词，介词连接主宾 |
| 上下文感知 | 同名方法在不同stage有不同行为 |
| 类型安全 | Python 3.11+ `Self`类型支持子类方法链 |
| 测试策略 | 验证每个链步骤的状态变化 |

---

**上一章**：[构建者模式DSL](./10-builder-pattern.md)
**下一章**：[Python魔法方法DSL](./12-python-magic-dsl.md)
