# 第14章：流式接口模式

> "流式接口让代码从'调用序列'变成'意图叙述'。"

---

## 14.1 流式接口的设计原则

**流式接口（Fluent Interface）**是 Martin Fowler 提出的术语，指通过方法链创建可读性接近自然语言的 API。

核心原则：
1. **每个方法返回 this（self）**，除非方法语义上是"终结"操作
2. **方法名是动词短语**，描述正在做什么
3. **参数尽量少**，复杂配置用专门的子 DSL
4. **终结方法**（如 `build()`、`execute()`）执行实际操作

```python
# 对比：普通 API vs 流式 API

# 普通 API（命令式）
report = Report()
report.set_period_start(date(2024, 1, 1))
report.set_period_end(date(2024, 3, 31))
report.add_dimension("region")
report.set_metric("revenue")
report.set_sort_order("desc")
report.set_top_n(10)
result = report.generate()

# 流式 API（声明式）
result = (
    Report()
        .for_period(start=date(2024, 1, 1), end=date(2024, 3, 31))
        .group_by_region()
        .by_revenue()
        .top(10)
        .generate()
)
```

---

## 14.2 五种流式接口模式

### 模式1：渐进构建（Progressive Builder）

每次调用添加一层配置：

```python
class EmailBuilder:
    """邮件构建器"""
    
    def __init__(self):
        self._to = []
        self._subject = ""
        self._body = ""
        self._attachments = []
        self._cc = []
    
    def to(self, *recipients: str) -> 'EmailBuilder':
        self._to.extend(recipients)
        return self
    
    def cc(self, *recipients: str) -> 'EmailBuilder':
        self._cc.extend(recipients)
        return self
    
    def subject(self, text: str) -> 'EmailBuilder':
        self._subject = text
        return self
    
    def body(self, content: str) -> 'EmailBuilder':
        self._body = content
        return self
    
    def attach(self, filename: str) -> 'EmailBuilder':
        self._attachments.append(filename)
        return self
    
    def send(self) -> bool:
        """终结方法：发送邮件"""
        email = Email(
            to=self._to,
            cc=self._cc,
            subject=self._subject,
            body=self._body,
            attachments=self._attachments
        )
        return email_service.send(email)


# 使用
(
    EmailBuilder()
        .to("customer@example.com")
        .cc("support@company.com")
        .subject("您的订单已发货")
        .body(f"您好，您的订单 {order_id} 已于今天发出。")
        .attach("invoice.pdf")
        .send()
)
```

### 模式2：状态转换链（State Transition Chain）

每次调用触发状态转换：

```python
class OrderWorkflow:
    """订单工作流——每个方法触发一次状态转换"""
    
    def __init__(self, order: Order):
        self._order = order
        self._errors = []
    
    def validate(self) -> 'OrderWorkflow':
        """验证订单"""
        if not self._order.items:
            self._errors.append("订单不能为空")
        if self._order.total.amount <= 0:
            self._errors.append("订单金额必须大于0")
        return self
    
    def apply_pricing(self) -> 'OrderWorkflow':
        """计算价格（含折扣）"""
        if self._errors:
            return self  # 有错误时跳过
        self._order = pricing_service.apply_all_rules(self._order)
        return self
    
    def reserve_inventory(self) -> 'OrderWorkflow':
        """预留库存"""
        if self._errors:
            return self
        try:
            inventory_service.reserve(self._order.items)
        except InsufficientInventoryError as e:
            self._errors.append(str(e))
        return self
    
    def confirm(self) -> 'WorkflowResult':
        """终结方法：确认订单并返回结果"""
        if self._errors:
            return WorkflowResult.failed(errors=self._errors)
        
        self._order.confirm()
        return WorkflowResult.succeeded(order=self._order)


# 使用：整个工作流清晰可见
result = (
    OrderWorkflow(draft_order)
        .validate()
        .apply_pricing()
        .reserve_inventory()
        .confirm()
)
```

### 模式3：条件分支链（Conditional Branch Chain）

支持在链中做条件分支：

```python
class QueryDSL:
    """支持条件分支的查询 DSL"""
    
    def __init__(self):
        self._conditions = []
        self._pagination = None
    
    def where(self, condition) -> 'QueryDSL':
        self._conditions.append(condition)
        return self
    
    def when(self, predicate: bool, then_fn) -> 'QueryDSL':
        """条件应用：if predicate then apply then_fn"""
        if predicate:
            then_fn(self)
        return self
    
    def paginate(self, page: int, size: int) -> 'QueryDSL':
        self._pagination = (page, size)
        return self
    
    def execute(self) -> list:
        ...


# 使用：根据条件动态构建查询
def search_orders(customer_id=None, status=None, date_from=None, page=1):
    return (
        QueryDSL()
            .when(customer_id is not None,
                  lambda q: q.where(f"customer_id = '{customer_id}'"))
            .when(status is not None,
                  lambda q: q.where(f"status = '{status}'"))
            .when(date_from is not None,
                  lambda q: q.where(f"created_at >= '{date_from}'"))
            .paginate(page=page, size=20)
            .execute()
    )
```

### 模式4：嵌套 DSL（Nested DSL）

子 DSL 处理复杂的嵌套配置：

```python
class SchemaBuilder:
    """数据库 Schema DSL"""
    
    def __init__(self, table_name: str):
        self._table = table_name
        self._columns = []
        self._indexes = []
    
    def column(self, name: str) -> 'ColumnBuilder':
        """返回子 DSL 处理列定义"""
        col = ColumnBuilder(name, parent=self)
        self._columns.append(col)
        return col
    
    def index(self, *columns: str) -> 'IndexBuilder':
        idx = IndexBuilder(columns, parent=self)
        self._indexes.append(idx)
        return idx
    
    def build(self) -> str:
        ...


class ColumnBuilder:
    """列定义子 DSL"""
    
    def __init__(self, name: str, parent: SchemaBuilder):
        self._name = name
        self._type = "TEXT"
        self._nullable = True
        self._parent = parent
    
    def of_type(self, type_name: str) -> 'ColumnBuilder':
        self._type = type_name
        return self
    
    def not_null(self) -> 'ColumnBuilder':
        self._nullable = False
        return self
    
    def primary_key(self) -> 'ColumnBuilder':
        self._is_pk = True
        return self
    
    # 返回父 DSL，继续链式调用
    def and_column(self, name: str) -> 'ColumnBuilder':
        return self._parent.column(name)
    
    def with_index(self) -> 'SchemaBuilder':
        return self._parent


# 使用：Schema 定义读起来像 DDL 文档
schema = (
    SchemaBuilder("orders")
        .column("id").of_type("UUID").not_null().primary_key()
            .and_column("customer_id").of_type("UUID").not_null()
            .and_column("status").of_type("VARCHAR(20)").not_null()
            .and_column("created_at").of_type("TIMESTAMP").not_null()
        .build()
)
```

### 模式5：组合 DSL（Combinator DSL）

通过组合小 DSL 构建复杂 DSL：

```python
class CombinedValidator:
    """组合验证器：将多个验证规则合并为一个"""
    
    def __init__(self, check_fn):
        self._check_fn = check_fn
    
    def __call__(self, value) -> list[str]:
        return self._check_fn(value)
    
    def __and__(self, other) -> 'CombinedValidator':
        def combined(value):
            return self(value) + other(value)
        return CombinedValidator(combined)
    
    def __or__(self, other) -> 'CombinedValidator':
        def either(value):
            left = self(value)
            right = other(value)
            return [] if (not left or not right) else left
        return CombinedValidator(either)


class Validator:
    """验证规则的 Combinator DSL"""
    
    def __init__(self, check_fn, error_msg: str):
        self._check = check_fn
        self._error_msg = error_msg
    
    def __call__(self, value) -> list[str]:
        """验证并返回错误列表"""
        if not self._check(value):
            return [self._error_msg]
        return []
    
    def __and__(self, other: 'Validator') -> 'CombinedValidator':
        """组合：两个都要通过"""
        def combined_check(value):
            errors = self(value) + other(value)
            return errors
        return CombinedValidator(combined_check)
    
    def __or__(self, other: 'Validator') -> 'CombinedValidator':
        """组合：至少一个通过"""
        def either_check(value):
            left_errors = self(value)
            right_errors = other(value)
            if left_errors and right_errors:
                return left_errors  # 都失败，返回左边的错误
            return []
        return CombinedValidator(either_check)


# 预定义的验证器（小 DSL 元素）
not_empty = Validator(lambda v: bool(v), "不能为空")
max_100 = Validator(lambda v: len(str(v)) <= 100, "长度不能超过100")
is_email = Validator(lambda v: "@" in str(v), "必须是有效的邮箱地址")
is_phone = Validator(lambda v: str(v).isdigit() and len(str(v)) == 11, "必须是11位手机号")

# 组合验证规则——读起来就是业务规则
username_validator = not_empty & max_100
contact_validator = is_email | is_phone
```

---

## 14.3 避免流式接口的常见陷阱

### 陷阱1：链太长导致难以调试

```python
# ❌ 太长，出错时不知道是哪一步
result = (A().b().c().d().e().f().g().h().i().j().build())

# ✅ 分步，便于调试
step1 = A().b().c()
step2 = step1.d().e().f()
result = step2.g().h().i().j().build()
```

### 陷阱2：副作用隐藏在链中

```python
# ❌ 链中有不明显的副作用
order = (
    Order()
        .add_item(item)      # 修改状态
        .send_email()        # 副作用！发送了邮件
        .reserve_inventory() # 副作用！修改了库存
        .confirm()
)

# ✅ 副作用应该在终结方法中集中处理
order = (
    OrderBuilder()
        .add_item(item)
        .build()             # 纯对象构建
)
# 副作用单独处理
workflow.process(order)      # 明确的副作用位置
```

### 陷阱3：方法名太通用

```python
# ❌ with_() 什么都能叫这个名字
builder.with_(x).with_(y).with_(z)

# ✅ 用描述性名字
builder.with_customer(customer).with_coupon(coupon).with_items(items)
```

---

## 14.4 实战：为测试构建流式断言 DSL

```python
class AssertThat:
    """测试断言的流式 DSL"""
    
    def __init__(self, value):
        self._value = value
        self._description = ""
    
    @classmethod
    def the_order(cls, order: Order) -> 'OrderAssertions':
        return OrderAssertions(order)
    
    @classmethod
    def the_money(cls, money: Money) -> 'MoneyAssertions':
        return MoneyAssertions(money)


class OrderAssertions:
    def __init__(self, order: Order):
        self._order = order
    
    def is_confirmed(self) -> 'OrderAssertions':
        assert self._order.status == OrderStatus.CONFIRMED, \
            f"期望订单状态为 CONFIRMED，实际为 {self._order.status}"
        return self
    
    def has_total(self, expected: Money) -> 'OrderAssertions':
        assert self._order.total == expected, \
            f"期望总价为 {expected}，实际为 {self._order.total}"
        return self
    
    def has_items(self, count: int) -> 'OrderAssertions':
        assert len(self._order.items) == count, \
            f"期望 {count} 个商品，实际有 {len(self._order.items)} 个"
        return self
    
    def emitted_event(self, event_type) -> 'OrderAssertions':
        events = self._order.pull_events()
        assert any(isinstance(e, event_type) for e in events), \
            f"期望发出 {event_type.__name__} 事件"
        return self


# 测试中使用
def test_order_workflow():
    order = make_draft_order()
    order.confirm()
    
    (
        AssertThat.the_order(order)
            .is_confirmed()
            .has_items(2)
            .has_total(Money(Decimal("119.8"), "CNY"))
            .emitted_event(OrderConfirmed)
    )
```

---

## 总结

流式接口的五种模式：
1. **渐进构建**：逐步积累配置，最后 build
2. **状态转换链**：每步触发状态变化
3. **条件分支链**：链中支持 when/unless
4. **嵌套 DSL**：子 DSL 处理嵌套结构
5. **组合 DSL**：用运算符组合小规则

关键原则：**流式接口是为了可读性，不是为了炫技**。

---

**下一章**：[用 Lark 构建外部 DSL](15-external-dsl-with-lark.md)
