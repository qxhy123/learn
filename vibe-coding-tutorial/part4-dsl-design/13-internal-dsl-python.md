# 第13章：Python 内部 DSL

> "最好的 DSL 是你不需要额外学习就能读懂的语言。"

---

## 13.1 内部 DSL 的本质

**内部 DSL（Internal DSL）**：寄宿在宿主语言（Python）中的领域特定语言，利用宿主语言的语法特性来创建更贴近业务的表达方式。

与外部 DSL 的区别：
| 对比维度 | 内部 DSL | 外部 DSL |
|---------|---------|---------|
| 解析器 | 宿主语言本身 | 需要自己写 |
| 语法限制 | 受宿主语言限制 | 完全自由 |
| 开发成本 | 低 | 高 |
| 工具支持 | 继承宿主语言工具 | 需要单独支持 |
| 适用场景 | 配置、工作流、规则 | 领域专家直接编写 |

### Python 特别适合内部 DSL 的原因

```python
# Python 的这些特性让内部 DSL 成为可能：

# 1. 方法链（Method Chaining）
result = Query().select("name").from_("users").where(age > 18).limit(10)

# 2. 上下文管理器（Context Managers）
with transaction():
    order.confirm()
    payment.process()

# 3. 装饰器（Decorators）
@route("/orders")
@requires_auth
def get_orders(): ...

# 4. 运算符重载（Operator Overloading）
price = base_price * Percentage(10)  # 自定义 * 操作

# 5. __call__ 方法
validator = NotEmpty() & MaxLength(100) & NoSpecialChars()
validator("hello")  # 可调用

# 6. 关键字参数
Order.place(customer=alice, items=[book], coupon=summer_sale)
```

---

## 13.2 方法链 DSL（Builder Pattern）

最常见的内部 DSL 风格：

```python
from typing import Optional, List, TYPE_CHECKING
from decimal import Decimal

class QueryBuilder:
    """SQL-like 查询 DSL"""
    
    def __init__(self):
        self._table = None
        self._conditions = []
        self._fields = ["*"]
        self._limit_val = None
        self._order_field = None
        self._order_dir = "ASC"
    
    def select(self, *fields: str) -> 'QueryBuilder':
        self._fields = list(fields)
        return self  # 返回 self 使方法链成为可能
    
    def from_table(self, table: str) -> 'QueryBuilder':
        self._table = table
        return self
    
    def where(self, condition: str) -> 'QueryBuilder':
        self._conditions.append(condition)
        return self
    
    def order_by(self, field: str, direction: str = "ASC") -> 'QueryBuilder':
        self._order_field = field
        self._order_dir = direction
        return self
    
    def limit(self, n: int) -> 'QueryBuilder':
        self._limit_val = n
        return self
    
    def build(self) -> str:
        """构建最终的 SQL 字符串"""
        parts = [f"SELECT {', '.join(self._fields)}"]
        parts.append(f"FROM {self._table}")
        if self._conditions:
            parts.append(f"WHERE {' AND '.join(self._conditions)}")
        if self._order_field:
            parts.append(f"ORDER BY {self._order_field} {self._order_dir}")
        if self._limit_val:
            parts.append(f"LIMIT {self._limit_val}")
        return " ".join(parts)

# 使用：读起来像 SQL 但是 Python
query = (
    QueryBuilder()
        .select("id", "name", "email")
        .from_table("customers")
        .where("tier = 'vip'")
        .where("active = true")
        .order_by("created_at", "DESC")
        .limit(20)
        .build()
)
# 输出：SELECT id, name, email FROM customers WHERE tier = 'vip' AND active = true ORDER BY created_at DESC LIMIT 20
```

### TDD 测试 Builder DSL

```python
class TestQueryBuilder:
    
    def test_simple_select_all(self):
        query = QueryBuilder().from_table("users").build()
        assert query == "SELECT * FROM users"
    
    def test_select_specific_fields(self):
        query = QueryBuilder().select("id", "name").from_table("users").build()
        assert query == "SELECT id, name FROM users"
    
    def test_where_condition(self):
        query = (
            QueryBuilder()
                .from_table("users")
                .where("age > 18")
                .build()
        )
        assert "WHERE age > 18" in query
    
    def test_multiple_where_conditions_use_and(self):
        query = (
            QueryBuilder()
                .from_table("users")
                .where("age > 18")
                .where("active = true")
                .build()
        )
        assert "WHERE age > 18 AND active = true" in query
    
    def test_order_by_default_asc(self):
        query = QueryBuilder().from_table("users").order_by("name").build()
        assert "ORDER BY name ASC" in query
    
    def test_limit_appended_last(self):
        query = QueryBuilder().from_table("users").limit(10).build()
        assert query.endswith("LIMIT 10")
```

---

## 13.3 规则 DSL

```python
class PricingRule:
    """定价规则 DSL——让业务规则可读"""
    
    def __init__(self, name: str):
        self._name = name
        self._conditions = []
        self._actions = []
    
    def when(self, condition) -> 'PricingRule':
        """条件：何时应用这条规则"""
        self._conditions.append(condition)
        return self
    
    def then(self, action) -> 'PricingRule':
        """动作：应用什么折扣/变更"""
        self._actions.append(action)
        return self
    
    def applies_to(self, context: dict) -> bool:
        return all(cond(context) for cond in self._conditions)
    
    def apply(self, price: Money, context: dict) -> Money:
        result = price
        for action in self._actions:
            result = action(result, context)
        return result
    
    def __repr__(self):
        return f"PricingRule({self._name!r})"


class Discount:
    """折扣操作的 DSL 工厂"""
    
    @staticmethod
    def percent(rate: float):
        """按百分比折扣"""
        def apply(price: Money, context: dict) -> Money:
            factor = Decimal(str(1 - rate / 100))
            return price.multiply(factor)
        return apply
    
    @staticmethod
    def fixed(amount: Money):
        """固定金额折扣"""
        def apply(price: Money, context: dict) -> Money:
            return price.subtract(amount)
        return apply
    
    @staticmethod
    def free_shipping():
        """免运费"""
        def apply(price: Money, context: dict) -> Money:
            shipping = context.get("shipping_fee", Money(Decimal("0"), "CNY"))
            return price.subtract(shipping)
        return apply


class Condition:
    """条件的 DSL 工厂"""
    
    @staticmethod
    def customer_tier(tier: str):
        return lambda ctx: ctx.get("customer_tier") == tier
    
    @staticmethod
    def order_total_above(amount: Money):
        return lambda ctx: ctx.get("order_total", Money(Decimal("0"), "CNY")).amount >= amount.amount
    
    @staticmethod
    def has_coupon(code: str):
        return lambda ctx: ctx.get("coupon_code") == code


# 业务规则定义——读起来像业务文档！
pricing_rules = [
    PricingRule("VIP 专属折扣")
        .when(Condition.customer_tier("vip"))
        .then(Discount.percent(10)),
    
    PricingRule("满500减50")
        .when(Condition.order_total_above(Money(Decimal("500"), "CNY")))
        .then(Discount.fixed(Money(Decimal("50"), "CNY"))),
    
    PricingRule("SUMMER 优惠码")
        .when(Condition.has_coupon("SUMMER2024"))
        .when(Condition.order_total_above(Money(Decimal("100"), "CNY")))
        .then(Discount.percent(15))
        .then(Discount.free_shipping()),
]
```

### 测试规则 DSL

```python
class TestPricingRules:
    
    def test_vip_discount_applies_to_vip_customer(self):
        rule = (
            PricingRule("VIP折扣")
                .when(Condition.customer_tier("vip"))
                .then(Discount.percent(10))
        )
        context = {"customer_tier": "vip"}
        price = Money(Decimal("100"), "CNY")
        
        assert rule.applies_to(context) is True
        assert rule.apply(price, context) == Money(Decimal("90.00"), "CNY")
    
    def test_vip_discount_does_not_apply_to_normal_customer(self):
        rule = (
            PricingRule("VIP折扣")
                .when(Condition.customer_tier("vip"))
                .then(Discount.percent(10))
        )
        context = {"customer_tier": "normal"}
        
        assert rule.applies_to(context) is False
    
    def test_multiple_conditions_all_must_be_true(self):
        rule = (
            PricingRule("SUMMER优惠")
                .when(Condition.has_coupon("SUMMER2024"))
                .when(Condition.order_total_above(Money(Decimal("100"), "CNY")))
                .then(Discount.percent(15))
        )
        
        # 有优惠码但金额不够
        context = {"coupon_code": "SUMMER2024", "order_total": Money(Decimal("50"), "CNY")}
        assert rule.applies_to(context) is False
        
        # 金额够但没有优惠码
        context = {"coupon_code": None, "order_total": Money(Decimal("200"), "CNY")}
        assert rule.applies_to(context) is False
        
        # 两个条件都满足
        context = {"coupon_code": "SUMMER2024", "order_total": Money(Decimal("200"), "CNY")}
        assert rule.applies_to(context) is True
```

---

## 13.4 工作流 DSL

```python
from typing import Callable, Any
from functools import reduce

class Pipeline:
    """管道工作流 DSL"""
    
    def __init__(self, value: Any):
        self._value = value
        self._steps = []
    
    def pipe(self, func: Callable, *args, **kwargs) -> 'Pipeline':
        """添加处理步骤"""
        self._steps.append((func, args, kwargs))
        return self
    
    def execute(self) -> Any:
        """执行所有步骤"""
        result = self._value
        for func, args, kwargs in self._steps:
            result = func(result, *args, **kwargs)
        return result
    
    @classmethod
    def process(cls, value: Any) -> 'Pipeline':
        return cls(value)


# 使用：订单处理工作流
def validate_order(order: Order) -> Order:
    if not order.items:
        raise EmptyOrderError()
    return order

def apply_vip_discount(order: Order) -> Order:
    if order.customer_tier == CustomerTier.VIP:
        return order.with_discount(Decimal("0.1"))
    return order

def calculate_tax(order: Order) -> Order:
    return order.with_tax(rate=Decimal("0.09"))

def confirm_order(order: Order) -> Order:
    return order.confirm()


# 读起来像业务流程描述
result = (
    Pipeline.process(draft_order)
        .pipe(validate_order)
        .pipe(apply_vip_discount)
        .pipe(calculate_tax)
        .pipe(confirm_order)
        .execute()
)
```

---

## 13.5 配置 DSL

```python
class ServiceConfig:
    """服务配置 DSL"""
    
    def __init__(self, name: str):
        self.name = name
        self._host = "localhost"
        self._port = 8080
        self._timeout = 30
        self._retries = 3
        self._middleware = []
    
    def at(self, host: str, port: int = 8080) -> 'ServiceConfig':
        self._host = host
        self._port = port
        return self
    
    def with_timeout(self, seconds: int) -> 'ServiceConfig':
        self._timeout = seconds
        return self
    
    def retry(self, times: int) -> 'ServiceConfig':
        self._retries = times
        return self
    
    def with_middleware(self, *middlewares) -> 'ServiceConfig':
        self._middleware.extend(middlewares)
        return self
    
    def build(self) -> dict:
        return {
            "name": self.name,
            "host": self._host,
            "port": self._port,
            "timeout": self._timeout,
            "retries": self._retries,
            "middleware": self._middleware
        }


# 配置读起来像文档
payment_service = (
    ServiceConfig("payment-service")
        .at("payment.internal", port=9090)
        .with_timeout(5)
        .retry(2)
        .with_middleware(auth_middleware, logging_middleware)
        .build()
)
```

---

## 13.6 实战：为电商系统构建完整内部 DSL

```python
# 完整的订单处理 DSL

class OrderDSL:
    """订单操作的 DSL 入口"""
    
    @staticmethod
    def for_customer(customer_id: str) -> 'OrderBuilder':
        return OrderBuilder(customer_id=customer_id)

class OrderBuilder:
    def __init__(self, customer_id: str):
        self._customer_id = customer_id
        self._items = []
        self._coupon = None
    
    def add(self, product: str, qty: int = 1, price: Money = None) -> 'OrderBuilder':
        self._items.append(OrderItem(product=product, quantity=qty, unit_price=price))
        return self
    
    def with_coupon(self, code: str) -> 'OrderBuilder':
        self._coupon = Coupon.find(code)
        return self
    
    def place(self) -> 'OrderConfirmationBuilder':
        order = Order.place(customer_id=self._customer_id, items=self._items)
        if self._coupon:
            order.apply_coupon(self._coupon)
        return OrderConfirmationBuilder(order)

class OrderConfirmationBuilder:
    def __init__(self, order: Order):
        self._order = order
    
    def confirm(self) -> 'OrderPaymentBuilder':
        self._order.confirm()
        return OrderPaymentBuilder(self._order)
    
    def preview(self) -> OrderSummary:
        return OrderSummary(order=self._order)

class OrderPaymentBuilder:
    def __init__(self, order: Order):
        self._order = order
    
    def pay_with(self, method: str) -> Order:
        payment = Payment.process(order=self._order, method=method)
        self._order.attach_payment(payment)
        return self._order


# 使用：整个下单流程如同阅读业务文档
order = (
    OrderDSL
        .for_customer("alice")
        .add("Python编程", qty=2, price=Money(Decimal("59.9"), "CNY"))
        .add("代码整洁之道", qty=1, price=Money(Decimal("79.9"), "CNY"))
        .with_coupon("BOOK20")
        .place()
        .confirm()
        .pay_with("alipay")
)
```

---

## 总结

Python 内部 DSL 的设计原则：
1. **方法链**：每个方法返回 `self`，支持链式调用
2. **流式读写**：左到右读起来像自然语言
3. **领域词汇**：方法名来自统一语言词汇表
4. **延迟执行**：Builder 积累配置，最后 `.build()` 或 `.execute()` 执行

---

**下一章**：[流式接口模式](14-fluent-interface-patterns.md)
