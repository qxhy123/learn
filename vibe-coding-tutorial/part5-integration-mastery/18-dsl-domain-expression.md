# 第18章：DSL 表达领域语言

> "DSL 是统一语言的代码化形态——让业务逻辑读起来像业务文档。"

---

## 18.1 DSL 与统一语言的关系

DDD 的统一语言是词汇和概念，DSL 是让这些词汇和概念在代码中"活起来"的机制：

```
统一语言词汇表：
  "读者"、"借阅"、"预约"、"还书"、"逾期费"

DDD 领域模型：
  class Patron, class Borrowing, class Reservation

内部 DSL（让业务流程可读）：
  LibrarySystem
      .patron(patron_id)
      .borrow(book_isbn)
      .for_days(14)
      .execute()

外部 DSL（让业务规则可配置）：
  rule "逾期费计算"
  when borrowing.is_overdue
  then charge 1 yuan per day
  cap at 30 yuan
  end
```

三者是同一个概念的三种表达形式。

---

## 18.2 从统一语言到 DSL 设计

### 案例：积分系统的 DSL 设计

**统一语言词汇**：
- 积分账户（PointsAccount）
- 积分明细（PointsEntry）
- 获得积分（EarnPoints）
- 兑换积分（RedeemPoints）
- 积分有效期（ExpiresAt）
- 积分规则（PointsRule）

**从词汇设计 DSL**：

```python
# 第一步：列出业务场景（来自用户故事）
"""
场景1：用户消费100元，获得1000积分，365天有效
场景2：VIP用户消费获得双倍积分
场景3：用户用500积分兑换咖啡券
场景4：查询用户有效积分余额
"""

# 第二步：用 DSL 表达这些场景
# （先设计 API，再实现）

# 场景1
PointsSystem.for_customer("alice").earned(
    amount=Money(Decimal("100"), "CNY"),
    expires_in_days=365
)

# 场景2
PointsSystem.for_customer("alice").earned(
    amount=Money(Decimal("100"), "CNY"),
    multiplier=2  # VIP 双倍
)

# 场景3
PointsSystem.for_customer("alice").redeemed(
    points=500,
    for_reward="coffee_voucher"
)

# 场景4
balance = PointsSystem.for_customer("alice").active_balance
```

### 第三步：TDD 验证 DSL 设计

```python
class TestPointsSystemDSL:
    
    def test_earn_points_on_purchase(self):
        system = PointsSystem(repo=InMemoryPointsRepo())
        
        system.for_customer("alice").earned(
            amount=Money(Decimal("100"), "CNY"),
            expires_in_days=365
        )
        
        assert system.for_customer("alice").active_balance == 1000
    
    def test_vip_multiplier_doubles_points(self):
        system = PointsSystem(repo=InMemoryPointsRepo())
        
        system.for_customer("alice").earned(
            amount=Money(Decimal("100"), "CNY"),
            multiplier=2
        )
        
        assert system.for_customer("alice").active_balance == 2000
    
    def test_redeem_reduces_balance(self):
        system = PointsSystem(repo=InMemoryPointsRepo())
        system.for_customer("alice").earned(amount=Money(Decimal("100"), "CNY"))
        
        system.for_customer("alice").redeemed(points=500, for_reward="coffee")
        
        assert system.for_customer("alice").active_balance == 500
```

---

## 18.3 为领域事件设计 DSL

领域事件的发布和订阅可以用 DSL 表达：

```python
class DomainEvents:
    """领域事件 DSL"""
    
    _bus = EventBus()
    
    @classmethod
    def on(cls, event_type):
        """装饰器 DSL：订阅事件"""
        def decorator(handler):
            cls._bus.subscribe(event_type, handler)
            return handler
        return decorator
    
    @classmethod
    def when(cls, event_type) -> 'EventHandlerBuilder':
        """流式 DSL：订阅并处理事件"""
        return EventHandlerBuilder(event_type, cls._bus)


class EventHandlerBuilder:
    def __init__(self, event_type, bus: EventBus):
        self._event_type = event_type
        self._bus = bus
        self._conditions = []
        self._handlers = []
    
    def if_condition(self, predicate) -> 'EventHandlerBuilder':
        self._conditions.append(predicate)
        return self
    
    def then_do(self, handler) -> 'EventHandlerBuilder':
        self._handlers.append(handler)
        return self
    
    def register(self):
        def combined_handler(event):
            if all(c(event) for c in self._conditions):
                for h in self._handlers:
                    h(event)
        self._bus.subscribe(self._event_type, combined_handler)


# 使用：订阅逻辑读起来像业务规则
(
    DomainEvents
        .when(OrderConfirmed)
        .if_condition(lambda e: e.total.amount > 1000)  # 大额订单
        .then_do(send_vip_notification)
        .then_do(assign_priority_shipping)
        .register()
)

# 装饰器风格
@DomainEvents.on(OrderCancelled)
def release_inventory(event: OrderCancelled):
    inventory_service.release_for_order(event.order_id)
```

---

## 18.4 为业务规则设计 DSL

```python
class BusinessRule:
    """业务规则 DSL——让规则表达成为一等公民"""
    
    def __init__(self, name: str, description: str = ""):
        self._name = name
        self._description = description
        self._preconditions = []
        self._invariants = []
        self._postconditions = []
    
    def requires(self, *conditions) -> 'BusinessRule':
        """前置条件：必须满足才能执行"""
        self._preconditions.extend(conditions)
        return self
    
    def ensures(self, *conditions) -> 'BusinessRule':
        """后置条件：执行后必须满足"""
        self._postconditions.extend(conditions)
        return self
    
    def invariant(self, *conditions) -> 'BusinessRule':
        """不变量：始终必须满足"""
        self._invariants.extend(conditions)
        return self
    
    def validate(self, context: dict) -> 'ValidationResult':
        errors = []
        for pre in self._preconditions:
            if not pre.check(context):
                errors.append(f"前置条件失败：{pre.description}")
        for inv in self._invariants:
            if not inv.check(context):
                errors.append(f"不变量违反：{inv.description}")
        return ValidationResult(valid=not errors, errors=errors)


# 定义业务规则——读起来像规格说明书
order_confirmation_rule = (
    BusinessRule("订单确认规则", "客户确认订单的业务规则")
        .requires(
            Condition.order_has_items("订单必须包含商品"),
            Condition.customer_is_active("客户账户必须有效"),
            Condition.inventory_is_available("所有商品必须有库存")
        )
        .ensures(
            Condition.order_status_is(OrderStatus.CONFIRMED, "订单状态为已确认"),
            Condition.event_emitted(OrderConfirmed, "发出 OrderConfirmed 事件")
        )
        .invariant(
            Condition.total_is_positive("订单总额必须为正数")
        )
)
```

---

## 18.5 测试 DSL 与领域语言的对齐

```python
class TestDSLDomainAlignment:
    """验证 DSL 是否准确表达了领域语言"""
    
    def test_dsl_method_names_match_ubiquitous_language(self):
        """DSL 方法名必须与统一语言词汇表中的动词一致"""
        # 从词汇表加载业务动词
        ubiquitous_verbs = load_ubiquitous_language()["verbs"]
        # ['place', 'confirm', 'cancel', 'pay', 'ship', 'return']
        
        # 检查 DSL 类的公开方法
        dsl_methods = [
            m for m in dir(OrderDSL)
            if not m.startswith('_')
        ]
        
        # 每个 DSL 方法都应该在词汇表中能找到对应
        for method in dsl_methods:
            assert any(verb in method for verb in ubiquitous_verbs), \
                f"DSL 方法 '{method}' 不在统一语言词汇表中，请检查命名"
    
    def test_dsl_reads_like_business_scenario(self):
        """DSL 代码应该读起来像业务场景描述"""
        # 业务场景文档：
        # "VIP客户 alice 购买2本书，使用BOOK20优惠码，确认订单后支付"
        
        # DSL 代码（字面读起来和场景描述应该一一对应）
        order = (
            OrderDSL
                .for_customer("alice")          # "VIP客户 alice"
                .add("Python书", qty=2)          # "购买2本书"
                .with_coupon("BOOK20")           # "使用BOOK20优惠码"
                .place()                         # "确认订单"
                .confirm()
                .pay_with("alipay")              # "支付"
        )
        
        # 验证结果
        assert order.status == OrderStatus.PAID
```

---

## 18.6 组合三者：完整的 Vibe Coding DSL 层

```python
# 最终形态：三者融合的 Vibe Coding 代码

# === 领域模型（DDD）===
@dataclass(frozen=True)
class Money:
    amount: Decimal
    currency: str

@dataclass
class Order:
    # ... 完整的聚合根实现

# === 业务规则（外部 DSL）===
# pricing_rules.dsl
"""
rule "VIP专属"
when customer.tier = "vip"
then discount 10%
end
"""

# === 工作流（内部 DSL）===
class OrderFulfillment:
    
    @classmethod
    def begin(cls) -> 'OrderFulfillmentBuilder':
        return OrderFulfillmentBuilder()


class OrderFulfillmentBuilder:
    
    def for_customer(self, customer_id: str) -> 'ItemsBuilder':
        return ItemsBuilder(customer_id)


class ItemsBuilder:
    def __init__(self, customer_id: str):
        self._customer_id = customer_id
        self._items = []
    
    def with_item(self, product_id: str, qty: int) -> 'ItemsBuilder':
        self._items.append((product_id, qty))
        return self
    
    def and_coupon(self, code: str) -> 'PricingBuilder':
        return PricingBuilder(self._customer_id, self._items, code)


class PricingBuilder:
    def __init__(self, customer_id, items, coupon_code=None):
        self._customer_id = customer_id
        self._items = items
        self._coupon_code = coupon_code
    
    def apply_rules_from(self, rules_file: str) -> 'FulfillmentBuilder':
        return FulfillmentBuilder(
            self._customer_id, self._items,
            self._coupon_code, rules_file
        )


class FulfillmentBuilder:
    def execute(self) -> OrderResult:
        # ... 执行完整业务流程
        pass


# === 测试（TDD）===
class TestCompleteOrderFulfillment:
    
    def test_vip_order_with_coupon_gets_stacked_discount(self):
        """完整的 Vibe Coding 测试：测试名即需求文档"""
        result = (
            OrderFulfillment
                .begin()
                .for_customer("vip-alice")
                .with_item("python-book", qty=2)
                .and_coupon("SUMMER10")
                .apply_rules_from("tests/fixtures/pricing_rules.dsl")
                .execute()
        )
        
        assert result.success is True
        assert result.applied_discounts == ["VIP专属", "SUMMER优惠"]
        assert result.final_price < result.base_price
```

---

## 总结

DSL 让统一语言在代码中完整落地：
- **内部 DSL** = 统一语言的 API 层（业务流程可读）
- **外部 DSL** = 统一语言的规则层（业务规则可配置）
- **TDD** = 验证 DSL 准确表达了业务意图

三者融合，代码就成为了精确的业务文档。

---

**下一章**：[完整 Vibe 项目实战](19-full-vibe-project.md)
