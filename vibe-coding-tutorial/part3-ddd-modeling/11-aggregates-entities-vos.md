# 第11章：聚合根、实体与值对象

> "三个构建块，构成整个领域模型的骨架。"

---

## 11.1 三者的核心区别

| 概念 | 有身份？ | 可变？ | 比较方式 | 生命周期 |
|------|---------|--------|---------|---------|
| **值对象** | 否 | 不可变 | 按值比较 | 短暂，随引用消亡 |
| **实体** | 是 | 可变 | 按 ID 比较 | 有自己的生命周期 |
| **聚合根** | 是（特殊实体） | 可变 | 按 ID 比较 | 控制整个聚合的生命周期 |

---

## 11.2 值对象（Value Object）

### 核心特征

1. **不可变**：创建后不能修改
2. **按值比较**：两个相同值的对象是相等的
3. **自我验证**：保证自身始终有效
4. **无副作用**：操作返回新对象

```python
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

@dataclass(frozen=True)  # frozen=True 保证不可变
class Money:
    amount: Decimal
    currency: str
    
    def __post_init__(self):
        # 自我验证
        if self.amount < 0:
            raise ValueError(f"金额不能为负：{self.amount}")
        if not self.currency:
            raise ValueError("货币代码不能为空")
        if len(self.currency) != 3:
            raise ValueError(f"货币代码必须是3位字母：{self.currency}")
    
    # 操作返回新对象（不修改自身）
    def add(self, other: 'Money') -> 'Money':
        self._assert_same_currency(other)
        return Money(self.amount + other.amount, self.currency)
    
    def subtract(self, other: 'Money') -> 'Money':
        self._assert_same_currency(other)
        if other.amount > self.amount:
            raise InsufficientFundsError()
        return Money(self.amount - other.amount, self.currency)
    
    def multiply(self, factor: Decimal) -> 'Money':
        return Money(
            (self.amount * factor).quantize(Decimal("0.01")),
            self.currency
        )
    
    def _assert_same_currency(self, other: 'Money'):
        if self.currency != other.currency:
            raise CurrencyMismatchError(self.currency, other.currency)
    
    def __str__(self):
        return f"{self.amount:.2f} {self.currency}"
```

### 测试值对象

```python
class TestMoney:
    
    def test_equality_by_value(self):
        """值对象按值相等"""
        assert Money(Decimal("100"), "CNY") == Money(Decimal("100"), "CNY")
    
    def test_different_values_not_equal(self):
        assert Money(Decimal("100"), "CNY") != Money(Decimal("200"), "CNY")
    
    def test_immutability(self):
        """值对象不可变"""
        money = Money(Decimal("100"), "CNY")
        with pytest.raises(AttributeError):
            money.amount = Decimal("200")
    
    def test_add_returns_new_object(self):
        m1 = Money(Decimal("100"), "CNY")
        m2 = Money(Decimal("50"), "CNY")
        result = m1.add(m2)
        assert result == Money(Decimal("150"), "CNY")
        assert m1 == Money(Decimal("100"), "CNY")  # 原对象未变
    
    def test_negative_amount_raises(self):
        with pytest.raises(ValueError):
            Money(Decimal("-1"), "CNY")
    
    def test_cross_currency_addition_raises(self):
        with pytest.raises(CurrencyMismatchError):
            Money(Decimal("100"), "CNY").add(Money(Decimal("100"), "USD"))
    
    def test_multiplication_rounds_to_2_decimals(self):
        price = Money(Decimal("99.99"), "CNY")
        result = price.multiply(Decimal("0.1"))
        assert result == Money(Decimal("10.00"), "CNY")
```

### 更多值对象例子

```python
@dataclass(frozen=True)
class EmailAddress:
    value: str
    
    def __post_init__(self):
        if "@" not in self.value:
            raise InvalidEmailError(self.value)
    
    def domain(self) -> str:
        return self.value.split("@")[1]

@dataclass(frozen=True)
class DateRange:
    start: date
    end: date
    
    def __post_init__(self):
        if self.end < self.start:
            raise ValueError("结束日期不能早于开始日期")
    
    def contains(self, d: date) -> bool:
        return self.start <= d <= self.end
    
    def overlaps(self, other: 'DateRange') -> bool:
        return self.start <= other.end and other.start <= self.end
    
    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1

@dataclass(frozen=True)
class Percentage:
    value: Decimal
    
    def __post_init__(self):
        if not (0 <= self.value <= 100):
            raise ValueError(f"百分比必须在0-100之间：{self.value}")
    
    def apply_to(self, amount: Money) -> Money:
        return amount.multiply(self.value / 100)
    
    def discount_factor(self) -> Decimal:
        return 1 - self.value / 100
```

---

## 11.3 实体（Entity）

### 核心特征

1. **有唯一身份**：通过 ID 区分，而非值
2. **可变**：状态会随时间变化
3. **按 ID 比较**：两个 ID 相同的对象是同一个实体
4. **封装行为**：不暴露内部状态，通过方法修改

```python
import uuid
from dataclasses import dataclass, field

@dataclass
class OrderItem:
    """订单项实体"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    product_id: str = ""
    product_name: str = ""
    quantity: int = 0
    unit_price: Money = field(default_factory=lambda: Money(Decimal("0"), "CNY"))
    
    def __eq__(self, other):
        if not isinstance(other, OrderItem):
            return False
        return self.id == other.id  # 按 ID 比较
    
    def __hash__(self):
        return hash(self.id)
    
    @property
    def subtotal(self) -> Money:
        return self.unit_price.multiply(Decimal(str(self.quantity)))
    
    def update_quantity(self, new_qty: int) -> None:
        if new_qty <= 0:
            raise InvalidQuantityError(f"数量必须大于0：{new_qty}")
        self.quantity = new_qty
```

---

## 11.4 聚合根（Aggregate Root）

### 聚合根的规则

1. **聚合根是事务边界**：一次操作只修改一个聚合
2. **外部只能持有聚合根的引用**：不能直接持有内部实体的引用
3. **聚合根负责维护不变量**：保证聚合内部的一致性
4. **聚合根发布领域事件**：将内部变化通知外部

```python
from dataclasses import dataclass, field
from typing import List, Optional
import uuid
from datetime import datetime

@dataclass
class Order:
    """订单聚合根"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    customer_id: str = ""
    _items: List[OrderItem] = field(default_factory=list, repr=False)
    status: OrderStatus = OrderStatus.DRAFT
    coupon: Optional[Coupon] = None
    _events: List = field(default_factory=list, repr=False)
    
    # ========== 工厂方法（控制创建） ==========
    
    @classmethod
    def place(cls, customer: Customer, items: List[OrderItem]) -> 'Order':
        """下单——工厂方法保证创建时的不变量"""
        if not items:
            raise EmptyOrderError("订单必须包含至少一个商品")
        
        order = cls(customer_id=customer.id)
        for item in items:
            order._items.append(item)
        
        # 记录领域事件
        order._events.append(OrderPlaced(
            order_id=order.id,
            customer_id=customer.id,
            total=order.total,
            placed_at=datetime.now()
        ))
        return order
    
    # ========== 命令方法（修改状态，维护不变量） ==========
    
    def add_item(self, item: OrderItem) -> None:
        """添加商品——维护不变量：只有草稿状态可以修改"""
        self._assert_is_draft("添加商品")
        self._items.append(item)
    
    def remove_item(self, item_id: str) -> None:
        """移除商品——维护不变量：至少保留一个"""
        self._assert_is_draft("移除商品")
        items_after = [i for i in self._items if i.id != item_id]
        if not items_after:
            raise CannotRemoveLastItemError("订单必须保留至少一个商品")
        self._items = items_after
    
    def apply_coupon(self, coupon: Coupon) -> None:
        """应用优惠券"""
        self._assert_is_draft("应用优惠券")
        if self.coupon is not None:
            raise CouponAlreadyAppliedError("每个订单只能使用一张优惠券")
        coupon.validate()
        self.coupon = coupon
    
    def confirm(self) -> None:
        """确认订单（命令方法：只修改状态，不返回值，遵循 CQS 原则）"""
        self._assert_is_draft("确认订单")
        self.status = OrderStatus.CONFIRMED
        self._events.append(OrderConfirmed(
            order_id=self.id,
            confirmed_at=datetime.now()
        ))
    
    def cancel(self, reason: str) -> None:
        """取消订单（命令方法：只修改状态，不返回值，遵循 CQS 原则）"""
        if self.status not in (OrderStatus.DRAFT, OrderStatus.CONFIRMED):
            raise InvalidOrderStateError(
                f"无法取消 {self.status.value} 状态的订单"
            )
        self.status = OrderStatus.CANCELLED
        self._events.append(OrderCancelled(
            order_id=self.id,
            reason=reason,
            cancelled_at=datetime.now()
        ))
    
    # ========== 查询方法（只读，不修改状态） ==========
    
    @property
    def items(self) -> List[OrderItem]:
        return list(self._items)  # 返回副本，防止外部修改
    
    @property
    def total(self) -> Money:
        if not self._items:
            return Money(Decimal("0"), "CNY")
        subtotals = [item.subtotal for item in self._items]
        base = subtotals[0]
        for s in subtotals[1:]:
            base = base.add(s)
        if self.coupon:
            discount = self.coupon.calculate_discount(base)
            base = base.subtract(discount)
        return base
    
    def pull_events(self) -> List:
        """取出并清空领域事件"""
        events = list(self._events)
        self._events.clear()
        return events
    
    # ========== 私有辅助方法 ==========
    
    def _assert_is_draft(self, operation: str) -> None:
        if self.status != OrderStatus.DRAFT:
            raise InvalidOrderStateError(
                f"无法在 {self.status.value} 状态下执行：{operation}"
            )
```

---

## 11.5 设计聚合边界

### 错误的边界设计

```python
# ❌ 错误：聚合太大，包含了不相干的概念
class MegaOrder:
    items: List[OrderItem]
    customer: Customer          # 不应该！Customer 是另一个聚合
    payment: Payment            # 不应该！Payment 有自己的边界
    shipment: Shipment          # 不应该！Shipment 有自己的边界
    
# 问题：修改 Order 时需要锁定 Customer、Payment、Shipment
# 高并发时产生大量竞争
```

```python
# ✅ 正确：只通过 ID 引用其他聚合
@dataclass
class Order:
    customer_id: CustomerId     # 只存 ID！
    payment_id: Optional[PaymentId] = None   # 只存 ID！
    shipment_id: Optional[ShipmentId] = None # 只存 ID！
    items: List[OrderItem] = field(default_factory=list)
    # OrderItem 是 Order 的内部实体，不是独立聚合
```

### 聚合边界判断原则

```
问1: 这个对象需要和 Order 一起修改吗？
     是 → 可能是同一个聚合的内部实体
     否 → 用 ID 引用

问2: 这个对象有自己独立的生命周期吗？
     是 → 是独立聚合，用 ID 引用
     否 → 可能是内部实体或值对象

问3: 这个对象可以独立存在（不依附于 Order）吗？
     是 → 独立聚合
     否 → 内部实体或值对象
```

---

## 11.6 综合测试：聚合根的完整测试

```python
class TestOrderAggregate:
    
    class TestOrderPlacement:
        def test_places_order_with_items(self): ...
        def test_placing_emits_order_placed_event(self): ...
        def test_cannot_place_empty_order(self): ...
    
    class TestItemManagement:
        def test_can_add_item_to_draft_order(self): ...
        def test_cannot_add_item_to_confirmed_order(self): ...
        def test_can_remove_item_from_draft_order(self): ...
        def test_cannot_remove_last_item(self): ...
    
    class TestCouponApplication:
        def test_can_apply_coupon_to_draft_order(self): ...
        def test_cannot_apply_two_coupons(self): ...
        def test_coupon_reduces_total(self): ...
        def test_expired_coupon_raises_error(self): ...
    
    class TestOrderConfirmation:
        def test_confirms_draft_order(self): ...
        def test_confirmation_emits_event(self): ...
        def test_cannot_confirm_already_confirmed_order(self): ...
    
    class TestOrderTotal:
        def test_total_sums_all_items(self): ...
        def test_total_applies_coupon_discount(self): ...
        def test_empty_order_total_is_zero(self): ...
    
    class TestOrderCancellation:
        def test_can_cancel_draft_order(self): ...
        def test_can_cancel_confirmed_order(self): ...
        def test_cannot_cancel_shipped_order(self): ...
        def test_cancellation_emits_event_with_reason(self): ...
```

---

## 11.7 领域服务：跨聚合的业务逻辑

当业务逻辑不自然地属于任何一个聚合根时，使用**领域服务**：

```python
class PricingService:
    """领域服务：跨聚合的定价计算"""
    
    def calculate_final_price(
        self, 
        order: Order, 
        customer: Customer,
        active_promotions: list
    ) -> Money:
        """综合订单、客户等级和促销活动计算最终价格"""
        base_price = order.subtotal
        
        # 客户等级折扣
        tier_discount = customer.tier.discount_rate
        price_after_tier = base_price.multiply(1 - tier_discount)
        
        # 叠加促销
        for promo in active_promotions:
            if promo.is_applicable(order, customer):
                price_after_tier = promo.apply(price_after_tier)
        
        return price_after_tier
```

### 何时使用领域服务

| 场景 | 放在哪里 |
|------|---------|
| 逻辑只涉及一个聚合 | 聚合根方法 |
| 逻辑跨越多个聚合 | **领域服务** |
| 协调多个领域对象 + 基础设施 | 应用服务 |

---

## 11.8 AI 辅助聚合边界设计

AI 可以帮你审查聚合设计决策：

### 提示词模板

```
我在设计电商订单系统的聚合。当前设计：
- Order 聚合根包含 OrderItem 列表
- Customer 是独立聚合
- Coupon 是值对象，属于 Order

请审查：
1. OrderItem 应该是实体还是值对象？
2. Coupon 放在 Order 内部是否合适？
3. 这个聚合的事务边界是否合理？
```

### 审查要点

- AI 建议拆分聚合时，是否会导致事务一致性问题？
- AI 建议合并聚合时，是否会导致并发冲突？

> **Vibe Coding 心法**：让 AI 挑战你的聚合设计，但最终的一致性边界由你决定。

---

## 总结

三个构建块的使用判断：

```
是否需要通过唯一 ID 区分？
├── 否 → 值对象（按值比较，不可变）
└── 是 → 是否是独立的事务边界？
    ├── 是 → 聚合根（外部通过 ID 引用，内部维护不变量）
    └── 否 → 实体（属于某个聚合，由聚合根管理）
```

---

**下一章**：[领域事件与 Saga](12-domain-events-sagas.md)
