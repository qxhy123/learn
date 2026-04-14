# 基础类型定义

> 本文件包含教程中所有章节共享的基础类型。
> 在阅读任何章节的代码示例前，请确保理解这些类型的接口和用法。

---

## 异常类

```python
class CurrencyMismatchError(ValueError):
    """货币类型不匹配"""
    pass


class EmptyOrderError(ValueError):
    """空订单操作错误"""
    pass


class InsufficientPointsError(ValueError):
    """积分不足"""
    pass


class CouponAlreadyAppliedError(ValueError):
    """优惠券已使用"""
    pass


class InvalidOrderStateError(ValueError):
    """无效的订单状态操作"""
    pass


class DSLParseError(Exception):
    """DSL 解析错误"""
    pass


class DSLSemanticError(Exception):
    """DSL 语义错误"""
    pass
```

---

## 值对象：Money

货币值对象，始终使用 `Decimal` 确保精度，**绝不使用 `float`**。

```python
from decimal import Decimal
from dataclasses import dataclass

@dataclass(frozen=True)
class Money:
    """货币值对象：始终使用 Decimal 确保精度"""
    amount: Decimal
    currency: str

    def __post_init__(self):
        if not isinstance(self.amount, Decimal):
            object.__setattr__(self, 'amount', Decimal(str(self.amount)))

    def __add__(self, other: Money) -> Money:
        if isinstance(other, int) and other == 0:
            return self  # 支持 sum() 的起始值
        if not isinstance(other, Money):
            return NotImplemented
        if self.currency != other.currency:
            raise CurrencyMismatchError(
                f"不能将 {self.currency} 与 {other.currency} 相加"
            )
        return Money(self.amount + other.amount, self.currency)

    def __radd__(self, other) -> Money:
        """支持 sum() 内置函数：sum() 从 0 开始累加"""
        if isinstance(other, int) and other == 0:
            return self
        if isinstance(other, Money):
            return other.__add__(self)
        return NotImplemented

    def __sub__(self, other: Money) -> Money:
        if not isinstance(other, Money):
            return NotImplemented
        if self.currency != other.currency:
            raise CurrencyMismatchError(
                f"不能将 {self.currency} 与 {other.currency} 相减"
            )
        return Money(self.amount - other.amount, self.currency)

    def __mul__(self, factor) -> Money:
        if isinstance(factor, (int, float, Decimal)):
            return Money(self.amount * Decimal(str(factor)), self.currency)
        return NotImplemented

    def __rmul__(self, factor) -> Money:
        return self.__mul__(factor)

    def __gt__(self, other: Money) -> bool:
        self._check_currency(other)
        return self.amount > other.amount

    def __ge__(self, other: Money) -> bool:
        self._check_currency(other)
        return self.amount >= other.amount

    def __lt__(self, other: Money) -> bool:
        self._check_currency(other)
        return self.amount < other.amount

    def __le__(self, other: Money) -> bool:
        self._check_currency(other)
        return self.amount <= other.amount

    def multiply(self, factor) -> Money:
        """乘法运算（兼容教程中的方法调用风格）"""
        return self.__mul__(factor)

    def subtract(self, other: Money) -> Money:
        """减法运算（兼容教程中的方法调用风格）"""
        return self.__sub__(other)

    def _check_currency(self, other: Money) -> None:
        if self.currency != other.currency:
            raise CurrencyMismatchError(
                f"不能比较 {self.currency} 与 {other.currency}"
            )

    def __repr__(self) -> str:
        return f"Money({self.amount}, '{self.currency}')"
```

### 使用示例

```python
from decimal import Decimal

m1 = Money(Decimal("10"), "CNY")
m2 = Money(Decimal("20"), "CNY")

m1 + m2          # Money(30, 'CNY')
sum([m1, m2])    # Money(30, 'CNY')  ← __radd__ 使 sum() 可用
m1.multiply(3)   # Money(30, 'CNY')
```

---

## 枚举

```python
from enum import Enum
from decimal import Decimal

class OrderStatus(Enum):
    DRAFT = "draft"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    SHIPPED = "shipped"
    DELIVERED = "delivered"


class CustomerTier(Enum):
    REGULAR = "regular"
    VIP = "vip"
    PREMIUM = "premium"

    @property
    def discount_rate(self) -> Decimal:
        rates = {
            CustomerTier.REGULAR: Decimal("0"),
            CustomerTier.VIP: Decimal("0.1"),
            CustomerTier.PREMIUM: Decimal("0.2"),
        }
        return rates[self]
```

---

## 领域事件

```python
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

@dataclass(frozen=True)
class DomainEvent:
    """领域事件基类"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    occurred_at: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True)
class OrderPlaced(DomainEvent):
    """订单已创建事件"""
    order_id: str = field(default="")
    customer_id: str = field(default="")
    total: Optional[Money] = field(default=None)


@dataclass(frozen=True)
class OrderConfirmed(DomainEvent):
    """订单已确认事件"""
    order_id: str = field(default="")


@dataclass(frozen=True)
class OrderCancelled(DomainEvent):
    """订单已取消事件"""
    order_id: str = field(default="")
    reason: str = field(default="")
```

---

## 事件总线

支持装饰器方式订阅，修复了教程早期版本中 `subscribe` 接口不一致的问题：

```python
from typing import Callable, Type

class EventBus:
    """事件总线：支持装饰器方式订阅"""

    def __init__(self):
        self._handlers: dict[Type, list[Callable]] = {}

    def subscribe(self, event_type: Type) -> Callable:
        """装饰器方式订阅事件

        用法：
            @bus.subscribe(OrderConfirmed)
            def handle_confirmed(event):
                ...
        """
        def decorator(handler: Callable) -> Callable:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(handler)
            return handler
        return decorator

    def publish(self, event: DomainEvent) -> None:
        """发布事件，通知所有订阅者"""
        event_type = type(event)
        for handler in self._handlers.get(event_type, []):
            handler(event)
```

---

## 优惠券

```python
from decimal import Decimal

@dataclass(frozen=True)
class Coupon:
    """优惠券值对象"""
    code: str
    discount_rate: Decimal

    def validate(self) -> None:
        """验证优惠券有效性"""
        if self.discount_rate <= 0 or self.discount_rate >= 1:
            raise ValueError(f"折扣率必须在 0~1 之间，当前：{self.discount_rate}")

    def calculate_discount(self, price: Money) -> Money:
        """计算折扣金额"""
        self.validate()
        discount_amount = price.amount * self.discount_rate
        return Money(discount_amount, price.currency)
```

---

## 订单项（值对象）

在 DDD 中，`OrderItem` 是**值对象**而非实体——它没有独立于订单的生命周期，
两个属性完全相同的订单项在业务上是等价的。

```python
@dataclass(frozen=True)
class OrderItem:
    """订单项值对象：通过属性值比较相等性"""
    product_name: str
    price: Money
    quantity: int

    @property
    def subtotal(self) -> Money:
        return self.price.multiply(self.quantity)
```

---

## 订单聚合根

命令方法（`confirm`、`cancel`）遵循 CQS 原则：只修改状态，不返回值。

```python
from typing import List, Optional

@dataclass
class Order:
    """订单聚合根"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    customer_id: str = ""
    status: OrderStatus = OrderStatus.DRAFT
    items: List[OrderItem] = field(default_factory=list)
    coupon: Optional[Coupon] = None
    _events: List[DomainEvent] = field(default_factory=list, repr=False)

    @classmethod
    def place(cls, customer_id: str, items: List[OrderItem]) -> Order:
        """创建订单"""
        if not items:
            raise EmptyOrderError("订单必须包含至少一个商品")
        order = cls(customer_id=customer_id, items=items)
        order._record_event(OrderPlaced(
            order_id=order.id,
            customer_id=customer_id,
            total=order.total,
        ))
        return order

    @property
    def total(self) -> Money:
        if not self.items:
            return Money(Decimal("0"), "CNY")
        subtotal = sum(item.subtotal for item in self.items)
        if self.coupon:
            discount = self.coupon.calculate_discount(subtotal)
            return subtotal - discount
        return subtotal

    def add_item(self, item: OrderItem) -> None:
        """添加订单项"""
        self._assert_is_draft("添加商品")
        self.items.append(item)

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
        if not self.items:
            raise EmptyOrderError("不能确认空订单")
        self.status = OrderStatus.CONFIRMED
        self._record_event(OrderConfirmed(order_id=self.id))

    def cancel(self, reason: str = "") -> None:
        """取消订单（命令方法：只修改状态，不返回值，遵循 CQS 原则）"""
        if self.status not in (OrderStatus.DRAFT, OrderStatus.CONFIRMED):
            raise InvalidOrderStateError(
                f"状态 {self.status.value} 的订单不能取消"
            )
        self.status = OrderStatus.CANCELLED
        self._record_event(OrderCancelled(order_id=self.id, reason=reason))

    def _assert_is_draft(self, action: str) -> None:
        if self.status != OrderStatus.DRAFT:
            raise InvalidOrderStateError(
                f"只有草稿状态的订单才能{action}，当前状态：{self.status.value}"
            )

    def _record_event(self, event: DomainEvent) -> None:
        self._events.append(event)

    def pull_events(self) -> List[DomainEvent]:
        """获取并清空待发布的领域事件"""
        events = self._events.copy()
        self._events.clear()
        return events
```

---

## 客户

```python
@dataclass
class Customer:
    """客户实体"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    tier: CustomerTier = CustomerTier.REGULAR
```
