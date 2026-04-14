# 第09章：实体与值对象

## 学习目标

- 理解实体与值对象的本质区别（标识 vs 值）
- 掌握设计实体的关键原则
- 掌握值对象的不变性设计
- 学会识别哪些概念应建模为值对象

---

## 9.1 两种看待世界的方式

现实世界中的事物，我们可以用两种方式来看待：

**方式一：通过标识来追踪**

> 你的银行账户，无论里面有多少钱，它都是"你的那个账户"。即使余额从100元变成10000元，账户的标识没有变。

**方式二：通过值本身来判断**

> 你钱包里的一张100元和另一张100元，对你来说没有任何区别——你不会说"我要用那张特定的100元"，任何100元对你都一样。

这两种方式，对应DDD中的**实体**和**值对象**：

```
实体（Entity）：通过唯一标识来区分，即使属性变化也是"同一个东西"
值对象（Value Object）：通过属性值来区分，值相同就是同一个东西
```

---

## 9.2 实体（Entity）

### 核心特征

**实体的本质**：有唯一标识（Identity），生命周期内属性可能变化，但标识不变。

```python
class BankAccount:
    """银行账户（实体）"""
    
    def __init__(self, account_id: AccountId, owner: str):
        self._id = account_id   # 唯一标识：账户号
        self._owner = owner
        self._balance = Money(Decimal("0"), "CNY")
    
    # 通过标识判断相等性
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BankAccount):
            return False
        return self._id == other._id  # 只比较标识！
    
    def __hash__(self) -> int:
        return hash(self._id)
    
    # 属性可以变化，但仍然是"同一个账户"
    def deposit(self, amount: Money) -> None:
        self._balance = self._balance + amount
    
    def withdraw(self, amount: Money) -> None:
        if amount > self._balance:
            raise InsufficientFundsError("余额不足")
        self._balance = self._balance - amount
```

### 实体设计原则

#### 原则1：标识设计

```python
# 坏的做法：用自然键作为标识（可能变化）
class User:
    def __init__(self, email: str):  # ❌ email可能变化
        self._id = email

# 好的做法：使用不可变的人工标识
from uuid import UUID, uuid4

class UserId:
    """强类型标识，防止混淆"""
    def __init__(self, value: UUID = None):
        self._value = value or uuid4()
    
    def __eq__(self, other):
        return isinstance(other, UserId) and self._value == other._value
    
    def __hash__(self):
        return hash(self._value)
    
    def __str__(self):
        return str(self._value)

class User:
    def __init__(self, user_id: UserId, email: str):
        self._id = user_id  # ✅ 永远不变的标识
        self._email = email  # 可以变化的属性
```

#### 原则2：封装状态变化（行为而非setter）

```python
# ❌ 贫血模型：只有getter/setter，没有行为
class Order:
    def set_status(self, status): self._status = status
    def set_payment_id(self, pid): self._payment_id = pid
    # 业务规则在哪里？全部散落在Service层

# ✅ 充血模型：业务规则封装在实体中
class Order:
    def confirm_payment(self, payment: Payment) -> None:
        """确认支付 - 业务规则内聚在此"""
        if self._status != OrderStatus.PENDING_PAYMENT:
            raise OrderException(f"订单{self._id}当前状态不支持支付确认")
        if payment.amount < self._total:
            raise OrderException("支付金额不足")
        self._status = OrderStatus.PAID
        self._payment_id = payment.id
        self._paid_at = datetime.now()
        self._record_event(PaymentConfirmed(self._id, payment.id))
    
    def cancel(self, reason: CancelReason) -> None:
        """取消订单"""
        cancellable_statuses = {OrderStatus.PENDING_PAYMENT, OrderStatus.PAID}
        if self._status not in cancellable_statuses:
            raise OrderException("当前状态的订单不可取消")
        self._status = OrderStatus.CANCELLED
        self._cancel_reason = reason
        self._cancelled_at = datetime.now()
        self._record_event(OrderCancelled(self._id, reason))
```

#### 原则3：保护不变量

```python
class Order:
    def add_item(self, product: Product, quantity: int) -> None:
        # 业务不变量：只有草稿状态可以添加商品
        if self._status != OrderStatus.DRAFT:
            raise OrderException("只有草稿订单才能添加商品")
        
        # 业务不变量：数量必须为正
        if quantity <= 0:
            raise ValueError("商品数量必须大于0")
        
        # 业务不变量：同一商品不重复添加，而是更新数量
        existing = self._find_item(product.id)
        if existing:
            existing.increase_quantity(quantity)
        else:
            self._items.append(OrderItem(product, quantity))
```

---

## 9.3 值对象（Value Object）

### 核心特征

**值对象的本质**：无唯一标识，通过属性值判断相等性，**不可变**（immutable）。

```python
from dataclasses import dataclass
from decimal import Decimal

@dataclass(frozen=True)  # frozen=True 使其不可变
class Money:
    """货币金额（值对象）"""
    amount: Decimal
    currency: str
    
    def __post_init__(self):
        # 约束验证在创建时进行
        if self.amount < 0:
            raise ValueError("金额不能为负数")
        if not self.currency:
            raise ValueError("货币代码不能为空")
    
    # 通过值判断相等性（dataclass自动生成）
    # Money(100, "CNY") == Money(100, "CNY") → True
    
    # 值对象的操作返回新对象，而非修改自身
    def add(self, other: "Money") -> "Money":
        if self.currency != other.currency:
            raise ValueError(f"不同货币不能相加: {self.currency} vs {other.currency}")
        return Money(self.amount + other.amount, self.currency)
    
    def subtract(self, other: "Money") -> "Money":
        result = self.amount - other.amount
        if result < 0:
            raise ValueError("减法结果不能为负")
        return Money(result, self.currency)
    
    def multiply(self, factor: Decimal) -> "Money":
        return Money((self.amount * factor).quantize(Decimal("0.01")), self.currency)
    
    def __add__(self, other: "Money") -> "Money":
        return self.add(other)
    
    def __str__(self) -> str:
        return f"{self.currency} {self.amount:.2f}"
    
    @classmethod
    def zero(cls, currency: str) -> "Money":
        return cls(Decimal("0"), currency)
```

### 不变性的重要性

```python
# ❌ 可变的值对象会导致意外问题
class MutableMoney:
    def __init__(self, amount, currency):
        self.amount = amount
        self.currency = currency

price = MutableMoney(100, "CNY")
order_total = price  # 共享同一个对象

# 某处代码修改了 price
price.amount = 200

# order_total 也被意外修改了！
print(order_total.amount)  # 200，而不是100

# ✅ 不可变值对象：安全共享
price = Money(Decimal("100"), "CNY")
discount = Money(Decimal("20"), "CNY")
final_price = price - discount  # 返回新对象
# price 和 discount 本身没有变化
```

### 值对象的常见例子

```python
# 地址（一组不可分割的值）
@dataclass(frozen=True)
class Address:
    province: str
    city: str
    district: str
    detail: str
    postal_code: str
    
    def full_address(self) -> str:
        return f"{self.province}{self.city}{self.district}{self.detail}"

# 日期范围
@dataclass(frozen=True)
class DateRange:
    start: date
    end: date
    
    def __post_init__(self):
        if self.end < self.start:
            raise ValueError("结束日期不能早于开始日期")
    
    def contains(self, d: date) -> bool:
        return self.start <= d <= self.end
    
    def overlaps(self, other: "DateRange") -> bool:
        return self.start <= other.end and other.start <= self.end
    
    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1

# 手机号
@dataclass(frozen=True)
class PhoneNumber:
    number: str
    
    def __post_init__(self):
        import re
        if not re.match(r'^1[3-9]\d{9}$', self.number):
            raise ValueError(f"无效的手机号: {self.number}")
    
    def masked(self) -> str:
        """返回脱敏格式：138****1234"""
        return f"{self.number[:3]}****{self.number[-4:]}"

# 地理坐标
@dataclass(frozen=True)
class GeoCoordinate:
    latitude: float
    longitude: float
    
    def distance_to(self, other: "GeoCoordinate") -> float:
        """计算两点间距离（公里）"""
        import math
        R = 6371
        lat1, lon1 = math.radians(self.latitude), math.radians(self.longitude)
        lat2, lon2 = math.radians(other.latitude), math.radians(other.longitude)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
        return R * 2 * math.asin(math.sqrt(a))
```

---

## 9.4 如何判断：实体还是值对象？

这是实践中最常见的问题。以下是一个决策框架：

### 核心问题：这个概念需要被追踪吗？

```
问题1：两个"相同"的实例是否可以互换？
  - 是（如两张100元可以互换） → 值对象
  - 否（如两个订单，即使内容相同也不同） → 实体

问题2：这个概念有独立的生命周期吗？
  - 有（如用户、订单、账户） → 实体
  - 没有（它依附于某个实体存在） → 考虑值对象

问题3：我们需要追踪它的历史变化吗？
  - 需要（如账户余额变更历史） → 实体
  - 不需要 → 值对象
```

### 一些容易混淆的例子

```python
# 例1：地址 - 值对象还是实体？
# 
# 电商场景：送货地址是用户的一个属性值，可替换 → 值对象
# 地理信息系统：每个地址有唯一ID，需要被追踪 → 实体
#
# 结论：取决于上下文！

# 例2：产品/商品
# 
# 商品目录上下文：Product是实体（有唯一商品ID）
# 订单上下文：OrderItem中的商品快照是值对象
#   （订单创建时记录的价格/名称，即使商品后来变了也不影响历史订单）
#
@dataclass(frozen=True)
class ProductSnapshot:  # 值对象！不随商品变化而变化
    product_id: str
    name: str
    price: Money
    captured_at: datetime  # 记录快照时间

class OrderItem:
    def __init__(self, product_snapshot: ProductSnapshot, quantity: int):
        self._product = product_snapshot  # 用快照，不用实体引用
        self._quantity = quantity
```

---

## 9.5 实体与值对象的持久化

```python
# 实体持久化：有自己的表，有主键
# orders 表：
# | id | customer_id | status | total_amount | total_currency |

# 值对象持久化策略1：嵌入到宿主实体的表中
# (Address 嵌入 User 表)
# users 表：
# | id | name | addr_province | addr_city | addr_district | addr_detail |

# 值对象持久化策略2：序列化为JSON字段
# orders 表：
# | id | shipping_address (JSON) | items (JSON) |

# 实现示例（使用 SQLAlchemy）
from sqlalchemy import Column, String, Numeric
from sqlalchemy.orm import composite

class OrderORM(Base):
    __tablename__ = "orders"
    
    id = Column(String(36), primary_key=True)
    
    # Money 值对象映射为两个列
    total_amount = Column(Numeric(10, 2))
    total_currency = Column(String(3))
    
    # 使用 composite 将两个列映射为 Money 值对象
    total = composite(Money, total_amount, total_currency)
    
    # Address 值对象映射为 JSON
    shipping_address_json = Column(String)
    
    @property
    def shipping_address(self) -> Address:
        data = json.loads(self.shipping_address_json)
        return Address(**data)
    
    @shipping_address.setter
    def shipping_address(self, addr: Address):
        self.shipping_address_json = json.dumps(dataclasses.asdict(addr))
```

---

## 9.6 综合示例：电商订单

```python
from __future__ import annotations
from dataclasses import dataclass, field
from decimal import Decimal
from datetime import datetime
from enum import Enum
from typing import List
from uuid import UUID, uuid4

# ===== 值对象 =====

@dataclass(frozen=True)
class OrderId:
    value: UUID = field(default_factory=uuid4)
    def __str__(self): return str(self.value)

@dataclass(frozen=True)
class Money:
    amount: Decimal
    currency: str = "CNY"
    
    def __add__(self, other: Money) -> Money:
        assert self.currency == other.currency
        return Money(self.amount + other.amount, self.currency)
    
    def __mul__(self, factor: Decimal) -> Money:
        return Money((self.amount * factor).quantize(Decimal("0.01")), self.currency)
    
    def __gt__(self, other: Money) -> bool:
        assert self.currency == other.currency
        return self.amount > other.amount

@dataclass(frozen=True)
class ProductSnapshot:
    """订单中商品的快照（不可变记录）"""
    product_id: str
    name: str
    unit_price: Money

class OrderStatus(Enum):
    DRAFT = "draft"
    PLACED = "placed"
    PAID = "paid"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"

# ===== 实体（OrderItem，属于Order聚合） =====

class OrderItem:
    def __init__(self, product: ProductSnapshot, quantity: int):
        if quantity <= 0:
            raise ValueError("数量必须大于0")
        self._product = product
        self._quantity = quantity
    
    @property
    def subtotal(self) -> Money:
        return self._product.unit_price * Decimal(self._quantity)
    
    @property
    def product_id(self) -> str:
        return self._product.product_id

# ===== 实体（Order，作为聚合根） =====

class Order:
    """订单实体（聚合根）"""
    
    def __init__(self, order_id: OrderId, customer_id: str):
        self._id = order_id
        self._customer_id = customer_id
        self._items: List[OrderItem] = []
        self._status = OrderStatus.DRAFT
        self._created_at = datetime.now()
    
    def __eq__(self, other: object) -> bool:
        return isinstance(other, Order) and self._id == other._id
    
    def __hash__(self) -> int:
        return hash(self._id)
    
    def add_item(self, product: ProductSnapshot, quantity: int) -> None:
        if self._status != OrderStatus.DRAFT:
            raise ValueError("只有草稿订单可以添加商品")
        
        # 同商品合并
        for item in self._items:
            if item.product_id == product.product_id:
                raise ValueError("商品已存在，请修改数量")
        
        self._items.append(OrderItem(product, quantity))
    
    def place(self) -> None:
        """下单"""
        if not self._items:
            raise ValueError("订单中没有商品")
        if self._status != OrderStatus.DRAFT:
            raise ValueError("只有草稿订单可以下单")
        self._status = OrderStatus.PLACED
    
    @property
    def total(self) -> Money:
        if not self._items:
            return Money(Decimal("0"))
        totals = [item.subtotal for item in self._items]
        result = totals[0]
        for t in totals[1:]:
            result = result + t
        return result
    
    @property
    def id(self) -> OrderId:
        return self._id
    
    @property
    def status(self) -> OrderStatus:
        return self._status


# ===== 使用示例 =====

# 创建订单（实体）
order = Order(OrderId(), customer_id="user-123")

# 商品快照（值对象）
iphone_snapshot = ProductSnapshot(
    product_id="prod-001",
    name="iPhone 15",
    unit_price=Money(Decimal("7999"))
)

# 添加商品
order.add_item(iphone_snapshot, quantity=1)

# 两个快照值相同则相等（值对象特性）
snap1 = ProductSnapshot("prod-001", "iPhone 15", Money(Decimal("7999")))
snap2 = ProductSnapshot("prod-001", "iPhone 15", Money(Decimal("7999")))
assert snap1 == snap2  # True：值对象，按值比较

# 两个订单即使内容相同，标识不同则不等（实体特性）
order1 = Order(OrderId(), "user-123")
order2 = Order(OrderId(), "user-123")
assert order1 != order2  # True：实体，按标识比较
```

---

## 本章小结

| 维度 | 实体 | 值对象 |
|------|------|--------|
| 标识 | 有唯一标识 | 无标识 |
| 相等性 | 标识相同则相等 | 所有属性值相同则相等 |
| 可变性 | 状态可变 | 不可变（返回新对象） |
| 生命周期 | 独立生命周期 | 依附于实体 |
| 例子 | 用户、订单、账户 | 金额、地址、日期范围 |

---

## 思考练习

1. 以下概念在你的系统中应该是实体还是值对象？为什么？
   - 商品SKU
   - 配送地址
   - 优惠券
   - 操作日志记录
   - 文件标签

2. 找一个你系统中的"实体"，检查它是否封装了业务规则，还是只是一个数据容器？

3. 为你系统中一个重要的业务概念设计一个值对象，确保它是不可变的，并包含自我验证逻辑。

---

**上一章：** [第08章：事件风暴](../part2-strategic-design/08-event-storming.md)  
**下一章：** [第10章：聚合与聚合根](./10-aggregates.md)
