# 第10章：聚合与聚合根

## 学习目标

- 深刻理解聚合作为一致性边界的本质
- 掌握设计聚合的四条核心规则
- 学会合理确定聚合的大小
- 理解聚合根的唯一访问入口原则

---

## 10.1 为什么需要聚合

先看一个没有聚合的问题：

```python
# 没有聚合时的并发问题
# 两个并发操作同时修改订单的不同部分

# 操作A（添加商品）：
order = order_repo.find("order-001")
order.items.append(new_item)
order_repo.save(order)

# 操作B（应用折扣）：
order = order_repo.find("order-001")  # 同时读取同一个订单
order.discount = 0.9
order_repo.save(order)  # 可能覆盖操作A的修改！

# 结果：数据不一致！
# 不变量被破坏：订单总价计算错误
```

更深层的问题：谁来保护**业务不变量**？

```
业务不变量（Business Invariant）：
  必须始终为真的业务规则

例子：
  - 订单总价 = 所有订单项的小计之和（修改任何项都必须同步总价）
  - 会议室在某时段只能有一个预订
  - 账户余额不能为负（除非有透支协议）
  
谁来保证这些规则？必须有一个"守门员"！
```

**聚合（Aggregate）** 就是这个守门员——它定义了一个**强一致性边界**，边界内的所有对象作为一个整体进行修改，不变量由聚合来保护。

---

## 10.2 聚合的核心概念

### 聚合（Aggregate）

```
聚合 = 一组相关对象的集合，作为一个数据一致性单元
     = 业务不变量的边界
     = 事务的边界（一次事务最多修改一个聚合）
```

### 聚合根（Aggregate Root）

```
聚合根 = 聚合的"大门"
       = 聚合中唯一可以被外部引用的实体
       = 所有外部操作的入口点

外界                聚合根           聚合内部对象
  ─────────────►   Order          OrderItem
                   （聚合根）    ← 只能通过Order访问
                                   Address（值对象）
```

---

## 10.3 聚合的四条核心规则

这四条规则来自Vaughn Vernon的著作，是设计聚合的黄金法则：

### 规则1：通过聚合根修改聚合内的对象

```python
# ❌ 违反规则：直接修改聚合内部对象
order = order_repo.get(order_id)
order.items[0].quantity = 5  # 绕过了聚合根！不变量可能被破坏

# ✅ 正确：通过聚合根方法修改
order = order_repo.get(order_id)
order.update_item_quantity("item-001", 5)  # 聚合根保护不变量

class Order:
    def update_item_quantity(self, item_id: str, new_quantity: int) -> None:
        """通过聚合根修改订单项数量"""
        if self._status not in (OrderStatus.DRAFT, OrderStatus.PLACED):
            raise OrderException("只有待付款前的订单可以修改")
        
        item = self._find_item(item_id)
        item.update_quantity(new_quantity)  # 聚合根委托给内部对象
        
        # 聚合根验证不变量
        self._validate_total_price()
```

### 规则2：跨聚合只使用标识引用

```python
# ❌ 违反规则：跨聚合持有对象引用
class Order:
    def __init__(self):
        self._customer: Customer = ...  # 持有Customer实例！
        # 这会把Order和Customer耦合在一起，形成一个巨大的聚合

# ✅ 正确：跨聚合只引用标识
class Order:
    def __init__(self, customer_id: CustomerId):
        self._customer_id = customer_id  # 只引用ID！
    
    # 需要Customer信息时，通过仓储加载
    # customer = customer_repo.get(self._customer_id)
```

**为什么只引用标识？**

```
对象引用 → 加载时会把整个关联对象图都加载 → 聚合变大
        → 事务边界扩大 → 并发冲突增加
        → 数据一致性更难保证

标识引用 → 各聚合独立加载 → 事务边界清晰
        → 可以分布在不同服务中 → 更好扩展
```

### 规则3：每次事务只修改一个聚合

```python
# ❌ 违反规则：一个事务修改多个聚合
with transaction():
    order = order_repo.get(order_id)
    order.place()
    
    inventory = inventory_repo.get(product_id)
    inventory.reserve(quantity)  # 同时修改库存！
    
    # 两个聚合都被修改：Order 和 Inventory
    # 如果任何一步失败，如何补偿？

# ✅ 正确：通过领域事件实现最终一致性
class Order:
    def place(self) -> None:
        self._status = OrderStatus.PLACED
        # 发布领域事件，而不是直接修改其他聚合
        self._record_event(OrderPlaced(
            order_id=self._id,
            items=[(item.product_id, item.quantity) for item in self._items]
        ))

# 库存上下文订阅事件，在独立事务中处理
class InventoryEventHandler:
    def on_order_placed(self, event: OrderPlaced) -> None:
        with transaction():
            for product_id, quantity in event.items:
                inventory = self._repo.get(product_id)
                inventory.reserve(quantity)
                self._repo.save(inventory)
```

### 规则4：通过最终一致性实现边界外的更新

```
当"一个聚合内更改时，另一个聚合需要响应"：
  不要用同步事务！
  用领域事件 + 最终一致性

时序：
  1. 修改聚合A，发布事件
  2. 提交事务，保存聚合A和事件
  3. 事件被异步处理
  4. 修改聚合B，提交事务

好处：
  ├── 每个聚合的事务独立
  ├── 部分失败可以重试（事件消费的幂等性）
  └── 系统更具弹性（聚合B暂时不可用，不影响聚合A的操作）
```

---

## 10.4 聚合的大小设计

聚合大小是一个需要权衡的问题：

### 过大的聚合

```python
# ❌ 过大的聚合：把所有相关的东西都放进来
class Order:
    _items: list[OrderItem]       # 订单项
    _customer: Customer            # 客户信息（应该是标识引用！）
    _payment: Payment              # 支付信息
    _shipment: Shipment            # 发货信息
    _refunds: list[Refund]         # 退款列表
    _reviews: list[Review]         # 评价列表
    _customer_service_tickets: ... # 客服工单

问题：
  - 并发操作频繁冲突（多人同时操作订单不同部分）
  - 加载缓慢（每次加载都拉取大量数据）
  - 边界模糊（订单是什么？什么都是？）
```

### 过小的聚合

```python
# 过小的聚合：把本应内聚的东西拆开
class Order:
    order_id: OrderId
    customer_id: CustomerId
    status: OrderStatus
    # 没有订单项！

class OrderItem:
    item_id: ItemId
    order_id: OrderId    # 外键引用
    product_id: str
    quantity: int

# 问题：
# 保护不变量变困难：
# "订单项数量不能超过库存" ← 跨越了两个聚合，谁来保证？
```

### 合理大小的判断标准

```python
# ✅ 合理的聚合：包含保护不变量所需的最小集合
class Order:
    """订单聚合
    
    聚合边界原则：包含以下不变量所需的全部对象：
    1. 订单总价 = 所有订单项小计之和
    2. 订单状态变更必须遵循状态机
    3. 单个订单不能超过最大商品种数限制
    """
    
    _id: OrderId
    _customer_id: CustomerId     # 引用，不是对象
    _items: list[OrderItem]      # 需要维护总价不变量，所以包含
    _status: OrderStatus
    _total: Money                # 冗余存储，保护不变量
    
    # ❌ 不包含：
    # _payment: Payment  → Payment是独立聚合
    # _shipment: Shipment → Shipment是独立聚合

class Payment:
    """支付聚合（独立于Order）"""
    _id: PaymentId
    _order_id: OrderId          # 只引用订单ID
    _amount: Money
    _status: PaymentStatus
    
class Shipment:
    """发货单聚合（独立于Order）"""
    _id: ShipmentId
    _order_id: OrderId          # 只引用订单ID
    _recipient: Address
    _tracking_number: str
    _status: ShipmentStatus
```

**决策规则**：
```
把X放入聚合Y的判断标准：
  √ 修改X时，Y的某个不变量需要被重新验证
  √ X没有自己独立的生命周期（X必须和Y一起创建/删除）
  √ X不会被外部频繁单独访问
  
  ✗ 修改X与Y的不变量无关 → X不属于聚合Y
  ✗ X有自己独立的生命周期 → X是独立聚合
  ✗ X会被大量并发操作 → X可能需要独立出去
```

---

## 10.5 聚合的完整实现

```python
from __future__ import annotations
from abc import ABC
from dataclasses import dataclass, field
from decimal import Decimal
from datetime import datetime
from enum import Enum
from typing import List, Optional
from uuid import UUID, uuid4

# ===== 值对象 =====

@dataclass(frozen=True)
class OrderId:
    value: UUID = field(default_factory=uuid4)

@dataclass(frozen=True)
class Money:
    amount: Decimal
    currency: str = "CNY"
    
    def __add__(self, other: Money) -> Money:
        if self.currency != other.currency:
            raise ValueError("货币不一致")
        return Money(self.amount + other.amount, self.currency)
    
    def __mul__(self, factor: Decimal) -> Money:
        return Money((self.amount * factor).quantize(Decimal("0.01")), self.currency)

@dataclass(frozen=True)
class ProductSnapshot:
    product_id: str
    name: str
    unit_price: Money

class OrderStatus(Enum):
    DRAFT = "draft"
    PLACED = "placed"
    PAID = "paid"
    CANCELLED = "cancelled"

# ===== 聚合内部实体（只能通过聚合根访问）=====

class OrderItem:
    """订单项（聚合内的实体，不是聚合根）"""
    
    def __init__(self, item_id: str, product: ProductSnapshot, quantity: int):
        self._id = item_id
        self._product = product
        self._quantity = quantity
    
    def update_quantity(self, new_quantity: int) -> None:
        if new_quantity <= 0:
            raise ValueError("数量必须大于0")
        self._quantity = new_quantity
    
    @property
    def subtotal(self) -> Money:
        return self._product.unit_price * Decimal(self._quantity)
    
    @property
    def id(self) -> str:
        return self._id
    
    @property
    def product_id(self) -> str:
        return self._product.product_id
    
    @property
    def quantity(self) -> int:
        return self._quantity

# ===== 聚合根 =====

MAX_ITEMS_PER_ORDER = 50  # 业务规则：单个订单最多50种商品

class Order:
    """订单聚合根
    
    保护的不变量：
    1. 订单项数量不超过 MAX_ITEMS_PER_ORDER
    2. 订单总价 = 所有订单项的小计之和
    3. 只有PLACED状态的订单才能被支付
    4. 只有DRAFT或PLACED状态的订单才能被取消
    """
    
    def __init__(self, order_id: OrderId, customer_id: str):
        self._id = order_id
        self._customer_id = customer_id
        self._items: List[OrderItem] = []
        self._status = OrderStatus.DRAFT
        self._created_at = datetime.now()
        self._events: List[DomainEvent] = []
    
    # ===== 命令方法（修改状态）=====
    
    def add_item(self, product: ProductSnapshot, quantity: int) -> None:
        """添加商品到订单"""
        self._ensure_status(OrderStatus.DRAFT, "添加商品")
        
        if len(self._items) >= MAX_ITEMS_PER_ORDER:
            raise OrderException(f"订单商品种数不能超过{MAX_ITEMS_PER_ORDER}种")
        
        if any(i.product_id == product.product_id for i in self._items):
            raise OrderException(f"商品 {product.product_id} 已在订单中")
        
        item_id = f"{self._id.value}-{len(self._items)+1}"
        self._items.append(OrderItem(item_id, product, quantity))
    
    def remove_item(self, product_id: str) -> None:
        """移除订单项"""
        self._ensure_status(OrderStatus.DRAFT, "移除商品")
        
        item = next((i for i in self._items if i.product_id == product_id), None)
        if not item:
            raise OrderException(f"订单中没有商品 {product_id}")
        
        self._items.remove(item)
    
    def update_item_quantity(self, product_id: str, new_quantity: int) -> None:
        """修改商品数量"""
        self._ensure_status(OrderStatus.DRAFT, "修改数量")
        
        item = next((i for i in self._items if i.product_id == product_id), None)
        if not item:
            raise OrderException(f"订单中没有商品 {product_id}")
        
        item.update_quantity(new_quantity)
    
    def place(self) -> None:
        """下单（提交订单）"""
        self._ensure_status(OrderStatus.DRAFT, "下单")
        
        if not self._items:
            raise OrderException("订单中没有商品，无法下单")
        
        self._status = OrderStatus.PLACED
        self._placed_at = datetime.now()
        
        # 记录领域事件（不直接修改其他聚合）
        self._record_event(OrderPlaced(
            order_id=self._id,
            customer_id=self._customer_id,
            items=[(i.product_id, i.quantity) for i in self._items],
            total=self.total,
            occurred_at=datetime.now()
        ))
    
    def mark_paid(self, payment_id: str, amount: Money) -> None:
        """标记为已支付"""
        self._ensure_status(OrderStatus.PLACED, "支付")
        
        if amount.amount < self.total.amount:
            raise OrderException(f"支付金额 {amount} 不足，应付 {self.total}")
        
        self._status = OrderStatus.PAID
        self._payment_id = payment_id
        self._paid_at = datetime.now()
        
        self._record_event(OrderPaid(
            order_id=self._id,
            payment_id=payment_id,
            occurred_at=datetime.now()
        ))
    
    def cancel(self, reason: str) -> None:
        """取消订单"""
        cancellable = {OrderStatus.DRAFT, OrderStatus.PLACED}
        if self._status not in cancellable:
            raise OrderException(f"状态为 {self._status.value} 的订单不可取消")
        
        self._status = OrderStatus.CANCELLED
        self._cancel_reason = reason
        self._cancelled_at = datetime.now()
        
        self._record_event(OrderCancelled(
            order_id=self._id,
            reason=reason,
            occurred_at=datetime.now()
        ))
    
    # ===== 查询方法（不修改状态）=====
    
    @property
    def id(self) -> OrderId:
        return self._id
    
    @property
    def status(self) -> OrderStatus:
        return self._status
    
    @property
    def total(self) -> Money:
        """计算订单总价（保证与订单项同步）"""
        if not self._items:
            return Money(Decimal("0"))
        result = self._items[0].subtotal
        for item in self._items[1:]:
            result = result + item.subtotal
        return result
    
    @property
    def item_count(self) -> int:
        return len(self._items)
    
    def get_item(self, product_id: str) -> Optional[OrderItem]:
        return next((i for i in self._items if i.product_id == product_id), None)
    
    # ===== 领域事件 =====
    
    def collect_events(self) -> List[DomainEvent]:
        """收集并清空待发布的事件"""
        events = list(self._events)
        self._events.clear()
        return events
    
    # ===== 私有辅助方法 =====
    
    def _ensure_status(self, expected: OrderStatus, operation: str) -> None:
        if self._status != expected:
            raise OrderException(
                f"操作'{operation}'要求订单状态为'{expected.value}'，"
                f"当前状态为'{self._status.value}'"
            )
    
    def _record_event(self, event: DomainEvent) -> None:
        self._events.append(event)
    
    def __eq__(self, other: object) -> bool:
        return isinstance(other, Order) and self._id == other._id
    
    def __hash__(self) -> int:
        return hash(self._id)
    
    def __repr__(self) -> str:
        return f"Order(id={self._id}, status={self._status.value}, total={self.total})"
```

---

## 10.6 聚合加载策略

```python
# 延迟加载 vs 即时加载
class OrderRepository:
    
    def get_with_items(self, order_id: OrderId) -> Order:
        """加载完整的订单（包含所有订单项）"""
        # 大多数操作都需要订单项来保护不变量
        ...
    
    def get_summary(self, order_id: OrderId) -> OrderSummary:
        """只加载订单汇总（适合只查询状态的场景）"""
        # 使用读模型，不需要完整聚合
        ...
```

---

## 本章小结

| 规则 | 内容 |
|------|------|
| 规则1 | 通过聚合根修改聚合内部 |
| 规则2 | 跨聚合只使用标识引用 |
| 规则3 | 每次事务只修改一个聚合 |
| 规则4 | 通过最终一致性实现边界外更新 |

**聚合大小**：包含保护不变量所需的最小集合，不多也不少。

---

## 思考练习

1. 在你系统中找一个"过大的聚合"，分析哪些部分可以被拆分出来
2. 找一个你系统中违反"每次事务只修改一个聚合"的场景，如何用领域事件改造它？
3. 以下哪些应该在同一个聚合内？为什么？
   - 博客文章 & 文章的评论
   - 博客文章 & 文章作者

---

**上一章：** [第09章：实体与值对象](./09-entities-value-objects.md)  
**下一章：** [第11章：领域事件](./11-domain-events.md)
