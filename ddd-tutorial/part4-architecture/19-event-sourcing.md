# 第19章：事件溯源（Event Sourcing）

## 学习目标

- 理解事件溯源的核心思想：事件即状态
- 掌握事件溯源聚合的实现方式
- 理解快照（Snapshot）优化机制
- 了解事件溯源的优缺点及适用场景

---

## 19.1 传统存储 vs 事件溯源

### 传统方式：存储当前状态

```
传统存储：只保存最新状态
                                      数据库中
时间    事件                          orders表当前状态
─────  ──────────────────────────    ─────────────────────
T1     订单已创建（total=1000）   →   {status: "placed", total: 1000}
T2     优惠券已应用（discount=100）→  {status: "placed", total: 900}   (覆盖)
T3     支付已完成              →      {status: "paid", total: 900}     (覆盖)
T4     订单已发货              →      {status: "shipped", total: 900}  (覆盖)

问题：
  - 历史丢失：不知道折扣前原价是多少
  - 无法审计：谁在什么时候改了什么？
  - 无法回溯：系统在T2时刻的状态是什么？
```

### 事件溯源：存储事件序列

```
事件溯源：保存所有事件，当前状态由事件推导而来
                                      事件存储
时间    事件                          （只追加，不修改）
─────  ──────────────────────────    ─────────────────────
T1     OrderCreated(total=1000)  →   [OrderCreated(total=1000)]
T2     CouponApplied(disc=100)   →   [OrderCreated, CouponApplied]
T3     PaymentCompleted          →   [OrderCreated, CouponApplied, PaymentCompleted]
T4     OrderShipped              →   [OrderCreated, CouponApplied, PaymentCompleted, OrderShipped]

优势：
  ✅ 完整历史：每次变更都有记录
  ✅ 完整审计：谁在何时做了什么
  ✅ 时间旅行：可以重放到任意时刻
  ✅ 事件自然成为集成的基础
```

---

## 19.2 事件溯源的核心原理

> **聚合的当前状态 = 从空状态开始，按顺序应用所有历史事件的结果**

```python
# 伪代码演示原理
def load_order(order_id: str, event_store: EventStore) -> Order:
    events = event_store.load(aggregate_id=order_id)
    
    order = Order.__new__(Order)  # 创建空对象
    for event in events:
        order.apply(event)         # 逐一应用事件
    
    return order

# 等价于：
# order = 空状态
# order.apply(OrderCreated(...))      → status=DRAFT, total=1000
# order.apply(CouponApplied(...))     → total=900
# order.apply(PaymentCompleted(...))  → status=PAID
# order.apply(OrderShipped(...))      → status=SHIPPED
# 最终状态：{status: SHIPPED, total: 900}
```

---

## 19.3 事件溯源聚合的实现

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import List, Type
from uuid import UUID, uuid4

# ===== 事件基类 =====

@dataclass(frozen=True)
class DomainEvent(ABC):
    aggregate_id: str
    event_id: UUID = field(default_factory=uuid4)
    version: int = 0               # 聚合的版本号（乐观锁）
    occurred_at: datetime = field(default_factory=datetime.now)

# ===== 具体事件 =====

@dataclass(frozen=True)
class OrderCreated(DomainEvent):
    customer_id: str = ""
    currency: str = "CNY"

@dataclass(frozen=True)
class OrderItemAdded(DomainEvent):
    product_id: str = ""
    product_name: str = ""
    unit_price: Decimal = Decimal("0")
    quantity: int = 0

@dataclass(frozen=True)
class OrderPlaced(DomainEvent):
    total_amount: Decimal = Decimal("0")

@dataclass(frozen=True)
class CouponApplied(DomainEvent):
    coupon_code: str = ""
    discount_amount: Decimal = Decimal("0")

@dataclass(frozen=True)
class PaymentCompleted(DomainEvent):
    payment_id: str = ""
    amount: Decimal = Decimal("0")

@dataclass(frozen=True)
class OrderCancelled(DomainEvent):
    reason: str = ""

# ===== 事件溯源聚合基类 =====

class EventSourcedAggregate(ABC):
    """事件溯源聚合基类"""
    
    def __init__(self):
        self._uncommitted_events: List[DomainEvent] = []
        self._version: int = 0
    
    def _record_event(self, event: DomainEvent) -> None:
        """记录事件：先应用到当前状态，再加入待提交列表"""
        self._apply(event)
        self._uncommitted_events.append(event)
        self._version += 1
    
    def _apply(self, event: DomainEvent) -> None:
        """分发事件到对应的 when_* 处理方法"""
        handler_name = f"_when_{type(event).__name__}"
        handler = getattr(self, handler_name, None)
        if handler:
            handler(event)
    
    @classmethod
    def reconstitute(cls, events: List[DomainEvent]) -> "EventSourcedAggregate":
        """从事件历史重建聚合"""
        instance = cls.__new__(cls)
        instance._uncommitted_events = []
        instance._version = 0
        
        for event in events:
            instance._apply(event)
            instance._version += 1
        
        return instance
    
    def collect_uncommitted_events(self) -> List[DomainEvent]:
        events = list(self._uncommitted_events)
        self._uncommitted_events.clear()
        return events
    
    @property
    def version(self) -> int:
        return self._version


# ===== 事件溯源的 Order 聚合 =====

class OrderStatus(Enum):
    NONE = "none"
    DRAFT = "draft"
    PLACED = "placed"
    PAID = "paid"
    SHIPPED = "shipped"
    CANCELLED = "cancelled"

class Order(EventSourcedAggregate):
    """事件溯源的订单聚合
    
    状态不直接修改，而是通过事件来变化。
    """
    
    def __init__(self):
        super().__init__()
        # 所有状态字段初始值（无意义的初始状态）
        self._id: str = ""
        self._customer_id: str = ""
        self._status: OrderStatus = OrderStatus.NONE
        self._items: List[dict] = []
        self._discount: Decimal = Decimal("0")
        self._total: Decimal = Decimal("0")
    
    # ===== 命令方法（产生事件）=====
    
    @classmethod
    def create(cls, customer_id: str) -> "Order":
        order = cls()
        order_id = str(uuid4())
        order._record_event(OrderCreated(
            aggregate_id=order_id,
            customer_id=customer_id,
        ))
        return order
    
    def add_item(self, product_id: str, product_name: str, unit_price: Decimal, quantity: int) -> None:
        if self._status != OrderStatus.DRAFT:
            raise ValueError("只有草稿订单才能添加商品")
        self._record_event(OrderItemAdded(
            aggregate_id=self._id,
            product_id=product_id,
            product_name=product_name,
            unit_price=unit_price,
            quantity=quantity,
        ))
    
    def place(self) -> None:
        if self._status != OrderStatus.DRAFT:
            raise ValueError("只有草稿订单才能下单")
        if not self._items:
            raise ValueError("订单为空")
        
        total = sum(i["unit_price"] * i["quantity"] for i in self._items)
        total -= self._discount
        
        self._record_event(OrderPlaced(
            aggregate_id=self._id,
            total_amount=total,
        ))
    
    def apply_coupon(self, coupon_code: str, discount_amount: Decimal) -> None:
        if self._status != OrderStatus.DRAFT:
            raise ValueError("只有草稿订单才能使用优惠券")
        self._record_event(CouponApplied(
            aggregate_id=self._id,
            coupon_code=coupon_code,
            discount_amount=discount_amount,
        ))
    
    def complete_payment(self, payment_id: str, amount: Decimal) -> None:
        if self._status != OrderStatus.PLACED:
            raise ValueError("只有已下单的订单才能完成支付")
        self._record_event(PaymentCompleted(
            aggregate_id=self._id,
            payment_id=payment_id,
            amount=amount,
        ))
    
    def cancel(self, reason: str) -> None:
        if self._status in (OrderStatus.SHIPPED,):
            raise ValueError("已发货订单不可取消")
        self._record_event(OrderCancelled(
            aggregate_id=self._id,
            reason=reason,
        ))
    
    # ===== 事件处理方法（when_* 方法，纯状态变更，无业务规则）=====
    
    def _when_OrderCreated(self, event: OrderCreated) -> None:
        self._id = event.aggregate_id
        self._customer_id = event.customer_id
        self._status = OrderStatus.DRAFT
        self._items = []
        self._discount = Decimal("0")
        self._total = Decimal("0")
    
    def _when_OrderItemAdded(self, event: OrderItemAdded) -> None:
        self._items.append({
            "product_id": event.product_id,
            "product_name": event.product_name,
            "unit_price": event.unit_price,
            "quantity": event.quantity,
        })
    
    def _when_CouponApplied(self, event: CouponApplied) -> None:
        self._discount += event.discount_amount
    
    def _when_OrderPlaced(self, event: OrderPlaced) -> None:
        self._status = OrderStatus.PLACED
        self._total = event.total_amount
    
    def _when_PaymentCompleted(self, event: PaymentCompleted) -> None:
        self._status = OrderStatus.PAID
    
    def _when_OrderCancelled(self, event: OrderCancelled) -> None:
        self._status = OrderStatus.CANCELLED
    
    @property
    def id(self) -> str: return self._id
    @property
    def status(self) -> OrderStatus: return self._status
    @property
    def total(self) -> Decimal: return self._total
```

---

## 19.4 事件存储（Event Store）实现

```python
from typing import List, Optional
import json

class EventStore:
    """事件存储：只追加，不修改（Append-Only）"""
    
    def append(
        self, 
        aggregate_id: str, 
        events: List[DomainEvent],
        expected_version: int  # 乐观锁：防止并发冲突
    ) -> None:
        """追加事件
        
        expected_version: 客户端认为的当前版本号
        如果不匹配，说明有并发修改，抛出异常
        """
        current_version = self._get_current_version(aggregate_id)
        
        if current_version != expected_version:
            raise ConcurrencyException(
                f"乐观锁冲突：期望版本 {expected_version}，实际版本 {current_version}"
            )
        
        for i, event in enumerate(events):
            self._db.execute("""
                INSERT INTO event_store 
                (aggregate_id, event_type, event_data, version, occurred_at)
                VALUES (:agg_id, :type, :data, :version, :occurred_at)
            """, {
                "agg_id": aggregate_id,
                "type": type(event).__name__,
                "data": json.dumps(self._serialize(event)),
                "version": current_version + i + 1,
                "occurred_at": event.occurred_at,
            })
    
    def load(self, aggregate_id: str, from_version: int = 0) -> List[DomainEvent]:
        """加载聚合的所有事件"""
        rows = self._db.execute("""
            SELECT event_type, event_data, version
            FROM event_store
            WHERE aggregate_id = :agg_id AND version > :from_version
            ORDER BY version ASC
        """, {"agg_id": aggregate_id, "from_version": from_version})
        
        return [self._deserialize(row["event_type"], row["event_data"]) for row in rows]
    
    def _serialize(self, event: DomainEvent) -> dict:
        from dataclasses import asdict
        return asdict(event)
    
    def _deserialize(self, event_type: str, data: str) -> DomainEvent:
        event_class = EVENT_REGISTRY[event_type]
        return event_class(**json.loads(data))
```

---

## 19.5 快照（Snapshot）优化

当聚合有大量历史事件时，每次加载都重放所有事件会很慢：

```python
class SnapshotStore:
    """快照存储：定期保存聚合快照"""
    
    def save(self, snapshot: AggregateSnapshot) -> None:
        self._db.upsert("snapshots", {
            "aggregate_id": snapshot.aggregate_id,
            "version": snapshot.version,
            "data": json.dumps(snapshot.state),
            "created_at": datetime.now()
        })
    
    def load(self, aggregate_id: str) -> Optional[AggregateSnapshot]:
        row = self._db.get_latest("snapshots", aggregate_id)
        if row:
            return AggregateSnapshot(
                aggregate_id=aggregate_id,
                version=row["version"],
                state=json.loads(row["data"])
            )
        return None


class EventSourcedOrderRepository:
    """使用事件溯源的订单仓储"""
    
    SNAPSHOT_THRESHOLD = 50  # 每50个事件创建一次快照
    
    def __init__(self, event_store: EventStore, snapshot_store: SnapshotStore):
        self._events = event_store
        self._snapshots = snapshot_store
    
    def get(self, order_id: str) -> Order:
        # 1. 尝试加载快照
        snapshot = self._snapshots.load(order_id)
        
        if snapshot:
            # 从快照恢复（不需要重放所有事件）
            order = Order.from_snapshot(snapshot.state)
            # 只加载快照之后的增量事件
            recent_events = self._events.load(order_id, from_version=snapshot.version)
        else:
            order = Order.__new__(Order)
            order._uncommitted_events = []
            order._version = 0
            recent_events = self._events.load(order_id)
        
        # 应用增量事件
        for event in recent_events:
            order._apply(event)
            order._version += 1
        
        return order
    
    def save(self, order: Order) -> None:
        uncommitted = order.collect_uncommitted_events()
        if not uncommitted:
            return
        
        # 追加事件（带乐观锁）
        self._events.append(
            aggregate_id=order.id,
            events=uncommitted,
            expected_version=order.version - len(uncommitted)
        )
        
        # 检查是否需要创建快照
        if order.version % self.SNAPSHOT_THRESHOLD == 0:
            self._snapshots.save(AggregateSnapshot(
                aggregate_id=order.id,
                version=order.version,
                state=order.to_snapshot_dict()
            ))
```

---

## 19.6 时间旅行：回放到任意时刻

```python
# 事件溯源的时间旅行能力
def replay_order_at(order_id: str, target_time: datetime) -> Order:
    """查看订单在某个历史时刻的状态"""
    all_events = event_store.load(order_id)
    
    # 只应用目标时间之前的事件
    events_before = [e for e in all_events if e.occurred_at <= target_time]
    
    return Order.reconstitute(events_before)

# 使用示例
# 查看昨天下午3点时订单的状态
historical_state = replay_order_at(
    "order-001", 
    datetime(2024, 1, 15, 15, 0, 0)
)
print(historical_state.status)  # 那时候的状态
```

---

## 19.7 事件溯源的适用场景

```
适合事件溯源：
  ✅ 审计需求强（金融、医疗、法律合规）
  ✅ 业务价值体现在历史中（如何到达当前状态）
  ✅ 需要时间旅行（重放分析、调试）
  ✅ 与CQRS结合，支持多种读模型
  ✅ 事件本身是核心业务概念（如交易记录）

不适合事件溯源：
  ❌ 简单CRUD业务，历史不重要
  ❌ 团队没有事件溯源经验
  ❌ 聚合事件非常频繁（如IoT传感器数据）
  ❌ 不需要审计和历史追溯
```

---

## 本章小结

| 特性 | 说明 |
|------|------|
| 核心原则 | 状态 = 事件的重放结果 |
| 存储方式 | 只追加，不修改，永久保留 |
| 加载方式 | 重放历史事件 |
| 快照 | 优化大量事件的加载性能 |
| 时间旅行 | 可重放到任意历史时刻 |
| 最佳搭档 | CQRS（读写分离） |

---

**上一章：** [第18章：CQRS](./18-cqrs.md)  
**下一章：** [第20章：DDD与微服务](../part5-advanced/20-ddd-microservices.md)
