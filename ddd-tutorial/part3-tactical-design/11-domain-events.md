# 第11章：领域事件（Domain Events）

## 学习目标

- 理解领域事件的语义及其与技术消息的区别
- 掌握领域事件的设计原则
- 实现领域事件的发布与订阅机制
- 理解领域事件在跨聚合协调中的作用

---

## 11.1 什么是领域事件

**领域事件（Domain Event）** 是领域中已经发生的重要事情的记录。

关键词：
- **已经发生**：过去时，不可撤销
- **重要的**：对业务有意义，不是每个状态变化都值得建模为领域事件
- **事情**：一个具体的业务动作的结果

```python
# 技术事件（不是领域事件）：
on_database_row_updated()
on_field_changed()
on_cache_invalidated()

# 领域事件：
class OrderPlaced:        # 订单已下单
class PaymentCompleted:   # 支付已完成
class ItemShipped:        # 商品已发货
class UserRegistered:     # 用户已注册
class PasswordResetRequested:  # 密码重置已申请
```

---

## 11.2 领域事件的设计原则

### 原则1：不可变性

事件一旦发生就不能改变：

```python
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

@dataclass(frozen=True)  # 不可变
class OrderPlaced:
    """订单已下单事件"""
    order_id: str
    customer_id: str
    total_amount: Decimal
    currency: str
    occurred_at: datetime  # 发生时间，不可变
    
    # 不能修改，只能读取
```

### 原则2：命名反映业务事实

```
命名规范：
  [领域概念] + [动词过去时]
  
好的命名：
  ✅ OrderPlaced（订单已下）
  ✅ PaymentFailed（支付失败）
  ✅ UserEmailVerified（用户邮箱已验证）
  ✅ ItemStockDepleted（商品库存耗尽）
  
不好的命名：
  ❌ OrderUpdated（太模糊，什么更新？）
  ❌ StatusChanged（完全没有业务语义）
  ❌ DatabaseRowInserted（技术事件，不是领域事件）
```

### 原则3：包含足够的上下文

```python
# ❌ 信息不足：订阅者需要再去查询才能处理
@dataclass(frozen=True)
class OrderPlaced:
    order_id: str  # 只有ID，订阅者还得去查订单内容

# ✅ 包含处理所需的信息
@dataclass(frozen=True)
class OrderPlaced:
    order_id: str
    customer_id: str
    items: tuple  # 用tuple保证不可变性
    total_amount: Decimal
    currency: str
    shipping_address: dict  # 地址信息快照
    occurred_at: datetime
    
    # 原则：事件中包含订阅者处理这个事件所需的全部信息
    # 避免订阅者再去查询——那会造成时序问题
```

### 原则4：事件是过去时，不是命令

```
事件 vs 命令：

命令（Command）：请求做某件事，可能被拒绝
  PlaceOrder（下单）
  SendEmail（发送邮件）
  ProcessPayment（处理支付）

事件（Event）：已经发生的事实，不可拒绝
  OrderPlaced（订单已下）
  EmailSent（邮件已发送）
  PaymentProcessed（支付已处理）

关键区别：
  命令 → 执行者可以拒绝 → 有失败的可能
  事件 → 已经发生 → 不能被"拒绝"，只能响应
```

---

## 11.3 领域事件的实现

### 基础设施

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Type
from uuid import UUID, uuid4

# 领域事件基类
@dataclass(frozen=True)
class DomainEvent(ABC):
    event_id: UUID = field(default_factory=uuid4)
    occurred_at: datetime = field(default_factory=datetime.now)

# 具体事件
@dataclass(frozen=True)
class OrderPlaced(DomainEvent):
    order_id: str = ""
    customer_id: str = ""
    total_amount: Decimal = Decimal("0")
    currency: str = "CNY"

@dataclass(frozen=True)
class PaymentCompleted(DomainEvent):
    payment_id: str = ""
    order_id: str = ""
    amount: Decimal = Decimal("0")

@dataclass(frozen=True)
class OrderCancelled(DomainEvent):
    order_id: str = ""
    reason: str = ""
```

### 事件发布机制

**方式一：聚合内收集，应用服务发布（推荐）**

```python
# 聚合收集事件
class Order:
    def __init__(self):
        self._events: List[DomainEvent] = []
    
    def place(self) -> None:
        # ... 修改状态 ...
        self._events.append(OrderPlaced(
            order_id=str(self._id),
            customer_id=self._customer_id,
            total_amount=self.total.amount,
        ))
    
    def collect_events(self) -> List[DomainEvent]:
        """收集事件（清空内部列表）"""
        events = list(self._events)
        self._events.clear()
        return events

# 应用服务在保存后发布事件
class OrderApplicationService:
    def place_order(self, command: PlaceOrderCommand) -> str:
        order = self._order_repo.get(command.order_id)
        order.place()
        
        # 先保存聚合
        self._order_repo.save(order)
        
        # 再发布事件（保存成功后）
        for event in order.collect_events():
            self._event_bus.publish(event)
        
        return str(order.id)
```

**方式二：事件总线（Event Bus）**

```python
# 简单的内存事件总线实现
class EventBus:
    def __init__(self):
        self._handlers: Dict[Type[DomainEvent], List[Callable]] = {}
    
    def subscribe(self, event_type: Type[DomainEvent], handler: Callable) -> None:
        """注册事件处理器"""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
    
    def publish(self, event: DomainEvent) -> None:
        """发布事件，同步调用所有处理器"""
        event_type = type(event)
        handlers = self._handlers.get(event_type, [])
        for handler in handlers:
            handler(event)


# 使用装饰器简化订阅
class EventHandler:
    def __init__(self, event_bus: EventBus):
        self._bus = event_bus
    
    def on(self, event_type: Type[DomainEvent]):
        """装饰器：订阅事件"""
        def decorator(func: Callable) -> Callable:
            self._bus.subscribe(event_type, func)
            return func
        return decorator


# 示例：库存上下文订阅订单事件
handler = EventHandler(event_bus)

@handler.on(OrderPlaced)
def reserve_stock_on_order_placed(event: OrderPlaced) -> None:
    """订单下单后，锁定库存"""
    # 这在库存上下文中执行，独立事务
    for item in event.items:
        inventory = inventory_repo.get(item["product_id"])
        inventory.reserve(item["quantity"])
        inventory_repo.save(inventory)

@handler.on(OrderCancelled)
def release_stock_on_order_cancelled(event: OrderCancelled) -> None:
    """订单取消后，释放库存"""
    ...
```

---

## 11.4 领域事件的跨聚合协调

领域事件是实现"每次事务只修改一个聚合"规则的关键机制：

```python
# 场景：用户注册后，发送欢迎邮件 + 初始化积分账户 + 发送推送通知

# ❌ 错误做法：一个事务做所有事
class UserRegistrationService:
    def register(self, email: str, password: str) -> None:
        with transaction():
            user = User.create(email, password)
            self._user_repo.save(user)
            
            # 跨上下文操作，违反规则3
            self._email_service.send_welcome_email(email)
            self._points_repo.create_account(user.id)
            self._push_service.send_registration_notification(user.id)
            
            # 如果邮件服务挂了，用户注册也失败？这不合理！

# ✅ 正确做法：通过领域事件解耦
class User:
    @classmethod
    def register(cls, email: str, password: str) -> "User":
        user = cls(UserId(), email, password)
        user._record_event(UserRegistered(
            user_id=str(user._id),
            email=email,
            registered_at=datetime.now()
        ))
        return user

class UserRegistrationService:
    def register(self, command: RegisterUserCommand) -> str:
        user = User.register(command.email, command.password)
        self._user_repo.save(user)  # 只保存用户
        
        # 发布事件，其他处理器异步响应
        for event in user.collect_events():
            self._event_bus.publish(event)
        
        return str(user.id)

# 各上下文独立响应事件
@handler.on(UserRegistered)
def send_welcome_email(event: UserRegistered) -> None:
    email_service.send_welcome(event.email)  # 失败不影响注册

@handler.on(UserRegistered)
def initialize_points_account(event: UserRegistered) -> None:
    points_service.create_account(event.user_id)

@handler.on(UserRegistered)
def send_registration_push(event: UserRegistered) -> None:
    push_service.notify(event.user_id)
```

---

## 11.5 持久化领域事件（Event Store）

在生产系统中，领域事件通常需要持久化：

```python
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class StoredEvent:
    """持久化存储的事件记录"""
    event_id: str
    event_type: str          # 事件类名
    aggregate_id: str        # 关联的聚合ID
    aggregate_type: str      # 聚合类名
    event_data: str          # 事件内容（JSON）
    occurred_at: datetime
    published: bool = False  # 是否已发布到消息队列

class EventStore:
    """事件存储（发件箱模式 Outbox Pattern）"""
    
    def save_events(
        self, 
        aggregate_id: str, 
        aggregate_type: str,
        events: List[DomainEvent]
    ) -> None:
        for event in events:
            stored = StoredEvent(
                event_id=str(event.event_id),
                event_type=type(event).__name__,
                aggregate_id=aggregate_id,
                aggregate_type=aggregate_type,
                event_data=self._serialize(event),
                occurred_at=event.occurred_at,
                published=False
            )
            self._db.save(stored)
    
    def get_unpublished_events(self) -> List[StoredEvent]:
        """获取尚未发布的事件（用于发件箱模式）"""
        return self._db.find_where(published=False)
    
    def mark_published(self, event_id: str) -> None:
        self._db.update(event_id, published=True)


# 发件箱模式（Outbox Pattern）：保证事件最终被发布
# 适用于消息队列（Kafka/RabbitMQ）场景

class OutboxPublisher:
    """定期扫描未发布事件并发送到消息队列"""
    
    def publish_pending(self) -> None:
        pending = self._event_store.get_unpublished_events()
        for stored_event in pending:
            try:
                self._message_queue.publish(
                    topic=stored_event.event_type,
                    message=stored_event.event_data
                )
                self._event_store.mark_published(stored_event.event_id)
            except Exception:
                # 失败时跳过，下次重试
                pass
```

---

## 11.6 事件的幂等处理

在分布式环境中，事件可能被重复消费，处理器必须是幂等的：

```python
class StockReservationHandler:
    def handle(self, event: OrderPlaced) -> None:
        """幂等处理：同一事件多次处理，结果相同"""
        
        # 使用事件ID作为幂等键
        if self._already_processed(event.event_id):
            return  # 已处理，直接返回
        
        # 执行业务逻辑
        for item in event.items:
            inventory = self._repo.get(item["product_id"])
            inventory.reserve(item["quantity"])
            self._repo.save(inventory)
        
        # 记录已处理
        self._mark_processed(event.event_id)
    
    def _already_processed(self, event_id: UUID) -> bool:
        return self._processed_events.exists(str(event_id))
    
    def _mark_processed(self, event_id: UUID) -> None:
        self._processed_events.add(str(event_id))
```

---

## 本章小结

| 特性 | 说明 |
|------|------|
| 不可变 | 事件已发生，不能修改 |
| 过去时命名 | 反映业务事实，非技术操作 |
| 充足信息 | 订阅者无需额外查询 |
| 解耦机制 | 跨聚合、跨上下文的核心集成手段 |
| 持久化 | 发件箱模式保证最终发布 |
| 幂等处理 | 分布式环境下必须保证 |

---

## 思考练习

1. 列出你系统中5个重要的业务"事实"，将它们设计为领域事件（命名、包含字段）
2. 找一个你系统中多个服务同步调用的业务流程，思考如何用领域事件重构为异步协作
3. 领域事件 vs 消息队列消息 vs 数据库触发器——它们有什么本质区别？

---

**上一章：** [第10章：聚合与聚合根](./10-aggregates.md)  
**下一章：** [第12章：仓储](./12-repositories.md)
