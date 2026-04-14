# 第12章：领域事件与 Saga

> "系统不只是由对象组成的，还由发生的事情组成的。"

---

## 12.1 领域事件是什么

**领域事件**：在领域中发生了某件重要的事情，需要记录和通知其他部分。

关键特征：
- **过去式命名**：`OrderConfirmed`（不是 `ConfirmOrder`）
- **不可变**：事件一旦发生，不可撤销（只能发补偿事件）
- **包含时间戳**：精确记录发生时间
- **包含上下文**：足够的信息让订阅者处理

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import uuid

@dataclass(frozen=True)
class DomainEvent:
    """所有领域事件的基类"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    occurred_at: datetime = field(default_factory=datetime.now)

@dataclass(frozen=True)
class OrderPlaced(DomainEvent):
    order_id: str = ""
    customer_id: str = ""
    total: Money = field(default_factory=lambda: Money(Decimal("0"), "CNY"))
    items: tuple = ()  # 使用 tuple 保持不可变

@dataclass(frozen=True)
class OrderConfirmed(DomainEvent):
    order_id: str = ""
    confirmed_at: datetime = field(default_factory=datetime.now)

@dataclass(frozen=True)
class OrderCancelled(DomainEvent):
    order_id: str = ""
    reason: str = ""

@dataclass(frozen=True)
class PaymentProcessed(DomainEvent):
    payment_id: str = ""
    order_id: str = ""
    amount: Money = field(default_factory=lambda: Money(Decimal("0"), "CNY"))

@dataclass(frozen=True)
class InventoryReserved(DomainEvent):
    reservation_id: str = ""
    order_id: str = ""

@dataclass(frozen=True)
class InventoryReservationFailed(DomainEvent):
    order_id: str = ""
    reason: str = ""
```

---

## 12.2 事件的发布与订阅

### 简单的内存事件总线

```python
from typing import Callable, Type, Dict, List
from collections import defaultdict

class EventBus:
    """简单的同步事件总线（测试和简单场景）"""
    
    def __init__(self):
        self._handlers: Dict[Type, List[Callable]] = defaultdict(list)
    
    def subscribe(self, event_type: Type) -> Callable:
        """装饰器方式订阅事件
        
        用法：
            @bus.subscribe(OrderConfirmed)
            def handle(event): ...
        """
        def decorator(handler: Callable) -> Callable:
            self._handlers[event_type].append(handler)
            return handler
        return decorator
    
    def publish(self, event: DomainEvent) -> None:
        handlers = self._handlers.get(type(event), [])
        for handler in handlers:
            handler(event)
    
    def publish_all(self, events: List[DomainEvent]) -> None:
        for event in events:
            self.publish(event)

# 使用示例
bus = EventBus()

# 订阅事件
@bus.subscribe(OrderConfirmed)
def send_confirmation_email(event: OrderConfirmed):
    email_service.send(order_id=event.order_id)

@bus.subscribe(OrderConfirmed)
def reserve_inventory(event: OrderConfirmed):
    inventory_service.reserve(order_id=event.order_id)

# 发布事件
order.confirm()
bus.publish_all(order.pull_events())
```

### 测试事件发布

```python
class TestOrderEvents:
    
    def test_placing_order_emits_order_placed_event(self):
        customer = make_customer()
        order = Order.place(customer=customer, items=[make_item()])
        
        events = order.pull_events()
        
        assert len(events) == 1
        event = events[0]
        assert isinstance(event, OrderPlaced)
        assert event.customer_id == customer.id
        assert event.order_id == order.id
    
    def test_confirming_order_emits_order_confirmed_event(self):
        order = make_draft_order()
        order.confirm()
        
        events = order.pull_events()
        
        assert any(isinstance(e, OrderConfirmed) for e in events)
    
    def test_pull_events_clears_event_list(self):
        order = make_draft_order()
        order.confirm()
        
        first_pull = order.pull_events()
        second_pull = order.pull_events()
        
        assert len(first_pull) == 1
        assert len(second_pull) == 0  # 第二次拉取为空
    
    def test_event_handler_called_on_order_confirmed(self):
        bus = EventBus()
        received_events = []
        bus.subscribe(OrderConfirmed, received_events.append)
        
        order = make_draft_order()
        order.confirm()
        bus.publish_all(order.pull_events())
        
        assert len(received_events) == 1
        assert received_events[0].order_id == order.id
```

---

## 12.3 Saga 模式：跨聚合的业务流程

### 为什么需要 Saga

在 DDD 中，每个事务只能修改一个聚合根。但业务流程常常跨越多个聚合：

```
下单流程涉及：
1. 创建 Order（订单聚合）
2. 预留库存（库存聚合）
3. 处理支付（支付聚合）
4. 创建物流单（物流聚合）

这是4个独立的聚合，不能在一个事务中完成！
```

**Saga** 是协调这种跨聚合业务流程的模式：
- 将长业务流程分解为多个本地事务
- 每步成功后发出事件，触发下一步
- 失败时执行补偿操作（回滚）

### Saga 的两种实现

**编排式 Saga（Choreography）**：没有中央控制，每个服务监听事件并决定下一步

```python
# 各个上下文自己处理
class InventoryContext:
    def on_order_confirmed(self, event: OrderConfirmed):
        try:
            reservation = self.reserve_for_order(event.order_id)
            self.bus.publish(InventoryReserved(order_id=event.order_id))
        except InsufficientInventoryError:
            self.bus.publish(InventoryReservationFailed(
                order_id=event.order_id,
                reason="库存不足"
            ))

class PaymentContext:
    def on_inventory_reserved(self, event: InventoryReserved):
        # 库存预留成功后，处理支付
        ...

class OrderContext:
    def on_inventory_reservation_failed(self, event: InventoryReservationFailed):
        # 库存不足，取消订单
        order = self.repo.find(event.order_id)
        order.cancel(reason=event.reason)
        self.repo.save(order)
```

**编制式 Saga（Orchestration）**：中央 Saga 编排器协调所有步骤

```python
class OrderFulfillmentSaga:
    """下单履行 Saga 编排器"""
    
    def __init__(self, inventory_service, payment_service, shipping_service):
        self._inventory = inventory_service
        self._payment = payment_service
        self._shipping = shipping_service
    
    def start(self, order: Order) -> SagaResult:
        saga_id = str(uuid.uuid4())
        
        # 步骤1：预留库存
        reservation = self._reserve_inventory(order, saga_id)
        if not reservation.success:
            return SagaResult.failed(reason=reservation.error)
        
        # 步骤2：处理支付
        payment = self._process_payment(order, saga_id)
        if not payment.success:
            # 补偿：释放库存
            self._inventory.release(reservation.id)
            return SagaResult.failed(reason=payment.error)
        
        # 步骤3：创建物流单
        shipment = self._create_shipment(order, saga_id)
        if not shipment.success:
            # 补偿：退款 + 释放库存
            self._payment.refund(payment.id)
            self._inventory.release(reservation.id)
            return SagaResult.failed(reason=shipment.error)
        
        return SagaResult.succeeded(
            reservation_id=reservation.id,
            payment_id=payment.id,
            shipment_id=shipment.id
        )
    
    def _reserve_inventory(self, order: Order, saga_id: str):
        try:
            reservation = self._inventory.reserve(order.items, saga_id=saga_id)
            return StepResult.succeeded(id=reservation.id)
        except InsufficientInventoryError as e:
            return StepResult.failed(error=str(e))
```

---

## 12.4 用 TDD 测试 Saga

```python
class TestOrderFulfillmentSaga:
    
    def test_successful_fulfillment_completes_all_steps(self):
        """成功的履行完成所有三步"""
        mock_inventory = Mock()
        mock_payment = Mock()
        mock_shipping = Mock()
        
        mock_inventory.reserve.return_value = Reservation(id="r1")
        mock_payment.process.return_value = Payment(id="p1")
        mock_shipping.create.return_value = Shipment(id="s1")
        
        saga = OrderFulfillmentSaga(mock_inventory, mock_payment, mock_shipping)
        order = make_confirmed_order()
        
        result = saga.start(order)
        
        assert result.success is True
        assert result.reservation_id == "r1"
        assert result.payment_id == "p1"
        assert result.shipment_id == "s1"
    
    def test_inventory_failure_cancels_saga_immediately(self):
        """库存不足时，Saga 立即失败，不执行后续步骤"""
        mock_inventory = Mock()
        mock_payment = Mock()
        
        mock_inventory.reserve.side_effect = InsufficientInventoryError()
        
        saga = OrderFulfillmentSaga(mock_inventory, mock_payment, Mock())
        result = saga.start(make_confirmed_order())
        
        assert result.success is False
        mock_payment.process.assert_not_called()  # 支付步骤没有执行
    
    def test_payment_failure_releases_inventory(self):
        """支付失败时，补偿操作释放已预留的库存"""
        mock_inventory = Mock()
        mock_payment = Mock()
        
        reservation = Reservation(id="r1")
        mock_inventory.reserve.return_value = reservation
        mock_payment.process.side_effect = PaymentFailedError("余额不足")
        
        saga = OrderFulfillmentSaga(mock_inventory, mock_payment, Mock())
        result = saga.start(make_confirmed_order())
        
        assert result.success is False
        # 验证补偿操作：库存被释放
        mock_inventory.release.assert_called_once_with("r1")
    
    def test_shipping_failure_refunds_and_releases_inventory(self):
        """物流创建失败时，退款并释放库存"""
        mock_inventory = Mock()
        mock_payment = Mock()
        mock_shipping = Mock()
        
        mock_inventory.reserve.return_value = Reservation(id="r1")
        mock_payment.process.return_value = Payment(id="p1")
        mock_shipping.create.side_effect = ShippingUnavailableError()
        
        saga = OrderFulfillmentSaga(mock_inventory, mock_payment, mock_shipping)
        result = saga.start(make_confirmed_order())
        
        assert result.success is False
        mock_payment.refund.assert_called_once_with("p1")   # 退款
        mock_inventory.release.assert_called_once_with("r1")  # 释放库存
```

---

## 12.5 事件溯源（Event Sourcing）

当需要完整历史记录时，可以将事件作为唯一的真相来源：

```python
class OrderEventStore:
    """订单事件存储——通过重放事件重建状态"""
    
    def __init__(self):
        self._events: Dict[str, List[DomainEvent]] = defaultdict(list)
    
    def append(self, order_id: str, events: List[DomainEvent]) -> None:
        self._events[order_id].extend(events)
    
    def load_events(self, order_id: str) -> List[DomainEvent]:
        return list(self._events[order_id])

class OrderProjection:
    """通过重放事件重建 Order 状态"""
    
    def rebuild(self, events: List[DomainEvent]) -> OrderState:
        state = OrderState()
        for event in events:
            self._apply(state, event)
        return state
    
    def _apply(self, state: OrderState, event: DomainEvent) -> None:
        if isinstance(event, OrderPlaced):
            state.order_id = event.order_id
            state.customer_id = event.customer_id
            state.status = OrderStatus.DRAFT
        elif isinstance(event, OrderConfirmed):
            state.status = OrderStatus.CONFIRMED
        elif isinstance(event, OrderCancelled):
            state.status = OrderStatus.CANCELLED
            state.cancel_reason = event.reason
```

---

## 12.6 综合实战：闪购系统的事件流

```python
# 闪购场景：多人同时抢购同一商品，用事件驱动保证一致性

class FlashSaleStarted(DomainEvent):
    product_id: str = ""
    available_qty: int = 0
    sale_price: Money = field(default_factory=lambda: Money(Decimal("0"), "CNY"))

class PurchaseAttempted(DomainEvent):
    attempt_id: str = ""
    customer_id: str = ""
    product_id: str = ""

class PurchaseSucceeded(DomainEvent):
    attempt_id: str = ""
    order_id: str = ""

class PurchaseFailed(DomainEvent):
    attempt_id: str = ""
    reason: str = ""  # "SOLD_OUT" | "DUPLICATE" | "RATE_LIMITED"

# 闪购 Saga
class FlashSaleSaga:
    def handle_purchase_attempt(self, event: PurchaseAttempted):
        # 1. 检查是否已购买（幂等）
        if self._has_purchased(event.customer_id, event.product_id):
            self.bus.publish(PurchaseFailed(
                attempt_id=event.attempt_id,
                reason="DUPLICATE"
            ))
            return
        
        # 2. 尝试扣减库存（原子操作）
        success = self._inventory.try_decrement(event.product_id)
        if not success:
            self.bus.publish(PurchaseFailed(
                attempt_id=event.attempt_id,
                reason="SOLD_OUT"
            ))
            return
        
        # 3. 创建订单
        order = Order.place(
            customer_id=event.customer_id,
            items=[FlashSaleItem(product_id=event.product_id)]
        )
        self.bus.publish(PurchaseSucceeded(
            attempt_id=event.attempt_id,
            order_id=order.id
        ))
```

---

## 12.6 AI 辅助事件流设计

在 Vibe Coding 工作流中，AI 可以帮助你设计事件流和 Saga 补偿逻辑：

### 提示词模板

```
我正在设计一个电商系统的领域事件流。

业务流程：用户下单 → 扣减库存 → 支付 → 发货

请帮我：
1. 列出每个步骤可能产生的领域事件
2. 设计 Saga 的补偿操作（每个步骤失败时如何回滚）
3. 指出哪些步骤需要幂等性保证
```

### AI 输出示例

```
事件流设计：
1. OrderPlaced → InventoryReserved → PaymentProcessed → ShipmentCreated
2. 补偿链：
   - 发货失败 → CancelPayment + ReleaseInventory
   - 支付失败 → ReleaseInventory
   - 库存不足 → RejectOrder
3. 幂等性：支付和库存操作必须支持幂等重试
```

### 审查要点

- AI 是否遗漏了补偿操作本身失败的情况？
- 事件命名是否符合统一语言词汇表？
- 是否考虑了并发场景（同时下单争抢库存）？

> **Vibe Coding 心法**：让 AI 生成 Saga 骨架，你专注于审查补偿逻辑的完整性。

---

## 12.7 仓储与应用服务概览

领域模型需要与外部世界连接。**仓储**负责持久化，**应用服务**负责协调：

### 仓储模式

```python
from abc import ABC, abstractmethod

class OrderRepository(ABC):
    """仓储接口（定义在领域层）"""
    
    @abstractmethod
    def find_by_id(self, order_id: str) -> Order:
        ...
    
    @abstractmethod
    def save(self, order: Order) -> None:
        ...


class InMemoryOrderRepository(OrderRepository):
    """内存实现（用于测试）"""
    
    def __init__(self):
        self._orders: dict[str, Order] = {}
    
    def find_by_id(self, order_id: str) -> Order:
        return self._orders[order_id]
    
    def save(self, order: Order) -> None:
        self._orders[order.id] = order
```

### 应用服务

```python
class PlaceOrderUseCase:
    """应用服务：协调领域对象，不包含业务逻辑"""
    
    def __init__(self, order_repo: OrderRepository, event_bus: EventBus):
        self.order_repo = order_repo
        self.event_bus = event_bus
    
    def execute(self, customer_id: str, items: list) -> str:
        # 1. 调用领域逻辑
        order = Order.place(customer_id=customer_id, items=items)
        # 2. 持久化
        self.order_repo.save(order)
        # 3. 发布事件
        for event in order.pull_events():
            self.event_bus.publish(event)
        return order.id
```

### 各层职责

| 层 | 职责 | 示例 |
|---|------|------|
| **领域层** | 业务规则 + 不变量 | `Order.confirm()` |
| **应用层** | 协调流程，无业务逻辑 | `PlaceOrderUseCase` |
| **基础设施层** | 技术实现 | `PostgresOrderRepository` |

---

## 总结

领域事件和 Saga 让复杂业务流程可管理：
- **领域事件**：记录"发生了什么"，解耦上下文
- **Saga**：协调跨聚合的复杂流程，用补偿替代分布式事务
- **TDD**：为每个步骤和补偿逻辑写测试，保证 Saga 的可靠性

---

**下一章**：[Python 内部 DSL](../part4-dsl-design/13-internal-dsl-python.md)
