# 第21章：Saga模式

## 学习目标

- 理解Saga解决的分布式事务问题
- 掌握编排式Saga（Orchestration）的实现
- 掌握协同式Saga（Choreography）的实现
- 设计有效的补偿操作

---

## 21.1 分布式事务的困境

微服务中，跨服务的操作不能用数据库事务保证一致性：

```
电商下单流程（跨3个服务）：

step1: 创建订单      → 订单服务       → 写订单DB
step2: 扣减库存      → 库存服务       → 写库存DB
step3: 扣款          → 支付服务       → 写支付DB

如果step2成功，step3失败：
  ├── 订单已创建 ✓
  ├── 库存已扣减 ✓
  └── 扣款失败  ✗ ← 数据不一致！

不能用2PC（两阶段提交）：
  └── 性能差、锁定时间长、参与方必须都支持XA
```

**Saga模式的解决方案**：
> 将长事务拆分为一系列本地事务，每个本地事务完成后发布事件。如果某步失败，执行**补偿事务（Compensating Transaction）**来撤销前面的操作。

---

## 21.2 补偿操作设计

Saga的关键是为每个操作设计对应的**补偿操作**：

```
正向操作（Forward）          补偿操作（Compensating）
──────────────────────────────────────────────────
创建订单                 →   取消订单
锁定库存                 →   释放库存
预授权支付               →   取消预授权
创建发货单               →   取消发货单
发送确认邮件             →   发送取消通知邮件（无法撤回，只能补偿）

注意：
  - 补偿操作不是回滚，而是语义上的"撤销"
  - 有些操作无法真正撤销（如发送邮件），只能补偿
  - 补偿操作必须是幂等的（可能执行多次）
```

---

## 21.3 编排式Saga（Orchestration-based Saga）

**中央协调者**负责协调所有步骤和补偿：

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime

class SagaStatus(Enum):
    STARTED = "started"
    COMPLETED = "completed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    FAILED = "failed"

class SagaStep(Enum):
    CREATE_ORDER = "create_order"
    RESERVE_STOCK = "reserve_stock"
    PROCESS_PAYMENT = "process_payment"
    CONFIRM_ORDER = "confirm_order"

@dataclass
class SagaState:
    saga_id: str
    order_id: str
    customer_id: str
    items: list
    status: SagaStatus
    current_step: SagaStep
    failed_at_step: Optional[SagaStep] = None
    error: Optional[str] = None
    created_at: datetime = None


class PlaceOrderSagaOrchestrator:
    """
    编排式Saga：中央协调者控制流程
    
    流程：
      创建订单 → 锁定库存 → 处理支付 → 确认订单
    
    补偿（反向）：
      取消订单 ← 释放库存 ← 退款（如已扣款）
    """
    
    def __init__(
        self,
        order_service,
        inventory_service,
        payment_service,
        saga_store,
    ):
        self._order_svc = order_service
        self._inventory_svc = inventory_service
        self._payment_svc = payment_service
        self._store = saga_store
    
    def start(self, saga_id: str, customer_id: str, items: list) -> None:
        """启动Saga"""
        state = SagaState(
            saga_id=saga_id,
            order_id=str(uuid4()),
            customer_id=customer_id,
            items=items,
            status=SagaStatus.STARTED,
            current_step=SagaStep.CREATE_ORDER,
            created_at=datetime.now()
        )
        self._store.save(state)
        self._execute_step(state)
    
    def _execute_step(self, state: SagaState) -> None:
        """执行当前步骤"""
        try:
            if state.current_step == SagaStep.CREATE_ORDER:
                self._step_create_order(state)
            
            elif state.current_step == SagaStep.RESERVE_STOCK:
                self._step_reserve_stock(state)
            
            elif state.current_step == SagaStep.PROCESS_PAYMENT:
                self._step_process_payment(state)
            
            elif state.current_step == SagaStep.CONFIRM_ORDER:
                self._step_confirm_order(state)
        
        except Exception as e:
            self._handle_failure(state, e)
    
    def _step_create_order(self, state: SagaState) -> None:
        self._order_svc.create_order(state.order_id, state.customer_id, state.items)
        state.current_step = SagaStep.RESERVE_STOCK
        self._store.save(state)
        self._execute_step(state)
    
    def _step_reserve_stock(self, state: SagaState) -> None:
        self._inventory_svc.reserve_for_order(state.order_id, state.items)
        state.current_step = SagaStep.PROCESS_PAYMENT
        self._store.save(state)
        self._execute_step(state)
    
    def _step_process_payment(self, state: SagaState) -> None:
        self._payment_svc.charge_order(state.order_id, state.customer_id)
        state.current_step = SagaStep.CONFIRM_ORDER
        self._store.save(state)
        self._execute_step(state)
    
    def _step_confirm_order(self, state: SagaState) -> None:
        self._order_svc.confirm_order(state.order_id)
        state.status = SagaStatus.COMPLETED
        self._store.save(state)
        print(f"Saga {state.saga_id} 完成！订单 {state.order_id} 已确认")
    
    def _handle_failure(self, state: SagaState, error: Exception) -> None:
        """处理步骤失败，开始补偿"""
        state.status = SagaStatus.COMPENSATING
        state.failed_at_step = state.current_step
        state.error = str(error)
        self._store.save(state)
        
        print(f"步骤 {state.current_step} 失败：{error}，开始补偿...")
        self._compensate(state)
    
    def _compensate(self, state: SagaState) -> None:
        """执行补偿操作（反向执行已完成的步骤）"""
        failed_step = state.failed_at_step
        
        # 根据失败的步骤，确定需要补偿的范围
        # 补偿顺序：与执行顺序相反
        
        if failed_step == SagaStep.CONFIRM_ORDER:
            # 支付已处理，需要退款
            self._try_compensate(lambda: self._payment_svc.refund(state.order_id))
            self._try_compensate(lambda: self._inventory_svc.release(state.order_id))
            self._try_compensate(lambda: self._order_svc.cancel_order(state.order_id, "确认失败"))
        
        elif failed_step == SagaStep.PROCESS_PAYMENT:
            # 库存已锁定，需要释放
            self._try_compensate(lambda: self._inventory_svc.release(state.order_id))
            self._try_compensate(lambda: self._order_svc.cancel_order(state.order_id, "支付失败"))
        
        elif failed_step == SagaStep.RESERVE_STOCK:
            # 只有订单被创建，取消它
            self._try_compensate(lambda: self._order_svc.cancel_order(state.order_id, "库存不足"))
        
        state.status = SagaStatus.COMPENSATED
        self._store.save(state)
        print(f"Saga {state.saga_id} 补偿完成")
    
    def _try_compensate(self, action) -> None:
        """执行补偿，忽略错误（补偿必须尽力执行）"""
        try:
            action()
        except Exception as e:
            # 补偿失败：记录日志，需要人工介入
            print(f"补偿操作失败：{e}，需要人工处理！")
```

---

## 21.4 协同式Saga（Choreography-based Saga）

**无中央协调者**，每个服务响应事件并发布下一个事件：

```python
# 订单服务
class OrderService:
    def create_order(self, command: CreateOrderCommand) -> None:
        order = Order.create(command.customer_id, command.items)
        self._repo.save(order)
        # 发布事件，触发下一步
        self._events.publish(OrderCreated(
            order_id=str(order.id),
            items=command.items,
            customer_id=command.customer_id
        ))

# 库存服务（响应 OrderCreated）
class InventoryService:
    def on_order_created(self, event: OrderCreated) -> None:
        try:
            for item in event.items:
                inventory = self._repo.get(item["product_id"])
                inventory.reserve(item["quantity"])
                self._repo.save(inventory)
            
            # 成功：发布事件，触发下一步
            self._events.publish(StockReserved(order_id=event.order_id))
        
        except InsufficientStockError:
            # 失败：发布失败事件，触发补偿
            self._events.publish(StockReservationFailed(
                order_id=event.order_id,
                reason="库存不足"
            ))

# 支付服务（响应 StockReserved）
class PaymentService:
    def on_stock_reserved(self, event: StockReserved) -> None:
        try:
            payment = self._process_payment(event.order_id)
            self._events.publish(PaymentProcessed(order_id=event.order_id))
        except PaymentFailedException as e:
            self._events.publish(PaymentFailed(
                order_id=event.order_id,
                reason=str(e)
            ))

# 订单服务（响应成功）
class OrderService:
    def on_payment_processed(self, event: PaymentProcessed) -> None:
        order = self._repo.get(event.order_id)
        order.confirm()
        self._repo.save(order)

# 订单服务（响应失败，执行补偿）
class OrderService:
    def on_stock_reservation_failed(self, event: StockReservationFailed) -> None:
        order = self._repo.get(event.order_id)
        order.cancel(event.reason)
        self._repo.save(order)
    
    def on_payment_failed(self, event: PaymentFailed) -> None:
        order = self._repo.get(event.order_id)
        order.cancel(event.reason)
        self._repo.save(order)
        # 同时触发库存释放
        self._events.publish(OrderCancelled(
            order_id=event.order_id,
            reason=event.reason
        ))

# 库存服务（响应订单取消，释放库存）
class InventoryService:
    def on_order_cancelled(self, event: OrderCancelled) -> None:
        self._release_reserved_stock(event.order_id)
```

---

## 21.5 两种Saga的对比

```
编排式Saga（Orchestration）：
  优点：
    ✅ 流程清晰，集中在一个类中
    ✅ 易于监控和调试（查看Saga状态）
    ✅ 补偿逻辑集中，不分散
  缺点：
    ❌ 中央协调者可能成为单点故障
    ❌ 服务可能知道太多其他服务的信息（耦合）

协同式Saga（Choreography）：
  优点：
    ✅ 完全解耦，服务只知道事件，不知道彼此
    ✅ 没有单点故障
    ✅ 更符合微服务的精神（松耦合）
  缺点：
    ❌ 流程分散，难以追踪整体状态
    ❌ 容易出现循环事件
    ❌ 测试更复杂

选择建议：
  ├── 步骤少（3步以内）：协同式更简单
  ├── 步骤多、复杂：编排式更清晰
  └── 需要监控Saga状态：编排式
```

---

## 21.6 Saga的幂等性设计

Saga中的每个步骤可能被重试，必须设计为幂等：

```python
class InventoryService:
    def reserve_for_order(self, order_id: str, items: list) -> None:
        """幂等的库存预留"""
        
        # 检查是否已经预留（幂等键）
        if self._is_already_reserved(order_id):
            print(f"订单 {order_id} 的库存已预留，跳过")
            return
        
        # 执行预留
        for item in items:
            inventory = self._repo.get(item["product_id"])
            inventory.reserve(item["quantity"])
            self._repo.save(inventory)
        
        # 记录已处理
        self._mark_reserved(order_id)
    
    def _is_already_reserved(self, order_id: str) -> bool:
        return self._reservation_log.exists(order_id)
```

---

## 本章小结

| 方面 | 编排式 | 协同式 |
|------|--------|--------|
| 流程控制 | 中央协调者 | 分散到各服务 |
| 耦合度 | 协调者了解各服务 | 服务只知道事件 |
| 可见性 | 流程清晰，易调试 | 流程分散，难追踪 |
| 适用场景 | 复杂流程 | 简单流程 |

**设计Saga的黄金法则**：每个正向操作都必须有对应的补偿操作，且补偿操作必须是幂等的。

---

**上一章：** [第20章：DDD与微服务](./20-ddd-microservices.md)  
**下一章：** [第22章：防腐层](./22-anti-corruption-layer.md)
