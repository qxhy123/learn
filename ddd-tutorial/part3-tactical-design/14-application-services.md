# 第14章：应用服务（Application Service）

## 学习目标

- 理解应用服务作为"用例编排层"的定位
- 掌握应用服务的职责边界
- 学会设计命令/查询对象（Command/Query）
- 理解应用服务与事务管理的关系

---

## 14.1 应用服务的角色

应用服务是**用例的入口**，它：
1. 接收外部请求（HTTP、消息、CLI）
2. 加载领域对象
3. 调用领域逻辑
4. 持久化变更
5. 发布领域事件
6. 返回结果

```
外部世界（REST API / 消息队列 / CLI）
           │
           ▼
    ┌─────────────────┐
    │   应用服务       │  ← 用例的编排层
    │                 │  ← 没有业务逻辑！
    │  1. 加载聚合    │
    │  2. 调用领域方法 │
    │  3. 保存聚合    │
    │  4. 发布事件    │
    └─────────────────┘
           │
           ▼
    ┌─────────────────┐
    │   领域层         │  ← 业务规则在这里
    │ (实体/值对象/聚合)│
    └─────────────────┘
           │
           ▼
    ┌─────────────────┐
    │  基础设施层      │  ← 持久化/消息队列
    └─────────────────┘
```

---

## 14.2 命令与查询分离

应用服务的方法通常分为两类：

**命令（Command）**：改变系统状态，无返回值（或只返回ID）
**查询（Query）**：读取数据，不改变状态

```python
# 命令：改变状态
class PlaceOrderCommand:
    customer_id: str
    items: list[dict]
    shipping_address: dict

class CancelOrderCommand:
    order_id: str
    reason: str

# 查询：读取数据
class GetOrderQuery:
    order_id: str

class ListCustomerOrdersQuery:
    customer_id: str
    status: Optional[str] = None
    page: int = 1
    page_size: int = 20
```

**为什么分离命令和查询？**

```python
# ❌ 混合：方法又查又改，难以理解
def process_order(order_id: str) -> dict:
    order = repo.get(order_id)
    order.confirm()              # 改变状态
    repo.save(order)
    return {"order": order.to_dict(), "status": "confirmed"}  # 也返回数据

# ✅ 分离：命令只改变状态，查询只读数据
def confirm_order(command: ConfirmOrderCommand) -> None:
    order = repo.get(OrderId(command.order_id))
    order.confirm()
    repo.save(order)
    # 只改变状态，不返回复杂数据

def get_order(query: GetOrderQuery) -> OrderDTO:
    return order_query_service.get(query.order_id)  # 只读
```

---

## 14.3 应用服务的完整实现

```python
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime

# ===== 命令对象 =====

@dataclass(frozen=True)
class PlaceOrderCommand:
    customer_id: str
    items: tuple  # [(product_id, product_name, unit_price, quantity), ...]
    shipping_address: dict

@dataclass(frozen=True)
class AddOrderItemCommand:
    order_id: str
    product_id: str
    product_name: str
    unit_price: Decimal
    quantity: int

@dataclass(frozen=True)
class CancelOrderCommand:
    order_id: str
    reason: str

# ===== 查询结果对象（DTO）=====

@dataclass
class OrderItemDTO:
    product_id: str
    product_name: str
    unit_price: Decimal
    quantity: int
    subtotal: Decimal

@dataclass
class OrderDTO:
    order_id: str
    customer_id: str
    status: str
    items: List[OrderItemDTO]
    total: Decimal
    currency: str
    created_at: datetime

# ===== 应用服务 =====

class OrderApplicationService:
    """订单应用服务
    
    职责：
    - 编排用例（不包含业务规则！）
    - 管理事务边界
    - 协调领域层和基础设施层
    """
    
    def __init__(
        self,
        order_repo: OrderRepository,
        product_catalog: ProductCatalogService,  # 外部服务（防腐层）
        pricing_service: OrderPricingService,    # 领域服务
        event_bus: EventBus,
    ):
        self._order_repo = order_repo
        self._catalog = product_catalog
        self._pricing = pricing_service
        self._event_bus = event_bus
    
    # ===== 命令方法 =====
    
    def place_order(self, command: PlaceOrderCommand) -> str:
        """用例：下单
        
        返回：新创建的订单ID
        """
        # 1. 创建聚合
        order = Order(OrderId(), command.customer_id)
        
        # 2. 添加订单项（通过领域方法，保护不变量）
        for item_data in command.items:
            product = self._catalog.get_product_snapshot(item_data["product_id"])
            order.add_item(product, item_data["quantity"])
        
        # 3. 下单
        order.place()
        
        # 4. 持久化
        self._order_repo.save(order)
        
        # 5. 发布领域事件
        self._publish_events(order)
        
        return str(order.id)
    
    def cancel_order(self, command: CancelOrderCommand) -> None:
        """用例：取消订单"""
        # 1. 加载聚合
        order = self._order_repo.get(OrderId(command.order_id))
        
        # 2. 调用领域方法（业务规则在领域对象中）
        order.cancel(command.reason)
        
        # 3. 持久化
        self._order_repo.save(order)
        
        # 4. 发布事件
        self._publish_events(order)
    
    def confirm_payment(self, command: ConfirmPaymentCommand) -> None:
        """用例：确认支付"""
        order = self._order_repo.get(OrderId(command.order_id))
        
        payment = Payment(
            payment_id=command.payment_id,
            amount=Money(command.amount, command.currency)
        )
        order.mark_paid(payment)
        
        self._order_repo.save(order)
        self._publish_events(order)
    
    # ===== 查询方法 =====
    
    def get_order(self, query: GetOrderQuery) -> OrderDTO:
        """查询：获取订单详情"""
        order = self._order_repo.get(OrderId(query.order_id))
        return self._to_dto(order)
    
    def list_customer_orders(self, query: ListCustomerOrdersQuery) -> List[OrderDTO]:
        """查询：获取客户订单列表"""
        orders = self._order_repo.find_by_customer(query.customer_id)
        if query.status:
            orders = [o for o in orders if o.status.value == query.status]
        return [self._to_dto(o) for o in orders]
    
    # ===== 私有方法 =====
    
    def _publish_events(self, order: Order) -> None:
        for event in order.collect_events():
            self._event_bus.publish(event)
    
    def _to_dto(self, order: Order) -> OrderDTO:
        """领域对象 → DTO（用于返回给调用方）"""
        # 注意：这里的转换不包含业务逻辑，只是数据映射
        ...
```

---

## 14.4 事务管理

应用服务是管理事务边界的正确位置：

```python
# 方式1：装饰器事务（推荐，清晰）
class OrderApplicationService:
    
    @transactional  # 整个方法在一个事务中
    def place_order(self, command: PlaceOrderCommand) -> str:
        order = Order(OrderId(), command.customer_id)
        # ... 业务逻辑 ...
        self._order_repo.save(order)
        # 事务在方法结束时提交
        # 异常时自动回滚
        return str(order.id)

# 方式2：显式事务（更灵活）
class OrderApplicationService:
    
    def place_order(self, command: PlaceOrderCommand) -> str:
        with self._unit_of_work as uow:
            order = Order(OrderId(), command.customer_id)
            # ... 业务逻辑 ...
            uow.orders.save(order)
            uow.commit()
            
            # 事务提交后再发布事件
            self._publish_events(order)
        
        return str(order.id)
```

### 事务与事件发布的顺序

```python
# ✅ 正确顺序：先提交事务，再发布事件
def place_order(self, command: PlaceOrderCommand) -> str:
    order = Order(...)
    order.place()
    
    # 1. 先持久化（事务）
    self._order_repo.save(order)
    # ↑ 如果这里失败，事件不会发布，数据一致
    
    # 2. 再发布事件（事务提交后）
    events = order.collect_events()
    for event in events:
        self._event_bus.publish(event)
    # ↑ 如果这里失败，订单已保存，用发件箱模式补偿
    
    return str(order.id)

# ❌ 错误顺序：先发布事件，再提交事务
def place_order_wrong(self, command):
    order = Order(...)
    order.place()
    
    # 先发布事件
    for event in order.collect_events():
        self._event_bus.publish(event)  # 事件已发出！
    
    # 再保存（如果这里失败，事件已发出但订单未保存！数据不一致）
    self._order_repo.save(order)
```

---

## 14.5 应用服务应该避免的事情

```python
# ❌ 在应用服务中写业务逻辑
class OrderApplicationService:
    def place_order(self, command: PlaceOrderCommand) -> str:
        order = Order(OrderId(), command.customer_id)
        
        # ❌ 这些是业务规则，应该在领域层
        if len(command.items) > 50:
            raise ValueError("订单商品不能超过50种")
        
        total = sum(item["price"] * item["qty"] for item in command.items)
        if total > 100000:
            raise ValueError("单笔订单不能超过10万元")
        
        # ❌ 手动计算总价（应该在Order聚合中）
        order._total = total
        ...

# ✅ 应用服务只编排，不包含业务规则
class OrderApplicationService:
    def place_order(self, command: PlaceOrderCommand) -> str:
        order = Order(OrderId(), command.customer_id)
        
        for item_data in command.items:
            product = self._catalog.get_product_snapshot(item_data["product_id"])
            order.add_item(product, item_data["quantity"])
            # ↑ add_item 内部有"不超过50种"的规则
        
        order.place()
        # ↑ place 内部有总价验证等规则
        
        self._order_repo.save(order)
        self._publish_events(order)
        return str(order.id)
```

---

## 14.6 与接口层的配合

```python
# REST API 接口层使用应用服务
from fastapi import APIRouter, Depends, HTTPException

router = APIRouter()

class PlaceOrderRequest(BaseModel):
    items: List[dict]
    shipping_address: dict

@router.post("/orders", status_code=201)
async def place_order(
    request: PlaceOrderRequest,
    current_user: User = Depends(get_current_user),
    order_service: OrderApplicationService = Depends(get_order_service)
) -> dict:
    try:
        command = PlaceOrderCommand(
            customer_id=current_user.id,
            items=tuple(request.items),
            shipping_address=request.shipping_address
        )
        order_id = order_service.place_order(command)
        return {"order_id": order_id}
    
    except OrderException as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="服务器内部错误")

# 接口层只负责：
# 1. HTTP协议转换（请求/响应格式）
# 2. 认证/授权（谁可以调用）
# 3. 错误到HTTP状态码的映射
# 不包含任何业务逻辑！
```

---

## 本章小结

| 要点 | 内容 |
|------|------|
| 核心职责 | 编排用例，无业务规则 |
| 命令/查询 | 分离变更和读取 |
| 事务边界 | 应用服务管理事务 |
| 事件发布 | 事务提交后再发布 |

**黄金原则**：如果你在应用服务里写了if/else判断业务条件，那这些逻辑应该下沉到领域层。

---

**上一章：** [第13章：领域服务](./13-domain-services.md)  
**下一章：** [第15章：工厂](./15-factories.md)
