# 第17章：六边形架构（Hexagonal Architecture）

## 学习目标

- 理解六边形架构（端口与适配器）的核心思想
- 掌握端口（Port）和适配器（Adapter）的概念
- 理解六边形架构如何实现极致的可测试性
- 对比六边形架构与分层架构的异同

---

## 17.1 六边形架构的核心思想

六边形架构由 Alistair Cockburn 提出，也称为**端口与适配器架构（Ports and Adapters）**。

核心思想：
> **让应用程序核心（领域 + 应用逻辑）完全独立于外部世界（UI、数据库、消息队列、第三方API），通过定义良好的接口（端口）与外界交互。**

```
传统分层架构：
  UI → 业务逻辑 → 数据库
  方向固定，UI和数据库依然耦合到业务

六边形架构：
         ┌───────────────────────────┐
  HTTP   │                           │   数据库
  ├──►   │  应用核心                 │  ◄───┤
  CLI    │  (领域 + 应用逻辑)        │       ORM
  ├──►   │                           │  ◄───┤
  测试   │  只知道"端口"，            │       内存存储
  └──►   │  不知道适配器              │  ◄───┘
         └───────────────────────────┘
              消息队列    第三方API

应用核心不依赖任何具体的外部技术！
可以用任何"适配器"连接到核心。
```

---

## 17.2 端口（Port）

**端口**是应用核心定义的**接口**，描述了"我需要什么"或"我提供什么"。

端口分为两种：

### 驱动端口（Driving Port / Primary Port）

应用核心**对外提供的功能**，被外界（UI、测试、消息消费者）调用。

```python
# 驱动端口（Primary Port）：应用提供的用例接口
from abc import ABC, abstractmethod

class OrderManagementPort(ABC):
    """订单管理端口——应用核心提供的功能"""
    
    @abstractmethod
    def place_order(self, command: PlaceOrderCommand) -> str: ...
    
    @abstractmethod
    def cancel_order(self, command: CancelOrderCommand) -> None: ...
    
    @abstractmethod
    def get_order(self, query: GetOrderQuery) -> OrderDTO: ...
```

### 被驱动端口（Driven Port / Secondary Port）

应用核心**需要外界提供的能力**，由基础设施适配器实现。

```python
# 被驱动端口（Secondary Port）：应用核心需要的依赖

class OrderStoragePort(ABC):
    """订单存储端口——应用核心需要存储能力"""
    @abstractmethod
    def save(self, order: Order) -> None: ...
    @abstractmethod
    def get(self, order_id: OrderId) -> Order: ...

class PaymentGatewayPort(ABC):
    """支付网关端口——应用核心需要支付能力"""
    @abstractmethod
    def charge(self, amount: Money, payment_method: str) -> PaymentResult: ...

class NotificationPort(ABC):
    """通知端口——应用核心需要发通知的能力"""
    @abstractmethod
    def send_order_confirmation(self, order: Order, email: str) -> None: ...
```

---

## 17.3 适配器（Adapter）

**适配器**是实现端口的具体技术代码，它将外部技术"适配"到端口接口。

### 驱动适配器（Driving Adapter）

将外部请求适配成对端口的调用：

```python
# 驱动适配器1：REST API适配器
class FastAPIOrderAdapter:
    """将HTTP请求适配为对应用核心的调用"""
    
    def __init__(self, order_port: OrderManagementPort):
        self._port = order_port
    
    def register_routes(self, app):
        @app.post("/orders")
        async def place_order(request: PlaceOrderHTTPRequest):
            command = PlaceOrderCommand(
                customer_id=request.customer_id,
                items=request.items,
            )
            order_id = self._port.place_order(command)
            return {"order_id": order_id}

# 驱动适配器2：CLI适配器
class CLIOrderAdapter:
    """将命令行输入适配为对应用核心的调用"""
    
    def __init__(self, order_port: OrderManagementPort):
        self._port = order_port
    
    def run(self):
        customer_id = input("客户ID: ")
        # ...解析输入...
        command = PlaceOrderCommand(customer_id=customer_id, items=[...])
        order_id = self._port.place_order(command)
        print(f"订单已创建：{order_id}")

# 驱动适配器3：消息队列消费者适配器
class KafkaOrderAdapter:
    """将Kafka消息适配为对应用核心的调用"""
    
    def __init__(self, order_port: OrderManagementPort):
        self._port = order_port
    
    def on_message(self, kafka_message: dict):
        command = PlaceOrderCommand(**kafka_message["data"])
        self._port.place_order(command)
```

### 被驱动适配器（Driven Adapter）

将端口接口适配到具体技术实现：

```python
# 被驱动适配器1：PostgreSQL适配器（实现存储端口）
class PostgreSQLOrderAdapter(OrderStoragePort):
    def save(self, order: Order) -> None:
        # 用 SQLAlchemy 实现
        ...

# 被驱动适配器2：MongoDB适配器（同样实现存储端口）
class MongoDBOrderAdapter(OrderStoragePort):
    def save(self, order: Order) -> None:
        # 用 pymongo 实现
        ...

# 被驱动适配器3：内存适配器（用于测试！）
class InMemoryOrderAdapter(OrderStoragePort):
    def __init__(self):
        self._store: Dict[str, Order] = {}
    
    def save(self, order: Order) -> None:
        self._store[str(order.id)] = order
    
    def get(self, order_id: OrderId) -> Order:
        return self._store[str(order_id)]

# 被驱动适配器4：支付宝支付适配器
class AlipayAdapter(PaymentGatewayPort):
    def charge(self, amount: Money, payment_method: str) -> PaymentResult:
        # 调用支付宝API
        ...

# 被驱动适配器5：模拟支付适配器（用于测试！）
class MockPaymentAdapter(PaymentGatewayPort):
    def __init__(self, should_succeed: bool = True):
        self._should_succeed = should_succeed
    
    def charge(self, amount: Money, payment_method: str) -> PaymentResult:
        if self._should_succeed:
            return PaymentResult.success(f"mock-payment-{uuid4()}")
        else:
            return PaymentResult.failed("模拟支付失败")
```

---

## 17.4 应用核心的实现

应用核心实现驱动端口，并依赖被驱动端口：

```python
class OrderApplicationService(OrderManagementPort):
    """应用核心：实现驱动端口，依赖被驱动端口"""
    
    def __init__(
        self,
        # 依赖被驱动端口（接口），不依赖具体适配器
        storage: OrderStoragePort,
        payment: PaymentGatewayPort,
        notification: NotificationPort,
    ):
        self._storage = storage
        self._payment = payment
        self._notification = notification
    
    def place_order(self, command: PlaceOrderCommand) -> str:
        order = Order(OrderId(), command.customer_id)
        for item in command.items:
            order.add_item(item["product_snapshot"], item["quantity"])
        order.place()
        
        self._storage.save(order)      # 通过端口
        self._notification.send_order_confirmation(order, command.customer_email)
        
        return str(order.id)
    
    def cancel_order(self, command: CancelOrderCommand) -> None:
        order = self._storage.get(OrderId(command.order_id))
        order.cancel(command.reason)
        self._storage.save(order)
    
    def get_order(self, query: GetOrderQuery) -> OrderDTO:
        order = self._storage.get(OrderId(query.order_id))
        return self._to_dto(order)
```

---

## 17.5 六边形架构的极致可测试性

这是六边形架构最大的优势——可以在没有任何基础设施的情况下测试所有应用逻辑：

```python
class TestOrderApplicationService:
    """纯单元测试：不需要数据库、不需要支付网关、不需要邮件服务"""
    
    def setup_method(self):
        # 全部使用测试适配器（内存/模拟）
        self.storage = InMemoryOrderAdapter()
        self.payment = MockPaymentAdapter(should_succeed=True)
        self.notification = MockNotificationAdapter()
        
        self.service = OrderApplicationService(
            storage=self.storage,
            payment=self.payment,
            notification=self.notification,
        )
    
    def test_place_order_successfully(self):
        command = PlaceOrderCommand(
            customer_id="cust-001",
            customer_email="test@example.com",
            items=[{
                "product_snapshot": ProductSnapshot("p1", "iPhone", Money(Decimal("7999"))),
                "quantity": 1
            }]
        )
        
        order_id = self.service.place_order(command)
        
        # 验证结果
        assert order_id is not None
        saved_order = self.storage.get(OrderId(order_id))
        assert saved_order.status == OrderStatus.PLACED
        assert self.notification.was_notified(order_id)  # 验证通知被发送
    
    def test_cancel_order_updates_storage(self):
        # 先创建一个待支付订单
        order_id = self._create_placed_order()
        
        # 取消
        self.service.cancel_order(CancelOrderCommand(order_id, "changed mind"))
        
        # 验证
        order = self.storage.get(OrderId(order_id))
        assert order.status == OrderStatus.CANCELLED
    
    def test_payment_failure_rolls_back(self):
        # 模拟支付失败
        self.payment = MockPaymentAdapter(should_succeed=False)
        self.service = OrderApplicationService(
            storage=self.storage,
            payment=self.payment,
            notification=self.notification,
        )
        
        with pytest.raises(PaymentFailedException):
            self.service.process_payment(...)
```

---

## 17.6 组合根（Composition Root）

所有的端口和适配器在**组合根**中装配在一起：

```python
# main.py 或 container.py：组合根
def create_application(config: Config) -> OrderManagementPort:
    """根据配置创建完整的应用，装配所有适配器"""
    
    if config.env == "test":
        # 测试环境：使用内存/模拟适配器
        storage = InMemoryOrderAdapter()
        payment = MockPaymentAdapter()
        notification = MockNotificationAdapter()
    else:
        # 生产环境：使用真实适配器
        db_session = create_database_session(config.database_url)
        storage = PostgreSQLOrderAdapter(db_session)
        payment = AlipayAdapter(config.alipay_config)
        notification = EmailNotificationAdapter(config.smtp_config)
    
    # 创建应用核心
    app_service = OrderApplicationService(
        storage=storage,
        payment=payment,
        notification=notification,
    )
    
    return app_service

# 在应用启动时
app_core = create_application(config)
http_adapter = FastAPIOrderAdapter(app_core)
http_adapter.register_routes(fastapi_app)
```

---

## 17.7 六边形 vs 分层架构

```
分层架构：
  ├── 层次化组织，上层调用下层
  ├── 依赖方向单一（上→下）
  └── 对"外部"（UI、DB）的位置有隐含假设

六边形架构：
  ├── 应用核心居中，外界围绕核心
  ├── 通过端口连接外界，对称的（左右没有高低）
  └── 更强调可替换性和可测试性

实践中：
  两者经常结合使用
  在限界上下文内部，用六边形架构组织代码
  在上下文之间，用上下文映射定义关系
```

---

## 本章小结

| 概念 | 说明 |
|------|------|
| 驱动端口 | 应用核心提供的功能接口（用例） |
| 被驱动端口 | 应用核心需要的外部能力接口 |
| 驱动适配器 | 将外部请求转为端口调用（HTTP/CLI/消息） |
| 被驱动适配器 | 将端口接口实现为具体技术（DB/API） |
| 核心优势 | 应用核心完全独立，可以纯单元测试 |

---

**上一章：** [第16章：分层架构](./16-layered-architecture.md)  
**下一章：** [第18章：CQRS](./18-cqrs.md)
