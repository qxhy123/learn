# 第24章：综合案例——电商系统完整DDD设计

## 学习目标

- 综合运用全部DDD知识设计一个完整系统
- 理解DDD各层次如何协同工作
- 感受从业务理解到代码设计的完整过程
- 掌握在真实项目中应用DDD的方法

---

## 24.1 案例背景

**系统**：一个中型B2C电商平台"EcoShop"

**业务范围**：
- 用户注册/登录
- 商品浏览和搜索
- 购物车和下单
- 支付（微信/支付宝）
- 物流跟踪
- 退换货

**技术团队**：约20人，分为5个小组

---

## 24.2 战略设计：子域与限界上下文

### 步骤一：子域识别

经过事件风暴工作坊（2天），识别出以下子域：

```
EcoShop 业务领域
│
├── 核心域（Core Domain）⭐
│   ├── 个性化推荐引擎  ← 差异化竞争力，用户留存关键
│   └── 动态定价系统    ← 利润优化，竞争优势
│
├── 支撑域（Supporting Subdomain）🔧
│   ├── 订单管理         ← 复杂状态机，业务特定规则
│   ├── 商品目录         ← 有业务规则，但不是差异化
│   ├── 库存管理         ← 有业务规则（锁定/释放/补货）
│   ├── 售后管理         ← 退换货规则较复杂
│   └── 用户会员体系     ← 积分、等级、特权
│
└── 通用域（Generic Subdomain）📦
    ├── 用户认证         → 使用 Keycloak
    ├── 支付             → 接入微信/支付宝
    ├── 物流跟踪         → 对接快递100 API
    ├── 消息通知         → 使用阿里云短信/邮件
    └── 搜索             → 使用 Elasticsearch
```

### 步骤二：限界上下文划分

```
┌─────────────────────────────────────────────────────────────────────┐
│                          EcoShop 平台                                │
│                                                                     │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐           │
│  │  推荐上下文   │  │  定价上下文   │  │  商品目录上下文│           │
│  │  Recommend   │  │   Pricing     │  │    Catalog    │           │
│  └───────────────┘  └───────────────┘  └───────────────┘           │
│                                                                     │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐           │
│  │  购物车上下文  │  │  订单上下文   │  │  库存上下文   │           │
│  │     Cart     │  │   Ordering    │  │  Inventory    │           │
│  └───────────────┘  └───────────────┘  └───────────────┘           │
│                                                                     │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐           │
│  │  支付上下文   │  │  物流上下文   │  │  售后上下文   │           │
│  │   Payment    │  │   Shipping    │  │  AfterSale    │           │
│  └───────────────┘  └───────────────┘  └───────────────┘           │
│                                                                     │
│  ┌───────────────┐                                                  │
│  │  会员上下文   │                                                  │
│  │  Membership  │                                                  │
│  └───────────────┘                                                  │
└─────────────────────────────────────────────────────────────────────┘
```

### 步骤三：上下文映射

```
                    OHS/PL（开放主机）
┌──────────┐ ──────────────────────► ┌──────────┐
│  定价    │                         │  订单    │  C-S（客户-供应商）
└──────────┘ ◄────── 价格查询 ───────  └────┬─────┘
                                           │
┌──────────┐  伙伴（Partnership）           │ 事件驱动
│  库存    │ ◄──────────────────────►      │
└──────────┘                               ▼
                              ┌──────────────────┐
┌──────────┐  ACL（防腐层）   │    支付上下文     │
│第三方支付 │ ──────────────► │   Payment ACL    │
└──────────┘                  └──────────────────┘
                                         │ 支付完成事件
                               ┌─────────┘
                               ▼
                       ┌──────────────┐
                       │  物流上下文   │
                       └──────────────┘
```

---

## 24.3 订单上下文的战术设计

以订单上下文为例，展示完整的战术设计：

### 聚合设计

```python
# ===== 值对象 =====

@dataclass(frozen=True)
class OrderId:
    value: UUID = field(default_factory=uuid4)
    def __str__(self): return str(self.value)

@dataclass(frozen=True)
class Money:
    amount: Decimal
    currency: str = "CNY"
    def __add__(self, other: "Money") -> "Money":
        assert self.currency == other.currency
        return Money(self.amount + other.amount, self.currency)
    def __mul__(self, factor: Decimal) -> "Money":
        return Money((self.amount * factor).quantize(Decimal("0.01")), self.currency)

@dataclass(frozen=True)
class ShippingAddress:
    """收货地址快照（值对象，防止地址变更影响历史订单）"""
    recipient_name: str
    phone: str
    province: str
    city: str
    district: str
    detail: str
    postal_code: str

@dataclass(frozen=True)
class ProductSnapshot:
    """商品快照（值对象，防止商品信息变更影响历史订单）"""
    product_id: str
    name: str
    unit_price: Money
    image_url: str = ""


# ===== 聚合内部实体 =====

class OrderItem:
    def __init__(self, item_id: str, product: ProductSnapshot, quantity: int):
        if quantity <= 0:
            raise ValueError(f"商品数量必须大于0，收到：{quantity}")
        self._id = item_id
        self._product = product
        self._quantity = quantity
    
    @property
    def subtotal(self) -> Money:
        return self._product.unit_price * Decimal(self._quantity)
    
    @property
    def product_id(self) -> str: return self._product.product_id
    
    @property
    def quantity(self) -> int: return self._quantity
    
    def increase_quantity(self, additional: int) -> None:
        if additional <= 0: raise ValueError("增加数量必须大于0")
        self._quantity += additional


# ===== 状态机 =====

class OrderStatus(Enum):
    PENDING_PAYMENT = "pending_payment"   # 待付款
    PAID = "paid"                          # 已付款
    PREPARING = "preparing"               # 备货中
    SHIPPED = "shipped"                   # 已发货
    DELIVERED = "delivered"               # 已签收
    CANCELLED = "cancelled"               # 已取消
    REFUNDING = "refunding"               # 退款中
    REFUNDED = "refunded"                 # 已退款

VALID_TRANSITIONS = {
    OrderStatus.PENDING_PAYMENT: {OrderStatus.PAID, OrderStatus.CANCELLED},
    OrderStatus.PAID: {OrderStatus.PREPARING, OrderStatus.REFUNDING},
    OrderStatus.PREPARING: {OrderStatus.SHIPPED},
    OrderStatus.SHIPPED: {OrderStatus.DELIVERED},
    OrderStatus.DELIVERED: {OrderStatus.REFUNDING},
    OrderStatus.REFUNDING: {OrderStatus.REFUNDED},
    OrderStatus.CANCELLED: set(),
    OrderStatus.REFUNDED: set(),
}


# ===== 聚合根 =====

class Order:
    """订单聚合根
    
    统一语言词汇：
    - 下单（place）：用户确认购买，等待付款
    - 付款（pay）：支付成功，进入备货
    - 发货（ship）：仓库发出商品
    - 签收（deliver）：用户收到商品
    - 取消（cancel）：取消订单
    - 申请退款（request_refund）：用户申请退款
    """
    
    MAX_ITEMS = 50
    
    def __init__(self, order_id: OrderId, customer_id: str, address: ShippingAddress):
        self._id = order_id
        self._customer_id = customer_id
        self._address = address
        self._items: List[OrderItem] = []
        self._status = OrderStatus.PENDING_PAYMENT
        self._created_at = datetime.now()
        self._events: List[DomainEvent] = []
    
    # --- 修改命令 ---
    
    def add_item(self, product: ProductSnapshot, quantity: int) -> None:
        """添加商品（仅在下单前可修改）"""
        # 业务规则：待付款状态才可以加商品（如购物车转订单时）
        if self._status != OrderStatus.PENDING_PAYMENT:
            raise OrderException("待付款状态才能修改订单商品")
        
        if len(self._items) >= self.MAX_ITEMS:
            raise OrderException(f"订单商品种数不能超过{self.MAX_ITEMS}种")
        
        existing = self._find_item(product.product_id)
        if existing:
            existing.increase_quantity(quantity)
        else:
            item_id = f"{self._id}-{len(self._items)+1:03d}"
            self._items.append(OrderItem(item_id, product, quantity))
    
    def pay(self, payment_id: str, paid_amount: Money) -> None:
        """确认支付"""
        self._transition_to(OrderStatus.PAID)
        
        if paid_amount.amount < self.total.amount:
            raise OrderException(
                f"支付金额 {paid_amount} 不足，应付 {self.total}"
            )
        
        self._payment_id = payment_id
        self._paid_at = datetime.now()
        self._emit(OrderPaid(
            order_id=str(self._id),
            payment_id=payment_id,
            amount=paid_amount,
            occurred_at=self._paid_at
        ))
    
    def start_preparing(self) -> None:
        """开始备货"""
        self._transition_to(OrderStatus.PREPARING)
        self._emit(OrderPreparationStarted(str(self._id)))
    
    def ship(self, tracking_number: str, carrier: str) -> None:
        """发货"""
        self._transition_to(OrderStatus.SHIPPED)
        self._tracking_number = tracking_number
        self._carrier = carrier
        self._shipped_at = datetime.now()
        self._emit(OrderShipped(
            order_id=str(self._id),
            tracking_number=tracking_number,
            carrier=carrier,
            occurred_at=self._shipped_at
        ))
    
    def mark_delivered(self) -> None:
        """确认签收"""
        self._transition_to(OrderStatus.DELIVERED)
        self._delivered_at = datetime.now()
        self._emit(OrderDelivered(str(self._id), self._delivered_at))
    
    def cancel(self, reason: str) -> None:
        """取消订单"""
        self._transition_to(OrderStatus.CANCELLED)
        self._cancel_reason = reason
        self._cancelled_at = datetime.now()
        self._emit(OrderCancelled(
            order_id=str(self._id),
            reason=reason,
            occurred_at=self._cancelled_at
        ))
    
    def request_refund(self, reason: str) -> None:
        """申请退款"""
        self._transition_to(OrderStatus.REFUNDING)
        self._refund_reason = reason
        self._refund_requested_at = datetime.now()
        self._emit(RefundRequested(
            order_id=str(self._id),
            reason=reason,
            amount=self.total,
            occurred_at=self._refund_requested_at
        ))
    
    # --- 查询方法 ---
    
    @property
    def id(self) -> OrderId: return self._id
    @property
    def status(self) -> OrderStatus: return self._status
    @property
    def customer_id(self) -> str: return self._customer_id
    
    @property
    def total(self) -> Money:
        if not self._items:
            return Money(Decimal("0"))
        result = self._items[0].subtotal
        for item in self._items[1:]:
            result = result + item.subtotal
        return result
    
    def collect_events(self) -> List[DomainEvent]:
        events = list(self._events)
        self._events.clear()
        return events
    
    # --- 私有方法 ---
    
    def _transition_to(self, new_status: OrderStatus) -> None:
        valid_targets = VALID_TRANSITIONS.get(self._status, set())
        if new_status not in valid_targets:
            raise OrderException(
                f"订单状态 '{self._status.value}' 不能转换为 '{new_status.value}'"
            )
        self._status = new_status
    
    def _find_item(self, product_id: str) -> Optional[OrderItem]:
        return next((i for i in self._items if i.product_id == product_id), None)
    
    def _emit(self, event: DomainEvent) -> None:
        self._events.append(event)
```

---

## 24.4 跨上下文的关键流程

### 下单 → 支付 → 库存 → 物流的完整流程

```python
# 使用协同式 Saga 实现跨上下文流程

# 1. 订单服务：用户下单
class OrderApplicationService:
    def place_order(self, cmd: PlaceOrderCommand) -> str:
        # 从定价服务获取最新价格（同步调用）
        priced_items = [
            self._pricing_client.get_snapshot(item["product_id"])
            for item in cmd.items
        ]
        
        order = Order(
            order_id=OrderId(),
            customer_id=cmd.customer_id,
            address=ShippingAddress(**cmd.shipping_address)
        )
        for snapshot, qty in zip(priced_items, [i["qty"] for i in cmd.items]):
            order.add_item(snapshot, qty)
        
        self._order_repo.save(order)
        
        for event in order.collect_events():
            self._event_bus.publish(event)
        
        return str(order.id)

# 2. 库存服务：监听订单创建，预占库存
class InventoryEventHandler:
    @handler(OrderPaid)
    def on_order_paid(self, event: OrderPaid) -> None:
        """支付成功后锁定库存（幂等）"""
        if self._already_reserved(event.order_id):
            return
        try:
            for item in event.items:
                inv = self._repo.get(item["product_id"])
                inv.reserve(item["quantity"])
                self._repo.save(inv)
            self._event_bus.publish(StockReserved(event.order_id))
        except InsufficientStockError:
            self._event_bus.publish(StockReservationFailed(event.order_id))

# 3. 订单服务：监听库存事件
class OrderEventHandler:
    @handler(StockReserved)
    def on_stock_reserved(self, event: StockReserved) -> None:
        order = self._repo.get(event.order_id)
        order.start_preparing()
        self._repo.save(order)
    
    @handler(StockReservationFailed)
    def on_stock_failed(self, event: StockReservationFailed) -> None:
        order = self._repo.get(event.order_id)
        order.request_refund("库存不足，系统自动退款")
        self._repo.save(order)

# 4. 物流服务：监听备货完成，创建物流单
class ShippingEventHandler:
    @handler(OrderPreparationStarted)
    def on_preparation_started(self, event: OrderPreparationStarted) -> None:
        shipment = Shipment.create(event.order_id, event.shipping_address)
        self._repo.save(shipment)
        self._logistics_client.create_waybill(shipment)
```

---

## 24.5 架构总览

```
                        ┌─────────────────┐
                        │   API Gateway   │
                        │  (认证/限流/路由)│
                        └────────┬────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        ▼                        ▼
┌──────────────┐      ┌──────────────────┐      ┌──────────────┐
│  Catalog     │      │  Ordering        │      │  Inventory   │
│  Service     │      │  Service         │      │  Service     │
│  (商品目录)   │      │  (订单)           │      │  (库存)       │
│  PostgreSQL  │      │  PostgreSQL      │      │  PostgreSQL  │
└──────────────┘      └────────┬─────────┘      └──────────────┘
                               │ 事件
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     ▼
          ┌──────────────┐      ┌──────────────┐
          │  Payment     │      │  Shipping    │
          │  Service     │      │  Service     │
          │  (支付)       │      │  (物流)       │
          │  PostgreSQL  │      │  PostgreSQL  │
          └──────────────┘      └──────────────┘
                    
          Kafka (事件总线，连接所有服务)
          Elasticsearch (搜索上下文)
          Redis (缓存，定价结果、库存缓存)
```

---

## 24.6 关键设计决策总结

| 决策 | 选择 | 原因 |
|------|------|------|
| 架构 | 模块化单体 → 渐进式微服务 | 先验证边界，再拆分 |
| 核心域 | 推荐算法、动态定价 | 自研精做，保持竞争优势 |
| 通用域 | Auth0、阿里云短信 | 购买，节省资源给核心 |
| 聚合大小 | Order包含OrderItem，不包含Payment | 不变量最小化 |
| 跨服务 | 协同式Saga + 领域事件 | 松耦合，高可用 |
| 查询 | 简单CQRS（读写分离代码） | 初期避免过度复杂 |
| 测试 | 大量领域测试 + 少量集成测试 | 快速反馈，保护业务规则 |

---

## 24.7 DDD实施路线图

```
Week 1-2：战略设计
  ├── 召开事件风暴工作坊（全团队）
  ├── 识别子域，确定核心域
  ├── 划分限界上下文
  └── 建立各上下文的词汇表

Week 3-4：核心域战术设计
  ├── 重点设计推荐/定价上下文的领域模型
  ├── 识别聚合、值对象、领域事件
  └── 编写领域测试（TDD）

Week 5-6：订单/支付上下文设计
  ├── 设计Order聚合和状态机
  ├── 实现跨上下文的Saga
  └── 建立防腐层（与第三方支付集成）

Week 7-8：架构落地
  ├── 选择分层架构 + 六边形架构
  ├── 实现仓储（先内存，再SQL）
  └── 建立事件总线

持续：模型精炼
  ├── 每Sprint回顾领域模型
  ├── 随业务理解深化，调整边界
  └── 保持统一语言词汇表更新
```

---

## 24.8 DDD的本质再回顾

经过24章的学习，让我们回到最初的问题：**DDD是什么？**

```
DDD是一种：

1. 沟通方式
   ├── 开发者与业务的共同语言
   └── 减少"翻译"带来的理解偏差

2. 思维框架
   ├── 先理解业务，再写代码
   └── 模型驱动设计，而非数据库驱动

3. 设计方法
   ├── 战略：找对地方做对事（子域、限界上下文）
   └── 战术：用清晰的模型表达业务规则（聚合、实体、值对象）

4. 持续学习过程
   └── 随着对业务理解的深化，模型不断演化精炼

"DDD不是一次完成的设计，而是团队对业务持续深化理解的旅程。"
```

---

## 本教程总结

```
Part 1：基础认知
  理解了"软件复杂度来自业务本身"，DDD让代码与业务同构

Part 2：战略设计
  学会了划分子域、限界上下文、上下文映射、事件风暴

Part 3：战术设计
  掌握了实体、值对象、聚合、领域事件、仓储、服务、工厂

Part 4：架构模式
  理解了分层架构、六边形架构、CQRS、事件溯源

Part 5：高阶实践
  应用了DDD在微服务、Saga、防腐层、测试、综合案例中

下一步：
  ├── 在真实项目中实践这些概念
  ├── 参加事件风暴工作坊
  ├── 阅读Eric Evans的蓝皮书（原典）
  └── 阅读Vaughn Vernon的红皮书（实践指南）
```

---

> "设计并不是让它看起来更好，而是让它工作得更好。" —— Steve Jobs  
> 
> 在DDD中：**好的设计让代码工作得更好，因为它与业务的运作方式保持一致。**

---

**上一章：** [第23章：测试策略](./23-testing-strategies.md)  
**回到目录：** [教程首页](../README.md)
