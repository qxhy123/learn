# 第10章：限界上下文设计

> "一个系统里的'订单'和另一个系统里的'订单'，可能是完全不同的东西。"

---

## 10.1 限界上下文的本质

**限界上下文（Bounded Context）**：统一语言的有效范围边界。

```
在"订单上下文"中：
  Product = 下单时的价格快照（不可变历史记录）

在"商品目录上下文"中：
  Product = 当前在售商品（可更新的当前状态）

在"物流上下文"中：
  Product = 包裹中的物品（有重量、体积的物理属性）
```

这不是一个"Product"类用参数控制不同行为，而是三个独立的概念，在各自的上下文中有清晰的语义。

---

## 10.2 识别限界上下文

### 方法1：事件风暴（Event Storming）

在白板上用便利贴识别领域事件，然后聚类：

```
发现的领域事件：
[橙色便利贴 = 事件]
- CustomerRegistered（客户已注册）
- OrderPlaced（订单已下）
- PaymentProcessed（支付已处理）
- InventoryReserved（库存已预留）
- ShipmentDispatched（货物已发出）
- PackageDelivered（包裹已送达）

聚类后发现的上下文：
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  用户上下文   │  │  订单上下文   │  │  支付上下文   │
│              │  │              │  │              │
│ Customer     │  │ Order        │  │ Payment      │
│ Registered   │  │ Placed       │  │ Processed    │
└──────────────┘  └──────────────┘  └──────────────┘

┌──────────────┐  ┌──────────────┐
│  库存上下文   │  │  物流上下文   │
│              │  │              │
│ Inventory    │  │ Shipment     │
│ Reserved     │  │ Dispatched   │
└──────────────┘  └──────────────┘
```

### 方法2：语言边界识别

当两个团队对同一个词有不同理解时，你找到了上下文边界：

```
问：什么是"用户"？

市场部说："访问过网站的任何人"
销售部说："下过单的客户"
IT部说："有账号的注册用户"
物流部说："收货地址的联系人"

→ 这里有至少4个不同的上下文！
```

---

## 10.3 上下文映射（Context Mapping）

上下文之间如何交互？常见的几种关系：

### 共享内核（Shared Kernel）

两个上下文共享一小部分模型：

```python
# shared_kernel/
# 被多个上下文共用的核心概念

@dataclass(frozen=True)
class CustomerId:
    """共享内核：客户 ID 在所有上下文中含义相同"""
    value: str

@dataclass(frozen=True)
class Money:
    """共享内核：金额在所有上下文中语义相同"""
    amount: Decimal
    currency: str
```

### 客户-供应商（Customer-Supplier）

一个上下文（供应商）为另一个（客户）提供服务：

```python
# 商品目录上下文（供应商）提供接口
class ProductCatalogService:
    def get_product_info(self, product_id: str) -> ProductInfo: ...

# 订单上下文（客户）使用接口
class OrderService:
    def __init__(self, product_catalog: ProductCatalogService):
        self._catalog = product_catalog
    
    def add_item_to_order(self, order: Order, product_id: str, qty: int):
        product_info = self._catalog.get_product_info(product_id)
        item = OrderItem(
            product_id=product_id,
            product_name=product_info.name,  # 取快照
            unit_price=product_info.current_price,  # 取快照
            quantity=qty
        )
        order.add_item(item)
```

### 防腐层（Anti-Corruption Layer, ACL）

当你无法控制上游时，用防腐层保护自己的领域模型：

```python
# 外部遗留系统的数据结构（你无法修改）
class LegacyOrderData:
    order_no: str          # 对应我们的 order_id
    cust_id: int           # 对应我们的 customer_id（但类型不同！）
    status_code: int       # 1=draft, 2=confirmed, 3=shipped
    total_amt: float       # 对应我们的 total（但类型不同！）

# 防腐层：将遗留系统的模型翻译为我们的领域模型
class LegacyOrderACL:
    """防腐层：隔离遗留系统，翻译为我们的领域语言"""
    
    STATUS_MAP = {1: OrderStatus.DRAFT, 2: OrderStatus.CONFIRMED, 3: OrderStatus.SHIPPED}
    
    def translate(self, legacy: LegacyOrderData) -> Order:
        return Order(
            id=OrderId(legacy.order_no),
            customer_id=CustomerId(str(legacy.cust_id)),  # 类型转换
            status=self.STATUS_MAP[legacy.status_code],    # 状态映射
            total=Money(Decimal(str(legacy.total_amt)), "CNY")  # 精度处理
        )
```

---

## 10.4 限界上下文的代码组织

### 目录结构

```
src/
├── shared_kernel/              # 共享内核（最小化！）
│   ├── customer_id.py
│   ├── money.py
│   └── domain_event.py
│
├── order_context/              # 订单上下文
│   ├── domain/
│   │   ├── order.py            # 聚合根
│   │   ├── order_item.py       # 实体
│   │   └── order_events.py     # 领域事件
│   ├── application/
│   │   ├── place_order.py      # 用例
│   │   └── confirm_order.py    # 用例
│   ├── infrastructure/
│   │   └── order_repository.py
│   └── acl/
│       └── product_catalog_acl.py  # 防腐层
│
├── catalog_context/            # 商品目录上下文
│   ├── domain/
│   │   ├── product.py
│   │   └── category.py
│   └── ...
│
└── payment_context/            # 支付上下文
    ├── domain/
    │   └── payment.py
    └── ...
```

### 上下文间的通信

```python
# 方式1：领域事件（解耦合的首选方式）
# 订单上下文发出事件
@dataclass(frozen=True)
class OrderConfirmed:
    order_id: OrderId
    customer_id: CustomerId
    items: list[OrderItemSummary]
    total: Money

# 库存上下文订阅事件
class InventoryEventHandler:
    def handle_order_confirmed(self, event: OrderConfirmed):
        for item in event.items:
            self._inventory.reserve(
                product_id=item.product_id,
                quantity=item.quantity
            )

# 方式2：查询（同步调用）
# 订单上下文查询商品信息
class ProductQuery:
    def get_current_price(self, product_id: str) -> Money: ...
```

---

## 10.5 测试上下文边界

### 验证上下文隔离

```python
# 确保订单上下文的代码不直接导入商品目录上下文
# tests/architecture/test_context_isolation.py
import ast
import os
from pathlib import Path

def test_order_context_does_not_import_catalog_context():
    """订单上下文不应直接依赖商品目录上下文"""
    order_files = Path("src/order_context").rglob("*.py")
    
    for file in order_files:
        tree = ast.parse(file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "catalog_context" not in alias.name, \
                        f"{file} 直接导入了 catalog_context，应该通过 ACL 或事件交互"
            elif isinstance(node, ast.ImportFrom):
                if node.module and "catalog_context" in node.module:
                    raise AssertionError(
                        f"{file} 从 {node.module} 导入，应该通过 ACL 或事件交互"
                    )
```

### 验证上下文接口稳定性

```python
# 测试防腐层的翻译正确性
class TestProductCatalogACL:
    
    def test_translates_catalog_product_to_order_item_snapshot(self):
        acl = ProductCatalogACL(catalog_service=MockCatalogService())
        
        snapshot = acl.get_item_snapshot("product-001")
        
        # 验证：快照包含正确信息
        assert isinstance(snapshot, OrderItemSnapshot)
        assert snapshot.product_id == "product-001"
        assert isinstance(snapshot.unit_price, Money)
    
    def test_acl_handles_missing_product_gracefully(self):
        acl = ProductCatalogACL(
            catalog_service=MockCatalogService(missing_products=["p999"])
        )
        
        with pytest.raises(ProductNotFoundError):
            acl.get_item_snapshot("p999")
```

---

## 10.6 实战：电商系统的上下文设计

### 完整电商系统上下文图

```
┌────────────────────────────────────────────────────────────┐
│                         电商平台                             │
│                                                            │
│  ┌─────────────┐      ┌─────────────┐      ┌───────────┐  │
│  │  用户上下文   │ ───▶ │  订单上下文   │ ──▶  │ 支付上下文 │  │
│  │             │      │             │      │           │  │
│  │ Customer    │      │ Order       │      │ Payment   │  │
│  │ Account     │      │ OrderItem   │      │ Invoice   │  │
│  │ Profile     │      │ Coupon      │      │           │  │
│  └─────────────┘      └──────┬──────┘      └───────────┘  │
│         ▲                    │                    │        │
│         │             ┌──────▼──────┐             │        │
│         │             │  库存上下文   │             │        │
│         │             │             │             │        │
│         │             │ Product     │             │        │
│         │             │ Inventory   │             │        │
│         │             │ Reservation │             │        │
│         │             └──────┬──────┘             │        │
│         │                    │                    │        │
│         │             ┌──────▼──────┐             │        │
│         └─────────────│  物流上下文   │◀────────────┘        │
│                       │             │                      │
│                       │ Shipment    │                      │
│                       │ Package     │                      │
│                       │ Tracking    │                      │
│                       └─────────────┘                      │
└────────────────────────────────────────────────────────────┘
```

### 上下文间的事件流

```python
# 事件流：下单 → 扣库存 → 创建物流单 → 通知用户

# 1. 订单上下文发出事件
order.confirm()
# 发出: OrderConfirmed(order_id, items, customer_id)

# 2. 库存上下文处理事件
class InventoryContext:
    def on_order_confirmed(self, event: OrderConfirmed):
        for item in event.items:
            self.reserve(item.product_id, item.quantity)
        self.publish(InventoryReserved(order_id=event.order_id))

# 3. 物流上下文处理事件  
class ShippingContext:
    def on_inventory_reserved(self, event: InventoryReserved):
        shipment = Shipment.create_for_order(event.order_id)
        self.publish(ShipmentCreated(shipment_id=shipment.id))

# 4. 通知上下文处理事件
class NotificationContext:
    def on_shipment_created(self, event: ShipmentCreated):
        order_info = self.order_query.get(event.order_id)
        self.email_service.send_shipping_notice(order_info.customer_email)
```

---

## 10.7 常见错误与对策

### 错误1：上下文太细（微服务地狱）

```
❌ 过度拆分：
   用户上下文 / 客户上下文 / 会员上下文 / 账户上下文
   → 实际上这些是一个上下文的不同方面

✅ 合理粒度：
   用户管理上下文（包含注册、登录、会员管理）
```

### 错误2：上下文边界模糊

```python
# ❌ 订单上下文直接操作库存
class OrderService:
    def confirm_order(self, order_id):
        order = self.order_repo.find(order_id)
        # 直接操作库存！违反上下文边界
        inventory = self.inventory_repo.find_by_product(order.items[0].product_id)
        inventory.quantity -= 1
        
# ✅ 通过事件解耦
class OrderService:
    def confirm_order(self, order_id):
        order = self.order_repo.find(order_id)
        confirmed = order.confirm()
        self.order_repo.save(confirmed)
        # 发布事件，让库存上下文自己处理
        self.event_bus.publish(confirmed.pull_events())
```

---

## 10.7 AI 辅助上下文发现

AI 可以加速事件风暴和上下文边界识别：

### 提示词模板

```
我正在为一个在线教育平台做事件风暴（Event Storming）。

核心业务：学员购买课程、学习课程、获得证书、讲师管理课程内容。

请帮我：
1. 列出所有可能的领域事件（用过去时命名）
2. 将事件按限界上下文分组
3. 标注上下文之间的事件传递关系
```

### AI 输出示例

```
课程管理上下文：CourseCreated, LessonAdded, CoursePublished
订单上下文：OrderPlaced, PaymentCompleted, OrderCancelled
学习追踪上下文：EnrollmentStarted, LessonCompleted, CourseCompleted
证书上下文：CertificateIssued

跨上下文事件流：
PaymentCompleted → EnrollmentStarted（订单→学习）
CourseCompleted → CertificateIssued（学习→证书）
```

### 审查要点

- AI 划分的上下文边界是否与团队组织结构匹配？
- 跨上下文的事件是否真的需要跨越，还是可以合并上下文？

> **Vibe Coding 心法**：AI 擅长穷举事件，你负责判断上下文边界是否合理。

---

## 总结

限界上下文是 DDD 最重要的战略工具：
- **每个上下文有自己的统一语言**
- **上下文通过接口/事件交互，不直接共享数据库**
- **防腐层保护领域模型不被外部污染**
- **TDD 测试可以验证上下文隔离**

---

**下一章**：[聚合根、实体与值对象](11-aggregates-entities-vos.md)
