# 第06章：上下文映射（Context Mapping）

## 学习目标

- 理解为什么需要上下文映射
- 掌握所有主要的集成模式及其适用场景
- 能够绘制上下文地图
- 理解上下游关系对团队的影响

---

## 6.1 上下文之间的关系

限界上下文不是孤岛，它们之间需要协作。**上下文映射（Context Mapping）** 就是描述这种协作关系的工具。

上下文映射关注两个问题：
1. **技术层面**：上下文之间如何交换数据？
2. **社交层面**：团队之间的关系如何？

> 注意：上下文映射不仅仅是技术集成模式，更是团队合作模式的体现。

---

## 6.2 上下游关系

在两个上下文之间，往往存在**上游（Upstream）** 和**下游（Downstream）** 的关系：

```
上游（Upstream / U）：
  - 数据/服务的提供方
  - 其变化会影响下游
  - 一般有更多"权力"

下游（Downstream / D）：
  - 数据/服务的消费方
  - 需要适应上游的接口
  - 一般处于"被动"地位

┌──────────────┐         ┌──────────────┐
│   上游       │────────►│   下游       │
│  Upstream    │         │  Downstream  │
└──────────────┘         └──────────────┘
```

**例子**：支付上下文是订单上下文的上游（订单需要知道支付是否成功）。

---

## 6.3 集成模式全览

DDD定义了多种上下文之间的集成模式，每种模式反映了不同的团队关系和技术选择：

```
合作关系 ← 关系紧密度 → 独立关系
    │                        │
    │  伙伴关系（Partnership）│
    │  共享内核（Shared Kernel）
    │  客户-供应商（Customer-Supplier）
    │  跟随者（Conformist）
    │  防腐层（ACL）
    │  开放主机服务（OHS）
    │  已发布语言（Published Language）
    │  各行其是（Separate Ways）
    └────────────────────────┘
```

---

## 6.4 详解各集成模式

### 模式1：伙伴关系（Partnership）

**场景**：两个上下文之间相互依赖，必须协调变更。

```
┌──────────────┐         ┌──────────────┐
│  订单上下文   │◄───────►│  库存上下文   │
│   Ordering  │  紧密协作 │  Inventory  │
└──────────────┘         └──────────────┘

特征：
- 两个团队同步发展，接口共同设计
- 任何接口变更需要双方协商
- 失败时两者一起负责

适用场景：
- 两个上下文本质上是同一业务流程的两部分
- 团队关系紧密，沟通成本低

⚠️ 风险：
- 紧密耦合，一方变慢会拖累另一方
- 长期来看可能需要合并为一个上下文
```

### 模式2：共享内核（Shared Kernel）

**场景**：两个上下文共享一部分领域模型（通常是基础类型）。

```python
# 共享内核：一个独立的共享库
# shared-kernel/
#   └── money.py, address.py, date_range.py 等基础类型

# 两个上下文都依赖这个共享库
from shared_kernel import Money, Address

# 订单上下文使用
class Order:
    total: Money
    shipping_address: Address

# 支付上下文使用
class Payment:
    amount: Money
```

```
特征：
- 共享部分非常小（通常只是基础值类型）
- 共享内核的变更需要双方同意
- 通常有专门的团队维护共享内核

适用场景：
- 多个上下文都需要相同的基础类型（如货币、日期范围）
- 这些类型变化极少

⚠️ 风险：
- 共享内核变大后，会变成共享数据库的同等问题
- 要保持克制，共享内核只放真正通用的东西
```

### 模式3：客户-供应商（Customer-Supplier）

**场景**：上下游关系，下游（客户）可以影响上游（供应商）的接口。

```
┌──────────────┐         ┌──────────────┐
│   支付上下文  │─[U/S]──►│   订单上下文  │
│   Payment   │ 供应商    │   Ordering  │ 客户
└──────────────┘         └──────────────┘

上游(供应商)特征：
- 提供接口和服务
- 会考虑下游的需求
- 有版本控制和变更通知

下游(客户)特征：
- 消费接口
- 可以提出需求，但不能强制
- 需要接受上游的变更节奏
```

```python
# 上游（供应商）定义的接口
class PaymentService:
    def charge(
        self, 
        order_id: str, 
        amount: Money,
        payment_method: PaymentMethod
    ) -> PaymentResult:
        ...
    
    def get_payment_status(self, payment_id: str) -> PaymentStatus:
        ...

# 下游（客户）消费此接口
class OrderApplicationService:
    def __init__(self, payment_service: PaymentService):
        self._payment_service = payment_service
    
    def process_payment(self, order: Order) -> None:
        result = self._payment_service.charge(
            order_id=str(order.id),
            amount=order.total,
            payment_method=order.payment_method
        )
        order.confirm_payment(result)
```

### 模式4：跟随者（Conformist）

**场景**：下游完全遵从上游的模型，不做任何转换。

```
┌──────────────┐         ┌──────────────┐
│   第三方支付   │────────►│  支付上下文   │
│   Alipay    │ 完全跟随  │   Payment   │
└──────────────┘         └──────────────┘

特征：
- 上游不会考虑下游的需求
- 下游直接使用上游的数据模型
- 下游模型与上游高度耦合

适用场景：
- 上游是强势的第三方（不受控制）
- 上游模型设计合理，跟随成本低
- 集成简单，无需转换逻辑

⚠️ 缺点：
- 下游与上游紧耦合，上游变化会直接冲击下游
- 上游的"坏设计"也传染给下游
```

```python
# 跟随者模式：直接使用上游（支付宝）的数据结构
class AlipayCallbackHandler:
    def handle(self, alipay_notification: dict) -> None:
        # 直接使用支付宝的字段名和数据格式
        # 没有转换，完全跟随
        order_id = alipay_notification["out_trade_no"]
        status = alipay_notification["trade_status"]
        # ...
        # 这里直接把支付宝的模型用到了我们系统里
```

### 模式5：防腐层（Anti-Corruption Layer，ACL）⭐

**场景**：下游需要与上游集成，但不想被上游的模型"污染"，建立一个翻译层。

```
┌──────────────┐   翻译   ┌──────────────┐   ┌──────────────┐
│   遗留系统   │─────────►│   防腐层     │──►│  新系统上下文 │
│   Legacy    │  (ACL)   │   Adapter   │   │   New BC    │
└──────────────┘         └──────────────┘   └──────────────┘

防腐层负责：
- 将上游的概念翻译成本上下文的语言
- 隔离上游模型的"污染"
- 提供稳定的内部接口，即使上游变化
```

```python
# 防腐层实现示例
# 上游是一个遗留的ERP系统，数据格式混乱

# 遗留系统的数据格式（不可修改）
class LegacyOrderData:
    ord_no: str        # 奇怪的字段名
    ord_stat: int      # 状态用数字表示，文档不清楚
    cust_cd: str       # 客户代码（需要去另一张表查名字）
    tot_amt: float     # 精度丢失的浮点数金额
    cre_dt: str        # 字符串日期 "20231201"

# 防腐层：将遗留格式翻译成我们的领域语言
class LegacyOrderTranslator:
    """将遗留ERP的订单数据翻译成本系统的领域对象"""
    
    LEGACY_STATUS_MAP = {
        1: OrderStatus.PENDING,
        2: OrderStatus.CONFIRMED,
        3: OrderStatus.SHIPPED,
        9: OrderStatus.CANCELLED,
    }
    
    def translate(self, legacy: LegacyOrderData) -> Order:
        return Order(
            id=OrderId(legacy.ord_no),
            status=self._translate_status(legacy.ord_stat),
            customer=self._load_customer(legacy.cust_cd),
            total=Money(Decimal(str(legacy.tot_amt)), "CNY"),
            created_at=datetime.strptime(legacy.cre_dt, "%Y%m%d")
        )
    
    def _translate_status(self, legacy_status: int) -> OrderStatus:
        status = self.LEGACY_STATUS_MAP.get(legacy_status)
        if status is None:
            raise TranslationError(f"未知的遗留状态码: {legacy_status}")
        return status
    
    def _load_customer(self, customer_code: str) -> Customer:
        # 通过客户代码查询完整的客户信息
        ...

# 本上下文的仓储使用防腐层
class OrderRepository:
    def __init__(self, legacy_system: LegacyERPSystem, translator: LegacyOrderTranslator):
        self._legacy = legacy_system
        self._translator = translator
    
    def find_by_id(self, order_id: OrderId) -> Order:
        legacy_data = self._legacy.get_order(str(order_id))
        return self._translator.translate(legacy_data)
```

**防腐层是保护自己的利器**，特别是在与遗留系统或设计糟糕的外部系统集成时。

### 模式6：开放主机服务（Open Host Service，OHS）

**场景**：上游提供通用的服务接口，供多个下游使用。

```python
# 开放主机服务：一个上下文对外提供标准化的服务
# 通常结合"已发布语言"使用

# 商品目录上下文的开放主机服务
class ProductCatalogService:
    """商品目录上下文的开放主机服务
    
    这是对外开放的接口，多个下游上下文可以消费此服务。
    接口设计需要考虑多方需求，变更需要谨慎，需要版本控制。
    """
    
    def get_product(self, product_id: str) -> ProductDTO:
        """获取商品信息（开放接口）"""
        ...
    
    def search_products(self, query: ProductSearchQuery) -> ProductSearchResult:
        """搜索商品（开放接口）"""
        ...
    
    def get_product_batch(self, product_ids: list[str]) -> list[ProductDTO]:
        """批量获取商品（开放接口，为性能优化提供）"""
        ...
```

### 模式7：已发布语言（Published Language，PL）

**场景**：使用行业标准或自定义的通用语言进行交换，通常与OHS配合。

```python
# 已发布语言：使用标准化的数据格式（如 JSON Schema、Protobuf、OpenAPI）
# 所有下游都理解这个格式

# 用Protobuf定义已发布语言
"""
// product.proto
message ProductDTO {
    string product_id = 1;
    string name = 2;
    string description = 3;
    repeated string image_urls = 4;
    MoneyDTO price = 5;
}

message MoneyDTO {
    int64 amount_in_cents = 1;
    string currency_code = 2;
}
"""

# 或者用 OpenAPI/JSON Schema
product_schema = {
    "type": "object",
    "properties": {
        "productId": {"type": "string"},
        "name": {"type": "string"},
        "price": {
            "type": "object",
            "properties": {
                "amountInCents": {"type": "integer"},
                "currencyCode": {"type": "string"}
            }
        }
    }
}
```

### 模式8：各行其是（Separate Ways）

**场景**：两个上下文完全独立，不集成。

```
特征：
- 宁可重复，也不集成
- 集成成本高于重复开发成本

适用场景：
- 两个上下文只需要很少的共享信息
- 集成的技术或沟通成本太高
- 可以通过简单的数据复制解决（如读取公共数据集）

示例：
- 报表上下文可能直接查数据库快照，而不与订单上下文集成
- 统计上下文可能有自己的非规范化数据，独立维护
```

---

## 6.5 绘制上下文地图

**上下文地图（Context Map）** 是将所有上下文及其关系可视化的工具：

```
                     已发布语言
           ┌─────────────────────────────────┐
           │                                 │
           │   ┌──────────┐    OHS/PL       ▼
     ACL   │   │  商品目录  │──────────►┌──────────┐
  ┌────────┤   └──────────┘            │  搜索    │
  │        │         │                 └──────────┘
  │     共享 │    C-S  │
  │     内核 │         ▼
  │        │   ┌──────────┐    伙伴    ┌──────────┐
  │        └──►│   订单    │◄─────────►│   库存    │
  ▼            └──────────┘           └──────────┘
┌──────────┐        │                      │
│  遗留ERP  │        │ C-S                  │ 事件
└──────────┘        ▼                      ▼
                ┌──────────┐    C-S   ┌──────────┐
                │   支付    │─────────►│   物流    │
                └──────────┘          └──────────┘

图例：
─────► : 上下游关系（箭头指向下游）
◄────► : 伙伴关系
ACL    : 防腐层
C-S    : 客户-供应商
OHS/PL : 开放主机服务/已发布语言
```

### 上下文地图的作用

1. **让集成关系可见**：团队清楚地知道谁依赖谁
2. **识别风险**：哪些上游变化会波及哪些下游
3. **指导团队协作**：不同集成模式对应不同的协作要求
4. **发现问题**：集成太复杂可能意味着边界划分有问题

---

## 6.6 集成模式选择指南

```
当你需要与外部系统集成时：
  └── 外部系统设计合理，可以直接跟随？
      ├── 是 → 考虑 Conformist 或 Customer-Supplier
      └── 否 → 使用 ACL 保护自己

当你是数据提供方，需要服务多个消费者时：
  └── 是否能标准化接口？
      ├── 是 → Open Host Service + Published Language
      └── 否 → Customer-Supplier（按需协商接口）

当两个上下文深度协作时：
  └── 团队关系如何？
      ├── 紧密协作一个团队 → Partnership
      ├── 共享少量基础类型 → Shared Kernel
      └── 明确上下游 → Customer-Supplier

当集成成本高，价值低时：
  └── Separate Ways
```

---

## 6.7 防腐层的深度实现

由于防腐层是最常用且最重要的模式，我们深入看一个完整实现：

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

# ============================================================
# 场景：电商平台需要集成第三方物流系统
# 第三方物流API设计混乱，我们需要防腐层保护自己
# ============================================================

# ------- 第三方物流系统的接口（不受我们控制）-------

class ThirdPartyLogisticsAPI:
    """第三方物流公司的原始API（设计混乱）"""
    
    def create_waybill(self, data: dict) -> dict:
        """创建运单，字段名混乱，文档不全"""
        # 返回格式：{"waybill_no": "...", "ec": 0, "em": "success"}
        ...
    
    def query_status(self, waybill_no: str) -> dict:
        """查询状态，用数字代码表示状态"""
        # 返回：{"status_code": "GTRY03", "status_desc": "派件中"}
        ...

# ------- 我们系统的领域模型（清晰的业务语言）-------

@dataclass
class ShippingOrder:
    """发货单（我们的领域概念）"""
    order_id: str
    sender: Address
    recipient: Address
    items: list[ShippingItem]

@dataclass  
class TrackingInfo:
    """物流跟踪信息（我们的领域概念）"""
    tracking_number: str
    current_status: ShipmentStatus
    estimated_delivery: Optional[date]
    events: list[TrackingEvent]

# ------- 防腐层 -------

class LogisticsAntiCorruptionLayer:
    """
    物流防腐层
    职责：将第三方物流的混乱接口翻译成我们清晰的领域语言
    """
    
    # 状态码映射（从第三方代码到我们的枚举）
    STATUS_CODE_MAP = {
        "GTRY01": ShipmentStatus.PICKED_UP,    # 已揽件
        "GTRY02": ShipmentStatus.IN_TRANSIT,   # 运输中
        "GTRY03": ShipmentStatus.OUT_FOR_DELIVERY,  # 派件中
        "GTRY04": ShipmentStatus.DELIVERED,    # 已签收
        "GTRY99": ShipmentStatus.EXCEPTION,    # 异常
    }
    
    def __init__(self, api: ThirdPartyLogisticsAPI):
        self._api = api
    
    def create_shipment(self, shipping_order: ShippingOrder) -> str:
        """
        创建发货单
        翻译：ShippingOrder → 第三方API格式 → tracking_number
        """
        # 将我们的领域对象翻译成第三方API的格式
        api_data = self._translate_to_api_format(shipping_order)
        
        # 调用第三方API
        response = self._api.create_waybill(api_data)
        
        # 检查错误
        if response.get("ec") != 0:
            raise ShippingException(f"创建运单失败: {response.get('em')}")
        
        # 提取运单号（我们只关心这一个结果）
        return response["waybill_no"]
    
    def get_tracking_info(self, tracking_number: str) -> TrackingInfo:
        """
        查询物流信息
        翻译：第三方API格式 → TrackingInfo
        """
        response = self._api.query_status(tracking_number)
        
        status = self.STATUS_CODE_MAP.get(
            response.get("status_code"), 
            ShipmentStatus.UNKNOWN
        )
        
        return TrackingInfo(
            tracking_number=tracking_number,
            current_status=status,
            estimated_delivery=self._parse_date(response.get("eta")),
            events=self._translate_events(response.get("events", []))
        )
    
    def _translate_to_api_format(self, order: ShippingOrder) -> dict:
        """将领域对象翻译成第三方的混乱格式"""
        return {
            "ordno": order.order_id,
            "snd_nm": order.sender.name,
            "snd_addr": f"{order.sender.province}{order.sender.city}{order.sender.detail}",
            "rcv_nm": order.recipient.name,
            "rcv_tel": order.recipient.phone,
            # ... 更多字段翻译
        }


# ------- 领域层使用防腐层 -------

class ShippingService:
    """物流领域服务（使用防腐层，不直接接触第三方API）"""
    
    def __init__(self, logistics_acl: LogisticsAntiCorruptionLayer):
        self._logistics = logistics_acl
    
    def ship_order(self, order: Order, recipient: Address) -> Shipment:
        shipping_order = ShippingOrder(
            order_id=str(order.id),
            sender=self._get_warehouse_address(order),
            recipient=recipient,
            items=[ShippingItem(item) for item in order.items]
        )
        
        tracking_number = self._logistics.create_shipment(shipping_order)
        return Shipment(order.id, tracking_number)
```

---

## 本章小结

| 模式 | 关键词 | 适用场景 |
|------|--------|---------|
| 伙伴关系 | 共同演化 | 深度协作的同级上下文 |
| 共享内核 | 共享基础类型 | 多个上下文的公共基础类型 |
| 客户-供应商 | 需求驱动 | 明确上下游，下游有议价权 |
| 跟随者 | 完全适应 | 上游强势，设计合理 |
| 防腐层 | 翻译隔离 | 遗留系统或糟糕的第三方API |
| 开放主机服务 | 标准接口 | 服务多个消费者 |
| 已发布语言 | 通用格式 | 接口标准化 |
| 各行其是 | 不集成 | 集成成本高于价值 |

---

## 思考练习

1. 在你的系统中，哪些集成关系最脆弱（一方变化，另一方立即出问题）？这对应哪种模式？
2. 如果你要集成一个设计糟糕的遗留系统，你会如何设计防腐层？
3. 绘制你当前系统的上下文地图（哪怕是草图），你发现了什么？

---

**上一章：** [第05章：限界上下文](./05-bounded-context.md)  
**下一章：** [第07章：子域类型与战略决策](./07-subdomain-types.md)
