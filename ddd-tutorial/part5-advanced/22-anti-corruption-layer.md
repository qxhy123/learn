# 第22章：防腐层（Anti-Corruption Layer）深度解析

## 学习目标

- 深入理解防腐层的设计模式和实现细节
- 掌握遗留系统集成的防腐层策略
- 学会设计防腐层的各个组件（翻译器、适配器、门面）
- 了解防腐层的测试策略

---

## 22.1 防腐层的本质

在第06章中我们初步介绍了防腐层。本章深入探讨其设计。

> **防腐层（Anti-Corruption Layer，ACL）** 是一个隔离层，将我们的领域模型与外部系统（遗留系统、第三方服务）的模型隔离开来，防止外部模型的概念"污染"我们的领域。

```
没有防腐层：
  我们的领域              外部系统
  ┌─────────────┐         ┌─────────────┐
  │  Order      │────────►│  LegacyERP  │
  │  Customer   │◄────────│  CUST_MAST  │  ← 外部的混乱概念
  │  Product    │         │  ORD_HDR    │    渗入到我们的模型
  └─────────────┘         └─────────────┘

有防腐层：
  我们的领域              防腐层              外部系统
  ┌─────────────┐      ┌──────────┐      ┌─────────────┐
  │  Order      │────►│ Translator│────►│  LegacyERP  │
  │  Customer   │◄────│ Adapter  │◄────│  CUST_MAST  │
  │  Product    │      │ Facade   │      │  ORD_HDR    │
  └─────────────┘      └──────────┘      └─────────────┘
  我们的模型干净          负责翻译           外部混乱不影响我们
```

---

## 22.2 防腐层的三个组件

### 组件1：门面（Facade）

简化外部系统的复杂接口：

```python
# 外部系统接口复杂（SOAP / 过时的API设计）
class LegacyERPSystem:
    def GETORDR(self, ORDNO: str, INCITM: bool = True) -> bytes:
        """获取订单，返回XML二进制"""
        ...
    
    def UPDORDR(self, ORDNO: str, STATCD: str, USRID: str) -> int:
        """更新订单状态，返回错误码"""
        ...
    
    def LSTCUST(self, CUSTTYP: str, MAXREC: int = 100) -> list:
        """列举客户"""
        ...


# 门面：提供简洁的接口，隐藏外部复杂性
class LegacyERPFacade:
    """门面：将复杂的ERP接口简化为我们需要的操作"""
    
    def __init__(self, erp: LegacyERPSystem):
        self._erp = erp
    
    def get_order_data(self, order_number: str) -> dict:
        """获取订单数据（简化接口）"""
        raw = self._erp.GETORDR(order_number, True)
        return self._parse_xml(raw)
    
    def update_order_status(self, order_number: str, status_code: str) -> None:
        """更新订单状态"""
        result = self._erp.UPDORDR(order_number, status_code, "SYSTEM")
        if result != 0:
            raise ERPException(f"更新订单状态失败，错误码：{result}")
    
    def _parse_xml(self, xml_bytes: bytes) -> dict:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml_bytes.decode('gbk'))  # 遗留系统用GBK编码
        return {child.tag: child.text for child in root}
```

### 组件2：适配器（Adapter）

将外部系统的接口适配到我们的端口定义：

```python
# 我们的端口（定义我们需要的能力）
class OrderSyncPort(ABC):
    @abstractmethod
    def sync_order_to_erp(self, order: Order) -> str: ...
    
    @abstractmethod
    def get_order_from_erp(self, erp_order_number: str) -> Order: ...

# 适配器：实现我们的端口，内部使用门面
class LegacyERPOrderAdapter(OrderSyncPort):
    def __init__(self, facade: LegacyERPFacade, translator: OrderTranslator):
        self._facade = facade
        self._translator = translator
    
    def sync_order_to_erp(self, order: Order) -> str:
        erp_data = self._translator.to_erp_format(order)
        erp_order_number = self._facade.create_order(erp_data)
        return erp_order_number
    
    def get_order_from_erp(self, erp_order_number: str) -> Order:
        erp_data = self._facade.get_order_data(erp_order_number)
        return self._translator.from_erp_format(erp_data)
```

### 组件3：翻译器（Translator）

在两套概念模型之间转换：

```python
class OrderTranslator:
    """在我们的领域模型和ERP模型之间翻译"""
    
    # ERP状态码 → 我们的订单状态
    ERP_STATUS_MAP = {
        "001": OrderStatus.DRAFT,
        "010": OrderStatus.PLACED,
        "020": OrderStatus.CONFIRMED,
        "030": OrderStatus.SHIPPED,
        "099": OrderStatus.CANCELLED,
    }
    
    # 我们的订单状态 → ERP状态码
    OUR_STATUS_TO_ERP = {v: k for k, v in ERP_STATUS_MAP.items()}
    
    def from_erp_format(self, erp_data: dict) -> Order:
        """将ERP数据翻译成我们的领域对象"""
        
        # 字段名映射（ERP使用缩写）
        order_id = erp_data.get("ORDNO") or erp_data.get("ord_no")
        customer_code = erp_data.get("CUSTCD")
        status_code = erp_data.get("ORDSTAT", "001")
        
        # 状态翻译
        status = self.ERP_STATUS_MAP.get(status_code)
        if status is None:
            raise TranslationError(f"未知ERP状态码: {status_code}")
        
        # 日期格式转换（ERP用 YYYYMMDD）
        created_str = erp_data.get("CRTDAT", "")
        try:
            created_at = datetime.strptime(created_str, "%Y%m%d") if created_str else datetime.now()
        except ValueError:
            raise TranslationError(f"无效日期格式: {created_str}")
        
        # 金额转换（ERP以分为单位）
        total_fen = int(erp_data.get("ORDAMT", 0))
        total = Money(Decimal(total_fen) / 100, "CNY")
        
        # 构建我们的领域对象
        return Order.reconstitute(
            order_id=OrderId(order_id),
            customer_id=self._translate_customer_code(customer_code),
            status=status,
            total=total,
            created_at=created_at,
            items=self._translate_items(erp_data.get("ITEMS", []))
        )
    
    def to_erp_format(self, order: Order) -> dict:
        """将我们的领域对象翻译成ERP格式"""
        return {
            "ORDNO": str(order.id),
            "CUSTCD": self._get_erp_customer_code(order.customer_id),
            "ORDSTAT": self.OUR_STATUS_TO_ERP.get(order.status, "001"),
            "ORDAMT": int(order.total.amount * 100),  # 转换为分
            "CRTDAT": order.created_at.strftime("%Y%m%d"),
            "ITEMS": [self._translate_item_to_erp(item) for item in order.items]
        }
    
    def _translate_customer_code(self, erp_code: str) -> str:
        """ERP客户代码 → 我们系统的客户ID（需要查询映射表）"""
        # 实际项目中可能需要查询数据库中的映射关系
        mapping = CustomerCodeMapping.find_by_erp_code(erp_code)
        if mapping is None:
            raise TranslationError(f"未找到ERP客户代码 {erp_code} 的映射")
        return mapping.our_customer_id
```

---

## 22.3 完整的防腐层架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         防腐层模块                               │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │   接口（Port）层：定义我们需要的能力                       │  │
│  │   OrderSyncPort / InventorySyncPort / CustomerPort        │  │
│  └──────────────────────────┬───────────────────────────────┘  │
│                              │ 实现                             │
│  ┌──────────────────────────▼───────────────────────────────┐  │
│  │   适配器（Adapter）层：实现端口                            │  │
│  │   LegacyERPOrderAdapter / LegacyInventoryAdapter          │  │
│  └──────┬──────────────────────────────────────┬────────────┘  │
│         │ 使用翻译器              使用门面        │             │
│  ┌──────▼──────────┐      ┌─────────────────────▼──────────┐   │
│  │  翻译器层        │      │  门面层                        │   │
│  │  OrderTranslator│      │  LegacyERPFacade               │   │
│  │  CustomerMapper │      │  （简化外部接口）               │   │
│  └─────────────────┘      └─────────────────────┬──────────┘   │
└──────────────────────────────────────────────────┼─────────────┘
                                                   │ 调用
                                    ┌──────────────▼──────────────┐
                                    │       外部系统               │
                                    │  LegacyERP / ThirdPartyAPI  │
                                    └─────────────────────────────┘
```

---

## 22.4 防腐层的测试策略

```python
class TestOrderTranslator:
    """翻译器的单元测试"""
    
    def setup_method(self):
        self.translator = OrderTranslator()
    
    def test_translate_erp_order_to_domain(self):
        """测试ERP格式转领域对象"""
        erp_data = {
            "ORDNO": "ERP-2024-001",
            "CUSTCD": "C001",
            "ORDSTAT": "010",     # 已下单
            "ORDAMT": "99900",    # 999.00元（分为单位）
            "CRTDAT": "20240115",
            "ITEMS": []
        }
        
        order = self.translator.from_erp_format(erp_data)
        
        assert str(order.id) == "ERP-2024-001"
        assert order.status == OrderStatus.PLACED
        assert order.total == Money(Decimal("999.00"), "CNY")
        assert order.created_at.date() == date(2024, 1, 15)
    
    def test_unknown_status_code_raises_error(self):
        """测试未知状态码的错误处理"""
        erp_data = {"ORDNO": "001", "CUSTCD": "C001", "ORDSTAT": "999"}
        
        with pytest.raises(TranslationError, match="未知ERP状态码: 999"):
            self.translator.from_erp_format(erp_data)


class TestLegacyERPOrderAdapter:
    """适配器的集成测试（使用Mock的门面）"""
    
    def setup_method(self):
        self.mock_facade = MagicMock(spec=LegacyERPFacade)
        self.translator = OrderTranslator()
        self.adapter = LegacyERPOrderAdapter(self.mock_facade, self.translator)
    
    def test_get_order_from_erp(self):
        """测试从ERP获取订单"""
        self.mock_facade.get_order_data.return_value = {
            "ORDNO": "ERP-001",
            "CUSTCD": "C001",
            "ORDSTAT": "010",
            "ORDAMT": "50000",
            "CRTDAT": "20240115",
            "ITEMS": []
        }
        
        order = self.adapter.get_order_from_erp("ERP-001")
        
        self.mock_facade.get_order_data.assert_called_once_with("ERP-001")
        assert order.status == OrderStatus.PLACED
        assert order.total == Money(Decimal("500.00"), "CNY")
```

---

## 22.5 何时建立防腐层

```
必须建立防腐层的情况：
  ✅ 集成遗留系统（设计老旧、命名混乱）
  ✅ 集成外部第三方服务（无法修改其API）
  ✅ 外部模型的概念与我们的领域模型有根本差异
  ✅ 需要保护自己免受外部变化影响

可以考虑不建立的情况：
  ❌ 外部系统与我们使用相同的概念（共享内核）
  ❌ 外部系统设计合理，直接使用更简单（跟随者模式）
  ❌ 集成关系极其简单（一两个字段的转换）
```

---

## 本章小结

| 组件 | 职责 |
|------|------|
| 门面（Facade） | 简化外部系统的复杂接口 |
| 翻译器（Translator） | 两套概念模型之间的双向转换 |
| 适配器（Adapter） | 实现我们的端口接口，使用门面和翻译器 |

**防腐层的价值**：让我们的领域模型保持纯净，不受外部设计决策的影响。

---

**上一章：** [第21章：Saga模式](./21-saga-pattern.md)  
**下一章：** [第23章：测试策略](./23-testing-strategies.md)
