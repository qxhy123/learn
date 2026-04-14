# 第6章：测试优先的设计思维

> "测试是设计工具，不只是验证工具。"

---

## 6.1 测试即设计文档

传统开发思路：先设计 → 再实现 → 最后测试  
TDD 思路：**测试就是设计**，测试驱动出实现

```python
# 这不只是测试，这是一份精确的 API 设计文档
class TestInventoryReservation:
    """
    库存预留服务的设计规约：
    - 预留操作必须是原子的
    - 预留失败时不改变任何状态
    - 支持超卖检测
    - 预留有时间限制
    """
    
    def test_reserve_reduces_available_quantity(self):
        """成功预留后，可用库存减少"""
        inventory = Inventory(product_id="p1", available=100)
        inventory.reserve(quantity=10, reservation_id="r1")
        assert inventory.available == 90
    
    def test_reserve_beyond_available_raises(self):
        """超卖时抛出明确异常"""
        inventory = Inventory(product_id="p1", available=5)
        with pytest.raises(InsufficientInventoryError) as exc_info:
            inventory.reserve(quantity=10, reservation_id="r1")
        assert exc_info.value.requested == 10
        assert exc_info.value.available == 5
    
    def test_failed_reserve_leaves_no_side_effects(self):
        """失败的预留不应改变任何状态"""
        inventory = Inventory(product_id="p1", available=5)
        initial_available = inventory.available
        
        try:
            inventory.reserve(quantity=10, reservation_id="r1")
        except InsufficientInventoryError:
            pass
        
        assert inventory.available == initial_available  # 状态未变
        assert len(inventory.reservations) == 0          # 无悬空预留
    
    def test_reservation_can_be_released(self):
        """预留可以被释放，释放后库存恢复"""
        inventory = Inventory(product_id="p1", available=100)
        inventory.reserve(quantity=10, reservation_id="r1")
        inventory.release("r1")
        assert inventory.available == 100
    
    def test_expired_reservation_auto_releases(self):
        """过期的预留自动释放"""
        inventory = Inventory(product_id="p1", available=100)
        inventory.reserve(
            quantity=10,
            reservation_id="r1",
            expires_at=datetime.now() - timedelta(hours=1)  # 已过期
        )
        # 触发过期清理（模拟时间流逝后的查询）
        available = inventory.get_available_after_cleanup()
        assert available == 100
```

**分析**：这5个测试就是 `Inventory` 的完整设计文档，任何人读完都知道它该如何工作。

---

## 6.2 可测试性即好设计

"测试困难"通常意味着设计问题。常见问题和 TDD 逼出的解决方案：

### 问题1：全局状态

```python
# 难以测试：依赖全局状态
import time

def calculate_order_age(order):
    now = time.time()  # 全局状态！测试时无法控制时间
    return now - order.created_at

# TDD 逼出的解法：注入时钟
def calculate_order_age(order, clock=None):
    if clock is None:
        clock = lambda: time.time()
    return clock() - order.created_at

# 测试中可以完全控制时间
def test_order_age_calculation():
    order = Order(created_at=1000.0)
    fake_clock = lambda: 1100.0  # 固定时间
    assert calculate_order_age(order, clock=fake_clock) == 100.0
```

### 问题2：隐藏依赖

```python
# 难以测试：隐藏的数据库依赖
class OrderService:
    def get_order(self, order_id):
        db = get_db_connection()  # 隐藏依赖！
        return db.query(Order).filter_by(id=order_id).first()

# TDD 逼出的解法：显式依赖注入
class OrderService:
    def __init__(self, order_repository: OrderRepository):
        self._repo = order_repository  # 显式依赖
    
    def get_order(self, order_id: str) -> Order:
        return self._repo.find_by_id(order_id)

# 测试中用 Mock 替代
def test_get_order_returns_correct_order():
    mock_repo = MagicMock()
    expected = Order(id="o1", ...)
    mock_repo.find_by_id.return_value = expected
    
    service = OrderService(mock_repo)
    result = service.get_order("o1")
    
    assert result == expected
```

### 问题3：职责过多（God Object）

```python
# 难以测试：一个类做太多事
class OrderManager:
    def process(self, order_data):
        # 1. 验证数据
        # 2. 计算价格
        # 3. 预留库存
        # 4. 创建订单
        # 5. 发送邮件
        # 6. 记录日志
        # 7. 发布事件
        pass  # 如何测试这7件事？

# TDD 写测试时，你会发现需要7个 Mock，这是警告信号
# TDD 逼出的解法：拆分职责
class OrderValidator:
    def validate(self, data) -> ValidationResult: ...

class PricingService:
    def calculate(self, order) -> PricedOrder: ...

class InventoryService:
    def reserve(self, items) -> Reservation: ...

class OrderFactory:
    def create(self, priced_order, reservation) -> Order: ...

# 每个类独立测试，简单清晰
```

---

## 6.3 从接口到实现（Outside-In TDD）

Outside-In TDD（也叫 London School TDD）：先写高层测试，再逐层向内实现。

### 第一层：验收测试（从用户视角）

```python
# tests/acceptance/test_order_placement_acceptance.py
class TestOrderPlacementAcceptance:
    """用户下单的完整流程验收测试"""
    
    def test_customer_can_place_order_and_receive_confirmation(self):
        """
        场景：客户成功下单
        给定：注册客户，有商品在库存中
        当：客户下单
        则：收到订单确认，库存减少
        """
        # 准备：构建完整的应用上下文
        app = build_test_application()
        
        # 执行：模拟用户操作
        response = app.place_order(
            customer_id="c001",
            items=[{"product_id": "p001", "qty": 2}]
        )
        
        # 验证：高层行为正确
        assert response.success is True
        assert response.order_id is not None
        assert response.confirmation_email_sent is True
```

### 第二层：应用服务测试

```python
# tests/unit/application/test_place_order_use_case.py
class TestPlaceOrderUseCase:
    
    def test_place_order_creates_confirmed_order(self):
        # 用 Mock 隔离所有基础设施
        mock_customer_repo = Mock()
        mock_order_repo = Mock()
        mock_inventory = Mock()
        mock_notifier = Mock()
        
        customer = Customer(id="c001", tier=CustomerTier.NORMAL)
        mock_customer_repo.find_by_id.return_value = customer
        mock_inventory.check_and_reserve.return_value = Reservation(id="r1")
        
        use_case = PlaceOrderUseCase(
            customer_repo=mock_customer_repo,
            order_repo=mock_order_repo,
            inventory=mock_inventory,
            notifier=mock_notifier
        )
        
        result = use_case.execute(PlaceOrderCommand(
            customer_id="c001",
            items=[OrderItemRequest(product_id="p001", qty=2)]
        ))
        
        assert result.success is True
        mock_order_repo.save.assert_called_once()
        mock_notifier.send_confirmation.assert_called_once()
```

### 第三层：领域测试（纯业务逻辑）

```python
# tests/unit/domain/test_order.py
class TestOrder:
    """纯领域逻辑测试，无任何 Mock"""
    
    def test_draft_order_can_be_confirmed(self):
        order = Order(status=OrderStatus.DRAFT, items=[make_item()])
        confirmed = order.confirm()
        assert confirmed.status == OrderStatus.CONFIRMED
    
    def test_already_confirmed_order_cannot_be_confirmed_again(self):
        order = Order(status=OrderStatus.CONFIRMED, items=[make_item()])
        with pytest.raises(InvalidOrderStateError):
            order.confirm()
```

---

## 6.4 测试命名即需求文档

好的测试名 = 活的需求文档：

```python
# 坏的测试名（不说明任何业务信息）
def test_1():
def test_order():
def test_order_method():

# 好的测试名（清晰的业务规约）
def test_vip_customer_gets_10_percent_discount_on_all_orders():
def test_order_cannot_be_confirmed_if_inventory_is_insufficient():
def test_premium_customer_discount_stacks_with_promotional_discount():
def test_order_total_rounds_to_2_decimal_places():

# 最好的命名模式
# test_{场景/状态}_{动作}_{期望结果}
def test_empty_cart_checkout_raises_empty_cart_error():
def test_vip_customer_apply_coupon_gets_stacked_discount():
def test_expired_coupon_apply_raises_coupon_expired_error():
```

### 用类组织相关测试

```python
class TestWhenOrderIsInDraftState:
    """测试草稿状态下的订单行为"""
    
    def test_can_add_items(self): ...
    def test_can_remove_items(self): ...
    def test_can_be_confirmed(self): ...
    def test_cannot_be_shipped(self): ...

class TestWhenOrderIsConfirmed:
    """测试已确认状态下的订单行为"""
    
    def test_cannot_add_items(self): ...
    def test_can_be_paid(self): ...
    def test_can_be_cancelled_within_24_hours(self): ...

class TestWhenOrderIsCancelled:
    """测试已取消状态下的订单行为"""
    
    def test_inventory_is_released(self): ...
    def test_refund_is_initiated_if_paid(self): ...
```

---

## 6.5 测试粒度的选择

### 四个象限

```
              │  快  │  慢  │
    ──────────┼──────┼──────┤
    隔 离 好  │  单  │  集  │
              │  元  │  成  │
    ──────────┼──────┼──────┤
    依 赖 真  │  组  │  端  │
    实 基 础  │  件  │  到  │
    设 施     │  测  │  端  │
              │  试  │      │
```

**Vibe Coding 的测试金字塔**：

```
        /\
       /  \     少量 E2E 测试（验收）
      /────\
     /      \   适量集成测试（场景）
    /────────\
   /          \ 大量单元测试（领域逻辑）
  /────────────\
```

### 测试粒度决策树

```
是否涉及外部依赖（DB/API/文件）？
├── 否 → 单元测试（pytest，无 Mock）
└── 是 → 需要 Mock？
    ├── 是 → 单元测试 + Mock
    └── 否 → 集成测试（需要真实基础设施）
```

---

## 6.6 实战：测试驱动设计促销引擎

```python
# 先写测试，让设计自然涌现
class TestPromotionEngine:
    
    def test_no_promotions_returns_original_price(self):
        engine = PromotionEngine(promotions=[])
        price = Money(100, "CNY")
        assert engine.apply(price, context=OrderContext()) == price
    
    def test_single_promotion_applied(self):
        promo = FixedDiscount(amount=Money(10, "CNY"))
        engine = PromotionEngine(promotions=[promo])
        assert engine.apply(Money(100, "CNY"), OrderContext()) == Money(90, "CNY")
    
    def test_multiple_promotions_stacked(self):
        promos = [
            FixedDiscount(amount=Money(10, "CNY")),
            PercentageDiscount(rate=Decimal("0.1"))
        ]
        engine = PromotionEngine(promotions=promos)
        # 100 - 10 = 90, 90 * 0.9 = 81
        assert engine.apply(Money(100, "CNY"), OrderContext()) == Money(81, "CNY")
    
    def test_promotion_not_applied_if_condition_not_met(self):
        promo = ConditionalDiscount(
            condition=lambda ctx: ctx.customer.tier == CustomerTier.VIP,
            discount=FixedDiscount(amount=Money(10, "CNY"))
        )
        engine = PromotionEngine(promotions=[promo])
        normal_customer_context = OrderContext(customer=Customer(tier=CustomerTier.NORMAL))
        
        # 普通客户不享受VIP折扣
        assert engine.apply(Money(100, "CNY"), normal_customer_context) == Money(100, "CNY")

# 测试驱动出的设计——注意这些接口是 TDD 自然涌现的
from abc import ABC, abstractmethod
from typing import Protocol

class Promotion(Protocol):
    def apply(self, price: Money, context: 'OrderContext') -> Money: ...

@dataclass
class PromotionEngine:
    promotions: list[Promotion]
    
    def apply(self, price: Money, context: 'OrderContext') -> Money:
        result = price
        for promotion in self.promotions:
            result = promotion.apply(result, context)
        return result
```

---

## 总结

测试优先的设计思维带来：
1. **强制清晰性**：写测试前必须清楚接口
2. **可测试即好设计**：难测的设计就是坏设计
3. **活的文档**：测试名即需求规约
4. **从外到内**：Outside-In 保证接口先于实现

---

**下一章**：[TDD 涌现出架构](07-emergent-architecture.md)
