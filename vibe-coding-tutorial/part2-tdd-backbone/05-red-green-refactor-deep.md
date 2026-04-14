# 第5章：Red-Green-Refactor 深度实践

> "把一个大问题分解成很多小的红绿循环，每一步都是确定的。"

---

## 5.1 理解三个阶段的深层含义

### Red：不仅仅是"写测试"

Red 阶段的真正目的是**明确下一步的终态**。

```
Red 阶段做的事：
1. 思考：这个功能完成后，可观察的行为是什么？
2. 建模：用领域语言描述这个行为
3. 编码：将行为翻译为可执行的测试
4. 确认：运行测试，看到它因为"正确的原因"失败
```

**"因为正确的原因失败"** 很重要：

```python
# 错误的 Red：因为语法错误失败（没有价值）
def test_order_confirmation():
    order = Order(
    # SyntaxError: 括号没关闭
    
# 错误的 Red：因为导入错误失败（不是业务逻辑失败）
def test_order_confirmation():
    from myproject.order import Order  # ModuleNotFoundError
    
# 正确的 Red：因为业务逻辑不存在而失败
def test_order_confirmation():
    order = Order(items=[OrderItem(name="书", price=Money(29.9, "CNY"))])
    result = order.confirm()
    # AssertionError: 因为 confirm() 方法还不存在
    assert result.status == OrderStatus.CONFIRMED
```

### Green：最小实现原则

Green 阶段只做一件事：**让测试通过，不多，不少**。

```python
# 测试
def test_order_total_with_single_item():
    order = Order(items=[OrderItem(price=Money(10, "CNY"), qty=3)])
    assert order.total == Money(30, "CNY")

# Green 1：最小实现（甚至可以硬编码）
class Order:
    def __init__(self, items):
        self.items = items
    
    @property
    def total(self):
        return Money(30, "CNY")  # 硬编码！先让测试通过

# 然后立即添加下一个测试，逼出真实实现
def test_order_total_with_multiple_items():
    order = Order(items=[
        OrderItem(price=Money(10, "CNY"), qty=3),
        OrderItem(price=Money(20, "CNY"), qty=1)
    ])
    assert order.total == Money(50, "CNY")

# 现在 Green 2：被两个测试逼出的真实实现
@property
def total(self):
    return sum(
        Money(item.price.amount * item.qty, item.price.currency)
        for item in self.items
    )
```

### Refactor：在绿灯保护下重构

Refactor 阶段的规则：**测试始终保持绿色**。

```python
# 重构前（Green 但丑陋）
@property
def total(self):
    total_amount = 0
    currency = None
    for item in self.items:
        total_amount += item.price.amount * item.qty
        currency = item.price.currency
    if currency is None:
        return Money(0, "CNY")
    return Money(total_amount, currency)

# 重构后（Green 且优雅）——每步小重构后都验证绿灯
@property
def total(self) -> Money:
    if not self.items:
        return Money(0, "CNY")
    subtotals = (item.subtotal for item in self.items)
    return sum(subtotals, start=Money(0, self.items[0].price.currency))
```

---

## 5.2 TDD 的节奏控制

### 步长太大的症状

```python
# 危险信号：一次写了多个测试，还都没实现
def test_vip_discount():      # 未实现
def test_bulk_discount():     # 未实现
def test_seasonal_discount(): # 未实现
def test_combined_discount(): # 未实现

# 问题：面对4个红灯，不知道先实现哪个
# 对策：一次只有一个红灯
```

### 正确的节奏示例

```python
# 循环1：最基本情况
def test_no_discount_for_normal_customer():
    order = Order(customer=Customer(tier="normal"), amount=Money(100, "CNY"))
    assert order.apply_discount() == Money(100, "CNY")
# [Red → Green] → commit

# 循环2：VIP 折扣
def test_vip_gets_10_percent_discount():
    order = Order(customer=Customer(tier="vip"), amount=Money(100, "CNY"))
    assert order.apply_discount() == Money(90, "CNY")
# [Red → Green → Refactor] → commit

# 循环3：Premium 折扣
def test_premium_gets_20_percent_discount():
    order = Order(customer=Customer(tier="premium"), amount=Money(100, "CNY"))
    assert order.apply_discount() == Money(80, "CNY")
# [Red → Green → Refactor] → commit
```

---

## 5.3 测试驱动出的设计演进

### 案例：积分系统的演进过程

**起点**：我们要实现积分系统

#### 第一轮（最简单情况）

```python
# Test
def test_earn_points_basic():
    account = PointsAccount()
    account.earn(100)
    assert account.balance == 100

# Implementation（最小）
class PointsAccount:
    def __init__(self):
        self.balance = 0
    
    def earn(self, points: int):
        self.balance += points
```

#### 第二轮（引入有效期）

```python
# 新测试逼出新设计
def test_expired_points_not_counted():
    account = PointsAccount()
    account.earn(100, expires_at=date(2020, 1, 1))  # 过期
    account.earn(50, expires_at=date(2099, 1, 1))   # 有效
    assert account.balance == 50  # 只有有效积分

# Implementation 演进：balance 不能是简单整数了
from dataclasses import dataclass, field
from datetime import date
from typing import List

@dataclass
class PointsEntry:
    amount: int
    expires_at: date

@dataclass
class PointsAccount:
    _entries: List[PointsEntry] = field(default_factory=list)
    
    def earn(self, points: int, expires_at: date = None):
        if expires_at is None:
            expires_at = date(9999, 12, 31)  # 不过期
        self._entries.append(PointsEntry(points, expires_at))
    
    @property
    def balance(self) -> int:
        today = date.today()
        return sum(e.amount for e in self._entries if e.expires_at >= today)
```

#### 第三轮（引入扣减）

```python
# 新测试
def test_deduct_points():
    account = PointsAccount()
    account.earn(100)
    account.deduct(30)
    assert account.balance == 70

def test_cannot_deduct_more_than_balance():
    account = PointsAccount()
    account.earn(50)
    with pytest.raises(InsufficientPointsError):
        account.deduct(100)

# Implementation 演进
def deduct(self, points: int):
    if points > self.balance:
        raise InsufficientPointsError(
            f"余额不足：需要 {points}，实际 {self.balance}"
        )
    self._entries.append(PointsEntry(-points, date(9999, 12, 31)))
```

**观察**：通过三轮 TDD，`PointsAccount` 的设计自然演进，每一步都有测试保护。这就是 TDD 的"设计涌现"。

---

## 5.4 测试分类与组织

### 三角测试法（Triangulation）

用多个测试从不同角度"三角定位"一个行为：

```python
class TestDiscountCalculation:
    """从多个角度三角验证折扣计算"""
    
    # 角度1：基准情况
    def test_no_discount_returns_original_price(self):
        assert calculate_discount(Money(100, "CNY"), rate=0) == Money(100, "CNY")
    
    # 角度2：边界情况
    def test_full_discount_returns_zero(self):
        assert calculate_discount(Money(100, "CNY"), rate=100) == Money(0, "CNY")
    
    # 角度3：典型业务值
    def test_ten_percent_discount(self):
        assert calculate_discount(Money(100, "CNY"), rate=10) == Money(90, "CNY")
    
    # 角度4：浮点精度
    def test_discount_with_decimal_result(self):
        result = calculate_discount(Money(99, "CNY"), rate=10)
        assert result == Money(89.10, "CNY")  # 精度验证
    
    # 角度5：错误情况
    def test_negative_discount_rate_raises(self):
        with pytest.raises(InvalidDiscountRateError):
            calculate_discount(Money(100, "CNY"), rate=-1)
    
    # 角度6：超过100%的折扣率
    def test_discount_rate_over_100_raises(self):
        with pytest.raises(InvalidDiscountRateError):
            calculate_discount(Money(100, "CNY"), rate=101)
```

### Given-When-Then 结构

```python
def test_vip_order_confirmation_workflow():
    # Given：前置条件（用 DDD 语言描述场景）
    customer = Customer(tier=CustomerTier.VIP)
    items = [
        OrderItem(product="Python书", price=Money(59.9, "CNY"), qty=2)
    ]
    order = Order.place(customer=customer, items=items)
    
    # When：执行动作
    confirmed_order = order.confirm()
    
    # Then：验证结果
    assert confirmed_order.status == OrderStatus.CONFIRMED
    events = confirmed_order.pull_events()
    assert len(events) == 1
    assert isinstance(events[0], OrderConfirmed)
    assert events[0].total == Money(119.8, "CNY")
```

---

## 5.5 处理复杂依赖

### 依赖注入 + Mock

```python
# 领域服务有外部依赖时
class OrderService:
    def __init__(
        self,
        order_repo: OrderRepository,      # 依赖
        inventory_service: InventoryService,  # 依赖
        event_publisher: EventPublisher   # 依赖
    ):
        self._order_repo = order_repo
        self._inventory = inventory_service
        self._publisher = event_publisher
    
    def confirm_order(self, order_id: str) -> Order:
        order = self._order_repo.find_by_id(order_id)
        self._inventory.reserve(order.items)
        confirmed = order.confirm()
        self._order_repo.save(confirmed)
        for event in confirmed.pull_events():
            self._publisher.publish(event)
        return confirmed

# 测试：用 Mock 替换所有依赖
from unittest.mock import MagicMock, create_autospec

def test_order_confirmation_publishes_event():
    # 准备
    mock_repo = create_autospec(OrderRepository)
    mock_inventory = create_autospec(InventoryService)
    mock_publisher = create_autospec(EventPublisher)
    
    order = Order(id="o1", status=OrderStatus.DRAFT, items=[...])
    mock_repo.find_by_id.return_value = order
    
    service = OrderService(mock_repo, mock_inventory, mock_publisher)
    
    # 执行
    service.confirm_order("o1")
    
    # 验证：事件被发布了
    mock_publisher.publish.assert_called_once()
    published_event = mock_publisher.publish.call_args[0][0]
    assert isinstance(published_event, OrderConfirmed)
```

---

## 5.6 TDD 的常见误区

### 误区1：写了测试就是 TDD

```python
# 这不是 TDD，这是"测试后置"
# 先写了100行实现，再补测试
class OrderService:
    # 100行实现...
    pass

# 补的测试（不是 TDD）
def test_everything_works():
    service = OrderService()
    result = service.do_something()
    assert result is not None  # 没有意义的断言
```

真正的 TDD：**测试先于实现存在**。

### 误区2：一个测试验证多件事

```python
# 坏：一个测试验证太多
def test_order_workflow():
    order = Order(...)
    order.add_item(...)
    order.confirm()
    order.pay()
    order.ship()
    
    assert order.status == "shipped"
    assert order.payment.status == "completed"
    assert order.shipping.tracking_number is not None
    # ...10个断言

# 好：每个测试验证一件事
def test_confirmed_order_has_confirmed_status():
    ...
    assert order.status == OrderStatus.CONFIRMED

def test_order_confirmation_emits_event():
    ...
    assert OrderConfirmed in [type(e) for e in events]
```

### 误区3：测试细节而非行为

```python
# 坏：测试实现细节（脆弱，容易破）
def test_order_uses_list_for_items():
    order = Order(...)
    assert isinstance(order._items, list)  # 测试私有实现

# 好：测试可观察行为
def test_order_can_have_multiple_items():
    order = Order(...)
    order.add_item(item1)
    order.add_item(item2)
    assert len(order.items) == 2
```

---

## 5.7 综合实战：用 TDD 实现优惠券系统

```python
# 步骤1：写失败的测试（Red）
import pytest
from decimal import Decimal

class TestCouponApplication:
    
    def test_fixed_coupon_reduces_price_by_fixed_amount(self):
        """固定金额优惠券减少固定金额"""
        coupon = Coupon(type=CouponType.FIXED, discount=Money(20, "CNY"))
        price = Money(100, "CNY")
        assert coupon.apply(price) == Money(80, "CNY")
    
    def test_percentage_coupon_reduces_by_percentage(self):
        """百分比优惠券按比例减少"""
        coupon = Coupon(type=CouponType.PERCENTAGE, discount_rate=Decimal("0.1"))
        price = Money(100, "CNY")
        assert coupon.apply(price) == Money(90, "CNY")
    
    def test_coupon_cannot_make_price_negative(self):
        """优惠后价格不能为负"""
        coupon = Coupon(type=CouponType.FIXED, discount=Money(200, "CNY"))
        price = Money(100, "CNY")
        result = coupon.apply(price)
        assert result == Money(0, "CNY")
    
    def test_expired_coupon_raises_error(self):
        """过期优惠券抛出错误"""
        coupon = Coupon(
            type=CouponType.FIXED,
            discount=Money(20, "CNY"),
            expires_at=date(2020, 1, 1)  # 已过期
        )
        with pytest.raises(CouponExpiredError):
            coupon.apply(Money(100, "CNY"))

# 步骤2：最小实现（Green）
from dataclasses import dataclass
from decimal import Decimal
from datetime import date
from enum import Enum
from typing import Optional

class CouponType(Enum):
    FIXED = "fixed"
    PERCENTAGE = "percentage"

@dataclass
class Coupon:
    type: CouponType
    discount: Optional[Money] = None
    discount_rate: Optional[Decimal] = None
    expires_at: Optional[date] = None
    
    def apply(self, price: Money) -> Money:
        if self.expires_at and self.expires_at < date.today():
            raise CouponExpiredError(f"优惠券已于 {self.expires_at} 过期")
        
        if self.type == CouponType.FIXED:
            discounted = price.amount - self.discount.amount
            return Money(max(0, discounted), price.currency)
        
        elif self.type == CouponType.PERCENTAGE:
            discounted = price.amount * (1 - self.discount_rate)
            return Money(float(discounted), price.currency)
        
        raise ValueError(f"未知的优惠券类型: {self.type}")

# 步骤3：重构（Refactor）
@dataclass
class Coupon:
    type: CouponType
    discount: Optional[Money] = None
    discount_rate: Optional[Decimal] = None
    expires_at: Optional[date] = None
    
    def apply(self, price: Money) -> Money:
        self._validate_not_expired()
        discounted_amount = self._calculate_discount(price)
        return Money(max(Decimal(0), discounted_amount), price.currency)
    
    def _validate_not_expired(self):
        if self.expires_at and self.expires_at < date.today():
            raise CouponExpiredError(f"优惠券已于 {self.expires_at} 过期")
    
    def _calculate_discount(self, price: Money) -> Decimal:
        if self.type == CouponType.FIXED:
            return Decimal(str(price.amount)) - Decimal(str(self.discount.amount))
        elif self.type == CouponType.PERCENTAGE:
            return Decimal(str(price.amount)) * (1 - self.discount_rate)
        raise ValueError(f"未知的优惠券类型: {self.type}")
```

---

## 总结

Red-Green-Refactor 的深层含义：
- **Red**：精确定义"完成"
- **Green**：最小代价到达"完成"
- **Refactor**：在保证"完成"的前提下提升设计质量

步长控制是关键：**一次只有一个红灯，每个绿灯都提交**。

---

**下一章**：[测试优先的设计思维](06-test-first-design-thinking.md)
