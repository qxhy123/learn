# 第7章：TDD 涌现出架构

> "好的架构不是设计出来的，是通过持续重构涌现出来的。"

---

## 7.1 什么是涌现式设计

传统做法：**大设计先行（Big Design Up Front, BDUF）**
- 先画详细的类图
- 先定义所有接口
- 然后实现

问题：
- 需求变化时，精心设计的架构崩塌
- 过度设计（YAGNI 违反）
- 实现时才发现设计问题

TDD 做法：**涌现式设计（Emergent Design）**
- 只为当前测试设计
- 通过持续重构提炼架构
- 架构随着需求增长自然成形

```
测试驱动的设计演进：

循环1: [简单实现]  → 满足1个测试
循环2: [小重构]    → 提取方法，满足2个测试
循环5: [中重构]    → 提取类，满足5个测试
循环10: [大重构]   → 提取模块，架构初现
循环20: [架构重构] → 分层清晰，模式浮现
```

---

## 7.2 重构的七个层次

### 层次1：提取变量（Extract Variable）

```python
# 重构前
def apply_vip_discount(price):
    return price * 0.9  # 0.9 是什么？

# 重构后
VIP_DISCOUNT_RATE = Decimal("0.9")

def apply_vip_discount(price: Money) -> Money:
    return price * VIP_DISCOUNT_RATE
```

### 层次2：提取函数（Extract Function）

```python
# 重构前
def confirm_order(order, customer):
    if customer.tier == "vip":
        discount = order.total * Decimal("0.1")
        order.total -= discount
    if not order.items:
        raise ValueError("空订单")
    order.status = "confirmed"
    order.confirmed_at = datetime.now()

# 重构后（每个函数一件事）
def confirm_order(order: Order, customer: Customer) -> ConfirmedOrder:
    _validate_order(order)
    discounted = _apply_customer_discount(order, customer)
    return _mark_as_confirmed(discounted)

def _validate_order(order: Order) -> None:
    if not order.items:
        raise EmptyOrderError("不能确认空订单")

def _apply_customer_discount(order: Order, customer: Customer) -> Order:
    if customer.tier == CustomerTier.VIP:
        return order.with_discount(Decimal("0.1"))
    return order

def _mark_as_confirmed(order: Order) -> ConfirmedOrder:
    return ConfirmedOrder(
        **order.__dict__,
        confirmed_at=datetime.now()
    )
```

### 层次3：提取类（Extract Class）

```python
# 重构前：Order 做了太多事
class Order:
    def confirm(self): ...
    def calculate_total(self): ...
    def apply_discount(self): ...
    def apply_tax(self): ...
    def validate(self): ...
    def serialize(self): ...  # 不属于领域！

# 重构后：职责分离
class Order:              # 纯领域聚合根
    def confirm(self): ...

class PricingCalculator: # 价格计算服务
    def calculate_total(self, order): ...
    def apply_discount(self, order, customer): ...
    def apply_tax(self, order, region): ...

class OrderValidator:    # 验证服务
    def validate(self, order): ...
```

### 层次4：提取接口（Extract Interface）

```python
# 重构前：直接依赖具体类
class OrderService:
    def __init__(self):
        self.repo = PostgresOrderRepository()  # 直接依赖具体实现

# 重构后：依赖抽象接口
from abc import ABC, abstractmethod

class OrderRepository(ABC):  # 提取接口
    @abstractmethod
    def find_by_id(self, order_id: str) -> Order: ...
    
    @abstractmethod
    def save(self, order: Order) -> None: ...

class OrderService:
    def __init__(self, repo: OrderRepository):  # 依赖抽象
        self._repo = repo
```

### 层次5：提取模块（Extract Module）

```python
# 重构前：所有代码在一个文件
# order.py
class Order: ...
class OrderItem: ...
class OrderService: ...
class OrderRepository: ...
class OrderValidator: ...
class PricingCalculator: ...

# 重构后：按职责分模块
# domain/order.py          ← 领域对象
# domain/pricing.py        ← 定价逻辑
# application/order_service.py  ← 应用服务
# infrastructure/order_repo.py  ← 基础设施
```

### 层次6：提取层（Extract Layer）

```python
# 重构后的分层结构（六边形架构）
"""
src/
├── domain/                    ← 纯业务逻辑，无框架依赖
│   ├── order/
│   │   ├── order.py          ← 聚合根
│   │   ├── order_item.py     ← 实体
│   │   └── events.py         ← 领域事件
│   └── pricing/
│       ├── pricing_rules.py  ← 领域服务
│       └── discount.py       ← 值对象
│
├── application/               ← 用例编排，依赖 domain
│   ├── place_order.py        ← 用例
│   └── confirm_order.py      ← 用例
│
└── infrastructure/            ← 技术实现，实现 domain 接口
    ├── repositories/
    │   └── pg_order_repo.py
    └── messaging/
        └── kafka_publisher.py
"""
```

### 层次7：提取服务（Extract Service）

当某个聚合或模块成为瓶颈时，提取为独立服务——这时 DDD 的限界上下文直接映射为微服务边界。

---

## 7.3 重构的时机：三原则

### 原则1：Rule of Three（三则重构）

```python
# 第一次：直接写
def get_vip_discount(price):
    return price * 0.9

# 第二次：再次出现，容忍复制
def get_premium_discount(price):
    return price * 0.8

# 第三次：出现了！提取抽象
def get_discount(price, rate):
    return price * (1 - rate)

VIP_RATE = 0.1
PREMIUM_RATE = 0.2
```

### 原则2：测试绿了再重构

```
❌ 红灯时重构 → 不知道重构破坏了什么
✅ 绿灯时重构 → 重构前后都是绿的，安全
```

### 原则3：小步重构，频繁提交

```bash
# 每次小重构后就提交
git commit -m "♻️ Extract: 将折扣逻辑提取到 DiscountCalculator"
git commit -m "♻️ Rename: price → unit_price 更准确"
git commit -m "♻️ Extract: OrderValidator 分离验证职责"
```

---

## 7.4 TDD 驱动出六边形架构

通过案例看 TDD 如何自然驱动出六边形架构：

### 阶段1：领域核心（什么都不依赖）

```python
# TDD 首先写领域测试 → 逼出纯领域代码
# tests/unit/domain/test_order.py
def test_order_confirm_changes_status():
    order = Order(items=[make_item()])
    order.confirm()
    assert order.status == OrderStatus.CONFIRMED

# 实现：纯 Python，零外部依赖
@dataclass
class Order:
    items: list[OrderItem]
    status: OrderStatus = OrderStatus.DRAFT
    
    def confirm(self) -> None:
        self.status = OrderStatus.CONFIRMED
```

### 阶段2：应用端口（抽象接口）

```python
# TDD 写用例测试 → 发现需要接口
def test_confirm_order_saves_to_repository():
    mock_repo = Mock()
    use_case = ConfirmOrderUseCase(repo=mock_repo)  # 发现需要 repo
    
    use_case.execute("order-1")
    
    mock_repo.save.assert_called_once()

# 这逼出了端口定义
class OrderRepository(Protocol):
    def find_by_id(self, id: str) -> Order: ...
    def save(self, order: Order) -> None: ...
```

### 阶段3：基础设施适配器（实现接口）

```python
# 最后才写：具体的数据库实现
class SqlOrderRepository:
    """实现 OrderRepository 端口的 SQL 适配器"""
    
    def __init__(self, session: Session):
        self._session = session
    
    def find_by_id(self, id: str) -> Order:
        row = self._session.query(OrderRow).filter_by(id=id).first()
        return self._to_domain(row)
    
    def save(self, order: Order) -> None:
        row = self._to_row(order)
        self._session.merge(row)
```

**观察**：TDD 的层次驱动出了六边形架构：
- 领域测试 → 纯领域核心
- 用例测试（Mock）→ 端口/接口
- 集成测试 → 适配器/基础设施

---

## 7.5 识别架构坏味道

TDD 让这些坏味道更容易被发现：

### 坏味道1：测试需要大量 Mock

```python
# 如果测试需要 Mock 5+ 个对象，这是类职责过多的信号
def test_something():
    mock_repo = Mock()
    mock_cache = Mock()
    mock_logger = Mock()
    mock_email = Mock()
    mock_sms = Mock()
    mock_audit = Mock()
    
    service = GodService(mock_repo, mock_cache, mock_logger, 
                         mock_email, mock_sms, mock_audit)
    # ... 这个类需要拆分
```

### 坏味道2：测试必须按顺序执行

```python
# 如果测试有隐式依赖，说明存在共享状态
def test_step1_create_user():
    db.execute("INSERT INTO users ...")

def test_step2_create_order():
    # 依赖 test_step1 先执行！
    user = db.query("SELECT * FROM users LIMIT 1")
    order = Order(customer_id=user.id)
```

### 坏味道3：测试理解需要阅读实现

```python
# 好的测试：读测试就够了，不需要看实现
def test_vip_discount_is_10_percent():
    customer = Customer(tier=CustomerTier.VIP)
    price = Money(100, "CNY")
    assert apply_discount(price, customer) == Money(90, "CNY")

# 坏的测试：必须读实现才能理解这在测什么
def test_pricing():
    x = process({"t": "v", "a": 100})
    assert x["r"] == 90
```

---

## 7.6 综合案例：从 Hello World 到分层架构

**需求**：实现一个简单的积分兑换礼品功能

#### 迭代1：最简单实现

```python
# Test
def test_can_redeem_gift():
    assert redeem_gift(points=500, gift_id="coffee") == True

# Implementation（最简）
def redeem_gift(points, gift_id):
    return True
```

#### 迭代3：加入规则

```python
# Tests
def test_insufficient_points_cannot_redeem():
    assert redeem_gift(points=100, gift_id="iphone") == False

def test_exact_points_can_redeem():
    assert redeem_gift(points=5000, gift_id="iphone") == True

# Implementation
GIFT_COSTS = {"coffee": 500, "iphone": 5000}

def redeem_gift(points: int, gift_id: str) -> bool:
    required = GIFT_COSTS.get(gift_id, 0)
    return points >= required
```

#### 迭代6：引入领域对象

```python
# Tests 开始用领域语言
def test_customer_can_redeem_coffee_with_500_points():
    customer = Customer(points=PointsBalance(500))
    gift = Gift(id="coffee", required_points=500)
    
    result = customer.redeem(gift)
    
    assert result.success is True
    assert customer.points.balance == 0

# Implementation 演进为领域对象
@dataclass
class PointsBalance:
    balance: int
    
    def deduct(self, amount: int) -> 'PointsBalance':
        if amount > self.balance:
            raise InsufficientPointsError()
        return PointsBalance(self.balance - amount)

@dataclass
class Customer:
    points: PointsBalance
    
    def redeem(self, gift: Gift) -> RedemptionResult:
        new_balance = self.points.deduct(gift.required_points)
        self.points = new_balance
        return RedemptionResult(success=True, redeemed_gift=gift)
```

#### 迭代12：架构分层出现

```
自然涌现的分层：
domain/
  customer.py       # Customer 聚合根
  gift.py           # Gift 实体
  points.py         # PointsBalance 值对象
  events.py         # GiftRedeemed 领域事件
  
application/
  redeem_gift.py    # RedeemGiftUseCase（用例）

infrastructure/
  customer_repo.py  # 数据库实现
  gift_catalog.py   # 礼品目录实现
```

---

## 总结

TDD 驱动出的架构具有：
- **单一职责**：每个类/函数只做一件事（测试逼出的）
- **依赖倒置**：依赖接口而非实现（Mock 逼出的）
- **松耦合**：模块间通过接口交互（可测试性要求）

架构不是设计的，是**用 TDD 循环"生长"出来的**。

---

**下一章**：[TDD 与 AI 结对编程](08-tdd-ai-pair-programming.md)
