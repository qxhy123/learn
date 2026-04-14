# 第9章：统一语言实战

> "如果领域专家和开发者说的是不同的语言，代码就永远无法正确地表达业务。"

---

## 9.1 为什么需要统一语言

### 典型的语言混乱

真实项目中，同一个概念往往有多种叫法：

```
业务部门叫：    开发团队叫：    数据库叫：
"客户"          user            tb_user
"下单"          create_order    insert_order
"优惠券"        discount        coupon_code
"库存"          stock           inventory_qty
"发货"          ship            update_status(4)
```

这导致：
- 沟通时需要翻译，容易出错
- 代码看不懂业务意图
- 需求变更时，影响范围难以评估

### 统一语言的效果

```python
# 没有统一语言的代码
def process_usr(u_id, itm_list, disc=None):
    u = get_usr(u_id)
    o = mk_ord(u, itm_list)
    if disc:
        apply_disc(o, disc)
    upd_stk(itm_list)
    send_mail(u.mail, o.id)

# 有统一语言的代码
def place_order(customer_id: CustomerId, items: list[OrderItem], coupon: Coupon | None = None) -> Order:
    customer = customer_repository.find_by_id(customer_id)
    order = Order.place(customer=customer, items=items)
    if coupon:
        order.apply_coupon(coupon)
    inventory_service.reserve(items)
    notification_service.send_order_confirmation(customer, order)
    return order
```

第二段代码，业务人员也能大致理解。

---

## 9.2 建立统一语言词汇表

### 词汇表模板

```markdown
# 订单上下文统一语言词汇表

## 核心概念

### Customer（客户）
**定义**：在平台上注册了账户并能下单的用户
**同义词（禁用）**：user, buyer, member, account
**等级**：NORMAL（普通）, VIP, PREMIUM（贵宾）
**说明**：每个客户有唯一的 CustomerId

### Order（订单）
**定义**：客户在一次购买意向中选择的商品集合
**同义词（禁用）**：purchase, transaction, cart（购物车是不同概念）
**状态**：DRAFT（草稿）→ CONFIRMED（已确认）→ PAID（已支付）→ SHIPPED（已发货）→ DELIVERED（已送达）
**说明**：订单一旦确认，商品和价格不可更改

### OrderItem（订单项）
**定义**：订单中的一个商品行项目，包含商品信息、数量和当时的单价快照
**说明**：OrderItem 记录的是下单时的价格快照，不随商品价格变化

### Money（金额）
**定义**：包含数值和货币单位的值对象
**说明**：不可变，不同货币不能直接相加

### Coupon（优惠券）
**定义**：可以在下单时减免部分费用的凭证
**同义词（禁用）**：discount_code, promo_code, voucher
**类型**：FIXED（固定金额减免）, PERCENTAGE（百分比减免）
**状态**：ACTIVE（有效）, USED（已用）, EXPIRED（已过期）

## 领域事件

### OrderPlaced（订单已下）
**触发时机**：客户成功创建订单时
**包含信息**：order_id, customer_id, total, placed_at

### OrderConfirmed（订单已确认）
**触发时机**：订单通过所有验证并确认时
**包含信息**：order_id, confirmed_at

### OrderCancelled（订单已取消）
**触发时机**：客户或系统取消订单时
**包含信息**：order_id, reason, cancelled_at
```

---

## 9.3 统一语言驱动代码设计

### 从词汇表到代码

```python
# 词汇表中的每个概念直接对应代码
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from decimal import Decimal
from datetime import datetime

# === 值对象 ===

@dataclass(frozen=True)
class CustomerId:
    value: str
    
    def __str__(self): return self.value

@dataclass(frozen=True)
class OrderId:
    value: str
    
    def __str__(self): return self.value

@dataclass(frozen=True)
class Money:
    amount: Decimal
    currency: str
    
    def __add__(self, other: 'Money') -> 'Money':
        if self.currency != other.currency:
            raise CurrencyMismatchError(
                f"无法相加：{self.currency} 和 {other.currency}"
            )
        return Money(self.amount + other.amount, self.currency)
    
    def multiply(self, factor: Decimal) -> 'Money':
        return Money(self.amount * factor, self.currency)
    
    def __str__(self):
        return f"{self.amount} {self.currency}"

# === 枚举（来自词汇表的状态和类型）===

class CustomerTier(Enum):
    NORMAL = "normal"
    VIP = "vip"
    PREMIUM = "premium"

class OrderStatus(Enum):
    DRAFT = "draft"
    CONFIRMED = "confirmed"
    PAID = "paid"
    SHIPPED = "shipped"
    DELIVERED = "delivered"

class CouponType(Enum):
    FIXED = "fixed"
    PERCENTAGE = "percentage"

# === 领域事件（来自词汇表的事件）===

@dataclass(frozen=True)
class OrderPlaced:
    order_id: OrderId
    customer_id: CustomerId
    total: Money
    placed_at: datetime

@dataclass(frozen=True)
class OrderConfirmed:
    order_id: OrderId
    confirmed_at: datetime

@dataclass(frozen=True)
class OrderCancelled:
    order_id: OrderId
    reason: str
    cancelled_at: datetime
```

---

## 9.4 统一语言的测试风格

测试名和测试内容都使用统一语言：

```python
class TestOrderPlacement:
    """订单下单行为规约"""
    
    def test_customer_can_place_order_with_valid_items(self):
        """客户可以用有效商品下单"""
        customer = Customer(id=CustomerId("c001"), tier=CustomerTier.NORMAL)
        items = [
            OrderItem(
                product_name="Python编程",
                quantity=1,
                unit_price=Money(Decimal("59.9"), "CNY")
            )
        ]
        
        order = Order.place(customer=customer, items=items)
        
        assert order.status == OrderStatus.DRAFT
        assert order.customer_id == CustomerId("c001")
        assert len(order.items) == 1
    
    def test_order_total_equals_sum_of_item_subtotals(self):
        """订单总价等于所有订单项小计之和"""
        items = [
            OrderItem(product_name="书A", quantity=2, unit_price=Money(Decimal("30"), "CNY")),
            OrderItem(product_name="书B", quantity=1, unit_price=Money(Decimal("50"), "CNY")),
        ]
        order = Order.place(customer=make_customer(), items=items)
        
        assert order.total == Money(Decimal("110"), "CNY")
    
    def test_placing_order_emits_order_placed_event(self):
        """下单时触发 OrderPlaced 领域事件"""
        customer = make_customer()
        
        order = Order.place(customer=customer, items=[make_item()])
        events = order.pull_events()
        
        assert len(events) == 1
        assert isinstance(events[0], OrderPlaced)
        assert events[0].customer_id == customer.id
```

---

## 9.5 处理不同上下文中的同一概念

DDD 的关键洞见：**同一个词在不同上下文有不同含义**。

```python
# 订单上下文中的 Product
@dataclass(frozen=True)
class OrderItemSnapshot:
    """订单项中的商品快照——记录下单时的历史信息"""
    product_id: str
    name: str
    unit_price: Money  # 下单时的价格，不可变
    
# 商品目录上下文中的 Product
@dataclass
class CatalogProduct:
    """商品目录中的商品——当前信息，随时可更新"""
    id: str
    name: str
    current_price: Money  # 当前价格，可能变化
    inventory: int

# 这两个"Product"是不同的！不要合并它们！
```

### 防腐层（Anti-Corruption Layer）

当两个上下文需要交互时，用防腐层翻译：

```python
class OrderContextProductAdapter:
    """将商品目录上下文的 Product 转换为订单上下文的 OrderItemSnapshot"""
    
    def __init__(self, catalog_service):
        self._catalog = catalog_service
    
    def get_order_snapshot(self, product_id: str) -> OrderItemSnapshot:
        """从商品目录获取商品，转换为订单快照"""
        catalog_product = self._catalog.find_product(product_id)
        
        # 防腐层：翻译不同上下文的概念
        return OrderItemSnapshot(
            product_id=catalog_product.id,
            name=catalog_product.name,
            unit_price=catalog_product.current_price  # 当前价格成为快照
        )
```

---

## 9.6 统一语言的维护

### 词汇演化的挑战

业务会变化，词汇也会变化：

```markdown
# 变更记录
2024-01-15: "用户" 改为 "客户"（Customer）
  原因：业务团队说"用户"太泛，"客户"更准确
  影响：所有代码从 User 改为 Customer
  工具：全局重命名，测试保证无功能变化

2024-03-01: 新增 "订阅客户"（SubscriberCustomer）
  原因：新增订阅业务，需要区分订阅客户和普通客户
  影响：Customer 成为基类，新增 SubscriberCustomer 子类型
```

### TDD 保护重命名

```bash
# 批量重命名时，TDD 是安全网
# 1. 运行测试，确认全绿
pytest -v  # 全绿

# 2. 执行重命名
find . -name "*.py" | xargs sed -i 's/class User/class Customer/g'
find . -name "*.py" | xargs sed -i 's/: User/: Customer/g'

# 3. 再次运行测试，确认全绿
pytest -v  # 全绿 → 重命名成功，没有遗漏
```

---

## 9.7 综合实战：建立图书馆系统的统一语言

**业务场景**：图书馆借阅系统

### 第一步：与领域专家对话，识别词汇

```
领域专家说：
"读者来图书馆借书，可以同时借多本。
 书有库存，如果库存不够，读者可以预约。
 读者最多借5本，每本最多借14天，可以续借一次。
 超期未还要收逾期费。"

识别出的核心词汇：
- 读者（Patron）：借书的人
- 图书（Book）：可以被借的资源
- 副本（Copy）：图书的物理副本（一本书可以有多个副本）
- 借阅（Borrowing）：读者借用某本书的副本
- 预约（Reservation）：副本不可用时的等待队列
- 续借（Renewal）：延长借阅期限
- 逾期费（OverdueFee）：超期未还的罚款
```

### 第二步：写词汇表

```python
# domain/ubiquitous_language.py
"""
图书馆系统统一语言

Patron（读者）：已注册的图书馆成员，可以借书和预约
Book（图书）：图书馆收藏的一个作品（一个ISBN）
Copy（副本）：Book 的物理副本，有唯一的条形码
Borrowing（借阅）：Patron 持有某个 Copy 的期间
    - 最多同时5本
    - 期限14天
    - 可续借1次
Reservation（预约）：当所有 Copy 不可用时，Patron 排队等待
OverdueFee（逾期费）：超过还书期限后的罚款，每天1元
"""
```

### 第三步：用 TDD 验证统一语言

```python
class TestBorrowingRules:
    
    def test_patron_can_borrow_available_copy(self):
        """读者可以借可用副本"""
        patron = Patron(id=PatronId("p1"), active_borrowings=[])
        copy = Copy(id=CopyId("c1"), status=CopyStatus.AVAILABLE)
        
        borrowing = patron.borrow(copy)
        
        assert borrowing.patron_id == PatronId("p1")
        assert borrowing.copy_id == CopyId("c1")
        assert copy.status == CopyStatus.ON_LOAN
    
    def test_patron_cannot_borrow_more_than_5_books(self):
        """读者最多同时借5本"""
        patron = Patron(
            id=PatronId("p1"),
            active_borrowings=[make_borrowing() for _ in range(5)]
        )
        copy = Copy(id=CopyId("c6"), status=CopyStatus.AVAILABLE)
        
        with pytest.raises(BorrowingLimitExceededError):
            patron.borrow(copy)
    
    def test_patron_cannot_borrow_unavailable_copy(self):
        """读者不能借出已被借走的副本"""
        patron = Patron(id=PatronId("p1"), active_borrowings=[])
        copy = Copy(id=CopyId("c1"), status=CopyStatus.ON_LOAN)
        
        with pytest.raises(CopyNotAvailableError):
            patron.borrow(copy)
    
    def test_patron_can_renew_within_due_date(self):
        """读者可以在到期前续借"""
        borrowing = Borrowing(
            patron_id=PatronId("p1"),
            copy_id=CopyId("c1"),
            due_date=date.today() + timedelta(days=3),
            renewals_used=0
        )
        
        renewed = borrowing.renew()
        
        assert renewed.due_date == date.today() + timedelta(days=3+14)
        assert renewed.renewals_used == 1
    
    def test_patron_cannot_renew_if_already_renewed_once(self):
        """每本书只能续借一次"""
        borrowing = Borrowing(
            patron_id=PatronId("p1"),
            copy_id=CopyId("c1"),
            due_date=date.today() + timedelta(days=3),
            renewals_used=1  # 已续借一次
        )
        
        with pytest.raises(MaxRenewalsExceededError):
            borrowing.renew()
```

---

## 9.8 AI 辅助统一语言建立

在 Vibe Coding 工作流中，AI 是你建立统一语言的得力助手：

### 提示词模板

```
我正在为一个电商系统建立统一语言（Ubiquitous Language）。

以下是业务方的需求描述：
"用户可以把商品加到购物车，然后下单，下单后可以取消。
VIP 用户有折扣，还可以用积分抵扣。"

请帮我：
1. 提取核心领域术语，用中英文对照表列出
2. 标记容易产生歧义的词（同一个词在不同场景下含义不同）
3. 建议用哪个词替代有歧义的词
```

### AI 输出示例

```
| 中文术语 | 英文术语 | 歧义风险 |
|---------|---------|---------|
| 下单    | PlaceOrder | "下单"可能指创建草稿或确认支付，建议拆分 |
| 取消    | CancelOrder | 已发货后的"取消"应叫"退货申请" |
| 折扣    | Discount | VIP折扣 vs 促销折扣 是不同概念 |
```

### 审查要点

- AI 提取的术语是否与业务方的实际用词一致？
- 是否有遗漏的隐含概念（如"库存锁定"）？

> **Vibe Coding 心法**：让 AI 扮演"领域专家"对话伙伴，快速迭代词汇表草稿。

---

## 总结

统一语言是 DDD 的核心：
- **词汇表先行**：先建词汇表，再写代码
- **代码即文档**：统一语言让代码自然可读
- **防腐层隔离**：不同上下文的同一概念要隔离
- **TDD 保护演化**：词汇变化时，测试保证无遗漏

---

**下一章**：[限界上下文设计](10-bounded-context-design.md)
