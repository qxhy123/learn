# 第2章：TDD + DDD + DSL 三位一体

> "单独的技术是工具，组合起来才是方法论。"

---

## 2.1 三种方法论的历史定位

### TDD（测试驱动开发）
- **起源**：Kent Beck，2002年《Test-Driven Development: By Example》
- **核心循环**：Red → Green → Refactor
- **哲学**：测试先于实现，让代码从测试中"生长"出来
- **解决的问题**：如何保证每次修改不破坏已有功能

### DDD（领域驱动设计）
- **起源**：Eric Evans，2003年《Domain-Driven Design》
- **核心思想**：软件模型应该反映业务领域的心智模型
- **哲学**：代码是领域知识的精确表达
- **解决的问题**：随着项目增大，代码和业务越来越脱节

### DSL（领域特定语言）
- **起源**：Martin Fowler，《Domain-Specific Languages》2010年
- **核心思想**：为特定问题域设计专用的语言
- **哲学**：让领域专家也能读懂（甚至写）关键逻辑
- **解决的问题**：通用编程语言表达特定业务逻辑时噪音过多

---

## 2.2 三者的关系图

```
                    ┌─────────────────────────────────┐
                    │          Vibe Coding             │
                    │    (意图驱动的AI协作编程)          │
                    └────────────┬────────────────────┘
                                 │ 由三个支柱支撑
              ┌──────────────────┼──────────────────┐
              │                  │                   │
     ┌────────▼──────┐  ┌────────▼──────┐  ┌────────▼──────┐
     │     TDD       │  │     DDD       │  │     DSL       │
     │  (验证层)      │  │  (语义层)     │  │  (表达层)     │
     │               │  │               │  │               │
     │ 定义"完成"     │  │ 定义"概念"    │  │ 定义"语法"    │
     │ 的标准         │  │ 的准确含义    │  │ 的表达方式    │
     └───────────────┘  └───────────────┘  └───────────────┘
```

### 层次职责

| 层次 | 方法 | 问题 | 输出 |
|------|------|------|------|
| 验证层 | TDD | 这个东西做完了吗？ | 通过的测试套件 |
| 语义层 | DDD | 这个东西是什么？ | 领域模型 |
| 表达层 | DSL | 怎么描述这个东西？ | 领域语言 |

---

## 2.3 TDD 深度回顾

### 三条规则（Uncle Bob 的 TDD 定律）

1. **除非让一个失败的测试通过，否则不能写任何产品代码**
2. **不能编写超过足以导致失败的测试代码**
3. **不能编写超过足以通过当前失败测试的产品代码**

这三条规则的精髓：**以分钟为单位的微循环**。

```
时间轴：
T=0:00  写测试（红灯）
T=0:03  写最小实现（绿灯）
T=0:05  重构（绿灯保持）
T=0:08  下一个测试（红灯）
...
```

### TDD 的设计力量

TDD 不仅是测试工具，更是**设计工具**。当你发现写测试很难，往往意味着设计有问题：

```python
# 难以测试的设计（耦合太紧）
def process_order(order_id):
    db = Database.get_instance()  # 全局状态，测试困难
    order = db.query(f"SELECT * FROM orders WHERE id={order_id}")
    email = EmailService()  # 副作用，测试困难
    email.send_confirmation(order['email'])
    db.execute(f"UPDATE orders SET status='confirmed' WHERE id={order_id}")

# TDD 逼出来的好设计（依赖注入）
def confirm_order(
    order: Order,
    repository: OrderRepository,  # 可以用 Mock
    notifier: OrderNotifier       # 可以用 Mock
) -> OrderConfirmed:
    confirmed = order.confirm()
    repository.save(confirmed)
    notifier.notify(confirmed)
    return confirmed
```

---

## 2.4 DDD 深度回顾

### 战略设计（Strategic Design）

战略设计解决"系统如何划分"的问题：

```
电商系统的限界上下文划分：

┌──────────────────────────────────────────────┐
│                 电商平台                       │
│                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │  订单上下 │  │  商品上下 │  │  用户上下 │  │
│  │  文       │  │  文       │  │  文       │  │
│  │          │  │          │  │          │  │
│  │ Order    │  │ Product  │  │ Customer │  │
│  │ OrderItem│  │ Inventory│  │ Account  │  │
│  │ Payment  │  │ Category │  │ Profile  │  │
│  └──────────┘  └──────────┘  └──────────┘  │
└──────────────────────────────────────────────┘

注意：每个上下文中的 "Product" 含义不同：
- 订单上下文：产品的价格和数量（历史快照）
- 商品上下文：产品的详细信息和库存
- 用户上下文：用户的购买历史和偏好
```

### 战术设计（Tactical Design）

战术设计解决"如何实现领域对象"的问题：

```python
from dataclasses import dataclass, field
from typing import List
from datetime import datetime
import uuid

# 值对象（Value Object）：无身份，靠值比较
@dataclass(frozen=True)
class Money:
    amount: float
    currency: str
    
    def __add__(self, other: 'Money') -> 'Money':
        if self.currency != other.currency:
            raise CurrencyMismatchError()
        return Money(self.amount + other.amount, self.currency)

# 实体（Entity）：有身份，靠 ID 比较
@dataclass
class OrderItem:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    product_name: str = ""
    quantity: int = 0
    unit_price: Money = field(default_factory=lambda: Money(0, "CNY"))
    
    @property
    def subtotal(self) -> Money:
        return Money(self.unit_price.amount * self.quantity, self.unit_price.currency)

# 聚合根（Aggregate Root）：事务边界
@dataclass
class Order:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    customer_id: str = ""
    items: List[OrderItem] = field(default_factory=list)
    status: str = "draft"
    _events: List = field(default_factory=list, repr=False)
    
    def add_item(self, item: OrderItem) -> None:
        if self.status != "draft":
            raise OrderNotModifiableError("只有草稿状态的订单可以添加商品")
        self.items.append(item)
    
    def confirm(self) -> 'Order':
        if not self.items:
            raise EmptyOrderError("不能确认空订单")
        self.status = "confirmed"
        self._events.append(OrderConfirmed(
            order_id=self.id,
            total=self.total,
            confirmed_at=datetime.now()
        ))
        return self
    
    @property
    def total(self) -> Money:
        if not self.items:
            return Money(0, "CNY")
        return sum((item.subtotal for item in self.items), Money(0, "CNY"))
```

---

## 2.5 DSL 深度回顾

### 内部 DSL vs 外部 DSL

```python
# 外部 DSL：完全独立的语言
# order_rules.dsl（假设的 DSL 语法）
"""
rule "VIP折扣"
    when customer.tier == "vip"
    then apply discount 10%
end
"""

# 内部 DSL：寄宿在宿主语言中，利用语言特性
# Python 内部 DSL（利用方法链和上下文管理器）
discount_rule = (
    Rule("VIP折扣")
        .when(lambda order: order.customer.tier == "vip")
        .then(apply_discount(10))
)
```

### DSL 的三个层次

**层次1：命名 DSL** - 仅通过命名提升可读性
```python
# 普通代码
if user.level >= 3 and purchase_count > 5:
    apply_rate(0.9)

# 命名 DSL
if customer.is_eligible_for_loyalty_discount():
    apply_loyalty_discount()
```

**层次2：流式 DSL** - 方法链构建表达式
```python
report = (
    SalesReport()
        .for_period(Q1_2024)
        .grouped_by(Region)
        .sorted_by(Revenue, descending=True)
        .with_top(10)
        .build()
)
```

**层次3：语法 DSL** - 独立的语言，需要解析器
```
# 类 SQL 的查询语言
FIND orders
WHERE customer.tier = "vip" 
  AND amount > 1000
  AND placed_at WITHIN LAST 30 DAYS
ORDER BY amount DESC
LIMIT 10
```

---

## 2.6 三者如何协同工作

### 协同模式：由外到内

```
第1步（DDD战略）：划定限界上下文
  ↓ 明确了"订单上下文"是我们要做的
  
第2步（DDD战术）：设计领域模型
  ↓ 定义了 Order, Customer, Money 等概念
  
第3步（TDD）：为领域行为写测试
  ↓ 明确了 Order.confirm() 的行为规约
  
第4步（AI实现）：根据测试生成代码
  ↓ AI 知道领域模型和行为规约，生成高质量代码
  
第5步（DSL）：封装复杂业务逻辑
  ↓ 用流式 DSL 让业务规则可读
  
第6步（TDD验证DSL）：为DSL写测试
  ↓ 确保 DSL 的行为和预期一致
```

### 实际代码示例：三者融合

```python
# 这段代码展示了三者如何融合

# === TDD：测试定义行为规约 ===
class TestOrderDiscountWorkflow:
    
    def test_vip_order_gets_10_percent_off(self):
        """规约：VIP订单享受10%折扣"""
        # DDD：使用领域语言
        customer = Customer(id="c1", tier=CustomerTier.VIP)
        order = Order.place(
            customer=customer,
            items=[OrderItem(product="书", quantity=2, price=Money(50, "CNY"))]
        )
        
        # DSL：流式工作流表达业务逻辑
        priced_order = (
            PricingWorkflow(order)
                .apply_tier_discounts()
                .apply_promotions()
                .calculate_tax()
                .finalize()
        )
        
        # TDD：清晰的断言
        assert priced_order.discount_amount == Money(10, "CNY")
        assert priced_order.final_amount == Money(90, "CNY")
```

---

## 2.7 Vibe Coding 的 AI 提示模板

掌握了三者的概念后，这是与 AI 协作的标准提示模板：

```markdown
## 上下文
我在构建 [系统名称] 的 [限界上下文] 模块。

## 领域模型
[贴上 DDD 领域对象定义]

## 测试规约
[贴上 TDD 测试代码]

## 实现要求
请实现 [类名/方法名]，需要：
1. 通过所有测试
2. 遵循领域模型的命名
3. [其他特定要求]

## 可用的 DSL
[如果有 DSL，贴上 DSL 定义]
```

---

## 2.8 总结

三位一体的 Vibe Coding 给了你：

- **TDD**：安全网和前进节拍器
- **DDD**：清晰的领域词汇和边界
- **DSL**：业务逻辑的高密度表达

下面几章我们将深入每个维度，最后综合运用。

---

**下一章**：[心流状态与编程节奏](03-mindset-and-flow-state.md)
