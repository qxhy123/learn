# 第20章：高阶模式与未来展望

> "掌握基础之后，你才能打破规则。"

---

## 20.1 高阶模式概览

本章覆盖超越基础三位一体的进阶技术：

| 模式 | 解决的问题 | 技术复杂度 |
|------|-----------|-----------|
| 事件溯源（ES） | 完整历史记录，时间旅行调试 | ★★★★ |
| CQRS | 读写负载分离 | ★★★ |
| Specification 模式 | 复杂业务规则的可测试表达 | ★★★ |
| 函数式领域建模 | 不可变、无副作用的领域层 | ★★★★ |
| AI 生成领域模型 | 从自然语言直接生成 DDD 代码 | ★★ |

---

## 20.2 事件溯源（Event Sourcing）

### 核心思想

传统方式：存储**当前状态**
事件溯源：存储**状态变化历史**（事件流）

```python
# 传统：直接存储当前状态
class OrderRepository:
    def save(self, order: Order):
        db.execute("UPDATE orders SET status=? WHERE id=?",
                   order.status, order.id)

# 事件溯源：存储事件
class EventStore:
    def append(self, aggregate_id: str, events: List[DomainEvent], expected_version: int):
        for event in events:
            db.execute(
                "INSERT INTO events (aggregate_id, event_type, data, version) VALUES (?,?,?,?)",
                aggregate_id, type(event).__name__,
                serialize(event), expected_version + 1
            )

# 重建状态：重放事件
class OrderRepository:
    def find(self, order_id: str) -> Order:
        events = event_store.load(order_id)
        order = Order.empty()
        for event in events:
            order.apply(event)
        return order
```

### 事件溯源的 Apply 方法

```python
@dataclass
class Order:
    id: str = ""
    status: OrderStatus = OrderStatus.DRAFT
    items: List[OrderItem] = field(default_factory=list)
    version: int = 0
    _uncommitted_events: list = field(default_factory=list, repr=False)
    
    @classmethod
    def empty(cls) -> 'Order':
        return cls()
    
    def apply(self, event: DomainEvent) -> None:
        """重放事件，重建状态"""
        if isinstance(event, OrderPlaced):
            self.id = event.order_id
            self.status = OrderStatus.DRAFT
        elif isinstance(event, ItemAdded):
            self.items.append(OrderItem(
                product_id=event.product_id,
                quantity=event.quantity
            ))
        elif isinstance(event, OrderConfirmed):
            self.status = OrderStatus.CONFIRMED
        elif isinstance(event, OrderCancelled):
            self.status = OrderStatus.CANCELLED
        self.version += 1
    
    def confirm(self) -> None:
        # 只记录事件，不直接改状态
        self._uncommitted_events.append(
            OrderConfirmed(order_id=self.id)
        )
        # apply 方法真正改状态
        self.apply(self._uncommitted_events[-1])
```

### TDD 测试事件溯源

```python
class TestEventSourcingOrder:
    
    def test_order_rebuilt_from_events(self):
        """通过重放事件重建订单状态"""
        events = [
            OrderPlaced(order_id="o1", customer_id="c1"),
            ItemAdded(order_id="o1", product_id="p1", quantity=2),
            OrderConfirmed(order_id="o1"),
        ]
        
        order = Order.empty()
        for event in events:
            order.apply(event)
        
        assert order.id == "o1"
        assert order.status == OrderStatus.CONFIRMED
        assert len(order.items) == 1
    
    def test_time_travel_to_intermediate_state(self):
        """时间旅行：重建到某个历史时刻的状态"""
        all_events = load_all_events("o1")
        
        # 只重放到第3个事件
        order_at_step3 = Order.empty()
        for event in all_events[:3]:
            order_at_step3.apply(event)
        
        # 可以看到历史状态
        assert order_at_step3.status == OrderStatus.DRAFT
```

---

## 20.3 CQRS（命令查询职责分离）

### 核心思想

将**写操作（命令）**和**读操作（查询）**分离到不同的模型：

```python
# 命令侧：领域模型（复杂，保证一致性）
class PlaceOrderCommand:
    customer_id: str
    items: List[OrderItemRequest]

class PlaceOrderCommandHandler:
    def handle(self, cmd: PlaceOrderCommand) -> str:  # 返回 order_id
        customer = self._customer_repo.find(cmd.customer_id)
        order = Order.place(customer=customer, items=cmd.items)
        self._order_repo.save(order)
        return order.id

# 查询侧：读模型（简单，为展示优化）
@dataclass
class OrderSummaryView:
    order_id: str
    customer_name: str
    total_amount: float
    status: str
    item_count: int
    created_at: datetime

class OrderQueryService:
    def get_order_summary(self, order_id: str) -> OrderSummaryView:
        # 直接从优化过的读表查询，不走聚合
        row = self._read_db.query(
            "SELECT o.id, c.name, o.total, o.status, COUNT(i.id), o.created_at "
            "FROM orders o JOIN customers c ON o.customer_id = c.id "
            "LEFT JOIN order_items i ON o.id = i.order_id "
            "WHERE o.id = ? GROUP BY o.id",
            order_id
        ).first()
        return OrderSummaryView(**row)
```

### DSL 封装 CQRS

```python
class OrderCommands:
    """命令侧 DSL"""
    
    @staticmethod
    def place(customer_id: str) -> 'OrderCommandBuilder':
        return OrderCommandBuilder(customer_id)


class OrderQueries:
    """查询侧 DSL"""
    
    @staticmethod
    def summary(order_id: str) -> OrderSummaryView:
        return order_query_service.get_order_summary(order_id)
    
    @staticmethod
    def for_customer(customer_id: str) -> 'CustomerOrdersQuery':
        return CustomerOrdersQuery(customer_id)


# 使用：命令和查询用不同的 DSL
order_id = (
    OrderCommands
        .place("alice")
        .with_item("book", qty=2, price=Money(Decimal("29.9"), "CNY"))
        .execute()
)

# 查询用优化的读模型
summary = OrderQueries.summary(order_id)
print(f"订单 {summary.order_id}：{summary.total_amount} 元，{summary.item_count} 件商品")
```

---

## 20.4 Specification 模式

将复杂的业务规则封装为可组合的规约对象：

```python
from abc import ABC, abstractmethod

class Specification(ABC):
    """规约基类"""
    
    @abstractmethod
    def is_satisfied_by(self, candidate) -> bool: ...
    
    def __and__(self, other: 'Specification') -> 'AndSpecification':
        return AndSpecification(self, other)
    
    def __or__(self, other: 'Specification') -> 'OrSpecification':
        return OrSpecification(self, other)
    
    def __invert__(self) -> 'NotSpecification':
        return NotSpecification(self)


class AndSpecification(Specification):
    def __init__(self, left: Specification, right: Specification):
        self._left, self._right = left, right
    
    def is_satisfied_by(self, candidate) -> bool:
        return self._left.is_satisfied_by(candidate) and self._right.is_satisfied_by(candidate)


class OrSpecification(Specification):
    def __init__(self, left: Specification, right: Specification):
        self._left, self._right = left, right
    
    def is_satisfied_by(self, candidate) -> bool:
        return self._left.is_satisfied_by(candidate) or self._right.is_satisfied_by(candidate)


class NotSpecification(Specification):
    def __init__(self, spec: Specification):
        self._spec = spec
    
    def is_satisfied_by(self, candidate) -> bool:
        return not self._spec.is_satisfied_by(candidate)


# 具体业务规约
class IsVIPCustomer(Specification):
    def is_satisfied_by(self, customer: Customer) -> bool:
        return customer.tier == CustomerTier.VIP

class HasActiveSubscription(Specification):
    def is_satisfied_by(self, customer: Customer) -> bool:
        return customer.subscription is not None and customer.subscription.is_active

class HasSpentMoreThan(Specification):
    def __init__(self, amount: Money):
        self._threshold = amount
    
    def is_satisfied_by(self, customer: Customer) -> bool:
        return customer.total_spent.amount >= self._threshold.amount


# 组合规约——读起来像业务规则文档
eligible_for_premium = (
    IsVIPCustomer() | HasActiveSubscription()
) & HasSpentMoreThan(Money(Decimal("10000"), "CNY"))

# 测试
def test_premium_eligibility_requires_spending_threshold():
    vip_low_spend = Customer(tier=CustomerTier.VIP, total_spent=Money(Decimal("100"), "CNY"))
    assert not eligible_for_premium.is_satisfied_by(vip_low_spend)
    
    vip_high_spend = Customer(tier=CustomerTier.VIP, total_spent=Money(Decimal("15000"), "CNY"))
    assert eligible_for_premium.is_satisfied_by(vip_high_spend)
```

---

## 20.5 函数式领域建模

用函数式思想让领域层更纯粹：

```python
# 函数式风格：不可变数据 + 纯函数
from typing import NamedTuple, Tuple

class Order(NamedTuple):  # 完全不可变
    id: str
    status: str
    items: tuple  # tuple 是不可变的
    total: Decimal

# 纯函数（输入确定，输出确定，无副作用）
def confirm_order(order: Order) -> Tuple[Order, List[DomainEvent]]:
    """返回新的订单状态和产生的事件，不修改原对象"""
    if order.status != "draft":
        raise InvalidOrderStateError()
    
    confirmed_order = Order(
        id=order.id,
        status="confirmed",  # 新对象，原对象不变
        items=order.items,
        total=order.total
    )
    events = [OrderConfirmed(order_id=order.id)]
    return confirmed_order, events

# 测试：纯函数极易测试
def test_confirm_order_returns_new_order_with_confirmed_status():
    original = Order(id="o1", status="draft", items=(), total=Decimal("100"))
    
    confirmed, events = confirm_order(original)
    
    assert confirmed.status == "confirmed"
    assert original.status == "draft"  # 原对象不变！
    assert len(events) == 1
    assert isinstance(events[0], OrderConfirmed)
```

---

## 20.6 AI 生成领域模型

这是 Vibe Coding 的未来方向——用 AI 从业务描述直接生成 DDD 代码：

### 当前最佳实践

```python
# 提示词模板：从业务描述生成 DDD 骨架

DOMAIN_MODEL_PROMPT = """
## 任务
根据以下业务描述，生成 Python DDD 领域模型代码骨架。

## 业务描述
{business_description}

## 要求
1. 识别并列出所有领域概念（值对象、实体、聚合根）
2. 识别并列出所有领域事件
3. 生成 Python dataclass 代码
4. 值对象用 frozen=True
5. 聚合根继承 AggregateRoot 基类
6. 每个业务行为作为方法，带类型注解
7. 生成对应的测试骨架（不需要实现，只需要测试名）

## 输出格式
先输出"领域词汇表"，再输出"代码"，再输出"测试骨架"。
"""

# 使用
from anthropic import Anthropic
client = Anthropic()

def generate_domain_model(business_description: str) -> str:
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4000,
        messages=[{
            "role": "user",
            "content": DOMAIN_MODEL_PROMPT.format(
                business_description=business_description
            )
        }]
    )
    return response.content[0].text

# 示例
model_code = generate_domain_model("""
在线图书馆系统：
- 读者可以借阅图书，每次最多5本
- 借阅期限为14天，可续借一次
- 超期未还每天罚款1元，最多30元
- 所有副本都被借走时，读者可以预约
""")
```

### AI 辅助的 TDD 循环

```python
# 未来的 Vibe Coding 工作流

class VibeCodingWorkflow:
    """AI 辅助的 TDD+DDD 工作流"""
    
    def __init__(self, ai_client):
        self._ai = ai_client
    
    def from_user_story(self, story: str) -> 'VibeCodingSession':
        # Step 1: AI 生成统一语言词汇表
        vocabulary = self._ai.extract_vocabulary(story)
        
        # Step 2: AI 生成领域模型骨架
        domain_skeleton = self._ai.generate_domain_skeleton(vocabulary)
        
        # Step 3: AI 生成测试骨架
        test_skeleton = self._ai.generate_test_skeleton(domain_skeleton)
        
        return VibeCodingSession(
            vocabulary=vocabulary,
            domain=domain_skeleton,
            tests=test_skeleton
        )
    
    # 你审查后，AI 帮你填充实现
    def implement(self, session: 'VibeCodingSession') -> str:
        return self._ai.implement_from_tests(
            tests=session.tests,
            domain=session.domain,
            vocabulary=session.vocabulary
        )
```

---

## 20.7 Vibe Coding 的未来

### 已落地的技术（2024-2025）

1. **MCP（Model Context Protocol）**：让 AI 工具直接访问项目上下文（代码、文档、数据库 schema），无需手动粘贴
2. **结构化输出（Structured Output）**：用 JSON Schema 约束 LLM 输出，确保生成的代码符合类型系统
3. **Claude Code / Cursor**：AI 原生的集成开发环境，支持多文件编辑、自动测试、上下文感知
4. **AI 辅助架构决策记录（ADR）**：让 AI 帮助生成和维护架构决策文档，保持设计意图的可追溯性

### 发展中的趋势

1. **AI 生成测试**：从需求文档自动生成测试用例和属性测试
2. **AI 代码审查**：自动检测 DDD 违规（上下文边界泄漏等），本地 LLM（如 Ollama）可保证代码隐私
3. **自然语言到 DSL**：直接从自然语言生成业务规则 DSL，降低非程序员的使用门槛
4. **实时反馈**：IDE 实时提示 DDD 最佳实践，与领域词汇表联动

### 不变的核心

无论技术如何发展，这些核心原则不会改变：

```
1. 意图先于实现
   → 先定义"完成"，再实现

2. 领域语言是设计工具
   → 词汇准确，设计自然

3. 可验证的终态
   → TDD 的红绿灯永远是质量的基石

4. 持续演化
   → 没有一次设计到位的系统，只有持续改进的系统
```

### 给读者的建议

```
初学者路径（3个月）：
  月1：掌握 TDD——每天30分钟练习 Red-Green-Refactor
  月2：掌握 DDD 战术——用 dataclass 实现值对象和聚合根
  月3：掌握内部 DSL——为你的项目设计一个流式接口

进阶路径（6个月后）：
  - 在实际项目中应用 DDD 战略设计
  - 构建一个外部 DSL 并用于生产规则配置
  - 尝试事件溯源架构

专家路径：
  - 研究函数式领域建模（Haskell/Elm 的思想应用到 Python）
  - 探索 LLM 辅助的 DDD 建模
  - 为你的领域构建完整的 DSL 生态
```

---

## 20.8 在遗留代码中渐进引入

大多数读者面对的不是全新项目，而是已有的代码库。以下是渐进引入 Vibe Coding 方法论的策略：

### 绞杀者模式（Strangler Fig Pattern）

不要试图一次性重写，而是逐步用新代码包围旧代码：

1. **从一个限界上下文开始**：选择变更最频繁、业务价值最高的模块
2. **写特征测试（Characterization Tests）**：先为现有行为写测试，固定当前行为
3. **用防腐层隔离**：新代码通过 ACL 调用旧代码，互不污染
4. **每次只迁移一个聚合**：将旧模块中的一个核心概念用 DDD 重建
5. **逐步切换流量**：新旧实现并行运行，验证一致后切换

### 实施路径

```
月 1：选定目标上下文，写特征测试覆盖现有行为
月 2：建立统一语言词汇表，定义新的领域模型
月 3：用 TDD 实现新聚合，通过 ACL 桥接旧代码
月 4：将 DSL 引入配置/规则层，逐步替换硬编码逻辑
```

### 关键原则

- **不破坏现有功能**：特征测试是安全网
- **小步前进**：每次 PR 只迁移一个概念
- **AI 辅助**：让 AI 帮你分析旧代码的隐含业务规则

---

## 20.9 自我评估清单

完成本教程后，你应该能够：

```
TDD 能力：
□ 在写任何实现之前先写测试
□ 控制步长：每次只有一个红灯
□ 识别"难以测试"的设计并重构
□ 区分单元测试、集成测试、验收测试

DDD 能力：
□ 建立统一语言词汇表
□ 识别限界上下文边界
□ 正确区分值对象、实体、聚合根
□ 设计领域事件和 Saga

DSL 能力：
□ 用方法链设计内部 DSL
□ 用 Lark 构建简单的外部 DSL
□ 用 DSL 表达业务规则，让非程序员可读

Vibe Coding 能力：
□ 给 AI 提供测试+领域模型作为精确提示
□ 审查 AI 生成的代码（功能+设计+安全）
□ 在保持心流的同时保证代码质量
```

---

## 结语

Vibe Coding 不是放弃工程纪律，而是**用精确的工程纪律解放创造力**。

当你掌握了 TDD 的节奏、DDD 的语言、DSL 的表达力，你会发现：
- 与 AI 的协作更精准——测试覆盖率可量化，上下文表达更清晰
- 代码的可读性显著提升，团队沟通成本明显降低
- 重构不再恐惧，因为测试是你的安全网

**开始你的下一个项目，用 Vibe Coding 的方式。**

```python
# 从这里开始

def test_my_first_vibe_coding_session():
    """
    我的第一个 Vibe Coding 测试
    业务场景：[写下你的场景]
    """
    # 用领域语言建模
    # 先写测试，再写实现
    # 让 AI 帮你完成中间步骤
    pass  # 改成你的第一个测试 🚀
```

---

**全教程完结**

回到首页：[README](../README.md)
