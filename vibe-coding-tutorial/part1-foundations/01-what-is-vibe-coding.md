# 第1章：什么是 Vibe Coding？

> "最好的代码是你能用一句话解释清楚的代码。"

---

## 1.1 Vibe Coding 的诞生背景

2024年，Andrej Karpathy（前 Tesla AI 总监、前 OpenAI 研究科学家）提出了"Vibe Coding"这个概念：

> "There's a new kind of coding I call 'vibe coding', where you fully give in to the vibes, embrace exponentials, and forget that the code even exists."

这句话的核心不是"让 AI 写所有代码"，而是描述一种**意图驱动**的编程状态——程序员专注于**想要什么**，而不是**怎么实现它**。

### Vibe Coding 的误解

❌ **错误理解**：随便说个需求，让 AI 全部实现，自己不看代码  
✅ **正确理解**：建立一个精确的意图表达系统，让你和 AI 的协作效率最大化

---

## 1.2 传统编程 vs Vibe Coding

### 传统开发流程

```
需求 → 思考实现 → 写代码 → 调试 → 测试 → 重复
```

问题：
- 大量时间花在"怎么实现"而非"实现什么"
- AI 生成的代码缺乏上下文，需要大量修改
- 没有结构，随着项目增大，代码熵快速上升

### Vibe Coding 流程

```
意图 → 建模（DDD）→ 测试（TDD）→ 表达（DSL）→ AI 生成 → 验证
```

优势：
- 每次 AI 生成都有清晰的规约（测试）
- 领域建模保证语义一致性
- DSL 让意图直接映射到代码

---

## 1.3 为什么需要 TDD + DDD + DSL？

单独使用任何一个都有局限性：

### 只用 TDD
```python
# 测试很好，但不知道这个类代表什么业务概念
def test_process_order():
    order = Order(items=[...])
    result = order.process()
    assert result.status == "confirmed"
```
问题：测试通过了，但代码跟业务语言脱节，沟通成本高。

### 只用 DDD
```python
class OrderAggregate:
    def confirm(self) -> OrderConfirmed:
        ...
```
问题：没有测试保驾护航，重构时危险；没有 DSL，表达力有限。

### 只用 DSL
```python
order().for_customer("alice").with_items([...]).confirm()
```
问题：流畅，但没有测试和领域模型支撑，DSL 变成语法糖，内部一团乱麻。

### 三者结合

```python
# 测试（TDD）：明确终态
def test_order_confirmation_workflow():
    # Arrange - 用 DDD 语言建模
    customer = Customer.register("alice@example.com")
    order = Order.place(customer, items=[Item("book", price=29.9)])
    
    # Act - 用 DSL 表达业务流程
    result = (
        OrderWorkflow(order)
            .validate_inventory()
            .apply_discount(customer.tier)
            .confirm()
    )
    
    # Assert - 验证领域事件
    assert isinstance(result.events[0], OrderConfirmed)
    assert result.total_amount == Money(26.91, "CNY")
```

这段代码：
- **TDD**：先写测试，定义了"完成"的标准
- **DDD**：`Customer`、`Order`、`Item`、`OrderConfirmed` 是领域概念
- **DSL**：`OrderWorkflow(...).validate_inventory().apply_discount(...).confirm()` 是流式 DSL

---

## 1.4 Vibe Coding 的核心原则

### 原则一：意图优先于实现（Intent over Implementation）

在写任何代码之前，先用自然语言描述意图：

```
# 意图文档（写在测试文件顶部）
"""
业务场景：用户下单后，系统需要：
1. 验证库存
2. 根据用户等级应用折扣
3. 生成订单确认事件
4. 发送确认邮件

约束：
- 库存不足时抛出 InsufficientInventoryError
- VIP 用户享受 10% 折扣
- 所有操作在事务中完成
"""
```

然后将这个意图转化为测试，再用 AI 实现。

### 原则二：测试是规约，不是验证（Tests as Specification）

TDD 的测试不是"事后验证"，而是"事前规约"：

```python
# 这不是测试，这是规约文档
class OrderConfirmationSpec:
    """订单确认的业务规约"""
    
    def spec_vip_discount_applied(self):
        """规约：VIP 用户下单时自动应用10%折扣"""
        ...
    
    def spec_inventory_check_before_confirm(self):
        """规约：确认前必须验证库存"""
        ...
    
    def spec_event_emitted_on_success(self):
        """规约：成功后发出 OrderConfirmed 领域事件"""
        ...
```

### 原则三：语言即设计（Language is Design）

你使用的词汇决定了你的设计。如果你叫一个方法 `do_stuff()`，你根本不理解这个方法在做什么。

```python
# 坏：语言模糊，设计混乱
def do_stuff(data):
    processed = process(data)
    return handle(processed)

# 好：语言清晰，设计自然浮现
def apply_pricing_rules(order: Order) -> PricedOrder:
    discounted = apply_customer_discount(order)
    taxed = apply_tax_rules(discounted)
    return PricedOrder.from_order(taxed)
```

---

## 1.5 Vibe Coding 实战：第一个例子

让我们用一个简单例子感受完整的 Vibe Coding 节奏。

**业务场景**：实现一个简单的积分系统，用户消费后获得积分。

### 步骤1：用自然语言描述意图

```
场景：用户消费后获得积分
规则：
- 每消费 1 元获得 10 积分
- VIP 用户积分翻倍
- 积分有效期 365 天
- 积分余额不能为负
```

### 步骤2：建立领域词汇（DDD 统一语言）

```python
# 领域词汇表（先写下来，再写代码）
# Customer（客户）：有等级（normal/vip），有积分账户
# Points（积分）：有数量、有效期，不能为负
# Transaction（消费记录）：触发积分奖励的业务事件
# PointsEarned（积分获得）：领域事件，记录积分增加
```

### 步骤3：写测试（TDD Red）

```python
import pytest
from datetime import date, timedelta

class TestPointsSystem:
    
    def test_normal_customer_earns_10_points_per_yuan(self):
        """普通用户每消费1元获得10积分"""
        customer = Customer(tier="normal")
        transaction = Transaction(amount=Money(100, "CNY"))
        
        event = customer.earn_points(transaction)
        
        assert event.points == Points(1000)
    
    def test_vip_customer_earns_double_points(self):
        """VIP用户积分翻倍"""
        customer = Customer(tier="vip")
        transaction = Transaction(amount=Money(100, "CNY"))
        
        event = customer.earn_points(transaction)
        
        assert event.points == Points(2000)
    
    def test_earned_points_expire_after_365_days(self):
        """积分有效期365天"""
        customer = Customer(tier="normal")
        transaction = Transaction(amount=Money(100, "CNY"))
        
        event = customer.earn_points(transaction)
        
        assert event.expiry_date == date.today() + timedelta(days=365)
    
    def test_points_balance_cannot_be_negative(self):
        """积分余额不能为负"""
        account = PointsAccount(balance=Points(100))
        
        with pytest.raises(InsufficientPointsError):
            account.deduct(Points(200))
```

### 步骤4：让 AI 实现（提示词示范）

```
根据以下测试实现 Customer、Transaction、Points、PointsAccount 类：

[粘贴测试代码]

要求：
1. 使用 Python 数据类（dataclass）
2. Points 是值对象，不可变
3. Customer.earn_points 返回 PointsEarned 领域事件
4. 不需要持久化，只需内存实现
```

### 步骤5：设计 DSL（表达力提升）

```python
# 用流式 DSL 描述积分规则，让业务逻辑可读性极强
rules = (
    PointsRules()
        .base_rate(10)  # 每元10积分
        .for_tier("vip").multiply(2)  # VIP翻倍
        .expires_in(days=365)  # 365天有效
        .never_negative()  # 不能为负
)
```

---

## 1.6 总结

Vibe Coding 的本质是**建立精确的意图表达系统**：

1. **TDD** 提供了可验证的"完成"标准——红灯到绿灯就是完成
2. **DDD** 提供了准确的领域词汇——代码和业务说同一种语言
3. **DSL** 提供了表达力放大器——让业务逻辑直接可读

三者结合，你和 AI 的对话从：

> "帮我写一个处理订单的函数"

变成：

> "根据这些测试和领域模型，实现 `OrderWorkflow.confirm()` 方法"

后者的 AI 输出质量会高出一个数量级。

---

## 练习

1. 选择你当前项目中的一个模块，用自然语言写出它的"意图文档"
2. 识别出3个以上的领域概念，给它们取合适的名字
3. 为其中一个概念写出3个测试（不需要实现）

---

**下一章**：[TDD + DDD + DSL 三位一体概览](02-tdd-ddd-dsl-trinity.md)
