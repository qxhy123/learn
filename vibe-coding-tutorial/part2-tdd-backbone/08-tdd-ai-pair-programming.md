# 第8章：TDD 与 AI 结对编程

> "AI 是最不知疲倦的结对编程伙伴，但它需要你提供清晰的规约。"

---

## 8.1 为什么 TDD 让 AI 输出更好

### 没有 TDD 时与 AI 的对话

```
你：帮我实现一个订单折扣系统

AI：好的，这是订单折扣系统：
[生成了200行代码]
[包含了你不需要的功能]
[命名和你的项目不一致]
[可能有微妙的 bug]
```

**问题**：AI 不知道：
- 你的系统边界在哪里
- 你的领域模型是什么
- 什么情况算"完成"

### 有了 TDD 后与 AI 的对话

```
你：根据这些测试实现折扣计算器：

[贴上10行清晰的测试代码]

AI：根据测试规约，我实现如下：
[生成了精确满足测试的代码]
[命名与测试一致]
[包含测试覆盖的所有边界情况]
```

**为什么更好**：
- 测试就是规约，AI 有了明确目标
- 测试中的命名成为 AI 使用的词汇
- 测试用例覆盖了所有边界，AI 不会遗漏

---

## 8.2 TDD+AI 的协作模式

### 模式1：你写测试，AI 写实现（最常用）

```python
# 你写（Red 阶段）：
class TestPricingEngine:
    
    def test_base_price_without_discount(self):
        engine = PricingEngine()
        result = engine.calculate(
            base_price=Money(100, "CNY"),
            customer=Customer(tier=CustomerTier.NORMAL),
            promotions=[]
        )
        assert result.final_price == Money(100, "CNY")
        assert result.discount_applied == Money(0, "CNY")
    
    def test_vip_discount_10_percent(self):
        engine = PricingEngine()
        result = engine.calculate(
            base_price=Money(100, "CNY"),
            customer=Customer(tier=CustomerTier.VIP),
            promotions=[]
        )
        assert result.final_price == Money(90, "CNY")
        assert result.discount_applied == Money(10, "CNY")
    
    def test_promotion_stacks_with_tier_discount(self):
        engine = PricingEngine()
        promo = PercentagePromotion(rate=Decimal("0.05"))
        result = engine.calculate(
            base_price=Money(100, "CNY"),
            customer=Customer(tier=CustomerTier.VIP),
            promotions=[promo]
        )
        # VIP 10% + 促销 5% = 先VIP后促销
        # 100 * 0.9 = 90, 90 * 0.95 = 85.5
        assert result.final_price == Money(85.5, "CNY")

# 你对 AI 说：
"""
根据上面的测试，实现 PricingEngine、PricingResult、
PercentagePromotion 类。

约束：
- Money 是不可变值对象，amount: float, currency: str
- Customer 有 tier: CustomerTier 属性
- CustomerTier 是枚举：NORMAL, VIP, PREMIUM
- PREMIUM 享受 20% 折扣
- 折扣先应用等级折扣，再应用促销
"""
```

### 模式2：AI 建议测试，你审查

```
你：我要实现一个"库存预警"功能：
    当商品库存低于预警线时，触发补货提醒。
    
    请先生成10个测试用例（不要实现），覆盖：
    - 正常情况
    - 边界情况  
    - 异常情况

AI：[生成10个测试用例]

你：[审查，修改不准确的测试，删除多余的，添加遗漏的]

然后：根据我确认的这些测试，请实现代码。
```

### 模式3：AI 重构，你验证

```
你：这是当前通过测试的代码：

[贴代码]

这是所有测试：

[贴测试]

请重构这段代码，提升可读性和设计质量。
要求：所有测试必须仍然通过。
禁止：删除或修改任何测试。
```

---

## 8.3 高效的 AI 提示词结构

### 黄金模板：CTRI（Context-Tests-Requirements-Implementation）

```markdown
## Context（上下文）
我在构建电商系统的「定价模块」（限界上下文）。

已有的领域对象：
```python
@dataclass(frozen=True)
class Money:
    amount: Decimal
    currency: str

class CustomerTier(Enum):
    NORMAL = "normal"
    VIP = "vip"
    PREMIUM = "premium"
```

## Tests（测试规约——你已经写好的）
```python
class TestDiscountService:
    def test_vip_gets_10_percent(): ...
    def test_premium_gets_20_percent(): ...
    def test_discount_cannot_make_price_negative(): ...
```

## Requirements（额外要求）
1. 返回类型必须是 DiscountResult，包含 original_price, discount_amount, final_price
2. 支持将来扩展新的等级（不要 if-elif 链）
3. 使用 Python dataclass
4. 不需要持久化

## Implementation（让 AI 实现）
请实现满足上述测试和要求的代码。
```

### 迭代提示词（当 AI 输出不满意时）

```markdown
你的实现有以下问题：

1. [具体问题1]：你用了 if-elif 链，请改用策略模式
2. [具体问题2]：DiscountResult 缺少 discount_rate 字段（测试 line 23 需要）
3. [具体问题3]：Money 的加法需要处理不同货币的情况

请修复这三个问题，其他部分保持不变。
```

---

## 8.4 AI 生成代码的审查清单

**不要盲目接受 AI 的输出！** 每次都运行这个检查清单：

```markdown
## AI 代码审查清单

### 功能正确性
- [ ] 所有测试通过（运行 pytest）
- [ ] 边界情况处理正确
- [ ] 异常情况有明确的错误类型

### 领域一致性（DDD）
- [ ] 命名与统一语言词汇表一致
- [ ] 没有引入新的"技术词汇"污染领域层
- [ ] 聚合根的不变量被维护

### 设计质量
- [ ] 没有全局状态
- [ ] 依赖通过构造函数注入
- [ ] 方法长度合理（< 20行）
- [ ] 圈复杂度可接受

### 安全性
- [ ] 没有 SQL 注入漏洞（如果有数据库操作）
- [ ] 输入验证在边界处完成
- [ ] 没有敏感信息硬编码

### 可测试性
- [ ] 新代码已有测试覆盖
- [ ] 没有引入难以测试的耦合
```

---

## 8.5 典型 AI 错误模式与对策

### 错误1：AI 过度设计

```python
# AI 给你一个"大而全"的实现
class UniversalDiscountEngine:
    """
    支持：
    - 15种折扣类型
    - 插件系统
    - 配置驱动
    - 国际化
    - 缓存层
    ...
    """
# 你只需要一个简单的 VIP 折扣！

# 对策：在提示词中明确限制
"""
只实现满足这5个测试所需的最小代码。
不要添加任何测试没有覆盖的功能。
不要使用插件系统、工厂模式等复杂设计，除非测试需要。
"""
```

### 错误2：AI 使用错误的抽象层次

```python
# 你要的是领域层的纯 Python 代码
# AI 却给了你框架耦合的代码
class Order(db.Model):  # SQLAlchemy 模型！
    id = db.Column(db.Integer, primary_key=True)
    
# 对策：在提示词中明确说明
"""
这是领域层代码。
禁止导入任何框架（Flask/Django/SQLAlchemy/FastAPI）。
只使用 Python 标准库和 pydantic。
"""
```

### 错误3：AI 修改了测试

```python
# 你让 AI 修复 bug，它却改了测试来让测试通过
def test_discount_cannot_exceed_50_percent():
    # AI 把这个测试改成了
    # assert result >= 0  # 更宽松的条件，测试"通过"了但规约变了

# 对策：明确声明
"""
禁止修改任何测试代码。
如果你认为测试有误，请告诉我，不要自行修改。
"""
```

### 错误4：AI 忽略异常情况

```python
# AI 的实现
def reserve_inventory(product_id: str, qty: int) -> bool:
    # AI 只实现了成功情况，没有错误处理

# 测试补充后，提示词要改进：
"""
请特别注意这些测试中的异常情况测试：
- test_insufficient_inventory_raises_error
- test_product_not_found_raises_error
确保异常路径和正常路径都正确实现。
"""
```

---

## 8.6 TDD 驱动的 AI 协作工作流

### 完整工作流示例：实现购物车功能

**Step 1：建立领域模型（你主导）**
```python
# 你先定义领域对象骨架
@dataclass
class CartItem:
    product_id: str
    quantity: int
    unit_price: Money

@dataclass
class Cart:
    customer_id: str
    items: list[CartItem] = field(default_factory=list)
```

**Step 2：写测试规约（你主导）**
```python
class TestCart:
    def test_add_item_to_empty_cart(self): ...
    def test_add_same_item_increases_quantity(self): ...
    def test_remove_item_from_cart(self): ...
    def test_cart_total_sums_all_items(self): ...
    def test_apply_coupon_to_cart(self): ...
    def test_cart_checkout_creates_order(self): ...
```

**Step 3：AI 实现（AI 主导）**
```
提示：根据上面的 CartItem、Cart 骨架和6个测试，
实现 Cart 的所有方法。
```

**Step 4：你验证（你主导）**
```bash
pytest tests/unit/domain/test_cart.py -v
# 检查所有6个测试是否通过
# 审查 AI 的实现是否符合设计原则
```

**Step 5：DSL 封装（你主导，AI 辅助）**
```python
# 你设计 DSL 接口
# AI 实现细节

cart_dsl = (
    Cart.for_customer("alice")
        .add("book", qty=2, price=Money(29.9, "CNY"))
        .add("pen", qty=5, price=Money(3.5, "CNY"))
        .apply_coupon("SUMMER10")
        .checkout()
)
```

---

## 8.7 提升 AI 协作效率的 10 个技巧

1. **测试即提示**：把完整测试代码直接贴入提示词，比描述需求更精确

2. **小批次迭代**：每次只让 AI 实现 2-3 个测试，而不是全部

3. **领域词汇表前置**：提示词开头先给出领域词汇表

4. **明确禁止清单**：说明不要什么比说明要什么更有效

5. **要求 AI 解释**：让 AI 解释为什么这样设计，发现问题

6. **版本化提示词**：好用的提示词保存到 `prompts/` 目录

7. **测试失败截图**：把 pytest 失败输出贴给 AI，让它修复

8. **分步确认**：复杂任务分步执行，每步确认后再继续

9. **代码审查后提示**：发现问题后，用具体代码行指出，让 AI 修复

10. **让 AI 写测试**：对不确定的需求，先让 AI 提建议，你审查后再实现

---

## 8.8 AI 协作的边界

**AI 擅长的**：
- 根据测试生成满足规约的实现
- 重构现有代码提升可读性
- 生成样板代码（Repository、DTO 等）
- 将现有逻辑翻译成不同风格（如转换为 DSL）

**你必须主导的**：
- 领域建模（只有你理解业务）
- 测试规约（什么是"完成"）
- 架构决策（分层、边界、模式）
- 审查和验证（AI 的输出不总是对的）

---

## 总结

TDD 是与 AI 协作的最佳协议：
- **测试 = 需求规约**：AI 有了明确目标
- **绿灯 = 完成标准**：不需要主观判断"够了"
- **Red-Green-Refactor = 迭代节奏**：AI 参与生成，你保持控制

这是 Vibe Coding 的本质：你定义意图（测试），AI 完成实现，你验证结果。

---

**下一章**：[统一语言实战](../part3-ddd-modeling/09-ubiquitous-language.md)
