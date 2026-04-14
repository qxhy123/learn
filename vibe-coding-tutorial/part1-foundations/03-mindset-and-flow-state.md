# 第3章：心流状态与编程节奏

> "心流不是偶然发生的，它是设计出来的。"

---

## 3.1 什么是心流（Flow State）

心理学家 Mihaly Csikszentmihalyi 在 1975 年提出"心流"概念：当任务难度和个人能力完美匹配时，人会进入一种高度专注、高效且愉悦的状态。

```
         高
          │
          │              ╔════════╗
          │         焦虑 ║        ║ 心流
     挑   │              ║        ║
     战   │         ╔════╝        ╚════╗
     难   │         ║                  ║
     度   │   ╔═════╝  无聊             ╚═══╗
          │   ║                              ║
         低│   ╚══════════════════════════════╝
          └────────────────────────────────────
          低              能力                高
```

**编程中的心流障碍**：
- 需求不清晰（不知道要做什么）
- 没有反馈循环（不知道做对了没有）
- 上下文切换太频繁
- 陷入细节（为了一个 bug 调试2小时）

**TDD + DDD + DSL 如何解决这些障碍**：

| 障碍 | 解决方案 | 机制 |
|------|----------|------|
| 需求不清晰 | DDD 统一语言 | 先建模，先命名 |
| 没有反馈 | TDD 红绿循环 | 每几分钟一次验证 |
| 上下文切换 | DSL 封装 | 把细节藏在 DSL 后面 |
| 陷入细节 | TDD 步长控制 | 小步前进 |

---

## 3.2 TDD 创造编程节奏

### 节拍器比喻

TDD 的 Red-Green-Refactor 循环就像音乐的节拍器：
- **Red**（写测试）：规划下一小步
- **Green**（使测试通过）：专注实现
- **Refactor**（重构）：清理和优化

```
♩  ♩  ♩  ♩   ←  节拍：每个周期 3-10 分钟

Red → Green → Refactor → Red → Green → Refactor
```

### 步长控制

**太大的步**：
```python
# 一次写了50行实现，才发现测试还没过 → 失去节奏
def process_entire_order_workflow(order_id):
    # 50行代码...
    pass
```

**适当的步**：
```python
# 步骤1：先让最简单的情况通过
def test_empty_order_total_is_zero():
    order = Order([])
    assert order.total == Money(0, "CNY")

# 实现：
class Order:
    def __init__(self, items):
        self.items = items
    
    @property
    def total(self):
        return Money(0, "CNY")  # 先让测试通过！

# 步骤2：增加一个商品的情况
def test_single_item_order_total():
    order = Order([OrderItem(price=Money(10, "CNY"), qty=2)])
    assert order.total == Money(20, "CNY")

# 实现演进：
@property
def total(self):
    if not self.items:
        return Money(0, "CNY")
    # Money 实现了 __radd__，使 sum() 能从默认的 0 开始累加
    return sum(item.subtotal for item in self.items)
```

---

## 3.3 DDD 减少认知负荷

### 命名的力量

糟糕的命名是心流杀手：

```python
# 这段代码读到一半就不得不去查文档
def calc(d, t, r):
    return d * (1 - t * r / 100)

# 这段代码一读就懂，不打断思路
def apply_discount(base_price: Money, customer_tier: str, discount_rate: float) -> Money:
    return base_price * (1 - discount_rate / 100)
```

### 统一语言消除歧义

真实项目中常见的命名混乱：

```python
# 同一个概念，代码中出现了5种叫法
user_id         # 在登录模块
customer_id     # 在订单模块
buyer_id        # 在支付模块
account_id      # 在积分模块
member_id       # 在会员模块
```

DDD 要求你先建立词汇表，在同一个限界上下文中统一命名：

```python
# 订单上下文的统一语言
# 词汇表：
# Customer（客户）：在我们系统中有账户的人
# CustomerId：客户的唯一标识
# CustomerTier：客户等级（NORMAL, VIP, PREMIUM）

@dataclass(frozen=True)
class CustomerId:
    value: str
    
    def __str__(self):
        return self.value

class CustomerTier(Enum):
    NORMAL = "normal"
    VIP = "vip"
    PREMIUM = "premium"
```

---

## 3.4 DSL 实现"忘记细节"

真正的心流需要能够"忘记"底层细节，专注于业务逻辑。DSL 就是这个"遗忘"工具：

### 没有 DSL：需要记住所有细节

```python
# 每次都要记住：参数顺序、类型转换、异常处理
report = Report()
report.set_start_date(datetime(2024, 1, 1))
report.set_end_date(datetime(2024, 3, 31))
report.set_group_by("region")
report.set_sort_field("revenue")
report.set_sort_order("desc")
report.set_limit(10)
report.include_subtotals = True
result = report.generate()
```

### 有了 DSL：专注业务意图

```python
# 读起来就像业务需求文档
result = (
    SalesReport()
        .for_quarter(Q1_2024)
        .by_region()
        .top(10, by=Revenue)
        .with_subtotals()
        .generate()
)
```

---

## 3.5 Vibe Coding 的心流工作法

### 五步工作循环

```
1. INTENT（意图）
   用1-2句话描述这个功能要做什么
   工具：注释/文档
   
      ↓
      
2. MODEL（建模）
   识别领域概念，建立统一语言
   工具：DDD 战术设计
   
      ↓
      
3. SPEC（规约）
   用测试描述期望行为
   工具：TDD（写 Red 测试）
   
      ↓
      
4. GENERATE（生成）
   让 AI 根据模型和测试生成实现
   工具：LLM（Claude/ChatGPT）
   
      ↓
      
5. EXPRESS（表达）
   用 DSL 封装复杂逻辑
   工具：内部 DSL 设计
   
      ↓
   （回到步骤1，处理下一个功能）
```

### 实战时间分配参考

| 步骤 | 占比 | 说明 |
|------|------|------|
| INTENT | 5% | 想清楚再动手，省后面的时间 |
| MODEL | 20% | 建模是最难最值得的投入 |
| SPEC | 25% | 写测试就是写设计文档 |
| GENERATE | 10% | AI 做的事，你负责提示和审查 |
| EXPRESS | 20% | DSL 设计需要品味和迭代 |
| 验证/调试 | 20% | TDD 保证这个不会超出控制 |

---

## 3.6 管理"心流中断"

### 常见中断和对策

**中断1：遇到一个有趣的技术问题，想深入研究**
```
对策：记录到"技术债/待研究"列表，当前循环完成后再看
工具：在测试文件旁边维护 NOTES.md
```

**中断2：AI 生成的代码不对，开始手动调试**
```
对策：如果调试超过5分钟，退回上一个绿灯状态，重新描述测试
工具：git stash + git diff
```

**中断3：需求变了，要重新设计**
```
对策：DDD 的限界上下文保护了其他模块，只改当前上下文内的测试和模型
工具：TDD 测试套件告诉你改了哪些东西
```

**中断4：发现了 bug，想顺手修**
```
对策：除非 bug 阻止当前工作，否则记录到 Issue，当前功能完成后再修
工具：TODO 注释 + issue tracker
```

---

## 3.7 Vibe Coding 的"节奏感"练习

### 练习：25分钟番茄钟协议

```
准备（2分钟）：
  - 清空工作台（关闭无关窗口）
  - 写下这个番茄钟要完成的"一件事"
  - 确认当前测试是 Red 状态

执行（20分钟）：
  - 只做写在便利贴上的那一件事
  - 遇到其他想法，写下来，不要分心
  - 保持 Red-Green-Refactor 节奏

收尾（3分钟）：
  - 确认所有测试是 Green 状态
  - 提交代码（git commit）
  - 写下下一个番茄钟要做的事
```

### 练习：三个问题检查法

每次开始新功能前，回答这三个问题：

1. **这个功能属于哪个限界上下文？**（DDD 定位）
2. **怎么知道这个功能做完了？**（TDD 终止条件）
3. **这个功能的业务逻辑如何用一行 DSL 表达？**（DSL 设计方向）

如果回答不了，说明还没准备好开始编码。

---

## 3.8 总结

心流不是偶然发生的，而是通过以下方式设计出来的：

- **TDD** 提供了持续的反馈节奏（每几分钟一次红绿切换）
- **DDD** 减少了认知负荷（清晰的领域词汇，不用记杂乱细节）
- **DSL** 实现了细节封装（忘记实现，专注意图）

当三者结合时，你会进入一种状态：清楚知道下一步做什么，做完了知道做对了，代码读起来像业务文档。

---

**下一章**：[工具链配置](04-environment-setup.md)
