# 第十五章：TDD 最佳实践与反模式

## 15.1 TDD 的三种层次

```
Level 1：测试后置（Test-Last）
  └── 先写代码，再补测试
  └── 测试被实现绑架，缺乏设计驱动力

Level 2：测试先行（Test-First）
  └── 先写测试，再写实现
  └── 有 Red-Green-Refactor 循环

Level 3：测试驱动设计（Test-Driven Design）
  └── 测试揭示设计问题
  └── "测试难写"意味着"设计有问题"
  └── 以测试倒逼接口解耦
```

---

## 15.2 黄金规则

### 规则一：不要让测试通过的代码超过必要

```python
# 第一个测试
def test_returns_zero_for_empty(self):
    self.assertEqual(sum_list([]), 0)

# Green 阶段：只写到让测试通过
def sum_list(lst):
    return 0   # 不要"聪明地"直接写 return sum(lst)

# 第二个测试迫使你实现真正的逻辑
def test_sums_single_element(self):
    self.assertEqual(sum_list([5]), 5)
```

### 规则二：重构只在 Green 状态下进行

```
Red → Green → [Refactor only here] → Red → Green → ...
```

Never refactor while tests are red.

### 规则三：测试失败时，只写让测试通过的最少代码

```python
# 不要这样：测试只要求正整数，你却实现了所有数字
def test_square_of_three(self):
    self.assertEqual(square(3), 9)

# Green 阶段的诱惑：
def square(n):
    return n ** 2   # 这没问题，但你可能跳过了中间步骤
    
# 极端 TDD 会先写：
def square(n):
    return 9   # 只让当前测试通过，等下一个测试迫使泛化
```

---

## 15.3 测试命名的艺术

```python
# 反模式：描述实现
def test_for_loop_iterates_items(self): ...
def test_if_branch_returns_true(self): ...

# 反模式：测试方法名
def test_calculate(self): ...
def test_process(self): ...

# 最佳实践：描述行为（Given-When-Then 风格）
def test_total_is_zero_when_cart_is_empty(self): ...
def test_checkout_fails_when_payment_is_declined(self): ...
def test_discount_applied_when_coupon_is_valid(self): ...
```

**口诀**：测试名应该能回答"什么情况下（When），什么结果（Then）"。

---

## 15.4 测试结构：AAA 模式

每个测试遵循 **Arrange-Act-Assert** 结构：

```python
def test_apply_discount_reduces_total(self):
    # Arrange（准备）
    cart = ShoppingCart()
    cart.add_item("laptop", price=1000.0)
    coupon = Coupon(code="SAVE10", discount=0.10)

    # Act（执行）
    cart.apply_coupon(coupon)
    total = cart.total()

    # Assert（验证）
    self.assertAlmostEqual(total, 900.0)
```

**规范**：
- Arrange 和 Assert 之间用空行分隔
- Act 通常只有一行（被测的那一个动作）
- 如果 Arrange 很长，考虑提取到 setUp 或辅助方法

---

## 15.5 常见反模式

### 反模式一：测试过多断言（Giant Test）

```python
# 坏：一个测试验证太多事情
def test_user_registration(self):
    user = register("Alice", "alice@ex.com", "Passw0rd!")
    self.assertIsNotNone(user.id)
    self.assertEqual(user.name, "Alice")
    self.assertEqual(user.email, "alice@ex.com")
    self.assertTrue(user.is_active)
    self.assertIsNone(user.last_login)
    self.assertIsNotNone(user.created_at)
    self.assertEqual(user.role, "user")
    # 失败时不知道是哪个断言出问题

# 好：拆分为多个独立测试
def test_registration_assigns_id(self): ...
def test_registration_stores_name(self): ...
def test_registration_defaults_to_active(self): ...
def test_registration_sets_role_to_user(self): ...
```

### 反模式二：测试实现细节（White-Box Coupling）

```python
# 坏：测试私有方法/内部实现
def test_internal_cache_hit(self):
    service = UserService()
    service._cache["user_1"] = fake_user   # 直接操纵私有状态
    result = service.get_user(1)
    self.assertEqual(result, fake_user)

# 好：测试公共行为
def test_second_call_returns_same_user(self):
    service = UserService()
    first = service.get_user(1)
    second = service.get_user(1)
    self.assertEqual(first.id, second.id)
    # 不关心缓存是否命中，只关心行为一致
```

### 反模式三：测试间依赖（Interdependent Tests）

```python
# 坏：测试依赖顺序
class TestUserFlow(unittest.TestCase):
    created_user_id = None  # 类级别共享状态！

    def test_1_create_user(self):
        user = create_user("Alice")
        TestUserFlow.created_user_id = user.id  # 共享状态

    def test_2_get_user(self):
        user = get_user(TestUserFlow.created_user_id)  # 依赖 test_1
        self.assertEqual(user.name, "Alice")

# 好：每个测试独立
class TestUserOperations(unittest.TestCase):
    def setUp(self):
        self.user = create_user("Alice")  # 每个测试自己创建

    def test_get_user_by_id(self):
        found = get_user(self.user.id)
        self.assertEqual(found.name, "Alice")
```

### 反模式四：逻辑测试（Logic in Tests）

```python
# 坏：测试中有条件逻辑
def test_discount_logic(self):
    for price in [10, 50, 100]:
        if price > 50:
            expected = price * 0.9
        else:
            expected = price
        result = apply_discount(price)
        self.assertAlmostEqual(result, expected)
    # 测试逻辑和生产逻辑可能有同样的 bug！

# 好：用固定期望值
DISCOUNT_CASES = [
    (10,  10.0),   # 低价：无折扣
    (50,  50.0),   # 边界：无折扣
    (51,  45.9),   # 超过阈值：9折
    (100, 90.0),
]

def test_discount_applied_correctly(self):
    for price, expected in self.DISCOUNT_CASES:
        with self.subTest(price=price):
            self.assertAlmostEqual(apply_discount(price), expected, places=2)
```

### 反模式五：Mock 过度（Over-Mocking）

```python
# 坏：连纯函数都 Mock
def test_total_calculation(self):
    with patch('myapp.cart.sum') as mock_sum:   # Mock 了内置 sum！
        mock_sum.return_value = 100
        cart = ShoppingCart()
        total = cart.total()
    self.assertEqual(total, 100)
    # 测试完全脱离现实

# 好：只 Mock 真正的外部依赖
def test_total_with_tax_service(self):
    mock_tax = MagicMock()
    mock_tax.get_rate.return_value = 0.1
    cart = ShoppingCart(tax_service=mock_tax)
    cart.add_item("laptop", 1000)
    self.assertAlmostEqual(cart.total_with_tax(), 1100.0)
```

---

## 15.6 TDD 与设计的关系

### 难以测试 = 设计问题的信号

| 测试中的痛苦 | 设计问题 | 解决方案 |
|-------------|---------|---------|
| 需要大量 Mock | 依赖太多 | 依赖注入，减少耦合 |
| setUp 很复杂 | 对象构建困难 | Builder 模式，默认值 |
| 测试运行很慢 | 依赖了 I/O | 分层，隔离 I/O |
| 很难给依赖打桩 | 硬编码依赖 | 接口抽象，依赖注入 |
| 一个改动破坏多个测试 | 重复逻辑 | 消除重复，单一职责 |

### 依赖注入让代码可测

```python
# 难以测试：硬编码依赖
class OrderService:
    def __init__(self):
        self.db = DatabaseConnection("prod_host")  # 无法替换
        self.mailer = SMTPMailer("smtp.example.com")  # 无法替换

# 容易测试：依赖注入
class OrderService:
    def __init__(self, db, mailer):   # 依赖从外部注入
        self.db = db
        self.mailer = mailer

# 测试时注入 Mock/Fake
service = OrderService(db=FakeDB(), mailer=MagicMock())
```

---

## 15.7 TDD 在遗留代码中的应用

**策略：绞杀者模式（Strangler Fig Pattern）**

```
1. 不要试图一次性重写遗留代码
2. 在遗留代码周围建立测试网（characterization tests）
3. 对新功能/修复使用 TDD
4. 逐渐用有测试的新代码替换遗留代码
```

### 特征测试（Characterization Tests）

```python
# 记录现有行为（即使行为是"错的"）
def test_legacy_format_output(self):
    """记录遗留函数的实际输出，以便重构时检测回归"""
    result = legacy_format_date("2024-01-15")
    # 遗留函数有 bug：月/日颠倒了，但我们先记录这个行为
    self.assertEqual(result, "15/01/2024")
    # 重构后如果输出变了，这个测试会提醒我们
```

---

## 15.8 TDD 节奏与实践习惯

### 红-绿-重构的时间控制

```
理想节奏：每轮 5-15 分钟
├── Red：1-3 分钟（写一个小的失败测试）
├── Green：2-5 分钟（最少代码通过）
└── Refactor：2-7 分钟（清理，保持绿色）
```

**测试太难写**：暂停，重新思考设计。

**重构时间太长**：说明重构步骤太大，分解成更小的步骤。

### 提交策略

```bash
# TDD 工作流与 Git 结合
git add .
git commit -m "Red: add failing test for empty cart"

# Green
git add .
git commit -m "Green: implement cart total returns 0 when empty"

# Refactor
git add .
git commit -m "Refactor: use property for cart total"
```

---

## 15.9 高阶技巧

### 测试驱动出接口

```python
# 先写测试，让接口"浮现"
def test_notify_user_of_order(self):
    notifier = MagicMock()   # 这个 MagicMock 定义了 notifier 的接口
    service = OrderService(notifier=notifier)
    service.place_order(user_id=1, items=["apple"])
    notifier.send.assert_called_once()   # notifier 需要有 send 方法

# 从测试中提取接口定义
class Notifier(Protocol):
    def send(self, user_id: int, message: str) -> None: ...
```

### 三角法（Triangulation）

用多个测试用例来消除"巧合通过"：

```python
def test_double_1(self):
    self.assertEqual(double(1), 2)
# 此时可以用 return 2 通过

def test_double_2(self):
    self.assertEqual(double(2), 4)
# 此时可以用 return n*2 通过，但也可以用 if n==1: return 2; return 4

def test_double_3(self):
    self.assertEqual(double(5), 10)
# 现在必须实现真正的 return n * 2
```

---

## 15.10 TDD 检查清单

### 写测试前
- [ ] 我知道这个测试要验证什么行为吗？
- [ ] 这个测试独立于其他测试吗？
- [ ] 我能用一句话描述这个测试吗？

### 写完测试后（Red 阶段）
- [ ] 测试确实失败了吗？
- [ ] 失败原因是"功能未实现"而非"测试本身有 bug"吗？
- [ ] 失败信息清晰吗？

### Green 阶段
- [ ] 我写了最少的代码让测试通过吗？
- [ ] 我没有引入未测试的功能吗？

### Refactor 阶段
- [ ] 所有测试仍然通过吗？
- [ ] 消除了重复代码吗？
- [ ] 命名是否清晰表达意图？

---

## 15.11 本章小结

**核心原则**：
1. Red-Green-Refactor 是纪律，不是建议
2. 测试验证行为，不是实现细节
3. 难以测试的代码是设计问题的信号
4. 每个测试独立、快速、自我验证

**主要反模式**：
- Giant Test（过多断言）
- White-Box Coupling（测试私有实现）
- Interdependent Tests（测试间依赖）
- Logic in Tests（测试中有条件逻辑）
- Over-Mocking（过度 Mock）

**高阶实践**：
- 依赖注入是可测试性的基础
- 测试驱动出接口，而不是从实现提取接口
- 在遗留代码上先建立特征测试，再安全重构

---

## 结语

TDD 不仅是测试技术，它是一种**思维方式**：在实现之前先定义"什么是正确的"。

坚持这个习惯，你写出的代码会：
- 接口更清晰（被测试倒逼出来的）
- 耦合更低（难以测试=警告信号）
- 回归更少（每次重构都有安全网）
- 文档更准确（测试就是活文档）

**Practice, practice, practice.** TDD 是一门手艺，需要刻意练习。
