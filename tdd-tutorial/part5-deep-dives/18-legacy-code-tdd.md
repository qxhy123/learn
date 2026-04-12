# 第十八章：遗留代码的 TDD 策略

## 18.1 遗留代码的定义

Michael Feathers 在《修改代码的艺术》中给出了一个精准定义：

> **遗留代码 = 没有测试的代码**

不是"老旧"，不是"难看"，而是**缺乏测试保护**。没有测试，任何修改都是冒险的。

---

## 18.2 遗留代码的困境

```
┌─────────────────────────────────────────────────────┐
│                    遗留代码困境                       │
│                                                     │
│  想重构代码 → 需要测试保护 → 难以添加测试             │
│      ↑                              │               │
│      └──────── 因为代码不可测 ───────┘               │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 18.3 技术一：特征测试（Characterization Tests）

**目标**：在不理解代码的情况下，记录其当前行为。

```python
# 遗留代码（不要看懂，只记录行为）
def legacy_format_price(amount, currency="USD", locale="en_US"):
    if locale == "en_US":
        formatted = f"${amount:,.2f}"
    elif locale == "de_DE":
        formatted = f"{amount:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") + " €"
    else:
        formatted = str(amount)
    if currency != "USD" and locale == "en_US":
        formatted = formatted.replace("$", f"{currency} ")
    return formatted


class TestLegacyFormatPrice(unittest.TestCase):
    """特征测试：记录现有行为，不判断对错"""

    def test_us_dollar_default(self):
        # 第一步：运行并观察输出
        result = legacy_format_price(1234.5)
        # 填入实际结果（不是"期望"，是"现实"）
        self.assertEqual(result, "$1,234.50")

    def test_large_amount(self):
        result = legacy_format_price(1000000.0)
        self.assertEqual(result, "$1,000,000.00")

    def test_german_locale(self):
        result = legacy_format_price(1234.5, locale="de_DE")
        self.assertEqual(result, "1.234,50 €")

    def test_eur_in_us_locale(self):
        result = legacy_format_price(100.0, currency="EUR")
        self.assertEqual(result, "EUR 100.00")

    def test_unknown_locale(self):
        result = legacy_format_price(99.9, locale="zh_CN")
        self.assertEqual(result, "99.9")
```

**工作流**：
1. 用 `print` 或调试器观察函数输出
2. 把观察到的输出写成断言
3. 此时你不判断对错——这些测试记录的是**现状**
4. 有了这层保护网，再开始重构

---

## 18.4 技术二：接缝（Seam）

接缝是代码中**可以替换行为而不修改调用处**的地方。Michael Feathers 定义了几种接缝：

### 对象接缝（最常用）

通过依赖注入引入接缝：

```python
# 原始遗留代码：硬编码依赖
class LegacyReportGenerator:
    def generate(self, report_id):
        # 硬编码的数据库连接 ← 无法测试的接缝
        conn = psycopg2.connect("host=legacy-prod user=app")
        data = conn.execute(f"SELECT * FROM reports WHERE id={report_id}")
        
        # 硬编码的文件输出 ← 另一个无法测试的接缝
        with open(f"/var/reports/{report_id}.pdf", "wb") as f:
            f.write(self._render_pdf(data))
```

**步骤一**：引入参数，不改变默认行为（保持向后兼容）：

```python
class LegacyReportGenerator:
    def __init__(self, db_conn=None, output_dir=None):
        # 默认值保持原有行为
        self._db_conn = db_conn
        self._output_dir = output_dir or "/var/reports"

    def _get_connection(self):
        if self._db_conn:
            return self._db_conn   # 注入的连接（测试用）
        return psycopg2.connect("host=legacy-prod user=app")  # 原有行为

    def generate(self, report_id):
        conn = self._get_connection()  # 接缝！
        data = conn.execute(f"SELECT * FROM reports WHERE id={report_id}")
        path = f"{self._output_dir}/{report_id}.pdf"
        with open(path, "wb") as f:
            f.write(self._render_pdf(data))
```

**步骤二**：现在可以测试了：

```python
class TestLegacyReportGenerator(unittest.TestCase):

    def setUp(self):
        self.mock_conn = MagicMock()
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(__import__('shutil').rmtree, self.temp_dir)

        self.generator = LegacyReportGenerator(
            db_conn=self.mock_conn,
            output_dir=self.temp_dir,
        )

    def test_queries_correct_report_id(self):
        self.mock_conn.execute.return_value = [{"title": "Q1 Report"}]
        self.generator.generate(42)
        call_args = self.mock_conn.execute.call_args[0][0]
        self.assertIn("42", call_args)

    def test_output_file_created(self):
        self.mock_conn.execute.return_value = [{"title": "Test"}]
        self.generator.generate(1)
        output_path = os.path.join(self.temp_dir, "1.pdf")
        self.assertTrue(os.path.exists(output_path))
```

---

## 18.5 技术三：黄金主文件（Golden Master Testing）

当函数有复杂的输出（HTML、JSON、PDF）时，用黄金主文件：

```python
import json
import os

GOLDEN_DIR = "tests/golden"

class TestLegacyHTMLGenerator(unittest.TestCase):

    def _golden_path(self, name):
        return os.path.join(GOLDEN_DIR, f"{name}.html")

    def _assert_matches_golden(self, name, actual):
        golden_path = self._golden_path(name)

        if not os.path.exists(golden_path):
            # 首次运行：创建黄金文件
            os.makedirs(GOLDEN_DIR, exist_ok=True)
            with open(golden_path, 'w') as f:
                f.write(actual)
            self.skipTest(f"Created golden file: {golden_path}. Re-run to verify.")

        with open(golden_path) as f:
            expected = f.read()

        self.assertEqual(actual, expected,
            f"Output differs from golden file {golden_path}\n"
            f"To update: delete {golden_path} and re-run")

    def test_invoice_html_format(self):
        html = legacy_generate_invoice_html(
            order_id=1001,
            items=[("Widget", 2, 9.99), ("Gadget", 1, 49.99)],
            tax_rate=0.08,
        )
        self._assert_matches_golden("invoice_1001", html)
```

**黄金主文件工作流**：
1. 首次运行：生成并保存黄金文件（提交到 git）
2. 后续运行：对比输出与黄金文件
3. 当重构后输出合法地改变：删除黄金文件，重新运行生成新黄金文件

---

## 18.6 技术四：绞杀者模式（Strangler Fig Pattern）

逐步用新代码替换旧代码，同时保持系统运行：

```python
# 旧实现（遗留，有 bug 但在生产运行）
def legacy_calculate_discount(price, user_type, quantity):
    # 100 行复杂的 if-else 逻辑...
    if user_type == "vip":
        discount = 0.2
    elif quantity > 100:
        discount = 0.15
    # ... 更多分支
    return price * (1 - discount)


# 新实现（TDD 驱动，有完整测试）
class DiscountCalculator:
    def calculate(self, price: float, user_type: str, quantity: int) -> float:
        discount = self._get_discount_rate(user_type, quantity)
        return price * (1 - discount)

    def _get_discount_rate(self, user_type: str, quantity: int) -> float:
        if user_type == "vip":
            return 0.20
        if quantity > 100:
            return 0.15
        if quantity > 50:
            return 0.10
        return 0.0


# 过渡层：新旧接口并存
def calculate_discount(price, user_type, quantity, use_new=False):
    """Feature flag 控制新旧实现切换"""
    if use_new or os.environ.get("USE_NEW_DISCOUNT"):
        return DiscountCalculator().calculate(price, user_type, quantity)
    return legacy_calculate_discount(price, user_type, quantity)
```

**行为等价测试**（确保新旧实现行为一致）：

```python
class TestDiscountEquivalence(unittest.TestCase):
    """验证新旧实现对相同输入产生相同输出"""

    TEST_CASES = [
        ("vip",    1,   100.0),
        ("vip",    150, 200.0),
        ("normal", 1,   50.0),
        ("normal", 60,  75.0),
        ("normal", 120, 100.0),
    ]

    def test_new_matches_legacy_for_all_cases(self):
        new_calc = DiscountCalculator()
        for user_type, quantity, price in self.TEST_CASES:
            with self.subTest(user_type=user_type, quantity=quantity, price=price):
                legacy = legacy_calculate_discount(price, user_type, quantity)
                new = new_calc.calculate(price, user_type, quantity)
                self.assertAlmostEqual(legacy, new, places=6,
                    msg=f"Mismatch: legacy={legacy}, new={new}")
```

---

## 18.7 技术五：Sprout 方法（抽芽法）

在不修改遗留代码的情况下，把新逻辑长在旁边：

```python
# 遗留代码：不要动它
def legacy_process_payment(order_id, amount, card_number):
    # 200 行混乱代码...
    result = some_old_gateway.charge(card_number, amount)
    log_to_file(f"Payment {result} for order {order_id}")
    update_db(order_id, "paid")
    return result


# Sprout 方法：新增逻辑独立实现，有完整测试
def validate_payment_request(order_id: int, amount: float) -> None:
    """从遗留代码中抽出来的新验证逻辑"""
    if amount <= 0:
        raise ValueError(f"Amount must be positive: {amount}")
    if order_id <= 0:
        raise ValueError(f"Invalid order_id: {order_id}")


# 在遗留代码入口处调用新方法（最小侵入）
def legacy_process_payment(order_id, amount, card_number):
    validate_payment_request(order_id, amount)  # ← 唯一新增的一行
    # ... 原有代码不动
```

```python
class TestValidatePaymentRequest(unittest.TestCase):
    """测试抽芽出来的新方法"""

    def test_valid_request_passes(self):
        validate_payment_request(1, 100.0)  # 不抛出即通过

    def test_negative_amount_raises(self):
        with self.assertRaises(ValueError):
            validate_payment_request(1, -50.0)

    def test_zero_amount_raises(self):
        with self.assertRaises(ValueError):
            validate_payment_request(1, 0.0)

    def test_invalid_order_id_raises(self):
        with self.assertRaises(ValueError):
            validate_payment_request(-1, 100.0)
```

---

## 18.8 遗留代码 TDD 决策树

```
遇到遗留代码，需要修改或添加功能
│
├── 代码有测试？
│   ├── YES → 直接 TDD 方式添加/修改
│   └── NO ↓
│
├── 先写特征测试（记录现有行为）
│   │
│   ├── 代码可以依赖注入？
│   │   ├── YES → 引入对象接缝，然后 TDD
│   │   └── NO → 添加参数默认值引入接缝
│   │
│   ├── 输出复杂（HTML/PDF）？
│   │   └── 黄金主文件测试
│   │
│   └── 需要添加新功能？
│       ├── 逻辑可以独立 → Sprout 方法
│       └── 需要逐步替换 → 绞杀者模式
```

---

## 18.9 实战建议

**不要这样做**：
```python
# 危险！试图一次性测试+重构整个遗留模块
class TestLegacyEntireModule(unittest.TestCase):
    def test_everything(self):
        # 500行测试...
```

**应该这样**：
1. 每次只测试**你要修改的部分**周围的代码
2. 先建立**安全网**（特征测试）
3. 小步前进，每步后运行全部测试
4. 用 git 提交每个安全的小步骤

---

## 18.10 本章小结

| 技术 | 适用场景 | 核心价值 |
|------|----------|---------|
| 特征测试 | 理解和锁定现有行为 | 重构的安全网 |
| 接缝 | 硬编码依赖无法测试 | 引入可替换点 |
| 黄金主文件 | 复杂输出（HTML/PDF） | 回归检测 |
| 绞杀者模式 | 需要逐步替换旧系统 | 安全过渡 |
| Sprout 方法 | 最小侵入添加新逻辑 | 不破坏旧代码 |

**核心原则**：永远在修改遗留代码之前建立测试网，不管这个网有多薄。

**下一章**：Mock 的内部机制——理解 `__getattr__`、描述符协议和 call 录制，成为 Mock 高手。
