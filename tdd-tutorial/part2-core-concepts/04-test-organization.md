# 第四章：测试组织与命名

## 4.1 为什么组织很重要

随着项目增长，测试数量可能达到数百甚至数千个。好的组织结构让你能：
- 快速找到特定功能的测试
- 只运行受影响的测试子集
- 从测试名称读懂系统行为

---

## 4.2 测试命名规范

### 命名模式：`test_<行为>_when_<条件>`

```python
# 差：不表达意图
def test_1(self):
def test_cart(self):
def test_add(self):

# 好：行为即文档
def test_total_is_zero_when_cart_is_empty(self):
def test_total_increases_when_item_is_added(self):
def test_raises_value_error_when_price_is_negative(self):
```

### Given-When-Then 风格（BDD 影响）

```python
class TestShoppingCart(unittest.TestCase):

    # given an empty cart
    # when nothing happens
    # then total is 0
    def test_empty_cart_has_zero_total(self):
        cart = ShoppingCart()
        self.assertEqual(cart.total(), 0)

    # given a cart with items
    # when we call total()
    # then it returns sum of prices
    def test_total_reflects_sum_of_item_prices(self):
        cart = ShoppingCart()
        cart.add_item("a", 3.0)
        cart.add_item("b", 2.0)
        self.assertEqual(cart.total(), 5.0)
```

---

## 4.3 测试类的划分策略

### 策略一：按被测类划分（最常见）

```
tests/
├── test_shopping_cart.py    → class TestShoppingCart
├── test_user.py             → class TestUser
├── test_payment.py          → class TestPayment
└── test_order.py            → class TestOrder
```

### 策略二：按场景划分（复杂类时）

```python
# test_shopping_cart.py

class TestShoppingCartCreation(unittest.TestCase):
    """测试购物车的创建场景"""
    def test_new_cart_is_empty(self): ...
    def test_new_cart_total_is_zero(self): ...


class TestShoppingCartAddItem(unittest.TestCase):
    """测试添加商品的场景"""
    def test_can_add_valid_item(self): ...
    def test_rejects_negative_price(self): ...
    def test_rejects_zero_quantity(self): ...


class TestShoppingCartCheckout(unittest.TestCase):
    """测试结账场景"""
    def test_checkout_clears_cart(self): ...
    def test_checkout_returns_receipt(self): ...
```

### 策略三：按集成层级划分

```
tests/
├── unit/           # 纯单元测试，无 I/O
│   ├── test_calculator.py
│   └── test_validators.py
├── integration/    # 与数据库/API 交互
│   ├── test_user_repository.py
│   └── test_payment_gateway.py
└── e2e/            # 端到端测试
    └── test_checkout_flow.py
```

---

## 4.4 项目目录结构

### 小型项目

```
my_project/
├── shopping_cart.py
└── test_shopping_cart.py    # 与源文件同级
```

### 中型项目

```
my_project/
├── src/
│   ├── __init__.py
│   ├── cart.py
│   └── user.py
├── tests/
│   ├── __init__.py
│   ├── test_cart.py
│   └── test_user.py
└── setup.py
```

### 大型项目

```
my_project/
├── src/
│   └── myapp/
│       ├── domain/
│       ├── services/
│       └── api/
├── tests/
│   ├── unit/
│   │   └── domain/
│   ├── integration/
│   │   └── services/
│   └── conftest.py    # pytest 共享夹具（如果混用 pytest）
└── pyproject.toml
```

---

## 4.5 共享测试基类

当多个测试类有公共逻辑时，提取基类：

```python
class BaseCartTest(unittest.TestCase):
    """购物车测试的公共基类"""

    def setUp(self):
        self.cart = ShoppingCart()
        self.apple = ("apple", 3.0)
        self.banana = ("banana", 1.5)

    def _add_items(self, *items):
        """辅助方法：批量添加商品"""
        for name, price in items:
            self.cart.add_item(name, price)


class TestCartTotal(BaseCartTest):
    def test_empty_total(self):
        self.assertEqual(self.cart.total(), 0)

    def test_single_item(self):
        self._add_items(self.apple)
        self.assertEqual(self.cart.total(), 3.0)


class TestCartDiscount(BaseCartTest):
    def test_ten_percent_discount(self):
        self._add_items(self.apple, self.banana)
        self.assertAlmostEqual(self.cart.total_with_discount(0.1), 4.05)
```

> ⚠️ 注意：基类方法名不要以 `test_` 开头，否则 unittest 会尝试直接运行它们。

---

## 4.6 测试辅助方法（Helper Methods）

```python
class TestEmailValidator(unittest.TestCase):

    def _assert_valid(self, email):
        """断言邮箱有效的辅助方法"""
        result = validate_email(email)
        self.assertTrue(result.is_valid, f"Expected '{email}' to be valid")

    def _assert_invalid(self, email, reason=None):
        """断言邮箱无效的辅助方法"""
        result = validate_email(email)
        self.assertFalse(result.is_valid, f"Expected '{email}' to be invalid")
        if reason:
            self.assertIn(reason, result.error_message)

    def test_standard_email_is_valid(self):
        self._assert_valid("user@example.com")

    def test_no_at_sign_is_invalid(self):
        self._assert_invalid("userexample.com", reason="@")

    def test_multiple_at_signs_invalid(self):
        self._assert_invalid("a@b@c.com")
```

---

## 4.7 测试文件的 import 规范

```python
# test_cart.py - 标准 import 顺序

# 1. 标准库
import unittest
from unittest.mock import MagicMock, patch

# 2. 第三方库（如有）
# import pytest  # 通常混用时才引入

# 3. 本项目模块
from myapp.cart import ShoppingCart
from myapp.models import Item, User
```

---

## 4.8 测试发现规则

`python -m unittest discover` 的默认规则：
- 搜索目录：当前目录
- 文件模式：`test*.py`
- 顶层目录：当前目录

自定义：
```bash
# 在 tests/ 目录下搜索 test_*.py 文件
python -m unittest discover -s tests -p "test_*.py"

# 从项目根目录运行，发现 tests/ 子目录
python -m unittest discover -s . -p "test_*.py" -t .
```

---

## 4.9 本章小结

- 命名：`test_<行为>_when_<条件>` 让测试自文档化
- 测试类：按被测类或按场景划分
- 项目结构：`tests/` 目录与 `src/` 并列
- 共享逻辑：提取到基类或辅助方法，不以 `test_` 命名
- 测试发现：`-m unittest discover` 自动找测试

**下一章**：深入 unittest 的断言体系，掌握每种断言的最佳适用场景。
