# 第三章：第一个完整的 TDD 循环

## 3.1 任务描述

我们要实现一个**购物车**，需求如下：
- 空购物车总价为 0
- 可以添加商品
- 总价是所有商品价格之和
- 不允许添加价格为负数的商品

用 TDD 方式，一步步来。

---

## 3.2 第一轮：空购物车

### Step 1 - Red：写第一个失败的测试

```python
# test_shopping_cart.py
import unittest

class TestShoppingCart(unittest.TestCase):

    def test_empty_cart_has_zero_total(self):
        cart = ShoppingCart()
        self.assertEqual(cart.total(), 0)

if __name__ == '__main__':
    unittest.main()
```

运行：
```
ERROR: test_empty_cart_has_zero_total
NameError: name 'ShoppingCart' is not defined
```

**这正是我们期望的！** 失败原因是"类不存在"，说明测试是真实的。

### Step 2 - Green：写最少的代码

```python
# shopping_cart.py
class ShoppingCart:
    def total(self):
        return 0
```

在测试文件中引入：
```python
from shopping_cart import ShoppingCart
```

运行：
```
test_empty_cart_has_zero_total ... ok
Ran 1 test in 0.001s  OK
```

### Step 3 - Refactor：暂无需重构

代码够简单，继续下一轮。

---

## 3.3 第二轮：添加商品

### Step 1 - Red

```python
def test_adding_one_item_sets_total(self):
    cart = ShoppingCart()
    cart.add_item("apple", price=3.0)
    self.assertEqual(cart.total(), 3.0)
```

运行：
```
ERROR: AttributeError: 'ShoppingCart' object has no attribute 'add_item'
```

### Step 2 - Green

```python
class ShoppingCart:
    def __init__(self):
        self._items = []

    def add_item(self, name, price):
        self._items.append(price)

    def total(self):
        return sum(self._items)
```

运行：两个测试都通过。

### Step 3 - Refactor

用具名元组让数据更清晰：
```python
from collections import namedtuple

Item = namedtuple('Item', ['name', 'price'])

class ShoppingCart:
    def __init__(self):
        self._items = []

    def add_item(self, name, price):
        self._items.append(Item(name=name, price=price))

    def total(self):
        return sum(item.price for item in self._items)
```

运行所有测试：全部通过。重构成功！

---

## 3.4 第三轮：多商品累加

### Step 1 - Red

```python
def test_multiple_items_sum_correctly(self):
    cart = ShoppingCart()
    cart.add_item("apple", price=3.0)
    cart.add_item("banana", price=1.5)
    cart.add_item("cherry", price=5.0)
    self.assertAlmostEqual(cart.total(), 9.5)
```

此时代码其实已经能通过这个测试，但写测试本身是有价值的——它**明确记录了行为**。

### Step 2 - Green：测试直接通过

运行：三个测试全部通过。

---

## 3.5 第四轮：拒绝负价格

### Step 1 - Red

```python
def test_negative_price_raises_error(self):
    cart = ShoppingCart()
    with self.assertRaises(ValueError):
        cart.add_item("fraud", price=-1.0)
```

运行：
```
FAIL: test_negative_price_raises_error
AssertionError: ValueError not raised
```

### Step 2 - Green

```python
def add_item(self, name, price):
    if price < 0:
        raise ValueError(f"Price cannot be negative: {price}")
    self._items.append(Item(name=name, price=price))
```

### Step 3 - Refactor

加一个更有意义的异常消息测试：
```python
def test_negative_price_error_message(self):
    cart = ShoppingCart()
    with self.assertRaisesRegex(ValueError, "negative"):
        cart.add_item("fraud", price=-1.0)
```

---

## 3.6 完整代码回顾

### 生产代码（shopping_cart.py）

```python
from collections import namedtuple

Item = namedtuple('Item', ['name', 'price'])


class ShoppingCart:
    def __init__(self):
        self._items = []

    def add_item(self, name, price):
        if price < 0:
            raise ValueError(f"Price cannot be negative: {price}")
        self._items.append(Item(name=name, price=price))

    def total(self):
        return sum(item.price for item in self._items)
```

### 测试代码（test_shopping_cart.py）

```python
import unittest
from shopping_cart import ShoppingCart


class TestShoppingCart(unittest.TestCase):

    def test_empty_cart_has_zero_total(self):
        cart = ShoppingCart()
        self.assertEqual(cart.total(), 0)

    def test_adding_one_item_sets_total(self):
        cart = ShoppingCart()
        cart.add_item("apple", price=3.0)
        self.assertEqual(cart.total(), 3.0)

    def test_multiple_items_sum_correctly(self):
        cart = ShoppingCart()
        cart.add_item("apple", price=3.0)
        cart.add_item("banana", price=1.5)
        cart.add_item("cherry", price=5.0)
        self.assertAlmostEqual(cart.total(), 9.5)

    def test_negative_price_raises_error(self):
        cart = ShoppingCart()
        with self.assertRaises(ValueError):
            cart.add_item("fraud", price=-1.0)

    def test_negative_price_error_message(self):
        cart = ShoppingCart()
        with self.assertRaisesRegex(ValueError, "negative"):
            cart.add_item("fraud", price=-1.0)


if __name__ == '__main__':
    unittest.main()
```

---

## 3.7 TDD 循环的节奏感

```
需求卡片 → Red → Green → Refactor → Red → Green → Refactor → ...
            ↑                              ↑
         5分钟以内                       循环往复
```

**TDD 的关键纪律**：
1. **永远不要跳过 Red 阶段** — 没有看到失败的测试，就不知道测试是否有效
2. **Green 阶段保持克制** — 不要写超出测试需要的代码
3. **Refactor 阶段要保持绿色** — 每次小步重构后立即运行测试

---

## 3.8 本章小结

通过购物车示例，我们完整经历了 4 个 TDD 循环：
- 空购物车 → 添加商品 → 多商品累加 → 拒绝负价格

每个功能点都有对应的测试，重构是安全的，代码的行为被精确记录。

**下一部分**：深入 unittest 的核心概念——测试组织、断言技巧、夹具生命周期。
