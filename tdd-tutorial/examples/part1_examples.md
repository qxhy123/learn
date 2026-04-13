# Part 1 示例：TDD 基础与 unittest 入门

运行方式：`python -m unittest examples/part1_examples.py -v`

```python
"""
Part 1 示例：TDD 基础与 unittest 入门
运行方式：python -m unittest examples/part1_examples.py -v
"""
import unittest
from collections import namedtuple


# ── 生产代码 ────────────────────────────────────────────────────────────────

Item = namedtuple('Item', ['name', 'price'])


class ShoppingCart:
    """购物车：第三章 TDD 循环的完整示例"""

    def __init__(self):
        self._items = []

    def add_item(self, name: str, price: float) -> None:
        if price < 0:
            raise ValueError(f"Price cannot be negative: {price}")
        self._items.append(Item(name=name, price=price))

    def total(self) -> float:
        return sum(item.price for item in self._items)

    def item_count(self) -> int:
        return len(self._items)

    def items(self):
        return list(self._items)


# ── 测试代码 ────────────────────────────────────────────────────────────────

class TestShoppingCart(unittest.TestCase):
    """第三章：购物车 TDD 循环"""

    # 第一轮：空购物车
    def test_empty_cart_has_zero_total(self):
        cart = ShoppingCart()
        self.assertEqual(cart.total(), 0)

    def test_empty_cart_has_zero_items(self):
        cart = ShoppingCart()
        self.assertEqual(cart.item_count(), 0)

    # 第二轮：添加单商品
    def test_adding_one_item_sets_total(self):
        cart = ShoppingCart()
        cart.add_item("apple", price=3.0)
        self.assertEqual(cart.total(), 3.0)

    def test_adding_item_increases_count(self):
        cart = ShoppingCart()
        cart.add_item("apple", price=3.0)
        self.assertEqual(cart.item_count(), 1)

    # 第三轮：多商品累加
    def test_multiple_items_sum_correctly(self):
        cart = ShoppingCart()
        cart.add_item("apple", price=3.0)
        cart.add_item("banana", price=1.5)
        cart.add_item("cherry", price=5.0)
        self.assertAlmostEqual(cart.total(), 9.5)

    # 第四轮：拒绝负价格
    def test_negative_price_raises_value_error(self):
        cart = ShoppingCart()
        with self.assertRaises(ValueError):
            cart.add_item("fraud", price=-1.0)

    def test_negative_price_error_message_mentions_negative(self):
        cart = ShoppingCart()
        with self.assertRaisesRegex(ValueError, "negative"):
            cart.add_item("fraud", price=-1.0)

    # 边界条件
    def test_zero_price_is_valid(self):
        cart = ShoppingCart()
        cart.add_item("freebie", price=0.0)
        self.assertEqual(cart.total(), 0.0)
        self.assertEqual(cart.item_count(), 1)

    def test_items_returns_copy_of_list(self):
        cart = ShoppingCart()
        cart.add_item("apple", price=3.0)
        items = cart.items()
        items.clear()  # 修改返回的列表
        self.assertEqual(cart.item_count(), 1)  # 不影响购物车


class TestUnittestBasics(unittest.TestCase):
    """第二章：unittest 基础特性演示"""

    @classmethod
    def setUpClass(cls):
        cls.shared_data = [1, 2, 3, 4, 5]

    def setUp(self):
        self.cart = ShoppingCart()

    # 各种断言演示
    def test_equal_assertion(self):
        self.assertEqual(1 + 1, 2)

    def test_float_comparison(self):
        self.assertAlmostEqual(0.1 + 0.2, 0.3, places=10)

    def test_membership_assertion(self):
        self.assertIn(3, self.shared_data)
        self.assertNotIn(10, self.shared_data)

    def test_none_assertion(self):
        self.assertIsNone(None)
        self.assertIsNotNone(0)

    def test_type_assertion(self):
        self.assertIsInstance(self.cart, ShoppingCart)

    def test_container_unordered(self):
        self.assertCountEqual([1, 3, 2], [2, 1, 3])

    # subTest 演示
    def test_multiple_cases_with_subtest(self):
        test_cases = [
            (0,   "zero"),
            (1,   "one"),
            (-1,  "negative"),
        ]
        for n, label in test_cases:
            with self.subTest(n=n, label=label):
                self.assertIsInstance(n, int, f"{label} should be an int")

    @unittest.skip("示例：跳过测试")
    def test_skipped(self):
        self.fail("This should be skipped")

    @unittest.expectedFailure
    def test_known_bug(self):
        # 已知 bug：演示 expectedFailure
        self.assertEqual(1, 2)  # 这会失败，但被标记为预期失败


if __name__ == '__main__':
    unittest.main(verbosity=2)
```
