# 第十六章：基于属性的测试（Property-Based Testing）

## 16.1 示例测试的局限

传统的基于示例的测试（Example-Based Testing）：
```python
def test_reverse_list(self):
    self.assertEqual(reverse_list([1, 2, 3]), [3, 2, 1])
    self.assertEqual(reverse_list([]),        [])
```

问题：**你只测试了你想到的用例**。开发者的盲点就是测试的盲点。

基于属性的测试（Property-Based Testing）：**描述输入的规律和输出的不变性，让框架自动生成数千个用例**。

---

## 16.2 核心思想：不变量（Invariants）

好的属性测试找到的是**对任意合法输入都成立的不变量**：

```
reverse_list 的不变量：
  1. len(result) == len(original)        # 长度不变
  2. reverse(reverse(x)) == x            # 双重反转还原
  3. result[0] == original[-1]           # 首尾互换
  4. set(result) == set(original)        # 元素集合不变
```

这些属性比"[1,2,3] → [3,2,1]"传递了更多的约束信息。

---

## 16.3 安装 Hypothesis

```bash
pip install hypothesis
```

Hypothesis 与 `unittest.TestCase` 完全兼容：

```python
from hypothesis import given, settings, assume
from hypothesis import strategies as st
```

---

## 16.4 第一个属性测试

```python
import unittest
from hypothesis import given
from hypothesis import strategies as st


def reverse_list(lst):
    return lst[::-1]


class TestReverseList(unittest.TestCase):

    @given(st.lists(st.integers()))
    def test_double_reverse_is_identity(self, lst):
        """对任意整数列表：双重反转等于原列表"""
        self.assertEqual(reverse_list(reverse_list(lst)), lst)

    @given(st.lists(st.integers()))
    def test_length_preserved(self, lst):
        """反转不改变长度"""
        self.assertEqual(len(reverse_list(lst)), len(lst))

    @given(st.lists(st.integers(), min_size=1))
    def test_first_element_becomes_last(self, lst):
        """非空列表：原来的第一个元素变成最后一个"""
        result = reverse_list(lst)
        self.assertEqual(result[-1], lst[0])
        self.assertEqual(result[0], lst[-1])

    @given(st.lists(st.integers()))
    def test_elements_preserved(self, lst):
        """所有元素被保留（频次不变）"""
        from collections import Counter
        self.assertEqual(Counter(reverse_list(lst)), Counter(lst))
```

运行输出：
```
test_double_reverse_is_identity: Hypothesis 生成了 100 个测试用例
test_length_preserved: 100 个用例全部通过
...
```

---

## 16.5 策略（Strategies）：生成测试数据

Hypothesis 用"策略"描述如何生成数据：

### 基础类型策略

```python
st.integers()                    # 任意整数（包括极值）
st.integers(min_value=0, max_value=100)  # 有界整数
st.floats(allow_nan=False)       # 浮点数（排除 NaN）
st.text()                        # Unicode 字符串
st.text(alphabet=st.characters(whitelist_categories=('Ll',)))  # 仅小写字母
st.booleans()                    # True/False
st.none()                        # None
```

### 容器策略

```python
st.lists(st.integers())                     # 整数列表
st.lists(st.integers(), min_size=1, max_size=10)
st.sets(st.integers())                      # 整数集合
st.dictionaries(st.text(), st.integers())   # 字典
st.tuples(st.integers(), st.text())         # 固定结构元组
```

### 组合策略

```python
# 从几个值中选一个
st.sampled_from([1, 2, 3, "a", None])

# 多种类型之一
st.one_of(st.integers(), st.text(), st.none())

# 映射：从策略生成值再变换
st.integers().map(lambda n: n * 2)   # 生成偶数

# 过滤：只保留满足条件的值
st.integers().filter(lambda n: n % 2 == 0)  # 偶数（效率低，用 map 更好）
```

---

## 16.6 自定义域对象策略

```python
from hypothesis import given, strategies as st
from dataclasses import dataclass


@dataclass
class Product:
    name: str
    price: float
    quantity: int


# 构建 Product 的策略
product_strategy = st.builds(
    Product,
    name=st.text(min_size=1, max_size=50),
    price=st.floats(min_value=0.01, max_value=10000.0, allow_nan=False),
    quantity=st.integers(min_value=0, max_value=9999),
)


class TestCartInvariant(unittest.TestCase):

    @given(st.lists(product_strategy, min_size=1))
    def test_total_equals_sum_of_item_totals(self, products):
        """购物车总价 = 各商品(价格 × 数量)之和"""
        cart = ShoppingCart()
        for p in products:
            cart.add(p)

        expected = sum(p.price * p.quantity for p in products)
        self.assertAlmostEqual(cart.total(), expected, places=5)

    @given(st.lists(product_strategy))
    def test_total_is_non_negative(self, products):
        """总价永远非负"""
        cart = ShoppingCart()
        for p in products:
            cart.add(p)
        self.assertGreaterEqual(cart.total(), 0)
```

---

## 16.7 assume()：过滤无效输入

```python
from hypothesis import assume

@given(st.integers(), st.integers())
def test_division_result(self, a, b):
    assume(b != 0)   # 跳过 b=0 的用例，而不是报错
    result = a / b
    self.assertAlmostEqual(result * b, a, places=10)
```

> 注意：`assume` 过多会降低效率。优先用 `.filter()` 或有界策略。

---

## 16.8 Hypothesis 的收缩（Shrinking）

当 Hypothesis 找到失败用例时，它会自动**收缩**到最小失败用例：

```python
def buggy_sort(lst):
    """有 bug 的排序：忽略了负数"""
    return sorted(x for x in lst if x >= 0)

class TestBuggySort(unittest.TestCase):

    @given(st.lists(st.integers()))
    def test_sort_preserves_all_elements(self, lst):
        result = buggy_sort(lst)
        self.assertEqual(sorted(result), sorted(lst))
```

Hypothesis 发现失败后，不会报告原始的大列表，而是自动收缩到：
```
Falsifying example: test_sort_preserves_all_elements(lst=[-1])
```

这是最小的失败用例——一个包含负数的列表。

---

## 16.9 有状态属性测试

Hypothesis 的 `RuleBasedStateMachine` 可以测试有状态系统：

```python
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, invariant
from hypothesis import strategies as st


class CartStateMachine(RuleBasedStateMachine):
    """用状态机测试购物车的不变量"""

    def __init__(self):
        super().__init__()
        self.cart = ShoppingCart()
        self.expected_items = []

    @initialize()
    def create_cart(self):
        self.cart = ShoppingCart()
        self.expected_items = []

    @rule(name=st.text(min_size=1, max_size=20),
          price=st.floats(min_value=0.01, max_value=1000.0, allow_nan=False))
    def add_valid_item(self, name, price):
        self.cart.add_item(name, price)
        self.expected_items.append(price)

    @rule()
    def clear_cart(self):
        self.cart.clear()
        self.expected_items.clear()

    @invariant()
    def total_matches_expected(self):
        """不变量：任意操作序列后，总价都应该正确"""
        expected = sum(self.expected_items)
        assert abs(self.cart.total() - expected) < 1e-9, \
            f"Total {self.cart.total()} != expected {expected}"

    @invariant()
    def total_is_non_negative(self):
        assert self.cart.total() >= 0


# 将状态机转为 unittest TestCase
TestCartStateMachine = CartStateMachine.TestCase
```

---

## 16.10 属性测试 vs 示例测试 的协作

二者不是替代关系，而是互补：

```
示例测试：
  ✓ 验证具体的已知场景
  ✓ 作为文档（人类可读）
  ✓ 快速（精确用例）

属性测试：
  ✓ 发现开发者未想到的边界条件
  ✓ 验证系统不变量
  ✓ 自动收缩到最小失败用例
  ✗ 运行稍慢（通常生成 100+ 用例）
  ✗ 属性本身需要仔细设计
```

**最佳实践**：
1. 用示例测试表达核心行为（规范性）
2. 用属性测试验证不变量（防御性）
3. 当属性测试找到 bug，将最小失败用例提取为示例测试（回归保护）

---

## 16.11 @settings 控制测试行为

```python
from hypothesis import settings, HealthCheck
from hypothesis import Phase

class TestWithCustomSettings(unittest.TestCase):

    @settings(max_examples=500)   # 生成更多用例（默认 100）
    @given(st.integers())
    def test_thorough(self, n):
        ...

    @settings(max_examples=10)    # 快速检查（CI 中节省时间）
    @given(st.lists(st.integers()))
    def test_quick(self, lst):
        ...

    @settings(
        max_examples=200,
        suppress_health_check=[HealthCheck.too_slow],  # 禁用慢速检查
        deriving=True,              # 从数据库中重放历史失败用例
    )
    @given(st.text())
    def test_text_processing(self, text):
        ...
```

---

## 16.12 发现真实 Bug 的案例

```python
def parse_date(s: str) -> tuple:
    """解析 'YYYY-MM-DD' 格式的日期"""
    year, month, day = s.split('-')
    return int(year), int(month), int(day)


class TestParseDate(unittest.TestCase):

    # 示例测试（通过）
    def test_standard_date(self):
        self.assertEqual(parse_date("2024-01-15"), (2024, 1, 15))

    # 属性测试（发现 bug！）
    @given(st.dates())
    def test_roundtrip(self, date):
        s = date.strftime("%Y-%m-%d")
        year, month, day = parse_date(s)
        self.assertEqual(year,  date.year)
        self.assertEqual(month, date.month)
        self.assertEqual(day,   date.day)
    # Hypothesis 会发现：年份 < 1000 时 strftime 不补零
    # → 发现了 parse_date 无法处理 "999-01-01" 的 bug
```

---

## 16.13 本章小结

- 属性测试描述**不变量**，而非具体示例
- Hypothesis 自动生成数据，**收缩**到最小失败用例
- 核心策略：`st.integers/text/lists/builds/one_of`
- `assume()` 过滤无效输入，`@settings` 控制生成数量
- `RuleBasedStateMachine` 测试有状态系统的不变量
- 属性测试和示例测试**互补**，不是替代

**下一章**：六边形架构——TDD 如何自然导向端口与适配器设计。
