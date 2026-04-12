# 第五章：断言深度解析

## 5.1 断言的本质

断言是测试的核心——它们定义了"什么是正确的"。选错断言方法会导致：
- 失败信息不清晰，难以诊断
- 误报（测试通过但行为错误）
- 漏报（测试失败但原因无关）

---

## 5.2 assertEqual 的陷阱

### 顺序问题（actual vs expected）

```python
# 推荐：assertEqual(actual, expected)
self.assertEqual(cart.total(), 5.0)   # actual=cart.total(), expected=5.0

# 失败消息清晰：
# AssertionError: 3.0 != 5.0

# 反例：顺序颠倒导致失败消息混乱
self.assertEqual(5.0, cart.total())
# AssertionError: 5.0 != 3.0   ← 让人困惑：是期望还是实际？
```

> 约定：`assertEqual(actual, expected)` — 实际值在前，期望值在后。

### 类型敏感

```python
self.assertEqual(1, 1.0)    # 通过：Python 的 == 允许 int/float 比较
self.assertEqual("1", 1)    # 失败：字符串 != 整数
self.assertEqual([], ())    # 失败：列表 != 元组（即使都为空）
self.assertEqual([1,2], (1,2))  # 失败：类型不同
```

---

## 5.3 浮点数断言

永远不要用 `assertEqual` 比较浮点数：

```python
# 危险！
self.assertEqual(0.1 + 0.2, 0.3)  # 失败！0.30000000000000004 != 0.3

# 正确：使用 assertAlmostEqual
self.assertAlmostEqual(0.1 + 0.2, 0.3, places=7)  # 精确到小数点后 7 位
self.assertAlmostEqual(0.1 + 0.2, 0.3, delta=1e-9) # 差值不超过 delta
```

### 自定义精度

```python
class TestFinancialCalculation(unittest.TestCase):

    def assertMoney(self, actual, expected):
        """货币精确到分（2位小数）"""
        self.assertAlmostEqual(actual, expected, places=2,
            msg=f"Money mismatch: expected {expected:.2f}, got {actual:.2f}")

    def test_tax_calculation(self):
        tax = calculate_tax(100.00, rate=0.085)
        self.assertMoney(tax, 8.50)
```

---

## 5.4 容器断言详解

### assertIn vs assertTrue(x in y)

```python
# 两者等价，但 assertIn 失败消息更好
self.assertIn("apple", cart.items())
# 失败：'apple' not found in ['banana', 'cherry']

self.assertTrue("apple" in cart.items())
# 失败：False is not true  ← 没有上下文信息
```

### assertCountEqual：忽略顺序的列表比较

```python
# 仅验证元素相同，不关心顺序
self.assertCountEqual(
    get_user_permissions(user),
    ["read", "write", "delete"]
)
```

### assertDictEqual 的细节

```python
actual = {"name": "Alice", "age": 30, "role": "admin"}
expected = {"name": "Alice", "age": 30, "role": "admin"}

self.assertDictEqual(actual, expected)
# 失败时显示详细的 diff：
# - {'name': 'Alice', 'age': 30, 'role': 'user'}
# + {'name': 'Alice', 'age': 30, 'role': 'admin'}
```

### 只检查字典的子集

```python
def assertDictContains(self, actual, subset):
    """验证 actual 包含 subset 中的所有键值对"""
    for key, value in subset.items():
        self.assertIn(key, actual, f"Key '{key}' missing from dict")
        self.assertEqual(actual[key], value,
            f"Key '{key}': expected {value!r}, got {actual[key]!r}")

def test_user_has_required_fields(self):
    user = create_user("Alice", "alice@example.com")
    self.assertDictContains(user.to_dict(), {
        "name": "Alice",
        "email": "alice@example.com",
    })
```

---

## 5.5 字符串断言

```python
# 包含子串
self.assertIn("error", response.message.lower())

# 正则匹配
self.assertRegex(log_output, r"\d{4}-\d{2}-\d{2} ERROR")

# 多行字符串（每个字符都比较，包括空白）
expected = "line1\nline2\nline3"
self.assertMultiLineEqual(generate_report(), expected)
```

### 实用技巧：忽略空白的字符串比较

```python
def assertStrippedEqual(self, actual, expected):
    """忽略首尾空白的字符串比较"""
    self.assertEqual(actual.strip(), expected.strip(),
        f"String mismatch:\nActual:   {actual!r}\nExpected: {expected!r}")
```

---

## 5.6 异常断言的完整用法

```python
class TestExceptionHandling(unittest.TestCase):

    def test_raises_correct_type(self):
        with self.assertRaises(ValueError):
            parse_age(-1)

    def test_raises_with_message(self):
        with self.assertRaisesRegex(ValueError, r"age.*negative"):
            parse_age(-1)

    def test_exception_attributes(self):
        """检查异常对象的属性"""
        with self.assertRaises(ValidationError) as ctx:
            validate_form({"email": "invalid"})
        
        exc = ctx.exception
        self.assertEqual(exc.field, "email")
        self.assertIn("format", exc.message)

    def test_no_exception_raised(self):
        """验证不抛出异常"""
        try:
            parse_age(25)  # 应该正常执行
        except Exception as e:
            self.fail(f"Unexpected exception raised: {e}")
```

---

## 5.7 自定义断言方法

封装复杂的验证逻辑，提高可读性和复用性：

```python
class TestHTTPResponse(unittest.TestCase):

    def assertOK(self, response):
        self.assertEqual(response.status_code, 200,
            f"Expected 200 OK, got {response.status_code}: {response.text[:200]}")

    def assertCreated(self, response):
        self.assertEqual(response.status_code, 201,
            f"Expected 201 Created, got {response.status_code}")

    def assertJSONContains(self, response, key, value):
        data = response.json()
        self.assertIn(key, data, f"Key '{key}' not in response JSON")
        self.assertEqual(data[key], value,
            f"response['{key}']: expected {value!r}, got {data[key]!r}")

    def test_create_user(self):
        response = self.client.post("/users", json={"name": "Alice"})
        self.assertCreated(response)
        self.assertJSONContains(response, "name", "Alice")
```

---

## 5.8 断言选择指南

| 场景 | 推荐断言 | 避免 |
|------|----------|------|
| 值相等 | `assertEqual` | `assertTrue(a == b)` |
| 浮点相等 | `assertAlmostEqual` | `assertEqual` |
| 布尔真假 | `assertTrue/False` | `assertEqual(x, True)` |
| None 检查 | `assertIsNone` | `assertEqual(x, None)` |
| 容器成员 | `assertIn` | `assertTrue(x in y)` |
| 无序列表 | `assertCountEqual` | `assertEqual(sorted(a), sorted(b))` |
| 异常类型 | `assertRaises` (with) | `try/except/fail` |
| 异常消息 | `assertRaisesRegex` | 手动检查 `str(e)` |
| 类型检查 | `assertIsInstance` | `assertEqual(type(x), T)` |

---

## 5.9 失败信息的艺术

好的失败信息能大幅减少调试时间：

```python
# 差：失败信息无意义
self.assertTrue(len(users) > 0)
# AssertionError: False is not true

# 好：失败信息包含上下文
self.assertGreater(len(users), 0,
    f"Expected at least one user, got {len(users)}. "
    f"Query params: {query_params}")

# 差：手动构建信息
if user.is_active != expected_active:
    self.fail(f"Wrong")

# 好：利用断言的内置信息 + 自定义补充
self.assertEqual(user.is_active, expected_active,
    f"User {user.id} ({user.name}) active status mismatch")
```

---

## 5.10 本章小结

- `assertEqual(actual, expected)` — actual 在前
- 浮点数用 `assertAlmostEqual`，指定 `places` 或 `delta`
- 容器比较：`assertCountEqual`（忽略顺序）、`assertDictEqual`（有 diff）
- 异常检查：优先用 `assertRaises` 上下文管理器
- 封装重复的断言逻辑为辅助方法
- 提供清晰的失败消息，包含足够的上下文

**下一章**：测试夹具（Fixtures）的完整生命周期管理。
