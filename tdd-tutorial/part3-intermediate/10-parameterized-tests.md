# 第十章：参数化测试

## 10.1 为什么需要参数化

没有参数化时，测试多个输入的代码：

```python
# 重复！重复！重复！
def test_validate_email_valid_1(self):
    self.assertTrue(is_valid_email("user@example.com"))

def test_validate_email_valid_2(self):
    self.assertTrue(is_valid_email("user+tag@sub.domain.com"))

def test_validate_email_invalid_1(self):
    self.assertFalse(is_valid_email("not-an-email"))

def test_validate_email_invalid_2(self):
    self.assertFalse(is_valid_email("@example.com"))
```

参数化后：
```python
def test_valid_emails(self):
    for email in VALID_EMAILS:
        with self.subTest(email=email):
            self.assertTrue(is_valid_email(email))
```

---

## 10.2 subTest：最简单的参数化

`subTest` 是 `unittest` 内置的参数化机制，适合简单场景。

```python
class TestEmailValidator(unittest.TestCase):

    VALID_CASES = [
        "user@example.com",
        "user+tag@example.com",
        "user@sub.domain.co.uk",
        "123@example.com",
    ]

    INVALID_CASES = [
        ("not-an-email",   "missing @"),
        ("@example.com",   "missing local part"),
        ("user@",          "missing domain"),
        ("user @ex.com",   "space in local part"),
        ("user@ex com",    "space in domain"),
    ]

    def test_valid_emails_are_accepted(self):
        for email in self.VALID_CASES:
            with self.subTest(email=email):
                result = is_valid_email(email)
                self.assertTrue(result,
                    f"Expected '{email}' to be valid, but was rejected")

    def test_invalid_emails_are_rejected(self):
        for email, reason in self.INVALID_CASES:
            with self.subTest(email=email, reason=reason):
                result = is_valid_email(email)
                self.assertFalse(result,
                    f"Expected '{email}' to be invalid ({reason}), but was accepted")
```

`subTest` 的关键优势：**一个子测试失败，其余继续运行**。

---

## 10.3 基于数据驱动的测试表格

将测试数据组织成表格，提高可读性：

```python
class TestAgeParser(unittest.TestCase):

    # (输入, 期望输出 or 期望异常)
    TEST_TABLE = [
        # 有效年龄
        (0,   0,           None),
        (1,   1,           None),
        (25,  25,          None),
        (150, 150,         None),
        # 无效年龄
        (-1,  None,        ValueError),
        (151, None,        ValueError),
        (None, None,       TypeError),
        ("25", None,       TypeError),
    ]

    def test_parse_age(self):
        for age_input, expected_result, expected_exc in self.TEST_TABLE:
            with self.subTest(input=age_input):
                if expected_exc:
                    with self.assertRaises(expected_exc):
                        parse_age(age_input)
                else:
                    result = parse_age(age_input)
                    self.assertEqual(result, expected_result)
```

---

## 10.4 生成测试方法：动态创建测试

`unittest` 本身不支持参数化装饰器，但可以动态生成测试方法：

```python
import unittest


def generate_tests(test_class):
    """动态为测试类生成参数化测试方法"""

    test_cases = [
        ("zero", 0, 0),
        ("one", 1, 1),
        ("negative", -5, 5),
        ("large", 1000, 1000),
    ]

    def make_test(n, expected):
        def test(self):
            self.assertEqual(abs(n), expected)
        return test

    for name, n, expected in test_cases:
        method_name = f"test_abs_{name}"
        setattr(test_class, method_name, make_test(n, expected))

    return test_class


@generate_tests
class TestAbsoluteValue(unittest.TestCase):
    pass
# 自动生成：test_abs_zero, test_abs_one, test_abs_negative, test_abs_large
```

---

## 10.5 使用 parameterized 库

第三方库 `parameterized` 提供更优雅的参数化语法：

```bash
pip install parameterized
```

```python
from parameterized import parameterized, parameterized_class
import unittest


class TestCalculator(unittest.TestCase):

    @parameterized.expand([
        ("add_positives",   2, 3,  5),
        ("add_negatives",  -2, -3, -5),
        ("add_mixed",      -2, 3,  1),
        ("add_zeros",       0, 0,  0),
    ])
    def test_add(self, name, a, b, expected):
        # 生成: test_add_0_add_positives, test_add_1_add_negatives, ...
        self.assertEqual(a + b, expected)

    @parameterized.expand([
        (10, 2,  5.0),
        (9,  3,  3.0),
        (7,  2,  3.5),
    ])
    def test_divide(self, dividend, divisor, expected):
        self.assertAlmostEqual(divide(dividend, divisor), expected)

    @parameterized.expand([
        (10, 0),   # 除以零
        (-5, 0),
    ])
    def test_divide_by_zero_raises(self, dividend, divisor):
        with self.assertRaises(ZeroDivisionError):
            divide(dividend, divisor)
```

### parameterized_class：参数化整个测试类

```python
@parameterized_class([
    {"storage_type": "memory",   "config": {}},
    {"storage_type": "file",     "config": {"path": "/tmp"}},
    {"storage_type": "redis",    "config": {"host": "localhost"}},
])
class TestStorage(unittest.TestCase):
    storage_type = None   # 会被 parameterized_class 替换
    config = None

    def setUp(self):
        self.storage = StorageFactory.create(self.storage_type, **self.config)

    def test_save_and_retrieve(self):
        self.storage.save("key", "value")
        self.assertEqual(self.storage.get("key"), "value")

    def test_overwrite(self):
        self.storage.save("key", "v1")
        self.storage.save("key", "v2")
        self.assertEqual(self.storage.get("key"), "v2")
```

---

## 10.6 数学和算法的参数化测试

参数化特别适合数学函数和算法：

```python
import math
import unittest


class TestMathFunctions(unittest.TestCase):

    SQRT_CASES = [
        (0,   0.0),
        (1,   1.0),
        (4,   2.0),
        (9,   3.0),
        (2,   math.sqrt(2)),   # 无理数
        (0.25, 0.5),
    ]

    def test_sqrt(self):
        for n, expected in self.SQRT_CASES:
            with self.subTest(n=n):
                self.assertAlmostEqual(math.sqrt(n), expected, places=10)

    def test_sqrt_of_negative_raises(self):
        negatives = [-1, -0.5, -100]
        for n in negatives:
            with self.subTest(n=n):
                with self.assertRaises(ValueError):
                    safe_sqrt(n)   # 自定义的 sqrt，负数抛 ValueError


class TestSortingAlgorithm(unittest.TestCase):
    """用同一套测试验证多种排序算法"""

    SORT_CASES = [
        ([],              []),
        ([1],             [1]),
        ([3, 1, 2],       [1, 2, 3]),
        ([5, 5, 5],       [5, 5, 5]),
        ([-3, 0, 3],      [-3, 0, 3]),
        ([9, 8, 7, 6, 5], [5, 6, 7, 8, 9]),
    ]

    def _run_sort_test(self, sort_func):
        for input_list, expected in self.SORT_CASES:
            with self.subTest(input=input_list):
                result = sort_func(input_list.copy())
                self.assertEqual(result, expected)

    def test_bubble_sort(self):
        self._run_sort_test(bubble_sort)

    def test_merge_sort(self):
        self._run_sort_test(merge_sort)

    def test_quick_sort(self):
        self._run_sort_test(quick_sort)
```

---

## 10.7 测试数据的组织原则

```python
class TestPasswordValidator(unittest.TestCase):

    # 将测试数据定义为类属性，集中管理
    VALID_PASSWORDS = [
        "Abc123!@#",          # 标准有效密码
        "P@ssw0rd",           # 典型例子
        "A" * 8 + "1!",       # 最短有效长度
        "A" * 64 + "1!",      # 最长有效长度
    ]

    INVALID_TOO_SHORT = [
        "Ab1!",       # 4 字符
        "Abc12!",     # 6 字符
        "Abc1234",    # 7 字符，无特殊字符
    ]

    INVALID_NO_UPPERCASE = [
        "abc123!@",
        "password1!",
    ]

    INVALID_NO_DIGIT = [
        "Abcdef!@",
        "Password!",
    ]

    def test_valid_passwords_accepted(self):
        for pwd in self.VALID_PASSWORDS:
            with self.subTest(password=f"{pwd[:4]}..."):
                self.assertTrue(validate_password(pwd))

    def test_short_passwords_rejected(self):
        for pwd in self.INVALID_TOO_SHORT:
            with self.subTest(password=pwd, reason="too short"):
                result = validate_password(pwd)
                self.assertFalse(result.is_valid)
                self.assertIn("length", result.errors)

    def test_no_uppercase_rejected(self):
        for pwd in self.INVALID_NO_UPPERCASE:
            with self.subTest(password=pwd, reason="no uppercase"):
                result = validate_password(pwd)
                self.assertFalse(result.is_valid)
                self.assertIn("uppercase", result.errors)
```

---

## 10.8 本章小结

- `subTest`：unittest 内置参数化，失败不影响其他子测试
- 测试表格：将输入/期望/异常组织成结构化数据
- 动态生成测试方法：`setattr(TestClass, 'test_xxx', fn)`
- `parameterized` 库：提供 `@parameterized.expand` 装饰器
- `parameterized_class`：对整个测试类参数化（多存储后端等场景）
- 测试数据集中定义为类属性，易于维护和扩展

**下一章**：在真实 I/O 场景（文件、数据库）下的测试策略。
