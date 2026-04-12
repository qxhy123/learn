# 第二章：Python unittest 框架基础

## 2.1 unittest 概览

`unittest` 是 Python 标准库内置的测试框架，灵感来自 Java 的 JUnit。

```
unittest 核心组件
├── TestCase     # 测试用例基类
├── TestSuite    # 测试套件（测试集合）
├── TestLoader   # 加载测试用例
├── TestRunner   # 运行测试并输出结果
└── assertions   # 断言方法（assertEqual, assertTrue 等）
```

---

## 2.2 第一个测试用例

```python
# test_calculator.py
import unittest

class TestCalculator(unittest.TestCase):

    def test_add_two_numbers(self):
        result = 1 + 1
        self.assertEqual(result, 2)

if __name__ == '__main__':
    unittest.main()
```

运行方式：
```bash
# 方式一：直接运行文件
python test_calculator.py

# 方式二：使用 -m unittest（推荐）
python -m unittest test_calculator

# 方式三：详细输出
python -m unittest test_calculator -v
```

输出示例：
```
test_add_two_numbers (test_calculator.TestCalculator) ... ok

----------------------------------------------------------------------
Ran 1 test in 0.001s

OK
```

---

## 2.3 TestCase 类的结构

```python
import unittest

class TestSomething(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """整个测试类运行前执行一次（类级别）"""
        print("setUpClass: 初始化类级别资源")

    @classmethod
    def tearDownClass(cls):
        """整个测试类运行后执行一次（类级别）"""
        print("tearDownClass: 清理类级别资源")

    def setUp(self):
        """每个测试方法运行前执行（方法级别）"""
        print("setUp: 准备测试数据")

    def tearDown(self):
        """每个测试方法运行后执行（方法级别）"""
        print("tearDown: 清理测试数据")

    def test_first(self):
        """测试方法：必须以 test_ 开头"""
        self.assertTrue(True)

    def test_second(self):
        self.assertEqual(1 + 1, 2)
```

**执行顺序**：
```
setUpClass
  setUp → test_first → tearDown
  setUp → test_second → tearDown
tearDownClass
```

---

## 2.4 常用断言方法

### 相等性断言

```python
# 值相等
self.assertEqual(actual, expected)      # actual == expected
self.assertNotEqual(actual, expected)   # actual != expected

# 对象同一性
self.assertIs(a, b)                     # a is b
self.assertIsNot(a, b)                  # a is not b

# None 检查
self.assertIsNone(x)                    # x is None
self.assertIsNotNone(x)                 # x is not None
```

### 布尔断言

```python
self.assertTrue(expr)      # bool(expr) is True
self.assertFalse(expr)     # bool(expr) is False
```

### 成员断言

```python
self.assertIn(member, container)        # member in container
self.assertNotIn(member, container)     # member not in container
```

### 类型断言

```python
self.assertIsInstance(obj, cls)         # isinstance(obj, cls)
self.assertNotIsInstance(obj, cls)
```

### 数值比较

```python
self.assertGreater(a, b)               # a > b
self.assertGreaterEqual(a, b)          # a >= b
self.assertLess(a, b)                  # a < b
self.assertLessEqual(a, b)             # a <= b

# 浮点数比较（避免精度问题）
self.assertAlmostEqual(a, b, places=7) # round(a-b, places) == 0
self.assertNotAlmostEqual(a, b)
```

### 字符串断言

```python
self.assertRegex(text, pattern)        # re.search(pattern, text)
self.assertNotRegex(text, pattern)
```

### 容器断言

```python
# 比较序列（忽略顺序）
self.assertCountEqual([1, 2, 3], [3, 1, 2])  # 元素相同，顺序无关

# 比较多行字符串
self.assertMultiLineEqual(first, second)

# 比较列表
self.assertListEqual([1, 2], [1, 2])

# 比较字典
self.assertDictEqual({'a': 1}, {'a': 1})

# 比较集合
self.assertSetEqual({1, 2}, {1, 2})
```

### 异常断言

```python
# 断言抛出异常
with self.assertRaises(ValueError):
    int("not a number")

# 断言抛出异常并检查消息
with self.assertRaisesRegex(ValueError, "invalid literal"):
    int("not a number")
```

---

## 2.5 自定义失败消息

所有断言方法都接受可选的 `msg` 参数，提供更清晰的失败信息：

```python
def test_user_age(self):
    user = get_user(id=1)
    self.assertEqual(
        user.age, 
        25, 
        msg=f"Expected age 25, got {user.age} for user {user.name}"
    )
```

---

## 2.6 跳过测试

```python
import unittest
import sys

class TestPlatformSpecific(unittest.TestCase):

    @unittest.skip("暂时跳过：功能未完成")
    def test_not_implemented(self):
        pass

    @unittest.skipIf(sys.platform == 'win32', "不支持 Windows")
    def test_unix_only(self):
        pass

    @unittest.skipUnless(sys.version_info >= (3, 10), "需要 Python 3.10+")
    def test_requires_310(self):
        pass

    @unittest.expectedFailure
    def test_known_bug(self):
        # 这个测试预期会失败（已知 bug）
        self.assertEqual(1, 2)
```

---

## 2.7 subTest：循环中的细粒度测试

```python
def test_even_numbers(self):
    test_cases = [2, 4, 6, 8, 10]
    for n in test_cases:
        with self.subTest(n=n):
            self.assertEqual(n % 2, 0, f"{n} should be even")
```

`subTest` 的优势：即使某个子测试失败，其他子测试仍会继续运行。

---

## 2.8 运行选项

```bash
# 运行单个测试类
python -m unittest test_module.TestClass

# 运行单个测试方法
python -m unittest test_module.TestClass.test_method

# 自动发现当前目录下所有测试
python -m unittest discover

# 指定发现目录和模式
python -m unittest discover -s tests/ -p "test_*.py" -v

# 失败时立即停止
python -m unittest -f test_module

# 显示本地变量（调试用）
python -m unittest -v --locals test_module
```

---

## 2.9 本章小结

- `unittest.TestCase` 是所有测试类的基类
- 测试方法必须以 `test_` 开头
- 生命周期：`setUpClass → setUp → test → tearDown → tearDownClass`
- 丰富的断言方法覆盖各种验证需求
- `subTest` 适合循环测试场景

**下一章**：用这些工具完成第一个完整的 TDD 开发循环。
