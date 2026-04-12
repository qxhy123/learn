# 第七章：测试套件与自动发现

## 7.1 TestSuite 基础

`TestSuite` 是测试用例的容器，可以手动组装或自动发现。

```python
import unittest

# 手动构建套件
def build_suite():
    suite = unittest.TestSuite()
    
    # 添加单个测试方法
    suite.addTest(TestCart('test_empty_cart_has_zero_total'))
    suite.addTest(TestCart('test_adding_item_increases_total'))
    
    # 添加整个测试类
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestUser))
    
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(build_suite())
```

---

## 7.2 TestLoader：自动加载测试

```python
loader = unittest.TestLoader()

# 从测试类加载
suite1 = loader.loadTestsFromTestCase(TestCart)

# 从模块加载（该模块中所有 TestCase 子类）
import test_cart
suite2 = loader.loadTestsFromModule(test_cart)

# 按名称加载（字符串形式）
suite3 = loader.loadTestsFromName('test_cart.TestCart.test_empty_cart')

# 从多个名称加载
suite4 = loader.loadTestsFromNames([
    'test_cart.TestCart',
    'test_user.TestUser',
])
```

---

## 7.3 自动发现（discover）

### 命令行使用

```bash
# 发现当前目录下所有测试（默认：test*.py）
python -m unittest discover

# 指定目录、模式、顶层目录
python -m unittest discover \
    -s tests/           \   # 搜索起始目录
    -p "test_*.py"      \   # 文件名模式
    -t .                \   # 顶层目录（影响 import）
    -v                      # 详细输出

# 只运行集成测试
python -m unittest discover -s tests/integration -v
```

### 编程方式使用

```python
loader = unittest.TestLoader()
suite = loader.discover(
    start_dir='tests/',
    pattern='test_*.py',
    top_level_dir='.'
)

runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)
```

---

## 7.4 TestRunner 的输出控制

```python
# verbosity 级别
# 0: 静默（只显示总结）
# 1: 默认（每个测试显示 . F E）
# 2: 详细（显示测试名称和结果）

runner = unittest.TextTestRunner(
    verbosity=2,
    stream=sys.stderr,          # 输出到 stderr
    failfast=True,              # 第一个失败就停止
    buffer=True,                # 捕获 stdout/stderr
    resultclass=MyTestResult,   # 自定义结果类
)
```

---

## 7.5 自定义 TestResult

```python
class ColorTestResult(unittest.TextTestResult):
    """带颜色的测试结果输出"""

    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'

    def addSuccess(self, test):
        super().addSuccess(test)
        if self.verbosity > 1:
            self.stream.write(f"{self.GREEN}PASS{self.RESET}\n")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        if self.verbosity > 1:
            self.stream.write(f"{self.RED}FAIL{self.RESET}\n")

    def addError(self, test, err):
        super().addError(test, err)
        if self.verbosity > 1:
            self.stream.write(f"{self.YELLOW}ERROR{self.RESET}\n")


class ColorTestRunner(unittest.TextTestRunner):
    resultclass = ColorTestResult


if __name__ == '__main__':
    ColorTestRunner(verbosity=2).run(suite)
```

---

## 7.6 测试过滤与标签

`unittest` 原生不支持标签，但可以用约定实现：

### 方案一：按目录分层

```
tests/
├── unit/
│   └── test_cart.py
├── integration/
│   └── test_checkout.py
└── slow/
    └── test_report_generation.py
```

```bash
# 只跑单元测试
python -m unittest discover tests/unit

# 只跑集成测试
python -m unittest discover tests/integration
```

### 方案二：命名约定

```python
class TestCart(unittest.TestCase):
    
    def test_fast_empty_cart(self):      # 快速测试
        ...

    def test_slow_generate_report(self): # 慢速测试（命名标识）
        ...
```

### 方案三：自定义装饰器 + 过滤运行器

```python
# 定义标签装饰器
def tag(*tags):
    def decorator(func):
        func._tags = set(tags)
        return func
    return decorator


class FilteredTestLoader(unittest.TestLoader):
    def __init__(self, include_tags=None, exclude_tags=None):
        super().__init__()
        self.include_tags = set(include_tags or [])
        self.exclude_tags = set(exclude_tags or [])

    def loadTestsFromTestCase(self, testCaseClass):
        suite = super().loadTestsFromTestCase(testCaseClass)
        filtered = unittest.TestSuite()
        for test in suite:
            tags = getattr(test._testMethodDoc and 
                          getattr(testCaseClass, test._testMethodName), 
                          '_tags', set())
            if self.include_tags and not tags & self.include_tags:
                continue
            if self.exclude_tags and tags & self.exclude_tags:
                continue
            filtered.addTest(test)
        return filtered


# 使用示例
class TestPerformance(unittest.TestCase):

    @tag('slow', 'benchmark')
    def test_large_dataset_processing(self):
        ...

    @tag('fast', 'smoke')
    def test_basic_operation(self):
        ...

# 只运行快速测试
loader = FilteredTestLoader(include_tags=['fast'])
suite = loader.loadTestsFromTestCase(TestPerformance)
```

---

## 7.7 在 CI 中组织测试套件

### Makefile 示例

```makefile
.PHONY: test test-unit test-integration test-all

test-unit:
	python -m unittest discover tests/unit -v

test-integration:
	python -m unittest discover tests/integration -v

test-all:
	python -m unittest discover tests/ -v

test-fast:
	python -m unittest discover tests/unit -p "test_*.py" -v

# 带覆盖率
test-coverage:
	python -m coverage run -m unittest discover tests/
	python -m coverage report -m
	python -m coverage html
```

### 主套件文件

```python
# tests/suite.py
import unittest

def unit_suite():
    loader = unittest.TestLoader()
    return loader.discover('tests/unit', pattern='test_*.py')

def integration_suite():
    loader = unittest.TestLoader()
    return loader.discover('tests/integration', pattern='test_*.py')

def full_suite():
    suite = unittest.TestSuite()
    suite.addTests(unit_suite())
    suite.addTests(integration_suite())
    return suite

if __name__ == '__main__':
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    
    suites = {
        'unit': unit_suite,
        'integration': integration_suite,
        'all': full_suite,
    }
    
    runner = unittest.TextTestRunner(verbosity=2, failfast='--failfast' in sys.argv)
    result = runner.run(suites[mode]())
    sys.exit(0 if result.wasSuccessful() else 1)
```

---

## 7.8 测试运行结果分析

```python
result = runner.run(suite)

# result 对象的属性
result.wasSuccessful()     # bool: 全部通过？
result.testsRun            # 运行了多少测试
result.failures            # [(test, traceback_str), ...]
result.errors              # [(test, traceback_str), ...]
result.skipped             # [(test, reason), ...]
result.expectedFailures    # [(test, traceback_str), ...]
result.unexpectedSuccesses # [test, ...]

# 退出码约定
import sys
sys.exit(0 if result.wasSuccessful() else 1)
```

---

## 7.9 本章小结

- `TestSuite` 可手动或自动组装测试集合
- `TestLoader.discover()` 是推荐的自动发现方式
- `TextTestRunner(verbosity=2)` 提供详细输出
- 可自定义 `TestResult` 实现彩色输出、报告生成等
- 按目录分层（unit/integration/slow）是 CI 中管理测试子集的最佳实践

**下一部分**：进入中级阶段——掌握 Mock、异常测试、参数化测试等强大工具。
