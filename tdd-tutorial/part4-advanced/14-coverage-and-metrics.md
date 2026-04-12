# 第十四章：覆盖率与质量度量

## 14.1 覆盖率不是目标，是工具

> "100% 的覆盖率不等于 100% 的正确性。"

覆盖率的真正价值：**发现你没有测试到的代码**，而不是炫耀一个数字。

---

## 14.2 安装与基础用法

```bash
pip install coverage
```

### 命令行运行

```bash
# 运行测试并收集覆盖率
coverage run -m unittest discover tests/ -v

# 显示报告（终端）
coverage report -m

# 生成 HTML 报告（可交互浏览）
coverage html
open htmlcov/index.html

# 生成 XML 报告（CI 系统集成）
coverage xml -o coverage.xml
```

### 典型输出

```
Name                    Stmts   Miss  Cover   Missing
-----------------------------------------------------
myapp/cart.py              28      2    93%   45-46
myapp/payment.py           42      8    81%   23, 35-41
myapp/validators.py        15      0   100%
myapp/utils.py             10      5    50%   8-12
-----------------------------------------------------
TOTAL                      95     15    84%
```

---

## 14.3 覆盖率类型

### 语句覆盖（Statement Coverage）

默认模式，统计被执行的语句比例：

```python
def process(x):
    if x > 0:        # 行 1
        return "positive"  # 行 2
    return "non-positive"  # 行 3

# 只测试 x=1 → 语句覆盖 67%（行 3 未执行）
```

### 分支覆盖（Branch Coverage）

统计每个条件的 True/False 分支是否都被执行：

```bash
coverage run --branch -m unittest discover tests/
coverage report -m --show-missing
```

```python
def validate(x):
    if x is None:          # 分支 1T: x=None, 1F: x!=None
        raise ValueError()
    if x < 0:              # 分支 2T: x<0, 2F: x>=0
        raise ValueError()
    return x

# 只测试 x=1 → 语句覆盖 50%, 分支覆盖 25%（1F + 2F 被覆盖）
```

### 推荐：启用分支覆盖

```ini
# .coveragerc
[run]
branch = True
source = myapp
omit =
    */tests/*
    */migrations/*
    setup.py

[report]
show_missing = True
skip_covered = False
fail_under = 80

[html]
directory = htmlcov
```

---

## 14.4 配置 .coveragerc

```ini
[run]
branch = True
source = myapp          # 只统计这个包的覆盖率
omit =
    */tests/*           # 排除测试代码本身
    */venv/*            # 排除虚拟环境
    */migrations/*      # 排除数据库迁移
    */__init__.py       # 排除 __init__ 文件（通常无逻辑）
    setup.py

[report]
show_missing = True     # 显示未覆盖的行号
precision = 2           # 小数点精度
fail_under = 80         # 低于 80% 时报错（CI 检查用）
exclude_lines =
    pragma: no cover    # 用注释排除特定行
    def __repr__
    raise NotImplementedError
    if __name__ == .__main__.:
    class .*\(Protocol\):

[html]
directory = htmlcov
title = My Project Coverage Report
```

---

## 14.5 排除特定代码

```python
def debug_info(self):   # pragma: no cover
    """调试专用，不需要测试覆盖"""
    return f"Cart(items={self._items!r})"

class AbstractBase:
    def process(self):
        raise NotImplementedError   # pragma: no cover

if __name__ == '__main__':   # pragma: no cover
    main()
```

---

## 14.6 在 CI 中集成覆盖率检查

### GitHub Actions 示例

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -r requirements-dev.txt

      - name: Run tests with coverage
        run: |
          coverage run --branch -m unittest discover tests/ -v
          coverage report --fail-under=80
          coverage xml

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          file: coverage.xml
```

### Makefile 集成

```makefile
test:
	python -m unittest discover tests/ -v

coverage:
	coverage run --branch -m unittest discover tests/
	coverage report -m
	coverage html

coverage-check:
	coverage run --branch -m unittest discover tests/
	coverage report --fail-under=80

.PHONY: test coverage coverage-check
```

---

## 14.7 覆盖率的正确解读

### 高覆盖率但测试无效

```python
# 测试覆盖了代码，但没有验证任何行为！
def test_process_payment(self):
    process_payment(user_id=1, amount=50.0)
    # 没有 assert → 覆盖率 100%，但测试毫无价值
```

### 低覆盖率指向的问题

```
myapp/error_handler.py   12    10    17%   15-28
```

可能的原因：
- 错误处理代码未被测试（需要补充错误场景测试）
- 死代码（可以删除）
- 第三方代码（应该在 `.coveragerc` 中排除）

---

## 14.8 其他质量度量

### 测试质量度量

```python
# 统计测试数量和通过率
result = unittest.TextTestRunner().run(suite)
print(f"Total: {result.testsRun}")
print(f"Failed: {len(result.failures)}")
print(f"Errors: {len(result.errors)}")
print(f"Pass rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun:.1%}")
```

### 测试速度分析

```python
import time
import unittest

class TimedTestResult(unittest.TextTestResult):
    """记录每个测试的运行时间"""

    def startTest(self, test):
        self._start_time = time.perf_counter()
        super().startTest(test)

    def stopTest(self, test):
        elapsed = time.perf_counter() - self._start_time
        if elapsed > 0.1:  # 超过 100ms 的测试
            self.stream.write(f"  ⚠ SLOW: {elapsed:.3f}s - {test}\n")
        super().stopTest(test)
```

### 突变测试（Mutation Testing）

突变测试是比覆盖率更强的测试质量度量：

```bash
pip install mutmut
mutmut run --paths-to-mutate myapp/
mutmut results
mutmut html
```

突变测试的原理：自动修改生产代码（如把 `>` 改成 `>=`），检查是否有测试失败。
- **存活的突变体**：没有测试能检测到这个变化 → 测试不充分
- **被杀死的突变体**：有测试捕获了这个变化 → 测试有效

```
Mutation score = killed / (killed + survived) * 100%
```

---

## 14.9 覆盖率目标设定建议

| 代码类型 | 建议覆盖率 | 原因 |
|----------|------------|------|
| 核心业务逻辑 | ≥ 90% | 错误代价高 |
| 工具函数 | ≥ 85% | 被广泛复用 |
| API 层 | ≥ 80% | 契约必须可靠 |
| 配置/脚手架 | ≥ 60% | 变化频繁，测试成本高 |
| UI/展示层 | ≥ 50% | 视觉逻辑难以自动化 |

---

## 14.10 本章小结

- `coverage run --branch` + `coverage report -m` 是核心工作流
- `.coveragerc` 配置 `fail_under` 在 CI 中强制最低覆盖率
- 覆盖率是**发现盲区**的工具，不是**衡量质量**的唯一标准
- 高覆盖率 + 有效断言 = 真正有价值的测试
- 突变测试（`mutmut`）是评估测试质量的进阶工具

**下一章**：TDD 最佳实践与反模式——成为高阶 TDD 实践者的终极指南。
