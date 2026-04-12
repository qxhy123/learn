# 第十九章：Mock 内部机制深度解析

## 19.1 为什么要理解 Mock 内部

不理解 Mock 工作原理时，你会遇到各种"奇怪"的行为：
- `assert_called_once_with` 明明应该匹配但报错
- patch 的目标不对导致 Mock 没生效
- `side_effect` 的执行顺序与预期不符
- `spec` 检查了不该检查的东西

理解原理，这些问题都变得透明。

---

## 19.2 MagicMock 的核心机制：`__getattr__`

```python
from unittest.mock import MagicMock

m = MagicMock()

# 访问任意属性 → 返回子 MagicMock
m.foo          # MagicMock（自动创建）
m.foo.bar      # MagicMock（链式自动创建）
m.foo.bar.baz  # MagicMock

# 调用 → 返回 MagicMock（默认）
m()            # MagicMock
m(1, 2)        # MagicMock

# 重要：同一属性每次访问返回同一个对象
assert m.foo is m.foo   # True！子 Mock 被缓存
```

### 内部实现原理

```python
# 简化版 Mock 实现（理解原理用）
class SimpleMock:
    def __init__(self):
        self._children = {}
        self._calls = []
        self.return_value = SimpleMock()

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        if name not in self._children:
            self._children[name] = SimpleMock()
        return self._children[name]

    def __call__(self, *args, **kwargs):
        self._calls.append((args, kwargs))
        return self.return_value

    def assert_called_with(self, *args, **kwargs):
        if not self._calls:
            raise AssertionError("Not called")
        last_args, last_kwargs = self._calls[-1]
        assert last_args == args and last_kwargs == kwargs
```

---

## 19.3 call 对象的结构

```python
from unittest.mock import call, MagicMock

m = MagicMock()
m(1, 2, key="val")
m.method(3, x=4)

# call_args：最后一次调用
print(m.call_args)
# call(1, 2, key='val')

# call_args.args 和 call_args.kwargs
print(m.call_args.args)    # (1, 2)
print(m.call_args.kwargs)  # {'key': 'val'}

# call_args_list：所有调用
print(m.call_args_list)
# [call(1, 2, key='val')]

# 链式调用记录
print(m.method_calls)
# [call.method(3, x=4)]

# 全部调用（包括自身和子 Mock 的调用）
print(m.mock_calls)
# [call(1, 2, key='val'), call.method(3, x=4)]
```

### call 对象的比较

```python
from unittest.mock import call, ANY

# 精确匹配
m.charge(user_id=1, amount=50.0)
m.charge.assert_called_with(user_id=1, amount=50.0)  # 通过

# ANY 通配符：忽略特定参数
m.charge.assert_called_with(user_id=ANY, amount=50.0)  # 通过
m.charge.assert_called_with(user_id=1, amount=ANY)     # 通过

# 手动构建 call 对象进行比较
expected_calls = [
    call(user_id=1, amount=50.0),
    call(user_id=2, amount=30.0),
]
m.charge.assert_has_calls(expected_calls)              # 验证包含这些调用（顺序）
m.charge.assert_has_calls(expected_calls, any_order=True)  # 忽略顺序
```

---

## 19.4 side_effect 的全部用法

```python
m = MagicMock()

# 用法一：固定异常
m.side_effect = ValueError("boom")
m()   # 抛出 ValueError

# 用法二：异常类（每次调用都抛出新实例）
m.side_effect = ConnectionError
m()   # 抛出 ConnectionError()

# 用法三：可迭代（每次调用返回/抛出下一个元素）
m.side_effect = [10, 20, ValueError("end"), 30]
m()   # 10
m()   # 20
m()   # 抛出 ValueError
m()   # 30
m()   # StopIteration（序列耗尽）

# 用法四：函数（完全自定义逻辑）
def smart_side_effect(n):
    if n < 0:
        raise ValueError(f"negative: {n}")
    return n * 2

m.side_effect = smart_side_effect
m(5)   # 10
m(-1)  # 抛出 ValueError

# 清除 side_effect
m.side_effect = None
m()    # 恢复返回 return_value
```

### side_effect 与 return_value 的优先级

```python
m = MagicMock()
m.return_value = 42
m.side_effect = lambda: 100

m()  # 100（side_effect 优先）

m.side_effect = None
m()  # 42（回到 return_value）
```

---

## 19.5 spec vs spec_set vs autospec

### spec：接口检查

```python
class MyClass:
    def method_a(self): ...
    attribute_x = 5

from unittest.mock import MagicMock

# 没有 spec
plain = MagicMock()
plain.nonexistent_method()  # OK，不报错

# spec：只允许真实接口上的属性/方法
specced = MagicMock(spec=MyClass)
specced.method_a()              # OK
specced.nonexistent_method()    # AttributeError！

# spec_set：更严格，连属性赋值也检查
strict = MagicMock(spec_set=MyClass)
strict.new_attr = "value"       # AttributeError！
```

### autospec：深度接口检查（推荐）

`spec` 只检查方法是否存在，`autospec` 还检查**签名**：

```python
from unittest.mock import create_autospec

class Calculator:
    def add(self, a: int, b: int) -> int: ...
    def divide(self, a: int, b: int) -> float: ...

# spec：不检查签名
specced = MagicMock(spec=Calculator)
specced.add(1, 2, 3, 4, extra="wrong")  # 不报错！签名没被检查

# autospec：检查签名
auto = create_autospec(Calculator)
auto.add(1, 2)            # OK
auto.add(1, 2, 3)         # TypeError：too many positional arguments
auto.add("str", "str")    # OK（Python 不做类型检查，但参数数量会检查）
```

### 什么时候用 autospec

```python
# 原则：只要你在 Mock 类的实例，优先用 autospec
with patch('myapp.payment.PaymentGateway', autospec=True) as MockGW:
    instance = MockGW.return_value
    # instance 的所有方法都有正确的签名检查
    instance.charge(amount=100)  # OK
    instance.charge(wrong_param=1)  # TypeError！提前发现问题
```

---

## 19.6 patch 的目标定位原理

这是最常见的困惑来源：**patch 要打在使用位置，不是定义位置**。

```python
# myapp/utils.py
import os

def get_home():
    return os.path.expanduser("~")


# myapp/reporter.py
from myapp.utils import get_home   # ← get_home 被绑定到这个模块的命名空间

def generate_report():
    home = get_home()
    return f"Report saved to {home}/reports"
```

```python
# 错误：patch 了 utils 中的定义，但 reporter 已经有了自己的引用
with patch('myapp.utils.get_home') as mock:  # 不生效！
    result = generate_report()

# 正确：patch reporter 模块命名空间中的引用
with patch('myapp.reporter.get_home') as mock:  # 生效！
    mock.return_value = "/mock/home"
    result = generate_report()
    assert "mock/home" in result
```

### 规则汇总

| 导入方式 | patch 目标 |
|---------|----------|
| `import os` + `os.path.join()` | `patch('myapp.module.os.path.join')` |
| `from os.path import join` + `join()` | `patch('myapp.module.join')` |
| `from myapp.utils import helper` + `helper()` | `patch('myapp.module.helper')` |
| `import myapp.utils` + `myapp.utils.helper()` | `patch('myapp.utils.helper')` |

---

## 19.7 configure_mock：批量配置

```python
m = MagicMock()

# 逐个配置（冗长）
m.name = "Alice"
m.age = 30
m.address.city = "London"
m.get_score.return_value = 95

# 批量配置（更清晰）
m.configure_mock(
    name="Alice",
    age=30,
    **{"address.city": "London"},  # 嵌套属性用点号
    **{"get_score.return_value": 95}
)

# 或者在创建时配置
m = MagicMock(
    name="Alice",
    age=30,
    **{"get_score.return_value": 95}
)
```

---

## 19.8 reset_mock：重置调用记录

```python
m = MagicMock()
m(1, 2)
m(3, 4)

print(m.call_count)  # 2

m.reset_mock()       # 重置调用记录

print(m.call_count)  # 0
m.assert_not_called()  # 通过

# reset_mock 不重置 return_value 和 side_effect（默认）
m.return_value = 42
m.reset_mock()
print(m.return_value)  # 42（未被重置）

# 重置所有（包括 return_value 和 side_effect）
m.reset_mock(return_value=True, side_effect=True)
```

---

## 19.9 MagicMock 的魔术方法支持

`MagicMock` vs `Mock` 的核心区别：MagicMock 预实现了所有魔术方法：

```python
from unittest.mock import MagicMock, Mock

# MagicMock：支持魔术方法
mm = MagicMock()
mm.__len__.return_value = 5
len(mm)   # 5

mm.__iter__.return_value = iter([1, 2, 3])
list(mm)  # [1, 2, 3]

mm.__contains__.return_value = True
"key" in mm  # True

mm.__enter__.return_value = mm
mm.__exit__.return_value = False
with mm as ctx:   # 可以用作上下文管理器
    pass

# Mock：不自动支持魔术方法
m = Mock()
len(m)   # TypeError: object of type 'Mock' has no len()
```

---

## 19.10 patch.object 的进阶用法

```python
class DatabasePool:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls._create()
        return cls._instance

    @classmethod
    def _create(cls):
        return {"conn": "real_db"}


class TestDatabasePool(unittest.TestCase):

    def test_singleton_returns_same_instance(self):
        # patch 类方法
        with patch.object(DatabasePool, '_create', return_value={"conn": "fake"}):
            DatabasePool._instance = None  # 重置单例
            inst1 = DatabasePool.get_instance()
            inst2 = DatabasePool.get_instance()

        self.assertIs(inst1, inst2)
        self.assertEqual(inst1["conn"], "fake")

    def test_patch_property(self):
        """patch 属性（property）"""
        with patch.object(type(DatabasePool), 'connection_count',
                         new_callable=lambda: property(lambda self: 5)):
            count = DatabasePool.connection_count
            self.assertEqual(count, 5)
```

---

## 19.11 断言方法完整参考

```python
m = MagicMock()
m(1); m(2); m(3)

# 存在性
m.assert_called()                        # 至少调用一次
m.assert_called_once()                   # 恰好一次（失败：3次）
m.assert_not_called()                    # 从未调用（失败）

# 参数
m.assert_called_with(3)                  # 最后一次调用是 m(3)
m.assert_called_once_with(3)             # 恰好一次且参数是 3（失败：3次）
m.assert_any_call(2)                     # 曾用 m(2) 调用过（通过）

# 调用序列
m.assert_has_calls([call(1), call(2), call(3)])          # 按顺序包含
m.assert_has_calls([call(3), call(1)], any_order=True)  # 任意顺序包含

# 计数
assert m.call_count == 3

# 调用参数详情
assert m.call_args == call(3)           # 最后一次调用
assert m.call_args.args == (3,)
assert m.call_args_list == [call(1), call(2), call(3)]
```

---

## 19.12 本章小结

- `MagicMock` 通过 `__getattr__` 动态创建子 Mock，子 Mock 被缓存
- `call` 对象记录调用参数，`ANY` 通配符忽略特定参数
- `side_effect` 优先于 `return_value`，支持异常、可迭代、函数三种形式
- `autospec` > `spec` > 无 spec：越严格，越能提前发现接口不匹配
- patch 必须打在**使用位置**（被测模块的命名空间），不是定义位置
- `MagicMock` 预实现了魔术方法（`len`、`iter`、`with` 等），`Mock` 不行

**下一章**：TDD 如何驱动出经典设计模式——Strategy、Observer、Command 从测试中自然浮现。
