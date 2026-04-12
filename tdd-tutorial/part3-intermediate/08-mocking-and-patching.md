# 第八章：Mock 与打桩

## 8.1 为什么需要 Mock

真实系统中，代码总是依赖外部组件：数据库、HTTP API、文件系统、时间……

```
ShoppingCart
    └── calls → PaymentGateway.charge()     # 真实 API 调用，有费用
    └── calls → EmailService.send()         # 真实邮件，影响用户
    └── calls → datetime.now()              # 时间不固定，测试不稳定
```

**Mock 的作用**：用可控的替代品代替真实依赖，让测试：
- **快**：不等待网络/数据库
- **稳定**：不受外部状态影响
- **隔离**：专注测试单元本身
- **可控**：模拟各种边界条件（超时、错误等）

---

## 8.2 unittest.mock 核心对象

```
unittest.mock
├── MagicMock      # 万能 Mock 对象（推荐默认选择）
├── Mock           # 基础 Mock（不自动创建魔术方法）
├── patch          # 装饰器/上下文管理器，替换命名空间中的对象
├── patch.object   # 替换对象的属性/方法
├── patch.dict     # 替换字典内容
├── patch.multiple # 同时 patch 多个
├── sentinel       # 唯一标记值
├── call           # 断言调用参数用
└── ANY            # 匹配任意值的通配符
```

---

## 8.3 MagicMock 基础

```python
from unittest.mock import MagicMock

# 创建 Mock 对象
mock = MagicMock()

# 访问任意属性 → 返回新的 MagicMock
mock.name         # MagicMock()
mock.nested.deep  # MagicMock()

# 调用 → 返回 MagicMock（默认）
mock()            # MagicMock()
mock(1, 2, key="val")

# 配置返回值
mock.return_value = 42
mock()            # 42

# 配置副作用（每次调用返回不同值）
mock.side_effect = [10, 20, 30]
mock()  # 10
mock()  # 20
mock()  # 30

# 配置副作用（抛出异常）
mock.side_effect = ValueError("connection refused")
mock()  # 抛出 ValueError
```

---

## 8.4 断言 Mock 的调用

```python
from unittest.mock import MagicMock, call

gateway = MagicMock()
gateway.charge("card_123", 50.0)
gateway.charge("card_456", 30.0)

# 是否被调用
gateway.charge.assert_called()                     # 至少一次
gateway.charge.assert_called_once()                # 恰好一次 → 失败（调用了两次）

# 最后一次调用的参数
gateway.charge.assert_called_with("card_456", 30.0)

# 特定参数被调用过
gateway.charge.assert_any_call("card_123", 50.0)

# 调用次数
assert gateway.charge.call_count == 2

# 所有调用的参数列表
assert gateway.charge.call_args_list == [
    call("card_123", 50.0),
    call("card_456", 30.0),
]

# 从未被调用
gateway.refund.assert_not_called()
```

---

## 8.5 patch：在正确的地方打桩

**核心原则**：patch 必须应用在**被测模块导入的位置**，而非定义位置。

```
# myapp/email.py
import smtplib           # smtplib 在这里定义

# myapp/order.py
from myapp.email import send_email   # send_email 在这里被导入
```

```python
# 错误：patch 定义位置
@patch('smtplib.SMTP')

# 正确：patch 使用位置（被测模块的命名空间）
@patch('myapp.order.send_email')
```

---

## 8.6 patch 的三种用法

### 用法一：装饰器

```python
from unittest.mock import patch, MagicMock
import unittest

class TestOrderService(unittest.TestCase):

    @patch('myapp.order.send_email')
    @patch('myapp.order.PaymentGateway')
    def test_place_order_sends_confirmation(self, MockGateway, mock_send_email):
        # 注意：装饰器从下到上，参数从左到右
        MockGateway.return_value.charge.return_value = {"status": "success"}

        order = OrderService().place_order(user_id=1, items=["apple"])

        mock_send_email.assert_called_once()
        call_args = mock_send_email.call_args
        self.assertIn("confirmation", call_args.args[0].lower())
```

### 用法二：上下文管理器（推荐）

```python
def test_payment_failure_raises_error(self):
    with patch('myapp.order.PaymentGateway') as MockGateway:
        MockGateway.return_value.charge.side_effect = PaymentError("card declined")

        with self.assertRaises(OrderError) as ctx:
            OrderService().place_order(user_id=1, items=["apple"])

        self.assertIn("payment", str(ctx.exception).lower())
```

### 用法三：setUp/tearDown 手动管理

```python
class TestOrderService(unittest.TestCase):

    def setUp(self):
        patcher = patch('myapp.order.PaymentGateway')
        self.MockGateway = patcher.start()
        self.addCleanup(patcher.stop)  # 比 tearDown 更安全

        self.MockGateway.return_value.charge.return_value = {"status": "ok"}

    def test_successful_order(self):
        result = OrderService().place_order(1, ["apple"])
        self.assertEqual(result.status, "completed")

    def test_order_calls_payment(self):
        OrderService().place_order(1, ["apple"])
        self.MockGateway.return_value.charge.assert_called_once()
```

---

## 8.7 patch.object：替换对象属性

```python
class TestUserService(unittest.TestCase):

    def test_get_user_from_cache(self):
        user_service = UserService()
        fake_user = User(id=1, name="Alice")

        with patch.object(user_service, 'cache') as mock_cache:
            mock_cache.get.return_value = fake_user

            result = user_service.get_user(1)

            self.assertEqual(result.name, "Alice")
            mock_cache.get.assert_called_once_with(1)
```

---

## 8.8 Mock 时间和随机

### Mock datetime

```python
from unittest.mock import patch
from datetime import datetime

class TestTimestampedLog(unittest.TestCase):

    def test_log_entry_has_current_timestamp(self):
        fixed_time = datetime(2024, 1, 15, 10, 30, 0)

        with patch('myapp.logger.datetime') as mock_dt:
            mock_dt.now.return_value = fixed_time

            entry = create_log_entry("test message")

            self.assertEqual(entry.timestamp, fixed_time)
            self.assertEqual(entry.message, "test message")
```

### Mock random

```python
def test_shuffle_uses_random(self):
    items = [1, 2, 3, 4, 5]
    with patch('random.shuffle') as mock_shuffle:
        shuffle_deck(items)
        mock_shuffle.assert_called_once_with(items)
```

---

## 8.9 spec 参数：防止 Mock 过度宽松

```python
# 没有 spec：访问不存在的方法不报错（危险！）
mock = MagicMock()
mock.nonexistent_method()   # 不报错，默默返回 MagicMock

# 有 spec：只允许真实接口上存在的方法
from myapp.payment import PaymentGateway

mock_gateway = MagicMock(spec=PaymentGateway)
mock_gateway.charge("card", 50.0)     # OK：charge 是真实方法
mock_gateway.nonexistent()            # AttributeError！

# spec_set 更严格：连属性赋值也检查
mock_strict = MagicMock(spec_set=PaymentGateway)
mock_strict.new_attribute = "value"   # AttributeError！
```

---

## 8.10 captor 模式：捕获调用参数

```python
def test_email_content_is_correct(self):
    with patch('myapp.order.send_email') as mock_send:
        OrderService().place_order(user_id=1, items=["apple"])

        # 捕获实际传入的参数
        actual_call = mock_send.call_args
        to_address = actual_call.args[0]      # 位置参数
        subject = actual_call.kwargs.get('subject')  # 关键字参数
        body = actual_call.kwargs.get('body')

        self.assertEqual(to_address, "user1@example.com")
        self.assertIn("Order Confirmation", subject)
        self.assertIn("apple", body)
```

---

## 8.11 常见 Mock 反模式

```python
# 反模式1：Mock 了但没断言
def test_bad(self):
    with patch('myapp.email.send') as mock_send:
        place_order(...)
    # 忘记 mock_send.assert_called_once()  ← 测试没有验证任何事

# 反模式2：过度 Mock，测试不再测试真实逻辑
def test_overMocked(self):
    with patch('myapp.order.calculate_total') as mock_calc:
        mock_calc.return_value = 100  # 把核心逻辑也 mock 掉了
        result = place_order(...)
    # 这个测试什么都没测

# 反模式3：Mock 了内部实现细节（过度耦合）
def test_internal_coupling(self):
    with patch.object(OrderService, '_internal_helper') as mock_helper:
        ...  # 测试耦合了私有方法，重构时会破碎
```

---

## 8.12 本章小结

- Mock 的目的：隔离依赖，使测试快速、稳定、可控
- `MagicMock`：万能替代品，支持属性访问、调用、迭代等
- `patch` 必须应用在**被测模块使用的命名空间**
- `spec` 参数防止对不存在方法的误调用
- 推荐用 `addCleanup(patcher.stop)` 而非 `tearDown`
- 避免：Mock 忘断言、过度 Mock、Mock 私有实现

**下一章**：深入异常与错误场景的测试技巧。
