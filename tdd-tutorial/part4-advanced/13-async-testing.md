# 第十三章：异步代码测试

## 13.1 异步测试的挑战

```python
# 问题：普通 unittest 无法正确运行协程
class TestAsync(unittest.TestCase):
    def test_fetch_user(self):
        result = fetch_user(1)   # 返回 coroutine，不是结果！
        self.assertEqual(result.name, "Alice")  # 永远失败
```

异步代码需要事件循环来驱动，`unittest` 本身不提供这一能力。有三种解决方案：

---

## 13.2 方案一：asyncio.run()（Python 3.7+）

最简单的方案，适合无复杂夹具的场景：

```python
import asyncio
import unittest

async def fetch_user(user_id):
    await asyncio.sleep(0.01)  # 模拟网络延迟
    return {"id": user_id, "name": "Alice"}

class TestAsyncFetch(unittest.TestCase):

    def test_fetch_user_returns_correct_name(self):
        result = asyncio.run(fetch_user(1))
        self.assertEqual(result["name"], "Alice")

    def test_fetch_multiple_users(self):
        async def _test():
            users = await asyncio.gather(
                fetch_user(1),
                fetch_user(2),
            )
            return users

        users = asyncio.run(_test())
        self.assertEqual(len(users), 2)
```

**限制**：每次调用 `asyncio.run()` 都创建新的事件循环，夹具复用困难。

---

## 13.3 方案二：IsolatedAsyncioTestCase（Python 3.8+，推荐）

Python 3.8 引入了 `IsolatedAsyncioTestCase`，原生支持 `async def` 测试方法：

```python
import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

class TestAsyncUserService(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        """异步 setUp：可以 await"""
        self.service = AsyncUserService()
        await self.service.connect()

    async def asyncTearDown(self):
        """异步 tearDown：可以 await"""
        await self.service.disconnect()

    async def test_get_user_by_id(self):
        user = await self.service.get_user(1)
        self.assertEqual(user.id, 1)

    async def test_create_user(self):
        user = await self.service.create(name="Bob", email="bob@example.com")
        self.assertIsNotNone(user.id)
        self.assertEqual(user.name, "Bob")
```

生命周期：
```
setUpClass (同步) → asyncSetUp → async test → asyncTearDown → tearDownClass (同步)
```

---

## 13.4 Mock 异步函数

普通 `MagicMock` 不能作为协程使用，需要 `AsyncMock`：

```python
from unittest.mock import AsyncMock, patch

class TestAsyncOrderService(unittest.IsolatedAsyncioTestCase):

    async def test_place_order_calls_payment(self):
        # AsyncMock：可以被 await
        mock_payment = AsyncMock()
        mock_payment.charge.return_value = {"status": "success", "txn": "tx_123"}

        service = AsyncOrderService(payment=mock_payment)
        await service.place_order(user_id=1, amount=50.0)

        mock_payment.charge.assert_called_once_with(user_id=1, amount=50.0)

    async def test_payment_failure_raises_order_error(self):
        mock_payment = AsyncMock()
        mock_payment.charge.side_effect = PaymentError("declined")

        service = AsyncOrderService(payment=mock_payment)
        with self.assertRaises(OrderError):
            await service.place_order(user_id=1, amount=50.0)

    async def test_patch_async_function(self):
        with patch('myapp.service.fetch_user', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = {"id": 1, "name": "Alice"}

            result = await get_user_profile(1)

            mock_fetch.assert_awaited_once_with(1)
            self.assertEqual(result["name"], "Alice")
```

---

## 13.5 AsyncMock 的断言方法

```python
mock = AsyncMock()
await mock(1, 2, key="val")
await mock(3, 4)

# 调用次数
mock.assert_awaited()                  # 至少被 await 过
mock.assert_awaited_once()             # 恰好 await 一次（失败：2次）
mock.assert_awaited_once_with(1, 2, key="val")  # 失败：调用了两次
mock.assert_awaited_with(3, 4)         # 最后一次 await 的参数

# 等同于 call_args_list
print(mock.await_args_list)
# [call(1, 2, key='val'), call(3, 4)]
```

---

## 13.6 测试并发行为

```python
class TestConcurrentRequests(unittest.IsolatedAsyncioTestCase):

    async def test_concurrent_tasks_complete_independently(self):
        """验证多个并发任务互不干扰"""
        results = []

        async def task(n):
            await asyncio.sleep(0.01 * n)
            results.append(n)
            return n

        outputs = await asyncio.gather(task(3), task(1), task(2))

        # gather 保持参数顺序
        self.assertEqual(outputs, [3, 1, 2])
        # 实际完成顺序按延迟（1, 2, 3）
        self.assertEqual(sorted(results), [1, 2, 3])

    async def test_timeout_raises_error(self):
        """验证超时处理"""
        async def slow_operation():
            await asyncio.sleep(10)  # 很慢
            return "done"

        with self.assertRaises(asyncio.TimeoutError):
            await asyncio.wait_for(slow_operation(), timeout=0.1)

    async def test_gather_fails_fast_on_error(self):
        """验证 gather 遇到异常的行为"""
        async def failing_task():
            raise ValueError("boom")

        async def ok_task():
            return "ok"

        with self.assertRaises(ValueError):
            await asyncio.gather(failing_task(), ok_task())
```

---

## 13.7 测试事件循环中的回调

```python
class TestEventLoop(unittest.IsolatedAsyncioTestCase):

    async def test_callback_invoked_after_delay(self):
        received = []

        def on_event(data):
            received.append(data)

        loop = asyncio.get_event_loop()
        loop.call_later(0.05, on_event, {"type": "tick"})

        await asyncio.sleep(0.1)  # 等待回调执行

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0]["type"], "tick")
```

---

## 13.8 方案三：手动事件循环（兼容旧版本）

对于 Python 3.7 以下或需要精确控制循环时：

```python
class TestAsyncManual(unittest.TestCase):

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()
        asyncio.set_event_loop(None)

    def run_async(self, coro):
        """辅助方法：在测试中运行协程"""
        return self.loop.run_until_complete(coro)

    def test_async_function(self):
        result = self.run_async(fetch_user(1))
        self.assertEqual(result["id"], 1)
```

---

## 13.9 异步生成器测试

```python
class TestAsyncGenerator(unittest.IsolatedAsyncioTestCase):

    async def test_stream_yields_items(self):
        async def number_stream(count):
            for i in range(count):
                await asyncio.sleep(0)
                yield i

        results = []
        async for item in number_stream(5):
            results.append(item)

        self.assertEqual(results, [0, 1, 2, 3, 4])

    async def test_mock_async_generator(self):
        """Mock 异步生成器"""
        from unittest.mock import MagicMock

        async def async_gen_side_effect(*args, **kwargs):
            for item in [1, 2, 3]:
                yield item

        mock_stream = MagicMock()
        mock_stream.__aiter__ = async_gen_side_effect

        results = [item async for item in mock_stream]
        self.assertEqual(results, [1, 2, 3])
```

---

## 13.10 版本兼容性对照

| 方案 | Python 版本 | 复杂夹具 | 推荐场景 |
|------|------------|----------|----------|
| `asyncio.run()` | 3.7+ | 困难 | 简单协程测试 |
| `IsolatedAsyncioTestCase` | 3.8+ | 完整支持 | 所有异步测试（推荐） |
| 手动事件循环 | 所有版本 | 手动管理 | 需要精确控制或兼容旧版本 |

---

## 13.11 本章小结

- Python 3.8+ 推荐使用 `IsolatedAsyncioTestCase`，原生支持 `async def` 测试
- `AsyncMock` 是异步函数的 Mock，支持 `assert_awaited*` 系列断言
- `asyncio.gather` + `asyncio.wait_for` 可测试并发和超时行为
- `asyncSetUp/asyncTearDown` 管理异步夹具生命周期

**下一章**：覆盖率分析与质量度量——用数据指导测试策略。
