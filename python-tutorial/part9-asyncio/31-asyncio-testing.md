# 第31章：asyncio测试实战

> 测试异步代码有几个地方比同步测试难：事件循环的生命周期管理、时间控制、模拟协程、并发行为的确定性、取消和超时的验证。如果不了解这些问题，你写的测试可能"能跑通"却根本没有测到你以为测到的东西。

---

## 学习目标

完成本章学习后，你将能够：

1. 使用 `pytest-asyncio` 编写结构清晰的异步测试
2. 控制事件循环范围（function/class/module/session）
3. 使用 `unittest.mock` 和 `AsyncMock` 模拟协程
4. 控制测试中的时间（`freezegun`、手动 `sleep` mock）
5. 测试并发行为、取消传播和超时逻辑
6. 识别并避免异步测试中的常见陷阱

---

## 正文内容

## 31.1 为什么异步测试需要特别对待

同步测试直接调用函数取返回值。异步测试面临的额外问题：

1. **事件循环生命周期**：每个测试需要一个事件循环，测试结束后要正确关闭
2. **协程的惰性**：`async def` 函数调用返回协程对象，忘记 `await` 不会报错但什么也没测到
3. **时间依赖**：`asyncio.sleep()` 让测试变慢，而且难以控制时序
4. **并发不确定性**：多个任务的执行顺序在测试里需要确定性控制
5. **取消和异常**：需要验证取消时清理逻辑是否真正执行

---

## 31.2 `pytest-asyncio` 基础配置

### 31.2.1 安装

```bash
pip install pytest-asyncio
```

### 31.2.2 配置模式

在 `pyproject.toml` 或 `pytest.ini` 里配置：

```toml
# pyproject.toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

或者在 `pytest.ini`：

```ini
[pytest]
asyncio_mode = auto
```

`asyncio_mode = "auto"` 让所有 `async def` 测试函数自动被识别为异步测试，无需手动加 `@pytest.mark.asyncio`。

### 31.2.3 最简单的异步测试

```python
# test_basic.py
import asyncio
import pytest


async def fetch_data(delay: float) -> str:
    await asyncio.sleep(delay)
    return "data"


async def test_fetch_data():
    result = await fetch_data(0.01)
    assert result == "data"
```

### 31.2.4 手动标记（不用 auto 模式时）

```python
import pytest

@pytest.mark.asyncio
async def test_something():
    result = await some_coroutine()
    assert result == expected
```

---

## 31.3 事件循环范围控制

`pytest-asyncio` 允许控制事件循环的复用范围。

### 31.3.1 默认：每个测试一个 loop（function 范围）

```python
import pytest


@pytest.fixture
async def setup_resource():
    resource = await create_resource()
    yield resource
    await resource.close()


async def test_one(setup_resource):
    result = await setup_resource.do_something()
    assert result is not None
```

每个测试函数都有独立的事件循环，隔离性最好，推荐作为默认选择。

### 31.3.2 模块级复用 loop（适合集成测试）

```python
import pytest


pytestmark = pytest.mark.asyncio(loop_scope="module")


@pytest.fixture(scope="module")
async def db_connection():
    conn = await connect_db()
    yield conn
    await conn.close()


async def test_query_one(db_connection):
    result = await db_connection.execute("SELECT 1")
    assert result is not None


async def test_query_two(db_connection):
    result = await db_connection.execute("SELECT 2")
    assert result is not None
```

模块级 loop 适合需要**共享昂贵资源**（数据库连接、模型加载）的集成测试。

### 31.3.3 session 级复用 loop

```python
# conftest.py
import pytest

@pytest.fixture(scope="session")
def event_loop_policy():
    return asyncio.DefaultEventLoopPolicy()
```

注意：session 级 loop 共享范围越大，测试之间的隔离越差，谨慎使用。

---

## 31.4 模拟协程：`AsyncMock`

Python 3.8+ 内置了 `AsyncMock`，专门用于模拟 `async def` 函数。

### 31.4.1 基本用法

```python
from unittest.mock import AsyncMock, patch
import pytest


async def service_call(url: str) -> dict:
    # 假设这是一个真实的 HTTP 请求
    ...


async def process(url: str) -> str:
    data = await service_call(url)
    return data["result"]


async def test_process():
    with patch("mymodule.service_call", new_callable=AsyncMock) as mock:
        mock.return_value = {"result": "mocked_result"}
        result = await process("https://example.com")
        assert result == "mocked_result"
        mock.assert_awaited_once_with("https://example.com")
```

### 31.4.2 `AsyncMock` vs `MagicMock` 的区别

```python
from unittest.mock import MagicMock, AsyncMock

# 错误：用 MagicMock 模拟协程
sync_mock = MagicMock(return_value="value")
# await sync_mock()  ← TypeError: object str can't be used in 'await' expression

# 正确：用 AsyncMock
async_mock = AsyncMock(return_value="value")
result = await async_mock()   # ← 正确，result == "value"
```

### 31.4.3 模拟异步上下文管理器

```python
from unittest.mock import AsyncMock, MagicMock, patch


async def test_async_context_manager():
    mock_conn = MagicMock()
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=None)
    mock_conn.execute = AsyncMock(return_value=[{"id": 1}])

    with patch("mymodule.get_connection", return_value=mock_conn):
        async with get_connection() as conn:
            result = await conn.execute("SELECT * FROM users")
            assert result == [{"id": 1}]
```

### 31.4.4 模拟异步迭代器

```python
from unittest.mock import AsyncMock


class AsyncIteratorMock:
    def __init__(self, items):
        self.items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.items)
        except StopIteration:
            raise StopAsyncIteration


async def test_async_iterator():
    mock_stream = AsyncIteratorMock(["line1\n", "line2\n", "line3\n"])

    lines = []
    async for line in mock_stream:
        lines.append(line.strip())

    assert lines == ["line1", "line2", "line3"]
```

### 31.4.5 验证 mock 调用

```python
from unittest.mock import AsyncMock


async def test_mock_assertions():
    mock = AsyncMock()
    mock.return_value = "ok"

    await mock("arg1", key="val")
    await mock("arg2")

    assert mock.await_count == 2
    mock.assert_awaited_with("arg2")                    # 最后一次调用
    mock.assert_any_await("arg1", key="val")            # 任意一次调用
    mock.assert_awaited_once_with("arg2")               # 恰好一次（会失败，因为调用了两次）
```

---

## 31.5 时间控制

### 31.5.1 用 `asyncio.sleep` mock 加速测试

最简单的方式是直接 mock `asyncio.sleep`：

```python
from unittest.mock import patch, AsyncMock
import asyncio


async def retry(fn, retries=3, delay=1.0):
    for attempt in range(retries):
        try:
            return await fn()
        except Exception:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(delay)


async def test_retry_eventually_succeeds():
    call_count = 0

    async def flaky():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("not ready")
        return "success"

    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = await retry(flaky, retries=3, delay=1.0)

    assert result == "success"
    assert call_count == 3
    # 测试瞬间完成，没有真正等待 1 秒
```

### 31.5.2 使用 `freezegun` 控制 `time.time()` / `datetime.now()`

对于依赖 `time.monotonic()` 或 `datetime.now()` 的代码：

```bash
pip install freezegun
```

```python
from freezegun import freeze_time
from datetime import datetime


@freeze_time("2024-01-15 10:00:00")
async def test_timestamp_behavior():
    from mymodule import create_event_with_timestamp
    event = await create_event_with_timestamp()
    assert event.created_at == datetime(2024, 1, 15, 10, 0, 0)
```

### 31.5.3 手动控制事件循环时间（高级）

对于需要精确控制 `asyncio` 内部时间的场景，可以用 `asyncio.get_event_loop().time()` 的 mock：

```python
import asyncio
from unittest.mock import patch


async def test_timeout_boundary():
    """测试正好在超时临界点的行为"""
    real_time = asyncio.get_event_loop().time()

    async def slow_operation():
        await asyncio.sleep(0.1)
        return "done"

    # 给 0.15s 超时，操作需要 0.1s，应该成功
    async with asyncio.timeout(0.15):
        result = await slow_operation()
    assert result == "done"
```

---

## 31.6 测试并发行为

### 31.6.1 测试任务真正并发执行

```python
import asyncio
import time


async def slow_op(name: str, delay: float) -> str:
    await asyncio.sleep(delay)
    return name


async def test_tasks_run_concurrently():
    start = time.monotonic()

    results = await asyncio.gather(
        slow_op("A", 0.1),
        slow_op("B", 0.1),
        slow_op("C", 0.1),
    )

    elapsed = time.monotonic() - start

    assert set(results) == {"A", "B", "C"}
    assert elapsed < 0.2   # 并发：约 0.1s，而不是 0.3s
```

### 31.6.2 测试执行顺序的确定性

```python
import asyncio


async def test_ordering_with_events():
    """验证操作按预期顺序执行"""
    order = []
    gate = asyncio.Event()

    async def first():
        order.append("start-first")
        gate.set()
        order.append("end-first")

    async def second():
        await gate.wait()
        order.append("second-after-gate")

    await asyncio.gather(second(), first())

    assert order == ["start-first", "end-first", "second-after-gate"]
```

### 31.6.3 测试生产者/消费者交互

```python
import asyncio


async def test_producer_consumer():
    queue: asyncio.Queue[int] = asyncio.Queue(maxsize=3)
    results = []

    async def producer():
        for i in range(5):
            await queue.put(i)
        await queue.put(None)  # 哨兵

    async def consumer():
        while True:
            item = await queue.get()
            queue.task_done()
            if item is None:
                break
            results.append(item)

    await asyncio.gather(producer(), consumer())

    assert results == [0, 1, 2, 3, 4]
```

---

## 31.7 测试取消和异常行为

### 31.7.1 验证任务可以被取消

```python
import asyncio
import pytest


async def cancellable_worker():
    cleanup_done = False
    try:
        while True:
            await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        cleanup_done = True
        raise
    finally:
        assert cleanup_done   # 确认清理逻辑执行了


async def test_cancellation_cleanup():
    task = asyncio.create_task(cancellable_worker())
    await asyncio.sleep(0.05)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert task.cancelled()
```

### 31.7.2 验证取消不被吞掉

```python
import asyncio
import pytest


async def bad_worker():
    try:
        await asyncio.sleep(10)
    except asyncio.CancelledError:
        pass   # ← 吞掉取消！


async def good_worker():
    try:
        await asyncio.sleep(10)
    except asyncio.CancelledError:
        # 清理...
        raise   # ← 正确：重新抛出


async def test_cancellation_not_swallowed():
    task = asyncio.create_task(bad_worker())
    await asyncio.sleep(0.01)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    # bad_worker 吞掉了取消，task 不会是 cancelled 状态
    assert not task.cancelled()   # 说明被吞掉了
    assert task.done()


async def test_cancellation_propagates():
    task = asyncio.create_task(good_worker())
    await asyncio.sleep(0.01)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert task.cancelled()   # 正确传播
```

### 31.7.3 验证超时行为

```python
import asyncio
import pytest


async def slow_operation():
    await asyncio.sleep(1.0)
    return "result"


async def test_timeout_raises():
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.01):
            await slow_operation()


async def test_timeout_not_triggered_when_fast():
    async def fast_operation():
        await asyncio.sleep(0.001)
        return "fast result"

    async with asyncio.timeout(0.1):
        result = await fast_operation()
    assert result == "fast result"
```

### 31.7.4 测试 `TaskGroup` 的异常传播

```python
import asyncio
import pytest


async def test_task_group_exception_propagation():
    """验证 TaskGroup 里任一任务失败会取消其他任务"""
    started = []
    cancelled = []

    async def successful_task(name: str):
        started.append(name)
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.append(name)
            raise

    async def failing_task():
        await asyncio.sleep(0.01)
        raise ValueError("task failed")

    with pytest.raises(ExceptionGroup) as exc_info:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(successful_task("A"))
            tg.create_task(successful_task("B"))
            tg.create_task(failing_task())

    assert "A" in cancelled or "B" in cancelled   # 至少一个被取消
    assert len(exc_info.value.exceptions) == 1
    assert isinstance(exc_info.value.exceptions[0], ValueError)
```

---

## 31.8 测试 `asyncio.Event`、`Lock`、`Queue`

### 31.8.1 测试 Event 广播

```python
import asyncio


async def test_event_wakes_all_waiters():
    event = asyncio.Event()
    woken = []

    async def waiter(name: str):
        await event.wait()
        woken.append(name)

    tasks = [asyncio.create_task(waiter(f"w{i}")) for i in range(4)]
    await asyncio.sleep(0)   # 让 waiter 们先挂起

    assert len(woken) == 0   # event 未触发，没人醒
    event.set()
    await asyncio.gather(*tasks)

    assert set(woken) == {"w0", "w1", "w2", "w3"}   # 全部被唤醒
```

### 31.8.2 测试 Lock 互斥性

```python
import asyncio


async def test_lock_mutual_exclusion():
    lock = asyncio.Lock()
    inside = []

    async def critical_section(name: str):
        async with lock:
            inside.append(f"enter-{name}")
            await asyncio.sleep(0.01)
            # 在持有锁期间，其他协程不应进入
            assert len([x for x in inside if x.startswith("enter-")]) == 1
            inside.append(f"exit-{name}")

    await asyncio.gather(
        critical_section("A"),
        critical_section("B"),
        critical_section("C"),
    )

    # 验证进出配对，说明每次只有一个在临界区
    pairs = [(inside[i], inside[i+1]) for i in range(0, len(inside), 2)]
    for enter, exit_ in pairs:
        name = enter.split("-")[1]
        assert exit_ == f"exit-{name}"
```

---

## 31.9 常见异步测试陷阱

### 陷阱一：忘记 `await`，测试永远通过

```python
# 错误：这个测试永远通过，因为没有 await
async def test_bug():
    some_coroutine()   # ← 只是创建了协程对象，没有运行
    assert True        # ← 永远通过，但什么也没测到
```

解决方案：启用 Python 警告，或使用 `asyncio` 的 `PYTHONASYNCIODEBUG=1` 环境变量：

```bash
PYTHONASYNCIODEBUG=1 pytest
```

### 陷阱二：`asyncio.sleep(0)` 不等价于"等所有任务完成"

```python
# 错误：以为 sleep(0) 之后所有任务都完成了
async def test_wrong():
    task = asyncio.create_task(slow_op())
    await asyncio.sleep(0)   # 只让出一次控制权
    assert task.done()       # ← 可能失败，slow_op 还没完成

# 正确：
async def test_right():
    task = asyncio.create_task(slow_op())
    result = await task      # 等到真正完成
    assert result == expected
```

### 陷阱三：在测试之间共享事件循环导致状态污染

使用 `function` 范围的 loop（默认），而不是 `session` 级，除非有明确需要。

### 陷阱四：mock 了错误的路径

```python
# 错误：mock 了原始模块，但被测代码已经导入了
with patch("asyncio.sleep"):   # 没效果，被测代码用的是自己模块里的 asyncio
    ...

# 正确：mock 被测代码实际使用的路径
with patch("mymodule.asyncio.sleep"):
    ...
# 或者
with patch("mymodule.some_func") as mock:
    mock.return_value = ...
```

### 陷阱五：没有等待后台任务完成就断言

```python
# 错误
async def test_background_task():
    results = []
    asyncio.create_task(populate(results))   # 后台任务
    assert len(results) == 5                 # ← 可能为 0，任务还没跑

# 正确
async def test_background_task():
    results = []
    task = asyncio.create_task(populate(results))
    await task                               # 等待完成
    assert len(results) == 5
```

---

## 31.10 测试工具速查

```python
# conftest.py 推荐配置
import pytest
import asyncio


# 如果需要 session 级共享资源
@pytest.fixture(scope="session")
async def shared_client():
    client = await create_client()
    yield client
    await client.aclose()


# 测试时间控制的 fixture
@pytest.fixture
def mock_sleep(monkeypatch):
    from unittest.mock import AsyncMock
    sleep_mock = AsyncMock()
    monkeypatch.setattr("asyncio.sleep", sleep_mock)
    return sleep_mock
```

```python
# 常用断言模式
assert task.done()             # 任务已完成
assert task.cancelled()        # 任务被取消
assert not task.exception()    # 没有异常

# AsyncMock 断言
mock.assert_awaited()                        # 至少被 await 一次
mock.assert_awaited_once()                   # 恰好被 await 一次
mock.assert_awaited_with(arg1, key=val)      # 最后一次以这些参数调用
mock.assert_any_await(arg1)                  # 曾经以这些参数调用
assert mock.await_count == N                 # 被 await 了 N 次
```

---

## 本章小结

| 主题 | 关键工具/方法 |
|------|-------------|
| 测试框架 | `pytest-asyncio`，`asyncio_mode = "auto"` |
| loop 范围 | `function`（默认）/ `module` / `session` |
| 模拟协程 | `AsyncMock`，`patch(..., new_callable=AsyncMock)` |
| 模拟时间 | `patch("asyncio.sleep")` / `freezegun` |
| 验证取消 | `task.cancelled()`，`pytest.raises(CancelledError)` |
| 验证超时 | `pytest.raises(TimeoutError)` |
| 测试 TaskGroup | `pytest.raises(ExceptionGroup)` |
| 常见陷阱 | 忘记 await、sleep(0) 误解、mock 路径错误 |

---

## 深度学习应用

在深度学习和 AI Infra 的测试场景中：

- `AsyncMock` 模拟模型推理接口，不依赖真实 GPU
- mock `asyncio.sleep` 加速重试/退避逻辑测试
- 测试 `TaskGroup` 确保批量推理中一个失败不会静默丢结果
- 用 `Event` 测试模型加载屏障的正确触发时序
- `freeze_time` 测试 deadline 预算传递逻辑

---

## 练习题

1. 为什么在异步测试里忘记 `await` 不会报错，却可能导致测试永远通过？
2. `AsyncMock` 和 `MagicMock` 的根本区别是什么？什么时候必须用 `AsyncMock`？
3. 如何测试"取消被正确传播而不是被吞掉"？写出测试代码骨架。
4. `await asyncio.sleep(0)` 和"等待所有任务完成"有什么区别？
5. 设计一个测试，验证一个生产者/消费者系统在消费者崩溃时，生产者能正确感知并停止生产。
