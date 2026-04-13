# 第29章：asyncio高级并发控制

> 掌握了 `await`、`create_task()`、`gather()` 之后，你面对的下一层问题是：如何处理**一组任务中最先完成的那个**？如何**保护某个协程不被取消**？如何在运行时**自省系统里所有活跃任务**？如何编写**自定义的可等待对象**？这些能力在高复杂度系统中经常出现，但很少被系统地讲清楚。

---

## 学习目标

完成本章学习后，你将能够：

1. 使用 `asyncio.wait()` 实现"第一个完成就响应"的并发模式
2. 使用 `asyncio.as_completed()` 对一组任务按完成顺序处理
3. 使用 `asyncio.shield()` 保护关键协程不被外层取消传播中断
4. 区分 `asyncio.timeout()`、`asyncio.timeout_at()` 和 `asyncio.wait_for()` 的适用场景
5. 在运行时自省任务状态，用任务名称辅助调试
6. 理解 Python 协程协议，实现自定义 awaitable 对象

---

## 正文内容

## 29.1 `asyncio.wait()`：比 `gather()` 更细粒度的控制

### 29.1.1 `gather()` 的局限性

`asyncio.gather()` 会等待**所有**任务完成后才返回。这在以下场景中不够用：

- 你想在**第一个任务完成时立刻响应**
- 你想在**任何任务出错时立刻停止**
- 你需要**区分"已完成"和"仍在运行"**的任务

这正是 `asyncio.wait()` 的用武之地。

### 29.1.2 基本用法

```python
done, pending = await asyncio.wait(tasks, return_when=...)
```

返回两个集合：
- `done`：已完成（包括正常完成和异常）的任务
- `pending`：还没完成的任务

`return_when` 有三个选项：

| 值 | 含义 |
|----|------|
| `asyncio.ALL_COMPLETED` | 等全部完成（默认） |
| `asyncio.FIRST_COMPLETED` | 有一个完成就返回 |
| `asyncio.FIRST_EXCEPTION` | 有一个抛异常就返回（若无异常则等全部完成） |

### 29.1.3 `FIRST_COMPLETED`：竞速模式

竞速模式的场景：从多个数据源取同一份数据，谁先返回就用谁：

```python
import asyncio
import random


async def fetch_from(source: str) -> str:
    delay = random.uniform(0.1, 1.0)
    await asyncio.sleep(delay)
    return f"{source} result after {delay:.2f}s"


async def race_fetch(sources: list[str]) -> str:
    tasks = {asyncio.create_task(fetch_from(s), name=s) for s in sources}

    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

    # 取消还在跑的任务
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)

    winner = next(iter(done))
    return winner.result()


async def main():
    result = await race_fetch(["primary-db", "replica-db", "cache"])
    print(f"winner: {result}")


asyncio.run(main())
```

这是一种典型的"冗余请求"（redundant request）模式，常见于：
- 多副本数据库中取最快响应
- 多区域服务降低尾延迟（tail latency）

### 29.1.4 `FIRST_EXCEPTION`：出错立刻响应

```python
import asyncio


async def risky_task(name: str, should_fail: bool):
    await asyncio.sleep(0.2)
    if should_fail:
        raise ValueError(f"{name} failed")
    return f"{name} ok"


async def main():
    tasks = [
        asyncio.create_task(risky_task("A", False)),
        asyncio.create_task(risky_task("B", True)),
        asyncio.create_task(risky_task("C", False)),
    ]

    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

    for task in done:
        if task.exception():
            print(f"error detected: {task.exception()}")

    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)


asyncio.run(main())
```

**注意**：`FIRST_EXCEPTION` 在没有任何任务抛异常时，行为退化为 `ALL_COMPLETED`。

### 29.1.5 `asyncio.wait()` 的重要细节

- 传入的必须是**可等待的 Future/Task 集合**，不能直接传协程
- 返回的 `done` 集合是**任务引用**，需要调用 `.result()` 或 `.exception()` 取值
- 不像 `gather()`，`wait()` **不会**自动传播异常——你必须自己检查

```python
for task in done:
    exc = task.exception()
    if exc:
        print(f"task failed: {exc}")
    else:
        print(f"task result: {task.result()}")
```

---

## 29.2 `asyncio.as_completed()`：按完成顺序处理

`as_completed()` 解决的是另一类问题：

> 我有一批任务，不关心谁第一个完成，但想**每当有一个完成时就立刻处理它的结果**。

```python
import asyncio
import random


async def slow_task(name: str) -> str:
    await asyncio.sleep(random.uniform(0.1, 1.0))
    return f"{name} done"


async def main():
    tasks = [slow_task(f"task-{i}") for i in range(5)]

    async for coro in asyncio.as_completed(tasks):
        result = await coro
        print(f"completed: {result}")


asyncio.run(main())
```

输出顺序会按照实际完成时间，而不是任务创建顺序。

### 29.2.1 `as_completed()` vs `gather()` vs `wait()`

| 方法 | 结果返回时机 | 结果顺序 | 适用场景 |
|------|-------------|----------|----------|
| `gather()` | 全部完成后 | 按创建顺序 | 需要所有结果再继续 |
| `wait()` | 可配置 | 集合（无序）| 细粒度控制 / 竞速 |
| `as_completed()` | 每完成一个就返回 | 按完成时间 | 流式处理 / 尽快展示 |

### 29.2.2 实战：带进度反馈的并发下载

```python
import asyncio
import random
import time


async def download(url: str) -> tuple[str, float]:
    delay = random.uniform(0.2, 1.5)
    await asyncio.sleep(delay)
    return url, delay


async def main():
    urls = [f"https://example.com/file-{i}" for i in range(8)]
    start = time.monotonic()

    completed = 0
    async for coro in asyncio.as_completed([download(url) for url in urls]):
        url, elapsed = await coro
        completed += 1
        print(f"[{completed}/{len(urls)}] {url} ({elapsed:.2f}s) "
              f"wall={time.monotonic() - start:.2f}s")

    print(f"\nall done in {time.monotonic() - start:.2f}s")


asyncio.run(main())
```

---

## 29.3 `asyncio.shield()`：保护协程不被外层取消

### 29.3.1 问题场景

设想你有一个关键的"提交事务"协程，但它被包裹在一个有超时的外层操作里。  
当超时触发取消时，你**不希望**提交事务被取消——因为它必须完成以保证数据一致性。

```python
import asyncio


async def commit_transaction():
    """关键操作：必须完成，不能被取消"""
    print("committing transaction...")
    await asyncio.sleep(0.5)
    print("transaction committed!")


async def handler_with_timeout():
    try:
        async with asyncio.timeout(0.2):
            await asyncio.shield(commit_transaction())  # ← 保护内层
    except TimeoutError:
        print("handler timed out, but transaction keeps running")


async def main():
    await handler_with_timeout()
    await asyncio.sleep(0.6)   # 给事务时间完成


asyncio.run(main())
```

### 29.3.2 `shield()` 的语义

`asyncio.shield(coro_or_future)` 创建了一个"护盾"：

- **外层取消**：护盾本身会被取消，但**内层协程继续运行**
- 如果你需要知道内层协程的最终结果，需要持有内层的 Future 引用

```python
import asyncio


async def critical_work():
    await asyncio.sleep(1.0)
    return "important result"


async def main():
    inner = asyncio.ensure_future(critical_work())
    shield = asyncio.shield(inner)

    try:
        async with asyncio.timeout(0.3):
            await shield
    except TimeoutError:
        print("outer timed out")

    # inner 还在跑！
    result = await inner
    print(f"inner finished: {result}")


asyncio.run(main())
```

### 29.3.3 何时用 `shield()`，何时不该用

| 情况 | 建议 |
|------|------|
| 关键写操作（事务提交、日志刷盘）| 可以用 `shield()` |
| 需要"无论如何都要完成"的清理逻辑 | 可以用 `shield()` + `finally` |
| 让所有操作都 shield | 不建议——会让取消机制失效 |
| 因为"怕出错"就 shield 一切 | 错误用法——应修复取消逻辑 |

**关键原则**：`shield()` 不是逃避取消传播的快捷方式，而是用于**真正需要原子性完成的操作**。

---

## 29.4 超时 API 完整对比

asyncio 提供了三套超时机制，各有不同的语义和适用场景。

### 29.4.1 `asyncio.timeout()`（Python 3.11+）

```python
async with asyncio.timeout(1.5):
    await some_operation()
```

- 超时时抛出 `TimeoutError`
- 可以配合 `try/except TimeoutError` 处理
- 支持**嵌套**：内层超时不影响外层超时的剩余时间

```python
async with asyncio.timeout(3.0):          # 外层：3秒总预算
    async with asyncio.timeout(1.0):      # 内层：阶段1最多1秒
        await stage_one()
    async with asyncio.timeout(1.5):      # 内层：阶段2最多1.5秒
        await stage_two()
```

### 29.4.2 `asyncio.timeout_at()`（Python 3.11+）

接受一个**绝对截止时间**（`loop.time()` 格式），而不是相对延迟：

```python
import asyncio


async def main():
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 2.0   # 当前时间 + 2 秒

    async with asyncio.timeout_at(deadline):
        await asyncio.sleep(1.0)
        # 剩余预算 = deadline - loop.time()
        async with asyncio.timeout_at(deadline):   # 复用同一个截止时间
            await asyncio.sleep(0.8)

asyncio.run(main())
```

`timeout_at()` 非常适合**deadline 预算传递**场景（见第27章）：

```python
async def call_chain(deadline: float):
    async with asyncio.timeout_at(deadline):
        result = await stage_a()
        await stage_b(result, deadline)   # 传递同一个截止时间

async def stage_b(data, deadline: float):
    async with asyncio.timeout_at(deadline):
        # 自动使用剩余预算，不需要重新计算
        return await process(data)
```

### 29.4.3 `asyncio.wait_for()`

```python
result = await asyncio.wait_for(some_coroutine(), timeout=1.5)
```

- 是函数调用形式，不是上下文管理器
- 超时时**取消**内层协程，然后抛出 `TimeoutError`
- 与 `timeout()` 的主要区别：`wait_for()` 会主动取消，`timeout()` 也会取消但语义更干净

### 29.4.4 三种超时 API 对比

| API | 形式 | 时间类型 | 嵌套 | 适用场景 |
|-----|------|---------|------|----------|
| `asyncio.timeout()` | 上下文管理器 | 相对延迟 | 支持 | 通用超时 |
| `asyncio.timeout_at()` | 上下文管理器 | 绝对时间 | 支持 | deadline 预算传递 |
| `asyncio.wait_for()` | 函数调用 | 相对延迟 | 较复杂 | 简单单层超时 |

**推荐原则**：Python 3.11+ 优先使用 `asyncio.timeout()`/`asyncio.timeout_at()`，它们的语义更清晰，嵌套更自然。

---

## 29.5 Task 自省与调试

### 29.5.1 命名任务

给任务命名是异步调试的第一步：

```python
task = asyncio.create_task(my_coro(), name="fetch-user-profile")
print(task.get_name())   # "fetch-user-profile"
```

也可以在创建后修改：

```python
task.set_name("fetch-user-profile-42")
```

命名任务会出现在：
- 调试输出中
- `asyncio.all_tasks()` 的结果里
- 异常堆栈信息中

### 29.5.2 `asyncio.all_tasks()`

获取当前事件循环里所有活跃任务：

```python
import asyncio


async def worker(name: str):
    await asyncio.sleep(10)


async def inspector():
    await asyncio.sleep(0.1)   # 让 workers 先启动
    tasks = asyncio.all_tasks()
    for task in tasks:
        print(f"  task: {task.get_name()!r}, done={task.done()}")


async def main():
    workers = [
        asyncio.create_task(worker("w1"), name="worker-1"),
        asyncio.create_task(worker("w2"), name="worker-2"),
        asyncio.create_task(worker("w3"), name="worker-3"),
    ]
    asyncio.create_task(inspector(), name="inspector")

    await asyncio.sleep(0.2)
    for w in workers:
        w.cancel()
    await asyncio.gather(*workers, return_exceptions=True)


asyncio.run(main())
```

### 29.5.3 `asyncio.current_task()`

在协程内部获取当前任务的引用：

```python
async def my_coro():
    task = asyncio.current_task()
    print(f"I am task: {task.get_name()}")
    task.set_name("renamed-at-runtime")
```

常见用法：
- 在日志里自动附加任务名
- 在协程内动态重命名任务以反映当前处理状态

### 29.5.4 检查任务状态

```python
task = asyncio.create_task(some_coro())

task.done()        # 是否已完成（含取消和异常）
task.cancelled()   # 是否被取消
task.exception()   # 如果任务异常结束，返回异常对象；否则返回 None
task.result()      # 如果任务正常完成，返回结果；否则抛出异常
```

注意：对未完成的任务调用 `result()` 或 `exception()` 会抛出 `InvalidStateError`。

### 29.5.5 批量检查任务状态的工具函数

```python
import asyncio


def task_summary(tasks: list[asyncio.Task]) -> dict:
    done = [t for t in tasks if t.done() and not t.cancelled() and not t.exception()]
    failed = [t for t in tasks if t.done() and not t.cancelled() and t.exception()]
    cancelled = [t for t in tasks if t.cancelled()]
    pending = [t for t in tasks if not t.done()]

    return {
        "done":      len(done),
        "failed":    len(failed),
        "cancelled": len(cancelled),
        "pending":   len(pending),
        "errors":    [t.exception() for t in failed],
    }
```

---

## 29.6 自定义 awaitable 对象

### 29.6.1 Python 协程协议

当你写 `await expr` 时，Python 要求 `expr` 是一个 **awaitable**，也就是：

- 一个协程对象（`async def` 函数的返回值）
- 实现了 `__await__()` 方法的对象
- 一个 `asyncio.Future` 或 `asyncio.Task`

`__await__()` 必须返回一个迭代器（iterator）。

### 29.6.2 最小自定义 awaitable

```python
import asyncio


class SleepOnce:
    """一个自定义的 awaitable：await 时挂起一次调度循环"""

    def __await__(self):
        yield   # 主动挂起一次，把控制权交还给事件循环
        return "woke up"


async def main():
    result = await SleepOnce()
    print(result)   # "woke up"


asyncio.run(main())
```

`yield` 是协程协议的核心：它告诉事件循环"我现在不需要继续，先去处理别的"。

### 29.6.3 基于 Future 的自定义 awaitable

更实用的模式是把 `__await__` 委托给一个 `Future`：

```python
import asyncio


class Deferred:
    """一个可从外部设置结果的 awaitable"""

    def __init__(self):
        self._future: asyncio.Future | None = None

    def _ensure_future(self):
        if self._future is None:
            loop = asyncio.get_running_loop()
            self._future = loop.create_future()
        return self._future

    def resolve(self, value):
        """从外部设置结果"""
        self._ensure_future().set_result(value)

    def reject(self, exc: Exception):
        """从外部设置异常"""
        self._ensure_future().set_exception(exc)

    def __await__(self):
        return self._ensure_future().__await__()


async def main():
    d = Deferred()

    async def resolver():
        await asyncio.sleep(0.3)
        d.resolve("hello from resolver")

    asyncio.create_task(resolver())
    result = await d
    print(f"got: {result}")


asyncio.run(main())
```

### 29.6.4 自定义 awaitable 在哪些场景有价值

| 场景 | 说明 |
|------|------|
| 异步 RPC 框架 | 调用方等待远端结果，基础设施稍后回填 |
| 动态批处理器 | 请求等待批次被执行，批处理器回填各个 Future |
| 跨线程桥接 | 线程里完成工作后，通过 `call_soon_threadsafe` 回填 Future |
| 测试基础设施 | 手动控制测试中协程的唤醒时机 |

---

## 29.7 高级取消模式

第25章介绍了 `CancelledError` 的基本处理原则。本节补充更复杂的场景。

### 29.7.1 取消超时与 `shield()` 的组合

```python
import asyncio


async def must_complete():
    """这个操作必须完成，不能被取消"""
    print("critical work started")
    await asyncio.sleep(1.0)
    print("critical work done")
    return "result"


async def best_effort_wrapper():
    """尽力在 0.5s 内完成，超时不等了，但关键工作继续"""
    inner = asyncio.ensure_future(must_complete())
    try:
        async with asyncio.timeout(0.5):
            return await asyncio.shield(inner)
    except TimeoutError:
        print("wrapper timed out, critical work still running")
        return await inner   # 等待关键工作最终完成


async def main():
    result = await best_effort_wrapper()
    print(f"final: {result}")


asyncio.run(main())
```

### 29.7.2 `CancelledError` 携带消息（Python 3.9+）

```python
import asyncio


async def worker():
    try:
        await asyncio.sleep(10)
    except asyncio.CancelledError as exc:
        print(f"cancelled with message: {exc}")
        raise


async def main():
    task = asyncio.create_task(worker())
    await asyncio.sleep(0.1)

    task.cancel("shutdown requested by operator")
    await asyncio.gather(task, return_exceptions=True)


asyncio.run(main())
```

Python 3.9+ 的 `task.cancel(msg)` 支持传递原因字符串，可以在 `CancelledError` 中读取，对调试非常有用。

### 29.7.3 检测取消是否来自父级 TaskGroup

在 `TaskGroup` 场景中，如果一个子任务失败，其他子任务会被批量取消。可以用 `task.cancelling()` 方法（Python 3.11+）判断取消的来源：

```python
import asyncio


async def child(name: str, should_fail: bool):
    try:
        await asyncio.sleep(1.0)
    except asyncio.CancelledError:
        task = asyncio.current_task()
        cancelling_count = task.cancelling() if task else 0
        print(f"{name}: CancelledError, cancelling()={cancelling_count}")
        raise


async def main():
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(child("A", False), name="child-A")
            tg.create_task(child("B", True),  name="child-B")
            # B 会立刻抛出异常导致 A 被取消
    except* Exception as eg:
        print(f"ExceptionGroup: {eg.exceptions}")


async def failing_child():
    raise RuntimeError("B failed")


asyncio.run(main())
```

---

## 29.8 实战：带竞速、超时预算和自省的并发编排器

```python
import asyncio
import time
import random
from dataclasses import dataclass


@dataclass
class StageResult:
    name: str
    value: object
    elapsed_ms: float


async def run_stage(name: str, work_s: float, fail: bool = False) -> StageResult:
    start = time.monotonic()
    await asyncio.sleep(work_s)
    if fail:
        raise RuntimeError(f"{name} failed")
    return StageResult(
        name=name,
        value=f"{name}_output",
        elapsed_ms=(time.monotonic() - start) * 1000,
    )


async def orchestrate(total_budget_s: float = 2.0):
    deadline = asyncio.get_running_loop().time() + total_budget_s

    print(f"[orchestrator] starting, budget={total_budget_s}s")

    # 阶段1：竞速取数据（谁快用谁）
    primary   = asyncio.create_task(run_stage("primary",  0.3), name="primary")
    secondary = asyncio.create_task(run_stage("secondary", 0.6), name="secondary")

    done, pending = await asyncio.wait(
        {primary, secondary},
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)

    stage1_result = next(iter(done)).result()
    print(f"  stage1: {stage1_result.name} ({stage1_result.elapsed_ms:.0f}ms)")

    # 阶段2：并发处理，按完成顺序收集
    enrichments = [
        run_stage(f"enrich-{i}", random.uniform(0.1, 0.5))
        for i in range(4)
    ]

    enriched = []
    async for coro in asyncio.as_completed(enrichments):
        try:
            async with asyncio.timeout_at(deadline):
                r = await coro
                enriched.append(r)
                print(f"  enriched: {r.name} ({r.elapsed_ms:.0f}ms)")
        except TimeoutError:
            print("  enrichment timed out, moving on")
            break

    # 自省：检查当前活跃任务
    active = asyncio.all_tasks()
    print(f"\n  active tasks at end: {len(active)}")
    for t in active:
        if t is not asyncio.current_task():
            print(f"    - {t.get_name()}")

    return {"stage1": stage1_result, "enrichments": enriched}


async def main():
    result = await orchestrate(total_budget_s=1.5)
    print(f"\nfinal enrichments: {len(result['enrichments'])}")


asyncio.run(main())
```

---

## 本章小结

| API | 核心用途 |
|-----|---------|
| `asyncio.wait(FIRST_COMPLETED)` | 竞速：谁先完成就用谁 |
| `asyncio.wait(FIRST_EXCEPTION)` | 出错就立刻停止等待 |
| `asyncio.as_completed()` | 按完成顺序流式处理结果 |
| `asyncio.shield()` | 保护内层协程不被外层取消 |
| `asyncio.timeout()` | 相对时间超时，支持嵌套 |
| `asyncio.timeout_at()` | 绝对截止时间，适合 deadline 预算传递 |
| `asyncio.all_tasks()` | 自省所有活跃任务 |
| `asyncio.current_task()` | 获取当前任务引用 |
| `task.cancel(msg)` | 带原因字符串的取消 |
| `task.cancelling()` | 检查任务被取消的层数（Python 3.11+） |
| 自定义 `__await__` | 实现自定义可等待对象 |

---

## 深度学习应用

本章在深度学习系统中的典型落点：

- `wait(FIRST_COMPLETED)` + 竞速：多副本 embedding 服务中取最快响应
- `as_completed()`：并发预取特征，每完成一个就更新上下文
- `shield()`：确保模型参数保存操作不被停机信号中断
- `timeout_at()`：推理链路多阶段共享一个截止时间
- `all_tasks()` + 任务命名：实时监控推理网关里的并发请求数

---

## 练习题

1. `asyncio.wait(FIRST_COMPLETED)` 和 `asyncio.as_completed()` 的本质区别是什么？各适合什么场景？
2. 为什么对未完成的任务调用 `task.result()` 会报错？正确的检查顺序是什么？
3. `asyncio.shield()` 能保证内层协程"一定完成"吗？有什么前提条件？
4. `asyncio.timeout()` 和 `asyncio.timeout_at()` 分别适合什么场景？举一个需要 deadline 预算传递的例子。
5. 实现一个自定义 awaitable `TimedFuture`，支持从外部设置结果，并在等待超过指定时间后自动以 `TimeoutError` 失败。
