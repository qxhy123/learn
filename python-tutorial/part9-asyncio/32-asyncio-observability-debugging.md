# 第32章：asyncio可观测性与调试

> 异步系统最难排障的场景不是崩溃——而是"系统看起来在运行，但不再前进"。队列不增不减，任务数量稳定，CPU 和内存都正常，但请求就是没有响应。要找出这类问题，你需要一套能让你"看见"事件循环内部状态的工具。

---

## 学习目标

完成本章学习后，你将能够：

1. 启用 `asyncio` 调试模式，理解它报告哪些问题
2. 使用任务自省 API 实时观察系统状态
3. 识别并排查慢协程、任务泄漏、死锁和事件循环阻塞
4. 为异步系统添加结构化日志和指标
5. 使用 `tracemalloc` 检测异步代码的内存问题

---

## 正文内容

## 32.1 asyncio 调试模式

### 32.1.1 启用调试模式

有三种方式启用 asyncio 调试模式：

**方式一：环境变量（推荐用于开发）**

```bash
PYTHONASYNCIODEBUG=1 python myapp.py
```

**方式二：代码里显式设置**

```python
import asyncio

asyncio.run(main(), debug=True)
```

**方式三：在运行中动态设置**

```python
async def main():
    loop = asyncio.get_running_loop()
    loop.set_debug(True)
    ...
```

### 32.1.2 调试模式报告哪些问题

开启调试模式后，asyncio 会：

1. **检测协程对象从未被 await**
   ```
   RuntimeWarning: coroutine 'fetch_data' was never awaited
   ```

2. **检测执行时间过长的回调/协程**（默认阈值 0.1s）
   ```
   Executing <Task ...> took 0.512 seconds
   ```

3. **更详细的任务创建堆栈信息**

4. **未关闭的异步资源警告**

### 32.1.3 调整慢回调警告阈值

```python
import asyncio

async def main():
    loop = asyncio.get_running_loop()
    loop.slow_callback_duration = 0.05  # 超过 50ms 就报警（默认 100ms）
    ...
```

### 32.1.4 调试模式的性能影响

调试模式有额外开销（追踪堆栈、时间测量等），**不应在生产环境开启**。  
使用模式：

| 环境 | 建议 |
|------|------|
| 本地开发 | 始终开启 `PYTHONASYNCIODEBUG=1` |
| CI/CD | 开启（帮助发现测试里的问题）|
| 生产 | 关闭（有性能影响）|

---

## 32.2 任务自省：实时观察系统状态

### 32.2.1 快照当前所有活跃任务

```python
import asyncio


async def dump_tasks():
    """打印当前所有活跃任务的状态"""
    tasks = asyncio.all_tasks()
    current = asyncio.current_task()

    print(f"\n=== Task Snapshot ({len(tasks)} active) ===")
    for task in sorted(tasks, key=lambda t: t.get_name()):
        if task is current:
            print(f"  [CURRENT] {task.get_name()}")
            continue

        state = "running" if not task.done() else (
            "cancelled" if task.cancelled() else
            "failed" if task.exception() else "done"
        )
        print(f"  [{state:9s}] {task.get_name()}")

    print("=" * 40)
```

### 32.2.2 周期性健康检查协程

```python
import asyncio


async def health_monitor(interval_s: float = 5.0):
    """后台运行，定期报告系统任务状态"""
    while True:
        await asyncio.sleep(interval_s)
        tasks = asyncio.all_tasks()
        pending = [t for t in tasks if not t.done()]

        print(f"[health] active_tasks={len(pending)}")
        for task in pending:
            if task.get_name() != "health_monitor":
                print(f"  - {task.get_name()}")
```

### 32.2.3 检测长时间挂起的任务

```python
import asyncio
import time


async def watchdog(max_idle_s: float = 30.0):
    """检测超过 max_idle_s 没有完成的任务"""
    start_times: dict[str, float] = {}

    while True:
        await asyncio.sleep(5.0)
        now = time.monotonic()
        tasks = asyncio.all_tasks()

        for task in tasks:
            name = task.get_name()
            if name not in start_times:
                start_times[name] = now

        # 清理已完成的任务记录
        active_names = {t.get_name() for t in tasks}
        start_times = {k: v for k, v in start_times.items()
                       if k in active_names}

        # 警告超时任务
        for name, start in start_times.items():
            if now - start > max_idle_s:
                print(f"[watchdog] WARNING: task '{name}' "
                      f"has been running for {now - start:.0f}s")
```

---

## 32.3 检测事件循环阻塞

事件循环阻塞是 asyncio 系统最常见的性能杀手：一段同步代码卡住了 loop，导致所有其他任务无法调度。

### 32.3.1 症状识别

```
Executing <Task pending coro=<main() running at app.py:42>> took 0.823 seconds
```

或者：请求延迟突然飙升，但 CPU 不高，I/O 等待也不多。

### 32.3.2 自制 loop lag 检测器

```python
import asyncio
import time


async def loop_lag_monitor(warn_threshold_ms: float = 50.0):
    """
    通过测量 sleep(0) 的实际耗时来检测 loop 阻塞。
    如果 loop 是健康的，sleep(0) 应该几乎立刻返回。
    """
    while True:
        start = time.monotonic()
        await asyncio.sleep(0)
        lag_ms = (time.monotonic() - start) * 1000

        if lag_ms > warn_threshold_ms:
            print(f"[lag_monitor] loop lag = {lag_ms:.1f}ms "
                  f"(threshold={warn_threshold_ms}ms)")


async def main():
    # 后台运行 lag 检测
    asyncio.create_task(loop_lag_monitor(warn_threshold_ms=20.0),
                        name="loop-lag-monitor")

    # 你的业务逻辑
    ...
```

### 32.3.3 找到阻塞代码的位置

当 `loop lag` 告警出现时，说明有同步代码在 loop 线程里运行太久。常见原因：

| 原因 | 表现 | 解决方案 |
|------|------|----------|
| `time.sleep()` | lag 突然出现 | 换 `asyncio.sleep()` |
| 阻塞数据库驱动 | 周期性 lag | 换异步驱动或 `to_thread()` |
| 大量纯 Python 计算 | 持续 lag | `run_in_executor(ProcessPoolExecutor)` |
| 阻塞文件 I/O | 随机 lag | `asyncio.to_thread()` |
| `json.loads()` 超大 payload | 偶发 lag | 分块或放入线程 |

---

## 32.4 结构化日志：让异步系统可读

### 32.4.1 用 ContextVar 自动附加上下文

第30章介绍了 `contextvars`。在日志里使用它：

```python
import asyncio
import logging
from contextvars import ContextVar

trace_id: ContextVar[str] = ContextVar("trace_id", default="-")


class AsyncContextFilter(logging.Filter):
    """自动把当前 trace_id 注入日志记录"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = trace_id.get("-")
        record.task_name = (
            t.get_name() if (t := asyncio.current_task()) else "main"
        )
        return True


def setup_logging():
    handler = logging.StreamHandler()
    handler.addFilter(AsyncContextFilter())
    formatter = logging.Formatter(
        "%(asctime)s [%(trace_id)s] [%(task_name)s] %(levelname)s %(message)s"
    )
    handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[handler])


logger = logging.getLogger(__name__)


async def handle_request(request_id: str):
    trace_id.set(request_id)
    logger.info("request started")
    await asyncio.sleep(0.1)
    logger.info("request completed")


async def main():
    setup_logging()
    await asyncio.gather(
        handle_request("req-001"),
        handle_request("req-002"),
    )


asyncio.run(main())
```

输出示例：
```
2024-01-15 10:00:00 [req-001] [Task-1] INFO request started
2024-01-15 10:00:00 [req-002] [Task-2] INFO request started
2024-01-15 10:00:00 [req-001] [Task-1] INFO request completed
2024-01-15 10:00:00 [req-002] [Task-2] INFO request completed
```

### 32.4.2 任务生命周期日志

```python
import asyncio
import logging
import time
from functools import wraps
from typing import Callable, TypeVar, Awaitable

T = TypeVar("T")
logger = logging.getLogger(__name__)


def log_task_lifecycle(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    """装饰器：自动记录协程的开始、完成、失败和耗时"""
    @wraps(func)
    async def wrapper(*args, **kwargs) -> T:
        name = func.__name__
        start = time.monotonic()
        logger.debug(f"{name} started")
        try:
            result = await func(*args, **kwargs)
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.debug(f"{name} completed in {elapsed_ms:.1f}ms")
            return result
        except asyncio.CancelledError:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.warning(f"{name} cancelled after {elapsed_ms:.1f}ms")
            raise
        except Exception as exc:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.error(f"{name} failed after {elapsed_ms:.1f}ms: {exc}")
            raise
    return wrapper


@log_task_lifecycle
async def fetch_profile(user_id: int) -> dict:
    await asyncio.sleep(0.1)
    return {"id": user_id}
```

---

## 32.5 指标收集：让系统可量化

### 32.5.1 最小内置指标收集器

```python
import asyncio
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field


@dataclass
class Metrics:
    counts: dict = field(default_factory=lambda: defaultdict(int))
    durations: dict = field(default_factory=lambda: defaultdict(list))

    def increment(self, name: str, value: int = 1):
        self.counts[name] += value

    def record_duration(self, name: str, duration_ms: float):
        self.durations[name].append(duration_ms)

    def summary(self) -> dict:
        result = dict(self.counts)
        for name, durations in self.durations.items():
            if durations:
                sorted_d = sorted(durations)
                n = len(sorted_d)
                result[f"{name}.p50_ms"] = sorted_d[n // 2]
                result[f"{name}.p99_ms"] = sorted_d[int(n * 0.99)]
                result[f"{name}.count"] = n
        return result


metrics = Metrics()


@asynccontextmanager
async def measure(name: str):
    """异步上下文管理器：自动计时并记录指标"""
    start = time.monotonic()
    try:
        yield
        metrics.increment(f"{name}.success")
    except asyncio.CancelledError:
        metrics.increment(f"{name}.cancelled")
        raise
    except Exception:
        metrics.increment(f"{name}.error")
        raise
    finally:
        elapsed_ms = (time.monotonic() - start) * 1000
        metrics.record_duration(name, elapsed_ms)


async def fetch_data(source: str) -> dict:
    async with measure(f"fetch.{source}"):
        await asyncio.sleep(0.1)
        return {"source": source}


async def metrics_reporter(interval_s: float = 10.0):
    while True:
        await asyncio.sleep(interval_s)
        summary = metrics.summary()
        print(f"[metrics] {summary}")


async def main():
    asyncio.create_task(metrics_reporter(5.0), name="metrics-reporter")

    await asyncio.gather(
        fetch_data("db"),
        fetch_data("cache"),
        fetch_data("api"),
    )

    await asyncio.sleep(0.1)


asyncio.run(main())
```

### 32.5.2 队列积压监控

```python
import asyncio


async def queue_monitor(queue: asyncio.Queue, name: str, interval_s: float = 1.0):
    """监控队列积压，发现处理瓶颈"""
    prev_size = 0
    while True:
        await asyncio.sleep(interval_s)
        size = queue.qsize()
        maxsize = queue.maxsize
        fill_pct = (size / maxsize * 100) if maxsize > 0 else 0

        trend = "↑" if size > prev_size else ("↓" if size < prev_size else "→")
        print(f"[queue:{name}] size={size}/{maxsize} ({fill_pct:.0f}%) {trend}")

        if maxsize > 0 and fill_pct > 80:
            print(f"[queue:{name}] WARNING: queue nearly full!")

        prev_size = size
```

---

## 32.6 检测任务泄漏

任务泄漏是指创建了 `Task` 但从未 `await` 它，或者 `TaskGroup` 之外的 `create_task()` 调用没有被适当管理。

### 32.6.1 什么是任务泄漏

```python
async def leaky_code():
    # 创建任务但从不等待它
    asyncio.create_task(some_background_work())
    # 如果 some_background_work 抛出异常，异常会被静默丢弃
    # 如果程序退出，这个任务会被强制取消并产生警告
```

### 32.6.2 运行时检测泄漏

```python
import asyncio


async def task_leak_detector(warn_threshold: int = 50):
    """监控任务数量，超过阈值警告"""
    baseline = len(asyncio.all_tasks())

    while True:
        await asyncio.sleep(5.0)
        current = len(asyncio.all_tasks())

        if current > warn_threshold:
            print(f"[leak_detector] WARNING: {current} active tasks "
                  f"(baseline={baseline})")
            tasks = asyncio.all_tasks()
            for task in list(tasks)[:10]:   # 只显示前10个
                print(f"  - {task.get_name()}: done={task.done()}")
```

### 32.6.3 正确管理后台任务

```python
import asyncio
from contextlib import asynccontextmanager


class TaskRegistry:
    """跟踪所有后台任务，确保它们在退出时被清理"""

    def __init__(self):
        self._tasks: set[asyncio.Task] = set()

    def create_task(self, coro, *, name: str | None = None) -> asyncio.Task:
        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def cancel_all(self):
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    def __len__(self):
        return len(self._tasks)


registry = TaskRegistry()


async def main():
    registry.create_task(worker(), name="worker-1")
    registry.create_task(worker(), name="worker-2")

    await asyncio.sleep(5.0)

    # 退出时统一取消
    print(f"cancelling {len(registry)} tasks")
    await registry.cancel_all()
```

---

## 32.7 `tracemalloc`：检测异步代码的内存问题

### 32.7.1 启用内存追踪

```python
import asyncio
import tracemalloc


tracemalloc.start(10)   # 保留 10 帧的调用栈


async def main():
    snapshot1 = tracemalloc.take_snapshot()

    # 运行一段可能有内存问题的代码
    await do_work()

    snapshot2 = tracemalloc.take_snapshot()

    top_stats = snapshot2.compare_to(snapshot1, "lineno")
    print("[ Top memory allocations ]")
    for stat in top_stats[:5]:
        print(stat)


asyncio.run(main())
```

### 32.7.2 检测协程对象泄漏

协程对象本身也会占用内存。如果你创建了大量协程对象但从未 await，这些对象会积累：

```python
import gc
import asyncio


def count_coroutines() -> int:
    """统计当前存活的协程对象数量"""
    return sum(1 for obj in gc.get_objects() if asyncio.iscoroutine(obj))


async def main():
    print(f"before: {count_coroutines()} coroutine objects")

    # 模拟泄漏：创建协程但不 await
    leaked = [some_coro() for _ in range(100)]

    print(f"after leak: {count_coroutines()} coroutine objects")

    # 清理
    for coro in leaked:
        coro.close()

    print(f"after close: {count_coroutines()} coroutine objects")
```

---

## 32.8 综合调试工具箱

把前面所有工具组合成一个可复用的调试工具箱：

```python
import asyncio
import logging
import time
from collections import defaultdict
from contextvars import ContextVar

logger = logging.getLogger("asyncio.debug")
trace_id: ContextVar[str] = ContextVar("trace_id", default="-")


class AsyncDebugToolkit:
    """生产级异步系统调试工具箱"""

    def __init__(
        self,
        loop_lag_threshold_ms: float = 50.0,
        task_warn_threshold: int = 100,
        report_interval_s: float = 30.0,
    ):
        self._loop_lag_threshold = loop_lag_threshold_ms / 1000
        self._task_threshold = task_warn_threshold
        self._interval = report_interval_s
        self._request_latencies: dict[str, list[float]] = defaultdict(list)

    async def start(self):
        """启动所有监控协程"""
        asyncio.create_task(self._loop_lag_monitor(), name="debug:loop-lag")
        asyncio.create_task(self._task_monitor(),     name="debug:task-monitor")
        asyncio.create_task(self._periodic_report(),  name="debug:reporter")

    async def _loop_lag_monitor(self):
        while True:
            start = time.monotonic()
            await asyncio.sleep(0)
            lag = time.monotonic() - start
            if lag > self._loop_lag_threshold:
                logger.warning(
                    f"loop lag detected: {lag * 1000:.1f}ms "
                    f"(threshold={self._loop_lag_threshold * 1000:.0f}ms)"
                )

    async def _task_monitor(self):
        while True:
            await asyncio.sleep(5.0)
            count = len(asyncio.all_tasks())
            if count > self._task_threshold:
                logger.warning(f"high task count: {count} "
                               f"(threshold={self._task_threshold})")

    async def _periodic_report(self):
        while True:
            await asyncio.sleep(self._interval)
            tasks = asyncio.all_tasks()
            pending = [t for t in tasks if not t.done()]
            logger.info(f"[report] active_tasks={len(pending)}")

            for operation, latencies in self._request_latencies.items():
                if latencies:
                    sorted_l = sorted(latencies[-100:])  # 只看最近100个
                    n = len(sorted_l)
                    p50 = sorted_l[n // 2]
                    p99 = sorted_l[int(n * 0.99)]
                    logger.info(
                        f"[report] {operation}: "
                        f"p50={p50 * 1000:.1f}ms p99={p99 * 1000:.1f}ms "
                        f"count={n}"
                    )

    def record_latency(self, operation: str, latency_s: float):
        self._request_latencies[operation].append(latency_s)


# 使用示例
async def main():
    toolkit = AsyncDebugToolkit(
        loop_lag_threshold_ms=30.0,
        task_warn_threshold=50,
        report_interval_s=10.0,
    )
    await toolkit.start()

    # 你的业务逻辑
    while True:
        start = time.monotonic()
        await asyncio.sleep(0.1)
        toolkit.record_latency("main_loop", time.monotonic() - start)


asyncio.run(main())
```

---

## 32.9 常见异步系统故障排查路径

```text
症状：系统"冻住"，请求不响应
    ├─ 检查 loop lag monitor → 是否有长时间阻塞回调？
    ├─ 检查 all_tasks() → 任务数量正常吗？
    └─ 检查队列积压 → 是否所有 worker 都在等待某个 Future？

症状：内存持续增长
    ├─ 检查 all_tasks() 数量 → 是否有任务泄漏？
    ├─ 检查队列大小 → 是否积压了大量消息？
    └─ tracemalloc 对比快照 → 哪个对象持续增长？

症状：延迟越来越高但不崩溃
    ├─ 检查队列积压趋势 → 消费速度跟不上生产速度？
    ├─ 检查 Semaphore 计数 → 是否并发上限太低？
    └─ 检查 loop lag → 是否有阻塞代码导致调度延迟？

症状：某些请求超时但整体看起来正常
    ├─ 检查 deadline 预算传递是否正确 → 是否某层超时太短？
    ├─ 检查任务取消是否正确传播 → 是否有孤儿任务继续消耗资源？
    └─ 检查 shield() 使用 → 是否有关键路径被意外保护？
```

---

## 本章小结

| 工具/方法 | 用途 |
|----------|------|
| `PYTHONASYNCIODEBUG=1` | 开启调试模式，检测慢回调和协程泄漏 |
| `asyncio.all_tasks()` | 实时快照所有活跃任务 |
| loop lag monitor | 检测事件循环阻塞 |
| `ContextVar` + 日志过滤器 | 请求级上下文自动注入日志 |
| `@log_task_lifecycle` | 自动记录协程耗时和异常 |
| `measure()` 上下文管理器 | 自动收集操作延迟指标 |
| 任务注册表 | 防止任务泄漏，统一清理 |
| `tracemalloc` | 检测内存增长的根源 |
| 队列积压监控 | 发现生产/消费失衡 |
| 故障排查路径 | 系统性定位"冻住/内存增长/延迟增长"问题 |

---

## 深度学习应用

在深度学习基础设施中，本章内容的典型落点：

- loop lag monitor：及时发现模型推理路径里的阻塞调用
- 任务数量监控：实时观察推理并发负载
- 队列积压监控：检测 batching 层的消费能力
- `tracemalloc`：排查异步推理服务的内存泄漏
- 结构化日志 + trace_id：跨推理链路的请求追踪

---

## 练习题

1. `PYTHONASYNCIODEBUG=1` 会报告哪几类问题？为什么不应在生产环境开启？
2. "loop lag"是什么？如何用 `asyncio.sleep(0)` 检测它？
3. 任务泄漏的常见原因是什么？如何在运行时检测任务泄漏？
4. 如何把 `ContextVar` 和日志过滤器结合，实现请求级别的日志自动标记？
5. 当你发现异步服务"看起来在跑但请求没有响应"时，应该按什么顺序检查哪些指标？
