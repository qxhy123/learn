# 第30章：asyncio与同步代码集成

> `asyncio` 不是一座孤岛。在真实系统里，你几乎总会遇到需要把异步代码与同步库、阻塞 I/O、CPU 密集型计算、子进程或跨线程通信集成在一起的场景。理解如何跨越这条边界，是把 `asyncio` 用进生产系统的最后一块关键拼图。

---

## 学习目标

完成本章学习后，你将能够：

1. 使用 `asyncio.to_thread()` 和 `loop.run_in_executor()` 在不阻塞事件循环的前提下调用同步阻塞代码
2. 理解 `ThreadPoolExecutor` 与 `ProcessPoolExecutor` 的适用边界
3. 使用 `asyncio.subprocess` 异步地创建和管理子进程
4. 理解 `contextvars` 上下文传播机制，并在异步系统中正确使用请求级上下文
5. 掌握从同步代码中安全调用异步代码的几种方式

---

## 正文内容

## 30.1 为什么同步/异步边界是个难题

asyncio 的核心假设是：

> 所有 I/O 操作都通过 `await` 挂起，把控制权归还给事件循环。

但实际上，你会经常遇到：

- 没有异步版本的库（例如某些数据库驱动、OCR 库、C 扩展）
- 大量纯 Python 的 CPU 计算（矩阵运算、数据解析）
- 需要启动外部命令（编译、脚本、系统工具）
- 来自其他线程的回调需要通知事件循环

如果你直接在协程里调用这些阻塞操作，整个事件循环会被卡死：

```python
async def bad_idea():
    time.sleep(2)          # ← 阻塞整个事件循环 2 秒
    data = requests.get("https://example.com")   # ← 同样有问题
```

本章的核心问题就是：**如何在不破坏事件循环的前提下，与这些"不合作"的代码共存**。

---

## 30.2 `asyncio.to_thread()`：把同步函数跑进线程

### 30.2.1 基本用法

`asyncio.to_thread()` 是 Python 3.9 引入的高层 API。它把一个**同步阻塞函数**放入默认的线程池执行，同时不阻塞事件循环：

```python
import asyncio
import time


def blocking_io(path: str) -> str:
    time.sleep(0.5)   # 模拟慢 I/O
    return f"read from {path}"


async def main():
    result = await asyncio.to_thread(blocking_io, "/data/large_file.bin")
    print(result)


asyncio.run(main())
```

### 30.2.2 并发调用多个同步函数

`to_thread()` 最大的价值在于可以**并发执行**多个阻塞操作：

```python
import asyncio
import time


def fetch_user(user_id: int) -> dict:
    time.sleep(0.3)   # 模拟阻塞数据库查询
    return {"id": user_id, "name": f"user-{user_id}"}


async def main():
    start = time.monotonic()

    # 并发执行 3 个阻塞查询
    results = await asyncio.gather(
        asyncio.to_thread(fetch_user, 1),
        asyncio.to_thread(fetch_user, 2),
        asyncio.to_thread(fetch_user, 3),
    )

    elapsed = time.monotonic() - start
    print(f"got {len(results)} users in {elapsed:.2f}s")
    # 大约 0.3s，而不是 0.9s


asyncio.run(main())
```

### 30.2.3 传参和关键字参数

```python
def process(data: bytes, *, encoding: str = "utf-8") -> str:
    return data.decode(encoding)


async def main():
    result = await asyncio.to_thread(process, b"hello", encoding="utf-8")
    print(result)
```

### 30.2.4 `to_thread()` 的限制

- 底层使用 Python 的 `ThreadPoolExecutor`（默认线程数由系统决定）
- 线程池里的代码受 **GIL（全局解释器锁）** 限制——CPU 密集型纯 Python 代码不会因此真正并行
- 适合 I/O 密集型阻塞（文件、网络、数据库），**不适合纯 CPU 密集型计算**

---

## 30.3 `loop.run_in_executor()`：更细粒度的控制

`run_in_executor()` 是 `to_thread()` 的底层版本，可以指定自定义的 executor：

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


async def main():
    loop = asyncio.get_running_loop()

    # 1. 使用默认 executor（ThreadPoolExecutor）
    result = await loop.run_in_executor(None, blocking_func, arg1)

    # 2. 使用自定义 ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as pool:
        result = await loop.run_in_executor(pool, blocking_func, arg1)

    # 3. 使用 ProcessPoolExecutor（真正绕过 GIL）
    with ProcessPoolExecutor(max_workers=4) as pool:
        result = await loop.run_in_executor(pool, cpu_bound_func, data)
```

### 30.3.1 `ThreadPoolExecutor` vs `ProcessPoolExecutor`

| Executor | GIL | 适合 | 不适合 |
|----------|-----|------|--------|
| `ThreadPoolExecutor` | 受限 | 阻塞 I/O、C 扩展 | 纯 Python CPU 计算 |
| `ProcessPoolExecutor` | 不受限 | CPU 密集型纯 Python | 需要共享内存、频繁通信 |

### 30.3.2 ProcessPoolExecutor 实战：CPU 密集型计算

```python
import asyncio
from concurrent.futures import ProcessPoolExecutor
import math


def compute_primes(limit: int) -> int:
    """纯 CPU 计算：统计 limit 以内的质数个数"""
    count = 0
    for n in range(2, limit):
        if all(n % i != 0 for i in range(2, int(math.sqrt(n)) + 1)):
            count += 1
    return count


async def main():
    loop = asyncio.get_running_loop()

    with ProcessPoolExecutor(max_workers=4) as pool:
        # 4 个任务真正并行（分布在 4 个进程）
        tasks = [
            loop.run_in_executor(pool, compute_primes, 50_000)
            for _ in range(4)
        ]
        results = await asyncio.gather(*tasks)
        print(f"prime counts: {results}")


asyncio.run(main())
```

### 30.3.3 设置默认 executor

如果你的应用里大量使用 `to_thread()` 或 `run_in_executor(None, ...)`，可以在启动时设置自定义的默认 executor：

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor


async def main():
    loop = asyncio.get_running_loop()

    # 用更大的线程池替换默认 executor
    pool = ThreadPoolExecutor(max_workers=20, thread_name_prefix="async-worker")
    loop.set_default_executor(pool)

    # 之后的 to_thread() 和 run_in_executor(None, ...) 都使用这个池
    await asyncio.to_thread(some_blocking_func)


asyncio.run(main())
```

---

## 30.4 `asyncio.subprocess`：异步子进程管理

标准库的 `subprocess` 是阻塞的。`asyncio` 提供了异步版本：

### 30.4.1 `create_subprocess_exec()`

精确控制参数，不经过 shell 解释，推荐使用：

```python
import asyncio


async def run_command(cmd: list[str]) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await proc.communicate()
    return proc.returncode, stdout.decode(), stderr.decode()


async def main():
    returncode, stdout, stderr = await run_command(["ls", "-la", "/tmp"])
    print(f"exit={returncode}")
    print(stdout)


asyncio.run(main())
```

### 30.4.2 `create_subprocess_shell()`

通过 shell 执行命令，支持 shell 特性（管道、通配符等），但有注入风险：

```python
import asyncio


async def main():
    proc = await asyncio.create_subprocess_shell(
        "echo hello | tr a-z A-Z",
        stdout=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    print(stdout.decode())   # "HELLO\n"


asyncio.run(main())
```

**安全提醒**：如果命令中包含用户输入，必须使用 `create_subprocess_exec()` 并严格验证参数，**不要**用 `create_subprocess_shell()` 拼接用户输入——这会导致命令注入漏洞。

### 30.4.3 流式读取大输出

`proc.communicate()` 会把所有输出读入内存。对于大量输出，应该流式读取：

```python
import asyncio


async def stream_output(cmd: list[str]):
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        print(f"  > {line.decode().rstrip()}")

    await proc.wait()
    print(f"exit code: {proc.returncode}")


async def main():
    await stream_output(["ping", "-c", "3", "127.0.0.1"])


asyncio.run(main())
```

### 30.4.4 子进程超时和取消

```python
import asyncio


async def run_with_timeout(cmd: list[str], timeout_s: float):
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        async with asyncio.timeout(timeout_s):
            stdout, stderr = await proc.communicate()
            return proc.returncode, stdout.decode()
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise


async def main():
    try:
        code, output = await run_with_timeout(
            ["sleep", "10"], timeout_s=0.5
        )
    except TimeoutError:
        print("command timed out and was killed")


asyncio.run(main())
```

### 30.4.5 并发启动多个子进程

```python
import asyncio


async def run_job(job_id: int) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        "python3", "-c", f"import time; time.sleep(0.3); print({job_id})",
        stdout=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    return job_id, stdout.decode().strip()


async def main():
    results = await asyncio.gather(
        *[run_job(i) for i in range(5)]
    )
    for job_id, output in results:
        print(f"job {job_id}: {output}")


asyncio.run(main())
```

---

## 30.5 `contextvars`：异步上下文传播

### 30.5.1 为什么需要 `contextvars`

在同步程序里，全局变量或线程本地存储（`threading.local`）常被用来传递"当前请求上下文"，例如：

- 当前用户 ID
- 请求追踪 ID（trace ID）
- 当前语言 / 时区

在 `asyncio` 里，**线程本地存储不能用**——所有协程跑在同一个线程里，不同请求的协程会共享同一个 `threading.local` 对象。

`contextvars` 模块（Python 3.7+）解决了这个问题：

> `ContextVar` 的值在每个"上下文"中独立存在，`asyncio.Task` 创建时会自动复制父协程的上下文。

### 30.5.2 基本用法

```python
import asyncio
from contextvars import ContextVar

request_id: ContextVar[str] = ContextVar("request_id", default="unknown")


async def handle_request(rid: str):
    token = request_id.set(rid)   # 设置当前上下文中的值
    try:
        await process()
    finally:
        request_id.reset(token)   # 恢复上一个值（可选，防止泄漏）


async def process():
    rid = request_id.get()
    print(f"[{rid}] processing...")
    await asyncio.sleep(0.1)
    print(f"[{rid}] done")


async def main():
    # 两个并发请求，各自的 request_id 互不干扰
    await asyncio.gather(
        handle_request("req-001"),
        handle_request("req-002"),
    )


asyncio.run(main())
```

输出会清晰地显示两个请求的日志交错，但各自的 `request_id` 不会混淆。

### 30.5.3 Task 创建时自动复制上下文

```python
import asyncio
from contextvars import ContextVar

user_id: ContextVar[int] = ContextVar("user_id")


async def child_task():
    # 能读到父协程设置的 user_id
    uid = user_id.get()
    print(f"child sees user_id={uid}")
    
    # 子任务里修改不影响父协程
    user_id.set(999)
    print(f"child set user_id=999")


async def parent():
    user_id.set(42)
    print(f"parent set user_id=42")

    task = asyncio.create_task(child_task())
    await task

    # 父协程的 user_id 不受子任务影响
    print(f"parent still has user_id={user_id.get()}")


asyncio.run(parent())
```

这正是 `contextvars` 最重要的特性：

> Task 继承父协程上下文的**快照**，子任务里的修改不影响父任务，也不影响其他并发任务。

### 30.5.4 实战：请求级 trace ID 自动传播

```python
import asyncio
import uuid
from contextvars import ContextVar
from functools import wraps
from typing import Callable

trace_id: ContextVar[str] = ContextVar("trace_id", default="")


def traced(func: Callable) -> Callable:
    """装饰器：自动在日志中附加 trace_id"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        tid = trace_id.get()
        prefix = f"[{tid}]" if tid else "[no-trace]"
        print(f"{prefix} {func.__name__} started")
        try:
            result = await func(*args, **kwargs)
            print(f"{prefix} {func.__name__} completed")
            return result
        except Exception as exc:
            print(f"{prefix} {func.__name__} failed: {exc}")
            raise
    return wrapper


@traced
async def fetch_profile(user_id: int) -> dict:
    await asyncio.sleep(0.1)
    return {"id": user_id}


@traced
async def fetch_orders(user_id: int) -> list:
    await asyncio.sleep(0.15)
    return [1, 2, 3]


@traced
async def handle_request(user_id: int):
    profile, orders = await asyncio.gather(
        fetch_profile(user_id),
        fetch_orders(user_id),
    )
    return {"profile": profile, "orders": orders}


async def gateway():
    # 模拟两个并发请求，各自有独立的 trace_id
    async def serve(user_id: int):
        tid = str(uuid.uuid4())[:8]
        trace_id.set(tid)
        await handle_request(user_id)

    await asyncio.gather(serve(1), serve(2))


asyncio.run(gateway())
```

### 30.5.5 在 `run_in_executor()` 里的上下文传播

默认情况下，`run_in_executor()` 会把当前上下文**复制**到线程里：

```python
import asyncio
from contextvars import ContextVar

request_id: ContextVar[str] = ContextVar("request_id", default="unknown")


def blocking_task():
    rid = request_id.get()   # ← 能读到协程里设置的值
    print(f"thread sees request_id={rid}")
    return rid


async def main():
    request_id.set("req-xyz")
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, blocking_task)
    print(f"thread returned: {result}")


asyncio.run(main())
```

这意味着 trace ID、用户上下文等信息会自动流入线程池中的同步代码，无需额外传参。

---

## 30.6 从同步代码调用异步代码

### 30.6.1 `asyncio.run()`：最简单的入口

在程序顶层，直接用 `asyncio.run()` 启动一个完整的事件循环：

```python
import asyncio


async def main():
    await asyncio.sleep(0.1)
    return "done"


result = asyncio.run(main())
print(result)
```

这是**从同步主程序进入异步世界**的标准方式。

### 30.6.2 `loop.run_until_complete()`（旧式 API）

在一些旧代码或框架中，你会看到：

```python
loop = asyncio.new_event_loop()
result = loop.run_until_complete(some_coroutine())
loop.close()
```

现代代码优先用 `asyncio.run()`，它处理了更多边界情况（信号、资源清理等）。

### 30.6.3 从线程安全地通知事件循环

如果你在**另一个线程**里（例如 GUI 线程、后台工作线程）需要通知 asyncio 事件循环，必须用线程安全的方法：

```python
import asyncio
import threading


async def worker_coroutine(data: str):
    await asyncio.sleep(0.1)
    print(f"processed: {data}")


def thread_function(loop: asyncio.AbstractEventLoop, data: str):
    """在另一个线程里，安全地向事件循环提交协程"""
    future = asyncio.run_coroutine_threadsafe(worker_coroutine(data), loop)
    result = future.result(timeout=5.0)   # 阻塞等待结果
    return result


async def main():
    loop = asyncio.get_running_loop()

    # 在另一个线程里向当前 loop 提交任务
    thread = threading.Thread(
        target=thread_function,
        args=(loop, "hello from thread"),
    )
    thread.start()
    thread.join()


asyncio.run(main())
```

**关键 API**：`asyncio.run_coroutine_threadsafe(coro, loop)` 返回一个 `concurrent.futures.Future`（不是 asyncio.Future），可以从任何线程阻塞等待结果。

### 30.6.4 `loop.call_soon_threadsafe()`：从线程调用回调

如果只是想从其他线程触发一个回调（而不是等待结果），可以：

```python
def notify_from_thread(loop: asyncio.AbstractEventLoop, event: asyncio.Event):
    """从其他线程安全地触发 asyncio.Event"""
    loop.call_soon_threadsafe(event.set)
```

---

## 30.7 同步/异步边界常见陷阱

### 陷阱一：在协程里调用阻塞函数

```python
# 错误
async def bad():
    data = requests.get("https://api.example.com")   # 阻塞整个 loop

# 正确
async def good():
    data = await asyncio.to_thread(requests.get, "https://api.example.com")
    # 或者更好：使用原生异步库 httpx/aiohttp
```

### 陷阱二：在线程里直接 `await` 或调用 `asyncio.run()`

```python
# 在已有 loop 运行时，不能再 asyncio.run()
def bad_thread_func():
    result = asyncio.run(some_coro())   # 如果已有 loop 运行，会报错

# 正确：使用 run_coroutine_threadsafe
def good_thread_func(loop):
    future = asyncio.run_coroutine_threadsafe(some_coro(), loop)
    return future.result()
```

### 陷阱三：混淆 `asyncio.Future` 和 `concurrent.futures.Future`

| 类型 | 所在线程 | 等待方式 |
|------|---------|---------|
| `asyncio.Future` | 事件循环线程 | `await future` |
| `concurrent.futures.Future` | 任意线程 | `future.result(timeout=...)` |

`asyncio.wrap_future()` 可以把 `concurrent.futures.Future` 转换成 `asyncio.Future`：

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor


async def main():
    pool = ThreadPoolExecutor()
    sync_future = pool.submit(blocking_function, arg)

    # 把 concurrent.futures.Future 转成 asyncio.Future
    async_future = asyncio.wrap_future(sync_future)
    result = await async_future


asyncio.run(main())
```

### 陷阱四：不关闭 executor 导致线程泄漏

```python
# 错误：executor 没有被关闭
async def main():
    pool = ThreadPoolExecutor(max_workers=10)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(pool, some_func)
    # pool 永远不会被回收

# 正确
async def main():
    with ThreadPoolExecutor(max_workers=10) as pool:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(pool, some_func)
    # 退出 with 时 pool 被关闭
```

---

## 30.8 实战：异步推理服务与同步模型的集成

下面这个例子模拟了深度学习系统里常见的场景：

- 模型是同步的（假设是 `torch` 或 `sklearn`）
- 服务层是 `asyncio`
- 需要在不阻塞事件循环的前提下跑推理

```python
import asyncio
import time
import random
from concurrent.futures import ProcessPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass

request_id: ContextVar[str] = ContextVar("request_id", default="none")


@dataclass
class InferenceInput:
    features: list[float]


@dataclass
class InferenceOutput:
    score: float
    latency_ms: float


# ---- 同步推理（假设是真实的模型） ----

def sync_inference(features: list[float]) -> float:
    """CPU 密集型：真实场景里是模型 forward pass"""
    time.sleep(random.uniform(0.05, 0.15))  # 模拟推理耗时
    return sum(features) / len(features)


# ---- 异步服务层 ----

class InferenceService:
    def __init__(self, workers: int = 4):
        self._pool = ProcessPoolExecutor(max_workers=workers)
        self._sem = asyncio.Semaphore(workers * 2)

    async def infer(self, inp: InferenceInput) -> InferenceOutput:
        rid = request_id.get()
        start = time.monotonic()

        async with self._sem:
            loop = asyncio.get_running_loop()
            score = await loop.run_in_executor(
                self._pool, sync_inference, inp.features
            )

        latency_ms = (time.monotonic() - start) * 1000
        print(f"[{rid}] score={score:.3f} latency={latency_ms:.1f}ms")
        return InferenceOutput(score=score, latency_ms=latency_ms)

    async def close(self):
        self._pool.shutdown(wait=True)


async def handle_request(service: InferenceService, req_id: str, features: list[float]):
    token = request_id.set(req_id)
    try:
        inp = InferenceInput(features=features)
        return await service.infer(inp)
    finally:
        request_id.reset(token)


async def main():
    service = InferenceService(workers=4)

    # 模拟 10 个并发推理请求
    requests = [
        (f"req-{i:03d}", [random.random() for _ in range(128)])
        for i in range(10)
    ]

    results = await asyncio.gather(
        *[handle_request(service, rid, features)
          for rid, features in requests]
    )

    latencies = [r.latency_ms for r in results]
    print(f"\np50={sorted(latencies)[len(latencies)//2]:.1f}ms "
          f"p99={sorted(latencies)[-1]:.1f}ms")

    await service.close()


asyncio.run(main())
```

这个例子展示了几个关键模式：

1. **ProcessPoolExecutor** 让 CPU 密集型模型推理真正并行
2. **Semaphore** 防止并发推理数超过 executor 容量
3. **contextvars** 让请求 ID 自动流入日志
4. **async with** 管理 Semaphore，`with` 管理 executor

---

## 30.9 什么时候应该"升级"到原生异步库

`to_thread()` 和 `run_in_executor()` 是"让阻塞代码凑合跑"的方案，不是最优解。如果你的系统对这些操作有高并发需求，应该考虑换用原生异步库：

| 同步库 | 原生异步替代 |
|--------|-------------|
| `requests` | `httpx`（支持 async）/ `aiohttp` |
| `psycopg2`（PostgreSQL）| `asyncpg` / `psycopg3` |
| `pymysql`（MySQL）| `aiomysql` |
| `redis-py`（同步）| `aioredis` / `redis.asyncio` |
| `boto3`（AWS）| `aiobotocore` / `botocore async` |
| `elasticsearch-py`（同步）| `elasticsearch-async` |

原生异步库的优势：

- 不消耗线程池资源
- 更好的连接池管理
- 更低的延迟和内存占用

---

## 本章小结

| 机制 | 核心用途 | 适合场景 |
|------|---------|---------|
| `asyncio.to_thread()` | 不阻塞 loop 地运行同步函数 | I/O 密集型阻塞库 |
| `run_in_executor(ThreadPool)` | 线程池执行，可自定义 | 同上，需要更细粒度控制 |
| `run_in_executor(ProcessPool)` | 进程池执行，绕过 GIL | CPU 密集型纯 Python 计算 |
| `asyncio.create_subprocess_exec()` | 异步子进程（无 shell） | 安全地运行外部命令 |
| `asyncio.create_subprocess_shell()` | 异步子进程（有 shell）| 管道等 shell 特性（注意注入风险）|
| `ContextVar` | 任务级上下文隔离 | 请求 ID、用户信息、trace 上下文 |
| `run_coroutine_threadsafe()` | 从线程提交协程 | 多线程与 asyncio 混用 |
| `asyncio.wrap_future()` | 转换 concurrent Future | 线程池结果引入 asyncio |

---

## 深度学习应用

本章在深度学习和 AI Infra 系统中的典型落点：

- `ProcessPoolExecutor` + `run_in_executor()`：CPU 密集型模型 forward pass 并行化
- `to_thread()`：包裹同步数据预处理库（Pillow、OpenCV）
- `asyncio.subprocess`：异步调用命令行工具（ffmpeg、torchaudio）
- `ContextVar` + trace ID：分布式推理链路追踪
- `run_coroutine_threadsafe()`：PyTorch DataLoader 后台线程回调 asyncio 事件循环

---

## 练习题

1. `asyncio.to_thread()` 和 `loop.run_in_executor(ProcessPoolExecutor(...))` 的本质区别是什么？各适合什么场景？
2. 为什么在 `asyncio` 里不能用 `threading.local()` 来隔离请求上下文？`ContextVar` 解决了什么问题？
3. 从另一个线程安全地向 asyncio 事件循环提交任务，有哪两种方式？它们的区别是什么？
4. 异步调用 `create_subprocess_exec()` 和 `create_subprocess_shell()` 有什么安全上的区别？
5. 设计一个"混合推理服务"：接收 HTTP 请求（asyncio），在线程池里跑特征预处理（同步），在进程池里跑模型推理（CPU 密集型），结果用 ContextVar 传播 trace ID，说明每一步选择的理由。
