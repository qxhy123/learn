# 第34章：极端场景高级模式

> 第27章讲了异步系统的7个核心机制和微批处理网关骨架。本章往更深处走：熔断器、令牌桶限流、异步连接池、多租户加权公平调度、批量部分失败隔离，以及雷群效应（Thundering Herd）防止。这些模式在高可用生产系统中经常出现，理解它们的 asyncio 实现，能帮你在系统出问题时知道"哪里出了什么问题"。

---

## 学习目标

完成本章学习后，你将能够：

1. 实现基于状态机的异步熔断器（Circuit Breaker）
2. 实现令牌桶和漏桶限流算法
3. 设计带健康检查的异步连接池
4. 理解并实现多租户加权公平调度器
5. 隔离批量操作中的部分失败，避免"一个失败导致全部失败"
6. 识别和防止雷群效应（Thundering Herd）

---

## 正文内容

## 34.1 熔断器（Circuit Breaker）

### 34.1.1 为什么需要熔断器

假设你的服务依赖一个下游 API。下游开始抖动，请求超时，你的系统开始重试。  
如果没有熔断器，结果是：

- 每个请求都在等超时
- 线程/协程堆积
- 下游越来越慢（被雪崩的请求压垮）
- 你的服务也被拖垮

熔断器的核心思想：

> 当下游连续失败超过阈值时，**停止发送请求**，给下游时间恢复。

### 34.1.2 熔断器的三个状态

```text
CLOSED  ──(连续失败 >= N)──►  OPEN
  ▲                            │
  │                            │(超时后)
  │                            ▼
  └──(请求成功)────────── HALF_OPEN
                (允许一个试探请求)
```

- **CLOSED（关闭）**：正常转发请求，记录失败次数
- **OPEN（断开）**：直接拒绝所有请求，不访问下游
- **HALF_OPEN（半开）**：超时后允许一个试探请求，成功则回到 CLOSED，失败则回到 OPEN

### 34.1.3 异步熔断器实现

```python
import asyncio
import time
from enum import Enum
from typing import Callable, TypeVar, Awaitable

T = TypeVar("T")


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpen(Exception):
    """熔断器开路时抛出的异常"""
    pass


class AsyncCircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_s: float = 30.0,
        half_open_max_calls: int = 1,
        name: str = "circuit_breaker",
    ):
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout_s
        self._half_open_max = half_open_max_calls
        self._name = name

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    async def _check_state(self):
        async with self._lock:
            if self._state == CircuitState.OPEN:
                # 检查是否可以进入 HALF_OPEN
                if time.monotonic() - self._last_failure_time >= self._recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    print(f"[{self._name}] → HALF_OPEN (recovery timeout elapsed)")
                else:
                    raise CircuitBreakerOpen(
                        f"Circuit {self._name!r} is OPEN"
                    )

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self._half_open_max:
                    raise CircuitBreakerOpen(
                        f"Circuit {self._name!r} is HALF_OPEN (max trial calls reached)"
                    )
                self._half_open_calls += 1

    async def _on_success(self):
        async with self._lock:
            if self._state in (CircuitState.HALF_OPEN, CircuitState.CLOSED):
                self._failure_count = 0
                if self._state == CircuitState.HALF_OPEN:
                    self._state = CircuitState.CLOSED
                    print(f"[{self._name}] → CLOSED (probe succeeded)")

    async def _on_failure(self, exc: Exception):
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                print(f"[{self._name}] → OPEN (probe failed: {exc})")
            elif (self._state == CircuitState.CLOSED and
                  self._failure_count >= self._failure_threshold):
                self._state = CircuitState.OPEN
                print(f"[{self._name}] → OPEN "
                      f"(failure_count={self._failure_count})")

    async def call(
        self,
        fn: Callable[..., Awaitable[T]],
        *args,
        **kwargs,
    ) -> T:
        await self._check_state()
        try:
            result = await fn(*args, **kwargs)
            await self._on_success()
            return result
        except CircuitBreakerOpen:
            raise
        except Exception as exc:
            await self._on_failure(exc)
            raise


# ---- 使用示例 ----

async def unstable_api(fail: bool = False) -> str:
    if fail:
        raise RuntimeError("downstream error")
    await asyncio.sleep(0.01)
    return "ok"


async def main():
    cb = AsyncCircuitBreaker(failure_threshold=3, recovery_timeout_s=0.5)

    # 模拟失败触发熔断
    for i in range(5):
        try:
            await cb.call(unstable_api, fail=True)
        except (RuntimeError, CircuitBreakerOpen) as e:
            print(f"call {i}: {type(e).__name__}: {e}")

    # 等待恢复超时
    await asyncio.sleep(0.6)

    # 试探请求
    try:
        result = await cb.call(unstable_api, fail=False)
        print(f"probe succeeded: {result}, state={cb.state}")
    except Exception as e:
        print(f"probe failed: {e}")


asyncio.run(main())
```

---

## 34.2 令牌桶限流

### 34.2.1 令牌桶 vs 漏桶

| 算法 | 特点 |
|------|------|
| 令牌桶（Token Bucket）| 允许短时突发，但长期平均速率受限 |
| 漏桶（Leaky Bucket）| 平滑输出，完全不允许突发 |

令牌桶更适合 API 限流场景，因为它允许正常的突发流量。

### 34.2.2 异步令牌桶实现

```python
import asyncio
import time


class AsyncTokenBucket:
    """
    令牌桶限流器：
    - 每秒向桶中添加 rate 个令牌
    - 桶的容量为 capacity（允许突发）
    - 每次请求消耗 tokens 个令牌
    - 如果令牌不足，等待直到令牌充足
    """

    def __init__(self, rate: float, capacity: float):
        self._rate = rate           # 每秒生成令牌数
        self._capacity = capacity   # 最大令牌数（允许的最大突发）
        self._tokens = capacity     # 当前令牌数（初始满桶）
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self._last_refill
        new_tokens = elapsed * self._rate
        self._tokens = min(self._capacity, self._tokens + new_tokens)
        self._last_refill = now

    async def acquire(self, tokens: float = 1.0):
        """获取令牌，不足时等待"""
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                # 计算需要等待多久
                wait_time = (tokens - self._tokens) / self._rate

            await asyncio.sleep(wait_time)

    async def try_acquire(self, tokens: float = 1.0) -> bool:
        """尝试获取令牌，立即返回是否成功"""
        async with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False


async def limited_api_call(bucket: AsyncTokenBucket, name: str):
    await bucket.acquire()
    print(f"[{time.monotonic():.3f}] {name} executing")
    await asyncio.sleep(0.01)
    return f"{name} result"


async def main():
    # 每秒 5 个请求，允许最多 10 个突发
    bucket = AsyncTokenBucket(rate=5.0, capacity=10.0)

    # 同时发出 15 个请求
    tasks = [
        asyncio.create_task(limited_api_call(bucket, f"req-{i:02d}"))
        for i in range(15)
    ]
    results = await asyncio.gather(*tasks)
    print(f"\ncompleted {len(results)} requests")


asyncio.run(main())
```

### 34.2.3 每租户独立限流

```python
import asyncio
from collections import defaultdict


class PerTenantRateLimiter:
    def __init__(self, rate: float, capacity: float):
        self._rate = rate
        self._capacity = capacity
        self._buckets: dict[str, AsyncTokenBucket] = {}

    def _get_bucket(self, tenant_id: str) -> AsyncTokenBucket:
        if tenant_id not in self._buckets:
            self._buckets[tenant_id] = AsyncTokenBucket(
                rate=self._rate,
                capacity=self._capacity,
            )
        return self._buckets[tenant_id]

    async def acquire(self, tenant_id: str, tokens: float = 1.0):
        await self._get_bucket(tenant_id).acquire(tokens)

    async def try_acquire(self, tenant_id: str, tokens: float = 1.0) -> bool:
        return await self._get_bucket(tenant_id).try_acquire(tokens)
```

---

## 34.3 异步连接池

### 34.3.1 为什么需要连接池

每次操作都新建连接的问题：

- 连接建立有延迟（TCP 握手、认证）
- 频繁建立/销毁连接占用资源
- 可能超过数据库/服务的连接上限

连接池的作用：

- 预先建立一批连接，复用它们
- 限制最大并发连接数
- 定期检查连接健康状况

### 34.3.2 通用异步连接池实现

```python
import asyncio
import time
from typing import AsyncContextManager
from contextlib import asynccontextmanager


class AsyncConnection:
    """模拟一个异步连接对象"""

    def __init__(self, conn_id: int):
        self.conn_id = conn_id
        self.created_at = time.monotonic()
        self._healthy = True

    async def execute(self, query: str) -> str:
        await asyncio.sleep(0.05)   # 模拟查询
        return f"result from conn-{self.conn_id}: {query}"

    async def ping(self) -> bool:
        """健康检查"""
        await asyncio.sleep(0.005)
        return self._healthy

    async def close(self):
        self._healthy = False


class AsyncConnectionPool:
    def __init__(
        self,
        min_size: int = 2,
        max_size: int = 10,
        max_idle_time_s: float = 300.0,
        health_check_interval_s: float = 30.0,
    ):
        self._min_size = min_size
        self._max_size = max_size
        self._max_idle = max_idle_time_s
        self._health_interval = health_check_interval_s

        self._pool: asyncio.Queue[AsyncConnection] = asyncio.Queue()
        self._size = 0
        self._size_lock = asyncio.Lock()
        self._idle_times: dict[int, float] = {}
        self._conn_counter = 0
        self._closed = False

    async def _create_connection(self) -> AsyncConnection:
        self._conn_counter += 1
        conn = AsyncConnection(self._conn_counter)
        await asyncio.sleep(0.02)   # 模拟连接建立
        print(f"[pool] created conn-{conn.conn_id}")
        return conn

    async def initialize(self):
        """预热：建立最小连接数"""
        for _ in range(self._min_size):
            conn = await self._create_connection()
            self._pool.put_nowait(conn)
            self._idle_times[conn.conn_id] = time.monotonic()
            async with self._size_lock:
                self._size += 1

        asyncio.create_task(self._health_check_loop(), name="pool:health-check")
        print(f"[pool] initialized with {self._min_size} connections")

    @asynccontextmanager
    async def acquire(self) -> AsyncContextManager[AsyncConnection]:
        conn = await self._get_connection()
        try:
            yield conn
        finally:
            await self._return_connection(conn)

    async def _get_connection(self) -> AsyncConnection:
        # 先尝试从池里拿
        try:
            conn = self._pool.get_nowait()
            self._idle_times.pop(conn.conn_id, None)
            return conn
        except asyncio.QueueEmpty:
            pass

        # 池里没有空闲连接：检查是否能新建
        async with self._size_lock:
            if self._size < self._max_size:
                self._size += 1
                conn = await self._create_connection()
                return conn

        # 已达上限：等待空闲连接
        conn = await self._pool.get()
        self._idle_times.pop(conn.conn_id, None)
        return conn

    async def _return_connection(self, conn: AsyncConnection):
        if not conn._healthy:
            async with self._size_lock:
                self._size -= 1
            print(f"[pool] discarded unhealthy conn-{conn.conn_id}")
            return

        self._idle_times[conn.conn_id] = time.monotonic()
        await self._pool.put(conn)

    async def _health_check_loop(self):
        while not self._closed:
            await asyncio.sleep(self._health_interval)
            await self._check_idle_connections()

    async def _check_idle_connections(self):
        now = time.monotonic()
        to_close = []

        # 找出空闲超时的连接
        for conn_id, idle_since in list(self._idle_times.items()):
            if now - idle_since > self._max_idle:
                to_close.append(conn_id)

        # 关闭超时连接（保留 min_size）
        closed = 0
        try:
            while len(to_close) > 0 and self._size - closed > self._min_size:
                conn = self._pool.get_nowait()
                if conn.conn_id in to_close:
                    await conn.close()
                    to_close.remove(conn.conn_id)
                    closed += 1
                    print(f"[pool] closed idle conn-{conn.conn_id}")
                else:
                    await self._pool.put(conn)
        except asyncio.QueueEmpty:
            pass

        if closed > 0:
            async with self._size_lock:
                self._size -= closed

    async def close_all(self):
        self._closed = True
        while not self._pool.empty():
            conn = self._pool.get_nowait()
            await conn.close()


async def main():
    pool = AsyncConnectionPool(min_size=2, max_size=5)
    await pool.initialize()

    async def worker(worker_id: int):
        async with pool.acquire() as conn:
            result = await conn.execute(f"SELECT {worker_id}")
            print(f"  worker-{worker_id}: {result}")

    await asyncio.gather(*[worker(i) for i in range(8)])
    await pool.close_all()


asyncio.run(main())
```

---

## 34.4 多租户加权公平调度

### 34.4.1 问题描述

在多租户系统里，不同租户有不同的资源配额（例如：付费用户 > 免费用户）。  
如果只用简单的 FIFO 队列：

- 高并发的免费用户会抢占所有带宽
- 付费用户感受到和免费用户一样差的延迟

**加权公平调度（Weighted Fair Queuing, WFQ）** 的思路：

- 每个租户有独立的请求队列
- 每个租户有权重（决定相对带宽比例）
- 调度器按权重分配处理机会

### 34.4.2 简化版 WFQ 实现

```python
import asyncio
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TenantConfig:
    name: str
    weight: float          # 相对权重，越大得到越多处理机会
    max_queue_size: int = 100


@dataclass(order=True)
class ScheduledRequest:
    virtual_time: float    # 虚拟时钟（决定调度顺序）
    seq: int = field(compare=True)
    tenant: str = field(compare=False)
    payload: Any = field(compare=False)
    future: asyncio.Future = field(compare=False)


class WeightedFairScheduler:
    def __init__(self, tenants: list[TenantConfig], worker_count: int = 3):
        self._tenants = {t.name: t for t in tenants}
        self._queues: dict[str, asyncio.Queue[ScheduledRequest]] = {
            t.name: asyncio.Queue(maxsize=t.max_queue_size)
            for t in tenants
        }
        self._virtual_clocks: dict[str, float] = {t.name: 0.0 for t in tenants}
        self._seq = 0
        self._worker_count = worker_count
        self._main_queue: asyncio.PriorityQueue[ScheduledRequest] = asyncio.PriorityQueue()
        self._shutdown = asyncio.Event()

    async def submit(self, tenant: str, payload: Any, timeout_s: float = 5.0) -> Any:
        """提交请求，返回处理结果"""
        if tenant not in self._tenants:
            raise ValueError(f"unknown tenant: {tenant}")

        config = self._tenants[tenant]
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        # 计算虚拟时间：weight 越大，虚拟时钟走得越慢（排序靠前）
        self._seq += 1
        vt = self._virtual_clocks[tenant] + 1.0 / config.weight
        self._virtual_clocks[tenant] = vt

        req = ScheduledRequest(
            virtual_time=vt,
            seq=self._seq,
            tenant=tenant,
            payload=payload,
            future=future,
        )

        try:
            self._queues[tenant].put_nowait(req)
            await self._main_queue.put(req)
        except asyncio.QueueFull:
            future.set_exception(RuntimeError(f"tenant {tenant!r} queue full"))

        return await asyncio.wait_for(future, timeout=timeout_s)

    async def _worker(self, worker_id: int):
        while not self._shutdown.is_set():
            try:
                req = await asyncio.wait_for(
                    self._main_queue.get(), timeout=0.1
                )
            except TimeoutError:
                continue

            if req.future.done():
                continue   # 请求已被取消或超时

            try:
                # 模拟处理
                await asyncio.sleep(0.05)
                result = f"processed by worker-{worker_id}: {req.payload}"
                if not req.future.done():
                    req.future.set_result(result)
            except Exception as exc:
                if not req.future.done():
                    req.future.set_exception(exc)

    async def run(self):
        workers = [
            asyncio.create_task(self._worker(i), name=f"wfq-worker-{i}")
            for i in range(self._worker_count)
        ]
        await self._shutdown.wait()
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

    def stop(self):
        self._shutdown.set()


async def main():
    tenants = [
        TenantConfig("premium", weight=3.0),   # 优先级 3x
        TenantConfig("standard", weight=1.5),
        TenantConfig("free", weight=1.0),
    ]
    scheduler = WeightedFairScheduler(tenants, worker_count=3)

    asyncio.create_task(scheduler.run(), name="wfq-scheduler")

    # 模拟各租户并发请求
    async def send_requests(tenant: str, n: int):
        tasks = [
            scheduler.submit(tenant, f"{tenant}-req-{i}")
            for i in range(n)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        success = sum(1 for r in results if not isinstance(r, Exception))
        print(f"{tenant}: {success}/{n} succeeded")

    await asyncio.gather(
        send_requests("premium", 10),
        send_requests("standard", 10),
        send_requests("free", 10),
    )

    scheduler.stop()
    await asyncio.sleep(0.1)


asyncio.run(main())
```

---

## 34.5 批量操作部分失败隔离

### 34.5.1 问题：一个失败导致全部失败

`asyncio.gather()` 的默认行为是：一个任务抛异常，整个 `gather()` 立即抛出。

```python
results = await asyncio.gather(task_a(), task_b(), task_c())
# 如果 task_b 失败，task_a 和 task_c 的结果全部丢失
```

这在批量操作中往往不是我们想要的——我们希望尽量收集成功的结果，记录失败的条目。

### 34.5.2 `return_exceptions=True`

```python
results = await asyncio.gather(task_a(), task_b(), task_c(),
                               return_exceptions=True)
# results 是列表，每个元素是结果或异常对象
for i, result in enumerate(results):
    if isinstance(result, Exception):
        print(f"task {i} failed: {result}")
    else:
        print(f"task {i} succeeded: {result}")
```

### 34.5.3 批量操作结果聚合器

```python
import asyncio
from dataclasses import dataclass
from typing import TypeVar, Generic, Callable, Awaitable

T = TypeVar("T")


@dataclass
class BatchResult(Generic[T]):
    succeeded: list[T]
    failed: list[tuple[Any, Exception]]   # (item, exception)
    cancelled: list[Any]

    @property
    def success_rate(self) -> float:
        total = len(self.succeeded) + len(self.failed) + len(self.cancelled)
        return len(self.succeeded) / total if total > 0 else 0.0


async def batch_execute(
    items: list,
    fn: Callable[..., Awaitable[T]],
    *,
    concurrency: int = 10,
    timeout_s: float = 5.0,
    ignore_exceptions: tuple = (Exception,),
) -> BatchResult[T]:
    """
    并发批量执行，收集所有结果（成功/失败/取消），不因部分失败中止。
    """
    sem = asyncio.Semaphore(concurrency)
    succeeded = []
    failed = []
    cancelled = []

    async def safe_execute(item) -> tuple:
        async with sem:
            try:
                async with asyncio.timeout(timeout_s):
                    result = await fn(item)
                    return ("ok", item, result)
            except asyncio.CancelledError:
                return ("cancelled", item, None)
            except ignore_exceptions as exc:
                return ("fail", item, exc)

    tasks = [asyncio.create_task(safe_execute(item)) for item in items]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    for raw in raw_results:
        if isinstance(raw, Exception):
            # gather 本身失败（不应发生，因为 safe_execute 已经捕获）
            failed.append((None, raw))
        elif raw[0] == "ok":
            succeeded.append(raw[2])
        elif raw[0] == "fail":
            failed.append((raw[1], raw[2]))
        elif raw[0] == "cancelled":
            cancelled.append(raw[1])

    return BatchResult(succeeded=succeeded, failed=failed, cancelled=cancelled)


async def fetch_item(item_id: int) -> dict:
    import random
    await asyncio.sleep(random.uniform(0.01, 0.1))
    if random.random() < 0.2:
        raise ValueError(f"item {item_id} not found")
    return {"id": item_id, "data": f"value-{item_id}"}


async def main():
    result = await batch_execute(
        list(range(20)),
        fetch_item,
        concurrency=5,
        timeout_s=0.2,
    )
    print(f"succeeded: {len(result.succeeded)}")
    print(f"failed: {len(result.failed)}")
    print(f"cancelled: {len(result.cancelled)}")
    print(f"success rate: {result.success_rate:.0%}")


asyncio.run(main())
```

---

## 34.6 雷群效应（Thundering Herd）防止

### 34.6.1 什么是雷群效应

雷群效应发生在：

- 大量任务同时等待同一个条件（例如缓存失效、服务恢复）
- 条件成立时，所有等待任务同时唤醒
- 它们同时向下游发请求，造成瞬时冲击

典型场景：

- 热点缓存失效，所有请求同时穿透到数据库
- 熔断器从 OPEN 转为 HALF_OPEN，大量请求同时试探
- 服务重启后，积压的重试请求同时发出

### 34.6.2 解决方案一：随机抖动（Jitter）

```python
import asyncio
import random


async def retry_with_jitter(
    fn,
    *args,
    max_retries: int = 3,
    base_delay_s: float = 0.5,
    max_delay_s: float = 10.0,
    jitter_factor: float = 0.5,
):
    for attempt in range(max_retries):
        try:
            return await fn(*args)
        except Exception:
            if attempt == max_retries - 1:
                raise
            # 指数退避 + 随机抖动
            base = min(base_delay_s * (2 ** attempt), max_delay_s)
            jitter = random.uniform(0, base * jitter_factor)
            delay = base + jitter
            print(f"  attempt {attempt + 1} failed, retry in {delay:.2f}s")
            await asyncio.sleep(delay)
```

### 34.6.3 解决方案二：缓存锁定（Cache Stampede Prevention）

```python
import asyncio
import time
from typing import Any


class AsyncCache:
    """带防击穿的异步缓存：同一个 key 只有一个协程负责刷新"""

    def __init__(self, ttl_s: float = 60.0):
        self._cache: dict[str, tuple[Any, float]] = {}   # {key: (value, expire_at)}
        self._loading: dict[str, asyncio.Future] = {}    # 正在加载中的 key

    async def get_or_load(
        self,
        key: str,
        loader: Callable[[], Awaitable[Any]],
    ) -> Any:
        # 命中有效缓存
        if key in self._cache:
            value, expire_at = self._cache[key]
            if time.monotonic() < expire_at:
                return value

        # 已有协程在加载同一个 key，等待它完成
        if key in self._loading:
            print(f"  [{key}] waiting for in-flight load")
            return await asyncio.shield(self._loading[key])

        # 我来负责加载
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._loading[key] = future

        try:
            value = await loader()
            self._cache[key] = (value, time.monotonic() + 60.0)
            future.set_result(value)
            return value
        except Exception as exc:
            future.set_exception(exc)
            raise
        finally:
            self._loading.pop(key, None)


cache = AsyncCache()


async def slow_db_query(key: str) -> str:
    print(f"  [{key}] loading from DB...")
    await asyncio.sleep(0.3)
    return f"db_value_for_{key}"


async def main():
    # 10 个协程同时请求同一个 key，只触发一次 DB 查询
    results = await asyncio.gather(
        *[cache.get_or_load("hot_key", lambda: slow_db_query("hot_key"))
          for _ in range(10)]
    )
    print(f"all got: {set(results)}")


asyncio.run(main())
```

### 34.6.4 解决方案三：错峰启动（Staggered Startup）

```python
import asyncio
import random


async def staggered_start(
    workers: list[asyncio.Task],
    base_delay_s: float = 0.1,
    jitter_s: float = 0.05,
):
    """让 worker 们错峰启动，避免同时向下游发起连接"""
    for i, worker in enumerate(workers):
        delay = base_delay_s * i + random.uniform(0, jitter_s)
        await asyncio.sleep(delay)
        print(f"  started worker {i} (delay={delay:.2f}s)")
```

---

## 34.7 综合示例：防雪崩的推理服务

把本章所有模式组合起来：

```python
import asyncio
import random
import time
from collections import defaultdict


class ResilientInferenceGateway:
    """
    集成了熔断器、令牌桶限流、连接池和防雷群措施的推理网关骨架。
    """

    def __init__(self):
        # 限流：每秒 100 个请求，允许突发 200
        self._rate_limiter = AsyncTokenBucket(rate=100.0, capacity=200.0)

        # 熔断器：5 次失败触发
        self._circuit_breaker = AsyncCircuitBreaker(
            failure_threshold=5,
            recovery_timeout_s=10.0,
            name="inference-backend",
        )

        # 有界队列
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=500)

        self._stats = defaultdict(int)
        self._shutdown = asyncio.Event()

    async def infer(self, payload: dict, timeout_s: float = 2.0) -> dict:
        # 1. 令牌桶限流
        if not await self._rate_limiter.try_acquire():
            self._stats["rate_limited"] += 1
            raise RuntimeError("rate limited")

        # 2. 熔断器检查
        try:
            result = await self._circuit_breaker.call(
                self._do_infer, payload, timeout_s=timeout_s
            )
            self._stats["success"] += 1
            return result
        except CircuitBreakerOpen:
            self._stats["circuit_open"] += 1
            raise
        except Exception:
            self._stats["error"] += 1
            raise

    async def _do_infer(self, payload: dict, timeout_s: float) -> dict:
        async with asyncio.timeout(timeout_s):
            # 模拟后端推理，有一定失败率
            await asyncio.sleep(random.uniform(0.05, 0.2))
            if random.random() < 0.1:
                raise RuntimeError("backend inference failed")
            return {"result": sum(payload.get("features", [0]))}

    def get_stats(self) -> dict:
        return dict(self._stats)


async def main():
    gateway = ResilientInferenceGateway()

    async def send_request(i: int):
        payload = {"features": [i, i + 1, i + 2]}
        try:
            result = await gateway.infer(payload, timeout_s=0.5)
            return result
        except Exception as exc:
            return f"error: {exc}"

    # 并发发出 50 个请求
    results = await asyncio.gather(
        *[send_request(i) for i in range(50)],
        return_exceptions=True,
    )

    success = sum(1 for r in results if isinstance(r, dict))
    errors  = sum(1 for r in results if not isinstance(r, dict))
    print(f"\nsuccess={success} errors={errors}")
    print(f"stats: {gateway.get_stats()}")


asyncio.run(main())
```

---

## 本章小结

| 模式 | 解决的问题 | 关键机制 |
|------|-----------|---------|
| 熔断器 | 防止雪崩传播 | 状态机：CLOSED→OPEN→HALF_OPEN |
| 令牌桶 | 速率限制，允许突发 | 时间窗口内的令牌补充 |
| 连接池 | 减少连接建立开销，限制并发 | 队列 + 健康检查 |
| 加权公平调度 | 多租户公平分配资源 | 虚拟时钟 + 优先级队列 |
| 批量部分失败隔离 | 收集尽可能多的成功结果 | `return_exceptions=True` + 隔离 wrapper |
| 随机抖动 | 防止重试风暴 | 指数退避 + 随机延迟 |
| 缓存锁定 | 防止缓存击穿 | 单次加载 Future 共享 |
| 错峰启动 | 防止连接风暴 | 按序延迟启动 |

---

## 深度学习应用

本章模式在 AI Infra 中的典型落点：

- **熔断器**：推理网关对模型服务的调用，防止单个模型实例故障导致全链路卡死
- **令牌桶**：每用户/每租户的推理请求限速
- **连接池**：向向量数据库、特征存储的连接复用
- **WFQ**：付费用户优先于免费用户获得推理资源
- **部分失败隔离**：批量 embedding 请求中，容忍少数失败而不中止整批
- **缓存锁定**：热点 embedding 缓存失效时防止惊群

---

## 练习题

1. 熔断器的 HALF_OPEN 状态的作用是什么？如果没有这个状态会怎样？
2. 令牌桶和漏桶的本质区别是什么？为什么令牌桶更适合 API 限流场景？
3. 连接池中 `health_check` 的必要性是什么？没有健康检查的连接池可能出现什么问题？
4. 在多租户场景里，为什么简单的 FIFO 队列不够，而加权公平调度更合适？
5. 设计一个防止"缓存雷群效应"的系统：当多个协程同时请求一个刚过期的缓存 key 时，只触发一次后端查询，其他协程等待这次查询的结果。
