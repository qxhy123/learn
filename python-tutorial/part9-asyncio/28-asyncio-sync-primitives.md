# 第28章：asyncio同步原语完整指南

> `asyncio` 提供了一套与线程同步原语镜像对应的异步协调工具：`Lock`、`Condition`、`Semaphore`、`BoundedSemaphore`、`Event`，以及多种 `Queue` 变体。它们看起来像线程库的翻版，但内部机制和正确用法有显著差异。真正掌握它们，关键在于理解：**哪个原语表达"资源独占"，哪个表达"条件等待"，哪个表达"状态广播"，哪个表达"数据流动"。**

> **版本提示**：本章代码适用于 Python 3.11+。`asyncio.timeout()` 于 3.11 正式稳定；`asyncio.Lock` 等原语自 3.4 起即可用。

---

## 学习目标

完成本章学习后，你将能够：

1. 理解并正确使用 `asyncio.Lock` 保护共享状态，并识别异步死锁模式
2. 理解 `asyncio.Condition` 的语义，并能用它实现精确的"条件唤醒"
3. 区分 `Semaphore` 与 `BoundedSemaphore`，并说明各自的适用场景
4. 掌握 `Queue`、`PriorityQueue`、`LifoQueue` 的差异与适用模式
5. 知道如何在复杂系统中选择正确的同步原语

---

## 正文内容

## 28.1 为什么异步同步原语和线程同步原语不同

在线程模型中，同步原语的核心目的是：

> 防止两个线程同时修改同一块内存。

在 `asyncio` 中，情况有所不同：

> **asyncio 是单线程的**——在任意时刻，只有一个协程在执行。

这意味着：

- 你不会像多线程那样遇到真正的"数据竞争"（data race）
- 但你**仍然会遇到"交错修改"问题**——在一个 `await` 点暂停时，另一个协程可以修改共享状态

例如：

```python
# 看起来安全，实际上不安全
counter = 0

async def increment():
    global counter
    current = counter
    await asyncio.sleep(0)   # ← 在这里被打断
    counter = current + 1   # ← 此时另一个协程已经修改了 counter
```

这正是 `asyncio` 同步原语存在的意义：

> 不是防多线程竞争，而是防协程在 `await` 点之间的**交错修改**。

---

## 28.2 `asyncio.Lock`：互斥锁

### 28.2.1 基本语义

`Lock` 保证：**在任意时刻，最多只有一个协程持有锁**。

其他试图获取锁的协程会挂起，等待锁被释放。

```python
import asyncio

lock = asyncio.Lock()

async def safe_increment(counter: list):
    async with lock:
        val = counter[0]
        await asyncio.sleep(0)   # 模拟 I/O：没有 lock 时这里会导致交错
        counter[0] = val + 1
```

### 28.2.2 完整示例：无锁 vs 有锁的对比

```python
import asyncio


async def unsafe_run():
    counter = [0]

    async def increment():
        val = counter[0]
        await asyncio.sleep(0)  # 切出去，让其他协程运行
        counter[0] = val + 1

    tasks = [asyncio.create_task(increment()) for _ in range(100)]
    await asyncio.gather(*tasks)
    return counter[0]


async def safe_run():
    counter = [0]
    lock = asyncio.Lock()

    async def increment():
        async with lock:
            val = counter[0]
            await asyncio.sleep(0)
            counter[0] = val + 1

    tasks = [asyncio.create_task(increment()) for _ in range(100)]
    await asyncio.gather(*tasks)
    return counter[0]


async def main():
    unsafe_result = await unsafe_run()
    safe_result = await safe_run()
    print(f"unsafe: {unsafe_result}")   # 通常远小于 100
    print(f"safe:   {safe_result}")     # 总是 100


asyncio.run(main())
```

### 28.2.3 两种获取方式

推荐使用 `async with` 语法，它会自动处理释放：

```python
# 推荐
async with lock:
    ...

# 等价于
await lock.acquire()
try:
    ...
finally:
    lock.release()
```

永远优先用 `async with`，因为它能在异常时自动释放，不会留下锁死系统。

### 28.2.4 异步死锁怎么发生

和线程死锁一样，异步死锁通常来自：

1. 同一个协程两次 `acquire()` 同一把锁（asyncio.Lock 不可重入）
2. 协程 A 持锁等 B，协程 B 持锁等 A

```python
# 死锁示例（不可重入）
lock = asyncio.Lock()

async def bad():
    async with lock:
        async with lock:   # ← 永远挂在这里
            print("never reached")
```

**asyncio 没有可重入锁（RLock）**——如果你需要在同一个协程中重复获取"同一把锁"，应重新设计代码结构，避免递归持锁。

### 28.2.5 何时用 Lock

| 场景 | 适合 Lock |
|------|-----------|
| 保护共享的可变状态 | 是 |
| 序列化对外部资源的写操作 | 是 |
| 等待某个条件成立 | 否，用 Condition |
| 限制并发数量 | 否，用 Semaphore |

---

## 28.3 `asyncio.Condition`：条件等待

`Condition` 解决的问题是：

> 一个协程需要等待**某个条件真正成立**后才能继续，而不只是"等锁空闲"。

### 28.3.1 基本 API

```python
cond = asyncio.Condition()

# 等待者
async with cond:
    await cond.wait()      # 释放锁 + 挂起，等待被通知
    # 被唤醒后重新持有锁，在这里继续

# 通知者
async with cond:
    cond.notify()          # 唤醒一个等待者
    # 或
    cond.notify_all()      # 唤醒所有等待者
```

注意：`wait()` 会自动**先释放锁，再挂起**；被唤醒后再**重新持有锁**。这是 Condition 设计中最容易误解的地方。

### 28.3.2 完整示例：有界缓冲区

```python
import asyncio
from collections import deque


class BoundedBuffer:
    def __init__(self, maxsize: int):
        self._buf: deque = deque()
        self._maxsize = maxsize
        self._cond = asyncio.Condition()

    async def put(self, item):
        async with self._cond:
            while len(self._buf) >= self._maxsize:
                await self._cond.wait()   # 满了就等消费者消费
            self._buf.append(item)
            self._cond.notify_all()       # 通知消费者有新数据

    async def get(self):
        async with self._cond:
            while not self._buf:
                await self._cond.wait()   # 空了就等生产者生产
            item = self._buf.popleft()
            self._cond.notify_all()       # 通知生产者有空间了
            return item


async def producer(buf: BoundedBuffer, n: int):
    for i in range(n):
        await buf.put(i)
        print(f"put {i}")


async def consumer(buf: BoundedBuffer, n: int):
    for _ in range(n):
        item = await buf.get()
        print(f"  got {item}")


async def main():
    buf = BoundedBuffer(maxsize=3)
    async with asyncio.TaskGroup() as tg:
        tg.create_task(producer(buf, 8))
        tg.create_task(consumer(buf, 8))


asyncio.run(main())
```

### 28.3.3 `wait()` 必须用 `while` 而不是 `if`

这是使用 `Condition` 最重要的规则：

```python
# 错误写法
async with cond:
    if not condition_met:
        await cond.wait()     # 被虚假唤醒时出错

# 正确写法
async with cond:
    while not condition_met:  # 每次唤醒后重新检查条件
        await cond.wait()
```

原因：`notify()` 唤醒的协程不一定真的满足了你的条件——可能是另一个协程发了通知，但你关心的状态并没有变化。

### 28.3.4 `wait_for()` 方法

`Condition` 还提供了更简洁的 `wait_for()` 方法，接受一个判断函数：

```python
async with cond:
    await cond.wait_for(lambda: len(buf) > 0)
    # 等价于：
    # while not (len(buf) > 0):
    #     await cond.wait()
```

### 28.3.5 Condition vs Event 选哪个

| 需求 | 推荐 |
|------|------|
| 等待"某个状态变化"，醒来后继续执行 | `Condition` |
| 等待"某件事已经发生"，永久有效 | `Event` |
| 多个等待者关心不同条件 | `Condition` |
| 所有等待者只关心同一个"开关" | `Event` |

核心区别：

- `Event.set()` 后，状态**永久为已触发**，新来的等待者会立即返回
- `Condition` 被 `notify()` 后，新来的等待者仍然需要等下一次通知

---

## 28.4 `asyncio.Semaphore` 与 `BoundedSemaphore`

### 28.4.1 Semaphore 语义

`Semaphore` 维护一个内部计数器，表示"还剩多少并发名额"：

- `acquire()`：计数器减 1；为 0 时挂起等待
- `release()`：计数器加 1；唤醒一个等待者

```python
sem = asyncio.Semaphore(3)   # 最多 3 个协程同时执行

async def call_api(url: str):
    async with sem:
        # 最多 3 个并发进入这里
        ...
```

### 28.4.2 完整示例：并发抓取限速

```python
import asyncio


async def fetch(session_id: int, sem: asyncio.Semaphore):
    async with sem:
        print(f"[{session_id}] fetching...")
        await asyncio.sleep(0.3)
        print(f"[{session_id}] done")
        return session_id


async def main():
    sem = asyncio.Semaphore(3)   # 最多 3 个并发
    tasks = [asyncio.create_task(fetch(i, sem)) for i in range(10)]
    results = await asyncio.gather(*tasks)
    print(f"results: {results}")


asyncio.run(main())
```

### 28.4.3 BoundedSemaphore 和 Semaphore 的区别

`BoundedSemaphore` 和 `Semaphore` 唯一的区别是：

> `BoundedSemaphore` 不允许 `release()` 超过初始值。

```python
sem = asyncio.Semaphore(2)
sem.release()
sem.release()
sem.release()   # Semaphore: 允许，内部计数器变成 5
                # BoundedSemaphore: 抛出 ValueError

bsem = asyncio.BoundedSemaphore(2)
bsem.release()  # OK
bsem.release()  # OK
bsem.release()  # ← ValueError: BoundedSemaphore released too many times
```

适用场景判断：

| 情况 | 选择 |
|------|------|
| 控制外部资源并发（HTTP、DB 连接）| `Semaphore` |
| 信号量有明确"配额"语义、不能超发 | `BoundedSemaphore` |
| 不确定 | 优先用 `BoundedSemaphore`，更能暴露错误 |

---

## 28.5 `asyncio.Queue` 变体完整指南

第26章已经用 `Queue` 做了生产者/消费者的基础演示。本节补充三个重要主题：

1. 三种队列变体的区别
2. `task_done()` 和 `join()` 的语义
3. 常见陷阱

### 28.5.1 三种队列

| 类型 | 顺序 | 适用场景 |
|------|------|----------|
| `Queue` | FIFO（先进先出）| 通用任务队列 |
| `PriorityQueue` | 优先级最小先出 | 高优先级任务插队 |
| `LifoQueue` | LIFO（后进先出）| DFS、栈式处理 |

#### PriorityQueue 示例

```python
import asyncio


async def main():
    pq: asyncio.PriorityQueue = asyncio.PriorityQueue()

    # (优先级, 数据)，数值越小优先级越高
    await pq.put((3, "low priority"))
    await pq.put((1, "urgent"))
    await pq.put((2, "normal"))

    while not pq.empty():
        priority, item = await pq.get()
        print(f"priority={priority}: {item}")


asyncio.run(main())
# 输出顺序：urgent -> normal -> low priority
```

如果队列元素不可比较（例如自定义对象），可以用 `(priority, seq_num, item)` 的三元组来保证稳定排序：

```python
import itertools

counter = itertools.count()

async def put_with_priority(pq, priority, item):
    await pq.put((priority, next(counter), item))
```

### 28.5.2 `task_done()` 和 `join()` 的语义

`Queue.join()` 会等待所有已入队的条目被标记为 `task_done()`。  
这比单纯等待队列为空更准确——因为队列为空时，消费者可能还在处理最后一批数据。

```python
import asyncio


async def worker(queue: asyncio.Queue):
    while True:
        item = await queue.get()
        try:
            await asyncio.sleep(0.1)  # 模拟处理
            print(f"done: {item}")
        finally:
            queue.task_done()         # ← 必须调用，否则 join() 永远不返回


async def main():
    queue: asyncio.Queue[int] = asyncio.Queue()
    worker_task = asyncio.create_task(worker(queue))

    for i in range(5):
        await queue.put(i)

    await queue.join()        # 等所有条目被处理完
    worker_task.cancel()
    await asyncio.gather(worker_task, return_exceptions=True)
    print("all done")


asyncio.run(main())
```

**常见陷阱**：如果消费者在处理时抛出异常并被 `except` 吞掉，导致 `task_done()` 没有被调用，`join()` 会永远阻塞。解决方案是无论是否异常，都在 `finally` 里调用 `task_done()`。

### 28.5.3 `put_nowait()` 和 `get_nowait()`

这两个方法是**非阻塞**版本：

```python
try:
    queue.put_nowait(item)   # 队列满时抛出 QueueFull
except asyncio.QueueFull:
    print("queue full, dropping item")

try:
    item = queue.get_nowait()  # 队列空时抛出 QueueEmpty
except asyncio.QueueEmpty:
    print("nothing to process")
```

适合用在：
- 不想等待的快速路径
- 需要检测队列状态但不愿挂起的场合

### 28.5.4 队列作为"停机信号"的惯用法

使用哨兵值（sentinel）来优雅地通知 worker 停止：

```python
import asyncio

SENTINEL = object()  # 唯一哨兵


async def worker(queue: asyncio.Queue):
    while True:
        item = await queue.get()
        if item is SENTINEL:
            queue.task_done()
            break               # 收到哨兵，退出
        await asyncio.sleep(0.1)
        print(f"processed: {item}")
        queue.task_done()


async def main():
    queue: asyncio.Queue = asyncio.Queue()
    worker_task = asyncio.create_task(worker(queue))

    for i in range(5):
        await queue.put(i)

    await queue.put(SENTINEL)   # 发送停止信号
    await queue.join()
    print("worker finished")


asyncio.run(main())
```

---

## 28.6 深入 `asyncio.Event`：高级模式

第25章已经介绍了 `Event` 的基本用法。本节补充两个重要的高级模式。

### 28.6.1 一次性 vs 可复位 Event

`Event` 默认是可复位的：

```python
event = asyncio.Event()
event.set()
event.clear()   # 复位为未触发
```

但在很多系统中，"启动就绪"这类事件是**一次性的**——一旦触发就不应该再 `clear()`。

建议：如果你的 `Event` 是一次性的，在代码注释或命名中明确说明：

```python
model_ready = asyncio.Event()   # one-shot: never cleared after set
```

并在 set 后加保护：

```python
if not model_ready.is_set():
    model_ready.set()
```

### 28.6.2 多阶段启动屏障

用多个 `Event` 模拟多阶段启动：

```python
import asyncio


async def stage_runner(
    name: str,
    wait_for: asyncio.Event | None,
    signal: asyncio.Event,
    work_s: float,
):
    if wait_for:
        await wait_for.wait()
        print(f"  [{name}] prerequisite done, starting")
    else:
        print(f"  [{name}] starting immediately")

    await asyncio.sleep(work_s)
    print(f"  [{name}] complete")
    signal.set()


async def main():
    db_ready    = asyncio.Event()
    cache_ready = asyncio.Event()
    app_ready   = asyncio.Event()

    async with asyncio.TaskGroup() as tg:
        tg.create_task(stage_runner("db",    None,        db_ready,    0.3))
        tg.create_task(stage_runner("cache", None,        cache_ready, 0.2))
        tg.create_task(stage_runner("app",   db_ready,    app_ready,   0.1))
        # app 等待 db 就绪后才启动

    print("all stages done, app_ready:", app_ready.is_set())


asyncio.run(main())
```

---

## 28.7 同步原语选择决策树

```text
需要协调多个协程？
│
├── 需要"某一时刻只有一个协程操作共享状态"？
│       └── asyncio.Lock
│
├── 需要"等待某个条件成立，条件可能在等待期间多次变化"？
│       └── asyncio.Condition
│
├── 需要"某件事情发生了，广播给所有等待者"？
│       └── asyncio.Event
│
├── 需要"限制最多 N 个协程并发进入某段逻辑"？
│   ├── 可以多 release？ → asyncio.Semaphore
│   └── 不允许超发？   → asyncio.BoundedSemaphore
│
└── 需要"协程之间传递数据"？
    ├── FIFO顺序    → asyncio.Queue
    ├── 优先级顺序  → asyncio.PriorityQueue
    └── LIFO顺序    → asyncio.LifoQueue
```

---

## 28.8 实战：带优先级的异步任务调度器

下面这个例子组合 `PriorityQueue`、`Semaphore`、`Event` 和 `Lock`：

```python
import asyncio
import itertools
import time
from dataclasses import dataclass, field


@dataclass(order=True)
class Job:
    priority: int
    seq: int = field(compare=True)
    name: str = field(compare=False)
    payload: float = field(compare=False)


class PriorityScheduler:
    def __init__(self, concurrency: int = 3):
        self._queue: asyncio.PriorityQueue[Job] = asyncio.PriorityQueue()
        self._sem = asyncio.Semaphore(concurrency)
        self._shutdown = asyncio.Event()
        self._stats_lock = asyncio.Lock()
        self._stats = {"done": 0, "failed": 0}
        self._seq = itertools.count()

    async def submit(self, name: str, priority: int, work_s: float):
        job = Job(priority=priority, seq=next(self._seq),
                  name=name, payload=work_s)
        await self._queue.put(job)

    async def _run_job(self, job: Job):
        async with self._sem:
            try:
                print(f"  running [{job.priority}] {job.name}")
                await asyncio.sleep(job.payload)
                async with self._stats_lock:
                    self._stats["done"] += 1
            except Exception:
                async with self._stats_lock:
                    self._stats["failed"] += 1
            finally:
                self._queue.task_done()

    async def dispatch_loop(self):
        while not self._shutdown.is_set():
            try:
                job = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.01)
                continue
            asyncio.create_task(self._run_job(job))

    async def run_until_done(self):
        dispatch = asyncio.create_task(self.dispatch_loop())
        await self._queue.join()
        self._shutdown.set()
        dispatch.cancel()
        await asyncio.gather(dispatch, return_exceptions=True)
        return dict(self._stats)


async def main():
    scheduler = PriorityScheduler(concurrency=3)

    jobs = [
        ("low-1",    3, 0.2),
        ("urgent-1", 1, 0.1),
        ("normal-1", 2, 0.15),
        ("urgent-2", 1, 0.1),
        ("low-2",    3, 0.2),
        ("normal-2", 2, 0.15),
    ]

    for name, priority, work_s in jobs:
        await scheduler.submit(name, priority, work_s)

    stats = await scheduler.run_until_done()
    print(f"\nstats: {stats}")


asyncio.run(main())
```

这个例子把本章所有原语用在了真实场景：

- `PriorityQueue`：高优先级任务先被调度
- `Semaphore`：限制并发执行数量
- `Event`：停机信号
- `Lock`：保护统计计数器

---

## 本章小结

| 原语 | 核心语义 | 典型使用模式 |
|------|----------|-------------|
| `Lock` | 互斥访问共享状态 | `async with lock:` |
| `Condition` | 等待条件成立 | `while not cond: await c.wait()` |
| `Event` | 广播状态变化 | `await e.wait()` / `e.set()` |
| `Semaphore` | 限制并发数量 | `async with sem:` |
| `BoundedSemaphore` | 限制并发 + 不允许超发 | 同上，更严格 |
| `Queue` | FIFO 数据流 | `await q.put()` / `await q.get()` |
| `PriorityQueue` | 按优先级消费 | 元组 `(priority, data)` |
| `LifoQueue` | LIFO 数据流 | DFS、撤销栈 |

---

## 深度学习应用

在深度学习系统中，本章原语的典型落点：

- `Lock`：保护并发写入的指标计数器、共享缓存更新
- `Condition`：动态批处理中"等批次凑满再触发推理"
- `Event`：模型加载就绪信号、检查点保存完成通知
- `Semaphore`：控制并发向量数据库请求数量
- `PriorityQueue`：多租户场景按 SLA 等级排序推理请求

---

## 练习题

1. 解释为什么 `asyncio.Lock` 在协程里仍然必要，尽管 asyncio 是单线程的。
2. `Condition.wait()` 必须在 `while` 循环中调用，而不是 `if`，原因是什么？
3. `BoundedSemaphore` 和 `Semaphore` 的区别是什么？什么场景更适合用 `BoundedSemaphore`？
4. `queue.task_done()` 和 `queue.join()` 的配合语义是什么？如果消费者抛出异常没有调用 `task_done()` 会发生什么？
5. 设计一个"多阶段流水线"：阶段 A 完成后通知阶段 B，阶段 B 完成后通知阶段 C，说明应该用哪种原语，为什么。
