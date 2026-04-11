# 第26章：asyncio网络编程与异步服务实战

> 真正把 `asyncio` 用进生产系统后，问题就不再是“会不会写协程”，而是：如何处理慢客户端、背压、限流、优雅关闭、连接泄漏、半失败，以及这些机制如何统一进入 event loop 的生命周期管理。

---

## 学习目标

完成本章学习后，你将能够：

1. 使用 `asyncio` 的流式 API 构建异步 TCP 服务
2. 理解背压（backpressure）为什么是异步服务的生死线
3. 理解 stream API 与更底层 transport/protocol 的关系
4. 理解 `Event`、signal、任务监督如何共同构成服务生命周期控制面
5. 设计生产者 / 消费者、连接池、超时和限流策略

---

## 正文内容

## 26.1 用 `asyncio.start_server()` 搭一个最小异步服务

标准库已经提供了构建 TCP 服务的能力：

```python
import asyncio


async def handle_client(reader: asyncio.StreamReader,
                        writer: asyncio.StreamWriter):
    addr = writer.get_extra_info("peername")
    try:
        data = await reader.readline()
        if not data:
            return
        message = data.decode().strip()
        writer.write(f"echo:{message}\n".encode())
        await writer.drain()
    finally:
        writer.close()
        await writer.wait_closed()


async def main():
    server = await asyncio.start_server(handle_client, "127.0.0.1", 8888)
    async with server:
        await server.serve_forever()


asyncio.run(main())
```

这个例子虽然简单，但已经包含两个生产系统必须遵守的原则：

1. 写数据后要 `await writer.drain()`
2. 连接结束时要显式关闭和等待关闭

它也说明了一点：

> event loop 不只是“跑协程”，它还承担 socket 读写事件和连接生命周期管理。

### Stream API 只是更高层的封装

你在 `asyncio.start_server()` 里最常见的是：

- `StreamReader`
- `StreamWriter`

这套接口很友好，但它不是 `asyncio` 唯一的网络模型。  
更底层还有一套：

- **transport**
- **protocol**

可以把两层关系先记成：

```text
高层：StreamReader / StreamWriter
低层：Transport / Protocol
```

多数业务开发者直接用 stream API 就够了，但如果你想理解：

- 框架为什么能在底层“收到字节后再恢复协程”
- callback 风格与 coroutine 风格如何衔接
- 某些高性能网络库为什么更偏 protocol

那么 transport / protocol 必须讲清楚。

## 26.1.1 什么是 transport / protocol

这是 `asyncio` 较早的一套低层接口设计。

### transport

transport 负责：

- 底层连接
- 读写缓冲
- 关闭 / 半关闭
- pause / resume reading 等流控操作

你可以把它理解成：

> “把字节真的搬出去 / 搬进来”的那层对象。

### protocol

protocol 负责：

- 收到连接时怎么办
- 收到字节时怎么办
- 连接断开时怎么办
- 错误发生时怎么办

它更像一个 callback handler。

也就是说：

| 对象 | 更像什么 |
|------|----------|
| transport | 数据通道和缓冲控制层 |
| protocol | 面向事件的回调逻辑层 |

### 一个最小 protocol 示例

```python
import asyncio


class EchoProtocol(asyncio.Protocol):
    def connection_made(self, transport):
        self.transport = transport
        peer = transport.get_extra_info("peername")
        print("connection from", peer)

    def data_received(self, data):
        message = data.decode()
        print("received:", message)
        self.transport.write(f"echo:{message}".encode())

    def connection_lost(self, exc):
        print("connection closed", exc)


async def main():
    loop = asyncio.get_running_loop()
    server = await loop.create_server(EchoProtocol, "127.0.0.1", 9998)
    async with server:
        await server.serve_forever()


asyncio.run(main())
```

这个例子和 `start_server()` 的最大区别在于：  
它几乎完全是 callback / protocol 风格，而不是 `await reader.read()` 风格。

### 为什么现代教程仍然要讲它

因为它帮你建立两个认识：

1. `asyncio` 底层并不只有协程写法
2. stream API 是在低层 transport/protocol 之上的更友好的抽象

如果你以后读底层框架、网络库、旧代码或某些性能优化实现，这一层很容易遇到。

---

## 26.2 背压：为什么 `drain()` 不是可有可无

很多人会写：

```python
writer.write(big_bytes)
```

然后忘记 `await writer.drain()`。  
这样做的问题是：如果对端读得慢，发送缓冲会持续膨胀，最终让你的进程：

- 内存越来越高
- 任务越来越多
- 整体延迟恶化

`drain()` 的意义就是：

> 当下游慢时，当前协程也必须慢下来。

这就是背压的核心哲学：  
**不要让上游无限快地产生工作，而下游已经明显处理不过来。**

### transport / protocol 视角下的背压

在 stream API 里，你常见的是：

- `writer.write(...)`
- `await writer.drain()`

在更底层的 transport / protocol 模型里，则更接近：

- `pause_reading()`
- `resume_reading()`
- 写缓冲高低水位控制

这说明背压不是某个单一 API，而是一个更底层的流控思想：

> 读得太快 / 写得太快时，必须让上游慢下来。

### 用有界队列表达背压

除了 `drain()`，另一个经典做法是使用 `Queue(maxsize=N)`：

```python
queue = asyncio.Queue(maxsize=1000)
await queue.put(item)
```

当队列满时，生产者会阻塞在 `put()`，这等价于在系统里明确地插入背压。

---

## 26.3 生产者 / 消费者模型

异步服务最常见的结构之一，就是生产者 / 消费者：

```text
请求入口 -> Queue -> Worker Pool -> 下游依赖 / 响应
```

优点：

- 把入口与处理解耦
- 方便做限流和批处理
- 方便记录积压长度

但也会带来两个新问题：

1. 队列该多大？
2. 积压时要丢弃、等待还是降级？

这说明队列不是缓冲区那么简单，而是策略决策点。

### 一个有界 worker pool

```python
import asyncio


async def worker(name: str, queue: asyncio.Queue):
    while True:
        item = await queue.get()
        try:
            await asyncio.sleep(0.2)  # 模拟 I/O
            print(f"{name} processed {item}")
        finally:
            queue.task_done()


async def main():
    queue: asyncio.Queue[int] = asyncio.Queue(maxsize=10)

    workers = [asyncio.create_task(worker(f"w{i}", queue)) for i in range(3)]

    for i in range(20):
        await queue.put(i)  # 队列满时形成背压

    await queue.join()

    for w in workers:
        w.cancel()
    await asyncio.gather(*workers, return_exceptions=True)


asyncio.run(main())
```

这个结构是很多复杂异步服务的基础。

---

## 26.4 超时、限流、重试和抖动（jitter）

在生产服务里，“请求失败”通常不是单一原因，而是几类问题叠加：

- 上游流量暴增
- 下游依赖变慢
- 某一批请求全超时
- 所有客户端同时重试，形成重试风暴

因此常见保护手段包括：

### 超时

```python
async with asyncio.timeout(1.5):
    await call_dependency()
```

### 限流

```python
sem = asyncio.Semaphore(50)
async with sem:
    await call_dependency()
```

### 重试 + 抖动

```python
import asyncio
import random


async def retry_with_jitter(fn, retries=3, base=0.2):
    for attempt in range(retries):
        try:
            return await fn()
        except Exception:
            if attempt == retries - 1:
                raise
            delay = base * (2 ** attempt) + random.uniform(0, 0.1)
            await asyncio.sleep(delay)
```

这里“抖动”很关键，因为如果所有请求都在同样时间重试，就会造成同步冲击。

---

## 26.5 `asyncio.Event` 在服务生命周期中的作用

上一章把 `Event` 作为基础协调原语来讲，这一章要把它放到服务生命周期里看。

很多异步服务其实都隐含着几个关键状态：

- 服务是否已启动完成
- 是否开始 draining
- 是否收到停机信号
- 某个后台 worker 是否允许继续接活

这些都很适合用 `Event` 来表达。

### 为什么 `Event` 很适合做 shutdown signal

因为停机通常具有“一对多广播”特性：

- 网关协程要知道
- worker 要知道
- 批处理器要知道
- 连接清理器也要知道

因此共享一个：

```python
shutdown_event = asyncio.Event()
```

然后所有后台任务都监听它，是非常自然的设计。

### `Event` 与优雅停机不是同一个概念

这是必须讲透的一点：

- `Event` 只是**广播某个状态已经发生**
- 优雅停机还包括：
  - 停止接收新请求
  - 等待旧请求收尾
  - 取消后台任务
  - flush 日志和指标

所以：

> `Event` 是生命周期控制面的一个基础原语，而不是完整生命周期管理本身。

### 一个 startup / shutdown 协调示例

```python
import asyncio


async def service_worker(name: str,
                         ready_event: asyncio.Event,
                         shutdown_event: asyncio.Event):
    await ready_event.wait()
    print(f"{name} started")

    while not shutdown_event.is_set():
        await asyncio.sleep(0.2)

    print(f"{name} stopping")


async def main():
    ready_event = asyncio.Event()
    shutdown_event = asyncio.Event()

    tasks = [
        asyncio.create_task(service_worker("w1", ready_event, shutdown_event)),
        asyncio.create_task(service_worker("w2", ready_event, shutdown_event)),
    ]

    await asyncio.sleep(0.3)
    ready_event.set()      # 广播“系统已启动”
    await asyncio.sleep(0.6)
    shutdown_event.set()   # 广播“准备停机”

    await asyncio.gather(*tasks)


asyncio.run(main())
```

这个例子把 `Event` 从“API 名字”提升成了“服务生命周期状态机”的一部分。

---

## 26.6 优雅关闭：不是 `Ctrl+C` 就结束了

一个真实异步服务在关闭时需要做的事往往包括：

- 停止接收新请求
- 等待队列中已有任务处理完
- 取消后台任务
- 刷新日志 / 指标
- 关闭网络连接

如果你只是让进程退出，常见后果是：

- 队列里任务丢失
- 半写入响应
- 文件句柄和连接没释放

### 一个简化的优雅关闭骨架

```python
import asyncio
import signal


shutdown_event = asyncio.Event()


def install_signal_handlers():
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_event.set)


async def background_worker():
    try:
        while not shutdown_event.is_set():
            await asyncio.sleep(0.5)
    except asyncio.CancelledError:
        # cleanup here
        raise


async def main():
    install_signal_handlers()

    worker = asyncio.create_task(background_worker())
    await shutdown_event.wait()

    worker.cancel()
    await asyncio.gather(worker, return_exceptions=True)


asyncio.run(main())
```

这个例子虽然简化，但已经体现了生产级关闭顺序：

1. 收到 signal
2. 通过 `Event` 广播 shutdown intent
3. 停止新工作
4. 取消后台任务
5. 等待清理完成

### `loop.add_signal_handler()` 为什么值得讲

它很能体现 event loop 的本质：  
操作系统信号原本是进程级事件，而 `loop.add_signal_handler()` 把它桥接成了异步系统内部的事件。

也就是说，event loop 实际上统一管理了：

- I/O 事件
- timer 事件
- callback 事件
- signal 事件

这正是“event loop”这个名字的真正含义：  
**它不是只管协程，而是管整个异步程序的事件源。**

---

## 26.7 一个近似生产级的异步服务骨架

下面这个示例把前面所有内容串起来：

- TCP 服务入口
- 有界队列
- worker pool
- 限流
- shutdown event
- signal 桥接

```python
import asyncio
import json
import signal


shutdown_event = asyncio.Event()


async def worker(name: str, queue: asyncio.Queue, sem: asyncio.Semaphore):
    try:
        while not shutdown_event.is_set():
            item = await queue.get()
            try:
                async with sem:
                    await asyncio.sleep(0.1)
                    print(f"{name} handled {item}")
            finally:
                queue.task_done()
    except asyncio.CancelledError:
        raise


async def handle_client(reader: asyncio.StreamReader,
                        writer: asyncio.StreamWriter,
                        queue: asyncio.Queue):
    try:
        while not shutdown_event.is_set():
            data = await reader.readline()
            if not data:
                break
            msg = json.loads(data.decode())
            await queue.put(msg)
            writer.write(b"{\"status\":\"accepted\"}\n")
            await writer.drain()
    finally:
        writer.close()
        await writer.wait_closed()


async def main():
    queue = asyncio.Queue(maxsize=100)
    sem = asyncio.Semaphore(10)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_event.set)

    workers = [asyncio.create_task(worker(f"w{i}", queue, sem)) for i in range(3)]

    server = await asyncio.start_server(
        lambda r, w: handle_client(r, w, queue),
        "127.0.0.1",
        9999,
    )

    async with server:
        await shutdown_event.wait()

    for w in workers:
        w.cancel()
    await asyncio.gather(*workers, return_exceptions=True)


asyncio.run(main())
```

### 这个服务骨架解决了哪些现实问题

1. 不让入口无限吃请求：有界队列
2. 不让下游无限并发：Semaphore
3. 不让退出变成硬切：Event + signal + cancel
4. 不让慢客户端无限写爆内存：`drain()`

---

## 26.8 常见故障模式

### 模式一：服务没崩，但越来越慢

常见根因：

- 队列持续增长
- 背压缺失
- 某些 worker 被阻塞同步代码卡住

### 模式二：停机时请求永远不返回

常见根因：

- Future 没有被设异常
- shutdown 只停 worker，没停入口
- 背景任务没有统一等待

### 模式三：慢客户端把服务拖垮

常见根因：

- 写操作后没 `drain()`
- 没有为连接设置超时和限制
- 发送缓冲不断增长

### 排障建议

```text
先看 queue size
再看 active tasks / cancellation
再看 drain / slow client
最后看下游依赖和 signal/shutdown 路径
```

---

## 本章小结

| 机制 | 作用 |
|------|------|
| `drain()` | 把慢客户端压力反馈给上游协程 |
| `Queue(maxsize=N)` | 为入口积压设置上限 |
| `Semaphore` | 限制同时访问下游的并发数 |
| `Event` | 广播服务生命周期状态 |
| `add_signal_handler()` | 把 OS 信号接入 loop 控制面 |
| 优雅关闭 | 让系统退出前有明确收尾流程 |

---

## 深度学习应用

本章在模型服务和 AI Infra 里最常见的落点包括：

- 异步模型网关
- 日志 / 事件摄取
- 流式推理入口
- 推理前异步预处理
- 模型服务优雅停机

当你在真实服务里看到：

- `shutdown_event`
- `loop.add_signal_handler(...)`
- `drain()`
- 有界队列

它们都不是“样板代码”，而是在防止系统进入不可控状态。

---

## 练习题

1. 为什么 `drain()` 是异步服务里极其重要的一行代码？
2. `asyncio.Event` 为什么很适合做 shutdown signal，却不能单独等价于“优雅停机”？
3. 为什么“队列存在”不等于“系统就安全”，还必须考虑队列大小和降级策略？
4. `Semaphore` 和 `Queue(maxsize=N)` 分别更适合控制什么？
5. 设计一个支持优雅关闭的异步服务关闭流程，并标出 signal、Event、worker 之间的关系。
