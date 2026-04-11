# 第25章：asyncio基础与结构化并发

> `asyncio` 不是“更快的线程”，而是一套围绕**事件循环（event loop）**、**协程（coroutine）**、**任务（Task）**、**Future**、**取消传播**和**异步协调原语**建立起来的并发运行时。真正掌握它，关键不在于会写 `async def`，而在于你能解释：任务什么时候被调度、什么时候挂起、什么时候恢复、什么时候取消，以及这些行为到底由谁控制。

> **版本提示**：本章大多数代码在 Python 3.11+ 体验最佳，因为 `TaskGroup`、`asyncio.timeout()` 等现代接口更完整。如果你使用 Python 3.9/3.10，大部分思想仍然成立，但部分 API 需替换成旧写法。

---

## 学习目标

完成本章学习后，你将能够：

1. 理解协程、Task、Future 与事件循环之间的完整关系
2. 解释 event loop 一次 tick 内部大致会做什么
3. 理解 `selector / reactor` 视角下的 `asyncio` 事件分发模型，以及 `epoll / kqueue / IOCP` 的平台差异
4. 区分 `await`、`create_task()`、`call_soon()`、`call_later()`、`TaskGroup` 各自解决的问题
5. 理解 `uvloop` 与标准 `asyncio` loop 的差异，并深入掌握 `asyncio.Event` 的语义、适用场景与常见误用

---

## 正文内容

## 25.1 为什么需要 asyncio

同步程序的默认思维模型是“一个函数调用把当前线程占住，直到它执行结束”。  
这在 CPU 密集任务中没有问题，但在 I/O 密集任务中会造成明显浪费。

例如，同步网络请求：

```python
data = socket.recv(4096)   # 线程阻塞
process(data)
```

如果网络暂时没有数据，这个线程就会停在这里，什么也做不了。  
而在 `asyncio` 里，等待 I/O 的代码会写成：

```python
data = await reader.read(4096)   # 当前协程挂起，线程继续服务别的协程
process(data)
```

这说明 `asyncio` 的核心价值是：

> 当一个任务在等待时，把执行权交回事件循环，让同一线程继续推进其他任务。

因此，`asyncio` 最适合：

- 高并发 HTTP / TCP 请求
- 异步 API 聚合
- WebSocket / SSE 流式服务
- I/O 主导的数据处理流水线
- 模型推理前的特征并发抓取与编排

但它**不适合直接解决**：

- 大量 CPU 密集型纯 Python 计算
- 真正需要多核并行的数值处理
- 长时间阻塞且无法异步化的第三方库调用

一个最重要的判断表：

| 任务类型 | 更适合什么 |
|---------|-----------|
| I/O 密集型 | `asyncio` |
| CPU 密集型 | 线程池 / 进程池 / 原生扩展 / 向量化库 |

---

## 25.2 事件循环到底在做什么

如果你只记住一句话，那就是：

> 事件循环负责管理“现在谁能继续执行”，并决定把执行机会交给谁。

在 `asyncio` 中，协程遇到 `await` 时会主动挂起自己，把控制权交回给事件循环。  
事件循环随后会去处理两类事情：

1. 哪些 I/O 已经就绪
2. 哪些定时器已经到期

然后把对应任务恢复执行。

### 事件循环的一次 tick 可以近似看成什么

虽然不同 Python 实现和操作系统底层细节不同，但你可以用下面这个近似模型理解 event loop：

```text
while loop is running:
    1. 收集已经到期的 timer callbacks
    2. 轮询内核，查看哪些 fd / socket 已经就绪
    3. 把就绪任务和回调放入 ready queue
    4. 依次执行 ready queue 中的回调 / 任务
    5. 遇到新的 await 时，把任务重新挂回 future/timer/io watcher
```

这个模型会直接解释很多现象：

- 为什么 `time.sleep()` 会把整个异步系统卡死
- 为什么大量短回调也会拖慢 loop
- 为什么超时、定时器、Future 回填本质上都和 loop 调度有关

### 事件循环里最关键的三种“等待”

| 类型 | loop 眼中的本质 | 典型来源 |
|------|----------------|----------|
| I/O 等待 | 某个 fd 未来会就绪 | socket、pipe、流式读写 |
| 时间等待 | 某个时刻再恢复 | `sleep()`、timeout、延迟回调 |
| 结果等待 | 某个 Future 未来会被填值 | Task、底层回调、批处理器 |

换句话说，loop 并不关心“业务语义”，它只关心：

- 谁现在 ready
- 谁稍后才 ready
- 谁在等别人把结果填回来

### 一个时间线示意

```text
t=0.000 create_task(A), create_task(B)
t=0.001 loop 运行 A，A await socket.read()
t=0.002 loop 运行 B，B await sleep(0.5)
t=0.003 loop 开始等待 I/O / timer
t=0.120 socket 可读，A 进入 ready queue
t=0.121 loop 恢复 A
t=0.500 sleep 到期，B 进入 ready queue
t=0.501 loop 恢复 B
```

当你开始用这种时间线来看系统时，`asyncio` 的行为就不再只是“魔法”。

### 从 selector / reactor 视角再看一遍

很多更底层的异步框架会用两个词来描述这一类系统：

- **selector**
- **reactor**

它们不是 `asyncio` 独有术语，但用它们理解 `asyncio` 非常有效。

#### selector 是什么

selector 的核心任务是：

> 询问操作系统：哪些文件描述符（fd）现在已经可读 / 可写？

在 Linux 上，你可以把它联想到：

- `select`
- `poll`
- `epoll`

在 macOS / BSD 上，可以联想到：

- `kqueue`

在 Windows 上，则会走不同的事件机制。

对 Python 用户来说，最重要的不是记内核 API 名字，而是理解：

> 事件循环本身并不会“主动去读 socket”，而是先通过 selector 询问“现在谁 ready 了”。  
> 只有 ready 了，loop 才会把相关回调或任务放回 ready queue。

#### `epoll`、`kqueue`、`IOCP` 分别是什么

当我们说“selector”时，底层其实并不是一个统一接口，而是不同操作系统上的不同事件通知机制。

##### `epoll`（Linux）

- 是 Linux 上非常常见的高性能 I/O 事件通知机制
- 适合大量 fd / socket 的可读可写事件监听
- 典型特点是：更适合高并发网络服务器，而不是每次都线性扫描全部 fd

##### `kqueue`（macOS / BSD）

- 是 BSD 系列和 macOS 常见的事件通知机制
- 不只支持 socket，也支持更多类型的内核事件
- 在概念上和 `epoll` 很像，都是“把感兴趣的事件注册给内核，等内核告诉你谁 ready 了”

##### `IOCP`（Windows）

- 全称 *I/O Completion Ports*
- 与 `epoll` / `kqueue` 的使用风格不完全相同
- 更接近“完成通知”模型，而不只是“fd 现在 ready 了”

工程上你可以先抓住一个简单结论：

| 平台 | asyncio 常见底层思路 |
|------|----------------------|
| Linux | selector / reactor + `epoll` |
| macOS / BSD | selector / reactor + `kqueue` |
| Windows | 更偏 proactor 风格，底层依赖 `IOCP` |

这也是为什么跨平台异步程序有时会在：

- 信号支持
- 文件描述符语义
- 子进程与 socket 行为
- 性能特征

上表现不同。

#### selector 和 proactor 有什么区别

如果再抽象一层，可以这么理解：

- **selector / reactor**：关注“谁现在 ready 了”
- **proactor / completion model**：关注“哪个异步操作已经完成了”

Linux / BSD 上常见的是前一种思路；Windows 上更典型的是后一种思路。  
这也是为什么 `asyncio` 在 Windows 上的底层 loop 实现和 Unix 风格 loop 存在明显差异。

#### 为什么 Python 开发者也值得理解这些差别

即使你不直接调用内核 API，这些差异也会影响：

- 某些 API 是否跨平台一致
- 为什么某些第三方库只在 Unix 上更成熟
- 为什么 Windows 上的信号、pipe、subprocess 行为常常和 Unix 不完全一致
- 为什么 benchmark 在不同平台上会差很多

#### reactor 是什么

reactor 模式可以粗略理解为：

1. 注册感兴趣的事件
2. 等待事件发生
3. 事件发生后调用对应 handler / callback

把它套进 `asyncio`：

```text
socket.read() -> 注册“我关心这个 fd 的可读事件”
loop poll -> 内核说“它可读了”
loop dispatch -> 恢复等待该事件的 Task / callback
```

这就是为什么说：

> `asyncio` 的事件循环本质上是一个 reactor。

#### coroutine 风格和 reactor 之间是什么关系

很多初学者会以为：

- callback 风格是一套系统
- coroutine 风格是另一套系统

其实更准确地说：

- **reactor / selector** 是底层事件分发模型
- **coroutine / await** 是让你不用手写 callback 链的一层更高级语法接口

也就是说，`await reader.read()` 并没有消灭 callback 或事件分发；它只是把这些复杂性封装到了运行时里。

#### 一个 mental model

可以把 `asyncio` 看成两层：

```text
高层：coroutine / Task / await / TaskGroup
底层：reactor / selector / callback / future resolution
```

高层负责让代码可读，底层负责让事件真的被调度起来。

### 再往下一层：为什么 loop 实现也会影响性能

到这里你已经知道：

- 高层是 coroutine / Task / await
- 底层是 selector / reactor / callback / Future resolution

但实际运行时还有一个经常被忽视的问题：

> 即使高层代码完全一样，不同 loop 实现的调度成本、事件分发成本和实现语言也会不同。

这正是 `uvloop` 会登场的地方。

---

## 25.3 协程、Task、Future 的关系

这三个对象是 `asyncio` 最核心也最容易混淆的概念。

### 25.3.1 协程对象

协程函数由 `async def` 定义：

```python
async def fetch_user(user_id: int) -> dict:
    await asyncio.sleep(0.1)
    return {"user_id": user_id}
```

调用它时，不会立刻执行，而是得到一个协程对象：

```python
coro = fetch_user(42)
print(coro)
# <coroutine object fetch_user at 0x...>
```

协程对象可以理解成：

> 一段“尚未真正跑完”的异步逻辑描述。

### 25.3.2 Task

`Task` 是被事件循环调度执行的协程包装器。

```python
task = asyncio.create_task(fetch_user(42))
```

一旦变成 `Task`，它就进入了事件循环的管理范围。

可以这样理解：

- 协程对象：待执行的逻辑
- Task：事件循环中的运行单元

### 25.3.3 Future

`Future` 是一个“稍后会被填入结果”的占位符。  
很多时候你不直接写它，但复杂系统里它非常重要，因为它能把“接收请求”和“稍后回填结果”解耦。

示例：

```python
import asyncio


async def complete_later(fut: asyncio.Future):
    await asyncio.sleep(0.2)
    fut.set_result("done")


async def main():
    loop = asyncio.get_running_loop()
    fut = loop.create_future()

    asyncio.create_task(complete_later(fut))
    result = await fut
    print(result)


asyncio.run(main())
```

### 一句话对比

| 对象 | 本质 |
|------|------|
| 协程对象 | “一段未来要执行的异步逻辑” |
| Task | “事件循环正在管理的协程执行实例” |
| Future | “未来才会有结果的占位承诺” |

### 为什么 Future 在复杂系统里很关键

如果你要写：

- 动态批处理器
- 异步 RPC 聚合器
- 推理请求到后台 worker 的桥接层

你几乎一定会用到 `Future`，因为调用方需要一个“先返回的句柄”，后台系统稍后再把结果写回去。

---

## 25.4 loop 生命周期与低层 API

很多教程只教你：

```python
asyncio.run(main())
```

这没错，但如果你只停留在这一层，就会看不懂很多高级代码。

### `asyncio.run()` 做了什么

它可以近似理解为：

1. 创建一个新的事件循环
2. 把 `main()` 放进去执行
3. 等待所有必要收尾完成
4. 关闭 loop

所以它不是一个普通 helper，而是异步程序的**生命周期边界**。

### `get_running_loop()` vs `get_event_loop()`

#### `asyncio.get_running_loop()`

返回当前**正在执行当前协程 / 回调**的那个 loop。  
这是现代 `asyncio` 代码里更推荐的 API。

```python
async def main():
    loop = asyncio.get_running_loop()
    print(loop)
```

#### `asyncio.get_event_loop()`

这是一个历史更久、语义更复杂的 API。  
在不同上下文和版本里的行为不完全一致，不适合作为现代教程里的默认推荐。

工程上，你可以先记住：

> 协程内部优先用 `get_running_loop()`。

### `loop.create_future()`

如果你在写底层异步基础设施，而不是只消费高层 API，常常会看到：

```python
loop = asyncio.get_running_loop()
future = loop.create_future()
```

这表示创建一个和当前 loop 生命周期绑定的 Future。

### `loop.call_soon()` 与 `loop.call_later()`

这两个 API 很能帮助你理解“event loop 本质上也是 callback scheduler”。

#### `call_soon`

```python
loop.call_soon(callback, *args)
```

意思是：把回调尽快放入 ready queue。

#### `call_later`

```python
loop.call_later(delay, callback, *args)
```

意思是：未来某个时间点再把回调放进调度队列。

最小示例：

```python
import asyncio


def say(msg: str):
    print(msg)


async def main():
    loop = asyncio.get_running_loop()
    loop.call_soon(say, "soon callback")
    loop.call_later(0.2, say, "later callback")
    await asyncio.sleep(0.3)


asyncio.run(main())
```

这些 API 的工程价值在于：

- 帮你理解定时器是怎么进入 loop 的
- 帮你理解 Future / Task 恢复本质上也是某种“回调重新入队”

### callback 风格和 coroutine 风格的关系

可以这样理解：

- callback 风格：你手工组织“事件来了之后要调用谁”
- coroutine 风格：你把“暂停 / 恢复”的控制权交给运行时，代码看起来像顺序写法

例如，callback 风格常见于：

```text
当 socket 可读 -> 调 callback_A
当 callback_A 完成 -> 调 callback_B
当 timer 到期 -> 调 callback_C
```

而 coroutine 风格看起来像：

```python
data = await reader.read(...)
result = await handle(data)
await writer.drain()
```

但本质上，底层仍然需要：

- 等待事件
- 安排恢复点
- 回填 Future
- 调度下一步

所以 coroutine 风格不是“替换掉 reactor”，而是“把 reactor 封装成更接近顺序程序的写法”。

### `run_in_executor()` 与 `to_thread()`

当你必须调用阻塞同步函数时，有两种常用办法：

- `loop.run_in_executor()`
- `asyncio.to_thread()`

它们都在解决同一个问题：

> 不要让阻塞同步代码卡死事件循环线程。

### `uvloop` 和标准 loop 的差异

`uvloop` 可以粗略理解为：

> 一个兼容 `asyncio` API 的、更高性能的事件循环实现。

从使用者视角看，很多高层代码几乎不用改，只需要把 loop policy 切换过去。

#### 标准 loop 的特点

- 来自 Python 标准库
- 默认可用、跨平台语义更稳定
- 是学习 `asyncio` 的基准参考实现

#### `uvloop` 的特点

- 基于更底层、更高性能的事件循环实现思路
- 在 Unix 类平台的网络 I/O 场景里常常更快
- 通常更适合高并发网络服务、网关、代理、抓取器

#### 一个重要边界

`uvloop` 不会神奇地解决所有性能问题。  
它更容易改善的是：

- 事件循环本身的调度和 I/O 分发开销

它**不会**直接解决：

- 你的业务逻辑阻塞
- 你的数据库慢
- 你的 HTTP 客户端设计不合理
- 你的系统没有背压和限流

因此，更准确的理解是：

| 方案 | 更可能改善什么 |
|------|----------------|
| 标准 loop | 稳定、默认、易理解 |
| `uvloop` | 高并发网络 I/O 的 loop 开销 |

#### 什么时候值得考虑 `uvloop`

当你满足以下条件时更值得：

- 服务是典型网络 I/O 主导
- 高并发连接很多
- 你已经排除了明显业务瓶颈
- 平台是 Unix 风格环境

#### 什么时候不该把 `uvloop` 当银弹

- 程序主要耗时在 CPU 计算
- 问题根本在下游数据库 / 模型推理
- 你还没做好 timeout / 背压 / 限流

一个工程化判断是：

> 先把异步系统的控制流和资源流设计正确，再去追求 loop 实现差异带来的性能收益。

---

## 25.5 `await`、`create_task()`、`gather()`、`TaskGroup` 分别解决什么

### `await`

表示当前协程等待另一个 awaitable 的结果，并在等待期间把执行权交给 loop。

```python
user = await fetch_user(42)
```

### `create_task()`

表示把协程注册成 Task，让它并发运行，而不是立刻顺序等待。

```python
task = asyncio.create_task(fetch_user(42))
profile = await fetch_profile(42)
user = await task
```

### `gather()`

表示我有一组任务，希望并发跑并一起等它们完成。

```python
results = await asyncio.gather(a(), b(), c())
```

### `TaskGroup`

表示我有一组具备共同生命周期的并发任务，希望统一管理异常和取消。

```python
async with asyncio.TaskGroup() as tg:
    t1 = tg.create_task(a())
    t2 = tg.create_task(b())
```

它比“到处裸 `create_task()`”更适合复杂系统，因为它更强调结构化并发。

---

## 25.6 `asyncio.Event`：不是样例配角，而是核心协调原语

这正是你指出当前教程不足的地方之一。  
在很多异步系统中，`Event` 不是一个小工具，而是非常重要的**状态广播原语**。

### `asyncio.Event` 是什么

它表达的是：

> 某个条件现在是否已经成立。

常见接口：

- `event.set()`：把状态设为已触发，并唤醒所有等待者
- `event.clear()`：把状态重置为未触发
- `await event.wait()`：等待状态变为已触发

最小示例：

```python
import asyncio


async def waiter(name: str, ready: asyncio.Event):
    print(f"{name} waiting")
    await ready.wait()
    print(f"{name} resumed")


async def main():
    ready = asyncio.Event()

    t1 = asyncio.create_task(waiter("A", ready))
    t2 = asyncio.create_task(waiter("B", ready))

    await asyncio.sleep(0.2)
    ready.set()

    await asyncio.gather(t1, t2)


asyncio.run(main())
```

### `Event` 最适合的场景

1. **系统启动屏障**  
   某些 worker 必须等“模型加载完成”后才能继续。

2. **服务停机广播**  
   所有后台任务都监听同一个 `shutdown_event`。

3. **配置刷新通知**  
   一批协程等待“新配置已就绪”这个状态。

### `Event` 与 `Queue`、`Lock`、`Semaphore` 的区别

| 原语 | 更适合表达什么 |
|------|----------------|
| `Event` | 某个条件是否已成立 |
| `Queue` | 数据项异步流动 |
| `Lock` | 同一时刻只能一个协程修改共享状态 |
| `Semaphore` | 最多允许 N 个协程同时进入临界区 |

一个简单判断：

- 你需要传递**数据**：用 `Queue`
- 你需要表达**状态变化**：用 `Event`
- 你需要做**互斥**：用 `Lock`
- 你需要做**并发上限**：用 `Semaphore`

### `Event` 为什么常见于 shutdown 逻辑

因为 shutdown 具有明显的一对多广播特征：

- 网关要知道
- worker 要知道
- batcher 要知道
- 指标上报协程也要知道

共享一个：

```python
shutdown_event = asyncio.Event()
```

然后各个协程都监听它，是非常自然的设计。

### 一个服务启动屏障示例

```python
import asyncio


async def model_loader(ready: asyncio.Event):
    print("loading model...")
    await asyncio.sleep(1.0)
    ready.set()
    print("model ready")


async def request_worker(name: str, ready: asyncio.Event):
    await ready.wait()
    print(f"{name} begins serving")


async def main():
    ready = asyncio.Event()

    async with asyncio.TaskGroup() as tg:
        tg.create_task(model_loader(ready))
        for i in range(3):
            tg.create_task(request_worker(f"worker-{i}", ready))


asyncio.run(main())
```

这段代码比单纯讲 `Event` API 更重要的地方在于：

> 它说明 `Event` 可以把“系统状态”从一个局部变量提升成多任务共享的协调机制。

---

## 25.7 取消传播与结构化并发

很多初学者的异步 demo 都缺少一个关键视角：  
**取消不是异常边角料，而是复杂系统的正常控制流。**

常见触发取消的原因：

- 上游请求超时
- 用户主动断开连接
- 服务准备停机
- 某个父任务失败，需要统一取消子任务

### 一个取消示例

```python
import asyncio


async def worker(name: str):
    try:
        while True:
            print(f"{name}: working")
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        print(f"{name}: cleanup before exit")
        raise


async def main():
    task = asyncio.create_task(worker("A"))
    await asyncio.sleep(2.5)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        print("task cancelled")


asyncio.run(main())
```

### 关键原则

1. `CancelledError` 通常不应该被吞掉
2. 清理逻辑要么放在 `except CancelledError`，要么放在 `finally`
3. 如果你把取消吞掉，系统会出现幽灵任务和状态不一致

### `TaskGroup` 为什么更适合复杂系统

因为它能把“父任务 -> 子任务”的生命周期绑定起来。  
如果一组任务属于同一个业务动作，那么它们就不应该在失败时各自散落。

---

## 25.8 实战：带启动屏障、限流、超时和取消的异步聚合器

下面这个例子把 loop、Task、Future、Event 组合起来：

```python
import asyncio
import random
from collections import defaultdict


class FeatureAggregator:
    def __init__(self, concurrency: int = 3, timeout_s: float = 1.0):
        self._ready = asyncio.Event()
        self._sem = asyncio.Semaphore(concurrency)
        self._timeout_s = timeout_s
        self._stats = defaultdict(int)

    async def warmup(self):
        await asyncio.sleep(0.3)
        self._ready.set()

    async def _fetch_one(self, name: str) -> str:
        async with self._sem:
            delay = random.uniform(0.1, 1.2)
            await asyncio.sleep(delay)
            if random.random() < 0.15:
                raise RuntimeError(f"{name} failed")
            return f"{name}:{delay:.2f}s"

    async def aggregate(self, names: list[str]) -> dict:
        await self._ready.wait()

        results = {}
        async with asyncio.TaskGroup() as tg:
            tasks = {name: tg.create_task(self._guarded_fetch(name)) for name in names}

        for name, task in tasks.items():
            results[name] = task.result()
        return results

    async def _guarded_fetch(self, name: str) -> str:
        try:
            async with asyncio.timeout(self._timeout_s):
                result = await self._fetch_one(name)
                self._stats["ok"] += 1
                return result
        except Exception as exc:
            self._stats["fail"] += 1
            return f"ERROR:{exc}"


async def main():
    agg = FeatureAggregator()

    async with asyncio.TaskGroup() as tg:
        tg.create_task(agg.warmup())
        task = tg.create_task(agg.aggregate(["profile", "risk", "recommendation"]))

    print(task.result())


asyncio.run(main())
```

### 这个例子里的关键对象

- `Event`：表达“服务是否已准备好”
- `Semaphore`：限制下游并发
- `TaskGroup`：让并发抓取有统一生命周期
- `timeout`：防止单个上游无限挂起

### 深度学习应用

这非常像模型前处理阶段：

```text
用户请求
  -> 等待模型 / 缓存 ready
  -> 并发抓 profile / risk / retrieval
  -> 聚合上下文
  -> 送入模型
```

很多“异步推理服务”的前半段，本质就是这个模式。

---

## 25.9 常见错误与调试建议

### 错误一：忘记 `await`

```python
result = fetch_user(42)   # 这是协程对象，不是结果
```

### 错误二：到处 `create_task()` 但没人回收

这会带来：

- 任务泄漏
- 异常无人处理
- 停机时一堆后台任务还活着

### 错误三：在协程里调用阻塞同步函数

例如：

- `time.sleep()`
- 阻塞数据库驱动
- 大量 CPU 计算

### 错误四：把 `Event` 当作消息队列

`Event` 只表达“状态”，不表达“多条数据流”。  
如果你想传多个对象，请用 `Queue`。

### 调试建议

```python
import asyncio

loop = asyncio.get_running_loop()
loop.set_debug(True)
```

并配合：

- `asyncio.all_tasks()`
- `task.get_name()`
- loop 时间线打点
- timeout / cancel 日志

一个更高级的排障问题是：

> 现在是 loop 没在调度，还是任务都在等某个永远不会到来的事件？

很多异步系统“没挂但不动了”，根因就在这里。

---

## 本章小结

| 概念 | 作用 |
|------|------|
| 事件循环 | 调度“谁现在能继续执行” |
| selector | 告诉 loop 哪些 fd / socket 已 ready |
| reactor | 统一管理事件注册、等待和分发的模式 |
| `epoll / kqueue / IOCP` | 不同平台上用于事件通知或完成通知的底层机制 |
| 协程对象 | 一段尚未完成的异步逻辑 |
| Task | 被 loop 管理的执行单元 |
| Future | 未来结果的占位承诺 |
| `Event` | 广播“某个条件已经成立” |
| `call_soon/call_later` | 把回调放入当前或未来调度队列 |
| `uvloop` | 面向高并发 I/O 的高性能 loop 实现 |

---

## 深度学习应用

在模型系统中，本章最常见的落点是：

- 模型启动就绪信号
- 异步特征聚合
- 模型前处理管道
- 批量推理控制面

真正理解 loop 和 `Event` 后，你会更容易读懂：

- 模型网关的启动逻辑
- 后台 batch worker
- 流式推理前的状态协调

---

## 练习题

1. 为什么 `asyncio` 更适合 I/O 密集型任务，而不适合直接解决 CPU 密集型任务？
2. 解释 event loop 一次 tick 内部大致在做什么，并说明 ready queue、timer、I/O watcher 的关系。
3. `selector` / `reactor` 视角下，`asyncio` 的 loop 到底扮演什么角色？`epoll / kqueue / IOCP` 各自又处在哪一层？
4. `uvloop` 和标准 loop 的差异主要体现在哪？为什么它不是“所有异步程序的银弹”？
5. 设计一个“并发抓取 4 个特征服务并聚合结果”的最小异步架构，并说明其中 loop、Task、Event 各自扮演什么角色。
