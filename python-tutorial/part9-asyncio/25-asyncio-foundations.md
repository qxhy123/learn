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

当你开始用这种时间线来看系统时，`asyncio` 的行为就不再只是”魔法”。

### 事件循环如何驱动协程：`send()` 机制

到目前为止，你知道”协程遇到 `await` 就挂起，事件循环恢复它”。  
但这句话跳过了最关键的问题：**loop 是怎么”恢复”一个协程的？**

答案是：通过 `coroutine.send(value)` 这个低层协议。

#### 协程在底层是增强版生成器

Python 中，`async def` 协程在底层与生成器（`yield`）使用同一套协议。  
一个协程对象有 `send()` 方法：

- `coro.send(None)` — 启动或推进协程，直到下一个挂起点（`await`）
- `coro.send(value)` — 向协程注入一个值，从上次暂停的地方继续执行

你可以把 `await expr` 在底层近似理解为：

```python
# 伪代码示意，非真实实现
result = yield expr   # 把 expr（通常是一个 Future）交给上层，暂停，等待结果注回来
```

协程把 Future 交给上层（loop），然后暂停。loop 收到这个 future，等它就绪后，再 `send(result)` 把结果注回来。

#### loop 的调度内核：`Task.__step()`

`Task` 对象内部有一个 `__step()` 方法（CPython 内部实现），可以近似看成：

```python
def __step(self, exc=None):
    try:
        if exc is None:
            # 推进协程，直到它 yield 出一个 Future（即遇到 await）
            result = self._coro.send(None)
        else:
            # 向协程注入异常（用于取消传播）
            result = self._coro.throw(type(exc), exc)
    except StopIteration as e:
        # 协程执行到 return，正常结束
        self.set_result(e.value)
    except Exception as e:
        # 协程内部抛出了未捕获异常
        self.set_exception(e)
    else:
        # result 是协程 yield 出来的 Future
        # 在这个 Future 完成时，再次调用 __step
        result.add_done_callback(self.__wakeup)

def __wakeup(self, future):
    self.__step()
```

把这个过程展开成时间线：

```text
loop 把 task.__step 放入 ready queue
    │
    ▼
task.__step() 调用 coro.send(None)
    │  协程运行，直到遇到 await some_io_future
    │  协程 yield 出 some_io_future 给 Task
    ▼
Task 在 some_io_future.add_done_callback(self.__wakeup)
控制权返回 loop

（loop 继续处理其他就绪任务 / 等待 I/O 事件）

I/O 完成 → some_io_future.set_result(data)
    │  触发 done_callbacks
    ▼
task.__wakeup() 调用 task.__step()
    │  coro.send(data) 把结果注回协程
    ▼
协程从 await 处恢复，拿到 data 继续执行
```

#### 这个机制解释了三个关键现象

**现象一：`time.sleep()` 为什么卡死整个系统**

`time.sleep()` 是同步阻塞，不会 `yield` 出任何 Future。  
整个 OS 线程卡在那里，loop 的 `send()` 调用无法返回，无法推进任何其他任务。

**现象二：两次 `await` 之间的代码为什么是”原子”的**

`send()` 会一直推进协程，直到下一个 `yield`（await 点）才返回控制权给 loop。  
在这段时间里，loop 完全无法插入其他任务。这就是为什么：

- 在两次 `await` 之间做大量计算会”饿死”其他任务
- `await asyncio.sleep(0)` 能主动让出，就是因为它立刻 yield 出一个已就绪的 Future，让 loop 可以先处理其他事

**现象三：单线程如何”并发”**

多个 Task 共享同一个 loop 线程，但每个 Task 都有自己独立的协程对象和执行位置。  
Loop 不断地依次 `send()` 每个就绪的 Task，形成”交错执行”的效果。

> 这是**协作式并发（cooperative concurrency）**，不是多线程并行。  
> 关键词：**单线程、主动让出、交错推进**。

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

### 25.3.3.1 为什么协程是串行的，Task 是并发的

这是学习 `asyncio` 最重要的认知跳跃点。  
一句话版本：

> `await coro()` 是”我等你完成再继续”；`create_task(coro())` 是”你去跑，我同时继续”。

#### 用计时实验看清楚

**顺序 `await`（串行）：**

```python
import asyncio
import time


async def slow(name: str, delay: float) -> str:
    await asyncio.sleep(delay)
    return name


async def main_sequential():
    t0 = time.perf_counter()

    r1 = await slow(“A”, 1.0)   # 当前协程暂停，等 A 跑完（1 秒）
    r2 = await slow(“B”, 1.0)   # A 完成后才开始 B（又 1 秒）

    print(f”results: {r1}, {r2}”)
    print(f”elapsed: {time.perf_counter() - t0:.2f}s”)   # ≈ 2.0 秒


asyncio.run(main_sequential())
```

输出：
```
results: A, B
elapsed: 2.01s
```

**并发 `create_task`：**

```python
async def main_concurrent():
    t0 = time.perf_counter()

    task_a = asyncio.create_task(slow(“A”, 1.0))   # 立即注册到 loop，不等待
    task_b = asyncio.create_task(slow(“B”, 1.0))   # 立即注册到 loop，不等待

    r1 = await task_a   # 等 A 的结果
    r2 = await task_b   # 等 B 的结果（此时 B 可能已经在跑或完成了）

    print(f”results: {r1}, {r2}”)
    print(f”elapsed: {time.perf_counter() - t0:.2f}s”)   # ≈ 1.0 秒


asyncio.run(main_concurrent())
```

输出：
```
results: A, B
elapsed: 1.01s
```

A 和 B 的等待时间**重叠**，总时间只有 ≈ 1 秒。

#### 为什么 `await` 让协程串行

`await` 对当前协程的含义是：

> “我现在暂停，等这个 awaitable 完成，拿到结果后再执行下面的代码。”

虽然暂停时不阻塞线程（loop 可以推进其他已有 Task），但对**当前协程自身**来说，执行是完全有序的：  
`await A` 没完成，`await B` 绝不会开始。

```text
main_sequential 协程的时间线：

时间 0────────────────1s────────────────2s
     │                 │                 │
     └──await A──────►┤└──await B──────►┤
        （A 在跑）     A 完成            B 完成
                       ↑ 才轮到 B 开始
```

#### 为什么 `create_task` 能并发

`create_task()` 做的事情是：

> “把这个协程变成一个**独立的 Task**，交给 loop 管理。从现在起，它和我是两个平行的调度单元。”

此后，两个 Task 在 loop 眼里是**完全独立的调度对象**：  
Task A 在 `await` 处让出时，loop 可以推进 Task B，反之亦然。

```text
main_concurrent 协程和两个 Task 的时间线：

时间 0────────────────────────────1s
     │                             │
     main: create_task(A), create_task(B), await task_a ...
     Task A:     ───await sleep(1.0)──────────►完成
     Task B:     ───await sleep(1.0)──────────►完成
     loop 线程:  A.__step → B.__step → 等待 I/O → A.__step → B.__step
```

A 和 B 的 `sleep` 在时间上完全重叠，所以总耗时只有 ≈ 1 秒。

#### 核心区别总结

| 方式 | 对当前协程的效果 | 时间叠加方式 | 适合场景 |
|------|----------------|-------------|----------|
| `await coro()` | 暂停，等结果 | A时间 + B时间 | B 的输入依赖 A 的输出 |
| `create_task(coro())` + `await task` | 注册后继续，稍后等结果 | ≈ max(A时间, B时间) | A 和 B 互相独立 |
| `asyncio.gather(a(), b())` | 两者并发，统一等全部完成 | ≈ max(A时间, B时间) | 批量并发，统一收结果 |

#### 一个常见误解

有些人认为：”`async def` 函数天生是并发的”。

这是错误的。**`async def` 只是让函数可以被挂起，不代表它会并发执行。**  
只有通过 `create_task()`、`gather()`、`TaskGroup` 等手段，才能让多个协程真正并发。

> **`await` 让线程不阻塞** ≠ **`create_task` 让任务并发**。  
> 前者是异步的基础，后者才是并发的来源。

### 为什么 Future 在复杂系统里很关键

如果你要写：

- 动态批处理器
- 异步 RPC 聚合器
- 推理请求到后台 worker 的桥接层

你几乎一定会用到 `Future`，因为调用方需要一个”先返回的句柄”，后台系统稍后再把结果写回去。

### 25.3.4 协程的生命周期状态机

理解协程 / Task 的状态转换，有助于正确处理结果、异常和取消。

```text
async def func() 被调用
        │
        ▼
   协程对象创建（未被调度）
        │
        │ create_task() 或 asyncio.run()
        ▼
    PENDING（等待 loop 调度）
        │
        │ loop 第一次调度
        ▼
    RUNNING（正在执行）
        │
        ├─ 遇到 await ──► SUSPENDED（挂起，等待 IO / timer / Future）
        │                       │
        │                       │ 条件就绪，loop 恢复
        │                       └──────────────────► RUNNING
        │
        ├─ 执行到 return ──► DONE（正常完成）
        ├─ 未捕获异常 ──────► DONE（含异常）
        └─ task.cancel() ──► CancelledError 注入 ──► CANCELLED
```

对应到代码可以查询的接口：

```python
task = asyncio.create_task(some_coro())

task.done()        # 已结束（完成 / 异常 / 取消）中的任意一种
task.cancelled()   # 是否被取消
task.result()      # 正常结果（未完成则 raise InvalidStateError）
task.exception()   # 捕获的异常（正常完成则 raise InvalidStateError）
```

一个重要细节：

> `task.done()` 为 `True` 不代表正常完成，它也涵盖”被取消”和”抛出了异常”。

因此，完整判断顺序通常是：

```python
if task.cancelled():
    # 处理取消
    ...
elif task.exception() is not None:
    # 处理异常
    exc = task.exception()
    ...
else:
    result = task.result()
```

#### `add_done_callback()`

Task 完成时可以注册同步回调，这在把 Task 结果桥接到非 async 系统时很有用：

```python
def on_done(t: asyncio.Task) -> None:
    if t.cancelled():
        print(“task was cancelled”)
    elif t.exception():
        print(f”task failed: {t.exception()}”)
    else:
        print(f”task result: {t.result()}”)

task = asyncio.create_task(some_coro())
task.add_done_callback(on_done)
```

注意：回调是**同步函数**，不是协程，且在 loop 线程里调用。

### 25.3.5 Awaitable 协议

`await` 能等的不只是 Task 和 Future，任何实现了 `__await__` 的对象都可以。

Python 中”可被 await 的对象”共三类：

| 类型 | 来源 |
|------|------|
| 协程对象 | `async def` 函数调用的返回值 |
| Task | `asyncio.create_task()` 的返回值 |
| Future | `loop.create_future()` 的返回值 |

偶尔还会看到自定义 awaitable，例如：

```python
class ReadyAfter:
    “””等待一段时间后返回一个值的自定义 awaitable。”””

    def __init__(self, delay: float, value: str):
        self._delay = delay
        self._value = value

    def __await__(self):
        # 委托给已有 awaitable 的 __await__ 实现
        yield from asyncio.sleep(self._delay).__await__()
        return self._value


async def main():
    result = await ReadyAfter(0.2, “hello”)
    print(result)  # hello


asyncio.run(main())
```

工程中很少直接手写 `__await__`，但理解它有助于你看清：

> `await` 不是魔法语法，而是一个标准协议——凡是实现了 `__await__` 的对象，都可以被 await。

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

### `asyncio.sleep(0)`：主动让出执行权

`asyncio.sleep(0)` 是一个非常有用但经常被忽视的工具。

它的含义是：

> 当前协程把执行权交回给事件循环，让 loop 先处理其他就绪任务，然后再回来继续执行我。

为什么这很重要？

因为 `asyncio` 是**协作式多任务**（cooperative multitasking）：协程只有在遇到 `await` 时才会主动挂起。如果一个协程在 `await` 之间做了大量同步计算，它会独占事件循环，导致其他任务得不到执行机会。

```python
import asyncio


async def cpu_heavy():
    total = 0
    for i in range(1_000_000):
        total += i
        if i % 100_000 == 0:
            await asyncio.sleep(0)   # 每 10 万次让出一次执行权
    return total


async def reporter():
    for _ in range(5):
        print("reporter: still alive")
        await asyncio.sleep(0.1)


async def main():
    async with asyncio.TaskGroup() as tg:
        tg.create_task(cpu_heavy())
        tg.create_task(reporter())


asyncio.run(main())
```

如果去掉 `cpu_heavy` 里的 `await asyncio.sleep(0)`，`reporter` 就会在 `cpu_heavy` 跑完之前完全得不到调度。

一句话总结：

> `await asyncio.sleep(0)` 是协作式调度系统中"主动让出"的标准惯用法。

### Task 命名与自省接口

Task 提供了一组自省接口，在调试和监控中非常有用。

#### 命名 Task

```python
# 创建时命名
task = asyncio.create_task(some_coro(), name="fetch-user-42")

# 之后修改名字
task.set_name("fetch-user-42-retry")

# 获取名字
print(task.get_name())  # fetch-user-42-retry
```

#### 获取当前 Task 和全部 Task

```python
async def my_coro():
    current = asyncio.current_task()    # 获取当前正在执行的 Task
    print(f"I am: {current.get_name()}")

    all_tasks = asyncio.all_tasks()     # 获取当前 loop 中所有未完成的 Task
    print(f"total running tasks: {len(all_tasks)}")
```

`asyncio.all_tasks()` 在调试"为什么程序不退出"时非常有用：如果有 Task 泄漏，可以在这里看到。

#### Task 自省小结

| 接口 | 作用 |
|------|------|
| `task.get_name()` | 获取 Task 名称 |
| `task.set_name(name)` | 设置 Task 名称 |
| `asyncio.current_task()` | 当前协程对应的 Task |
| `asyncio.all_tasks()` | 当前 loop 全部未完成 Task |
| `task.done()` | 是否已结束 |
| `task.cancelled()` | 是否被取消 |
| `task.result()` | 获取结果（未完成则抛 `InvalidStateError`） |
| `task.exception()` | 获取异常（正常完成则抛 `InvalidStateError`） |

### `wait_for()` 与 `asyncio.timeout()` 详解

超时控制是异步系统中最基本的防御手段。Python 提供了两种方式。

#### `asyncio.wait_for()`

```python
try:
    result = await asyncio.wait_for(slow_coro(), timeout=2.0)
except asyncio.TimeoutError:
    print("timed out")
```

`wait_for()` 在超时后会：
1. 向被等待的协程发送取消信号
2. 等取消完成后，把异常转换成 `asyncio.TimeoutError` 抛出

#### `asyncio.timeout()`（Python 3.11+，推荐）

```python
try:
    async with asyncio.timeout(2.0):
        result = await slow_coro()
        result2 = await another_coro()  # 整个 block 共享一个 deadline
except asyncio.TimeoutError:
    print("timed out")
```

`asyncio.timeout()` 是上下文管理器风格，更适合**一个 deadline 管辖多个 await** 的场景。

两者对比：

| | `wait_for()` | `asyncio.timeout()` |
|---|---|---|
| 风格 | 函数包装 | 上下文管理器 |
| 管辖范围 | 单个 awaitable | 整个 `async with` 块 |
| Python 版本 | 3.4+ | 3.11+ |
| 推荐用法 | 包裹单个操作 | 多步操作共享 deadline |

#### `timeout_at()` 与绝对时间截止

```python
deadline = asyncio.get_running_loop().time() + 5.0  # 5 秒后的绝对时间

async with asyncio.timeout_at(deadline):
    await step_one()
    await step_two()
```

`timeout_at()` 适合在多层函数间传递同一个截止时间，避免每层都重新算相对时间。

#### 检查剩余时间

```python
async with asyncio.timeout(10.0) as deadline_ctx:
    await step_one()
    remaining = deadline_ctx.deadline() - asyncio.get_running_loop().time()
    print(f"still have {remaining:.2f}s")
    await step_two()
```

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

它比”到处裸 `create_task()`”更适合复杂系统，因为它更强调结构化并发。

### `gather()` 的 `return_exceptions` 参数

默认情况下，`gather()` 中任何一个子任务抛出异常，`gather()` 会立刻把异常传播给调用方，**其余任务继续运行但结果被丢弃**，这很容易造成资源泄漏。

`return_exceptions=True` 改变这个行为：

```python
results = await asyncio.gather(
    a(), b(), c(),
    return_exceptions=True
)

for r in results:
    if isinstance(r, Exception):
        print(f”task failed: {r}”)
    else:
        print(f”task result: {r}”)
```

此时所有任务都会跑完，异常作为普通返回值混在结果列表里，调用方自行判断。

使用建议：

| 场景 | 推荐做法 |
|------|----------|
| “有一个失败就整体失败” | 默认 `gather()`，或用 `TaskGroup` |
| “每个任务独立，失败后继续” | `gather(return_exceptions=True)` |
| 需要结构化生命周期管理 | `TaskGroup`（Python 3.11+） |

### `asyncio.wait()`：更灵活的等待控制

`wait()` 提供比 `gather()` 更细粒度的控制：

```python
tasks = {asyncio.create_task(a()), asyncio.create_task(b())}

done, pending = await asyncio.wait(tasks, timeout=2.0)

for t in done:
    print(“done:”, t.result())
for t in pending:
    t.cancel()
```

`wait()` 的三种等待模式（`return_when` 参数）：

| 值 | 含义 |
|----|------|
| `ALL_COMPLETED`（默认） | 全部完成后返回 |
| `FIRST_COMPLETED` | 第一个完成即返回 |
| `FIRST_EXCEPTION` | 第一个抛异常即返回 |

`FIRST_COMPLETED` 常用于”多个副本竞速取最快结果”的场景。

注意：`wait()` 不会自动取消 `pending` 集合里的任务，**需要调用方手动取消**，否则会有任务泄漏。

### `asyncio.shield()`：保护任务不被取消

有时你不想让外部取消传播到某个正在执行的操作——例如数据库写入、日志上报、资源清理。

`shield()` 可以为任务提供保护：

```python
async def important_write():
    await asyncio.sleep(1)
    print(“write committed”)


async def main():
    task = asyncio.create_task(important_write())
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=0.5)
    except asyncio.TimeoutError:
        print(“timeout, but write continues in background”)
    await task   # 仍然等待写入完成
```

`shield()` 的工作原理：

- 外层取消信号打在 `shield()` 包裹的那个 Future 上，不会穿透到内层 `task`
- 内层 `task` 继续运行，不感知外部取消
- 但如果内层 `task` 本身被直接 `.cancel()`，shield 也保护不了

一个工程原则：

> `shield()` 不是”永久豁免”，而是”隔离当前这一层的取消信号”。

### `asyncio.as_completed()`：按完成顺序处理结果

`gather()` 等所有任务完成才统一返回。  
`as_completed()` 则让你**按任务完成的先后顺序**逐个处理：

```python
import asyncio
import random


async def fetch(name: str) -> str:
    await asyncio.sleep(random.uniform(0.1, 1.0))
    return f”{name} done”


async def main():
    coros = [fetch(“A”), fetch(“B”), fetch(“C”)]
    for coro in asyncio.as_completed(coros):
        result = await coro
        print(result)   # 谁先完成谁先打印


asyncio.run(main())
```

`as_completed()` 在以下场景比 `gather()` 更合适：

- 需要及时响应最早完成的任务
- 流式展示部分结果（不必等所有任务完成）
- 批量请求中尽快处理已就绪的响应

### 五种并发控制方式总览

| 方式 | 适合场景 |
|------|----------|
| `await coro` | 顺序等待单个操作 |
| `create_task()` | 后台启动，稍后再等 |
| `gather()` | 并发运行一组任务，统一等待 |
| `TaskGroup` | 结构化并发，统一生命周期和异常管理 |
| `wait()` | 需要 FIRST_COMPLETED / 超时后处理剩余 |
| `as_completed()` | 流式按完成顺序处理结果 |
| `shield()` | 隔离取消信号，保护关键操作 |

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

因为它能把”父任务 -> 子任务”的生命周期绑定起来。  
如果一组任务属于同一个业务动作，那么它们就不应该在失败时各自散落。

### 取消是如何传播的

理解取消传播路径，对写出正确的清理逻辑很关键。

```text
task.cancel()
    │
    ▼
loop 在下一次 task 运行到 await 时，注入 CancelledError
    │
    ▼
协程内部 await 的那一层收到 CancelledError
    │
    ├─ 如果没有被捕获 → 继续向上传播，直到 Task 结束
    └─ 如果被 except 捕获但没有 raise → 取消被吞掉（危险！）
```

取消的注入点是**下一个 `await`**。如果协程在两次 `await` 之间做了大量同步计算，取消信号要等到下一次 `await` 才能被注入。

#### `cancel(msg=...)` 传递取消原因（Python 3.9+）

```python
task.cancel(“shutdown: service restarting”)

async def worker():
    try:
        await asyncio.sleep(10)
    except asyncio.CancelledError as e:
        print(f”cancelled: {e}”)   # 打印取消原因
        raise
```

这在调试多级取消传播时很有帮助，可以沿着调用链追踪是谁发起了取消。

#### `finally` 是取消场景的首选清理位置

```python
async def worker():
    conn = await get_connection()
    try:
        await do_work(conn)
    finally:
        await conn.close()   # 无论正常完成还是被取消，都会执行
```

`finally` 比 `except CancelledError + raise` 更安全，因为它不会意外拦截其他异常。

### 取消风暴与 `uncancel()`（Python 3.11+）

在嵌套 TaskGroup 中，如果一个子任务失败，TaskGroup 会取消同组所有其他子任务。  
有时内部代码需要用 `try/except CancelledError` 做临时保护，又不想影响外部的取消计数。

Python 3.11 引入了 `task.uncancel()` 来处理这种情况：

```python
async def careful_worker():
    task = asyncio.current_task()
    try:
        await risky_operation()
    except asyncio.CancelledError:
        # 我内部消化了这次取消，通知外部”少算一次”
        task.uncancel()
        await fallback_operation()
```

工程上直接使用 `uncancel()` 较少见，但了解它能帮助你读懂 Python 3.11+ TaskGroup 的内部实现。

### 优雅停机的基本模式

生产系统里最常见的取消场景是**优雅停机**：收到 SIGTERM 后，让所有正在处理的请求完成，再退出。

```python
import asyncio
import signal


async def main():
    shutdown = asyncio.Event()

    def on_signal():
        print(“shutdown signal received”)
        shutdown.set()

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, on_signal)
    loop.add_signal_handler(signal.SIGINT, on_signal)

    # 启动工作任务
    async with asyncio.TaskGroup() as tg:
        tg.create_task(serve_requests(shutdown))
        tg.create_task(background_flush(shutdown))


async def serve_requests(shutdown: asyncio.Event):
    while not shutdown.is_set():
        await asyncio.sleep(0.1)   # 处理请求
    print(“serve_requests: draining and exiting”)


async def background_flush(shutdown: asyncio.Event):
    while not shutdown.is_set():
        await asyncio.sleep(1.0)
    print(“background_flush: flushing final batch”)


asyncio.run(main())
```

关键设计点：
- 用 `Event` 广播停机信号，而不是直接 `cancel()` 所有 Task
- 让每个 worker 自己检查信号并做完手头的事再退出
- TaskGroup 确保全部 worker 完成后主流程才退出

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
| 事件循环 | 调度”谁现在能继续执行” |
| selector | 告诉 loop 哪些 fd / socket 已 ready |
| reactor | 统一管理事件注册、等待和分发的模式 |
| `epoll / kqueue / IOCP` | 不同平台上用于事件通知或完成通知的底层机制 |
| 协程对象 | 一段尚未完成的异步逻辑 |
| Task 状态机 | PENDING → RUNNING → DONE / CANCELLED |
| Task | 被 loop 管理的执行单元 |
| Future | 未来结果的占位承诺 |
| Awaitable 协议 | 实现 `__await__` 即可被 await |
| `asyncio.sleep(0)` | 主动让出执行权，协作式调度的标准惯用法 |
| Task 自省 | `get_name()`、`current_task()`、`all_tasks()`、`done()`、`result()` |
| `wait_for()` | 给单个 awaitable 加超时，超时后取消并抛 TimeoutError |
| `asyncio.timeout()` | 上下文管理器风格超时，多个 await 共享一个 deadline |
| `gather()` | 并发运行一组任务，统一等待 |
| `gather(return_exceptions=True)` | 所有任务跑完，异常作为普通返回值处理 |
| `wait()` | 细粒度等待控制（FIRST_COMPLETED / ALL_COMPLETED） |
| `as_completed()` | 按任务完成顺序逐个处理结果 |
| `shield()` | 隔离外层取消信号，保护内层关键操作 |
| `TaskGroup` | 结构化并发，统一生命周期和异常管理 |
| `Event` | 广播”某个条件已经成立” |
| 取消传播 | CancelledError 在下一个 await 处注入，不应被吞掉 |
| 优雅停机 | 用 Event 广播停机信号，worker 自行排空后退出 |
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
4. `uvloop` 和标准 loop 的差异主要体现在哪？为什么它不是”所有异步程序的银弹”？
5. 设计一个”并发抓取 4 个特征服务并聚合结果”的最小异步架构，并说明其中 loop、Task、Event 各自扮演什么角色。
6. 画出协程 / Task 的状态机，说明 `task.done()` 为 `True` 时可能对应哪几种状态，以及如何用代码正确区分它们。
7. `asyncio.sleep(0)` 的作用是什么？在什么场景下必须使用它？如果不用会发生什么？
8. `wait_for()` 和 `asyncio.timeout()` 各自适合什么场景？两者在超时后对被等待任务的行为有何相同点？
9. `gather()` 默认行为和 `gather(return_exceptions=True)` 有什么区别？`wait(return_when=FIRST_COMPLETED)` 又解决什么不同的问题？
10. 解释 `shield()` 的保护机制：它能阻止什么？不能阻止什么？结合代码说明一个合理的使用场景。

