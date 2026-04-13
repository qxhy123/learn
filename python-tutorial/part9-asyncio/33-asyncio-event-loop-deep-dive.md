# 第33章：事件循环深度解析

> 第25章从"为什么需要 asyncio"的角度讲了事件循环的宏观工作原理。本章往下一层：`asyncio.Runner`、底层 I/O 注册 API、`run_forever()` 生命周期、自定义 loop policy、Windows ProactorEventLoop 的差异，以及在 Jupyter 和嵌套环境中事件循环会遇到的真实问题。这些内容在调试复杂系统、编写框架代码或理解第三方库源码时不可或缺。

---

## 学习目标

完成本章学习后，你将能够：

1. 使用 `asyncio.Runner` 精确控制事件循环生命周期
2. 使用 `loop.add_reader()`/`add_writer()` 直接注册低层 I/O 事件
3. 理解 `call_soon()`、`call_later()`、`call_at()` 三种调度 API 的关系
4. 正确使用 `loop.run_forever()` + `loop.stop()` 构建长生命周期服务
5. 理解 Windows 上 ProactorEventLoop 与 SelectorEventLoop 的差异
6. 理解并解决 Jupyter / 嵌套环境下的事件循环冲突问题

---

## 正文内容

## 33.1 `asyncio.Runner`：精确控制 loop 生命周期

### 33.1.1 为什么 `asyncio.run()` 有时不够用

`asyncio.run()` 每次调用都会创建一个全新的事件循环，运行完就关闭。  
在大多数场景下这是正确的。但有时你需要：

- 在同一个 loop 里**依次运行多个顶层协程**
- 在 loop 关闭前**执行一些收尾操作**
- 在测试框架里**手动控制 loop 的生命周期**

这就是 `asyncio.Runner`（Python 3.11+）出现的原因。

### 33.1.2 `asyncio.Runner` 基本用法

```python
import asyncio


async def setup() -> dict:
    await asyncio.sleep(0.01)
    return {"config": "loaded"}


async def run(config: dict) -> str:
    await asyncio.sleep(0.01)
    return f"ran with {config}"


async def teardown(state: dict) -> None:
    await asyncio.sleep(0.01)
    print(f"teardown: {state}")


def main():
    with asyncio.Runner() as runner:
        config = runner.run(setup())
        result = runner.run(run(config))
        runner.run(teardown({"result": result}))
        print(f"final: {result}")


main()
```

关键点：
- `asyncio.Runner` 作为上下文管理器使用
- 同一个 `Runner` 实例内，每次 `runner.run()` 使用**同一个事件循环**
- 退出 `with` 块时，loop 会被正确关闭

### 33.1.3 `Runner` vs `asyncio.run()` 对比

| 特性 | `asyncio.run()` | `asyncio.Runner` |
|------|----------------|------------------|
| loop 生命周期 | 每次调用新建+关闭 | 手动控制 |
| 运行多个协程 | 需要嵌套在一个 main 里 | 可以多次 `runner.run()` |
| 适合场景 | 简单入口点 | 测试框架、CLI工具、精细控制 |

### 33.1.4 在测试框架中使用 Runner

```python
import asyncio
import pytest


@pytest.fixture(scope="module")
def runner():
    with asyncio.Runner() as r:
        yield r


def test_first(runner):
    result = runner.run(some_async_operation())
    assert result == expected_1


def test_second(runner):
    # 与 test_first 共享同一个事件循环
    result = runner.run(another_async_operation())
    assert result == expected_2
```

---

## 33.2 底层 I/O 注册：`add_reader()` 和 `add_writer()`

### 33.2.1 什么时候需要这些 API

在大多数业务代码里，你不会直接用 `add_reader()`/`add_writer()`——`StreamReader`/`StreamWriter` 已经帮你封装好了。

但在以下场景你会遇到它们：

- 把**非 asyncio 的 socket**（例如来自第三方库的 fd）集成进事件循环
- 编写**自定义协议层**或底层网络框架
- 阅读 asyncio 本身的源码或第三方异步驱动

### 33.2.2 基本语义

```python
loop.add_reader(fd, callback, *args)
loop.add_writer(fd, callback, *args)
loop.remove_reader(fd)
loop.remove_writer(fd)
```

- `add_reader(fd, cb)`：当 fd 变为**可读**时，调用 cb
- `add_writer(fd, cb)`：当 fd 变为**可写**时，调用 cb
- 回调是**一次性**的——触发后如果还想继续监听，需要重新注册
- 不是协程，而是普通回调

### 33.2.3 用 `add_reader` 把普通 socket 接入 asyncio

```python
import asyncio
import socket


async def echo_server():
    loop = asyncio.get_running_loop()

    # 创建一个普通的阻塞 socket
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("127.0.0.1", 9876))
    server_sock.listen(5)
    server_sock.setblocking(False)   # ← 关键：必须设为非阻塞

    future: asyncio.Future = loop.create_future()

    def accept_connection():
        client_sock, addr = server_sock.accept()
        print(f"connection from {addr}")
        client_sock.setblocking(False)
        loop.add_reader(client_sock.fileno(), handle_client, client_sock)
        # 重新注册自己，继续监听新连接
        loop.add_reader(server_sock.fileno(), accept_connection)

    def handle_client(client_sock: socket.socket):
        data = client_sock.recv(1024)
        if data:
            client_sock.sendall(b"echo:" + data)
            loop.add_reader(client_sock.fileno(), handle_client, client_sock)
        else:
            loop.remove_reader(client_sock.fileno())
            client_sock.close()

    loop.add_reader(server_sock.fileno(), accept_connection)
    print("server listening on :9876")

    # 等待（实际使用中可以是 shutdown_event.wait()）
    await asyncio.sleep(5)
    loop.remove_reader(server_sock.fileno())
    server_sock.close()


asyncio.run(echo_server())
```

这个例子虽然冗长，但非常清楚地展示了：

> `add_reader()` 本质上就是在告诉 selector："当这个 fd 可读时，运行这个回调。"

### 33.2.4 `add_reader`/`add_writer` 在 Windows 上不可用

这是一个重要的平台差异：

- Unix（Linux/macOS）：`SelectorEventLoop` 支持 `add_reader`/`add_writer`
- Windows 默认（3.8+）：`ProactorEventLoop` **不支持** `add_reader`/`add_writer`

如果你的代码依赖这些 API，需要在 Windows 上显式切换到 `SelectorEventLoop`（见 33.5 节）。

---

## 33.3 定时器 API：`call_soon()`、`call_later()`、`call_at()`

第25章已经讲了 `call_soon()` 和 `call_later()`，本节补充完整对比和 `call_at()`。

### 33.3.1 三种定时器的关系

| API | 时机 | 时间参数 |
|-----|------|---------|
| `loop.call_soon(cb)` | 下一次 loop 迭代尽快执行 | 无 |
| `loop.call_later(delay, cb)` | `delay` 秒后执行 | 相对延迟（秒） |
| `loop.call_at(when, cb)` | `loop.time()` 达到 `when` 时执行 | 绝对时间（loop 时钟） |

### 33.3.2 `call_at()` 的语义

`loop.time()` 返回事件循环的单调时钟值（不是 `time.monotonic()` 的直接映射，但通常非常接近）。

```python
import asyncio


def say(msg: str):
    print(f"[{asyncio.get_event_loop().time():.3f}] {msg}")


async def main():
    loop = asyncio.get_running_loop()
    now = loop.time()

    loop.call_soon(say, "call_soon")
    loop.call_later(0.1, say, "call_later 0.1s")
    loop.call_at(now + 0.2, say, "call_at now+0.2s")
    loop.call_at(now + 0.05, say, "call_at now+0.05s")

    await asyncio.sleep(0.3)


asyncio.run(main())
# 输出顺序：call_soon → call_at 0.05s → call_later 0.1s → call_at 0.2s
```

### 33.3.3 `call_at()` 的工程用途

`call_at()` 最常见的用途是在**已知截止时间**的场景里设置定时回调，而不必每次都计算剩余时间：

```python
import asyncio


class DeadlinePropagator:
    """演示 call_at 在 deadline 传递场景里的用途"""

    def __init__(self, deadline: float):
        self._deadline = deadline

    def schedule_at_deadline(self, callback, *args):
        loop = asyncio.get_running_loop()
        loop.call_at(self._deadline, callback, *args)


async def main():
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 0.5

    prop = DeadlinePropagator(deadline)
    prop.schedule_at_deadline(print, "deadline callback fired!")

    await asyncio.sleep(0.6)


asyncio.run(main())
```

### 33.3.4 `call_soon_threadsafe()`：跨线程调用

在**其他线程**里向事件循环提交回调（不是协程），使用：

```python
loop.call_soon_threadsafe(callback, *args)
```

这是线程安全的 `call_soon()`。常见于：
- 线程里的回调通知 asyncio 事件循环
- GUI 线程 → asyncio 线程通信

---

## 33.4 `run_forever()` + `loop.stop()`：手动管理 loop 生命周期

### 33.4.1 什么时候用 `run_forever()`

`asyncio.run()` 适合"运行一个顶层协程"。如果你在构建：

- 长生命周期的服务器框架
- 嵌入 asyncio 到已有事件循环系统（例如 Tkinter/Qt）
- 需要在外部代码控制 loop 启停的场景

那么 `loop.run_forever()` + `loop.stop()` 更合适。

### 33.4.2 基本模式

```python
import asyncio
import signal


async def background_worker():
    try:
        while True:
            print("working...")
            await asyncio.sleep(1.0)
    except asyncio.CancelledError:
        print("worker cancelled")
        raise


def main():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    worker_task = loop.create_task(background_worker())

    # 信号处理：优雅停机
    def stop():
        worker_task.cancel()
        loop.call_soon(loop.stop)

    loop.add_signal_handler(signal.SIGINT,  stop)
    loop.add_signal_handler(signal.SIGTERM, stop)

    print("loop starting...")
    loop.run_forever()   # ← 阻塞，直到 loop.stop() 被调用

    # run_forever() 返回后，等待任务收尾
    loop.run_until_complete(asyncio.gather(*asyncio.all_tasks(loop),
                                           return_exceptions=True))
    loop.close()
    print("loop closed")


main()
```

### 33.4.3 `run_forever()` 和 `run_until_complete()` 的关系

```text
asyncio.run(coro)
  = loop.run_until_complete(coro) + 收尾 + loop.close()

loop.run_forever()
  = 持续运行，直到 loop.stop() 被调用（通常在回调或信号处理里）
```

### 33.4.4 不能在 `run_forever()` 正在运行时再次 `run_until_complete()`

```python
# 错误：loop 正在 run_forever() 时调用 run_until_complete() 会报错
loop.run_forever()   # ← blocking
loop.run_until_complete(coro)   # ← 永远到不了这里
```

正确做法是用 `call_soon_threadsafe()` 或在回调里处理需要额外运行的逻辑。

---

## 33.5 自定义 Event Loop Policy

### 33.5.1 什么是 loop policy

`asyncio` 的 loop policy 控制：
- 如何创建新的事件循环
- 如何获取当前线程的事件循环

```python
policy = asyncio.get_event_loop_policy()
```

### 33.5.2 内置 policy 类型

| Policy | 平台 | Loop 类型 |
|--------|------|-----------|
| `DefaultEventLoopPolicy` | Unix | `SelectorEventLoop` |
| `DefaultEventLoopPolicy` | Windows 3.8+ | `ProactorEventLoop` |
| `WindowsSelectorEventLoopPolicy` | Windows | `SelectorEventLoop` |

### 33.5.3 切换 policy

```python
import asyncio
import sys


if sys.platform == "win32":
    # 在 Windows 上强制使用 SelectorEventLoop
    # （当你需要 add_reader/add_writer 时）
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def main():
    loop = asyncio.get_running_loop()
    print(type(loop))   # SelectorEventLoop on Windows


asyncio.run(main())
```

### 33.5.4 编写自定义 loop policy（高级）

```python
import asyncio


class CustomLoopPolicy(asyncio.DefaultEventLoopPolicy):
    """一个在每次创建 loop 时自动设置调试模式的自定义 policy"""

    def new_event_loop(self) -> asyncio.AbstractEventLoop:
        loop = super().new_event_loop()
        loop.set_debug(True)
        loop.slow_callback_duration = 0.05
        return loop


asyncio.set_event_loop_policy(CustomLoopPolicy())

asyncio.run(main())   # 创建的 loop 会自动开启调试模式
```

自定义 policy 的工程用途：

| 场景 | 用途 |
|------|------|
| 测试框架 | 每个测试自动用新 loop + 调试模式 |
| `uvloop` 集成 | `uvloop.install()` 本质上就是设置了一个 policy |
| 监控框架 | 自动给所有 loop 挂上 instrumentation |

---

## 33.6 Windows 上的 ProactorEventLoop

### 33.6.1 为什么 Windows 不同

Unix 上的 `SelectorEventLoop` 使用 `select`/`poll`/`epoll`/`kqueue`——这些是"就绪通知"模型。  
Windows 上没有等价的高效 fd 监听机制，但有 **IOCP（I/O Completion Ports）**——这是一种"完成通知"模型。

因此，Python 在 Windows 上默认使用 `ProactorEventLoop`：

| 特性 | SelectorEventLoop (Unix) | ProactorEventLoop (Windows) |
|------|--------------------------|------------------------------|
| 底层机制 | epoll/kqueue/select | IOCP |
| `add_reader()`/`add_writer()` | 支持 | **不支持** |
| subprocess 支持 | 需要额外配置 | 原生支持 |
| 管道（pipe）支持 | 支持 | 部分支持 |
| 性能特征 | 高并发 socket | 高吞吐文件/网络 I/O |

### 33.6.2 实际影响

最常见的 Windows asyncio 坑：

```python
# 在 Windows 上运行时，某些库可能报错：
# NotImplementedError: add_reader() is not supported by ProactorEventLoop
```

解决方法：

```python
import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
```

但这样做会失去 ProactorEventLoop 的一些优势（如 subprocess 支持），需要权衡。

### 33.6.3 subprocess 在 Windows 上的特殊处理

`asyncio.create_subprocess_exec()` 在 Windows 上需要 `ProactorEventLoop`：

```python
import asyncio
import sys


async def main():
    proc = await asyncio.create_subprocess_exec(
        "ping", "-n", "2", "127.0.0.1",
        stdout=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    print(stdout.decode())


# Windows 上需要确保使用 ProactorEventLoop（Python 3.8+ 是默认值）
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

asyncio.run(main())
```

---

## 33.7 Jupyter / 嵌套事件循环

### 33.7.1 为什么 Jupyter 里直接用 `asyncio.run()` 会报错

Jupyter Notebook 和 Jupyter Lab 本身已经在运行一个事件循环。  
当你在 cell 里调用 `asyncio.run()` 时：

```python
# ← 这会报错：
asyncio.run(some_coroutine())
# RuntimeError: This event loop is already running
```

原因：`asyncio.run()` 要求**当前线程没有正在运行的事件循环**，但 Jupyter 的内核已经有了一个。

### 33.7.2 Jupyter 里的正确用法

在 Jupyter 里，可以直接 `await` 顶层协程——Jupyter 会自动处理：

```python
# Jupyter cell 里可以直接 await（IPython 7.0+）
result = await some_coroutine()
print(result)
```

如果你在使用旧版 Jupyter 或者某些环境不支持顶层 await，可以用 `nest_asyncio`：

```bash
pip install nest_asyncio
```

```python
import nest_asyncio
import asyncio

nest_asyncio.apply()   # 打补丁，允许嵌套 event loop

# 之后可以在有 loop 的环境里调用 asyncio.run()
result = asyncio.run(some_coroutine())
```

**注意**：`nest_asyncio` 是一个 hack，不应在生产代码中使用。它仅适合 notebook 探索和快速原型。

### 33.7.3 框架集成：把 asyncio 嵌入到其他事件系统

有时你需要把 asyncio 集成到已有事件循环系统（例如 Tkinter、Qt、wxPython）。  
核心思路是利用 `loop.run_until_complete()` 或在 GUI 线程定时推进 asyncio loop：

```python
import asyncio
import threading


class AsyncioThread(threading.Thread):
    """在独立线程里运行 asyncio event loop，用于与同步框架集成"""

    def __init__(self):
        super().__init__(daemon=True)
        self.loop = asyncio.new_event_loop()
        self._ready = threading.Event()

    def run(self):
        asyncio.set_event_loop(self.loop)
        self._ready.set()
        self.loop.run_forever()

    def submit(self, coro):
        """线程安全地向 asyncio loop 提交协程"""
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def stop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)


# 使用
async_thread = AsyncioThread()
async_thread.start()
async_thread._ready.wait()


# 从主线程（同步）调用异步代码
future = async_thread.submit(some_async_func())
result = future.result(timeout=5.0)
```

---

## 33.8 低层 Transport 和 Protocol 进阶

第26章已经介绍了 transport/protocol 的基本概念。本节补充两个常用的变体。

### 33.8.1 DatagramProtocol：UDP 支持

```python
import asyncio


class UDPEchoServerProtocol(asyncio.DatagramProtocol):
    def __init__(self):
        self.transport = None

    def connection_made(self, transport: asyncio.DatagramTransport):
        self.transport = transport

    def datagram_received(self, data: bytes, addr: tuple):
        message = data.decode()
        print(f"received {message!r} from {addr}")
        self.transport.sendto(f"echo:{message}".encode(), addr)

    def error_received(self, exc: Exception):
        print(f"error: {exc}")

    def connection_lost(self, exc):
        print("connection closed")


async def main():
    loop = asyncio.get_running_loop()
    transport, protocol = await loop.create_datagram_endpoint(
        UDPEchoServerProtocol,
        local_addr=("127.0.0.1", 9999),
    )
    print("UDP server listening on :9999")
    try:
        await asyncio.sleep(10)
    finally:
        transport.close()


asyncio.run(main())
```

### 33.8.2 连接管道（pipe）：与子进程通信

```python
import asyncio
import os


class PipeProtocol(asyncio.Protocol):
    def data_received(self, data: bytes):
        print(f"pipe received: {data.decode().strip()}")

    def connection_lost(self, exc):
        print("pipe closed")


async def main():
    loop = asyncio.get_running_loop()
    read_fd, write_fd = os.pipe()

    # 把读端接入 asyncio loop
    read_transport, protocol = await loop.connect_read_pipe(
        PipeProtocol,
        os.fdopen(read_fd),
    )

    # 向写端写数据
    os.write(write_fd, b"hello from pipe\n")
    os.write(write_fd, b"another line\n")

    await asyncio.sleep(0.1)
    read_transport.close()
    os.close(write_fd)


asyncio.run(main())
```

---

## 本章小结

| 主题 | 关键点 |
|------|--------|
| `asyncio.Runner` | 同一 loop 里依次运行多个顶层协程 |
| `add_reader`/`add_writer` | 把任意非阻塞 fd 直接接入 selector |
| `call_at()` | 基于 loop 绝对时间的定时回调 |
| `run_forever()` + `loop.stop()` | 手动管理 loop 生命周期 |
| 自定义 loop policy | 控制 loop 创建方式（uvloop、调试模式等）|
| Windows ProactorEventLoop | 不支持 `add_reader`；subprocess 需要它 |
| Jupyter 嵌套 loop | 直接 `await` 或用 `nest_asyncio` |
| `DatagramProtocol` | UDP 异步服务 |
| `connect_read_pipe` | 把 pipe 接入 asyncio |

---

## 深度学习应用

本章内容在深度学习系统中的典型落点：

- `asyncio.Runner`：测试框架里复用 loop 加载大型模型一次
- 自定义 loop policy：生产环境统一开启 debug 模式 + lag 监控
- `call_at()`：推理服务 deadline 到达时的超时触发
- `run_forever()`：长生命周期推理网关服务主循环
- Jupyter 嵌套 loop：在 notebook 里探索异步推理接口

---

## 练习题

1. `asyncio.Runner` 和 `asyncio.run()` 在 loop 生命周期上的核心区别是什么？什么场景更适合用 `Runner`？
2. `loop.add_reader()` 注册的回调是一次性的还是持续触发的？如果要持续监听需要怎么做？
3. `call_soon()`、`call_later()`、`call_at()` 三者的参数差异是什么？什么时候应该用 `call_at()` 而不是 `call_later()`？
4. 为什么 Jupyter Notebook 里不能直接使用 `asyncio.run()`？有哪两种解决方式？
5. Windows 上的 `ProactorEventLoop` 不支持 `add_reader()`，但为什么 Python 在 Windows 上仍然选择它作为默认 loop？
