# 第27章：asyncio极端复杂场景实战

> 真正困难的 `asyncio` 场景，不是几十个协程一起跑，而是：多租户、突发流量、超时预算、取消风暴、微批处理、背压、限流、部分失败、优雅关闭和可观测性同时出现。

---

## 学习目标

完成本章学习后，你将能够：

1. 理解高复杂度异步系统中的核心失效模式
2. 设计带背压、限流、超时预算和取消传播的异步架构
3. 理解 structured concurrency 在极端场景中的价值
4. 实现一个近似生产级的异步微批处理服务骨架
5. 知道什么时候 `asyncio` 是对的工具，什么时候需要更强的系统边界

---

## 正文内容

## 27.1 极端复杂场景到底“复杂”在哪里

一个高复杂度异步系统往往同时满足以下条件：

- 高并发：瞬时有大量请求进入
- 高不确定性：请求耗时差异很大
- 强约束：有严格超时和 SLA
- 高状态性：队列、缓存、批次、租户策略都在动态变化
- 高失败率：下游依赖会慢、会超时、会半失败

典型场景包括：

1. 异步推理网关
2. 高并发爬虫 / 抓取平台
3. 流式日志 / 事件摄取系统
4. RAG 查询编排器
5. 多租户任务调度器

这些场景难的不是“如何写一个 await”，而是如何控制系统在压力下仍然保持可解释。

---

## 27.2 复杂系统的 7 个核心机制

一个近似生产级的 asyncio 系统，通常至少要具备：

1. **有界队列**：防止无限积压
2. **并发限制**：防止把下游打爆
3. **超时预算**：防止请求无限挂起
4. **取消传播**：上游失败时统一止损
5. **批处理 / 聚合**：提高吞吐
6. **优雅关闭**：确保已接收任务有明确命运
7. **观测指标**：知道系统到底卡在哪里

你可以把它们理解成异步系统的“生命维持器官”。缺一个，系统在低压下也许还看起来正常，但一遇到峰值就容易失控。

---

## 27.3 超时预算不是“每一层都写 5 秒”

一个常见错误是：  
每一层都给下游一个固定超时，例如 5 秒。  
结果是：

- 总请求超时远超用户 SLA
- 上游已经放弃了，底层任务还在继续跑
- 系统里充满失去业务价值的工作

更合理的做法是 **预算传递（deadline propagation）**：

```text
用户总预算 2.0s
  -> 路由 0.1s
  -> 检索 0.4s
  -> 重排 0.3s
  -> 模型推理 1.0s
  -> 后处理 0.2s
```

在代码里，这通常意味着：

- 上游把 deadline 传给下游
- 下游按“剩余预算”设置 timeout

### 一个简化版预算计算

```python
import asyncio
import time


def remaining_budget(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


async def call_with_budget(deadline: float):
    timeout_s = remaining_budget(deadline)
    async with asyncio.timeout(timeout_s):
        await asyncio.sleep(0.2)
```

---

## 27.4 背压、限流、降级必须同时设计

在极端场景下，单独只有一种机制通常不够。

### 只有队列，没有限流

系统会不断积压，最后变成“大型延迟垃圾场”。

### 只有限流，没有背压

上游会继续疯狂制造任务，只是这些任务被更晚地拒绝。

### 只有超时，没有降级

所有请求都一起超时，系统表现为全面抖动。

因此，生产系统通常需要组合策略：

```text
有界队列
  + Semaphore 限流
  + deadline timeout
  + 当积压过高时快速拒绝或降级
```

### 一个简单降级策略表

| 状态 | 行为 |
|------|------|
| 队列正常 | 正常处理 |
| 队列接近上限 | 限制低优先级请求 |
| 队列已满 | 直接拒绝非核心流量 |
| 下游持续超时 | 熔断、返回降级结果 |

---

## 27.5 微批处理（micro-batching）为什么困难

微批处理常见于推理系统，例如：

- 把 1~10ms 内到达的请求收集成一批
- 一起送到模型
- 再把结果拆回各自请求

它的挑战在于同时要处理：

- 等待太久会拉高延迟
- 批太小吞吐不高
- 长短请求混杂会形成不公平
- 取消时要正确移除等待中的请求

所以微批处理的真实难点不是“凑个列表”，而是：

> 在极短时间窗口里，把吞吐、延迟、公平性和取消治理一起做对。

---

## 27.6 实战：异步微批处理推理网关

下面这个例子模拟一个近似生产级的异步推理网关，特点包括：

- 有界输入队列
- 每租户并发限制
- 请求超时预算
- 微批处理
- 取消传播
- 优雅关闭

```python
import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class InferenceRequest:
    tenant: str
    payload: list[float]
    deadline: float
    future: asyncio.Future


class AsyncBatchGateway:
    def __init__(self, max_queue=1000, max_batch=8, batch_wait_ms=20):
        self.queue: asyncio.Queue[InferenceRequest] = asyncio.Queue(maxsize=max_queue)
        self.max_batch = max_batch
        self.batch_wait_s = batch_wait_ms / 1000
        self.shutdown_event = asyncio.Event()
        self.tenant_limits = defaultdict(lambda: asyncio.Semaphore(4))
        self.stats = defaultdict(int)

    async def submit(self, tenant: str, payload: list[float], timeout_s: float):
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        req = InferenceRequest(
            tenant=tenant,
            payload=payload,
            deadline=time.monotonic() + timeout_s,
            future=future,
        )
        await self.queue.put(req)  # 队列满时背压
        self.stats["accepted"] += 1
        return await future

    async def _run_batch(self, batch: list[InferenceRequest]):
        await asyncio.sleep(0.03)  # 模拟模型推理
        for req in batch:
            if not req.future.done():
                req.future.set_result(sum(req.payload))
                self.stats["completed"] += 1

    async def batch_loop(self):
        try:
            while not self.shutdown_event.is_set():
                first = await self.queue.get()
                batch = [first]
                deadline = time.monotonic() + self.batch_wait_s

                while len(batch) < self.max_batch:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                        batch.append(item)
                    except TimeoutError:
                        break

                alive_batch = []
                for req in batch:
                    if req.deadline <= time.monotonic():
                        if not req.future.done():
                            req.future.set_exception(TimeoutError("request deadline exceeded"))
                            self.stats["timeout"] += 1
                        self.queue.task_done()
                        continue
                    alive_batch.append(req)

                if alive_batch:
                    await self._run_batch(alive_batch)
                    for _ in alive_batch:
                        self.queue.task_done()
        except asyncio.CancelledError:
            raise

    async def serve(self):
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self.batch_loop())
            await self.shutdown_event.wait()

    async def shutdown(self):
        self.shutdown_event.set()
```

### 这个示例有哪些关键点

1. **有界队列**：保护系统不被无限流量压垮
2. **deadline**：请求不是无限等待的
3. **微批处理窗口**：用 `batch_wait_ms` 换吞吐
4. **future 回填结果**：把批处理结果映射回单请求
5. **TaskGroup**：服务生命周期可管理

### 这个示例还缺什么

如果要更接近生产，还应继续补：

- 每租户真实配额策略
- 熔断器
- 指标上报
- 请求取消时从等待队列剔除
- 批内异常隔离
- 多 worker / 多副本协调

这恰恰说明：**极端复杂场景的难点不在单个技巧，而在这些技巧如何组合。**

---

## 27.7 什么时候 `asyncio` 已经不是最佳边界

`asyncio` 很强，但也不是万能。以下情况常常要引入更明确的系统边界：

### CPU 密集型重任务

应考虑：

- `asyncio.to_thread()`
- 线程池 / 进程池
- 专门任务队列

### 需要跨进程 / 跨机器分发

应考虑：

- 消息队列
- 作业编排系统
- 外部服务化拆分

### 需要严格持久化和恢复

单机内存队列往往不够，需要：

- durable queue
- 持久化状态
- 明确的幂等语义

所以成熟工程里常见分工是：

```text
asyncio: 单进程内 I/O 编排与协调
message queue / workflow engine: 跨进程、跨节点、可恢复编排
```

---

## 27.8 深度学习应用：异步推理网关与批量推理协调器

在深度学习系统里，本章模式特别适合：

- 异步模型网关
- 检索 + 重排 + LLM 的链路编排
- 批量推理入口层
- 高并发 embedding 服务

一个典型边界是：

```text
asyncio
  负责：队列、限流、批处理、deadline、取消传播、优雅关闭
模型框架 / 推理引擎
  负责：真正的数值计算
```

如果你把两者角色混淆，就容易出现：

- 事件循环被重 CPU / 重 GPU 工作卡死
- 模型服务没有被正确批处理
- 控制流和计算流互相污染

---

## 本章小结

| 主题 | 核心结论 |
|------|----------|
| 极端复杂场景 | 难在多种约束同时出现，而不是 API 难记 |
| deadline 预算 | 比“每层固定超时”更合理 |
| 背压 / 限流 / 降级 | 必须组合设计 |
| 微批处理 | 是吞吐、延迟、公平性、取消治理的综合题 |
| 工具边界 | `asyncio` 强在单进程 I/O 协调，不等于全栈调度平台 |

---

## 练习题

1. 为什么说“每一层都写死 5 秒超时”不是一个合格的异步系统设计？
2. 设计一个带 deadline 预算传递的异步调用链，至少包含 3 个下游阶段。
3. 为什么微批处理系统必须同时考虑背压、取消和公平性？
4. 在什么情况下你应该把任务从 `asyncio` 事件循环移交给线程池、进程池或外部队列？
5. 设计一个近似生产级的异步推理网关：要求支持有界队列、每租户限流、超时预算、微批处理和优雅关闭。
