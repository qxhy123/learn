# 第9章：调度器与请求管理

> 在当前仓库的 vLLM V1 里，调度器不再把“prefill 阶段”和“decode 阶段”当成两套完全不同的流程。它只关心一件事：这一轮还应该为每个请求补算多少 token，以及这些 token 值不值得占用有限的 KV Cache 块和 batch 预算。

---

## 学习目标

学完本章，你将能够：

1. 理解 vLLM V1 调度器的核心抽象和调度目标
2. 掌握当前仓库里请求状态的真实定义
3. 理解统一 token 预算如何同时覆盖 chunked prefill、decode、prefix caching 和 speculative decoding
4. 理解 V1 中的抢占语义为什么以 recompute 为主，而不是 CPU swap
5. 根据源码中的真实参数调优调度行为

---

## 9.1 调度器到底在解决什么问题？

调度器每一轮都要同时回答下面几件事：

1. 当前已经在跑的请求，哪些还能继续推进？
2. 新来的请求里，哪些可以被接纳进这一轮？
3. 这一轮最多能发出去多少 token，才不会把 batch、显存和编码预算压爆？
4. 如果 KV Cache 块不够，应该让谁先让路？

在 V1 中，这些问题被统一到一个很简单的视角里：

- `request.num_tokens_with_spec`：这个请求当前“理论上”需要处理到哪里
- `request.num_computed_tokens`：这个请求“实际上”已经算到哪里
- 调度器每轮只是在努力让后者追上前者

这就是 `vllm/vllm/v1/core/sched/scheduler.py` 开头 `schedule()` 注释里那句很关键的话的含义：
**没有硬编码的 prefill phase / decode phase，只有 token 进度差。**

这套抽象的好处是，一个调度循环可以自然覆盖：

- 普通 prefill
- 普通 decode
- chunked prefill
- prefix caching 命中的“已计算 token”
- speculative decoding 带来的草稿 token

---

## 9.2 当前仓库里的请求状态

很多旧资料会把 vLLM 调度讲成三态：`waiting / running / swapped`。
但当前仓库的 V1 实现不是这个模型。

`vllm/vllm/v1/request.py` 中的 `RequestStatus` 主要是这些状态：

| 状态 | 含义 |
|------|------|
| `WAITING` | 普通等待态，尚未被本轮接纳 |
| `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR` | 结构化输出请求在等 grammar 编译完成 |
| `WAITING_FOR_REMOTE_KVS` | 在等外部 KV connector 的远端 KV 数据 |
| `WAITING_FOR_STREAMING_REQ` | 在等流式输入补全 |
| `RUNNING` | 已有 KV 分配，正在活跃执行 |
| `PREEMPTED` | 被抢占后重新回到等待队列，后续需要重算部分或全部 token |
| `FINISHED_*` | 各类完成/停止/错误终态 |

可以用下面这个图理解主路径：

```text
新请求
  ↓
WAITING ───────────────→ RUNNING ───────────────→ FINISHED_*
  ↑                         │
  │                         │ KV 不够 / 被更高优先级挤出
  └──────── PREEMPTED ←─────┘
```

如果请求带结构化输出、远端 KV、流式输入，还会先进入某个“阻塞等待态”，等条件满足后再回到 `WAITING`。

---

## 9.3 V1 调度循环怎么工作？

当前仓库的主调度入口是：

- `vllm/vllm/v1/engine/core.py` 中的 `EngineCore.step()`
- `vllm/vllm/v1/core/sched/scheduler.py` 中的 `Scheduler.schedule()`

调度器每轮大致做这几步：

```python
def schedule():
    token_budget = max_num_scheduled_tokens or max_num_batched_tokens

    # 1. 先处理 running 请求
    for request in running:
        num_new_tokens = request.num_tokens_with_spec - request.num_computed_tokens
        new_blocks = kv_cache_manager.allocate_slots(request, num_new_tokens)
        if new_blocks is None:
            preempt_lowest_priority_request()
        else:
            schedule_running(request)

    # 2. 再处理 waiting / skipped_waiting 请求
    while waiting and token_budget > 0:
        request = pick_next_waiting_request()
        computed_blocks = kv_cache_manager.get_computed_blocks(request)
        if can_admit(request):
            allocate_slots(...)
            move_to_running(request)
        else:
            break

    return SchedulerOutput(...)
```

这段伪代码和源码并不逐行相同，但抓住了 3 个真实原则。

### 原则 1：先保住已在运行的请求

源码先遍历 `self.running`，优先尝试给运行中的请求继续分配新 slot。
原因很简单：这些请求已经吃掉了 KV Cache，也已经做过一部分计算，优先继续推进，浪费最小。

### 原则 2：调度预算是 token 预算，不只是请求数预算

V1 里最重要的预算是：

- `max_num_seqs`
- `max_num_scheduled_tokens`，如果没单独设置，就退化为 `max_num_batched_tokens`

也就是说，调度器不是简单地“每轮挑 128 个请求”，而是要保证：

- 这一轮活跃序列数不过多
- 这一轮总共发给 GPU 的 token 数不过多

这就是为什么长 prompt 会和 decode 请求直接竞争同一份预算。

### 原则 3：waiting 队列接纳前会先看缓存命中

对于一个从未运行过的新请求，调度器不会马上把整段 prompt 都当成“未计算”：

1. 先调用 `kv_cache_manager.get_computed_blocks(request)`
2. 让 prefix cache 尝试找出已经命中的完整 block
3. 只为真正还没算过的 token 申请新块和计算预算

这一步直接把前缀缓存和调度耦合在了一起。

---

## 9.4 抢占：当前 V1 里真正发生了什么？

### 旧说法：swap 到 CPU

旧版教程或早期资料常会说：

- 显存不够时，把某些请求的 KV Cache 换出到 CPU
- 等显存有空再换回 GPU

### 当前仓库：主路径是 `PREEMPTED + recompute`

当前仓库 V1 的核心语义是：

- 运行中的请求如果拿不到新块，会触发 `_preempt_request(...)`
- 请求状态进入 `PREEMPTED`
- 它会被重新放回等待队列
- 后续再次被调度时，从已有可复用状态重新推进

换句话说，**当前 V1 调度器默认不是围绕本地 GPU↔CPU swap 队列设计的**。

这点也能从官方文档 `docs/usage/v1_guide.md` 看出来：
V1 已经把“GPU <> CPU KV Cache Swapping”列为 removed feature。

### 为什么这样改？

V1 的取舍是：

- 保持核心调度器抽象更简单
- 让 prefix caching、chunked prefill、spec decode 在同一套流程里组合
- 避免 CPU swap 把调度器本身变成一个复杂的内存迁移状态机

需要注意的是：

- 仓库里仍然存在 `kv_connector`、`simple_kv_offload` 等代码
- 这些更多是远端 KV / offload / disaggregation 相关能力
- 不等于旧式 “swapped 队列 + swap-space 参数” 又回来了

---

## 9.5 公平性、优先级和被谁抢占

### 默认策略：FCFS

调度策略配置来自：

- `vllm/vllm/config/scheduler.py`

当前支持：

- `fcfs`
- `priority`

`fcfs` 的核心思想依然是先来先服务，但要注意一个实现细节：

- 运行队列里如果发生抢占
- 在 FCFS 模式下，源码默认 `pop()` 运行队列尾部的请求
- 这通常意味着“最新进入 running 的请求”更容易被抢占

直觉上，这是在保护更早开始执行、已经投入更多计算的请求。

### 优先级模式

如果启用 `priority`，源码会选择：

- 优先级更低
- 或同优先级但到达时间更晚

的请求先被挤出。

这和 `Request.__lt__()` 的定义、以及 `scheduler.py` 里：

```python
preempted_req = max(
    self.running,
    key=lambda r: (r.priority, r.arrival_time),
)
```

是对应的。

---

## 9.6 Chunked Prefill 为什么会直接影响调度？

V1 里 chunked prefill 已经不是“附属优化”，而是调度器本身的一部分。

配置项都在 `SchedulerConfig`：

- `enable_chunked_prefill`
- `max_num_partial_prefills`
- `max_long_partial_prefills`
- `long_prefill_token_threshold`
- `scheduler_reserve_full_isl`

### 统一 token 预算视角下的 chunked prefill

对于超长 prompt，请求第一次进入系统时，不一定要一次把整个 prompt 都算完。
调度器可以只发一部分 token，让剩余预算留给别的请求或 decode 请求。

这会带来两个直接效果：

| 现象 | 含义 |
|------|------|
| TTFT 可能变长 | 因为一个长 prompt 需要多轮才能 prefill 完 |
| TPOT 更稳定 | 因为 decode 不会被超长 prefill 长时间阻塞 |

### `scheduler_reserve_full_isl` 的意义

这项配置默认是 `True`。
它会让调度器在接纳一个新请求前，先问一句：

> “如果把这个请求完整跑完，KV Cache 总容量到底够不够？”

这样可以避免一种很糟糕的情况：

- chunked prefill 的第一小段看起来能塞进去
- 但完整序列根本放不下
- 结果系统过度接纳，随后频繁抢占和重算

---

## 9.7 当前仓库里真正该调哪些参数？

下面这些参数和当前源码是一一对应的：

```bash
vllm serve model \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 256 \
  --max-num-partial-prefills 2 \
  --max-long-partial-prefills 1 \
  --long-prefill-token-threshold 1024 \
  --scheduling-policy fcfs \
  --async-scheduling \
  --stream-interval 1
```

### 参数解释

| 参数 | 作用 | 调大后的典型影响 |
|------|------|------------------|
| `--max-num-batched-tokens` | 单轮最大 token 预算 | 吞吐更高，但长 prefill 更容易压住 decode |
| `--max-num-seqs` | 单轮可容纳的最大活跃序列数 | 并发更高，但 host 开销和 KV 压力更大 |
| `--max-num-partial-prefills` | 可并发执行的部分 prefill 数 | 长 prompt 更容易被切片穿插 |
| `--max-long-partial-prefills` | 同时允许的长 prompt 部分 prefill 数 | 可防止几个超长 prompt 把系统拖死 |
| `--long-prefill-token-threshold` | 超过这个阈值就按“长 prefill”对待 | 影响 TTFT 与 TPOT 的平衡 |
| `--scheduling-policy` | `fcfs` 或 `priority` | 决定等待队列和抢占对象 |
| `--async-scheduling` | 减少 host 侧空转和调度间隙 | 通常能改善 GPU 利用率 |
| `--stream-interval` | 流式输出粒度 | 1 最丝滑，但 host 负担更高 |

### 两个已经过时的旧参数

如果你在旧文章里看到这些选项，要知道它们并不对应当前仓库的 V1 主路径：

- `--swap-space`
- `--preemption-mode recompute|swap`

---

## 9.8 源码对照：本章该看哪些文件？

| 主题 | 关键文件 | 你会看到什么 |
|------|----------|--------------|
| 调度入口 | `vllm/vllm/v1/engine/core.py` | `EngineCore.step()` 如何调用 scheduler 和 executor |
| 核心调度器 | `vllm/vllm/v1/core/sched/scheduler.py` | token 预算、运行队列、等待队列、抢占 |
| 调度配置 | `vllm/vllm/config/scheduler.py` | 所有用户可调参数的真实定义 |
| 请求状态 | `vllm/vllm/v1/request.py` | `RequestStatus`、`num_computed_tokens` 等字段 |
| 输出汇总 | `vllm/vllm/v1/engine/output_processor.py` | 流式输出如何被整理和返回 |

推荐阅读顺序：

1. 先看 `RequestStatus` 和 `Request` 的关键字段
2. 再看 `Scheduler.schedule()` 顶部注释
3. 接着跟 `self.running` 和 `self.waiting` 两段主循环
4. 最后回到 `EngineCore.step()`，把“调度 -> 执行 -> 更新输出”连起来

---

## 9.9 观察调度行为的实验建议

### 实验 1：对比短 prompt 和长 prompt 竞争

做一组混合压测：

- 一半请求 prompt 长度 64
- 一半请求 prompt 长度 4096

对比以下配置：

1. `--max-num-batched-tokens` 小
2. `--max-num-batched-tokens` 大
3. `--max-num-partial-prefills=1`
4. `--max-num-partial-prefills=2`

观察：

- TTFT
- TPOT
- `vllm:kv_cache_usage_perc`

### 实验 2：观察优先级调度

给部分请求加更高优先级：

```python
client.chat.completions.create(
    model="model-name",
    messages=[{"role": "user", "content": "ping"}],
    extra_body={"priority": 0},
)
```

再和普通请求混跑，观察高优先级请求是否更快进入 `RUNNING`。

---

## 本章小结

| 概念 | 当前仓库中的真实含义 |
|------|----------------------|
| 调度抽象 | 统一 token 预算，而不是硬拆 prefill/decode 两阶段 |
| 请求状态 | `WAITING / RUNNING / PREEMPTED / FINISHED_*` 为主，不是旧的 swapped 模型 |
| 抢占 | 以 `PREEMPTED + 重新调度/重算` 为主 |
| Chunked Prefill | 调度器的一等公民，不是外围优化 |
| 调优重点 | `max_num_batched_tokens`、`max_num_seqs`、partial prefill 相关参数 |

---

## 练习题

### 基础题

1. 为什么说 V1 调度器没有严格的 prefill/decode 两阶段？
2. `num_tokens_with_spec` 和 `num_computed_tokens` 分别表示什么？
3. 当前仓库里的 `PREEMPTED` 和旧资料里的 `swapped` 有什么本质区别？

### 实践题

4. 在 `scheduler.py` 中找到 running 队列和 waiting 队列的两段主循环，并标出各自负责的代码范围。
5. 找到 `_preempt_request(...)` 的调用点，梳理一次失败分配后是如何触发抢占的。

### 思考题

6. 为什么 `scheduler_reserve_full_isl=True` 能减少 KV Cache thrashing？
7. 如果你的业务以超长 prompt 为主，但又非常在意交互式 decode 的稳定 TPOT，应该优先动哪些参数？
