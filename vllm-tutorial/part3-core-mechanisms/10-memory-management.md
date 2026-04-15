# 第10章：内存管理与块引擎

> 如果说 PagedAttention 解决了“显存应该怎样被切块”这个问题，那么当前仓库中的 `KVCacheManager` 解决的就是“这些块到底如何被分配、缓存、回收和复用”。

---

## 学习目标

学完本章，你将能够：

1. 理解当前仓库中 `KVCacheManager` 在 V1 架构里的职责
2. 掌握 KV block pool、free queue、prefix cache 和 request blocks 的关系
3. 跟踪一次请求从命中缓存到分配新块、再到释放块的完整路径
4. 理解为什么当前 V1 不再把 CPU swap 当成主路径
5. 结合源码估算 KV Cache 容量并定位显存瓶颈

---

## 10.1 从 BlockSpaceManager 到 KVCacheManager

如果你读过较早期的 vLLM 资料，可能见过 `BlockSpaceManager` 这个名字。
但当前仓库的 V1 主实现里，真正负责 KV 块生命周期的核心类是：

- `vllm/vllm/v1/core/kv_cache_manager.py` 中的 `KVCacheManager`（对外接口层）
- `vllm/vllm/v1/core/kv_cache_coordinator.py` 中的 `KVCacheCoordinator`（协调多 KV group 的分配策略）
- `vllm/vllm/v1/core/block_pool.py` 中的 `BlockPool` 和 `BlockHashToBlockMap`（底层 block 管理和 prefix cache）
- `vllm/vllm/v1/core/kv_cache_utils.py` 中的 `KVCacheBlock`、`FreeKVCacheBlockQueue` 和 hash 工具

可以把它理解成四层：

```text
Scheduler
  ↓
KVCacheManager        ← 对外接口：get_computed_blocks / allocate_slots / free
  ↓
KVCacheCoordinator    ← 协调多 KV cache group 的分配策略、prefix cache 读写
  ↓
BlockPool             ← 底层 block 管理：block hash map、free queue、LRU 逐出
  ↓
FreeKVCacheBlockQueue ← 双向链表实现的空闲块队列
```

`KVCacheManager` 给调度器暴露的是比较干净的接口：

- `get_computed_blocks(request)`
- `can_fit_full_sequence(request, ...)`
- `allocate_slots(request, ...)`
- `free(request)`
- `reset_prefix_cache()`

这意味着调度器不用直接操心：

- block pool 的空闲链表
- block hash 的维护
- prefix cache 的触碰与逐出
- 多个 KV cache group 的细节

---

## 10.2 启动时，vLLM 怎么决定能放多少 KV Cache？

当前仓库的初始化主路径在：

- `vllm/vllm/v1/engine/core.py`

关键函数是：

- `EngineCore._initialize_kv_caches()`

它大致做这几件事：

1. 调 `model_executor.get_kv_cache_specs()` 询问模型需要哪些 KV cache 规格
2. 调 `model_executor.determine_available_memory()` 估算除模型权重之外还能留多少显存给 KV cache
3. 用 `get_kv_cache_configs(...)` 计算每组 cache 能分成多少个 block
4. 把结果同步回 `vllm_config.cache_config`
5. 调 `model_executor.initialize_from_config(...)` 真正初始化 worker 侧 KV cache

随后，调度器初始化时再创建：

```python
self.kv_cache_manager = KVCacheManager(
    kv_cache_config=kv_cache_config,
    max_model_len=self.max_model_len,
    enable_caching=self.cache_config.enable_prefix_caching,
    use_eagle=self.use_eagle,
    log_stats=self.log_stats,
    enable_kv_cache_events=self.enable_kv_cache_events,
    dcp_world_size=self.dcp_world_size,
    pcp_world_size=self.pcp_world_size,
    hash_block_size=self.block_size,
    metrics_collector=self.kv_metrics_collector,
)
```

`KVCacheManager` 内部会通过 `get_kv_cache_coordinator(...)` 工厂函数，根据是否启用 prefix caching 和 KV cache group 数量，选择不同的 `KVCacheCoordinator` 实现。

### 直觉公式

虽然源码实际还要考虑：

- 多个 KV cache group
- attention 类型差异
- context parallel
- hybrid attention / mamba cache

但粗略估算仍可以先用这个公式：

```text
可用 KV 显存
= GPU 总显存 * gpu_memory_utilization - 模型权重/激活/工作区

总 block 数
= 可用 KV 显存 / 每个 block 的字节数
```

一旦总 block 数确定，系统能容纳的并发上限，本质上就受它约束。

---

## 10.3 当前源码里的核心数据结构

### 1. `KVCacheBlock`

在 `vllm/vllm/v1/core/kv_cache_utils.py` 里，`KVCacheBlock` 是最底层的元数据单元。

它记录的字段包括：

- `block_id`：全局唯一标识，范围 0 ~ num_gpu_blocks-1
- `ref_cnt`：引用计数
- `_block_hash`：类型为 `BlockHashWithGroupId | None`，只有完整且已缓存的 block 才有
- `prev_free_block / next_free_block`：双向链表指针，只由 `FreeKVCacheBlockQueue` 操作
- `is_null`：标记 null block（不应被缓存的特殊 block）

也就是说，一个 block 既是：

- “某一段 KV cache 的身份”
- 又是 prefix cache 的可缓存对象
- 同时还是 free queue 里的链表节点

### 2. BlockPool

当前仓库已把 block pool 抽取为独立模块 `vllm/vllm/v1/core/block_pool.py`。

`BlockPool` 管理着：

- `FreeKVCacheBlockQueue`：空闲块队列
- `BlockHashToBlockMap`：hash → block 的缓存映射（用于 prefix caching）
- block 的分配、缓存、逐出和释放

启动时会一次性创建整池 block，而不是按需 new Python 对象。
这样做的原因很工程化：

- 少 GC 压力
- 元数据总量固定
- 空闲/占用/缓存状态都好追踪

### 3. free queue

当前仓库没有简单地用 `deque` 来存空闲块，而是把双向链表指针直接挂在 `KVCacheBlock` 上。
这样可以做到：

- O(1) 从中间移除某个 block
- O(1) 追加到队尾

这对 prefix cache 的 LRU 逐出尤其关键。

### 4. request blocks

每个请求都会在管理器里对应一组 block 列表。
调度器看到的是 `KVCacheBlocks` 这种更高层的包装，用它来拿 block id、拼接块组等。

### 5. cache blocks

如果某个 block 已经“完整填满”，并且 prefix caching 打开，它就可能进入 cache map，供后续请求复用。

---

## 10.4 一次请求的块生命周期

### 阶段 1：先查能不能复用

当新请求第一次进入调度器时，waiting 流程会先调用：

```python
computed_blocks, num_cached_tokens = kv_cache_manager.get_computed_blocks(request)
```

这一步的含义是：

1. 对请求已有 token 按 block 计算 hash
2. 找最长的前缀命中
3. 返回已经算好的完整 block

当前仓库里有两个关键约束：

- **只有完整 block 才会进入 prefix cache**
- **即使整段 prompt 都命中，最后一个 token 仍常常要重算一次来拿 logits**

这正是 `get_computed_blocks()` 里把最大 cache hit 长度限制为 `prompt_length - 1` 的原因。

### 阶段 2：为真正需要推进的 token 申请 slot

真正的重头戏在 `KVCacheManager.allocate_slots(...)`。源码里有一段非常直观的 block layout 注释，展示了不同 token 段的关系：

```text
----------------------------------------------------------------------
| < comp > | < new_comp > | < ext_comp >  | < new >  | < lookahead > |
----------------------------------------------------------------------
                                          |   < to be computed >     |
----------------------------------------------------------------------
                          |            < to be allocated >           |
----------------------------------------------------------------------
                          | < to be cached (roughly) >  |
----------------------------------------------------------------------
```

各缩写含义：

| 缩写 | 含义 |
|------|------|
| `comp` | `request.num_computed_tokens`，已经算过并持有 block 的 token |
| `new_comp` | `num_new_computed_tokens`，本轮 prefix cache 新命中的 token |
| `ext_comp` | `num_external_computed_tokens`，KV connector 远端命中的 token |
| `new` | `num_new_tokens`，本轮要真正计算的新 token（含未验证的 draft token）|
| `lookahead` | `num_lookahead_tokens`，speculative decoding 的前瞻 token |

### 分配的三个阶段

源码把 `allocate_slots` 拆成明确的三个阶段：

**阶段 A：释放不再需要的旧块**

比如在 sliding window 场景下，已经滑出窗口的 block 不再被注意力访问，可以提前释放来腾出空间。

**阶段 B：处理 prefix token（comp + new_comp + ext_comp）**

- 把新命中的缓存 block 接到请求上（通过 `coordinator.allocate_new_computed_blocks`）
- 为远端 KV（ext_comp）分配对应 block
- 释放 sliding window 之外的不需要的 prefix block

**阶段 C：为要计算的 token 分配新块（new + lookahead）**

- 调用 `coordinator.allocate_new_blocks` 从 `BlockPool` 的空闲队列获取新 block
- 如果空闲块不够，返回 `None`（调度器据此触发抢占）
- 如果有新的完整 block 生成，通过 `coordinator.cache_blocks()` 写入 prefix cache

这个接口虽然名字简单，实际上是 V1 内存管理的汇合点——它统一处理本地缓存、远端缓存、speculative decoding 和 encoder-decoder cross-attention 的 block 分配。

---

## 10.5 KVCacheCoordinator：三种策略

当前仓库的 `get_kv_cache_coordinator()` 工厂函数会根据配置选择三种不同的 coordinator：

| Coordinator | 适用场景 | 特点 |
|-------------|---------|------|
| `KVCacheCoordinatorNoPrefixCache` | prefix caching 未启用 | 最简单，不做 hash、不查缓存 |
| `UnitaryKVCacheCoordinator` | 只有 1 个 KV cache group | 纯 attention 模型的标准路径 |
| `HybridKVCacheCoordinator` | 多个 KV cache group | 混合模型（如 attention + mamba）|

选择逻辑：

```python
if not enable_caching:
    return KVCacheCoordinatorNoPrefixCache(...)
if len(kv_cache_config.kv_cache_groups) == 1:
    return UnitaryKVCacheCoordinator(...)
return HybridKVCacheCoordinator(...)
```

每个 coordinator 内部持有一组 `SingleTypeKVCacheManager`，每个 manager 负责一种 KV cache group 的 block 管理。coordinator 负责协调多个 manager 之间的一致性。

### 阶段 3：请求完成后释放

请求结束时，调度器会走到：

```python
kv_cache_manager.free(request)
```

释放时有两个关键点：

1. 不是所有 block 都会立刻消失，只有 `ref_cnt == 0` 的才真正回收
2. 回收到 free queue 时，通常按“尾块优先逐出”的思路反向挂回

这能让：

- 更短命、更不可能被复用的尾块
- 在未来更早被 LRU 逐出

---

## 10.6 前缀缓存在当前实现里怎么落地？

本章重点是内存管理，但前缀缓存已经和它绑死了，所以必须一起看。

### 当前仓库的 block hash 由什么组成？

依据 `docs/design/prefix_caching.md` 和 `kv_cache_utils.py`，一个 block hash 不只是“这一块的 token”：

- 父 block 的 hash
- 当前 block 的 token
- 额外哈希信息

额外信息可能包括：

- LoRA ID
- 多模态输入 hash
- `cache_salt`

所以它更像：

```text
block_hash = hash(parent_hash, block_tokens, extra_hashes)
```

这能保证：

- 同一个 block 内容，但前缀不同，不会误命中
- 多租户环境下，不同 salt 的请求不会共享缓存

### 只缓存完整块

这是理解 prefix caching 命中率的关键：

- 前缀虽然很长
- 但如果最后一段只填满了半个 block
- 那半块不会被缓存，也不会被后续请求直接复用

所以命中往往是“按完整 block 的最长公共前缀”命中，而不是按 token 逐个命中。

### V1 的一个实现细节：append-only block table

当前 V1 的 block table 设计偏向 append-only。
这带来一个实际后果：

- 如果一个新请求又生成出一个内容完全重复的完整 block
- 它短时间内可能和旧 block 同时存在
- 重复块通常要到请求释放时才被彻底消掉

这也是官方 prefix caching 设计文档专门解释的一点。

---

## 10.7 抢占时，当前 V1 为什么不走 CPU swap？

这是本章最需要“用源码纠偏”的地方。

旧资料常见叙述：

- 块管理器同时维护 GPU 块池和 CPU swap 块池
- OOM 时先 swap out，恢复时再 swap in

但当前仓库的 V1 主路径不是这样。

### 真实情况

当 `allocate_slots()` 返回 `None` 时：

- 调度器会触发抢占
- 被抢占请求标记为 `PREEMPTED`
- 其相关 KV block 被释放
- 后续再次调度时，通过重新计算和 prefix cache 复用来恢复进度

也就是说：

- **V1 主线是 recompute-oriented**
- **不是本地 CPU swap-oriented**

官方 `docs/usage/v1_guide.md` 也明确把：

- `GPU <> CPU KV Cache Swapping`

标成了 removed feature。

### 这不意味着仓库里没有 offload 代码

当前仓库仍然有：

- `distributed/kv_transfer/...`
- `v1/simple_kv_offload/...`

但这些更多是：

- connector
- 远端 KV 传输
- disaggregated / offload 场景

不能把它们简单等同于“旧教程里的 swapped 队列”。

---

## 10.8 如何估算容量？

### 一个够用的工程估算式

```python
def estimate_max_concurrent(
    gpu_memory_gb: float,
    model_memory_gb: float,
    gpu_memory_utilization: float,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    avg_seq_len: int,
    dtype_bytes: int = 2,
) -> int:
    available_gb = gpu_memory_gb * gpu_memory_utilization - model_memory_gb
    block_gb = (
        2 * num_layers * num_kv_heads * head_dim * block_size * dtype_bytes
    ) / (1024**3)
    total_blocks = int(available_gb / block_gb)
    blocks_per_req = (avg_seq_len + block_size - 1) // block_size
    return total_blocks // blocks_per_req
```

这个估算忽略了：

- hybrid attention
- 多 group cache
- encoder cache
- lookahead/spec decode

但对理解“为什么显存一满系统就开始频繁 preempt”已经很够用了。

### 线上最值得盯的几个指标

当前仓库文档 `docs/design/metrics.md` 里明确列出了这些指标：

- `vllm:kv_cache_usage_perc`
- `vllm:prefix_cache_queries`
- `vllm:prefix_cache_hits`

如果你发现：

- `kv_cache_usage_perc` 长期接近 1
- 同时 TTFT 和 TPOT 都开始恶化

那大概率不是“模型变慢了”，而是 KV Cache 容量已经在逼近极限。

---

## 10.9 源码阅读地图

| 主题 | 关键文件 | 重点关注 |
|------|----------|----------|
| 启动时计算 KV 容量 | `vllm/vllm/v1/engine/core.py` | `_initialize_kv_caches()` |
| 内存管理外观接口 | `vllm/vllm/v1/core/kv_cache_manager.py` | `get_computed_blocks` / `allocate_slots` / `free` |
| 协调器层 | `vllm/vllm/v1/core/kv_cache_coordinator.py` | `KVCacheCoordinator` 及其子类 |
| block pool 和 prefix cache | `vllm/vllm/v1/core/block_pool.py` | `BlockPool`、`BlockHashToBlockMap` |
| block 元数据和 free queue | `vllm/vllm/v1/core/kv_cache_utils.py` | `KVCacheBlock`、`FreeKVCacheBlockQueue` |
| 前缀缓存设计文档 | `vllm/docs/design/prefix_caching.md` | hash 组成、LRU、append-only block table |
| 调度器如何使用内存管理 | `vllm/vllm/v1/core/sched/scheduler.py` | waiting/running 两段分配逻辑 |

推荐顺序：

1. 先看 `KVCacheBlock`（kv_cache_utils.py）
2. 再看 `BlockPool`（block_pool.py），理解底层 block 管理
3. 然后看 `KVCacheCoordinator`（kv_cache_coordinator.py），理解多 group 协调
4. 再看 `KVCacheManager.allocate_slots()`，理解对外接口
5. 接着看 `Scheduler.schedule()` 在 waiting 路径里怎么调用它
6. 最后看 prefix caching 设计文档，把 hash 和 LRU 的设计补上

---

## 本章小结

| 概念 | 当前仓库中的真实实现 |
|------|----------------------|
| 内存管理核心类 | `KVCacheManager` → `KVCacheCoordinator` → `BlockPool` 三层架构 |
| 空闲块管理 | `BlockPool` + `FreeKVCacheBlockQueue`（双向链表） |
| 前缀缓存 | 只缓存完整 block，通过 `BlockHashToBlockMap` 按 hash 复用 |
| 抢占恢复 | `PREEMPTED + recompute` 为主 |
| 关键瓶颈 | block 数不够时，会直接传导到调度器抢占和 TTFT/TPOT 波动 |

---

## 练习题

### 基础题

1. 当前仓库里负责 KV 块生命周期管理的核心类叫什么？
2. 为什么 free queue 要用双向链表而不是简单 `deque`？
3. 为什么 prefix cache 只缓存完整 block？

### 实践题

4. 在 `kv_cache_manager.py` 中找到 `allocate_slots()`，标出“检查空闲块是否足够”和“写回 cache_blocks”分别发生在什么位置。
5. 在 `kv_cache_utils.py` 中找到 `KVCacheBlock` 和 `FreeKVCacheBlockQueue`，梳理 block 被回收到空闲队列后的 LRU 语义。

### 思考题

6. 为什么 V1 选择让调度器围绕 recompute 而不是本地 CPU swap 建模？
7. 如果 `kv_cache_usage_perc` 很高但 `prefix_cache_hits` 很低，你更应该先调 prompt 结构、调度参数，还是直接加显存？
