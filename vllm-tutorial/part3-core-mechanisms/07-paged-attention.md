# 第7章：PagedAttention

> PagedAttention 是 vLLM 的核心创新。它用一个简单而优雅的类比——操作系统的虚拟内存分页——解决了 LLM 推理中最棘手的显存管理问题。

---

## 学习目标

学完本章，你将能够：

1. 理解 PagedAttention 的设计动机和它解决的核心问题
2. 掌握逻辑块、物理块、块表的概念及其映射关系
3. 理解 PagedAttention 如何消除显存碎片和预分配浪费
4. 理解 Copy-on-Write 机制如何支持并行采样和共享前缀
5. 定量分析 PagedAttention 带来的显存效率提升

---

## 7.1 从操作系统虚拟内存说起

### 操作系统面临的问题

在操作系统中，多个进程需要共享物理内存。如果每个进程直接使用物理地址：

- 进程不知道其他进程占了哪些内存
- 内存分配和释放后会产生碎片
- 无法高效地共享内存页面

### 操作系统的解决方案：虚拟内存

操作系统的核心思想是引入一个**间接层**：

- 每个进程有自己的**虚拟地址空间**（连续的）
- 虚拟地址通过**页表**映射到**物理页帧**（可以不连续）
- 物理页帧按固定大小分配（通常 4KB）
- 按需分配，用到时才分配物理页帧

### vLLM 面临的类似问题

在 LLM 推理中：

- 多个请求需要共享 GPU 显存来存储 KV Cache
- 请求的长度不同且动态增长
- 预分配导致浪费，释放导致碎片
- 共享前缀时 KV Cache 被重复存储

**PagedAttention 的核心洞察：KV Cache 的管理问题与物理内存的管理问题本质相同，可以用相同的方案解决。**

---

## 7.2 PagedAttention 的核心概念

### 逻辑块与物理块

PagedAttention 将 KV Cache 分成固定大小的**块（block）**：

- **逻辑块**：从请求的视角看到的连续 KV Cache 空间
- **物理块**：GPU 显存中实际存储 KV 数据的固定大小区域
- **块大小**：每个块存储固定数量的 token 的 KV 向量（默认 16 个 token）

```
一个物理块的结构（以 block_size=16 为例）：

┌────────────────────────────────────────────┐
│  K vectors: [k₁, k₂, k₃, ..., k₁₆]       │
│  V vectors: [v₁, v₂, v₃, ..., v₁₆]       │
│                                            │
│  大小 = 2 × num_kv_heads × head_dim × 16 × dtype_size  │
└────────────────────────────────────────────┘
```

### 块表（Block Table）

块表维护每个请求的逻辑块到物理块的映射：

```
请求 A 的块表:
  逻辑块 0 → 物理块 7
  逻辑块 1 → 物理块 3
  逻辑块 2 → 物理块 12

请求 B 的块表:
  逻辑块 0 → 物理块 1
  逻辑块 1 → 物理块 9
```

关键特性：
- 逻辑块是连续的（0, 1, 2, ...）
- 物理块可以不连续（散布在显存各处）
- 映射关系通过块表维护

### 与操作系统的对应关系

| 操作系统 | PagedAttention | 作用 |
|----------|---------------|------|
| 虚拟页面 | 逻辑块 | 进程/请求看到的连续空间 |
| 物理页帧 | 物理块 | 实际存储位置 |
| 页表 | 块表 | 虚拟到物理的映射 |
| 页面大小（4KB） | 块大小（16 tokens） | 分配的最小单位 |
| 按需分页 | 按需分配块 | 用到时才分配 |
| 页面共享 | 块共享（CoW） | 多个进程/请求共享同一物理页/块 |

---

## 7.3 按需分配：消除预分配浪费

### 朴素方法的浪费

传统做法在请求开始时，按 max_seq_len 一次性分配所有 KV Cache：

```
请求开始时（max_seq_len=2048, 实际只需 200 tokens）：

朴素分配: [████████████████████████████████████████████████████]
           ↑ 预分配 2048 tokens 的空间

实际使用: [███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
           ↑ 只用了 200 tokens

浪费率: 90%
```

### PagedAttention 的按需分配

PagedAttention 只在需要时才分配新的物理块：

```
Token 1-16:   分配物理块 7     [████████████████]
Token 17-32:  分配物理块 3     [████████████████]
Token 33-40:  分配物理块 12    [████████░░░░░░░░]  ← 最后一个块可能未满
...生成结束

浪费: 只有最后一个块的内部碎片（最多 block_size - 1 个 token）
```

### 浪费率分析

对于块大小为 B 的 PagedAttention：

- 每个请求最多浪费 B-1 个 token 的空间
- 平均浪费 (B-1)/2 个 token
- 以 B=16 为例，平均浪费 7.5 个 token

如果请求平均长度为 200 token：
- 朴素方法（max_len=2048）：浪费 90%
- PagedAttention（B=16）：浪费 7.5/200 = 3.75%

**从 90% 降到 4%——这就是 PagedAttention 最直接的价值。**

---

## 7.4 不连续存储：消除外部碎片

### 朴素方法的碎片

传统方法要求每个请求的 KV Cache 在显存中连续存储：

```
分配后：[AAAA][BBBB][CCCC][DDDD]

释放 B 和 D 后：[AAAA][    ][CCCC][    ]

新请求 E 需要 6 个单位：
虽然空闲空间 = 8 个单位，但最大连续空间只有 4 个单位
→ 无法分配！（外部碎片）
```

### PagedAttention 消除碎片

物理块可以分散在显存中的任何位置：

```
显存物理布局：[A₁][C₁][ ][A₂][ ][C₂][ ][ ]

新请求 E 需要 3 个块：
直接分配空闲块：[A₁][C₁][E₁][A₂][E₂][C₂][E₃][ ]
→ 成功！无需连续空间
```

因为注意力计算通过块表间接访问 KV Cache，物理上是否连续不影响计算正确性。

---

## 7.5 Copy-on-Write：高效共享

### 场景 1：并行采样（n>1）

当一个请求需要生成多个结果时（如 `n=3`），多个采样序列共享相同的 prompt KV Cache。

```
请求："写一首诗" → 生成 3 个不同版本

Prompt 的 KV Cache 是相同的，无需复制：

序列 1: [共享prompt块][独立块1a][独立块1b]
序列 2: [共享prompt块][独立块2a]
序列 3: [共享prompt块][独立块3a][独立块3b][独立块3c]
         ↑ 同一个物理块，引用计数=3
```

### Copy-on-Write 机制

当多个序列共享同一个物理块，且其中一个需要修改该块时：

1. 检查物理块的引用计数
2. 如果引用计数 > 1，先复制一份新的物理块
3. 在新块上修改
4. 减少原块的引用计数

```
初始状态（共享块 7）：
序列 1 → 块 7 (ref=2)
序列 2 → 块 7 (ref=2)

序列 1 需要修改块 7 中的内容：
1. ref_count(块7) = 2 > 1 → 触发 CoW
2. 分配新块 15，复制块 7 的内容
3. 序列 1 → 块 15 (ref=1)，序列 2 → 块 7 (ref=1)
```

### 场景 2：前缀共享

多个请求使用相同的系统 prompt 时，可以共享前缀的 KV Cache 块：

```
系统 prompt: "你是一个有帮助的助手..."  → KV Cache 占 3 个块

请求 A: [系统prompt块0][系统prompt块1][系统prompt块2][用户A的块...]
请求 B: [系统prompt块0][系统prompt块1][系统prompt块2][用户B的块...]
请求 C: [系统prompt块0][系统prompt块1][系统prompt块2][用户C的块...]
         ↑ 3个物理块被 3 个请求共享，节省 6 个块的显存
```

---

## 7.6 PagedAttention Kernel

PagedAttention 需要自定义的注意力 CUDA kernel 来处理非连续的 KV Cache 存储。

### 标准注意力 vs PagedAttention

**标准注意力**：

```python
# K, V 在连续内存中
# K: [batch, num_heads, seq_len, head_dim]
# V: [batch, num_heads, seq_len, head_dim]
attention = softmax(Q @ K.T / sqrt(d)) @ V
```

**PagedAttention**：

```python
# K, V 分散在物理块中
# 需要通过块表找到实际位置
for each query token:
    for each block in block_table:
        k_block = physical_blocks[block_table[block_idx]]
        v_block = physical_blocks[block_table[block_idx]]
        # 对这个块中的 token 计算注意力
        scores += query @ k_block.T
    # softmax 和 V 加权
```

### 实际实现的优化

vLLM 的 PagedAttention kernel 做了大量优化：

1. **分块计算**：每个 CUDA thread block 处理一个查询 token 对一个 KV 块的注意力
2. **在线 softmax**：使用数值稳定的在线算法，避免额外的 pass
3. **向量化内存访问**：合并显存读取，最大化带宽利用
4. **支持 GQA**：正确处理 Query 头和 KV 头的多对一映射

### 在 V1 源码中的落点

当前仓库里，PagedAttention 的算子层实现在 `vllm/vllm/v1/attention/ops/paged_attn.py`：

```python
class PagedAttention:
    @staticmethod
    def split_kv_cache(kv_cache, num_kv_heads, head_size):
        """将合并的 KV cache 张量拆成 key_cache 和 value_cache"""
        x = 16 // kv_cache.element_size()
        key_cache = kv_cache[0].view(num_blocks, num_kv_heads, head_size // x, -1, x)
        value_cache = kv_cache[1].view(num_blocks, num_kv_heads, head_size, -1)
        return key_cache, value_cache

    @staticmethod
    def write_to_paged_cache(key, value, key_cache, value_cache, slot_mapping, ...):
        """把新生成的 KV 向量写入 paged cache 的指定 slot"""
        ops.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping, ...)
```

`split_kv_cache` 展示了 KV cache 的真实物理布局——key 的最内层维度做了 `16 / element_size()` 的向量化拆分（`x` 维度），这正是为了优化 CUDA kernel 的内存访问模式。

### GPU 侧的 Block Table 管理

更上层的 block table 管理在 `vllm/vllm/v1/worker/gpu/block_table.py` 中的 `BlockTables` 类：

```python
class BlockTables:
    def __init__(self, block_sizes, max_num_reqs, max_num_batched_tokens, ...):
        # 每个 KV cache group 一张 block table: [max_num_reqs, max_num_blocks]
        self.block_tables: list[StagedWriteTensor] = [...]
        # slot_mapping: [num_kv_cache_groups, max_num_batched_tokens]
        self.slot_mappings = torch.zeros(...)
```

这里能看到两个工程细节：

1. **多 KV cache group 支持**：block table 不是一张，而是每个 KV cache group 一张（用于混合 attention+mamba 模型）
2. **StagedWriteTensor**：block table 使用分段写入策略，在 CPU 侧准备数据后批量传输到 GPU

### 注意力后端不只是 PagedAttention

当前 V1 的注意力后端已经远超原始 PagedAttention kernel。真正被调用的是 `v1/attention/backends/` 下的高级后端：

- **Flash Attention**（`flash_attn.py`）：使用 `reshape_and_cache_flash` 写 KV cache，然后调 `flash_attn_varlen_func` 做注意力计算
- **FlashInfer**（`flashinfer.py`）：更新版的高性能后端
- **MLA 后端族**（`mla/`）：DeepSeek 的 Multi-head Latent Attention

这些后端内部都包含了 paged KV cache 的写入和读取逻辑，但使用了更高效的融合 kernel 而非原始的分步 PagedAttention kernel。

---

## 7.7 块大小的选择

### 块大小的权衡

| 块大小 | 优势 | 劣势 |
|--------|------|------|
| 小（如 1） | 几乎零内部碎片 | 块表更大，管理开销高 |
| 中（如 16） | 碎片和开销平衡 | 默认选择 |
| 大（如 64） | 管理开销低 | 内部碎片增大 |

### vLLM 的默认选择

vLLM 默认使用 `block_size=16`，这在大多数场景下是一个好的平衡点。

```python
# 修改块大小（通常不需要）
llm = LLM(model="...", block_size=16)
```

### 块大小对显存效率的影响

```python
# 模拟不同块大小下的显存浪费
import random

num_requests = 1000
seq_lengths = [random.randint(50, 2000) for _ in range(num_requests)]

for block_size in [1, 4, 16, 32, 64]:
    total_allocated = sum(
        ((s + block_size - 1) // block_size) * block_size
        for s in seq_lengths
    )
    total_actual = sum(seq_lengths)
    waste = (total_allocated - total_actual) / total_allocated * 100
    print(f"Block size = {block_size:2d}: waste = {waste:.1f}%")
```

---

## 本章小结

| 概念 | 要点 |
|------|------|
| 核心类比 | KV Cache 管理 ≈ 操作系统虚拟内存管理 |
| 逻辑块/物理块 | 请求看到连续空间，实际可以不连续存储 |
| 块表 | 维护逻辑块到物理块的映射 |
| 按需分配 | 生成过程中按需分配新块，消除预分配浪费 |
| 消除外部碎片 | 物理块不需要连续，任何空闲块都可分配 |
| Copy-on-Write | 多序列共享块，修改时才复制 |
| 显存浪费 | 从 60-80% 降至 < 4% |

---

## 动手实验

### 实验 1：可视化块分配

```python
# 模拟 PagedAttention 的块分配过程
class SimpleBlockAllocator:
    def __init__(self, num_blocks, block_size=16):
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))
        self.block_tables = {}  # request_id -> [physical_block_ids]

    def allocate_request(self, request_id, num_tokens):
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        if len(self.free_blocks) < num_blocks_needed:
            raise RuntimeError("OOM: not enough free blocks")
        allocated = []
        for _ in range(num_blocks_needed):
            block = self.free_blocks.pop(0)
            allocated.append(block)
        self.block_tables[request_id] = allocated
        return allocated

    def free_request(self, request_id):
        blocks = self.block_tables.pop(request_id)
        self.free_blocks.extend(blocks)

    def status(self):
        used = sum(len(v) for v in self.block_tables.values())
        free = len(self.free_blocks)
        print(f"Used: {used}, Free: {free}, Requests: {len(self.block_tables)}")

# 模拟
allocator = SimpleBlockAllocator(num_blocks=100, block_size=16)

allocator.allocate_request("A", num_tokens=200)  # 需要 13 个块
allocator.allocate_request("B", num_tokens=50)   # 需要 4 个块
allocator.status()

allocator.free_request("A")
allocator.status()

allocator.allocate_request("C", num_tokens=300)  # 可以使用 A 释放的块
allocator.status()
```

### 实验 2：对比浪费率

计算 1000 个随机长度请求在朴素分配和 PagedAttention 下的显存浪费率。

---

## 练习题

### 基础题

1. PagedAttention 的"Page"借鉴了操作系统的什么概念？
2. 逻辑块和物理块有什么区别？为什么需要这个间接层？
3. Copy-on-Write 在什么场景下被触发？

### 实践题

4. 用 Python 模拟一个块分配器，实现 allocate、free 和 status 方法。
5. 计算 block_size 分别为 1、16、64 时，对于平均长度 500 token 的请求，内部碎片的浪费率。

### 思考题

6. PagedAttention 使得 KV Cache 在物理上不连续，这对注意力计算的 CUDA kernel 有什么影响？性能开销大吗？
7. 如果所有请求的长度都完全相同，PagedAttention 相比朴素分配还有优势吗？
