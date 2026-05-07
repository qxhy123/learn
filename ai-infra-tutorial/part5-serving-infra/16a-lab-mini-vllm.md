# 第 16a-lab 章 · Mini-vLLM 实战:从零写一个带 PagedAttention 的迷你推理引擎

> 上一章[16a vLLM 内部机制](16a-vllm-internals.md)从工程师视角拆解了 vLLM 的设计取舍。本章把这些取舍**变成可读、可跑、可改的代码**:在 `code/mini-vllm/` 下的几百行 Python 里,从零实现 PagedAttention、continuous batching、prefix caching、chunked prefill、swap、streaming 和完整 sampler,并与 HF transformers 在 TinyLlama-1.1B 上做位级精确对拍。
>
> **目标读者:** 已经读过 [第 15 章](15-batching-scheduling-and-kv-cache.md) 和 [16a 主章](16a-vllm-internals.md),想"动手把这些机制串一遍"的人。本章不替代主章的概念解释,它是**带读者一步步加 feature 的实战手册**。
>
> **不期待:** 性能与真实 vLLM 对标(我们用纯 PyTorch reference attention,慢得多)。期待的是看清每个机制如何在最小可运行代码里成立。

---

## 配套代码

源码在 `ai-infra-tutorial/code/mini-vllm/`,跟随教程仓库一起 clone。安装与跑通:

```bash
cd code/mini-vllm
pip install -e ".[dev]"
python examples/run_toy.py            # toy GPT,随机权重,秒跑通
python examples/run_tinyllama.py      # TinyLlama-1.1B,首次下载 ~2.2 GB
pytest tests/ -v                      # 54 个快速测试
pytest tests/ -m slow -v              # 加上 HF parity(下载权重)
python examples/bench.py              # 5 配置 throughput 对比表
```

代码结构(已在 `code/mini-vllm/README.md` 列出):

```
mini_vllm/
├── config.py                 # ModelConfig / EngineConfig / SamplingParams / CacheConfig
├── sequence.py               # Sequence (per-request state)
├── block_manager.py          # GPU+CPU pools, hash chain, ref count, swap, CoW
├── cache_engine.py           # KV tensor pools (GPU + optional CPU)
├── scheduler.py              # 三队列 + chunked prefill + 自动 swap-out
├── model_runner.py           # ModelInput 组装,调用 model.forward
├── engine.py                 # 顶层 step / generate / generate_stream
├── sampler.py                # greedy / temperature / top-p / top-k
├── tokenizer.py              # 包 HF tokenizers
├── backends/{interface,reference,torch_backend}.py    # AttentionBackend
└── models/{base,toy_gpt,llama,llama_loader}.py        # ToyGPT + Llama + HF loader
```

## 路线图

我们按 feature 渐进的顺序展开。**读者跑的是同一份代码**,通过 `EngineConfig` 的 4 个 flag 切换形态:

| 节 | feature | flag | 关闭后退化为 |
|---|---|---|---|
| §2 | block-based KV cache + 一次性 prefill | (始终开) | — |
| §3 | PagedAttention kernel(reference 实现) | (始终开) | — |
| §4 | TinyLlama 1.1B + HF safetensors loader | (模型层) | toy GPT |
| §5 | continuous batching | `enable_continuous_batching` | running 队列空才能接新请求 |
| §6 | chunked prefill | `enable_chunked_prefill` | 长 prompt 一次吃完 |
| §7 | prefix caching + CoW | `enable_prefix_caching` | 每个 prompt 从零 prefill |
| §8 | swap to CPU + LRU 抢占 | `enable_swap` + `num_cpu_blocks` | 显存满则 raise |
| §9 | 从 mini 到真实 vLLM | — | — |

每节末尾 2-3 个练习,鼓励改代码自己验证。

---

## §1. 为什么需要 PagedAttention

### Naive 实现的显存碎片问题

如果你不用 PagedAttention,KV Cache 自然的写法是这样:

```python
# 每个请求一个连续 (max_seq_len, num_heads, head_dim) 的 K/V 张量
kv_cache = torch.zeros(max_seq_len, num_heads, head_dim, ...)
```

这个写法在静态 batch + 等长请求时没问题。但服务真实流量时,你立刻撞到三堵墙:

1. **过度预留**:你不知道这个请求会生成多长(可能 32 token,也可能 2048 token)。如果按 max_seq_len 分配,99% 的空间永远闲置。
2. **碎片化**:多请求并发时,有的占满 max_seq_len,有的只用了 50 token,但中间空间被锁住,新请求无法塞进来。
3. **共享前缀失效**:100 个请求都用相同的 system prompt(比如 "You are a helpful assistant..."),那段 K/V 被算了 100 遍,显存里也存了 100 份。

vLLM 的 PagedAttention 一句话:**把 KV Cache 切成固定大小的 block(默认 16 token),让 attention kernel 通过 block table 间接寻址,允许同一个 logical sequence 的 K/V 散落在不连续的物理 block 上。**

这跟操作系统的虚拟内存几乎一一对应:

| OS 虚拟内存 | PagedAttention |
|---|---|
| 进程 | 一个生成请求(Sequence) |
| 虚拟地址 | seq 内的 token 位置(0, 1, 2, ...) |
| 物理页 | 物理 block(`PhysicalBlock` with `block_id`) |
| 页表 | `BlockTable`(逻辑顺序的 block 列表) |
| 页大小 | `block_size`(默认 16) |
| Page sharing(COW) | 引用计数 + CoW 写时复制 |
| Swap | swap-to-CPU |

**这就是为什么我们的 `BlockManager` 长得像内存分配器,`Scheduler` 长得像进程调度器。**

### 练习

- E1.1 — 估算:Llama-3-70B,vocab=128K,num_layers=80,num_kv_heads=8,head_dim=128,fp16。一个请求 max_seq_len=8K 时,KV Cache 单请求占多少 MB?如果按 max_seq_len 静态预分配,80 GB A100 单卡最多并发多少请求?
- E1.2 — 假设 100 个请求平均长度 256 token,但 max_seq_len=8K。静态分配 vs PagedAttention(block_size=16)显存利用率差多少?

---

## §2. 第一版:Block KV Cache + Naive Scheduler

### `BlockManager`(`mini_vllm/block_manager.py`)

最小可工作的 BlockManager 只需要四个方法:

```python
class BlockManager:
    def can_allocate(self, seq) -> AllocStatus:           # OK / LATER / NEVER
    def allocate(self, seq) -> BlockTable                  # 一次性分配 prompt 所需的所有 block
    def append_slot(self, seq) -> Optional[Tuple[int,int]] # decode 时按需扩 1 个 block;返回 CoW 映射
    def free(self, seq) -> None                            # 引用计数 -1,归 0 回池
```

`PhysicalBlock` 一开始就声明了 `ref_count` / `block_hash` / `device` 三个字段 —— Plan 1 用不到,但让 Plan 5(prefix caching)和 Plan 6(swap)不需要改数据结构:

```python
@dataclass
class PhysicalBlock:
    block_id: int
    ref_count: int = 1
    block_hash: Optional[int] = None     # Plan 5 fills
    device: str = "gpu"                  # Plan 6 toggles
```

**关键设计 1:slot mapping 从抽象 token 位置到物理写入地址**。每个 token 写到哪个槽位,由 `get_slot_mapping()` 算:

```python
def get_slot_mapping(self, seq, start, end):
    mapping = []
    for pos in range(start, end):
        block_idx = pos // self.block_size
        offset = pos % self.block_size
        pb = seq.block_table.physical_blocks[block_idx]
        mapping.append(pb.block_id * self.block_size + offset)
    return mapping
```

返回的整数 `block_id * block_size + offset` 是物理 K/V tensor 的扁平索引。这个抽象让 attention backend 完全不需要知道 BlockTable 的内部布局。

**关键设计 2:`KV pool` 与 `BlockManager` 解耦**。`CacheEngine`(`mini_vllm/cache_engine.py`)只负责持有大张量:

```python
shape = (num_blocks, num_kv_heads, head_dim, block_size)
key_cache = torch.zeros(shape, ...)
value_cache = torch.zeros(shape, ...)
```

注意最后一维是 `block_size`(不是 `num_blocks` 或 `head_dim`)—— 这让"读一整个 block 的 K/V"变成 contiguous load。真实 vLLM 在 CUDA kernel 里依赖这个布局做 cache-line 友好读取。

### Scheduler(Plan 1 形态)

最简调度只有两个队列(`waiting` + `running`),且**不做 continuous batching**:

```python
def schedule(self):
    # 1. 把所有 running 的 seq 放进 prefill_seqs 或 decode_seqs
    # 2. 仅当 running 为空时,从 waiting 队首拉一批进 running
    if not self.running:
        while self.waiting and bm.can_allocate(self.waiting[0]) == OK:
            ...
```

这相当于 vLLM 的 `enable_continuous_batching=False` 退化模式。**它跑得通,但显存利用率差** —— 一旦有请求生成超长序列,新请求就得等到这一整批跑完。我们在 §5 把这个限制拆掉。

### 练习

- E2.1 — 在 `tests/test_block_manager.py` 加一个测试:连续 alloc/free 100 次不同长度的 seq,断言 `num_free_blocks` 始终在合法范围内,且最终回到初始值。
- E2.2 — 把 `block_size` 从 16 改成 1(每个 token 自己一个 block)。跑 `examples/run_toy.py`。性能差多少?为什么 vLLM 默认 16 而不是 1 或 256?
- E2.3 — `BlockManager` 当前的 free list 用 list pop。换成 deque 或者 SortedList 性能会变吗?写微基准证明。

---

## §3. PagedAttention Kernel(参考实现)

我们的 `AttentionBackend` 暴露三个方法(`mini_vllm/backends/interface.py`):

```python
class AttentionBackend(Protocol):
    def reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)
    def prefill(q, key_cache, value_cache, block_table, seq_lens, query_lens, scale)
    def decode (q, key_cache, value_cache, block_table, context_lens, scale)
```

注意 prefill 和 decode 都**只接收 cache + block_table,不直接接收 K/V 参数**。新写入的 K/V 在 attention 之前已通过 `reshape_and_cache` 写入 cache,kernel 通过 block_table 反查回来。这看起来浪费(刚写又读),但它统一了"前缀已 cached"和"chunk 内新写入"两条路径,让 chunked prefill 和 prefix caching 不需要 kernel 分支。

### Reference decode kernel(`mini_vllm/backends/reference.py`)

经典 PagedAttention 形态。Q 长度恒为 1,K/V 散落在 block_table 指向的不连续物理 block 里:

```python
def reference_decode(q, key_cache, value_cache, block_table, context_lens, scale):
    for b in range(B):
        ctx = context_lens[b]
        # Gather K/V for this seq from its block_table
        K = []
        for blk_idx in range(...):
            block_id = block_table[b, blk_idx]
            K.append(key_cache[block_id, :, :, :take].permute(2, 0, 1))
        K = torch.cat(K, dim=0)             # [ctx, H_kv, D]
        K = K.repeat_interleave(group, dim=1)   # GQA broadcast → [ctx, H, D]
        # 标准 attention
        scores = einsum("hd,thd->ht", q[b], K) * scale
        attn = softmax(scores, dim=-1)
        out[b] = einsum("ht,thd->hd", attn, V)
```

**这是慢但绝对正确的版本**。vLLM 的 CUDA paged attention v1 kernel 把这套流程改写成:每个 thread block 负责 `(seq_idx, head_idx)`,沿 block_table 顺序加载 block(每次 `block_size×head_dim` 的 K tile),累加在线 softmax(`m_i, l_i, acc`)。功能等价,数学等价,但走 SMEM tile + 张量核心。我们把性能版留给 [Plan 2](../docs/superpowers/specs/2026-05-06-mini-vllm-paged-attention-design.md)(等 GPU 机器)。

### Reference prefill kernel —— 关键扩展

prefill 的 reference 实现支持 **causal mask 在"已 cached 前缀 + 新 chunk"上正确计算**:

```python
def reference_prefill(q, key_cache, value_cache, block_table,
                      seq_lens, query_lens, scale):
    for b in range(B):
        n_q  = query_lens[b]                    # 这一步要算的新 query 数
        n_kv = seq_lens[b]                       # 累计 KV 长度(prior cached + 新 chunk)
        n_cached = n_kv - n_q                    # 之前 cached 的部分
        # Gather K/V from cache(包含 prior cached + 刚通过 reshape_and_cache 写入的新 chunk)
        K, V = gather_from(block_table[b], 0, n_kv)
        # Causal mask: query 位置 q(绝对位置 = n_cached + q)attends to KV [0, n_cached + q]
        mask[q, kv] = 0 if kv <= n_cached + q else -inf
        scores = (Q @ K^T) * scale + mask
        attn = softmax(scores)
        out[b] = attn @ V
```

n_cached=0 时这就是普通的 causal triangular。n_cached>0 时新 chunk 完全可见 prior 部分,自身仍 causal。**这一行 mask 计算就是 chunked prefill + prefix caching 共用的数学基础。**

### 测试 contract

`tests/test_attention.py` 锁三件事:
1. `reshape_and_cache` 写到正确的物理 slot(block_id*block_size + offset)
2. Torch backend 的 prefill/decode 与 reference 完全一致(allclose atol=1e-5,fp32)
3. **Chunked prefill 与 unchunked 输出 bit-identical** —— 把一个长度 8 的 prompt 分成 chunk1(5 token)+ chunk2(3 token)算,与一次算 8 token,结果完全相同。这个测试是 Plan 5 chunked prefill 正确性的根基。

### 练习

- E3.1 — 验证 GQA broadcast 正确性:Q=8 head,KV=2 head(group=4)。手算 group 内每个 query head 应该看到哪个 KV head,确认 `repeat_interleave(group, dim=1)` 是对的而不是 `repeat(group, dim=1)`。
- E3.2 — 把 `reference_prefill` 改成走 `F.scaled_dot_product_attention` + `attn_mask`(用 mask 张量代替手动 -inf 写入)。验证 allclose。
- E3.3 — 在 `_build_kv_cache_from_kv` 测试辅助函数里,改 block_size 为 8、3、1,看测试是否还过。block_size=1 时 cache layout 退化为什么形态?

---

## §4. 接上真实模型:TinyLlama-1.1B

ToyGPT 用随机权重,生成的是乱码;读者真正的获得感来自跑通真实 LLM。我们做了 TinyLlama-1.1B-Chat-v1.0,**不依赖 `transformers` 的 modeling code**。

### Llama 架构(`mini_vllm/models/llama.py`)

包含 5 个组件:

1. **`RotaryEmbedding`**:precomputed cos/sin tables,绝对位置 indexed。卷积约定 `[N, H, D]`(我们的 layout)而不是 HF 的 `[B, H, T, D]`。
2. **`apply_rotary_pos_emb`**:rotate-half 风格(`[-x_high, x_low]`)对齐 HF。Test atol=1e-5。
3. **`LlamaRMSNorm`**:fp32 计算稳定性,gamma scale。
4. **`LlamaMLP`**:**fused** `gate_up_proj` + `down_proj`,SwiGLU。HF 把 gate 和 up 分开存,我们 fuse 成一个矩阵(对齐 vLLM 真实做法)。
5. **`LlamaAttention`**:GQA + RoPE,**fused** `qkv_proj`。RoPE 在写 cache **之前**应用 —— 这个顺序与 HF/vLLM 一致,paged cache 持有 rotated K。

### HF safetensors loader(`mini_vllm/models/llama_loader.py`)

`load_hf_to_llama_model(model, "TinyLlama/TinyLlama-1.1B-Chat-v1.0")` 干两件事:

```python
# 1. Key remapping:HF 的 model.layers.X.self_attn.q_proj.weight 等
#    → 我们的 layers.X.self_attn.qkv_proj.weight
# 2. Tensor fusing:把 q/k/v 三份 weight concat 成 [Q | K | V] 的 fused QKV matrix
fused_qkv = torch.cat([q_weight, k_weight, v_weight], dim=0)
fused_gate_up = torch.cat([gate_weight, up_weight], dim=0)
```

split 顺序在 `LlamaAttention.forward` 里要对得上:`qkv.split([q_size, kv_size, kv_size], dim=-1)`。这个对应关系如果错了,parity test 会立刻 fail。

### Parity test(`tests/test_llama_parity.py`)

**最严格的正确性闸门**:

```python
@pytest.mark.slow
def test_logits_match_hf_top5():
    ours = LlamaModel(cfg, TorchBackend())
    load_hf_to_llama_model(ours, "TinyLlama/...")
    hf = LlamaForCausalLM.from_pretrained("TinyLlama/...")

    prompt = [1, 15043, 29892, 1373, 526, 366, 2599, 29973]   # 8 tokens
    our_top5 = topk(ours(prompt).logits[-1], 5)
    hf_top5  = topk(hf(prompt).logits[0, -1], 5)
    assert overlap(our_top5, hf_top5) >= 4
```

实测 5/5 完全相同。`tests/test_llama_e2e.py` 进一步跑 8-token greedy 生成对比,**逐 token 8/8 完全一致**(参见 commit `fcc7bd6` 的实测输出)。

这意味着我们的实现在数学上与 HF transformers 等价。不是"接近",是"位级精确"。

### 练习

- E4.1 — 改一下 `_fuse_qkv` 的顺序,把 v 放在最前面。重跑 parity test,看哪个 assert 先 fail。
- E4.2 — 把 `LlamaRMSNorm` 的 fp32 cast 拿掉(全程 input dtype 计算)。在 fp16 下 parity 还成立吗?atol 需要放到多少?
- E4.3 — 加载 Qwen2.5-0.5B(架构相近,主要差 vocab_size 和 max_pos)。需要改什么?

---

## §5. Continuous Batching

### 朴素调度的瓶颈

§2 的 Scheduler 在 running 非空时拒绝接新请求。这意味着**一个 8K 长 prompt 占着不放,后到的 short prompts 全都等着**。在线服务里这是灾难:p99 TTFT 直接爆。

### 改动:`enable_continuous_batching=True`(默认)

`mini_vllm/scheduler.py` 的 `schedule()` 主循环现在做两件事:

1. 把 running 队列处理完(prefill 中的 → 继续 prefill,prefill 完的 → decode)
2. **如果还有 token budget,从 waiting 队首继续拉新请求加入这一步的 batch**

```python
can_admit = self.enable_continuous_batching or not self.running
if can_admit:
    while self.waiting and budget > 0:
        seq = self.waiting[0]
        if seq.num_prompt_tokens > budget:
            break    # FCFS:头大请求 budget 不够,后面小请求也等
        ...allocate, admit, budget -= chunk_len
```

**Token budget**:`max_num_batched_tokens`(默认 2048)。每个 decode 占 1,每个 prefill 占其 chunk_len(Plan 4 = 整个 prompt)。一旦超 budget 就 break。

### 模型层为什么不用动

我们在 §3 已经把 `prefill` kernel 设计成支持**混合 batch**:`prefill_seq_lens` 是数组,可以同时容纳多条 seq。这是为什么 continuous batching 只改 scheduler 一个文件 —— Plan 1 接口已经为它准备好了。

### Bench 数据(toy GPT)

```
config                | tok/s   | TTFT p50
naive (off)           | 2469.2  | 0.008
+ continuous batching | 2785.4  | 0.006
```

toy GPT 上提升不大(prompts 都很短);真实 LLM + 长尾 prompt 上的差距能有数倍。bench.py 的价值是**演示 flag 切换的代码框架**,不是绝对吞吐数字。

### 练习

- E5.1 — 把 `max_num_batched_tokens` 改成 4(极端小)。跑 `examples/run_toy.py` 4 个 prompt,观察一步只能跑得动多少。
- E5.2 — `tests/test_e2e.py::test_e2e_continuous_batching_vs_baseline_same_output` 断言 flag on/off **输出完全一致**。如果生成结果依赖调度顺序(比如 token attention 数值受 batch 内其它 seq 影响),这个测试会怎么 fail?为什么我们的实现不会?
- E5.3 — 实现 LCFS(Last-Come-First-Served)admission policy 替代 FCFS,看 TTFT 中位数变化。

---

## §6. Chunked Prefill

### 长 prompt 阻塞 decode 的故事

scenario:已有 4 个 seq 在 decode,各 1 token/step。来了一个 8000 token 的 prompt。Plan 4 调度会怎么做?

- Token budget 2048,8000 > 2048 → 这个 prompt **不被 admit**
- 那它就一直在 waiting 队列
- 直到所有当前 running 都跑完它才开 prefill
- **它的 TTFT = 已有 seq 全部完成的时间** —— 灾难

vLLM 的解法:**把长 prompt 切片**。8000 token → 16 个 chunk × 500。每个 step 只算 500 个 prefill token + 4 个 decode token(共 504,fits budget)。decoding 用户感知不到长 prompt 的存在。

### 数学正确性的关键

切片之后的第二个 chunk 必须能 attend 到第一个 chunk 已写入 cache 的 K/V。这就是为什么 §3 的 `prefill` kernel 必须从 cache 读 K/V。**chunked prefill 在 kernel 层就是"上一个 chunk 已 cached + 这个 chunk 是新写入"**:

```python
# Step 1: prefill chunk 1(token 0-499)
seq_lens = [500];  query_lens = [500]   # n_cached = 0
# Step 2: prefill chunk 2(token 500-999)
seq_lens = [1000]; query_lens = [500]   # n_cached = 500
# Step 3: prefill chunk 3
seq_lens = [1500]; query_lens = [500]   # n_cached = 1000
...
```

`tests/test_attention.py::test_prefill_chunked_matches_unchunked` 验证:对一个 length=8 的 prompt,无论分 1+8 还是 5+3 还是 1+1+1+...+1,**逐位置输出完全相同**。

### Scheduler 改动(`mini_vllm/scheduler.py`)

加一个 `_chunk_for(seq, budget)` 方法决定本步给这个 seq 分多少 token:

```python
def _chunk_for(self, seq, budget):
    remaining = seq.num_prompt_tokens - seq.num_prefilled
    chunk = remaining
    if self.enable_chunked_prefill:
        chunk = min(chunk, self.chunked_prefill_size)   # 默认 512
    chunk = min(chunk, budget)
    return max(0, chunk)
```

`Sequence.scheduled_chunk_len` 跟踪每步进度。`Sequence.num_prefilled` 在 `Engine.step` 之后增加 chunk_len:

```python
for seq in sched.prefill_seqs:
    seq.num_prefilled += seq.scheduled_chunk_len
```

只有当 `num_prefilled == num_prompt_tokens` 时才 sample 这个 seq 的 logit(中途 chunks 没有 logit 输出)。

### 练习

- E6.1 — `chunked_prefill_size=4`,跑 `tests/test_e2e.py::test_e2e_chunked_prefill_matches_unchunked`(已存在),改 prompt 到 50 token。测试还过吗?
- E6.2 — chunked prefill 增加了什么 overhead?数 `Engine.step` 调用次数:不切片时 1 步搞定 prefill,切 16 片就要 16 步。每步 KV gather + softmax 都重做,但每次的计算量约 1/16。绝对 cost 多了多少?
- E6.3 — 把 chunked prefill 写得"懒一点":`chunked_prefill_size=∞` 但仍然受 budget 约束。和当前实现等价吗?何时不等价?

---

## §7. Prefix Caching + CoW

### 问题:相同 system prompt 的 100 个请求

OpenAI API 上常见 pattern:`messages=[{"role":"system","content":"You are..."}, {"role":"user","content":"..."}]`。100 个并发请求的 system prompt 一字不差,user 部分各不相同。

朴素 vLLM 会算 100 遍 system prompt 的 K/V,显存里 100 份。Prefix caching 能把这 100 份共享成 1 份。

### 实现:Block Hash Chain

`mini_vllm/block_manager.py` 的核心:

```python
def _hash_block(prev_hash, token_ids):
    return hash((prev_hash, token_ids))

# 注册:每当一个 block 被填满(block_size 个 token 写入),计 hash 入表
def register_filled_blocks(self, seq):
    for i in range(seq.seq_len // block_size):
        pb = seq.block_table.physical_blocks[i]
        tokens = tuple(seq.token_ids[i*bs:(i+1)*bs])
        h = _hash_block(prev_hash, tokens)
        if h not in self._hash_to_block:
            self._hash_to_block[h] = pb
        prev_hash = h

# 查找:新 seq 来时按前缀扫
def _lookup_cached_prefix(self, token_ids):
    out = []
    prev_hash = None
    for i in range(len(token_ids) // block_size):
        h = _hash_block(prev_hash, tuple(token_ids[i*bs:(i+1)*bs]))
        pb = self._hash_to_block.get(h)
        if pb is None: break
        out.append(pb)
        prev_hash = h
    return out
```

只对 **填满的** block 计 hash —— 部分填的最后一个 block 不参与共享(因为它的剩余 slot 内容未定)。

### 引用计数

`allocate(seq)` 命中 cached prefix 时:

```python
cached = self._lookup_cached_prefix(seq.prompt_token_ids)
for pb in cached:
    pb.ref_count += 1   # 共享:多个 seq 持有同一个 block
n_fresh = n_total - len(cached)
fresh = [self._take_free_block() for _ in range(n_fresh)]
seq.block_table = BlockTable(physical_blocks=cached + fresh)
seq.num_prefilled = len(cached) * block_size   # ← 跳过这部分计算
```

**Scheduler 自动配合**:`_chunk_for` 在 `seq.num_prefilled` 已经被 allocate 设到非零时,只需要算剩余的 `num_prompt_tokens - num_prefilled`。整套机制在 `Engine.step` 的层面完全透明 —— 只是这个 seq 的"prefill 起点"不再是 0。

### Evictable Cache

朴素实现里 ref_count → 0 的 block 立刻回 free pool,导致**重复 prompt 在第二次跑时 cache 已经被清掉**。我们加了 evictable list:

```python
def free(self, seq):
    for pb in seq.block_table.physical_blocks:
        pb.ref_count -= 1
        if pb.ref_count == 0:
            if self.enable_prefix_caching and pb.block_hash is not None:
                self._evictable.append(pb.block_id)   # 留在 cache,可被驱逐
            else:
                self._free_block_ids.append(pb.block_id)
                ...

def _take_free_block(self):
    if self._free_block_ids:
        block_id = self._free_block_ids.pop()
    elif self._evictable:
        # FIFO 驱逐最老的 cached block
        block_id = self._evictable.pop(0)
        del self._hash_to_block[old.block_hash]
    ...
```

`_lookup_cached_prefix` 命中一个 evictable block 时,在 `allocate` 里 `_rescue_from_evictable` 把它从 evictable 列表中移出 + ref_count++。

### "保留至少 1 个 fresh token" 规则

如果 prompt 完全是 cached prefix(prompt_len 是 block_size 整数倍且全命中),`num_prefilled` 会等于 `num_prompt_tokens`,**没有 token 走 forward,也就没有 logit 用来 sample 下一个 token**。

解法:`allocate` 在全命中时主动丢掉最后一个 cached block,保证至少 1 个 fresh token 进 forward。

```python
if cached and len(cached) * block_size >= seq.num_prompt_tokens:
    cached = cached[:-1]
```

### CoW(plumbing 完整,Plan 5 不触发)

`append_slot` 检测 `physical_blocks[-1].ref_count > 1` 时返回 `(src, dst)` 让 runner 复制 K/V 数据。但 Plan 5 单 completion greedy 路径下,decode 写入的总是私有 block(cached blocks 在 prefix 起点,decode 总在尾部)—— 所以**测试里手动构造场景验证 CoW 返回值正确**(`test_prefix_cache.py::test_cow_triggered_when_appending_into_shared_block`),实际生成路径不会触发。Parallel sampling 或 beam search 才会真正用到 CoW。

### 练习

- E7.1 — 跑 `tests/test_prefix_cache.py::test_e2e_prefix_cache_skips_compute_on_repeat_prompt`,加 print 把第二次 admit 时的 `seq.num_prefilled` 打出来。手算应该是几?
- E7.2 — `_hash_block` 用 Python 内置 `hash()`(per-process salted)。在分布式 vLLM(多 worker)下这会出问题吗?如何改成 deterministic hash?
- E7.3 — Evictable cache 当前 FIFO。改成 LRU 看哪些场景受益(hint:重复 prompt 频率高时)。

---

## §8. Swap to CPU + LRU 抢占

### 显存超订阅

GPU pool 默认 256 个 block。如果某一刻并发请求要 300 个 block,3 个选择:

1. **Reject** 后到的:简单粗暴,影响 SLA
2. **Recompute**:把某个 seq 的 KV 全丢掉,重新走 prefill。短 prompt 场景下其实最便宜
3. **Swap to CPU**:把 GPU 上的 block 拷到 CPU pool 暂存,等显存腾出再拷回来。CPU RAM 通常比 HBM 大 4-10 倍

我们实现 swap 路径(spec 里 recompute 留作扩展)。

### CacheEngine 加 CPU pool(`mini_vllm/cache_engine.py`)

```python
if cache_cfg.num_cpu_blocks > 0:
    cpu_shape = (num_cpu_blocks, num_kv_heads, head_dim, block_size)
    self.cpu_kv_caches = [(torch.zeros(cpu_shape, device='cpu'), ...) ...]

def swap_out(self, mapping):       # {gpu_id: cpu_id}
    for layer in range(self.num_layers):
        for g_id, c_id in mapping.items():
            self.cpu_kv_caches[layer][0][c_id].copy_(self.kv_caches[layer][0][g_id].to('cpu'))
            self.cpu_kv_caches[layer][1][c_id].copy_(self.kv_caches[layer][1][g_id].to('cpu'))
```

CPU pool 的 layout 与 GPU 完全一致(只是 device='cpu')。tensor copy 走 `.to('cpu')` / `.to(self.device)`(真实 vLLM 用 `cudaMemcpyAsync` + 多 stream 重叠)。

### BlockManager 双池(`mini_vllm/block_manager.py`)

GPU/CPU 两个独立 id namespace。`PhysicalBlock.device` 字段标记当前所在池。

**`can_swap_out(seq)` 关键约束**:

```python
def can_swap_out(self, seq):
    gpu_blocks = [pb for pb in seq.block_table.physical_blocks if pb.device == 'gpu']
    if any(pb.ref_count > 1 for pb in gpu_blocks):
        return False    # 共享 block 不能 swap(否则破坏 prefix cache 一致性)
    return self.num_free_cpu_blocks >= len(gpu_blocks)
```

**共享 block 不能 swap** —— 这是 prefix caching 与 swap 互动的关键不变量。如果允许 swap 共享 block,其它仍在 GPU 上 running 的 seq 突然找不到自己的 K/V 了。

### Scheduler 自动抢占(`mini_vllm/scheduler.py`)

每个 step 头部尝试 `_try_swap_in`(把 swapped 队列里能装下的拉回 GPU)。每个 decode seq 在 `append_slot` 之前调 `_ensure_room_for_append`:

```python
def _ensure_room_for_append(self, decoding_seq, out):
    extra = max(0, needed - have)              # 这个 seq 这步要扩几个 block
    while extra > self.bm.num_free_blocks:
        victim = self._pick_swap_victim(exclude=decoding_seq)
        if victim is None: break
        mapping = self.bm.swap_out(victim)
        out.swap_out.update(mapping)
        self.running.remove(victim)
        self.swapped.append(victim)
```

LRU 受害者选择:running 队尾(最近 admit 的)倒着走,跳过 ineligible 的(共享 block 等)。

### Engine 应用 swap(`mini_vllm/engine.py`)

```python
def step(self):
    sched = self.scheduler.schedule()
    if sched.swap_out:
        self.cache_engine.swap_out(sched.swap_out)   # K/V tensor 拷到 CPU
    if sched.swap_in:
        self.cache_engine.swap_in(sched.swap_in)     # 拷回 GPU
    # ...然后 forward
```

顺序很重要:swap 必须在 forward 之前完成,否则 attention 读到的 block_id 对应不上 tensor 内容。

### E2E 验证(`tests/test_swap.py`)

最严格的测试:

```python
def test_e2e_swap_matches_no_swap_baseline():
    # baseline: 大 GPU pool,swap 关
    eng_big = build(num_gpu_blocks=64, num_cpu_blocks=0, enable_swap=False)
    # 紧张场景: 小 GPU + 大 CPU,swap 自动触发
    eng_small = build(num_gpu_blocks=8, num_cpu_blocks=32, enable_swap=True)
    out_big   = eng_big.generate(prompts, sp)
    out_small = eng_small.generate(prompts, sp)
    assert out_big == out_small   # 输出 token 完全一致
```

Swap 改变的是**时序**(blocks 在 GPU/CPU 之间搬运),不改变**数学**(K/V 内容 bit-identical 经过 round-trip)。

### 练习

- E8.1 — 改 bench 加一行"+ swap on plenty"(`num_gpu_blocks=64, num_cpu_blocks=64, enable_swap=True`)。throughput 下降多少?为什么(hint:swap 没必要触发,但代码路径多了什么?)。
- E8.2 — 实现 recompute 抢占策略:被抢占的 seq 重置 `num_prefilled=0`、free 它的所有 GPU block,重新进 waiting 队列。短 prompt(<128 token)下 recompute vs swap 哪个更快?写测试比对。
- E8.3 — 当前 `_ensure_room_for_append` 的"挑队尾"是 LIFO 抢占。改成 LRU(running 队列里最久没产生 token 的 seq)。

---

## §9. 从 Mini 到真实 vLLM

### 我们简化掉的东西

| 真实 vLLM | mini-vLLM |
|---|---|
| CUDA paged attention v1/v2 kernel | Reference Python loop(慢但正确) |
| Triton/Flash kernel for prefill | 同上 |
| Async engine + AsyncLLMEngine | 同步 `step()`,无 asyncio |
| Multi-worker(TP/PP)+ NCCL all-reduce | 单进程 |
| Ray cluster orchestration | 无 |
| OpenAI-compatible HTTP server | 无(只有 Python API + streaming generator) |
| AWQ/GPTQ/FP8 量化 | 无 |
| LoRA / multi-LoRA | 无 |
| Speculative decoding(draft model / Medusa / EAGLE) | 无 |
| Sliding window / ALiBi / linear attention | 无 |
| Beam search / parallel sampling | sampler 单 completion |
| CUDA graph capture for decode | 无 |
| Persistent batching with prefix cache aware admission ordering | FCFS only |
| Recompute preemption strategy | 仅 swap |
| Prefetch / pipelining of swap_in/out | 同步阻塞拷贝 |

### 概念对照表(读真实 vLLM 源码时)

| mini_vllm.X | vllm.X(粗略) |
|---|---|
| `engine.LLMEngine` | `vllm.engine.llm_engine.LLMEngine`(V0)/ `vllm.v1.engine.core.EngineCore`(V1) |
| `scheduler.Scheduler` | `vllm.core.scheduler.Scheduler` / `vllm.v1.core.scheduler.Scheduler` |
| `block_manager.BlockManager` | `vllm.core.block_manager.BlockSpaceManager` / `vllm.v1.core.kv_cache_manager.KVCacheManager` |
| `model_runner.ModelRunner` | `vllm.worker.model_runner.GPUModelRunner` |
| `cache_engine.CacheEngine` | `vllm.worker.cache_engine.CacheEngine` |
| `backends.AttentionBackend` | `vllm.attention.backends.abstract.AttentionBackend` |
| `models.llama.LlamaModel` | `vllm.model_executor.models.llama.LlamaModel` |
| `models.llama_loader.load_hf_to_llama_model` | `vllm.model_executor.model_loader.weight_utils` + 各 model 的 `load_weights` |
| `sampler.Sampler` | `vllm.model_executor.layers.sampler.Sampler` |
| `sequence.Sequence` | `vllm.sequence.Sequence` / `vllm.v1.request.Request` |

### 哪些机制我们做对了

如果你用 mini-vLLM 学完后再去看真实 vLLM 源码,这些**核心抽象一一对应**:

- **Block table 间接寻址** —— 数据结构、`slot_mapping` 的含义、attention kernel 的接口形态完全一致
- **Continuous batching 调度模型** —— FCFS + token budget + admission policy
- **Prefix caching 的 hash chain + 引用计数 + evictable** —— 算法等价
- **Chunked prefill 在 kernel 层的"prior cached + new chunk"统一表达** —— 数学等价
- **Swap-to-CPU 双池 + LRU 抢占** —— 简化但同构

如果要扩展真实 vLLM 的某个 feature,你应该已经知道**这个 feature 影响哪几个文件、改动什么 API**。这是这套教程最重要的产出。

### 哪些机制我们没做(留作扩展)

- **Triton/CUDA decode kernel**(Plan 2 spec 已写,待 GPU 机器):block_table 索引 + 在线 softmax(`m_i, l_i, acc`)+ GQA-aware tile
- **Recompute preemption**:相比 swap 更便宜的短 prompt 路径
- **Async engine**:`step()` 改成 `asyncio.run(...)` 包一层,引入 producer/consumer queue
- **TP**:每个 layer 的 weight 沿 head 切,attention 后 all-reduce。最小实现 ~200 行
- **OpenAI server**:FastAPI + SSE streaming,本质是把 `generate_stream` 接到 HTTP

### 练习

- E9.1 — 选一个真实 vLLM 的 feature(比如 `enable_prefix_caching` 在 V1 引擎里的实现),clone vllm 仓库,grep 找到对应代码。它的复杂度是 mini-vLLM 的几倍?多出来的复杂度服务于什么?
- E9.2 — 把 mini-vLLM 的 `sampler.py` 加上 repetition penalty(对已生成的 token logit 减 penalty)。需要传什么 state 到 sampler?
- E9.3 — 设计一个 "chat template" wrapper:接受 `messages=[{"role":..., "content":...}]`,生成正确的 ChatML/Llama-style prompt 字符串,送进 `engine.add_request`。

---

## 总结

我们用 1500 行左右的纯 Python 实现了 vLLM 的核心:**PagedAttention 寻址、continuous batching、chunked prefill、prefix caching with CoW、swap-to-CPU、full sampler、streaming**。每个 feature 通过 `EngineConfig` flag 切换,**与 HF transformers 在 TinyLlama-1.1B 上 8/8 token greedy 完全一致**。

性能我们完全没追求 —— reference attention 比真实 CUDA kernel 慢 50-100×。但**架构对得上**:你现在能在 PaperPaged 上读懂 PagedAttention 论文里每个图,在 vLLM 源码里认出每个核心数据结构,在生产 issue 里推断 metric 异常的可能根因。

下一步:
- 想跑性能?装 NVIDIA GPU,做 [Plan 2(Triton kernel)](../docs/superpowers/specs/2026-05-06-mini-vllm-paged-attention-design.md)
- 想理解更深?把上面 9 节的练习 E*.1/E*.2/E*.3 全做一遍
- 想看真实生产形态?读 vLLM `v1/engine/core.py` 和 `v1/core/scheduler.py`,带着我们的概念对照表

代码、测试、bench 都在 `code/mini-vllm/` 里。`pytest tests/ -m slow` 是最大的信任来源 —— 如果它过了,你的实现与 HF 数学一致;如果它挂了,某个 kernel/loader 写错了。

---

## 配套源码 commit 历史(供索引)

```
9bcdcb0  Plan 7: streaming + sampler + bench
9453c7f  Plan 6: swap to CPU + LRU preemption
5ac4b4a  Plan 5: prefix caching + CoW + chunked prefill
497b727  Plan 4: continuous batching + token budget
0a07b0a  Plan 3: TinyLlama 1.1B + HF safetensors loader (8/8 greedy parity)
6eb4b35  Plan 1: skeleton (toy GPT, Torch backend, naive scheduler)
```

每个 Plan 都有独立 spec/plan 文档在 `docs/superpowers/{specs,plans}/`,可以追溯每一步的设计决策与执行细节。
