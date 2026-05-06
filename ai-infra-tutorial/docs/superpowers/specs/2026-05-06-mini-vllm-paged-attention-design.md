# Mini-vLLM with PagedAttention — 设计文档

- **日期**: 2026-05-06
- **目标**: 实现一个端到端可跑的 mini 推理引擎,核心是从零写 PagedAttention,作为 `part5-serving-infra/16a-vllm-internals.md` 的配套实战章节
- **配套章节**: `part5-serving-infra/16a-lab-mini-vllm.md`(新建)
- **代码位置**: `code/mini-vllm/`

---

## 1. 范围与目标

### 1.1 想做的(In-scope)

**核心(全部必做):**
- Block-based KV cache + BlockTable(PagedAttention 的存储基础)
- Prefill / Decode 双形态 attention kernel
- Batched decode(单请求 → 多请求)
- 最简 FCFS Scheduler(waiting / running / swapped 三队列)

**进阶(全部必做):**
1. Continuous batching —— 每步动态接入新请求 *(flag: `enable_continuous_batching`)*
2. Prefix caching / block sharing(含 CoW)*(flag: `enable_prefix_caching`)*
3. Chunked prefill —— 长 prompt 不阻塞 decode *(flag: `enable_chunked_prefill`)*
4. Swap to CPU —— 显存超订阅时换出 *(flag: `enable_swap`)*
5. Sampling —— greedy / temperature / top-p / top-k(始终开启,通过 `SamplingParams` 控制)
6. Streaming token 输出(始终开启,通过 `generate_stream` API 使用)

上述 1-4 是可关闭的对照开关,5-6 是始终开启的能力。

**双后端 attention kernel:**
- `TorchPagedAttention`:CPU/MPS 可跑,正确性基线
- `TritonPagedAttention`:GPU 性能版

**两份模型(共用同一 backend 接口):**
- `models/toy_gpt.py`:自包含 GPT-2 风格 toy 模型(~30M),教学用
- `models/llama.py`:TinyLlama-1.1B 兼容,加载 HF safetensors

### 1.2 不做的(Out-of-scope, YAGNI)

- Tensor parallel / pipeline parallel / 多 GPU
- 量化(AWQ/GPTQ/FP8)
- LoRA / multi-LoRA
- Speculative decoding
- 跨进程 worker、Ray、async engine
- 完整 OpenAI-compatible HTTP server(仅提供 streaming Python API + 简单 CLI)
- Sliding window / ALiBi
- Beam search

---

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────┐
│  CLI / Streaming API     (examples/run.py, server.py)   │
└────────────────────────┬────────────────────────────────┘
                         │ generate(prompts, sampling_params)
┌────────────────────────▼────────────────────────────────┐
│  LLMEngine                                              │
│   ├─ add_request() / step() / abort()                   │
│   └─ 拥有 Scheduler / ModelRunner / Tokenizer / Sampler │
└──────┬──────────────────────────────────┬───────────────┘
       │                                  │
┌──────▼──────────────┐       ┌───────────▼──────────────┐
│  Scheduler          │       │  ModelRunner             │
│   ├─ waiting queue  │       │   ├─ load model          │
│   ├─ running queue  │       │   ├─ build batch input   │
│   ├─ swapped queue  │       │   ├─ forward(...)        │
│   └─ schedule() →   │       │   └─ 调 AttentionBackend │
│      SchedulerOut   │       └───────────┬──────────────┘
└──────┬──────────────┘                   │
       │ allocate/free/swap/share blocks  │ K/V via block table
┌──────▼──────────────────────────────────▼───────────────┐
│  BlockManager  +  KVCache (physical block pool)         │
│   ├─ BlockTable per request                             │
│   ├─ 引用计数(prefix caching / CoW)                    │
│   └─ GPU pool ⇄ CPU pool(swap)                         │
└─────────────────────────────────────────────────────────┘
                         ▲
┌────────────────────────┴────────────────────────────────┐
│  AttentionBackend (运行时按 device 选)                   │
│   ├─ TorchPagedAttention  (CPU/MPS,正确性基线)          │
│   └─ TritonPagedAttention (GPU 性能版)                   │
│   各自暴露 prefill(...) / decode(...) 两入口             │
└─────────────────────────────────────────────────────────┘
```

**关键原则:**
1. 每个进阶特性是一个 `EngineConfig` flag,默认全开;关掉就退化到对照基线。
2. AttentionBackend 是 thin interface;Triton 版必须与 Torch 版 `torch.allclose` 对齐。
3. ModelRunner / Scheduler / BlockManager 完全不感知模型种类。
4. 同步引擎,`step()` 阻塞;streaming 通过 step 间 yield 实现。

---

## 3. BlockManager & KVCache

### 3.1 物理 KV pool

```python
key_cache:   [num_blocks, num_kv_heads, head_dim, block_size]
value_cache: [num_blocks, num_kv_heads, head_dim, block_size]
```
- `block_size` 默认 16(可配)
- `num_blocks` 启动时由 `CacheEngine.profile_num_blocks()` 算:`(显存预算 - 模型权重 - 激活 buffer) / per_block_bytes`
- 每层一对 K/V cache,真实张量是 `List[Tuple[K, V]]` 长度 = `num_layers`

### 3.2 BlockTable

每请求维护 `List[PhysicalBlock]`,长度 = `ceil(seq_len / block_size)`。

```python
@dataclass
class PhysicalBlock:
    block_id: int
    ref_count: int
    block_hash: Optional[int]  # 仅填满的 block 才计 hash
    device: Literal["gpu", "cpu"]
```

### 3.3 BlockManager 接口

```python
class BlockManager:
    def can_allocate(self, seq) -> AllocStatus           # OK / LATER / NEVER
    def allocate(self, seq) -> BlockTable
    def append_slot(self, seq) -> Optional[CowMapping]
    def free(self, seq)
    def fork(self, parent_seq, child_seq)                # 接口预留
    def get_cached_blocks(self, token_ids) -> List[PhysicalBlock]
    def swap_out(self, seq) -> Dict[int,int]
    def swap_in(self, seq)  -> Dict[int,int]
```

### 3.4 Prefix caching

- `block_hash = hash(prev_block_hash, tuple(token_ids_in_block))`
- 仅对填满的 block 计 hash
- `BlockManager` 维护 `hash_to_block: Dict[int, PhysicalBlock]`
- allocate 时按前缀 hash 链查命中,命中则 `ref_count += 1` 直接复用
- decode 写入若 `ref_count > 1` 触发 CoW:分配新 block、拷贝原内容、更新 BlockTable、原 block `ref_count -= 1`

### 3.5 Swap

- 启动除 GPU pool 外再分配 CPU pool(`num_cpu_blocks` 默认 3 × `num_gpu_blocks`)
- 显存不足时 scheduler 选 victim(LRU),`swap_out` 把所有 GPU block 拷到 CPU pool 槽位
- 该请求进 `swapped` 队列,下一轮尝试 `swap_in`

### 3.6 不变量(测试严格校验)

1. 每个 GPU block_id 在任意时刻要么在 free pool,要么被 BlockTable 持有,`ref_count` 之和 = 持有它的 BlockTable 数
2. swap_out 后 BlockTable 持有 CPU block_id;swap_in 后必须恢复成 GPU block_id
3. 同一 hash 的 block 在 `hash_to_block` 表里有且只有一个条目

---

## 4. PagedAttention Kernel

### 4.1 统一接口 `AttentionBackend`

```python
class AttentionBackend(Protocol):
    def reshape_and_cache(
        key, value,                   # [num_tokens, num_kv_heads, head_dim]
        key_cache, value_cache,
        slot_mapping,                 # [num_tokens]: block_id*block_size + offset
    ) -> None

    def prefill(
        q, k, v,                      # [num_prefill_tokens, num_heads, head_dim]
        kv_cache, block_table,        # block_table 仅 prefix 命中时非空
        seq_lens, query_lens, scale,
    ) -> Tensor

    def decode(
        q,                            # [batch, num_heads, head_dim]
        kv_cache, block_table,        # [batch, max_blocks]
        context_lens, scale,
    ) -> Tensor
```

### 4.2 为什么 prefill 与 decode 分开

- **Prefill**:Q/K/V 都长,适合 FlashAttention 风格 tile 计算
- **Decode**:Q 长度恒为 1,K/V 散落在不连续物理 block,需要按 block_table gather + 在线 softmax —— 这是经典 PagedAttention 形态

### 4.3 Torch 参考实现(`backends/torch_backend.py`)

- `reshape_and_cache`:按 `slot_mapping` 做 scatter
- `prefill`:命中前缀按 block_table gather 拼回连续 → 与新 K/V concat → `F.scaled_dot_product_attention(causal=True)`
- `decode`:循环 batch,按 block_table gather K/V → naive matmul + softmax(慢但正确,作为 Triton 版基线)

### 4.4 Triton 实现(`backends/triton_backend.py`)

- `decode` kernel:每个 program 负责 `(batch_idx, head_idx)`,沿 block_table 循环加载 block(每次 `block_size×head_dim` 的 K tile),累加在线 softmax(`m_i, l_i, acc`)。简化版 vLLM v1 paged attention v1 kernel。
- `prefill` kernel:GQA-aware 的简化 flash-attn fwd,支持读取已 cached 的前缀 block。

### 4.5 GQA

kernel 接受 `num_heads` 和 `num_kv_heads`,group 内 query head 共享同一 K/V head(broadcast on load)。

### 4.6 正确性约束

```python
torch.allclose(triton_decode(...), torch_decode(...), atol=1e-2, rtol=1e-2)  # fp16
torch.allclose(torch_decode(...),  reference_naive(...), atol=1e-4)          # fp32 ref
```
`reference_naive` 是把 KV 完全拼回连续张量后调 `F.scaled_dot_product_attention` 的金标准。

### 4.7 取舍

- 不实现 backward(纯推理)
- Triton kernel 用 fp16/bf16,fp32 仅 Torch 版兜底
- 不做 sliding window / ALiBi

---

## 5. Scheduler

### 5.1 三队列

- `waiting`:新加入但未分配 KV
- `running`:正在 decode
- `swapped`:换出到 CPU,等显存腾出换回

### 5.2 SchedulerOutput

```python
@dataclass
class SchedulerOutput:
    prefill_seqs:   List[SeqMeta]
    decode_seqs:    List[SeqMeta]
    swap_in:        Dict[int, int]   # CPU -> GPU
    swap_out:       Dict[int, int]   # GPU -> CPU
    blocks_to_copy: List[Tuple[int, int]]  # CoW
```

### 5.3 单步调度顺序

1. **swap_in**:把 swapped 队列里能装下的请求换回 GPU
2. **continue running decode**:为每个 running seq 尝试 `append_slot`;失败则触发 preemption(swap 或 recompute,可配),腾出后重试
3. **admit waiting prefill**:在 token budget 内尽量多拉 waiting seq;先查 prefix cache 命中前缀(命中部分直接挂载、不计算),剩余 token 数若超 `chunked_prefill_size` 就只 prefill 前段(chunked prefill)
4. **token budget**:`max_num_batched_tokens` 上限(默认 2048),decode(每 seq 1 token)+ prefill chunk 总和不超

### 5.4 Continuous batching

每步重新调度;prefill 和 decode 可同 batch 跑(ModelRunner 内部分别调 prefill/decode kernel,输出 concat)。新请求下一步立刻被纳入。

### 5.5 关键参数

- `max_num_batched_tokens` = 2048
- `chunked_prefill_size` = 512
- `block_size` = 16
- `num_cpu_blocks` = 3 × `num_gpu_blocks`

### 5.6 Preemption

- 默认 `swap`(进 swapped 队列)
- 可配 `recompute`(丢 KV,重回 waiting)
- LRU 选 victim

### 5.7 Config flag → 行为退化

| flag=False | 退化为 |
|---|---|
| `enable_continuous_batching` | 每步只调度同一批,直到全部完成才接新请求 |
| `enable_chunked_prefill` | prefill 整段一次性吃完 |
| `enable_prefix_caching` | 跳过 hash 链,prefill 全从零 |
| `enable_swap` | 显存不够直接 abort 或纯 recompute |

---

## 6. ModelRunner & 模型层

### 6.1 ModelInput

```python
@dataclass
class ModelInput:
    input_ids:        Tensor  # [num_prefill_tokens + num_decode_tokens]
    positions:        Tensor
    slot_mapping:     Tensor
    block_tables:     Tensor  # [batch, max_blocks]  decode 用
    seq_lens:         Tensor
    query_lens:       Tensor  # prefill: chunk len; decode: 1
    num_prefill_tokens: int
    num_decode_tokens:  int
```
prefill tokens 排前,decode tokens 排后。Attention 层按 `num_prefill_tokens` 切开分别走 kernel。

### 6.2 模型接口

```python
class CausalLM(Protocol):
    config: ModelConfig
    def forward(self, model_input, kv_caches) -> Tensor: ...
    def sample_indices(self, model_input) -> Tensor: ...
```

### 6.3 toy_gpt.py

- 标准 GPT-2:Pre-LN、MHA、学习式位置嵌入、tied embedding
- 6 层 / d=384 / 6 heads / vocab=50257 / max_pos=1024,~30M
- 权重:随机初始化或可选 finetune 小权重
- 用途:lab 章节前半段教学,CPU 秒跑通

### 6.4 llama.py(TinyLlama-1.1B 兼容)

- RMSNorm + RoPE + SwiGLU + GQA(`num_kv_heads=4`,`num_heads=32`)
- 直接加载 HF `TinyLlama/TinyLlama-1.1B-Chat-v1.0` safetensors
- Loader 在 `models/llama_loader.py`,手写 HF→本格式 key 映射,**不依赖 transformers modeling code**
- CPU/MPS/CUDA 都能跑

### 6.5 共享 attention 调用链

```python
q, k, v = self.qkv_proj(x).split(...)
q, k = apply_rope(q, k, positions)        # llama 才有
backend.reshape_and_cache(k, v, kv_cache, slot_mapping)
out_p = backend.prefill(q[:np], ..., block_tables, query_lens, seq_lens)
out_d = backend.decode (q[np:], ..., block_tables, seq_lens)
out = self.o_proj(torch.cat([out_p, out_d]))
```

### 6.6 Sampler

独立 `sampler.py`,接 logits + `SamplingParams` → token id。最小集 greedy/temperature/top-p/top-k,~80 行。Beam 不做。

### 6.7 Streaming

`LLMEngine.step()` 返回新生成的 `(request_id, token_id, is_finished)` list;上层推到各请求的 `queue.Queue`;`generate_stream(prompt)` 是 Python generator。

---

## 7. 目录结构

```
ai-infra-tutorial/
├── code/mini-vllm/
│   ├── README.md
│   ├── pyproject.toml
│   ├── mini_vllm/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── engine.py
│   │   ├── scheduler.py
│   │   ├── sequence.py
│   │   ├── block_manager.py
│   │   ├── cache_engine.py
│   │   ├── model_runner.py
│   │   ├── sampler.py
│   │   ├── tokenizer.py
│   │   ├── backends/
│   │   │   ├── interface.py
│   │   │   ├── torch_backend.py
│   │   │   ├── triton_backend.py
│   │   │   └── reference.py
│   │   └── models/
│   │       ├── base.py
│   │       ├── toy_gpt.py
│   │       ├── llama.py
│   │       └── llama_loader.py
│   ├── examples/
│   │   ├── run_toy.py
│   │   ├── run_tinyllama.py
│   │   ├── stream_chat.py
│   │   └── bench.py
│   └── tests/
│       ├── test_block_manager.py
│       ├── test_attention.py
│       ├── test_scheduler.py
│       ├── test_prefix_cache.py
│       └── test_e2e.py
└── part5-serving-infra/
    ├── 16a-vllm-internals.md       # 末尾加链接到 lab
    └── 16a-lab-mini-vllm.md        # 新增
```

---

## 8. Tutorial 章节大纲(`16a-lab-mini-vllm.md`)

按 feature 渐进叙事,但读者跑同一份代码 + flag 切换:

1. **为什么要 PagedAttention** — naive contiguous KV cache 显存碎片问题(配图)
2. **第一版:Block KV Cache + naive Scheduler** — 跑通 toy GPT;讲 BlockManager 三队列、`block_size`、`slot_mapping`
3. **PagedAttention Kernel** — 先 Torch 参考实现讲清 gather + 在线 softmax,再 Triton kernel 逐行注释 decode 主循环
4. **接上真实模型 TinyLlama 1.1B** — RoPE/GQA/RMSNorm/HF 权重对齐 backend
5. **Continuous Batching** — 关 vs 开 throughput 曲线(bench.py 出图)
6. **Chunked Prefill** — 长 prompt 阻塞 decode 的 latency spike,开启后变平
7. **Prefix Caching + CoW** — 系统 prompt 共享场景的 TTFT 改善;hash chain 与 ref_count
8. **Swap to CPU** — 超额订阅显存;preemption 策略对比(swap vs recompute)
9. **从 mini 到真实 vLLM** — 简化掉了什么(async engine、TP、worker、量化、speculative)、源码对照表(`mini_vllm.X` ↔ `vllm.Y`)

每节末 2-3 个读者练习。

---

## 9. 测试策略

- **单元(pytest,CPU only)**:`test_block_manager` / `test_scheduler` / `test_prefix_cache`,完全 mock,不依赖 kernel,CI 必跑
- **Kernel 正确性(CPU + GPU)**:`test_attention` 随机输入,Triton ↔ Torch ↔ reference 三向 `allclose`;GPU 不可用自动 skip Triton
- **端到端 smoke(CPU,toy GPT)**:遍历 16 种 flag 组合(2^4),每种跑 5 个请求,断言:输出确定性 + 引擎结束后 `BlockManager.num_free_blocks == num_total_blocks`
- **端到端真实(GPU 可选,TinyLlama)**:固定 prompt + greedy,与 HF transformers 同模型同 prompt 比对 top-5 重叠度 ≥ 4
- **不做**:fuzz、property-based、性能回归门禁

---

## 10. Benchmark(`examples/bench.py`)

固定模型 + ShareGPT 抽样 50 prompts,跑 5 组对比并出 markdown 表 + matplotlib 图(章节直接贴):

| 配置 | throughput (tok/s) | TTFT p50/p99 | TBT p50/p99 | peak GPU mem |
|---|---|---|---|---|
| naive (全关) | | | | |
| + continuous batching | | | | |
| + chunked prefill | | | | |
| + prefix caching | | | | |
| + swap (oversubscribe) | | | | |

---

## 11. 风险与开放问题

1. **Triton kernel 正确性收敛时间**:GQA + paged + chunked prefill 三者叠加是 kernel 最难的部分,可能需要单独一轮调试。计划在实施时先做 `decode` kernel(最经典),`prefill` kernel 复用度低可后做。
2. **MPS 上 Triton 不可用**:Mac 用户只能跑 Torch backend,benchmark 章节明确标注"以下数据来自 NVIDIA GPU"。
3. **TinyLlama 权重 loader**:不依赖 transformers modeling code,但仍依赖 `safetensors` 读权重 + `tokenizers` 做分词。这两个是合理依赖。
4. **Swap 性能**:Python 同步实现的 GPU↔CPU 拷贝会比 vLLM 的 async cudaMemcpy 慢一截,benchmark 表里要诚实标注"swap 开销可能被高估"。
