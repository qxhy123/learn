# 第2章：KV Cache 原理

> KV Cache 是 LLM 推理从"不可用"变成"勉强可用"的关键技术。但它也带来了新的问题——显存管理，而这正是 vLLM 要解决的核心挑战。

---

## 学习目标

学完本章，你将能够：

1. 理解注意力计算中 Q、K、V 的角色以及为什么 K 和 V 可以被缓存
2. 解释 KV Cache 如何将 decode 阶段的计算复杂度从 O(n²) 降为 O(n)
3. 精确计算任意模型配置下的 KV Cache 显存占用
4. 理解 KV Cache 管理为什么困难，以及朴素实现的浪费在哪里
5. 建立对 KV Cache 优化技术（MQA、GQA、量化）的初步认识

---

## 2.1 从注意力计算说起

### Q、K、V 是什么？

在 Transformer 的自注意力机制中，每个 token 的表示会被线性变换为三个向量：

- **Query (Q)**：当前 token "想要查询"的信息
- **Key (K)**：每个 token "可以被查到"的标签
- **Value (V)**：每个 token 实际携带的信息

注意力计算的核心公式：

```
Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V
```

直觉上：
- Q × K^T 计算当前 token 与所有历史 token 的"相关度"
- softmax 将相关度转化为注意力权重
- 乘以 V 得到加权汇总的信息

### 为什么 K 和 V 可以缓存？

关键观察：**在自回归生成中，已经生成的 token 不会改变**。

当生成第 t 个 token 时：
- 需要计算第 t 个 token 的 Q_t
- 需要用 Q_t 与所有 token（1 到 t-1）的 K 和 V 做注意力计算
- 但 token 1 到 t-1 的 K 和 V 在之前的步骤中已经计算过了

如果不缓存，每生成一个新 token，都要重新计算所有历史 token 的 K 和 V——这是巨大的浪费。

### 有无 KV Cache 的对比

**无 KV Cache（朴素实现）：**

```
第 1 步: 计算 K₁, V₁        → 输出 token₁
第 2 步: 重新计算 K₁,K₂, V₁,V₂  → 输出 token₂
第 3 步: 重新计算 K₁,K₂,K₃, V₁,V₂,V₃ → 输出 token₃
...
第 n 步: 重新计算全部 K₁..Kₙ, V₁..Vₙ  → 输出 tokenₙ

总注意力计算量: O(1 + 2 + 3 + ... + n) = O(n²)
```

**有 KV Cache：**

```
第 1 步: 计算并缓存 K₁, V₁      → 输出 token₁
第 2 步: 计算 K₂,V₂, 从缓存读 K₁,V₁ → 输出 token₂
第 3 步: 计算 K₃,V₃, 从缓存读 K₁..K₂,V₁..V₂ → 输出 token₃
...
第 n 步: 计算 Kₙ,Vₙ, 从缓存读 K₁..Kₙ₋₁,V₁..Vₙ₋₁ → 输出 tokenₙ

每步新增计算量: O(1) 的 K/V 投影 + O(n) 的注意力
总注意力计算量: O(n + (n-1) + ... + 1) = O(n²)  (注意力本身)
但避免了 O(n²) 的 K/V 投影重复计算
```

KV Cache 的核心价值：**避免对历史 token 的 K、V 投影进行重复计算**。在 decode 阶段，每步只需要为 1 个新 token 做投影计算，然后与缓存中的 K、V 做注意力。

---

## 2.2 KV Cache 的数据结构

### 单层的 KV Cache

对于 Transformer 的一个注意力层，KV Cache 存储的是：

```
K Cache: [batch_size, num_kv_heads, seq_len, head_dim]
V Cache: [batch_size, num_kv_heads, seq_len, head_dim]
```

其中：
- `num_kv_heads`：KV 头数（在标准 MHA 中等于注意力头数，在 GQA/MQA 中更少）
- `seq_len`：当前已生成的序列长度（会随生成过程增长）
- `head_dim`：每个头的维度（通常为 128）

### 多层的总 KV Cache

整个模型的 KV Cache 是所有层的 KV Cache 的总和：

```
总 KV Cache = num_layers × 2(K和V) × num_kv_heads × seq_len × head_dim × dtype_size
```

### 实际模型的 KV Cache 大小

以几个常见模型为例（FP16，单请求，序列长度 2048）：

| 模型 | 层数 | KV头数 | head_dim | 单请求 KV Cache |
|------|------|--------|----------|----------------|
| Llama-3.1-8B | 32 | 8 (GQA) | 128 | 0.5 GB |
| Llama-3.1-70B | 80 | 8 (GQA) | 128 | 1.25 GB |
| Llama-2-7B | 32 | 32 (MHA) | 128 | 2.0 GB |
| Mistral-7B | 32 | 8 (GQA) | 128 | 0.5 GB |

注意 Llama-3 使用了 GQA（分组查询注意力），KV 头数只有 8，大幅减少了 KV Cache 大小。

---

## 2.3 KV Cache 的显存计算

精确计算 KV Cache 显存是评估推理系统容量的基础能力。

### 通用公式

```
KV Cache (bytes) = 2 × L × H_kv × D × S × B × sizeof(dtype)
```

其中：
- `2`：K 和 V 各一份
- `L`：Transformer 层数
- `H_kv`：KV 头数
- `D`：每个头的维度
- `S`：序列长度
- `B`：batch size
- `sizeof(dtype)`：数据类型大小（FP16=2, FP8=1, INT8=1）

### 计算示例：Llama-3.1-8B

```python
# Llama-3.1-8B 的配置
num_layers = 32
num_kv_heads = 8       # GQA: 8 个 KV 头
head_dim = 128
seq_len = 4096
batch_size = 1
dtype_bytes = 2        # FP16

kv_cache_bytes = 2 * num_layers * num_kv_heads * head_dim * seq_len * batch_size * dtype_bytes
kv_cache_gb = kv_cache_bytes / (1024**3)
print(f"KV Cache: {kv_cache_gb:.2f} GB")  # ≈ 1.0 GB

# 如果并发 64 个请求
print(f"64 并发: {kv_cache_gb * 64:.1f} GB")  # ≈ 64 GB —— 超过单卡显存！
```

### 关键洞察：KV Cache 是显存大户

对于 Llama-3.1-8B（FP16）：
- 模型权重：~16 GB
- 单请求 KV Cache（seq=4096）：~1 GB
- 64 并发 KV Cache：~64 GB

在高并发场景下，**KV Cache 的显存占用远超模型权重**。这就是为什么 KV Cache 的内存管理如此关键。

---

## 2.4 朴素 KV Cache 管理的问题

### 问题 1：预分配浪费

传统做法是在请求开始时，按 `max_seq_len` 预分配 KV Cache 空间。

```
预分配策略：

请求 A (实际 200 token): [██░░░░░░░░░░░░░░░░░░]  max=2048
请求 B (实际 500 token): [█████░░░░░░░░░░░░░░░]  max=2048
请求 C (实际 100 token): [█░░░░░░░░░░░░░░░░░░░]  max=2048

█ = 已使用    ░ = 预分配但未使用（浪费）

浪费率：1 - (200+500+100)/(2048×3) = 87%
```

### 问题 2：外部碎片

不同请求的 KV Cache 大小不同，分配和释放后会产生外部碎片，导致虽然总显存充足，但无法分配给新请求。

```
显存状态：

[请求A的KV][空闲][请求B的KV][空闲][请求C的KV][空闲]

三段空闲区域加起来可以容纳一个新请求，
但因为不连续，无法分配。
```

### 问题 3：无法共享

当多个请求共享相同前缀（如系统 prompt），每个请求仍然独立存储这部分 KV Cache：

```
请求 A: [系统prompt的KV][用户A的KV]
请求 B: [系统prompt的KV][用户B的KV]   ← 系统 prompt 的 KV Cache 重复存储
请求 C: [系统prompt的KV][用户C的KV]
```

### 这些问题有多严重？

vLLM 的论文中指出，在朴素实现中，**60-80% 的 KV Cache 显存被浪费**。这意味着：

- 本来可以同时服务 100 个请求的 GPU，实际只能服务 20-40 个
- 吞吐量损失 2-5 倍
- 相同的服务能力需要更多的 GPU，成本直接翻倍

这正是 vLLM 提出 PagedAttention 要解决的核心问题，我们将在第 7 章详细讲解。

---

## 2.5 KV Cache 优化技术概览

在深入 vLLM 的解决方案之前，先了解几种主要的 KV Cache 优化方向。

### 减少 KV Cache 大小

**GQA（Grouped-Query Attention）**

将多个 Query 头共享同一组 K、V 头。例如 Llama-3 的 32 个 Query 头共享 8 个 KV 头，KV Cache 缩小为 MHA 的 1/4。

```
MHA (Multi-Head Attention):     32 Q 头, 32 KV 头  → KV Cache = 32 × head_dim
GQA (Grouped-Query Attention):  32 Q 头,  8 KV 头  → KV Cache =  8 × head_dim (节省 4×)
MQA (Multi-Query Attention):    32 Q 头,  1 KV 头  → KV Cache =  1 × head_dim (节省 32×)
```

**KV Cache 量化**

将 KV Cache 从 FP16 量化为 INT8 或 FP8，显存减半，但可能影响精度。

### 减少 KV Cache 的重复计算

**前缀缓存（Prefix Caching）**

当多个请求共享相同前缀时，只计算一次并缓存 KV Cache，后续请求直接复用。vLLM 原生支持这一特性（第 16 章详解）。

### 更高效的内存管理

**PagedAttention**

vLLM 的核心创新：借鉴操作系统的虚拟内存和分页机制，将 KV Cache 按固定大小的块分配，消除碎片和预分配浪费（第 7 章详解）。

---

## 本章小结

| 概念 | 要点 |
|------|------|
| KV Cache 的作用 | 缓存已计算的 K、V 向量，避免 decode 阶段的重复计算 |
| 显存占用公式 | 2 × 层数 × KV头数 × head_dim × 序列长度 × batch × dtype_size |
| 高并发下 KV Cache 是显存大户 | 并发请求多时，KV Cache 远超模型权重 |
| 朴素管理的浪费 | 预分配浪费、外部碎片、无法共享 → 60-80% 显存被浪费 |
| GQA 的作用 | 减少 KV 头数，直接缩小 KV Cache |
| PagedAttention 的目标 | 消除碎片和浪费，这是 vLLM 的核心创新 |

---

## 动手实验

### 实验 1：计算不同模型的 KV Cache 显存

```python
def calc_kv_cache_gb(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int = 1,
    dtype_bytes: int = 2,  # FP16
) -> float:
    """计算 KV Cache 的显存占用 (GB)"""
    total_bytes = (
        2 * num_layers * num_kv_heads * head_dim
        * seq_len * batch_size * dtype_bytes
    )
    return total_bytes / (1024**3)

# 常见模型配置
models = {
    "Llama-3.1-8B":  {"num_layers": 32, "num_kv_heads": 8,  "head_dim": 128},
    "Llama-3.1-70B": {"num_layers": 80, "num_kv_heads": 8,  "head_dim": 128},
    "Llama-2-7B":    {"num_layers": 32, "num_kv_heads": 32, "head_dim": 128},
    "Mistral-7B":    {"num_layers": 32, "num_kv_heads": 8,  "head_dim": 128},
    "Qwen-2.5-72B":  {"num_layers": 80, "num_kv_heads": 8,  "head_dim": 128},
}

for name, cfg in models.items():
    for seq_len in [2048, 8192, 32768]:
        gb = calc_kv_cache_gb(**cfg, seq_len=seq_len)
        print(f"{name:20s} seq={seq_len:5d}  KV Cache = {gb:.2f} GB")
    print()
```

### 实验 2：观察 GQA 对 KV Cache 的影响

```python
# 对比 MHA 和 GQA 的 KV Cache 大小
seq_len = 4096

# Llama-2-7B (MHA: 32 KV heads)
mha = calc_kv_cache_gb(32, 32, 128, seq_len)
# Llama-3.1-8B (GQA: 8 KV heads)
gqa = calc_kv_cache_gb(32, 8, 128, seq_len)

print(f"MHA (32 KV heads): {mha:.2f} GB")
print(f"GQA (8 KV heads):  {gqa:.2f} GB")
print(f"GQA 节省:           {(1 - gqa/mha)*100:.0f}%")
```

### 实验 3：估算单 GPU 的最大并发

```python
# 假设使用 A100 80GB GPU，FP16 推理 Llama-3.1-8B
gpu_memory_gb = 80
model_weight_gb = 16  # FP16 权重
overhead_gb = 2       # CUDA context, activations 等

available_for_kv = gpu_memory_gb - model_weight_gb - overhead_gb

kv_per_request = calc_kv_cache_gb(32, 8, 128, seq_len=4096)

max_concurrent = int(available_for_kv / kv_per_request)
print(f"可用显存: {available_for_kv:.0f} GB")
print(f"每请求 KV Cache: {kv_per_request:.2f} GB")
print(f"最大并发请求数: {max_concurrent}")

# 思考：如果使用 INT8 KV Cache 量化，能多服务多少请求？
```

---

## 练习题

### 基础题

1. 为什么 KV Cache 只缓存 K 和 V，不缓存 Q？
2. 如果一个模型有 40 层、8 个 KV 头、head_dim=128，序列长度为 4096，在 FP16 下单请求的 KV Cache 占用多少 GB？
3. 什么是 GQA？它如何减少 KV Cache 的大小？

### 实践题

4. 使用上面的 `calc_kv_cache_gb` 函数，计算 Llama-3.1-70B 在序列长度 32768、并发 16 个请求时的总 KV Cache 占用。这需要几张 A100 80GB？
5. 对比 Llama-2-7B（MHA）和 Llama-3.1-8B（GQA）在相同条件下的最大并发数差异。

### 思考题

6. 为什么说"KV Cache 管理是 LLM 推理系统的核心挑战"？如果 KV Cache 很小（比如模型只有 4 层），这个问题还重要吗？
7. 预分配 KV Cache 的策略为什么会导致 60-80% 的浪费？在什么场景下浪费最严重？
