# 第二十三章：推理优化工程实践

> **本章导读**：生产环境中，扩散模型的推理速度和内存占用是关键约束。本章介绍主流推理优化技术：量化（INT8/FP8/GGUF）、Flash Attention、torch.compile、xformers加速、模型并行，以及针对扩散模型的特殊优化（步数缓存、TeaCache等），帮助你将理论模型部署到实际应用中。

**前置知识**：第16-19章（架构），PyTorch基础
**预计学习时间**：110分钟

---

## 学习目标

完成本章学习后，你将能够：
1. 实施权重量化（FP16/BF16/INT8/INT4）并分析精度-速度权衡
2. 使用Flash Attention减少注意力层的内存开销
3. 使用torch.compile对扩散模型进行图优化
4. 理解扩散模型特有的加速技术（步骤缓存、CFG优化）
5. 设计高吞吐量推理服务（批处理、动态分辨率、模型预热）

---

## 23.1 精度降低

### 浮点格式对比

| 格式 | 位宽 | 指数位 | 尾数位 | 精度 | 显存 |
|------|------|--------|--------|------|------|
| FP32 | 32 | 8 | 23 | 高 | 基准 |
| BF16 | 16 | 8 | 7 | 中（动态范围大） | 50% |
| FP16 | 16 | 5 | 10 | 中（动态范围小） | 50% |
| INT8 | 8 | — | — | 低 | 25% |
| FP8 | 8 | 4/5 | 3/2 | 低-中 | 25% |
| INT4 | 4 | — | — | 极低 | 12.5% |

**BF16 vs FP16**：BF16的指数范围与FP32相同，训练时梯度不容易溢出；FP16精度更高但范围小。推理时两者效果接近。

### BF16/FP16推理

最简单的优化——使用半精度：

```python
# 方法1：模型直接转换
model = model.to(torch.bfloat16)

# 方法2：autocast（推理时）
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(x, t, text)
```

**SD1.5的显存对比**：
- FP32：~6GB
- BF16：~3GB
- INT8：~2GB

### INT8量化

使用bitsandbytes或torch.ao.quantization：

```python
import bitsandbytes as bnb

# 线性层量化为INT8
model = bnb.nn.Linear8bitLt(in_features, out_features, has_fp16_weights=False)

# 或用 load_in_8bit 参数（HuggingFace集成）
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    device_map="auto"
)
```

### GGUF/GGML量化（CPU友好）

llama.cpp引入的GGUF格式支持混合精度量化：
- Q4_K_M：4位量化，关键层保留更高精度
- Q8_0：8位均匀量化

FLUX等模型的GGUF版本在消费级GPU（甚至CPU）上可运行。

---

## 23.2 Flash Attention

### 标准注意力的内存瓶颈

标准注意力的显存复杂度为 $O(N^2)$（$N$是序列长度/空间维度）：

```python
# 标准实现
QK = Q @ K.T    # (B, h, N, N) — 当N=4096时，这个矩阵约64GB！
attn = QK.softmax(-1)
out = attn @ V
```

对于 $512 \times 512$ 图像，空间维度 $N = 512^2 / 64 = 4096$（在潜在空间 $64\times64$，经过patch化后）。

### Flash Attention原理

Dao et al. (2022) 的**Flash Attention**通过分块计算（Tiling）避免存储完整注意力矩阵：

1. 将Q、K、V分块到SRAM（快速片上内存）
2. 分块计算softmax（利用在线softmax算法）
3. 分块输出，无需存储 $N\times N$ 矩阵

**内存**：从 $O(N^2)$ 降至 $O(N)$（只存储输入输出）
**速度**：快2-4倍（减少HBM读写，提高计算密度）

### 使用Flash Attention

```python
# PyTorch 2.0+ 内置（SDPA - Scaled Dot Product Attention）
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,
    enable_math=False,
    enable_mem_efficient=False
):
    out = F.scaled_dot_product_attention(Q, K, V)

# 或直接使用（PyTorch自动选择最优实现）
out = F.scaled_dot_product_attention(Q, K, V)  # 自动选择flash/efficient

# xFormers库
import xformers.ops as xops
out = xops.memory_efficient_attention(Q, K, V)
```

---

## 23.3 torch.compile

### 图编译加速

`torch.compile`（PyTorch 2.0）将动态计算图编译为优化的静态图：

```python
import torch

# 编译整个模型
model_compiled = torch.compile(model, mode='reduce-overhead')

# 选择编译模式：
# 'default': 均衡（编译时间 ~1min）
# 'reduce-overhead': 减少Python开销，适合生产（需热身）
# 'max-autotune': 最大性能（编译时间长）
# 'fullgraph': 全图编译（限制更严格，速度最快）
```

**扩散模型特殊问题**：每步时间步 $t$ 不同，如果模型分支依赖 $t$，compile可能无效（动态形状问题）。

解决方案：
1. 使用`torch.compile(dynamic=True)`
2. 或将 $t$ 通过嵌入（Embedding/sinusoidal）注入，消除控制流对 $t$ 的依赖

**实测加速（SD1.5，A100）**：
- baseline（FP16）：~3s/image
- + compile：~2s/image（约1.5x加速）

---

## 23.4 扩散模型特有优化

### CFG并行

标准CFG需要两次串行前向（条件+无条件）。优化：将batch维度拼接：

```python
# 优化：合并批次，一次前向（但显存×2）
x_in = torch.cat([x] * 2)              # shape: (2B, C, H, W)
t_in = torch.cat([t] * 2)              # shape: (2B,)
cond_in = torch.cat([uncond, cond])     # shape: (2B, L, d)
eps_out = unet(x_in, t_in, cond_in)   # shape: (2B, C, H, W)
eps_uncond, eps_cond = eps_out.chunk(2)
```

**权衡**：一次前向 vs 两次前向：前者速度略快（减少Python overhead），但显存多一倍。

### 步骤缓存（Skip Steps / TGATE）

Ma et al. (2024) 发现，相邻步骤的注意力输出高度相似：可以复用。

**TGATE**：
- 前K步：正常计算所有注意力（包括交叉注意力）
- 后 $T - K$ 步：缓存交叉注意力输出，只计算自注意力

节省约25%计算量，图像质量几乎不变。

### TeaCache（时间步嵌入缓存）

Feng et al. (2024) 用时间步嵌入相似度预测输出差异：

```python
# 如果两个相邻步的时间嵌入相似度 > 阈值，复用上一步的中间特征
if similarity(t_emb[i], t_emb[i-1]) > threshold:
    features = cached_features  # 跳过部分计算
else:
    features = unet_forward(x_t, t)
    cached_features = features
```

---

## 23.5 显存优化

### 梯度检查点（推理时无需）

训练时使用梯度检查点减少显存：

```python
from torch.utils.checkpoint import checkpoint

def forward(x, t, context):
    for block in self.blocks:
        x = checkpoint(block, x, t, context, use_reentrant=False)
    return x
```

### CPU Offload

将不活跃的模型组件卸载到CPU：

```python
# 在diffusers中：
pipe.enable_model_cpu_offload()      # 逐模块offload（较慢但显存最省）
pipe.enable_sequential_cpu_offload() # 逐层offload（最省显存）

# 手动：将VAE卸载
pipe.vae.to('cpu')
# 使用时：
pipe.vae.to('cuda')
latent = pipe.vae.decode(z)
pipe.vae.to('cpu')
```

### VAE切片/Tiled VAE

高分辨率图像的VAE解码可能OOM，使用切片处理：

```python
pipe.enable_vae_slicing()    # 在批维度切片
pipe.enable_vae_tiling()     # 在空间维度切片（高分辨率图像）
```

---

## 代码实战

```python
"""
第二十三章代码实战：扩散模型推理优化
实现Flash Attention、torch.compile、CFG并行
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from contextlib import contextmanager
from typing import Optional


# ============================================================
# 1. 注意力实现对比：标准 vs Flash（SDPA）
# ============================================================

class StandardAttention(nn.Module):
    """标准注意力（显式计算注意力矩阵）"""
    
    def __init__(self, dim: int, n_heads: int = 8):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (B, N, dim)
        Returns:
            out: shape (B, N, dim)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                         # (3, B, h, N, d)
        Q, K, V = qkv.unbind(0)                                   # each: (B, h, N, d)
        
        # 标准注意力：O(N²) 显存
        scale = self.head_dim ** -0.5
        attn = (Q @ K.transpose(-2, -1)) * scale                  # (B, h, N, N)
        attn = attn.softmax(dim=-1)
        out = attn @ V                                             # (B, h, N, d)
        
        out = out.transpose(1, 2).reshape(B, N, C)                # (B, N, C)
        return self.proj(out)


class FlashAttention(nn.Module):
    """Flash Attention（通过PyTorch SDPA）"""
    
    def __init__(self, dim: int, n_heads: int = 8):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (B, N, dim)
        Returns:
            out: shape (B, N, dim)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv.unbind(0)                                   # each: (B, h, N, d)
        
        # Flash Attention：O(N) 显存，快2-4倍
        out = F.scaled_dot_product_attention(Q, K, V)             # (B, h, N, d)
        
        out = out.transpose(1, 2).reshape(B, N, C)                # (B, N, C)
        return self.proj(out)


# ============================================================
# 2. 精度基准测试
# ============================================================

def benchmark_precision(seq_len: int = 256, dim: int = 512, n_iters: int = 100):
    """对比不同精度的推理速度"""
    x_fp32 = torch.randn(4, seq_len, dim)
    x_fp16 = x_fp32.half()
    x_bf16 = x_fp32.bfloat16()
    
    model_fp32 = FlashAttention(dim)
    model_fp16 = FlashAttention(dim).half()
    model_bf16 = FlashAttention(dim).bfloat16()
    
    results = {}
    for dtype_name, model, x in [
        ('FP32', model_fp32, x_fp32),
        ('FP16', model_fp16, x_fp16),
        ('BF16', model_bf16, x_bf16),
    ]:
        # 预热
        with torch.no_grad():
            for _ in range(5):
                _ = model(x)
        
        start = time.time()
        with torch.no_grad():
            for _ in range(n_iters):
                _ = model(x)
        elapsed = (time.time() - start) / n_iters * 1000
        results[dtype_name] = elapsed
    
    print(f"注意力层精度对比（seq_len={seq_len}, dim={dim}，B=4）：")
    for dtype, ms in results.items():
        print(f"  {dtype}: {ms:.2f}ms/iter")
    return results


# ============================================================
# 3. CFG优化：批量合并 vs 串行
# ============================================================

class SimpleDiffusionBlock(nn.Module):
    """简单扩散块（用于演示）"""
    
    def __init__(self, dim: int = 256, seq_len: int = 64):
        super().__init__()
        self.attn = FlashAttention(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


def benchmark_cfg_batching():
    """对比串行CFG和批量CFG的速度"""
    dim, seq_len = 256, 64
    model = SimpleDiffusionBlock(dim, seq_len)
    B = 2
    
    x = torch.randn(B, seq_len, dim)
    n_iters = 50
    
    # 串行CFG（两次前向）
    start = time.time()
    with torch.no_grad():
        for _ in range(n_iters):
            out_uncond = model(x)         # 第一次：无条件
            out_cond = model(x)           # 第二次：有条件（此处相同，仅测速）
            eps = out_uncond + 7.5 * (out_cond - out_uncond)
    serial_time = (time.time() - start) / n_iters * 1000
    
    # 批量CFG（一次前向，batch×2）
    x_double = torch.cat([x, x], dim=0)  # shape: (2B, seq_len, dim)
    start = time.time()
    with torch.no_grad():
        for _ in range(n_iters):
            out = model(x_double)          # 一次前向
            out_uncond, out_cond = out.chunk(2, dim=0)
            eps = out_uncond + 7.5 * (out_cond - out_uncond)
    batch_time = (time.time() - start) / n_iters * 1000
    
    print(f"CFG方式对比（B={B}）：")
    print(f"  串行（2次前向）: {serial_time:.2f}ms")
    print(f"  批量（1次前向，batch×2）: {batch_time:.2f}ms")
    print(f"  加速比: {serial_time / batch_time:.2f}x")


# ============================================================
# 4. torch.compile演示
# ============================================================

def demo_torch_compile():
    """演示torch.compile加速"""
    model = SimpleDiffusionBlock(dim=256, seq_len=64)
    x = torch.randn(4, 64, 256)
    n_iters = 50
    
    # 预热基准
    with torch.no_grad():
        for _ in range(5):
            _ = model(x)
    
    # 基准速度
    start = time.time()
    with torch.no_grad():
        for _ in range(n_iters):
            _ = model(x)
    base_time = (time.time() - start) / n_iters * 1000
    
    # 编译模型
    compiled_model = torch.compile(model, mode='reduce-overhead')
    
    # 编译预热（第一次调用触发编译）
    print("编译中（首次调用）...")
    with torch.no_grad():
        for _ in range(3):  # 需要几次热身
            _ = compiled_model(x)
    
    # 编译后速度
    start = time.time()
    with torch.no_grad():
        for _ in range(n_iters):
            _ = compiled_model(x)
    compile_time = (time.time() - start) / n_iters * 1000
    
    print(f"torch.compile对比：")
    print(f"  基准: {base_time:.2f}ms/iter")
    print(f"  编译后: {compile_time:.2f}ms/iter")
    print(f"  加速比: {base_time / compile_time:.2f}x")


# ============================================================
# 5. 内存优化统计
# ============================================================

def memory_stats():
    """展示不同精度的显存占用"""
    # 模拟SD1.5 U-Net参数量（~860M参数）
    n_params = 860_000_000
    
    sizes = {
        'FP32': n_params * 4 / 1e9,
        'BF16/FP16': n_params * 2 / 1e9,
        'INT8': n_params * 1 / 1e9,
        'INT4 (GGUF Q4)': n_params * 0.5 / 1e9,
    }
    
    print("SD1.5 U-Net (~860M参数) 显存占用估算：")
    for fmt, gb in sizes.items():
        print(f"  {fmt:<20}: {gb:.2f} GB")
    
    print("\n推理总显存（含激活值，batch=1，512×512）：")
    # 近似估算（激活值约为权重的30-50%）
    for fmt, weight_gb in sizes.items():
        total = weight_gb * 1.4  # 激活值约40%
        print(f"  {fmt:<20}: ~{total:.2f} GB")


if __name__ == "__main__":
    print("=" * 60)
    print("扩散模型推理优化基准测试")
    print("=" * 60)
    
    print("\n[1] 精度对比")
    benchmark_precision(seq_len=64, dim=256, n_iters=20)
    
    print("\n[2] CFG批量优化")
    benchmark_cfg_batching()
    
    print("\n[3] 显存优化估算")
    memory_stats()
    
    # 注：torch.compile需要PyTorch 2.0+，在某些环境可能失败
    try:
        print("\n[4] torch.compile加速")
        demo_torch_compile()
    except Exception as e:
        print(f"torch.compile跳过（需要PyTorch 2.0+）: {e}")
    
    print("\n优化清单：")
    print("  ✅ BF16推理：显存减半，速度提升约30%")
    print("  ✅ Flash Attention：注意力层显存O(N)，速度提升2-4x")
    print("  ✅ torch.compile：整体加速1.3-2x")
    print("  ✅ CFG批量：减少Python调用overhead")
    print("  ✅ TGATE/TeaCache：跳步缓存，减少25%计算量")
    print("  ✅ CPU Offload：超大显存模型在小卡运行")
```

---

## 本章小结

| 优化方法 | 显存节省 | 速度提升 | 代码量 | 适用场景 |
|----------|----------|----------|--------|----------|
| BF16推理 | 50% | ~30% | 1行 | 几乎所有 |
| Flash Attention | ~80%(注意力) | 2-4x(注意力) | 1行 | 所有注意力层 |
| torch.compile | 无 | 1.3-2x | 1行+热身 | 固定分辨率推理 |
| CFG批量 | 无 | ~10-20% | 5行 | 有CFG的推理 |
| INT8量化 | 75% | ~略降 | 几行 | 显存受限场景 |
| CPU Offload | 节省GPU | 速度降低 | 1行 | 显存极度受限 |
| TGATE | 无 | ~25% | 中等 | 批量生产推理 |

---

## 练习题

### 基础题

**23.1** 在PyTorch中实现一个简单的BF16 vs FP32推理基准：使用一个多层MLP（4096→4096→4096），对比两种精度在GPU上的速度（使用`torch.cuda.Event`计时）和数值差异（最大绝对误差）。

**23.2** 解释为什么Flash Attention的显存复杂度是 $O(N)$ 而不是 $O(N^2)$。描述分块softmax（Online Softmax）的关键公式，说明如何在不存储完整 $N \times N$ 矩阵的情况下计算softmax。

### 中级题

**23.3** 实现步骤缓存（Step Caching）：在DDIM采样的后半段，跳过交叉注意力的重新计算（直接复用前一步的交叉注意力输出）。对比有/无缓存在20步采样时的时间和图像质量（用SSIM量化）。

**23.4** 对一个简单的扩散U-Net使用`torch.compile(mode='max-autotune')`，分析：(a) 编译时间；(b) 推理加速比；(c) 当输入分辨率变化时是否触发重新编译（使用`torch._dynamo.explain()`）。

### 提高题

**23.5** 实现一个简单的投机解码（Speculative Decoding）变体用于扩散模型：用一个小型"草稿模型"（少层U-Net）预测多步，再用完整模型验证。理论分析何时此方法有效（草稿模型与完整模型的预测相关性要求）。

---

## 练习答案

**23.1** 典型结果：FP32 ~5ms，BF16 ~2.5ms（GPU利用率高时差异明显）；最大绝对误差通常 <0.01（BF16精度足够用于推理）。

**23.2** 分块softmax：维护每块的最大值 $m_i$ 和指数和 $\ell_i$，通过归纳合并：$m_{new} = \max(m_i, m_{i+1})$，$\ell_{new} = \ell_i e^{m_i - m_{new}} + \ell_{i+1}e^{m_{i+1}-m_{new}}$。只存储当前块的 $Q$、$K$、$V$ 和累积输出，显存 $O(N \cdot B \cdot h \cdot d)$（线性）。

**23.3** 代码框架：在采样循环中，从第 $k$ 步开始缓存 `cross_attn_output`；后续步骤：跳过 `k = context @ W_k`，直接用缓存输出。SSIM通常 >0.95（高度相似）。

**23.4** `max-autotune` 编译时间约5-10分钟；加速比约1.5-2.5x（取决于模型大小）；分辨率变化通常触发重新编译，用`dynamic=True`可避免。

**23.5** 投机解码适用于草稿与完整模型预测高度一致的场景（Acceptance Rate >80%）。扩散模型中，相邻步骤预测相关性随 $\Delta t$ 减小而增大——适合少步（10步以内）采样。

---

## 延伸阅读

1. **Dao et al. (2022)**. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* — Flash Attention原文
2. **Ma et al. (2024)**. *TGATE: Cross-Attention Makes Inference Cumbersome in Text-to-Image Diffusion Models* — 步骤缓存
3. **Dettmers et al. (2022)**. *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale* — INT8量化（适用于扩散模型）
4. PyTorch官方文档. *torch.compile tutorial* — torch.compile使用指南

---

[← 上一章：流匹配与一致性模型](../part7-frontier-models/22-flow-matching-consistency-models.md)

[下一章：完整项目：文本到图像 →](./24-complete-project-text-to-image.md)

[返回目录](../README.md)
