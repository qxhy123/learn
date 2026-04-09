# 第11章：优化技术

> **学习目标**
>
> 1. 理解Adam和AdamW优化器的区别及各自的适用场景
> 2. 掌握权重衰减的正确使用方法，避免常见误区
> 3. 理解梯度累积的原理，能够在内存受限时模拟大批量训练
> 4. 掌握混合精度训练的实现，显著降低显存占用和训练时间
> 5. 能够将上述技术组合，实现高效的Transformer训练流程

---

## 11.1 Adam优化器

### 11.1.1 SGD的局限性

随机梯度下降（SGD）是最基础的优化算法：

$$\theta_{t+1} = \theta_t - \eta \cdot g_t$$

其中 $\eta$ 是学习率，$g_t$ 是当前批次的梯度。

SGD有几个明显的局限性：

**问题一：对所有参数使用相同的学习率**

神经网络中不同参数的梯度量级差异很大。Embedding层的参数很少被更新（稀疏梯度），而输出层的参数几乎每步都更新。用统一学习率对两者来说都不理想。

**问题二：在平坦区域收敛缓慢**

Loss曲面上存在大片平坦区域，SGD在这些区域步长太小，收敛极慢。

**问题三：容易在鞍点附近震荡**

高维空间中鞍点数量远多于局部最小值。SGD在鞍点附近梯度接近零，容易卡住。

带动量的SGD（SGD with Momentum）部分解决了问题二和三：

$$v_t = \mu v_{t-1} + g_t$$
$$\theta_{t+1} = \theta_t - \eta \cdot v_t$$

动量项 $v_t$ 积累了历史梯度方向，帮助穿越平坦区域。但问题一依然存在。

### 11.1.2 Adam：动量 + 自适应学习率

Adam（Adaptive Moment Estimation）同时解决了以上三个问题。它维护两个状态：

**一阶矩（梯度的指数移动平均，类似动量）：**

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

**二阶矩（梯度平方的指数移动平均，用于自适应学习率）：**

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

典型超参数值：$\beta_1 = 0.9$，$\beta_2 = 0.999$，$\epsilon = 10^{-8}$。

**参数更新规则：**

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

分母 $\sqrt{\hat{v}_t}$ 对每个参数单独做了"归一化"：如果某个参数的历史梯度很大（即该方向变化频繁），则有效学习率会缩小；反之则放大。这就是"自适应"的含义。

### 11.1.3 偏差修正

注意到 $m_t$ 和 $v_t$ 初始化为零，在训练早期会有较大的偏差（bias）。例如第一步：

$$m_1 = (1 - \beta_1) g_1 = 0.1 \cdot g_1$$

远小于真实的梯度均值。Adam通过偏差修正（bias correction）来补偿：

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

随着 $t$ 增大，$\beta_1^t \to 0$，修正项趋近于1，偏差修正自动消退。

### 11.1.4 Adam的完整算法

```
初始化：m_0 = 0, v_0 = 0, t = 0
for each step:
    t = t + 1
    g_t = ∇L(θ_{t-1})           # 计算梯度
    m_t = β₁·m_{t-1} + (1-β₁)·g_t   # 更新一阶矩
    v_t = β₂·v_{t-1} + (1-β₂)·g_t²  # 更新二阶矩
    m̂_t = m_t / (1 - β₁^t)      # 偏差修正
    v̂_t = v_t / (1 - β₂^t)      # 偏差修正
    θ_t = θ_{t-1} - η·m̂_t / (√v̂_t + ε)  # 参数更新
```

### 11.1.5 Adam的内存开销

Adam需要为每个参数维护两个额外状态（$m_t$ 和 $v_t$），因此内存占用是SGD的3倍。对于包含数十亿参数的大型Transformer，这是一个不可忽视的成本。

---

## 11.2 AdamW与权重衰减

### 11.2.1 L2正则化 vs 权重衰减

在深度学习中，防止过拟合的一个常用方法是**L2正则化**：在损失函数中加入参数平方和的惩罚项：

$$L_{reg} = L + \frac{\lambda}{2} \|\theta\|^2$$

对 $\theta$ 求梯度：

$$g_t^{reg} = g_t + \lambda \theta_t$$

将带正则化的梯度代入SGD更新：

$$\theta_{t+1} = \theta_t - \eta (g_t + \lambda \theta_t) = (1 - \eta\lambda)\theta_t - \eta g_t$$

注意到 $(1 - \eta\lambda)\theta_t$ 这一项：每步参数都会乘以一个小于1的系数，即参数在"衰减"。这就是为什么L2正则化在SGD中等价于**权重衰减（Weight Decay）**。

### 11.2.2 为什么Adam中两者不等价

将L2正则化梯度代入Adam更新：

$$\hat{m}_t \leftarrow \text{基于} (g_t + \lambda \theta_t) \text{的一阶矩}$$
$$\hat{v}_t \leftarrow \text{基于} (g_t + \lambda \theta_t)^2 \text{的二阶矩}$$

问题在于：Adam会对正则化项也做**自适应缩放**。

具体来说，如果某个参数的梯度 $g_t$ 历史上很大（$\hat{v}_t$ 大），那么正则化惩罚项 $\lambda \theta_t$ 也会被同样地缩小。结果是：**历史梯度大的参数受到的正则化效果更弱**，而这通常恰恰是我们最需要正则化的参数！

这意味着，在Adam中直接使用L2正则化，正则化效果是"失真的"，不能达到预期的防过拟合效果。

### 11.2.3 AdamW的修正

AdamW（Adam with decoupled Weight decay）的思路是：将权重衰减从梯度计算中**解耦**出来，直接作用于参数更新：

$$\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)$$

等价写法（先做Adam更新，再做权重衰减）：

$$\theta_{t+1}^{Adam} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$
$$\theta_{t+1} = \theta_{t+1}^{Adam} - \eta \lambda \theta_t$$

这样权重衰减的强度对所有参数是**均等的**，不受Adam自适应学习率的影响。实验证明AdamW在大多数Transformer训练任务上优于原始Adam。

### 11.2.4 哪些参数应该衰减

不是所有参数都应该施加权重衰减：

| 参数类型 | 是否衰减 | 原因 |
|----------|----------|------|
| 线性层权重（Weight matrix） | 是 | 防止权重过大，是主要正则化目标 |
| 线性层偏置（Bias） | 否 | 偏置参数少，衰减收益有限 |
| Embedding矩阵 | 否（通常）| 词向量本身不应被强迫趋近于零 |
| LayerNorm的 $\gamma$、$\beta$ | 否 | 归一化层的缩放参数不应衰减 |
| 位置编码（可训练） | 否（通常）| 同Embedding |

**判断规则**：一维参数（偏置、LayerNorm参数）通常不衰减；二维及以上参数（权重矩阵）通常衰减。

```python
def get_param_groups(model, weight_decay):
    """将参数分为两组：需要衰减的和不需要衰减的"""
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 一维参数（偏置、LayerNorm）不衰减
        if param.dim() == 1 or "bias" in name or "layernorm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
```

---

## 11.3 梯度累积

### 11.3.1 为什么需要梯度累积

在Transformer训练中，批量大小（batch size）对模型质量有重要影响：

- BERT预训练使用批量大小256
- GPT-3训练使用批量大小约3.2M tokens
- 大批量通常能带来更稳定的梯度估计和更好的最终性能

然而，GPU显存是有限的。一张A100（80GB）在训练7B参数模型时，批量大小可能只能到4-8。

**梯度累积（Gradient Accumulation）** 是解决这个矛盾的方法：将多个小批次的梯度**累加**后再做一次参数更新，效果等价于使用更大的批量。

### 11.3.2 等效批量大小

$$\text{等效批量大小} = \text{微批量大小（micro batch size）} \times \text{累积步数（accumulation steps）}$$

例如：
- 微批量大小 = 4（受显存限制）
- 累积步数 = 8
- 等效批量大小 = 32

### 11.3.3 数学原理

标准批量梯度：

$$g_{batch} = \frac{1}{N} \sum_{i=1}^{N} \nabla L(x_i, \theta)$$

梯度累积等价形式（将 $N$ 个样本分成 $K$ 个微批量，每批 $n = N/K$ 个样本）：

$$g_{batch} = \frac{1}{N} \sum_{k=1}^{K} \sum_{i \in \text{batch}_k} \nabla L(x_i, \theta) = \frac{1}{K} \sum_{k=1}^{K} g_k$$

其中 $g_k = \frac{1}{n} \sum_{i \in \text{batch}_k} \nabla L(x_i, \theta)$ 是第 $k$ 个微批量的梯度均值。

**注意**：需要对每个微批量计算梯度均值（除以微批量大小），然后将 $K$ 个均值**平均**（再除以 $K$），才能得到与大批量等价的梯度。

### 11.3.4 实现方法

```python
accumulation_steps = 8  # 累积8步 = 等效批量增大8倍

optimizer.zero_grad()

for step, batch in enumerate(dataloader):
    # 前向传播 + 损失计算
    outputs = model(**batch)
    loss = outputs.loss

    # 将损失除以累积步数，保证梯度均值等价
    loss = loss / accumulation_steps

    # 反向传播（梯度累积，不清零）
    loss.backward()

    # 每 accumulation_steps 步才更新参数
    if (step + 1) % accumulation_steps == 0:
        # 可选：梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

### 11.3.5 与分布式训练的关系

在数据并行（Data Parallel）训练中，每个GPU处理数据的一个子集，梯度在所有GPU间进行**AllReduce**同步。

梯度累积与数据并行的关系：

```
等效批量大小 = 微批量大小 × 累积步数 × GPU数量
```

在启用梯度累积时，为了避免在每个微批次都触发AllReduce（开销大），PyTorch提供了 `no_sync()` 上下文管理器：

```python
for step, batch in enumerate(dataloader):
    # 累积阶段：禁用梯度同步，节省通信开销
    if (step + 1) % accumulation_steps != 0:
        with model.no_sync():
            loss = model(**batch).loss / accumulation_steps
            loss.backward()
    else:
        # 最后一步：正常同步梯度
        loss = model(**batch).loss / accumulation_steps
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

---

## 11.4 混合精度训练

### 11.4.1 浮点数格式对比

| 格式 | 符号位 | 指数位 | 尾数位 | 范围 | 精度 | 显存 |
|------|--------|--------|--------|------|------|------|
| FP32 | 1 | 8 | 23 | $\pm 3.4 \times 10^{38}$ | 约7位十进制 | 4字节 |
| FP16 | 1 | 5 | 10 | $\pm 65504$ | 约3位十进制 | 2字节 |
| BF16 | 1 | 8 | 7 | $\pm 3.4 \times 10^{38}$ | 约2位十进制 | 2字节 |

**FP16的优势**：显存减半，现代GPU的FP16计算速度是FP32的2-8倍（Tensor Core加速）。

**FP16的风险**：
1. **溢出**：最大值仅65504，梯度若超过此值则变为 `inf`
2. **下溢**：最小正规化数约 $6 \times 10^{-5}$，很小的梯度会变为0
3. **精度损失**：尾数位少，累积误差较大

**BF16的优势**：与FP32相同的指数范围，不会溢出。Google TPU和NVIDIA Ampere架构（A100等）原生支持BF16。现代大模型训练通常优选BF16。

### 11.4.2 混合精度训练的核心思想

不是把所有计算都降低精度，而是**混合使用**：

- **前向传播和反向传播**：使用FP16（速度快、显存少）
- **参数主副本（Master Weights）**：保存FP32（保证精度）
- **优化器状态（$m_t$、$v_t$）**：FP32（精度关键）

更新流程：

```
FP32主参数 → 转换为FP16 → 前向计算 → 反向计算（FP16梯度）
→ 转换为FP32梯度 → FP32优化器更新 → 更新FP32主参数
```

### 11.4.3 动态损失缩放（Dynamic Loss Scaling）

为了解决FP16梯度下溢的问题，混合精度训练使用**损失缩放（Loss Scaling）**：

在反向传播前，将损失乘以一个大的缩放因子 $S$（如 $2^{15} = 32768$）：

$$L_{scaled} = L \times S$$

由于反向传播的线性性质，所有梯度也会乘以 $S$：

$$g_{scaled} = g \times S$$

这将小梯度放大到FP16可表示的范围。在优化器更新前，再将梯度除以 $S$ 恢复原始量级：

$$g_{original} = g_{scaled} / S$$

**动态缩放**策略：
- 如果连续 $N$（默认2000）步都没有出现梯度溢出（`inf`/`nan`），则将缩放因子翻倍
- 如果检测到梯度溢出，则将缩放因子减半，并**跳过本步更新**（溢出的梯度不可用）

### 11.4.4 PyTorch AMP实现

PyTorch的 `torch.cuda.amp` 模块提供了自动混合精度（Automatic Mixed Precision）支持：

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()  # 动态损失缩放器

for batch in dataloader:
    optimizer.zero_grad()

    # autocast自动选择FP16/FP32
    with autocast():
        outputs = model(**batch)
        loss = outputs.loss

    # 使用scaler进行缩放后的反向传播
    scaler.scale(loss).backward()

    # 梯度裁剪（需要先unscale）
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # 更新参数（内部会检查梯度是否有inf/nan）
    scaler.step(optimizer)

    # 更新缩放因子
    scaler.update()

    scheduler.step()
```

`autocast` 会自动决定每个操作使用哪种精度：
- 矩阵乘法（`mm`、`bmm`）：FP16（走Tensor Core）
- 归一化、Softmax、损失函数：FP32（精度敏感）

### 11.4.5 BF16训练（无需Loss Scaling）

在支持BF16的硬件（A100、H100、TPU）上，可以直接使用BF16而无需动态损失缩放：

```python
# PyTorch 1.10+
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    outputs = model(**batch)
    loss = outputs.loss

# BF16不需要GradScaler，直接反向传播
loss.backward()
optimizer.step()
```

---

## 11.5 分布式训练基础

### 11.5.1 数据并行

**DataParallel（DP）**：PyTorch的早期实现，将模型复制到多个GPU，每个GPU处理数据的一个子集，梯度在主GPU上汇总。

```python
model = nn.DataParallel(model)  # 简单，但效率低
```

DP的问题：主GPU成为瓶颈，GPU间负载不均衡，不支持多机训练。

**DistributedDataParallel（DDP）**：推荐使用。每个进程（GPU）有独立的模型副本，梯度通过高效的**Ring-AllReduce**算法同步：

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 初始化进程组
dist.init_process_group(backend="nccl")

model = DDP(model, device_ids=[local_rank])
```

DDP的关键优势：
- 通信与计算重叠（反向传播时同步梯度，不等待）
- 支持多机多卡
- 无主GPU瓶颈

### 11.5.2 模型并行

当单个GPU无法容纳整个模型时，需要**模型并行（Model Parallelism）**：

**张量并行（Tensor Parallelism）**：将单个层的权重矩阵切分到多个GPU。Megatron-LM使用此方法，将注意力头分配到不同GPU。

**流水线并行（Pipeline Parallelism）**：将模型的不同层分配到不同GPU，形成"流水线"。存在"流水线气泡"问题（部分GPU在等待）。

**序列并行（Sequence Parallelism）**：将序列维度切分，适合处理超长上下文。

### 11.5.3 ZeRO优化

ZeRO（Zero Redundancy Optimizer）是DeepSpeed提出的内存优化技术，解决数据并行中每个GPU都保存完整优化器状态的冗余问题。

**三个优化阶段：**

| ZeRO阶段 | 切分内容 | 内存节省 | 通信开销 |
|----------|----------|----------|----------|
| ZeRO-1 | 优化器状态（$m_t$、$v_t$） | 4倍 | 少 |
| ZeRO-2 | + 梯度 | 8倍 | 中 |
| ZeRO-3 | + 参数 | $N_d$倍（$N_d$=GPU数） | 多 |

**内存对比**（以7B参数模型、FP16训练为例）：

- 无ZeRO：每卡约112GB（超过单卡80GB上限）
- ZeRO-1：每卡约35GB
- ZeRO-3：每卡约4GB（16卡）

### 11.5.4 PyTorch FSDP

**Fully Sharded Data Parallel（FSDP）** 是PyTorch原生的ZeRO-3实现，无需依赖DeepSpeed：

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# 指定按Transformer层包装（每层独立分片）
auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={TransformerDecoderLayer}
)

model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,
    mixed_precision=MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    )
)
```

FSDP工作原理：
1. 前向传播时，在需要某一层时**AllGather**完整参数
2. 该层计算完成后**立即丢弃**参数（只保留本GPU的分片）
3. 反向传播时再次AllGather，计算梯度后丢弃
4. 梯度通过**ReduceScatter**同步（每GPU只保留对应分片的梯度）

---

## 本章小结

| 技术 | 解决的问题 | 主要开销 | 适用场景 |
|------|-----------|----------|----------|
| Adam | 梯度量级差异大、收敛慢 | 2倍额外内存（$m_t$、$v_t$） | 所有Transformer训练 |
| AdamW | Adam中L2正则化失真 | 几乎无额外开销 | 需要正则化时（推荐替代Adam） |
| 梯度累积 | 显存不足以使用大批量 | 每步多次前向/反向 | 单GPU/显存受限场景 |
| 混合精度（FP16） | 训练速度慢、显存占用大 | 需要维护FP32主参数 | 拥有Tensor Core的GPU |
| 混合精度（BF16） | 同上，且无溢出风险 | 无需Loss Scaling | Ampere及以上架构GPU |
| DDP | 单GPU无法充分利用算力 | 梯度通信开销 | 多GPU，模型可放入单卡 |
| ZeRO/FSDP | 模型过大无法放入单卡 | 参数AllGather通信 | 大模型训练 |

**推荐的Transformer训练配置**：

```
小模型（<1B参数）：AdamW + 梯度累积 + BF16
中型模型（1-13B）：AdamW + DDP + BF16 + 梯度累积
大型模型（>13B）：AdamW + FSDP/ZeRO-3 + BF16
```

---

## 代码实战：完整优化配置

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import functools


# ============================================================
# 1. AdamW参数组配置（区分衰减/不衰减参数）
# ============================================================

def configure_optimizer(model, learning_rate=3e-4, weight_decay=0.1, betas=(0.9, 0.95)):
    """
    配置AdamW优化器，正确区分需要权重衰减的参数。

    注意：
    - 二维及以上参数（权重矩阵）施加衰减
    - 一维参数（偏置、LayerNorm的gamma/beta）不衰减
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 判断标准：1D参数（偏置/归一化层参数）不衰减
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
            "lr": learning_rate,
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
            "lr": learning_rate,
        },
    ]

    # beta2=0.95适合语言模型（比默认的0.999更快适应梯度变化）
    optimizer = AdamW(param_groups, betas=betas, eps=1e-8)

    print(f"参数统计：")
    print(f"  需要衰减：{sum(p.numel() for p in decay_params):,} 个参数")
    print(f"  不衰减：  {sum(p.numel() for p in no_decay_params):,} 个参数")

    return optimizer


# ============================================================
# 2. 梯度累积训练循环
# ============================================================

def train_with_gradient_accumulation(
    model,
    dataloader,
    optimizer,
    scheduler,
    accumulation_steps=8,
    max_grad_norm=1.0,
    device="cuda",
):
    """
    带梯度累积的训练循环。

    等效批量大小 = dataloader batch size × accumulation_steps
    """
    model.train()
    total_loss = 0.0
    optimizer_steps = 0

    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        # 将数据移到设备
        batch = {k: v.to(device) for k, v in batch.items()}

        # 前向传播
        outputs = model(**batch)

        # 损失归一化：除以累积步数，保证梯度等价于大批量
        loss = outputs.loss / accumulation_steps
        total_loss += outputs.loss.item()

        # 反向传播（梯度累积，不清零）
        loss.backward()

        # 每 accumulation_steps 步执行一次参数更新
        if (step + 1) % accumulation_steps == 0:
            # 梯度裁剪（防止梯度爆炸）
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_grad_norm
            )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            optimizer_steps += 1

            avg_loss = total_loss / accumulation_steps
            print(f"步骤 {optimizer_steps}: loss={avg_loss:.4f}, grad_norm={grad_norm:.4f}")
            total_loss = 0.0

    return optimizer_steps


# ============================================================
# 3. 混合精度训练循环（FP16 + GradScaler）
# ============================================================

def train_with_amp(
    model,
    dataloader,
    optimizer,
    scheduler,
    use_bf16=False,
    accumulation_steps=4,
    max_grad_norm=1.0,
    device="cuda",
):
    """
    混合精度训练循环，支持FP16（GradScaler）和BF16。

    BF16在Ampere+架构上推荐使用（无需Loss Scaling）。
    """
    model.train()

    # BF16不需要GradScaler
    use_grad_scaler = not use_bf16
    scaler = GradScaler(enabled=use_grad_scaler)

    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}

        # 自动混合精度前向传播
        with autocast(device_type="cuda", dtype=amp_dtype):
            outputs = model(**batch)
            loss = outputs.loss / accumulation_steps

        if use_grad_scaler:
            # FP16：使用scaler缩放损失后反向传播
            scaler.scale(loss).backward()
        else:
            # BF16：直接反向传播
            loss.backward()

        if (step + 1) % accumulation_steps == 0:
            if use_grad_scaler:
                # 梯度裁剪前先unscale
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                # 更新参数（若梯度有inf/nan则跳过本步）
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()


# ============================================================
# 4. 完整优化配置示例（将所有技术组合）
# ============================================================

class TransformerConfig:
    vocab_size: int = 32000
    hidden_size: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    max_seq_len: int = 2048


def create_training_setup(model, total_steps, warmup_steps=2000):
    """
    创建完整的训练配置，包含：
    - AdamW优化器（参数分组）
    - Warmup + Cosine衰减学习率调度
    - 混合精度（BF16）
    - 梯度累积
    """
    # 1. 配置AdamW优化器
    optimizer = configure_optimizer(
        model,
        learning_rate=3e-4,
        weight_decay=0.1,
        betas=(0.9, 0.95),
    )

    # 2. 带Warmup的余弦退火学习率
    def lr_lambda(current_step):
        # Warmup阶段：线性增长
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # 余弦退火阶段
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.1, 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress)).item()))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 3. 返回完整配置
    config = {
        "optimizer": optimizer,
        "scheduler": scheduler,
        "use_bf16": torch.cuda.is_bf16_supported(),  # 自动检测硬件
        "accumulation_steps": 8,
        "max_grad_norm": 1.0,
    }

    print(f"训练配置：")
    print(f"  精度：{'BF16' if config['use_bf16'] else 'FP16'}")
    print(f"  梯度累积步数：{config['accumulation_steps']}")
    print(f"  Warmup步数：{warmup_steps}")
    print(f"  总训练步数：{total_steps}")

    return config


# ============================================================
# 5. 完整训练循环（整合所有技术）
# ============================================================

def full_training_loop(model, train_dataloader, eval_dataloader, config):
    """
    整合了以下技术的完整训练循环：
    - AdamW with weight decay
    - 梯度累积
    - 混合精度（BF16/FP16）
    - 梯度裁剪
    - 学习率调度
    """
    device = next(model.parameters()).device
    optimizer = config["optimizer"]
    scheduler = config["scheduler"]
    accumulation_steps = config["accumulation_steps"]
    use_bf16 = config["use_bf16"]
    max_grad_norm = config["max_grad_norm"]
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    scaler = GradScaler(enabled=not use_bf16)

    model.train()
    optimizer.zero_grad()

    global_step = 0
    log_interval = 10  # 每10个优化步打印一次

    for epoch in range(config.get("num_epochs", 3)):
        epoch_loss = 0.0

        for micro_step, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            # 混合精度前向传播
            with autocast(device_type="cuda", dtype=amp_dtype):
                outputs = model(**batch)
                loss = outputs.loss / accumulation_steps

            # 反向传播
            if not use_bf16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss += outputs.loss.item()

            # 梯度累积完成，执行优化步
            if (micro_step + 1) % accumulation_steps == 0:
                # 梯度裁剪
                if not use_bf16:
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                )

                # 参数更新
                if not use_bf16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # 日志
                if global_step % log_interval == 0:
                    avg_loss = epoch_loss / (log_interval * accumulation_steps)
                    current_lr = scheduler.get_last_lr()[0]
                    print(
                        f"Epoch {epoch+1} | Step {global_step} | "
                        f"Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | "
                        f"Grad Norm: {grad_norm:.4f}"
                    )
                    epoch_loss = 0.0
```

---

## 练习题

### 基础题

**练习 11.1**（基础）

以下代码使用Adam实现了L2正则化，请指出其问题所在，并改写为正确的AdamW实现：

```python
# 有问题的代码
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.01   # 问题在哪里？
)
```

---

**练习 11.2**（基础）

给定以下训练条件：
- GPU显存限制：实际批量大小只能设为 8
- 目标等效批量大小：256
- GPU数量：4

请计算需要的梯度累积步数，并在代码中标注正确的损失归一化位置：

```python
# 请填写 accumulation_steps 的值
accumulation_steps = ___

for step, batch in enumerate(dataloader):
    outputs = model(**batch)
    loss = outputs.loss

    # 在哪里需要除以 accumulation_steps？
    loss.backward()

    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

### 中级题

**练习 11.3**（中级）

请实现一个 `get_param_groups` 函数，满足以下要求：

1. Embedding层参数（`embed_tokens`）：不衰减
2. LayerNorm参数（`layer_norm`、`layernorm`）：不衰减
3. 所有偏置参数（`bias`）：不衰减
4. 其余参数：施加权重衰减 0.1

验证：调用函数后打印各组参数数量，确认分组正确。

---

**练习 11.4**（中级）

以下混合精度训练代码有一个常见错误，请找出并修正：

```python
scaler = GradScaler()

for batch in dataloader:
    with autocast():
        loss = model(**batch).loss

    loss.backward()          # 错误在哪里？

    optimizer.step()         # 还有问题吗？
    scaler.update()
    optimizer.zero_grad()
```

修正后，说明每一行代码的作用。

---

### 提高题

**练习 11.5**（提高）

实现一个 `TrainingMonitor` 类，用于在训练过程中监控以下指标：

1. **梯度溢出率**：统计FP16训练中 `GradScaler` 跳过更新的次数占总步数的比例
2. **梯度范数趋势**：记录每步的梯度范数，若连续10步梯度范数 > 10.0，输出警告
3. **学习率曲线**：记录每步的学习率

类接口如下：

```python
class TrainingMonitor:
    def __init__(self):
        pass

    def update(self, step: int, grad_norm: float, scaler: GradScaler, scheduler):
        """每个优化步调用一次"""
        pass

    def report(self):
        """打印训练统计摘要"""
        pass
```

---

## 练习答案

### 答案 11.1

**问题所在**：`torch.optim.Adam` 的 `weight_decay` 参数实现的是L2正则化，而非解耦的权重衰减。在Adam中，L2正则化项会被自适应学习率缩放，导致正则化效果不均匀——梯度历史越大的参数受到的正则化越弱。

**正确实现**：

```python
# 方法一：直接使用AdamW（推荐）
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.01  # AdamW中是解耦的权重衰减
)

# 方法二：区分参数组，避免对偏置/归一化参数衰减
def get_optimizer(model):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or "bias" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    return torch.optim.AdamW([
        {"params": decay,    "weight_decay": 0.01},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=3e-4)
```

---

### 答案 11.2

**计算梯度累积步数**：

$$\text{累积步数} = \frac{\text{目标批量}}{\text{微批量} \times \text{GPU数量}} = \frac{256}{8 \times 4} = 8$$

**正确代码**：

```python
accumulation_steps = 8  # 256 / (8 × 4) = 8

for step, batch in enumerate(dataloader):
    outputs = model(**batch)

    # 必须在backward之前除以accumulation_steps
    # 原因：loss.backward()计算的是 d(loss)/d(param)
    # 除以K后，累积K步的梯度之和 = (1/K)×Σg_k（梯度均值）
    loss = outputs.loss / accumulation_steps
    loss.backward()

    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

### 答案 11.3

```python
def get_param_groups(model, weight_decay=0.1):
    """
    将模型参数分为两组：
    - 需要权重衰减：权重矩阵等二维参数
    - 不需要权重衰减：嵌入层、归一化层、偏置
    """
    no_decay_keywords = {"embed_tokens", "layer_norm", "layernorm", "bias"}

    decay_params = []
    no_decay_params = []

    seen = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in seen:
            continue
        seen.add(id(param))

        # 检查名称是否包含不衰减关键词，或是否为1D参数
        should_decay = True
        if param.ndim <= 1:
            should_decay = False
        else:
            name_lower = name.lower()
            for kw in no_decay_keywords:
                if kw in name_lower:
                    should_decay = False
                    break

        if should_decay:
            decay_params.append((name, param))
        else:
            no_decay_params.append((name, param))

    # 验证：打印各组信息
    print(f"需要衰减的参数组：{len(decay_params)} 个张量，"
          f"{sum(p.numel() for _, p in decay_params):,} 个参数")
    print(f"不衰减的参数组：{len(no_decay_params)} 个张量，"
          f"{sum(p.numel() for _, p in no_decay_params):,} 个参数")

    # 示例：打印不衰减的参数名称
    print("\n不衰减的参数：")
    for name, _ in no_decay_params[:10]:  # 只显示前10个
        print(f"  {name}")

    return [
        {
            "params": [p for _, p in decay_params],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for _, p in no_decay_params],
            "weight_decay": 0.0,
        },
    ]


# 使用示例
# param_groups = get_param_groups(model, weight_decay=0.1)
# optimizer = torch.optim.AdamW(param_groups, lr=3e-4)
```

---

### 答案 11.4

**错误分析**：

```python
scaler = GradScaler()

for batch in dataloader:
    with autocast():
        loss = model(**batch).loss

    # 错误1：应使用 scaler.scale(loss).backward()
    # 不缩放直接backward，FP16梯度可能下溢为0
    loss.backward()

    # 错误2：应使用 scaler.step(optimizer)
    # 且在step之前应调用 scaler.unscale_(optimizer) 以支持梯度裁剪
    # 直接调用optimizer.step()时，梯度仍是缩放后的值，参数更新量级错误
    optimizer.step()

    # 错误3：scaler.update()应在scaler.step()之后立即调用
    scaler.update()
    optimizer.zero_grad()
```

**修正后的完整代码**：

```python
scaler = GradScaler()

for batch in dataloader:
    # 步骤1：混合精度前向传播
    with autocast():
        loss = model(**batch).loss

    # 步骤2：缩放损失，执行反向传播（梯度 × scale_factor，避免下溢）
    scaler.scale(loss).backward()

    # 步骤3（可选）：梯度裁剪需要先unscale
    scaler.unscale_(optimizer)  # 将梯度除以scale_factor，恢复真实量级
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # 步骤4：检查梯度是否有inf/nan；若有则跳过本步更新
    scaler.step(optimizer)  # 内部自动调用optimizer.step()（若梯度正常）

    # 步骤5：动态调整scale_factor（连续正常→翻倍；出现inf→减半）
    scaler.update()

    optimizer.zero_grad()
```

---

### 答案 11.5

```python
from collections import deque
import warnings


class TrainingMonitor:
    """
    训练过程监控器，跟踪以下指标：
    - 梯度溢出率（FP16训练）
    - 梯度范数趋势
    - 学习率曲线
    """

    def __init__(self, overflow_window=100, grad_norm_window=10, grad_norm_threshold=10.0):
        """
        Args:
            overflow_window: 计算溢出率的滑动窗口大小
            grad_norm_window: 梯度范数警告的连续步数阈值
            grad_norm_threshold: 梯度范数警告阈值
        """
        self.overflow_window = overflow_window
        self.grad_norm_window = grad_norm_window
        self.grad_norm_threshold = grad_norm_threshold

        # 溢出检测：使用滑动窗口
        self.overflow_history = deque(maxlen=overflow_window)
        self.total_steps = 0
        self.total_overflows = 0

        # 梯度范数跟踪
        self.grad_norm_history = []
        self.recent_grad_norms = deque(maxlen=grad_norm_window)

        # 学习率记录
        self.lr_history = []

        # 记录上次scaler的scale值，用于检测是否发生了overflow
        self._last_scale = None

    def _check_overflow(self, scaler: "GradScaler") -> bool:
        """
        通过检测scale是否减小来判断是否发生了梯度溢出。
        GradScaler在检测到inf/nan时会减小scale并跳过该步更新。
        """
        current_scale = scaler.get_scale()
        if self._last_scale is not None and current_scale < self._last_scale:
            overflow = True
        else:
            overflow = False
        self._last_scale = current_scale
        return overflow

    def update(self, step: int, grad_norm: float, scaler: "GradScaler", scheduler):
        """
        每个优化步调用一次（在scaler.update()之后）。

        Args:
            step: 当前全局步数
            grad_norm: 梯度裁剪后的梯度范数
            scaler: GradScaler实例
            scheduler: 学习率调度器
        """
        self.total_steps += 1

        # 1. 检测梯度溢出
        overflow = self._check_overflow(scaler)
        self.overflow_history.append(int(overflow))
        if overflow:
            self.total_overflows += 1

        # 2. 跟踪梯度范数
        self.grad_norm_history.append((step, grad_norm))
        self.recent_grad_norms.append(grad_norm)

        # 检查是否连续多步梯度范数过大
        if len(self.recent_grad_norms) == self.grad_norm_window:
            if all(g > self.grad_norm_threshold for g in self.recent_grad_norms):
                warnings.warn(
                    f"[Step {step}] 警告：连续 {self.grad_norm_window} 步梯度范数 > "
                    f"{self.grad_norm_threshold}（最近均值={sum(self.recent_grad_norms)/len(self.recent_grad_norms):.2f}）。"
                    f"建议检查学习率或梯度裁剪设置。",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # 3. 记录学习率
        current_lr = scheduler.get_last_lr()[0]
        self.lr_history.append((step, current_lr))

    def get_overflow_rate(self, window=None) -> float:
        """返回梯度溢出率（近期窗口内）"""
        if window is not None:
            recent = list(self.overflow_history)[-window:]
            if not recent:
                return 0.0
            return sum(recent) / len(recent)
        if self.total_steps == 0:
            return 0.0
        return self.total_overflows / self.total_steps

    def report(self):
        """打印训练统计摘要"""
        print("=" * 50)
        print("训练监控报告")
        print("=" * 50)
        print(f"总优化步数：{self.total_steps}")

        # 溢出统计
        overflow_rate = self.get_overflow_rate()
        recent_overflow_rate = self.get_overflow_rate(window=self.overflow_window)
        print(f"\n梯度溢出统计：")
        print(f"  总溢出次数：{self.total_overflows}")
        print(f"  全局溢出率：{overflow_rate:.2%}")
        print(f"  近期溢出率（最近{self.overflow_window}步）：{recent_overflow_rate:.2%}")

        if recent_overflow_rate > 0.1:
            print(f"  [警告] 近期溢出率较高，考虑降低初始loss scale或使用BF16")

        # 梯度范数统计
        if self.grad_norm_history:
            norms = [g for _, g in self.grad_norm_history]
            print(f"\n梯度范数统计：")
            print(f"  均值：{sum(norms)/len(norms):.4f}")
            print(f"  最大值：{max(norms):.4f}（步 {self.grad_norm_history[norms.index(max(norms))][0]}）")
            print(f"  最小值：{min(norms):.4f}")

        # 学习率统计
        if self.lr_history:
            lrs = [lr for _, lr in self.lr_history]
            print(f"\n学习率：")
            print(f"  初始：{lrs[0]:.2e}")
            print(f"  当前：{lrs[-1]:.2e}")
            print(f"  峰值：{max(lrs):.2e}")

        print("=" * 50)


# 使用示例
# monitor = TrainingMonitor(grad_norm_threshold=10.0)
#
# for step, batch in enumerate(dataloader):
#     with autocast():
#         loss = model(**batch).loss
#     scaler.scale(loss).backward()
#     scaler.unscale_(optimizer)
#     grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#     scaler.step(optimizer)
#     scaler.update()
#     scheduler.step()
#     optimizer.zero_grad()
#
#     monitor.update(step, grad_norm.item(), scaler, scheduler)
#
# monitor.report()
```

---

> **下一章预告**：第12章将介绍学习率调度策略，包括Warmup、余弦退火、以及针对大模型训练的WSD（Warmup-Stable-Decay）调度方案，并探讨如何根据任务特点选择最优的训练超参数。
