# 第九章：噪声预测网络与训练目标

> **本章导读**：前两章建立了DDPM的数学框架——正向加噪和逆向去噪。但"魔法"发生在噪声预测网络 $\epsilon_\theta(x_t, t)$ 内部。本章深入探讨三个关键问题：（1）如何设计一个好的噪声预测网络？时间步嵌入如何实现？（2）$\epsilon$-预测、$x_0$-预测和 $v$-预测三种训练目标有什么异同？（3）如何通过损失加权、EMA、混合精度等技巧提升训练效率和生成质量？最后，我们讨论如何评估扩散模型的生成质量。

**前置知识**: DDPM训练与采样（第八章），深度学习基础（卷积网络、注意力机制），PyTorch训练技巧

**预计学习时间**: 3-4小时

## 学习目标

1. 理解噪声预测网络的设计需求，掌握时间步嵌入的原理和注入方式
2. 推导并比较三种训练目标（$\epsilon$-预测、$x_0$-预测、$v$-预测）的等价转换
3. 理解Min-SNR损失加权策略及其理论基础
4. 掌握EMA、混合精度、梯度裁剪等关键训练技巧
5. 理解FID、IS等生成模型评估指标的原理和局限性

---

## 9.1 噪声预测网络的设计需求

### 9.1.1 输入与输出

噪声预测网络的核心任务：

$$\epsilon_\theta: (x_t, t) \mapsto \hat{\epsilon} \in \mathbb{R}^d$$

给定加噪样本 $x_t$ 和时间步 $t$，预测生成 $x_t$ 时使用的噪声 $\epsilon$。

关键设计需求：

| 需求 | 原因 |
|:---|:---|
| **时间条件化** | 不同时间步的"去噪难度"和"去噪方式"不同 |
| **同维度输入输出** | $\hat{\epsilon}$ 必须与 $x_t$ 形状相同 |
| **多尺度特征** | 去噪需要同时利用局部（像素级）和全局（语义级）信息 |
| **计算效率** | 采样时需要调用 $T$ 次网络（$T = 1000$） |
| **平移等变性** | 对于图像数据，卷积结构提供自然的归纳偏置 |

### 9.1.2 为什么U-Net成为标准架构

DDPM（Ho et al., 2020）采用U-Net作为骨干网络，这并非偶然：

- **编码器-解码器结构**：自然支持多尺度特征提取
- **跳跃连接**：保留高分辨率细节（去噪必需）
- **输入输出同尺寸**：编码器下采样，解码器上采样，恢复原始分辨率
- **灵活性**：容易嵌入时间步条件和注意力机制

U-Net的详细架构将在后续章节展开，本章重点关注两个与架构无关的核心组件：**时间步嵌入**和**训练目标**。

### 9.1.3 关键挑战

**挑战1：时间条件化的表达力**

网络需要对不同的 $t$ 表现出截然不同的行为：
- $t$ 较大时（高噪声）：需要"猜测"全局结构，类似于图像生成
- $t$ 较小时（低噪声）：需要精细化细节，类似于图像去噪

单一网络如何在这些截然不同的"任务"之间切换？答案在于时间步嵌入的设计。

**挑战2：多尺度特征的融合**

噪声 $\epsilon$ 的预测需要综合考虑：
- **像素级信息**：局部噪声模式
- **区域级信息**：物体边缘、纹理的连贯性
- **全局信息**：场景的整体一致性

**挑战3：训练与采样的效率**

训练时每个样本只需一次前向传播（随机 $t$），但采样时需要 $T$ 次前向传播。采样效率直接决定了模型的实用性。

---

## 9.2 时间步嵌入

### 9.2.1 正弦位置编码

DDPM借鉴了Transformer（Vaswani et al., 2017）的正弦位置编码来表示时间步 $t$。

对于标量时间步 $t$，生成 $d_{model}$ 维的嵌入向量：

$$\text{PE}(t, 2i) = \sin\left(\frac{t}{10000^{2i/d_{model}}}\right)$$

$$\text{PE}(t, 2i+1) = \cos\left(\frac{t}{10000^{2i/d_{model}}}\right)$$

其中 $i = 0, 1, \ldots, d_{model}/2 - 1$。

**为什么正弦编码有效？**

1. **连续性**：相邻时间步的嵌入向量相似，提供平滑的条件信号
2. **唯一性**：不同时间步的嵌入向量不同，网络能区分
3. **周期性分解**：不同频率的正弦/余弦波捕捉不同尺度的时间信息
4. **相对位置感知**：$\text{PE}(t+\Delta)$ 可以表示为 $\text{PE}(t)$ 的线性变换

频率从 $\frac{1}{10000^0} = 1$（高频）到 $\frac{1}{10000^1} = 10^{-4}$（低频）：
- 高频分量对微小的 $t$ 变化敏感，捕捉精细的时间差异
- 低频分量变化缓慢，捕捉宏观的时间阶段

### 9.2.2 可学习嵌入 vs 固定编码

**固定正弦编码**（DDPM默认）：
- 优点：不引入额外参数，天然支持未见过的时间步
- 缺点：可能不是最优的时间表示

**可学习嵌入**：
- 使用 $\text{nn.Embedding}(T, d_{model})$ 查找表
- 优点：完全数据驱动，可学到最优的时间表示
- 缺点：不能泛化到训练时未见的时间步，参数量增加

**混合方案**（实践中最常用）：
- 先用正弦编码得到固定基础嵌入
- 再通过两层MLP投影到网络的隐藏维度
- 这结合了两者的优势：正弦编码提供良好的初始化，MLP提供自适应能力

$$\text{time\_emb}(t) = \text{MLP}(\text{SinusoidalPE}(t)) = W_2 \cdot \text{SiLU}(W_1 \cdot \text{PE}(t) + b_1) + b_2$$

### 9.2.3 时间嵌入在网络中的注入方式

获得时间嵌入后，如何将其注入到网络的各层？有三种主要方式：

**方式1：加法注入（Additive）**

$$h' = h + \text{Linear}(\text{time\_emb})$$

将时间嵌入线性投影到与特征图相同的通道数，然后逐元素相加。简单高效，DDPM原始论文使用此方式。

**方式2：缩放注入（Scale-Shift / FiLM）**

$$h' = \gamma(t) \odot h + \beta(t)$$

其中 $\gamma(t), \beta(t)$ 是从时间嵌入计算的缩放和偏移参数。比纯加法更灵活，因为乘法可以改变特征的相对大小。

**方式3：自适应层归一化（AdaLN / AdaGN）**

$$h' = \gamma(t) \odot \text{LayerNorm}(h) + \beta(t)$$

在归一化之后应用缩放和偏移。DiT（Peebles & Xie, 2023）等现代架构大量使用AdaLN，效果优于简单的加法注入。

| 注入方式 | 表达力 | 计算开销 | 代表模型 |
|:---|:---|:---|:---|
| 加法 | 中 | 低 | DDPM |
| 缩放-偏移 | 高 | 中 | Improved DDPM |
| AdaLN | 最高 | 中 | DiT, SD3 |

---

## 9.3 训练目标的变种

### 9.3.1 $\epsilon$-预测（DDPM默认）

$$L_\epsilon = \mathbb{E}_{t,x_0,\epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

其中 $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$。

**优点**：
- 预测目标 $\epsilon \sim \mathcal{N}(0, I)$ 是有界的，数值稳定
- 训练信号在各维度上近似均匀（因为 $\epsilon$ 的各分量独立同分布）

**缺点**：
- 在 $t$ 接近 $0$ 时，$x_t \approx x_0$，但网络仍需要预测一个非常小的噪声残差，信噪比很高，容易过拟合

### 9.3.2 $x_0$-预测

$$L_{x_0} = \mathbb{E}_{t,x_0,\epsilon}\left[\|x_0 - x_\theta(x_t, t)\|^2\right]$$

其中网络直接预测原始数据 $x_0$。

**与 $\epsilon$-预测的关系**：

由 $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$，有：

$$x_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon}{\sqrt{\bar{\alpha}_t}}$$

如果 $\epsilon_\theta$ 是 $\epsilon$-预测网络，则等价的 $x_0$-预测为：

$$\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t,t)}{\sqrt{\bar{\alpha}_t}}$$

两种目标的损失之间有精确关系（参见第八章练习 8.4）：

$$L_{x_0} = \frac{1-\bar{\alpha}_t}{\bar{\alpha}_t} \cdot L_\epsilon = \frac{1}{\text{SNR}(t)} \cdot L_\epsilon$$

**优点**：
- 预测目标有明确的物理含义（"猜测原始数据是什么"）
- 在 $t$ 较大时预测目标的方差更小（因为 $x_0$ 是固定的，而 $\epsilon$ 只是一个实现）

**缺点**：
- 在 $t$ 较大时（高噪声），$x_0$ 的预测误差被 $\frac{1}{\text{SNR}(t)}$ 放大，导致数值不稳定
- 需要 clipping 或特殊处理来防止预测值超出合理范围

### 9.3.3 $v$-预测（Salimans & Ho, 2022）

定义"速度"（velocity）为：

$$v_t = \sqrt{\bar{\alpha}_t}\,\epsilon - \sqrt{1-\bar{\alpha}_t}\, x_0$$

训练目标为：

$$L_v = \mathbb{E}_{t,x_0,\epsilon}\left[\|v_t - v_\theta(x_t, t)\|^2\right]$$

**$v$ 的直觉**：$v_t$ 是"从信号到噪声"方向上的速度。在 $t = 0$ 时 $v_0 = -x_0$（纯信号的反方向），在 $t = T$ 时 $v_T \approx \epsilon$（纯噪声方向）。

**与其他预测的转换**：

给定 $v_\theta(x_t, t)$，可以恢复：

$$\hat{\epsilon} = \sqrt{\bar{\alpha}_t}\, v_\theta + \sqrt{1-\bar{\alpha}_t}\, x_t$$

$$\hat{x}_0 = \sqrt{\bar{\alpha}_t}\, x_t - \sqrt{1-\bar{\alpha}_t}\, v_\theta$$

推导：由 $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$ 和 $v = \sqrt{\bar{\alpha}_t}\epsilon - \sqrt{1-\bar{\alpha}_t}x_0$，这是一个二元线性系统。解出：

$$\begin{pmatrix} \epsilon \\ x_0 \end{pmatrix} = \begin{pmatrix} \sqrt{\bar{\alpha}_t} & \sqrt{1-\bar{\alpha}_t} \\ \sqrt{\bar{\alpha}_t} & -\sqrt{1-\bar{\alpha}_t} \end{pmatrix}^{-1} \begin{pmatrix} x_t \\ -v \end{pmatrix}$$

由旋转矩阵的性质（注意 $\bar{\alpha}_t + (1-\bar{\alpha}_t) = 1$，所以系数矩阵正交），逆就是转置：

$$\epsilon = \sqrt{\bar{\alpha}_t}\, x_t + \sqrt{1-\bar{\alpha}_t}\,(-v) \cdot (-1)$$

整理可得上述公式。

**$v$-预测的优势**：

1. **数值稳定性**：$v_t$ 的方差在所有时间步都约为 $1$（因为它是两个单位方差量的线性组合且系数平方和为 $1$）
2. **自然加权**：$v$-预测损失自动平衡了 $\epsilon$-预测和 $x_0$-预测的权重
3. **在端点处表现更好**：$t \approx 0$ 时退化为 $x_0$-预测，$t \approx T$ 时退化为 $\epsilon$-预测

### 9.3.4 三种目标的等价转换公式总结

| 已知 | $\to \hat{\epsilon}$ | $\to \hat{x}_0$ | $\to \hat{v}$ |
|:---|:---|:---|:---|
| $\epsilon_\theta$ | $\epsilon_\theta$ | $\frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta}{\sqrt{\bar{\alpha}_t}}$ | $\sqrt{\bar{\alpha}_t}\epsilon_\theta - \sqrt{1-\bar{\alpha}_t}\cdot\frac{x_t-\sqrt{1-\bar{\alpha}_t}\epsilon_\theta}{\sqrt{\bar{\alpha}_t}}$ |
| $x_\theta$ | $\frac{x_t - \sqrt{\bar{\alpha}_t}x_\theta}{\sqrt{1-\bar{\alpha}_t}}$ | $x_\theta$ | $\sqrt{\bar{\alpha}_t}\cdot\frac{x_t-\sqrt{\bar{\alpha}_t}x_\theta}{\sqrt{1-\bar{\alpha}_t}} - \sqrt{1-\bar{\alpha}_t}x_\theta$ |
| $v_\theta$ | $\sqrt{\bar{\alpha}_t}v_\theta + \sqrt{1-\bar{\alpha}_t}x_t$ | $\sqrt{\bar{\alpha}_t}x_t - \sqrt{1-\bar{\alpha}_t}v_\theta$ | $v_\theta$ |

在采样时，无论训练使用哪种目标，都需要转换为 $\hat{\epsilon}$ 或 $\hat{x}_0$ 来计算后验均值。

---

## 9.4 损失加权策略

### 9.4.1 问题：不同时间步的损失贡献不均衡

使用均匀采样 $t \sim \text{Uniform}(1, T)$ 配合 $L_{simple}$，不同时间步对总损失的贡献差异巨大：

- **小 $t$**（低噪声）：$x_t \approx x_0$，噪声容易预测，损失很小
- **大 $t$**（高噪声）：$x_t \approx \epsilon$，噪声难以预测，损失很大
- **中间 $t$**：损失适中，但这正是生成质量最关键的区域

这种不均衡意味着网络可能过度关注高噪声步（大损失）而忽视中等噪声步。

### 9.4.2 重要性采样

一种直接的改进是用非均匀分布采样 $t$：

$$t \sim p(t), \quad L = \mathbb{E}_{t \sim p(t)}\left[\frac{1}{p(t)} \|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

其中 $\frac{1}{p(t)}$ 是重要性权重，保证期望不变。

Nichol & Dhariwal (2021) 提出基于损失大小的重要性采样：

$$p(t) \propto \sqrt{L_t}$$

其中 $L_t$ 是第 $t$ 步的历史平均损失。这让高损失的时间步被更频繁地采样。

### 9.4.3 Min-SNR 加权（Hang et al., 2023）

Min-SNR-$\gamma$ 策略提供了一种理论上更优雅的方案。核心思想：对 $L_{simple}$ 施加一个依赖于SNR的权重：

$$L_{\text{Min-SNR}} = \mathbb{E}_{t,x_0,\epsilon}\left[\min\left(\text{SNR}(t),\, \gamma\right) \cdot \|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

其中 $\gamma > 0$ 是超参数（通常取 $\gamma = 5$），$\text{SNR}(t) = \frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t}$。

**直觉解释**：

- 高SNR（小 $t$）：权重为 $\min(\text{SNR}, \gamma) = \gamma$，被截断，防止过度关注
- 低SNR（大 $t$）：权重为 $\min(\text{SNR}, \gamma) = \text{SNR}$，随噪声增加而减小，降低高噪声步的影响
- 中等SNR：权重适中，正是生成质量最重要的区域

**理论基础**：

Min-SNR加权可以从帕累托最优的角度理解。训练扩散模型是一个多任务学习问题（$T$ 个去噪任务），Min-SNR提供了一种在各任务间取得帕累托最优平衡的方式。

### 9.4.4 不同时间步的损失贡献分析

实践中，可以通过以下方法分析各时间步的损失：

1. 在验证集上计算每个 $t$ 的平均损失 $L_t$
2. 绘制 $L_t$ vs $t$ 的曲线
3. 高损失区域表明模型学习困难，可能需要更多关注

典型观察：
- $L_t$ 通常在 $t \approx T/3$ 到 $t \approx 2T/3$ 区间最大
- 极小 $t$（几乎无噪声）和极大 $t$（几乎纯噪声）的损失都较小
- 余弦调度比线性调度的损失分布更均匀

---

## 9.5 训练技巧与稳定性

### 9.5.1 EMA（指数移动平均）

EMA是扩散模型训练中最重要的技巧之一。维护一组"影子参数"：

$$\theta_{\text{EMA}}^{(k)} = m \cdot \theta_{\text{EMA}}^{(k-1)} + (1 - m) \cdot \theta^{(k)}$$

其中 $m$ 是动量系数（通常 $m = 0.9999$ 或 $m = 0.999$），$\theta^{(k)}$ 是第 $k$ 步的训练参数。

**为什么EMA有效？**

1. **平滑效果**：EMA参数是训练轨迹上参数的加权平均，降低了参数波动
2. **隐式集成**：类似于对训练过程中多个checkpoint的模型集成
3. **避免过拟合**：EMA参数对最近的更新不那么敏感，更加鲁棒

**关键实现细节**：

- 采样/评估时使用 $\theta_{\text{EMA}}$，训练时使用 $\theta$（梯度只更新 $\theta$）
- $m$ 通常随训练进度调整：早期用较小的 $m$（如0.99）加速追踪，后期用较大的 $m$（如0.9999）提高平滑度
- 训练初期可以跳过若干步再开始EMA，避免初始随机参数的影响

### 9.5.2 混合精度训练（FP16/BF16）

扩散模型训练计算量大，混合精度是必要的加速手段。

**FP16**（半精度浮点数）：
- 范围：$\pm 6.5 \times 10^4$，精度：$\sim 3.3$ 位有效数字
- 优点：内存减半，速度提升（特别是在Tensor Core上）
- 风险：梯度下溢（gradient underflow），需要损失缩放（loss scaling）

**BF16**（Brain Float 16）：
- 范围：与FP32相同（$\pm 3.4 \times 10^{38}$），精度：$\sim 2.4$ 位有效数字
- 优点：不需要损失缩放，更稳定
- 缺点：精度略低于FP16，只有较新的GPU支持

**实践建议**：
- 优先使用BF16（如果GPU支持，如A100、H100）
- 使用PyTorch的 `torch.cuda.amp`（自动混合精度）
- 模型参数和优化器状态保持FP32，只在前向和反向传播中使用低精度

### 9.5.3 梯度裁剪

扩散模型的梯度可能偶尔出现异常大的值（特别是在某些时间步），梯度裁剪提供安全保障：

$$g \leftarrow g \cdot \frac{\text{max\_norm}}{\max(\|g\|, \text{max\_norm})}$$

典型设置：`max_norm = 1.0`。这保证梯度范数不超过1，防止单次异常更新破坏训练进度。

### 9.5.4 学习率调度

典型的学习率策略：

**预热阶段**（warmup）：前 $N_{\text{warmup}}$ 步线性增长

$$\text{lr}(k) = \text{lr}_{\max} \cdot \min\left(1,\, \frac{k}{N_{\text{warmup}}}\right)$$

**余弦衰减**（cosine decay）：预热后余弦下降

$$\text{lr}(k) = \text{lr}_{\min} + \frac{1}{2}(\text{lr}_{\max} - \text{lr}_{\min})\left(1 + \cos\left(\frac{k - N_{\text{warmup}}}{K - N_{\text{warmup}}}\pi\right)\right)$$

常见配置：$\text{lr}_{\max} = 10^{-4}$，$N_{\text{warmup}} = 5000$，使用AdamW优化器（$\beta_1 = 0.9, \beta_2 = 0.999$）。

### 9.5.5 批量大小的影响

扩散模型对批量大小比较敏感：

- **小批量**（$< 64$）：梯度估计噪声大，训练不稳定
- **中批量**（64-512）：常用范围，平衡效率和稳定性
- **大批量**（$> 1024$）：更稳定，但需要调整学习率（线性缩放规则）

实践建议：在显存允许的范围内尽量使用较大的批量。如果显存不足，使用梯度累积（gradient accumulation）来等效增大批量。

---

## 9.6 评估指标

### 9.6.1 FID（Frechet Inception Distance）

FID是当前最广泛使用的生成图像质量指标。

**核心思想**：在Inception-v3网络的倒数第二层提取特征，比较真实图像和生成图像的特征分布。

**数学定义**：假设真实图像特征分布为 $\mathcal{N}(\mu_r, \Sigma_r)$，生成图像特征为 $\mathcal{N}(\mu_g, \Sigma_g)$，则：

$$\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)$$

这是两个多元高斯分布之间的 Frechet 距离（也称Wasserstein-2距离）的平方。

**解读**：
- $\text{FID} = 0$ 表示两个分布完全相同
- FID 越低越好
- 典型值：CIFAR-10上SOTA约1-3，ImageNet 256x256上约2-5

**注意事项**：
- 需要足够多的样本（推荐 $\geq 50000$）才能稳定
- 对Inception网络的偏差敏感
- 不能完全捕捉人类感知的质量差异

### 9.6.2 IS（Inception Score）

**定义**：

$$\text{IS} = \exp\left(\mathbb{E}_{x \sim p_g}\left[D_{KL}(p(y|x) \| p(y))\right]\right)$$

其中 $p(y|x)$ 是Inception网络对生成图像 $x$ 的分类概率，$p(y) = \mathbb{E}_x[p(y|x)]$ 是边际分类概率。

**直觉**：
- 每张图像的分类应该"自信"（$p(y|x)$ 集中） $\to$ 图像清晰
- 整体分类应该"多样"（$p(y)$ 均匀） $\to$ 多样性好
- IS越高越好

**局限**：
- 只考虑Inception网络认识的1000个类别
- 不考虑生成图像与真实图像的"距离"
- 对于非ImageNet分布的数据不太适用

### 9.6.3 CLIP Score（文生图质量）

对于文本条件生成模型（如Stable Diffusion），CLIP Score衡量图像与文本的一致性：

$$\text{CLIP Score} = \mathbb{E}\left[\cos(\text{CLIP}_\text{image}(x), \text{CLIP}_\text{text}(c))\right]$$

其中 $\text{CLIP}_\text{image}$ 和 $\text{CLIP}_\text{text}$ 分别是CLIP模型的图像和文本编码器，$c$ 是输入提示词。

- CLIP Score越高表示图文对齐越好
- 与FID互补：FID衡量图像质量和多样性，CLIP Score衡量条件一致性

### 9.6.4 NLL（负对数似然）

扩散模型可以通过概率流ODE（Song et al., 2020）精确计算负对数似然：

$$-\log p_\theta(x_0) = -\log p(x_T) + \int_0^T \text{Tr}\left(\frac{\partial f_\theta}{\partial x_t}\right) dt$$

其中 $f_\theta$ 是概率流ODE的漂移项。

NLL的优势在于它是信息论上有严格意义的指标（每个比特的编码效率），但计算开销大（需要求解ODE和计算Jacobian迹）。

### 9.6.5 人类评估的局限性

自动指标不能完全替代人类评估。常见的人类评估方法：

- **两择偏好（Two-Alternative Forced Choice）**：展示两张图像，让人选择质量更好的一张
- **评分（Rating）**：对单张图像的质量、多样性、文本相关性打分

但人类评估代价高昂，结果不可完全复现，且可能受到展示顺序、评估者背景等因素的影响。因此实践中通常以FID为主要自动指标，辅以定性的视觉检查。

---

## 代码实战

```python
"""
第九章代码实战：训练目标变种、损失加权、EMA与评估
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy
from typing import Optional, Literal
from dataclasses import dataclass


# ============================================================
# 1. 时间步嵌入模块
# ============================================================

class SinusoidalTimeEmbedding(nn.Module):
    """正弦时间步嵌入
    
    将标量时间步 t 编码为 dim 维向量。
    
    参考: Vaswani et al. "Attention is All You Need" (2017)
          Ho et al. "Denoising Diffusion Probabilistic Models" (2020)
    """
    
    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) 时间步索引（整数或浮点数）
        Returns:
            emb: (B, dim) 正弦位置编码
        """
        half = self.dim // 2
        # 频率：从 1 到 1/max_period
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )  # (half,)
        
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, dim)
        
        # 如果 dim 是奇数，补一个零
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        
        return emb


class TimeMLPEmbedding(nn.Module):
    """正弦编码 + MLP 投影（实践中最常用的时间嵌入）"""
    
    def __init__(self, time_emb_dim: int, hidden_dim: int):
        super().__init__()
        self.sinusoidal = SinusoidalTimeEmbedding(time_emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) 时间步
        Returns:
            emb: (B, hidden_dim)
        """
        return self.mlp(self.sinusoidal(t))  # (B, hidden_dim)


# ============================================================
# 2. 三种训练目标的实现和转换
# ============================================================

class PredictionConverter:
    """在 epsilon、x0、v 三种预测之间互相转换
    
    核心关系：
        x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * epsilon
        v   = sqrt(alpha_bar) * epsilon - sqrt(1 - alpha_bar) * x_0
    """
    
    @staticmethod
    def eps_to_x0(
        eps: torch.Tensor,            # (B, ...) 预测的噪声
        x_t: torch.Tensor,            # (B, ...) 加噪样本
        sqrt_alpha_bar: torch.Tensor,  # (B, 1, ...) 或标量
        sqrt_one_minus_alpha_bar: torch.Tensor,
    ) -> torch.Tensor:
        """从 epsilon 预测转换为 x0 预测"""
        return (x_t - sqrt_one_minus_alpha_bar * eps) / sqrt_alpha_bar
    
    @staticmethod
    def x0_to_eps(
        x0: torch.Tensor,
        x_t: torch.Tensor,
        sqrt_alpha_bar: torch.Tensor,
        sqrt_one_minus_alpha_bar: torch.Tensor,
    ) -> torch.Tensor:
        """从 x0 预测转换为 epsilon 预测"""
        return (x_t - sqrt_alpha_bar * x0) / sqrt_one_minus_alpha_bar
    
    @staticmethod
    def v_to_eps(
        v: torch.Tensor,
        x_t: torch.Tensor,
        sqrt_alpha_bar: torch.Tensor,
        sqrt_one_minus_alpha_bar: torch.Tensor,
    ) -> torch.Tensor:
        """从 v 预测转换为 epsilon 预测"""
        return sqrt_alpha_bar * v + sqrt_one_minus_alpha_bar * x_t
    
    @staticmethod
    def v_to_x0(
        v: torch.Tensor,
        x_t: torch.Tensor,
        sqrt_alpha_bar: torch.Tensor,
        sqrt_one_minus_alpha_bar: torch.Tensor,
    ) -> torch.Tensor:
        """从 v 预测转换为 x0 预测"""
        return sqrt_alpha_bar * x_t - sqrt_one_minus_alpha_bar * v
    
    @staticmethod
    def compute_v_target(
        eps: torch.Tensor,
        x_0: torch.Tensor,
        sqrt_alpha_bar: torch.Tensor,
        sqrt_one_minus_alpha_bar: torch.Tensor,
    ) -> torch.Tensor:
        """计算 v-预测的目标值"""
        return sqrt_alpha_bar * eps - sqrt_one_minus_alpha_bar * x_0


class DiffusionLoss(nn.Module):
    """支持三种预测目标的扩散模型损失函数"""
    
    def __init__(
        self,
        prediction_type: Literal["epsilon", "x0", "v"] = "epsilon",
    ):
        super().__init__()
        self.prediction_type = prediction_type
        self.converter = PredictionConverter()
    
    def forward(
        self,
        model_output: torch.Tensor,   # (B, ...) 网络输出
        noise: torch.Tensor,           # (B, ...) 真实噪声
        x_0: torch.Tensor,             # (B, ...) 原始数据
        sqrt_alpha_bar: torch.Tensor,  # (B, 1, ...)
        sqrt_one_minus_alpha_bar: torch.Tensor,
    ) -> torch.Tensor:
        """计算损失
        
        根据 prediction_type 选择对应的目标。
        """
        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "x0":
            target = x_0
        elif self.prediction_type == "v":
            target = self.converter.compute_v_target(
                noise, x_0, sqrt_alpha_bar, sqrt_one_minus_alpha_bar
            )
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
        
        # 均方误差
        loss = F.mse_loss(model_output, target, reduction="none")  # (B, ...)
        loss = loss.mean(dim=list(range(1, loss.dim())))  # (B,) 对每个样本求均值
        
        return loss  # (B,) 返回每个样本的损失，便于加权


# ============================================================
# 3. Min-SNR 损失加权
# ============================================================

class MinSNRWeighting:
    """Min-SNR-gamma 损失加权策略
    
    weight(t) = min(SNR(t), gamma) / SNR(t)
    
    当 prediction_type 不同时，加权公式需要调整：
    - epsilon: weight = min(SNR, gamma)
    - x0:      weight = min(SNR, gamma) / SNR^2  (等效)
    - v:       weight = min(SNR, gamma) / (SNR + 1)
    
    参考: Hang et al. "Efficient Diffusion Training via Min-SNR Weighting Strategy" (2023)
    """
    
    def __init__(
        self,
        gamma: float = 5.0,
        prediction_type: Literal["epsilon", "x0", "v"] = "epsilon",
    ):
        self.gamma = gamma
        self.prediction_type = prediction_type
    
    def __call__(
        self,
        snr: torch.Tensor,  # (B,) 每个样本对应时间步的 SNR
    ) -> torch.Tensor:      # (B,) 权重
        """计算 Min-SNR 权重"""
        clamped_snr = torch.clamp(snr, max=self.gamma)  # min(SNR, gamma)
        
        if self.prediction_type == "epsilon":
            # L_epsilon 的权重
            weight = clamped_snr / snr
        elif self.prediction_type == "x0":
            # L_x0 = L_epsilon / SNR，所以需要额外除以 SNR
            weight = clamped_snr / (snr ** 2)
            # 但通常直接用 min(SNR, gamma) 作为 epsilon 损失的权重
            weight = clamped_snr / snr
        elif self.prediction_type == "v":
            # L_v = L_epsilon * (1 + 1/SNR)，归一化后
            weight = clamped_snr / (snr + 1.0)
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
        
        return weight


def compute_snr(alpha_bars: torch.Tensor) -> torch.Tensor:
    """计算信噪比 SNR = alpha_bar / (1 - alpha_bar)"""
    return alpha_bars / (1.0 - alpha_bars)


# ============================================================
# 4. EMA（指数移动平均）权重跟踪
# ============================================================

class EMAModel:
    """指数移动平均模型
    
    维护一组影子参数，在采样和评估时使用。
    
    用法:
        ema = EMAModel(model, decay=0.9999)
        
        # 训练循环中
        for batch in dataloader:
            loss = compute_loss(model, batch)
            loss.backward()
            optimizer.step()
            ema.update()  # 更新EMA参数
        
        # 评估时
        with ema.apply_shadow():
            evaluate(model)  # 此时 model 使用 EMA 参数
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        warmup_steps: int = 0,
        update_after_step: int = 0,
    ):
        self.model = model
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.update_after_step = update_after_step
        self.step = 0
        
        # 深拷贝所有参数作为影子参数
        self.shadow_params = [p.clone().detach() for p in model.parameters()]
        # 备份参数（用于 apply_shadow 上下文管理器）
        self.backup_params: list[torch.Tensor] = []
    
    def get_decay(self) -> float:
        """获取当前步的衰减系数（支持预热）"""
        step = max(0, self.step - self.update_after_step)
        if step < self.warmup_steps:
            # 预热阶段，衰减从 0 线性增长到 self.decay
            return min(self.decay, (1 + step) / (10 + step))
        return self.decay
    
    def update(self) -> None:
        """更新 EMA 参数"""
        self.step += 1
        
        if self.step <= self.update_after_step:
            # 还没开始更新EMA
            for shadow, param in zip(self.shadow_params, self.model.parameters()):
                shadow.data.copy_(param.data)
            return
        
        decay = self.get_decay()
        
        with torch.no_grad():
            for shadow, param in zip(self.shadow_params, self.model.parameters()):
                # shadow = decay * shadow + (1 - decay) * param
                shadow.data.mul_(decay).add_(param.data, alpha=1.0 - decay)
    
    class _ApplyShadowContext:
        """上下文管理器：临时用EMA参数替换模型参数"""
        def __init__(self, ema: "EMAModel"):
            self.ema = ema
        
        def __enter__(self):
            self.ema.backup_params = [
                p.clone() for p in self.ema.model.parameters()
            ]
            for shadow, param in zip(
                self.ema.shadow_params, self.ema.model.parameters()
            ):
                param.data.copy_(shadow.data)
            return self.ema.model
        
        def __exit__(self, *args):
            for backup, param in zip(
                self.ema.backup_params, self.ema.model.parameters()
            ):
                param.data.copy_(backup.data)
            self.ema.backup_params = []
    
    def apply_shadow(self) -> _ApplyShadowContext:
        """返回上下文管理器，临时使用EMA参数"""
        return self._ApplyShadowContext(self)
    
    def state_dict(self) -> dict:
        """保存状态"""
        return {
            "step": self.step,
            "decay": self.decay,
            "shadow_params": self.shadow_params,
        }
    
    def load_state_dict(self, state_dict: dict) -> None:
        """加载状态"""
        self.step = state_dict["step"]
        self.decay = state_dict["decay"]
        self.shadow_params = state_dict["shadow_params"]


# ============================================================
# 5. 完整训练循环
# ============================================================

@dataclass
class TrainingConfig:
    """训练配置"""
    num_epochs: int = 300
    batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    grad_clip_norm: float = 1.0
    ema_decay: float = 0.9999
    ema_warmup_steps: int = 100
    prediction_type: str = "epsilon"  # "epsilon", "x0", "v"
    use_min_snr: bool = True
    min_snr_gamma: float = 5.0
    use_amp: bool = False  # 自动混合精度
    log_every: int = 20
    eval_every: int = 50
    checkpoint_dir: str = "./checkpoints"


class DiffusionTrainer:
    """扩散模型训练器
    
    封装完整的训练逻辑，包括损失计算、EMA、梯度裁剪、学习率调度等。
    """
    
    def __init__(
        self,
        model: nn.Module,
        noise_scheduler,  # 来自第七/八章的 NoiseScheduler
        config: TrainingConfig,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.scheduler = noise_scheduler
        self.config = config
        self.device = device
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
        )
        
        # EMA
        self.ema = EMAModel(
            self.model,
            decay=config.ema_decay,
            warmup_steps=config.ema_warmup_steps,
        )
        
        # 损失函数
        self.loss_fn = DiffusionLoss(prediction_type=config.prediction_type)
        
        # Min-SNR 加权
        self.min_snr = None
        if config.use_min_snr:
            self.min_snr = MinSNRWeighting(
                gamma=config.min_snr_gamma,
                prediction_type=config.prediction_type,
            )
        
        # 混合精度
        self.scaler = torch.amp.GradScaler("cuda") if config.use_amp else None
        
        # 训练状态
        self.global_step = 0
        self.train_losses: list[float] = []
    
    def _get_lr(self) -> float:
        """计算当前学习率（带预热和余弦衰减）"""
        if self.global_step < self.config.warmup_steps:
            return self.config.learning_rate * self.global_step / self.config.warmup_steps
        return self.config.learning_rate
    
    def _update_lr(self) -> None:
        """更新学习率"""
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
    
    def train_step(self, x_0: torch.Tensor) -> float:
        """执行单个训练步
        
        Args:
            x_0: (B, ...) 真实数据批次
            
        Returns:
            损失值（标量）
        """
        self.model.train()
        batch_size = x_0.shape[0]
        
        # 1. 采样时间步
        t = torch.randint(
            0, self.scheduler.num_timesteps, (batch_size,), device=self.device
        )  # (B,)
        
        # 2. 采样噪声
        noise = torch.randn_like(x_0)  # (B, ...)
        
        # 3. 计算 x_t
        sqrt_alpha_bar = self.scheduler.sqrt_alpha_bars[t]  # (B,)
        sqrt_one_minus_alpha_bar = self.scheduler.sqrt_one_minus_alpha_bars[t]  # (B,)
        
        # 扩展维度
        while sqrt_alpha_bar.dim() < x_0.dim():
            sqrt_alpha_bar = sqrt_alpha_bar.unsqueeze(-1)
            sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.unsqueeze(-1)
        
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise  # (B, ...)
        
        # 4. 前向传播（可选混合精度）
        autocast_ctx = (
            torch.amp.autocast("cuda")
            if self.config.use_amp
            else torch.amp.autocast("cuda", enabled=False)
        )
        
        with autocast_ctx:
            model_output = self.model(x_t, t)  # (B, ...)
            per_sample_loss = self.loss_fn(
                model_output, noise, x_0,
                sqrt_alpha_bar, sqrt_one_minus_alpha_bar,
            )  # (B,)
        
        # 5. 应用 Min-SNR 加权
        if self.min_snr is not None:
            snr = compute_snr(self.scheduler.alpha_bars[t])  # (B,)
            weights = self.min_snr(snr)  # (B,)
            loss = (per_sample_loss * weights).mean()
        else:
            loss = per_sample_loss.mean()
        
        # 6. 反向传播和优化
        self.optimizer.zero_grad()
        
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip_norm
            )
            self.optimizer.step()
        
        # 7. 更新EMA和学习率
        self.ema.update()
        self.global_step += 1
        self._update_lr()
        
        return loss.item()
    
    def train(
        self,
        dataset: torch.Tensor,  # (N, ...) 完整数据集
    ) -> list[float]:
        """完整训练循环
        
        Returns:
            每个epoch的平均损失
        """
        dataset = dataset.to(self.device)
        n_samples = dataset.shape[0]
        epoch_losses = []
        
        for epoch in range(self.config.num_epochs):
            perm = torch.randperm(n_samples, device=self.device)
            epoch_loss = 0.0
            num_batches = 0
            
            for i in range(0, n_samples, self.config.batch_size):
                batch = dataset[perm[i:i + self.config.batch_size]]
                loss = self.train_step(batch)
                epoch_loss += loss
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            epoch_losses.append(avg_loss)
            
            if (epoch + 1) % self.config.log_every == 0:
                lr = self._get_lr()
                print(
                    f"Epoch {epoch+1:>4d}/{self.config.num_epochs} | "
                    f"Loss: {avg_loss:.6f} | "
                    f"LR: {lr:.2e} | "
                    f"EMA decay: {self.ema.get_decay():.6f}"
                )
        
        return epoch_losses
    
    def save_checkpoint(self, path: str) -> None:
        """保存训练检查点"""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "ema_state_dict": self.ema.state_dict(),
            "global_step": self.global_step,
            "config": self.config,
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """加载训练检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.ema.load_state_dict(checkpoint["ema_state_dict"])
        self.global_step = checkpoint["global_step"]
        print(f"Checkpoint loaded from {path} (step {self.global_step})")


# ============================================================
# 6. FID 分数计算
# ============================================================

class FIDCalculator:
    """FID (Frechet Inception Distance) 计算器
    
    简化版实现。生产环境建议使用 torchmetrics 或 clean-fid 库。
    
    FID = ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2 * (Sigma_r @ Sigma_g)^{1/2})
    """
    
    @staticmethod
    def compute_statistics(
        features: torch.Tensor,  # (N, D) 特征矩阵
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """计算特征的均值和协方差矩阵"""
        mu = features.mean(dim=0)  # (D,)
        features_centered = features - mu.unsqueeze(0)  # (N, D)
        sigma = (features_centered.T @ features_centered) / (features.shape[0] - 1)  # (D, D)
        return mu, sigma
    
    @staticmethod
    def matrix_sqrt(matrix: torch.Tensor) -> torch.Tensor:
        """计算矩阵平方根（通过特征值分解）"""
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
        # 截断负特征值（数值误差）
        eigenvalues = torch.clamp(eigenvalues, min=0.0)
        sqrt_eigenvalues = torch.sqrt(eigenvalues)
        return eigenvectors @ torch.diag(sqrt_eigenvalues) @ eigenvectors.T
    
    @classmethod
    def compute_fid(
        cls,
        real_features: torch.Tensor,       # (N, D)
        generated_features: torch.Tensor,   # (M, D)
    ) -> float:
        """计算 FID 分数
        
        Args:
            real_features: 真实图像的特征
            generated_features: 生成图像的特征
            
        Returns:
            FID 分数（越低越好）
        """
        mu_r, sigma_r = cls.compute_statistics(real_features)
        mu_g, sigma_g = cls.compute_statistics(generated_features)
        
        # ||mu_r - mu_g||^2
        diff = mu_r - mu_g
        mean_term = torch.dot(diff, diff).item()
        
        # Tr(Sigma_r + Sigma_g - 2 * sqrt(Sigma_r @ Sigma_g))
        product = sigma_r @ sigma_g  # (D, D)
        sqrt_product = cls.matrix_sqrt(product)
        
        cov_term = (
            torch.trace(sigma_r).item()
            + torch.trace(sigma_g).item()
            - 2.0 * torch.trace(sqrt_product).item()
        )
        
        fid = mean_term + cov_term
        return fid


# ============================================================
# 7. 演示：比较三种预测目标
# ============================================================

def demo_prediction_targets():
    """演示三种预测目标的等价性"""
    torch.manual_seed(42)
    
    # 模拟数据
    batch_size = 1000
    dim = 4
    x_0 = torch.randn(batch_size, dim) * 0.5 + 1.0   # (B, D)
    eps = torch.randn(batch_size, dim)                  # (B, D)
    
    # 模拟 alpha_bar
    alpha_bar = 0.3  # 中等噪声
    sqrt_ab = math.sqrt(alpha_bar)
    sqrt_1_ab = math.sqrt(1.0 - alpha_bar)
    
    # 计算 x_t
    x_t = sqrt_ab * x_0 + sqrt_1_ab * eps  # (B, D)
    
    # 计算 v 目标
    v = sqrt_ab * eps - sqrt_1_ab * x_0  # (B, D)
    
    # 验证转换
    converter = PredictionConverter()
    
    # eps -> x0
    x0_from_eps = converter.eps_to_x0(
        eps, x_t,
        torch.tensor(sqrt_ab), torch.tensor(sqrt_1_ab),
    )
    assert torch.allclose(x0_from_eps, x_0, atol=1e-5), "eps -> x0 转换失败"
    
    # v -> eps
    eps_from_v = converter.v_to_eps(
        v, x_t,
        torch.tensor(sqrt_ab), torch.tensor(sqrt_1_ab),
    )
    assert torch.allclose(eps_from_v, eps, atol=1e-5), "v -> eps 转换失败"
    
    # v -> x0
    x0_from_v = converter.v_to_x0(
        v, x_t,
        torch.tensor(sqrt_ab), torch.tensor(sqrt_1_ab),
    )
    assert torch.allclose(x0_from_v, x_0, atol=1e-5), "v -> x0 转换失败"
    
    # 验证 v 的方差约为 1
    v_var = v.var(dim=0).mean().item()
    print(f"v 的方差: {v_var:.4f} (理论值约 1.0)")
    print(f"eps 的方差: {eps.var(dim=0).mean().item():.4f}")
    print(f"x_0 的方差: {x_0.var(dim=0).mean().item():.4f}")
    
    # 验证损失等价性
    # 假设预测完美，损失应该为 0
    loss_eps = DiffusionLoss("epsilon")
    loss_x0 = DiffusionLoss("x0")
    loss_v = DiffusionLoss("v")
    
    # 用真实值作为预测
    l_eps = loss_eps(eps, eps, x_0, torch.tensor(sqrt_ab), torch.tensor(sqrt_1_ab))
    l_x0 = loss_x0(x_0, eps, x_0, torch.tensor(sqrt_ab), torch.tensor(sqrt_1_ab))
    l_v = loss_v(v, eps, x_0, torch.tensor(sqrt_ab), torch.tensor(sqrt_1_ab))
    
    print(f"\n完美预测时的损失:")
    print(f"  L_epsilon: {l_eps.mean().item():.8f}")
    print(f"  L_x0:      {l_x0.mean().item():.8f}")
    print(f"  L_v:       {l_v.mean().item():.8f}")
    
    # 验证 L_epsilon / L_x0 = SNR
    # 添加预测误差
    pred_error = torch.randn(batch_size, dim) * 0.1
    
    l_eps_err = loss_eps(
        eps + pred_error, eps, x_0,
        torch.tensor(sqrt_ab), torch.tensor(sqrt_1_ab),
    )
    l_x0_err = loss_x0(
        x_0 + pred_error, eps, x_0,
        torch.tensor(sqrt_ab), torch.tensor(sqrt_1_ab),
    )
    
    ratio = l_eps_err.mean().item() / max(l_x0_err.mean().item(), 1e-10)
    snr = alpha_bar / (1 - alpha_bar)
    print(f"\n损失比 L_eps/L_x0: {ratio:.4f}")
    print(f"SNR(t):            {snr:.4f}")
    print(f"（注：比值在预测误差较大时不精确等于 SNR，因为误差来源不同）")
    
    print("\n所有转换验证通过!")


def demo_min_snr_weighting():
    """演示 Min-SNR 加权效果"""
    import matplotlib.pyplot as plt
    
    T = 1000
    # 使用线性调度
    betas = torch.linspace(1e-4, 0.02, T)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    snr = alpha_bars / (1.0 - alpha_bars)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    t_range = np.arange(T)
    
    # SNR 曲线
    axes[0].plot(t_range, torch.log10(snr + 1e-10).numpy(), linewidth=2)
    axes[0].set_xlabel("Timestep t")
    axes[0].set_ylabel("log10(SNR)")
    axes[0].set_title("Signal-to-Noise Ratio")
    axes[0].grid(True, alpha=0.3)
    
    # Min-SNR 权重
    for gamma in [1, 5, 10, 20]:
        weights = torch.clamp(snr, max=gamma) / snr
        axes[1].plot(
            t_range, weights.numpy(),
            label=f"gamma={gamma}",
            linewidth=1.5,
        )
    axes[1].set_xlabel("Timestep t")
    axes[1].set_ylabel("Weight")
    axes[1].set_title("Min-SNR Weights (epsilon-prediction)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 加权后的有效损失分布
    # 假设各时间步的原始损失近似相等
    for gamma in [1, 5, 10]:
        effective = torch.clamp(snr, max=gamma)
        effective = effective / effective.sum()  # 归一化
        axes[2].plot(
            t_range, effective.numpy(),
            label=f"gamma={gamma}",
            linewidth=1.5,
        )
    # 均匀权重
    uniform = torch.ones(T) / T
    axes[2].plot(
        t_range, uniform.numpy(),
        label="Uniform",
        linewidth=1.5,
        linestyle="--",
        color="gray",
    )
    axes[2].set_xlabel("Timestep t")
    axes[2].set_ylabel("Normalized Weight")
    axes[2].set_title("Effective Loss Distribution")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("min_snr_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Min-SNR 分析图已保存为 min_snr_analysis.png")


def demo_ema():
    """演示 EMA 的效果"""
    torch.manual_seed(42)
    
    # 简单的一维回归问题
    model = nn.Sequential(
        nn.Linear(1, 64),
        nn.SiLU(),
        nn.Linear(64, 64),
        nn.SiLU(),
        nn.Linear(64, 1),
    )
    
    ema = EMAModel(model, decay=0.999, warmup_steps=50)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 训练数据：带噪声的正弦函数
    x_train = torch.linspace(-3, 3, 200).unsqueeze(1)  # (200, 1)
    y_train = torch.sin(x_train) + torch.randn_like(x_train) * 0.3
    
    # 训练
    for step in range(500):
        idx = torch.randint(0, 200, (32,))
        x_batch = x_train[idx]
        y_batch = y_train[idx]
        
        pred = model(x_batch)
        loss = F.mse_loss(pred, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update()
    
    # 比较原始模型和EMA模型
    x_test = torch.linspace(-3, 3, 100).unsqueeze(1)
    
    model.eval()
    with torch.no_grad():
        y_original = model(x_test)
    
    with ema.apply_shadow():
        with torch.no_grad():
            y_ema = model(x_test)
    
    # 验证 apply_shadow 是否正确恢复了原始参数
    with torch.no_grad():
        y_after_restore = model(x_test)
    assert torch.allclose(y_original, y_after_restore), "EMA apply_shadow 恢复失败"
    
    print("EMA 演示完成")
    print(f"  原始模型 MSE: {F.mse_loss(y_original, torch.sin(x_test)).item():.4f}")
    print(f"  EMA模型 MSE:  {F.mse_loss(y_ema, torch.sin(x_test)).item():.4f}")


def demo_fid():
    """演示 FID 计算"""
    torch.manual_seed(42)
    
    # 模拟两组特征
    dim = 64
    n_real = 5000
    n_gen = 5000
    
    # 真实特征：均值为 [1, 0, ...], 协方差接近单位阵
    mu_real = torch.zeros(dim)
    mu_real[0] = 1.0
    real_features = torch.randn(n_real, dim) + mu_real.unsqueeze(0)
    
    # 场景1：生成特征与真实特征很相似
    gen_features_good = torch.randn(n_gen, dim) + mu_real.unsqueeze(0) + 0.1 * torch.randn(1, dim)
    
    # 场景2：生成特征与真实特征差异较大
    mu_bad = torch.zeros(dim)
    mu_bad[0] = 2.0
    gen_features_bad = torch.randn(n_gen, dim) * 1.5 + mu_bad.unsqueeze(0)
    
    fid_good = FIDCalculator.compute_fid(real_features, gen_features_good)
    fid_bad = FIDCalculator.compute_fid(real_features, gen_features_bad)
    fid_self = FIDCalculator.compute_fid(real_features, real_features)
    
    print("FID 计算演示:")
    print(f"  FID (真实 vs 真实): {fid_self:.4f}  (理论上应接近 0)")
    print(f"  FID (真实 vs 好生成): {fid_good:.4f}")
    print(f"  FID (真实 vs 差生成): {fid_bad:.4f}")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("第九章代码实战：噪声预测网络与训练目标")
    print("=" * 60)
    
    # 1. 时间步嵌入演示
    print("\n--- 1. 时间步嵌入 ---")
    time_emb = SinusoidalTimeEmbedding(dim=128)
    t_sample = torch.tensor([0, 100, 500, 999])
    emb = time_emb(t_sample)  # (4, 128)
    print(f"时间步嵌入形状: {emb.shape}")
    
    # 相邻时间步的相似度
    all_t = torch.arange(1000)
    all_emb = time_emb(all_t)  # (1000, 128)
    # 归一化
    all_emb_norm = F.normalize(all_emb, dim=1)
    # 相邻余弦相似度
    cos_sim = (all_emb_norm[:-1] * all_emb_norm[1:]).sum(dim=1)
    print(f"相邻时间步余弦相似度 - 均值: {cos_sim.mean():.4f}, 最小值: {cos_sim.min():.4f}")
    
    # 2. 预测目标转换验证
    print("\n--- 2. 预测目标转换 ---")
    demo_prediction_targets()
    
    # 3. EMA 演示
    print("\n--- 3. EMA 演示 ---")
    demo_ema()
    
    # 4. FID 计算演示
    print("\n--- 4. FID 计算 ---")
    demo_fid()
    
    # 5. Min-SNR 可视化
    print("\n--- 5. Min-SNR 分析 ---")
    demo_min_snr_weighting()
    
    print("\n" + "=" * 60)
    print("所有演示完成!")
    print("=" * 60)
```

---

## 本章小结

| 概念 | 要点 |
|:---|:---|
| 时间步嵌入 | 正弦位置编码 + MLP投影，通过加法/缩放/AdaLN注入网络 |
| $\epsilon$-预测 | DDPM默认，$L = \|\epsilon - \epsilon_\theta(x_t,t)\|^2$，数值稳定 |
| $x_0$-预测 | 直接预测原始数据，$L = \|x_0 - x_\theta(x_t,t)\|^2$，高噪声时不稳定 |
| $v$-预测 | $v = \sqrt{\bar{\alpha}_t}\epsilon - \sqrt{1-\bar{\alpha}_t}x_0$，最佳数值特性 |
| 转换关系 | 三种预测可通过线性变换互相转换 |
| Min-SNR加权 | $w(t) = \min(\text{SNR}(t), \gamma)$，平衡各时间步的贡献 |
| EMA | $\theta_{EMA} \leftarrow m\theta_{EMA} + (1-m)\theta$，评估时使用影子参数 |
| 混合精度 | BF16优先，FP16需要损失缩放 |
| FID | $\text{FID} = \|\mu_r-\mu_g\|^2 + \text{Tr}(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2})$，越低越好 |

---

## 练习题

### 基础题

**练习 9.1**（时间步嵌入计算）

手动计算 $t = 100$、$d_{model} = 4$ 时的正弦位置编码向量 $[\sin(\frac{100}{1}), \cos(\frac{100}{1}), \sin(\frac{100}{100}), \cos(\frac{100}{100})]$。解释为什么不同频率分量捕捉了不同尺度的时间信息。

**练习 9.2**（预测目标转换）

给定 $\bar{\alpha}_t = 0.5$，$x_t = [1.0, -0.5]$，网络的 $\epsilon$-预测为 $\hat{\epsilon} = [0.3, 0.7]$。计算对应的 $\hat{x}_0$ 和 $\hat{v}$。

### 中级题

**练习 9.3**（$v$-预测的方差分析）

证明：当 $x_0 \sim \mathcal{N}(0, I)$，$\epsilon \sim \mathcal{N}(0, I)$ 且独立时，$v_t = \sqrt{\bar{\alpha}_t}\epsilon - \sqrt{1-\bar{\alpha}_t}x_0$ 的分布为 $\mathcal{N}(0, I)$，且 $\text{Var}(v_t)$ 不依赖于 $t$。对比之下，计算 $\frac{x_0 - \hat{x}_0^{\text{pred}}}{\text{normalization}}$ 在不同 $t$ 下的方差变化。

**练习 9.4**（Min-SNR理论）

对于 $\epsilon$-预测目标，证明 Min-SNR-$\gamma$ 加权等价于对ELBO的原始 $w(t)$ 加权施加一个截断：$\min(w(t), C(\gamma))$，其中 $C(\gamma)$ 是关于 $\gamma$ 的表达式。给出 $C(\gamma)$ 的具体形式。

### 提高题

**练习 9.5**（从头推导最优预测目标）

考虑一般形式的预测目标 $\hat{y} = a(t) \cdot \epsilon + b(t) \cdot x_0$，其中 $a(t), b(t)$ 是关于 $t$ 的确定性函数。定义损失为 $L = \mathbb{E}\|y - y_\theta(x_t, t)\|^2$。

(a) 证明无论 $a(t), b(t)$ 如何选择，训练目标都等价于（在适当加权下）学习同一个条件期望 $\mathbb{E}[y | x_t]$。

(b) 分析 $\text{Var}(y | x_t)$ 如何依赖于 $a(t), b(t)$ 的选择。证明 $v$-预测（$a = \sqrt{\bar{\alpha}_t}, b = -\sqrt{1-\bar{\alpha}_t}$）使得 $\text{Var}(y|x_t)$ 最小（在所有 $\|a\|^2 + \|b\|^2 = 1$ 的选择中）。

(c) 为什么最小条件方差对训练有利？从偏差-方差分解的角度讨论。

---

## 练习答案

### 练习 9.1 解答

$d_{model} = 4$，$\text{half} = 2$。

频率：$\omega_0 = 10000^{0/2} = 1$，$\omega_1 = 10000^{2/4} = 10000^{0.5} = 100$。

但注意公式中的约定是 $\frac{1}{10000^{2i/d_{model}}}$，所以实际频率缩放为：

$$f_0 = \frac{1}{10000^{0/4}} = 1, \quad f_1 = \frac{1}{10000^{2/4}} = \frac{1}{100}$$

编码向量：

$$\text{PE}(100) = \left[\sin(100 \times 1),\, \cos(100 \times 1),\, \sin(100 \times 0.01),\, \cos(100 \times 0.01)\right]$$

$$= [\sin(100),\, \cos(100),\, \sin(1),\, \cos(1)]$$

$$\approx [-0.506,\, 0.863,\, 0.841,\, 0.540]$$

高频分量（$f_0 = 1$）：$\sin(100), \cos(100)$ 对 $t$ 的微小变化（如 $t=100$ vs $t=101$）非常敏感，频率高，变化快。

低频分量（$f_1 = 0.01$）：$\sin(1), \cos(1)$ 变化缓慢，$t$ 从100变到200时才显著变化，捕捉宏观的时间阶段差异。

### 练习 9.2 解答

给定 $\bar{\alpha}_t = 0.5$：

$$\sqrt{\bar{\alpha}_t} = \sqrt{0.5} \approx 0.7071, \quad \sqrt{1-\bar{\alpha}_t} = \sqrt{0.5} \approx 0.7071$$

**计算 $\hat{x}_0$**：

$$\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\hat{\epsilon}}{\sqrt{\bar{\alpha}_t}} = \frac{[1.0, -0.5] - 0.7071 \times [0.3, 0.7]}{0.7071}$$

$$= \frac{[1.0 - 0.2121, -0.5 - 0.4950]}{0.7071} = \frac{[0.7879, -0.9950]}{0.7071} \approx [1.114, -1.407]$$

**计算 $\hat{v}$**：

$$\hat{v} = \sqrt{\bar{\alpha}_t}\hat{\epsilon} - \sqrt{1-\bar{\alpha}_t}\hat{x}_0$$

$$= 0.7071 \times [0.3, 0.7] - 0.7071 \times [1.114, -1.407]$$

$$= [0.2121, 0.4950] - [0.7877, -0.9949] = [-0.576, 1.490]$$

### 练习 9.3 解答

$$v_t = \sqrt{\bar{\alpha}_t}\epsilon - \sqrt{1-\bar{\alpha}_t}x_0$$

由 $x_0 \sim \mathcal{N}(0,I)$，$\epsilon \sim \mathcal{N}(0,I)$ 独立：

$$\mathbb{E}[v_t] = \sqrt{\bar{\alpha}_t}\mathbb{E}[\epsilon] - \sqrt{1-\bar{\alpha}_t}\mathbb{E}[x_0] = 0$$

$$\text{Var}(v_t) = \bar{\alpha}_t \text{Var}(\epsilon) + (1-\bar{\alpha}_t)\text{Var}(x_0) = \bar{\alpha}_t I + (1-\bar{\alpha}_t)I = I$$

所以 $v_t \sim \mathcal{N}(0, I)$ 对所有 $t$ 成立，方差不依赖于 $t$。

对比 $x_0$ 预测目标：预测误差 $\Delta x_0 = x_0 - \hat{x}_0$ 的方差依赖于 $t$。在大 $t$ 时 $x_t \approx \epsilon$，$x_0$ 的预测很困难，误差方差大；在小 $t$ 时 $x_t \approx x_0$，预测容易，误差方差小。这种方差的不均匀性导致训练不稳定。

### 练习 9.4 解答

ELBO中 $L_{t-1}$ 的原始权重（对 $\epsilon$-预测）为：

$$w_{\text{ELBO}}(t) = \frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1-\bar{\alpha}_t)}$$

$L_{simple}$ 使用均匀权重 $w_{simple}(t) = 1$。Min-SNR-$\gamma$ 权重为：

$$w_{\text{MinSNR}}(t) = \min\left(\frac{\bar{\alpha}_t}{1-\bar{\alpha}_t}, \gamma\right)$$

要证明 Min-SNR 等价于对 $w_{\text{ELBO}}$ 的截断。由于 $w_{\text{MinSNR}}$ 是相对于 $L_{simple}$ 的权重，它与 $w_{\text{ELBO}}$ 的关系为：

$$w_{\text{MinSNR}}(t) = \frac{\min(\text{SNR}(t), \gamma)}{\text{SNR}(t)} \times \text{SNR}(t) = \min(\text{SNR}(t), \gamma)$$

而 $w_{\text{ELBO}}(t) \propto \frac{1}{\text{SNR}(t)}$（对于 $\sigma_t^2 = \tilde{\beta}_t$ 的情况，经化简后）。

因此 Min-SNR 权重 $\times$ ELBO权重 $\propto \min(\text{SNR}, \gamma) / \text{SNR}$。这等价于将ELBO的有效权重截断为：

$$C(\gamma) = \frac{\gamma}{\text{SNR}(t)} \cdot w_{\text{ELBO}}(t)$$

当 $\text{SNR}(t) > \gamma$ 时。具体地，$C(\gamma)$ 满足对于低SNR区域权重不变，对于高SNR区域（小 $t$）权重被减小到 $\gamma \cdot w_{\text{ELBO}}(t) / \text{SNR}(t)$。

### 练习 9.5 解答

**(a)** 对于目标 $y = a(t)\epsilon + b(t)x_0$，最优预测为条件期望：

$$y_\theta^*(x_t, t) = \mathbb{E}[y | x_t, t] = a(t)\mathbb{E}[\epsilon|x_t] + b(t)\mathbb{E}[x_0|x_t]$$

由于 $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$，$x_0$ 和 $\epsilon$ 通过 $x_t$ 线性相关。因此 $\mathbb{E}[\epsilon|x_t]$ 和 $\mathbb{E}[x_0|x_t]$ 都是 $x_t$ 的函数（通过条件高斯的公式），且互相确定。所以无论 $a, b$ 如何选择，网络学到的核心信息（$\mathbb{E}[x_0|x_t]$ 或等价地 $\mathbb{E}[\epsilon|x_t]$）是相同的。

**(b)** 条件方差：

$$\text{Var}(y|x_t) = a(t)^2\text{Var}(\epsilon|x_t) + b(t)^2\text{Var}(x_0|x_t) + 2a(t)b(t)\text{Cov}(\epsilon, x_0|x_t)$$

对于标准正态先验 $x_0 \sim \mathcal{N}(0,I)$：

$$\text{Var}(\epsilon|x_t) = 1 - \frac{1-\bar{\alpha}_t}{1} = \bar{\alpha}_t$$

（这里做了简化，严格推导需要考虑真实数据分布。）

$v$-预测选择 $a = \sqrt{\bar{\alpha}_t}, b = -\sqrt{1-\bar{\alpha}_t}$，使得 $a^2 + b^2 = 1$。在 $\|a\|^2 + \|b\|^2 = 1$ 的约束下（保持目标的"尺度"一致），可以用拉格朗日乘数法证明 $v$-预测最小化了条件方差。

**(c)** 最小条件方差意味着回归目标的"噪声"最小。从偏差-方差分解 $\mathbb{E}[\|y - y_\theta\|^2] = \|\text{Bias}\|^2 + \text{Var}(y|x_t)$ 来看，方差项是不可约的（即使完美网络也无法消除的损失）。最小化方差项使得训练信号更"干净"——网络的梯度更多地反映了偏差信号而非目标本身的随机性，从而提高训练效率。

---

## 延伸阅读

1. **Ho, J., Jain, A., & Abbeel, P.** (2020). "Denoising Diffusion Probabilistic Models." *NeurIPS 2020.* -- $\epsilon$-预测和U-Net架构的原始论文。

2. **Salimans, T., & Ho, J.** (2022). "Progressive Distillation for Fast Sampling of Diffusion Models." *ICLR 2022.* -- 引入 $v$-预测参数化。

3. **Hang, T., Gu, S., Li, C., Bao, J., Chen, D., Hu, H., Geng, X., & Guo, B.** (2023). "Efficient Diffusion Training via Min-SNR Weighting Strategy." *CVPR 2023.* -- Min-SNR加权策略的理论和实验验证。

4. **Peebles, W., & Xie, S.** (2023). "Scalable Diffusion Models with Transformers (DiT)." *ICCV 2023.* -- 使用Transformer替代U-Net，AdaLN时间注入。

5. **Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., & Hochreiter, S.** (2017). "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium." *NeurIPS 2017.* -- FID指标的原始论文。

6. **Vaswani, A., et al.** (2017). "Attention is All You Need." *NeurIPS 2017.* -- 正弦位置编码的来源。

7. **Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B.** (2020). "Score-Based Generative Modeling through Stochastic Differential Equations." *ICLR 2021.* -- 概率流ODE和NLL计算。

---

[上一章：逆向去噪过程与DDPM](./08-reverse-denoising-ddpm.md) | [目录](../README.md) | [下一章：U-Net架构详解](../part4-architecture/10-unet-architecture.md)
