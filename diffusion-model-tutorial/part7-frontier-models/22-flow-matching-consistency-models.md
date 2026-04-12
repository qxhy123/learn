# 第二十二章：Flow Matching与一致性模型

> **本章导读**：扩散模型虽然在生成质量上取得了巨大成功，但其多步采样过程仍然是一个核心瓶颈——生成一张图像通常需要20-50步ODE/SDE求解。本章介绍两种突破这一瓶颈的前沿方法：Flow Matching通过学习更直的流轨迹使ODE更容易求解，而一致性模型则直接学习ODE轨迹到终点的映射，实现一步生成。这两种方法已经深刻影响了Stable Diffusion 3、FLUX、以及多种实时生成系统的设计。本章将从理论基础到代码实现，系统地讲解这些前沿技术。

**前置知识**：扩散模型（DDPM/DDIM）、概率流ODE、Score Matching基础、连续归一化流（CNF）基本概念

**预计学习时间**：4-5小时

---

## 学习目标

完成本章学习后，你将能够：

1. 理解Flow Matching的核心思想——条件流与直线插值路径的设计动机
2. 推导Flow Matching的训练目标并解释其与扩散模型的等价关系
3. 掌握Rectified Flow的直化（rectification）策略与ODE采样方法
4. 理解一致性模型的自一致性约束及其蒸馏与训练两种范式
5. 实现Flow Matching和一致性模型的核心代码并在2D数据集上训练验证

---

## 22.1 超越扩散：流模型

### 22.1.1 扩散模型的采样瓶颈

回顾扩散模型的采样过程。给定训练好的分数函数 $s_\theta(x_t, t)$ 或噪声预测 $\epsilon_\theta(x_t, t)$，需要求解概率流ODE：

$$\frac{dx}{dt} = f(x, t) - \frac{1}{2}g^2(t)\nabla_x \log p_t(x)$$

从 $x_T \sim \mathcal{N}(0, I)$ 积分到 $x_0$。这个ODE的求解需要多步迭代的原因是：

1. **弯曲轨迹**：扩散ODE定义的流线通常是弯曲的，欧拉法需要很小的步长才能精确积分
2. **非线性漂移**：漂移项 $f(x, t) - \frac{1}{2}g^2(t)s_\theta$ 高度非线性
3. **高斯先验约束**：前向过程必须以高斯噪声结束，这限制了噪声调度的选择

### 22.1.2 归一化流（Normalizing Flows）的回顾

归一化流（NF）是另一种基于变换的生成模型：

$$z \sim p_z(z), \quad x = f_\theta(z), \quad p_x(x) = p_z(f_\theta^{-1}(x)) \cdot |\det J_{f_\theta^{-1}}(x)|$$

NF的关键限制：
- **可逆性要求**：变换 $f_\theta$ 必须可逆，且雅可比行列式必须易于计算
- **架构限制**：这导致了coupling layers、autoregressive flows等受限架构
- **表达能力**：受限架构的表达能力弱于自由参数化的网络

**连续归一化流（CNF）**（Chen et al. 2018）通过神经ODE绕过了可逆性的显式约束：

$$\frac{dx}{dt} = v_\theta(x, t), \quad x(0) = z \sim p_z, \quad x(1) = x_{data}$$

但CNF的训练需要通过ODE求解器反向传播，计算成本高昂。

### 22.1.3 Flow Matching的核心直觉

Flow Matching的关键问题是：**能否直接指定一条从噪声到数据的路径，然后训练网络学习这条路径的速度场？**

答案是肯定的。如果我们选择最简单的路径——**直线**——那么：

- 路径：$x_t = (1-t)x_0 + t x_1$，其中 $x_0 \sim \mathcal{N}(0, I)$（噪声），$x_1 \sim p_{data}$（数据）
- 速度：$\frac{dx_t}{dt} = x_1 - x_0$

这就是Flow Matching的核心：训练速度网络 $v_\theta(x_t, t)$ 去逼近真实速度 $x_1 - x_0$。

---

## 22.2 Flow Matching（FM）

### 22.2.1 两篇独立工作

Flow Matching有两篇几乎同时的独立工作：

- **Lipman et al. (2022)**：*"Flow Matching for Generative Modeling"*
- **Liu et al. (2022)**：*"Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow"*（刘润森等）

两者的核心思想高度一致，但侧重点略有不同——前者更侧重理论框架，后者更强调直线轨迹的实用优势。

### 22.2.2 条件流路径

给定噪声样本 $x_0 \sim \mathcal{N}(0, I)$ 和数据样本 $x_1 \sim p_{data}$，定义条件流路径：

$$\phi_t(x_0 | x_1) = (1-t)x_0 + t x_1, \quad t \in [0, 1]$$

- $t=0$：$\phi_0 = x_0$（纯噪声）
- $t=1$：$\phi_1 = x_1$（数据）
- 中间：$\phi_t$ 是噪声到数据的线性插值

对应的条件速度场：

$$u_t(x | x_1) = \frac{d\phi_t}{dt} = x_1 - x_0$$

这是一个常数速度场——不依赖于 $t$，只取决于起终点。

### 22.2.3 Flow Matching训练目标

**边际速度场**是对条件速度场在数据分布上的期望：

$$u_t(x) = \mathbb{E}_{x_1 \sim p_{data}}[u_t(x | x_1) \cdot p_t(x | x_1)] / p_t(x)$$

直接计算 $u_t(x)$ 是困难的（需要知道 $p_t(x)$），但我们可以用**条件流匹配**（Conditional Flow Matching, CFM）目标来替代：

$$\mathcal{L}_{CFM} = \mathbb{E}_{t \sim U[0,1], \, x_0 \sim \mathcal{N}(0,I), \, x_1 \sim p_{data}} \left\| v_\theta(\phi_t(x_0|x_1), t) - (x_1 - x_0) \right\|^2$$

展开写就是：

$$\boxed{\mathcal{L}_{CFM} = \mathbb{E}_{t, x_0, x_1} \left\| v_\theta\big((1-t)x_0 + tx_1, \, t\big) - (x_1 - x_0) \right\|^2}$$

这个目标的梯度与边际流匹配目标的梯度一致（Lipman et al. 2022, Theorem 2），因此可以安全地用于训练。

### 22.2.4 与DDPM的对比

| 特性 | DDPM | Flow Matching |
|------|------|---------------|
| 前向过程 | $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$ | $x_t = (1-t)x_0 + tx_1$ |
| 时间范围 | $t \in \{0, 1, ..., T\}$（离散）或 $[0, T]$ | $t \in [0, 1]$ |
| 方向 | 数据→噪声（$t=0$为数据） | 噪声→数据（$t=0$为噪声） |
| 预测目标 | 噪声 $\epsilon$ 或分数 $\nabla \log p_t$ | 速度 $v = x_1 - x_0$ |
| 轨迹形状 | 弯曲（受噪声调度影响） | **直线**（线性插值） |
| 训练目标 | $\|\epsilon_\theta - \epsilon\|^2$ | $\|v_\theta - (x_1 - x_0)\|^2$ |
| ODE求解 | 概率流ODE（弯曲） | 直线ODE（简单） |

**直线轨迹的关键优势**：ODE越直，低阶数值方法（如Euler法）的误差越小，因此可以用更少的步数得到高质量样本。

### 22.2.5 Rectified Flow的直化策略

Rectified Flow（Liu et al. 2022）进一步提出了**Reflow**操作来让轨迹更直：

1. 训练第一轮Flow Matching模型 $v_\theta^{(1)}$
2. 使用 $v_\theta^{(1)}$ 从噪声 $x_0$ 生成对应的数据 $\hat{x}_1$，得到新的配对 $(x_0, \hat{x}_1)$
3. 用这些新配对重新训练 $v_\theta^{(2)}$

每一轮Reflow都使轨迹更直（因为 $(x_0, \hat{x}_1)$ 对应的是模型认为"正确"的起终点配对，而非随机配对）。理论上，经过无限轮Reflow，轨迹会收敛到**最优传输（OT）路径**——对于高斯先验到数据的映射，这些是不相交的直线。

### 22.2.6 更一般的插值路径

虽然线性插值 $(1-t)x_0 + tx_1$ 是最简单的选择，但Flow Matching框架支持更一般的路径：

$$\phi_t(x_0 | x_1) = \alpha_t x_1 + \sigma_t x_0$$

其中 $\alpha_0 = 0, \alpha_1 = 1, \sigma_0 = 1, \sigma_1 = 0$（满足边界条件）。

不同的 $(\alpha_t, \sigma_t)$ 对应不同的噪声调度。特别地：

- **线性**：$\alpha_t = t, \sigma_t = 1 - t$（Rectified Flow）
- **VP-type**：$\alpha_t = \sqrt{\bar{\alpha}_t}, \sigma_t = \sqrt{1-\bar{\alpha}_t}$（等价于DDPM的VP-SDE）
- **余弦**：$\alpha_t = \cos(\pi t / 2), \sigma_t = \sin(\pi t / 2)$

这揭示了Flow Matching与扩散模型的深层统一：**扩散模型是Flow Matching在特定噪声调度下的特例**。

---

## 22.3 FM的理论基础

### 22.3.1 连续归一化流与Flow Matching的关系

连续归一化流（CNF）定义了一个时间依赖的向量场 $v(x, t)$，通过ODE将先验分布传输到数据分布：

$$\frac{dx}{dt} = v(x, t), \quad x(0) \sim p_0 = \mathcal{N}(0, I)$$

概率密度沿着这个ODE的变化由**连续性方程**（即Liouville方程）描述：

$$\frac{\partial p_t}{\partial t} + \nabla \cdot (p_t v_t) = 0$$

传统CNF通过最大似然训练（需要反向传播通过ODE求解器），而Flow Matching直接通过回归速度场来训练——不需要求解任何ODE。

### 22.3.2 条件流匹配的正确性

为什么用条件目标 $\mathcal{L}_{CFM}$ 可以替代边际目标？关键定理：

**定理（Lipman et al. 2022）**：条件流匹配目标 $\mathcal{L}_{CFM}$ 和边际流匹配目标 $\mathcal{L}_{FM}$ 具有相同的梯度：

$$\nabla_\theta \mathcal{L}_{CFM} = \nabla_\theta \mathcal{L}_{FM}$$

直觉：虽然两个目标的值不同（$\mathcal{L}_{CFM}$ 有更高的方差），但它们的梯度方向一致。这是因为 $\mathcal{L}_{FM}$ 中减去的边际速度场 $u_t(x)$ 不依赖于 $\theta$（它是真实分布的性质），因此不影响梯度。

### 22.3.3 为什么FM的ODE更容易求解

直线轨迹的一个重要数学性质：

对于线性路径 $\phi_t = (1-t)x_0 + tx_1$，条件速度场 $u_t = x_1 - x_0$ 是常数。如果网络完美地学习了边际速度场 $v_\theta(x, t) = u_t(x)$，那么ODE：

$$\frac{dx}{dt} = v_\theta(x, t)$$

的轨迹是**尽可能直的**——在条件层面（给定 $x_0, x_1$）是完全笔直的直线，在边际层面（不同 $x_0$ 混合后）可能略有弯曲，但弯曲程度远小于VP-SDE对应的ODE。

**数值含义**：

- 一阶Euler法的局部截断误差为 $O(h^2)$，其中 $h$ 是步长
- 对于直线ODE，Euler法是精确的（误差为零）
- 对于接近直线的ODE，即使步长很大（如 $h = 0.5$，即2步），误差也很小

这就是Flow Matching可以用极少步数（1-5步）生成高质量样本的理论基础。

### 22.3.4 与最优传输的联系

Rectified Flow论文的一个重要理论结果：经过Reflow操作，流的轨迹趋向于**最优传输映射**。

最优传输（OT）寻找从分布 $p_0$ 到 $p_1$ 的代价最小的传输方案：

$$\min_{\pi \in \Pi(p_0, p_1)} \mathbb{E}_{(x_0, x_1) \sim \pi} [\|x_0 - x_1\|^2]$$

最优传输映射具有**不相交**的性质：不同的传输路径不会交叉。这意味着ODE的向量场在空间中是"光滑且有序的"，非常容易用低步数求解。

---

## 22.4 Stable Diffusion 3与FLUX

### 22.4.1 SD3的架构

Stable Diffusion 3（Esser et al. 2024）是Flow Matching在大规模文本到图像生成中的成功应用：

| 组件 | SD v1.5 | SD 3 |
|------|---------|------|
| 训练框架 | DDPM（$\epsilon$-prediction） | Flow Matching（$v$-prediction） |
| 骨干网络 | U-Net | MMDiT（MultiModal DiT） |
| 文本编码器 | CLIP ViT-L | CLIP-L + CLIP-G + T5-XXL |
| 噪声调度 | 线性/余弦β | Rectified Flow（线性插值） |
| 采样器 | DDIM/DPM-Solver | Euler ODE求解器 |

### 22.4.2 Rectified Flow在SD3中的应用

SD3使用Rectified Flow的线性插值路径：

$$z_t = (1-t) \cdot \epsilon + t \cdot z_0, \quad \epsilon \sim \mathcal{N}(0, I)$$

注意这里的时间方向约定与某些论文不同——$t=0$ 对应噪声，$t=1$ 对应数据。

训练目标：

$$\mathcal{L} = \mathbb{E}_{t, \epsilon, z_0}\left[\|v_\theta(z_t, t, c) - (z_0 - \epsilon)\|^2\right]$$

其中 $c$ 是文本条件（来自三个文本编码器的联合嵌入）。

### 22.4.3 SD3的时间步采样策略

SD3发现对训练时间步 $t$ 的采样分布有重要影响。它使用了**logit-normal分布**（而非均匀分布）来采样 $t$：

$$t = \sigma(\mathcal{N}(\mu, s^2))$$

其中 $\sigma$ 是sigmoid函数。这使得更多的训练资源集中在 $t \approx 0.5$ 附近（噪声和信号混合最复杂的区域），而减少对 $t \approx 0$ 和 $t \approx 1$（几乎纯噪声或纯信号）的关注。

### 22.4.4 MMDiT架构

SD3的MMDiT（详见第21章）在Flow Matching框架下工作，其条件化方式与DiT的adaLN-Zero类似，但扩展到处理多模态token：

```
文本token (CLIP-L + CLIP-G + T5-XXL)
    │
    ├──→ 投影到共同维度
    │
    └──→ 与图像token拼接
         │
         ▼
    Joint Self-Attention (文本+图像双向交互)
         │
         ▼
    独立的FFN (文本FFN + 图像FFN)
```

### 22.4.5 FLUX

FLUX.1（Black Forest Labs, 2024）是SD3团队成员创办的公司推出的改进版：

- **FLUX.1 [pro]**：商业API版本
- **FLUX.1 [dev]**：开源版（蒸馏版本，更快）
- **FLUX.1 [schnell]**：4步快速版（蒸馏+对齐）

FLUX的改进包括：
- 更大的MMDiT模型（~12B参数）
- 改进的文本编码策略
- Guidance Distillation用于减少采样步数
- 支持更灵活的分辨率和宽高比

---

## 22.5 一致性模型（Consistency Models）

### 22.5.1 核心动机

Song et al. (2023)提出了一个优雅的想法：如果我们已经知道ODE的终点 $x_0$，那何必沿着ODE一步步积分呢？

考虑概率流ODE的轨迹 $\{x_t\}_{t \in [\epsilon, T]}$。对于同一条轨迹上的任意两点 $x_s$ 和 $x_t$（$\epsilon \leq s < t \leq T$），它们应该映射到同一个终点 $x_\epsilon \approx x_0$。

一致性模型学习一个函数 $f_\theta(x_t, t)$，满足：

$$\boxed{f_\theta(x_s, s) = f_\theta(x_t, t), \quad \forall s, t \in [\epsilon, T], \text{ 在同一ODE轨迹上}}$$

这就是**自一致性（self-consistency）**约束。

### 22.5.2 一致性函数的定义

一致性函数 $f: (x, t) \mapsto x_\epsilon$ 需要满足：

1. **一致性条件**：ODE轨迹上的所有点映射到同一终点
2. **边界条件**：$f(x_\epsilon, \epsilon) = x_\epsilon$（在终点，函数是恒等映射）

实际参数化：

$$f_\theta(x, t) = c_{skip}(t) \cdot x + c_{out}(t) \cdot F_\theta(x, t)$$

其中 $c_{skip}(t)$ 和 $c_{out}(t)$ 是设计好的缩放函数，满足 $c_{skip}(\epsilon) = 1, c_{out}(\epsilon) = 0$（保证边界条件）。一种常见选择：

$$c_{skip}(t) = \frac{\sigma_{data}^2}{(t - \epsilon)^2 + \sigma_{data}^2}, \quad c_{out}(t) = \frac{\sigma_{data}(t - \epsilon)}{\sqrt{\sigma_{data}^2 + t^2}}$$

### 22.5.3 一致性蒸馏（Consistency Distillation, CD）

CD利用一个**预训练的扩散模型**（教师）来训练一致性模型（学生）。

核心想法：在教师模型的ODE轨迹上取相邻两点，要求学生对它们输出相同的值。

**训练步骤**：
1. 从数据采样 $x_0$，构造 $x_{t_n} = \sqrt{\bar{\alpha}_{t_n}}x_0 + \sqrt{1-\bar{\alpha}_{t_n}}\epsilon$
2. 用教师模型ODE一步估计前一个时间步的位置：$\hat{x}_{t_{n-1}} = \text{ODE\_step}(x_{t_n}, t_n, t_{n-1}; \theta_{teacher})$
3. 训练一致性模型使得 $f_\theta(x_{t_n}, t_n) \approx f_{\theta^-}(\hat{x}_{t_{n-1}}, t_{n-1})$

$$\mathcal{L}_{CD} = \mathbb{E}\left[d\big(f_\theta(x_{t_n}, t_n), \, f_{\theta^-}(\hat{x}_{t_{n-1}}, t_{n-1})\big)\right]$$

其中 $\theta^-$ 是指数移动平均（EMA）参数，$d(\cdot, \cdot)$ 是距离函数（如 $L_2$ 或 LPIPS）。

### 22.5.4 一致性训练（Consistency Training, CT）

CT不需要预训练的教师模型，直接从数据学习一致性函数。

核心想法：利用 $x_{t_n}$ 和 $x_{t_{n-1}}$ 在同一条（由数据点确定的）轨迹上这一事实。

$$x_{t_n} = \sqrt{\bar{\alpha}_{t_n}}x_0 + \sqrt{1-\bar{\alpha}_{t_n}}\epsilon$$
$$x_{t_{n-1}} = \sqrt{\bar{\alpha}_{t_{n-1}}}x_0 + \sqrt{1-\bar{\alpha}_{t_{n-1}}}\epsilon$$

注意这里 $x_0$ 和 $\epsilon$ 是相同的——这保证了两点在同一条ODE轨迹上。

$$\mathcal{L}_{CT} = \mathbb{E}\left[d\big(f_\theta(x_{t_n}, t_n), \, f_{\theta^-}(x_{t_{n-1}}, t_{n-1})\big)\right]$$

CT的优势：无需预训练教师模型，从头训练即可。但通常质量略低于CD。

### 22.5.5 一步生成与多步细化

**一步生成**：训练完成后，直接从噪声 $x_T \sim \mathcal{N}(0, T^2 I)$ 预测：

$$\hat{x}_0 = f_\theta(x_T, T)$$

一次前向传播即可生成图像。

**多步细化**：虽然一致性模型支持一步生成，但通过少量步数（2-4步）可以显著提升质量：

```
z_T ~ N(0, I)
    │
    ▼
x̂₀ = f_θ(z_T, T)            # 第1步：粗略预测
    │
    ▼
z_{t₁} = √ᾱ_{t₁} x̂₀ + √(1-ᾱ_{t₁}) ε₁    # 添加噪声到中间步
    │
    ▼
x̂₀ = f_θ(z_{t₁}, t₁)       # 第2步：精化
    │
    ▼
z_{t₂} = √ᾱ_{t₂} x̂₀ + √(1-ᾱ_{t₂}) ε₂    # 再次添加噪声
    │
    ▼
x̂₀ = f_θ(z_{t₂}, t₂)       # 第3步：进一步精化
```

每一步都是"噪声→预测→加噪→预测"的循环。由于每一步都直接预测 $x_0$（而非预测噪声后做一小步更新），质量提升比普通扩散模型的少步采样更显著。

### 22.5.6 性能对比

在ImageNet 64x64上的结果（Song et al. 2023）：

| 方法 | 步数 | FID↓ |
|------|------|------|
| DDPM | 256 | 2.07 |
| DDIM | 50 | 4.67 |
| DDIM | 10 | 8.23 |
| Progressive Distillation | 4 | 6.60 |
| Progressive Distillation | 1 | 9.12 |
| **Consistency Distillation** | **2** | **3.55** |
| **Consistency Distillation** | **1** | **6.20** |
| **Consistency Training** | **2** | **5.83** |
| **Consistency Training** | **1** | **8.70** |

CD在2步时已经接近256步DDPM的质量，这是巨大的加速。

---

## 22.6 未来方向

### 22.6.1 基于Flow Matching的视频生成

Flow Matching的直线轨迹特性使其特别适合视频生成——时间维度上的连续性要求更高效的采样：

- **CogVideoX**（智谱AI 2024）：使用3D VAE + Flow Matching Transformer生成高质量视频
- **Wan**（阿里2025）：基于Flow Matching的视频生成大模型
- **MovieGen**（Meta 2024）：大规模视频和音频生成

视频Flow Matching的关键挑战是计算量——视频的token数远大于图像（时间$\times$高$\times$宽），因此快速采样更加关键。

### 22.6.2 离散扩散模型

将扩散/Flow Matching扩展到离散空间（文本生成）：

- **D3PM**（Austin et al. 2021）：离散去噪扩散
- **MDLM**（Sahoo et al. 2024）：Masked Diffusion Language Models
- **离散Flow Matching**：在类别空间中定义连续时间的马尔可夫过程

这使得扩散模型可以应用于文本、代码、分子序列等离散数据。

### 22.6.3 科学应用

扩散模型在科学领域的突破性应用：

- **AlphaFold 3**（Abramson et al. 2024）：使用扩散模型预测蛋白质-配体复合物的3D结构
- **RFDiffusion**：蛋白质设计
- **分子生成**：Drug Discovery中的分子构象生成

这些应用利用了扩散模型对连续几何数据建模的强大能力。

### 22.6.4 世界模型

扩散模型作为世界模型（World Model）的核心组件：

- **Genie 2**（DeepMind 2024）：交互式3D环境生成
- **DIAMOND**（2024）：基于扩散的世界模型用于强化学习
- **UniSim**（2024）：统一的模拟器学习

世界模型需要根据动作生成下一帧——这本质上是条件扩散/流模型的应用。

### 22.6.5 技术融合趋势

未来的生成模型可能融合多种范式：

1. **Flow Matching + Transformer + 大规模训练** = 更强的基础模型
2. **一致性模型 + 蒸馏** = 实时生成能力
3. **多模态统一** = 图像/视频/3D/文本在同一框架中生成
4. **自回归 + 扩散混合** = 利用两者优势

---

## 代码实战

### 实战1：Flow Matching从零实现

```python
"""
Flow Matching与一致性模型完整实战
1. Flow Matching训练（2D toy数据集）
2. ODE采样（Euler + RK4）
3. 对比FM和DDPM的轨迹
4. 一致性模型简化实现
5. 一步生成演示
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Flow Matching从零实现
# ============================================================

class VelocityNetwork(nn.Module):
    """
    速度预测网络 v_θ(x_t, t)
    用于2D数据的简单MLP
    """
    
    def __init__(
        self,
        data_dim: int = 2,
        hidden_dim: int = 256,
        time_dim: int = 64,
    ) -> None:
        super().__init__()
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # 主网络
        self.net = nn.Sequential(
            nn.Linear(data_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D] 当前位置
            t: [B, 1] 时间步
        Returns:
            v: [B, D] 预测速度
        """
        t_emb = self.time_mlp(t)           # [B, time_dim]
        h = torch.cat([x, t_emb], dim=-1)  # [B, D + time_dim]
        return self.net(h)                  # [B, D]


class FlowMatchingTrainer:
    """
    Flow Matching训练器
    使用线性插值路径 x_t = (1-t)*x_0 + t*x_1
    """
    
    def __init__(
        self,
        model: VelocityNetwork,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.device = device
    
    def compute_loss(
        self,
        x_1: torch.Tensor,  # [B, D] 数据样本
    ) -> torch.Tensor:
        """
        计算Conditional Flow Matching损失
        
        L = E_{t, x_0, x_1} || v_θ(x_t, t) - (x_1 - x_0) ||^2
        """
        B = x_1.shape[0]
        
        # 采样噪声（先验分布）
        x_0 = torch.randn_like(x_1)  # [B, D], 标准高斯
        
        # 采样时间步
        t = torch.rand(B, 1, device=self.device)  # [B, 1], U[0,1]
        
        # 构造插值点
        x_t = (1 - t) * x_0 + t * x_1  # [B, D]
        
        # 真实速度（条件速度场）
        target_v = x_1 - x_0  # [B, D]
        
        # 网络预测
        pred_v = self.model(x_t, t)  # [B, D]
        
        # MSE损失
        loss = F.mse_loss(pred_v, target_v)
        return loss
    
    def train_step(self, x_1: torch.Tensor) -> float:
        """执行一步训练"""
        self.optimizer.zero_grad()
        loss = self.compute_loss(x_1)
        loss.backward()
        self.optimizer.step()
        return loss.item()


# ============================================================
# 2. ODE采样器（Euler + RK4）
# ============================================================

class ODESampler:
    """ODE采样器，支持Euler和RK4方法"""
    
    def __init__(
        self,
        model: VelocityNetwork,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.device = device
    
    @torch.no_grad()
    def euler_sample(
        self,
        num_samples: int,
        data_dim: int = 2,
        num_steps: int = 100,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """
        Euler法ODE采样
        从t=0（噪声）积分到t=1（数据）
        
        dx/dt = v_θ(x, t)
        x_{k+1} = x_k + h * v_θ(x_k, t_k)
        """
        dt = 1.0 / num_steps
        
        # 初始化噪声
        x = torch.randn(num_samples, data_dim, device=self.device)
        
        trajectory = [x.clone()] if return_trajectory else None
        
        for step in range(num_steps):
            t = torch.full(
                (num_samples, 1), step * dt, device=self.device
            )
            v = self.model(x, t)  # [B, D]
            x = x + dt * v       # Euler步进
            
            if return_trajectory:
                trajectory.append(x.clone())
        
        if return_trajectory:
            return x, torch.stack(trajectory)  # [B, D], [steps+1, B, D]
        return x
    
    @torch.no_grad()
    def rk4_sample(
        self,
        num_samples: int,
        data_dim: int = 2,
        num_steps: int = 20,
    ) -> torch.Tensor:
        """
        4阶Runge-Kutta ODE采样
        精度更高，相同步数下比Euler质量更好
        """
        dt = 1.0 / num_steps
        x = torch.randn(num_samples, data_dim, device=self.device)
        
        for step in range(num_steps):
            t_val = step * dt
            
            t = torch.full((num_samples, 1), t_val, device=self.device)
            t_mid = torch.full((num_samples, 1), t_val + dt/2, device=self.device)
            t_end = torch.full((num_samples, 1), t_val + dt, device=self.device)
            
            k1 = self.model(x, t)
            k2 = self.model(x + dt/2 * k1, t_mid)
            k3 = self.model(x + dt/2 * k2, t_mid)
            k4 = self.model(x + dt * k3, t_end)
            
            x = x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        
        return x


# ============================================================
# 3. 2D数据集
# ============================================================

def make_two_moons(n_samples: int = 1000, noise: float = 0.05) -> torch.Tensor:
    """生成2D双月形数据集"""
    from sklearn.datasets import make_moons
    data, _ = make_moons(n_samples=n_samples, noise=noise)
    return torch.tensor(data, dtype=torch.float32)


def make_swiss_roll_2d(n_samples: int = 1000) -> torch.Tensor:
    """生成2D瑞士卷数据集"""
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n_samples))
    x = t * np.cos(t) / 10
    y = t * np.sin(t) / 10
    data = np.stack([x, y], axis=1)
    return torch.tensor(data, dtype=torch.float32)


def make_8_gaussians(n_samples: int = 1000) -> torch.Tensor:
    """生成8个高斯分布混合的数据集"""
    centers = []
    for i in range(8):
        angle = 2 * np.pi * i / 8
        centers.append([2 * np.cos(angle), 2 * np.sin(angle)])
    centers = np.array(centers)
    
    # 每个中心生成 n_samples // 8 个点
    n_per = n_samples // 8
    data = []
    for center in centers:
        samples = np.random.randn(n_per, 2) * 0.15 + center
        data.append(samples)
    data = np.concatenate(data, axis=0)
    return torch.tensor(data, dtype=torch.float32)


# ============================================================
# 4. DDPM基线（用于轨迹对比）
# ============================================================

class NoisePredictor(nn.Module):
    """DDPM的噪声预测网络（与VelocityNetwork结构相同）"""
    
    def __init__(
        self,
        data_dim: int = 2,
        hidden_dim: int = 256,
        time_dim: int = 64,
    ) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(data_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        h = torch.cat([x, t_emb], dim=-1)
        return self.net(h)


class DDPMTrainer:
    """DDPM训练器（用于对比实验）"""
    
    def __init__(
        self,
        model: NoisePredictor,
        num_timesteps: int = 1000,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.device = device
        self.T = num_timesteps
        
        # 噪声调度
        betas = torch.linspace(1e-4, 0.02, num_timesteps, device=device)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    def train_step(self, x_0: torch.Tensor) -> float:
        B = x_0.shape[0]
        
        # 采样时间步
        t_idx = torch.randint(0, self.T, (B,), device=self.device)
        t_normalized = t_idx.float()[:, None] / self.T  # 归一化到[0,1]
        
        noise = torch.randn_like(x_0)
        
        sqrt_alpha = self.alphas_cumprod[t_idx].sqrt()[:, None]
        sqrt_one_minus = (1 - self.alphas_cumprod[t_idx]).sqrt()[:, None]
        
        x_t = sqrt_alpha * x_0 + sqrt_one_minus * noise
        
        noise_pred = self.model(x_t, t_normalized)
        loss = F.mse_loss(noise_pred, noise)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        data_dim: int = 2,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """DDIM采样（确定性，用于轨迹可视化）"""
        num_steps = 50
        step_indices = torch.linspace(self.T - 1, 0, num_steps).long()
        
        x = torch.randn(num_samples, data_dim, device=self.device)
        trajectory = [x.clone()] if return_trajectory else None
        
        for i in range(len(step_indices)):
            t_idx = step_indices[i]
            t_normalized = t_idx.float().unsqueeze(0).expand(num_samples, 1) / self.T
            
            alpha_t = self.alphas_cumprod[t_idx]
            noise_pred = self.model(x, t_normalized)
            
            # 预测x_0
            x0_pred = (x - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
            
            if i < len(step_indices) - 1:
                t_prev = step_indices[i + 1]
                alpha_prev = self.alphas_cumprod[t_prev]
                x = alpha_prev.sqrt() * x0_pred + (1 - alpha_prev).sqrt() * noise_pred
            else:
                x = x0_pred
            
            if return_trajectory:
                trajectory.append(x.clone())
        
        if return_trajectory:
            return x, torch.stack(trajectory)
        return x


# ============================================================
# 5. 一致性模型简化实现
# ============================================================

class ConsistencyModel(nn.Module):
    """
    一致性模型简化实现
    f_θ(x_t, t) 直接预测 x_0
    """
    
    def __init__(
        self,
        data_dim: int = 2,
        hidden_dim: int = 256,
        time_dim: int = 64,
        sigma_data: float = 0.5,
    ) -> None:
        super().__init__()
        self.sigma_data = sigma_data
        
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        self.net = nn.Sequential(
            nn.Linear(data_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),
        )
    
    def c_skip(self, t: torch.Tensor) -> torch.Tensor:
        """跳跃连接系数，保证边界条件 f(x, eps) = x"""
        return self.sigma_data ** 2 / ((t - 0.001) ** 2 + self.sigma_data ** 2)
    
    def c_out(self, t: torch.Tensor) -> torch.Tensor:
        """输出缩放系数"""
        return self.sigma_data * (t - 0.001) / (self.sigma_data ** 2 + t ** 2).sqrt()
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        一致性函数: f_θ(x_t, t) = c_skip(t) * x + c_out(t) * F_θ(x, t)
        
        Args:
            x: [B, D] 带噪输入
            t: [B, 1] 时间步
        Returns:
            x_pred: [B, D] 预测的干净数据
        """
        c_s = self.c_skip(t)  # [B, 1]
        c_o = self.c_out(t)   # [B, 1]
        
        t_emb = self.time_mlp(t)
        h = torch.cat([x, t_emb], dim=-1)
        F_out = self.net(h)  # [B, D]
        
        return c_s * x + c_o * F_out


class ConsistencyTrainer:
    """
    一致性训练（CT）
    不需要预训练教师模型
    """
    
    def __init__(
        self,
        model: ConsistencyModel,
        lr: float = 1e-4,
        ema_rate: float = 0.999,
        num_timesteps: int = 150,
        sigma_max: float = 80.0,
        sigma_min: float = 0.002,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.device = device
        
        # EMA模型（目标网络）
        self.ema_model = ConsistencyModel(
            data_dim=2, hidden_dim=256, sigma_data=model.sigma_data
        ).to(device)
        self.ema_model.load_state_dict(model.state_dict())
        self.ema_rate = ema_rate
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # 时间步离散化
        self.N = num_timesteps
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        
        # 生成时间步序列
        self.sigmas = torch.linspace(
            sigma_min, sigma_max, num_timesteps, device=device
        )
    
    @torch.no_grad()
    def update_ema(self) -> None:
        """更新EMA目标网络"""
        for p, p_ema in zip(
            self.model.parameters(), self.ema_model.parameters()
        ):
            p_ema.data.mul_(self.ema_rate).add_(p.data, alpha=1 - self.ema_rate)
    
    def train_step(self, x_0: torch.Tensor) -> float:
        """
        一致性训练的一步
        """
        B = x_0.shape[0]
        
        # 采样相邻时间步索引
        n = torch.randint(1, self.N, (B,), device=self.device)
        
        t_n = self.sigmas[n][:, None]      # [B, 1] 当前时间步
        t_n1 = self.sigmas[n - 1][:, None]  # [B, 1] 前一个时间步
        
        # 采样噪声
        noise = torch.randn_like(x_0)
        
        # 构造两个噪声点（同一轨迹上）
        x_tn = x_0 + t_n * noise     # [B, D]
        x_tn1 = x_0 + t_n1 * noise   # [B, D]，同一个noise保证同一轨迹
        
        # 在线模型预测
        pred_online = self.model(x_tn, t_n)           # f_θ(x_{t_n}, t_n)
        
        # EMA目标模型预测（不计算梯度）
        with torch.no_grad():
            pred_target = self.ema_model(x_tn1, t_n1)  # f_{θ⁻}(x_{t_{n-1}}, t_{n-1})
        
        # 一致性损失：两个预测应该相同
        loss = F.mse_loss(pred_online, pred_target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新EMA
        self.update_ema()
        
        return loss.item()
    
    @torch.no_grad()
    def one_step_generate(
        self,
        num_samples: int,
        data_dim: int = 2,
    ) -> torch.Tensor:
        """一步生成"""
        x_T = torch.randn(num_samples, data_dim, device=self.device) * self.sigma_max
        t = torch.full((num_samples, 1), self.sigma_max, device=self.device)
        return self.model(x_T, t)
    
    @torch.no_grad()
    def multi_step_generate(
        self,
        num_samples: int,
        data_dim: int = 2,
        num_steps: int = 3,
    ) -> torch.Tensor:
        """多步细化生成"""
        x = torch.randn(num_samples, data_dim, device=self.device) * self.sigma_max
        
        # 选择中间时间步
        step_sigmas = torch.linspace(
            self.sigma_max, self.sigma_min, num_steps + 1, device=self.device
        )
        
        for i in range(num_steps):
            t = torch.full((num_samples, 1), step_sigmas[i].item(), device=self.device)
            x_pred = self.model(x, t)  # 预测x_0
            
            if i < num_steps - 1:
                # 重新添加噪声到下一个时间步
                noise = torch.randn_like(x)
                x = x_pred + step_sigmas[i + 1] * noise
            else:
                x = x_pred
        
        return x


# ============================================================
# 6. 训练与可视化
# ============================================================

def train_and_compare(
    dataset_fn=make_8_gaussians,
    n_samples: int = 2000,
    n_train_steps: int = 10000,
    device: str = "cpu",
) -> None:
    """训练Flow Matching和DDPM并对比结果"""
    
    # 生成数据
    data = dataset_fn(n_samples).to(device)
    
    # ---------- 训练Flow Matching ----------
    fm_model = VelocityNetwork(data_dim=2).to(device)
    fm_trainer = FlowMatchingTrainer(fm_model, lr=1e-3, device=device)
    
    print("训练 Flow Matching...")
    for step in range(n_train_steps):
        idx = torch.randint(0, len(data), (256,))
        batch = data[idx]
        loss = fm_trainer.train_step(batch)
        if (step + 1) % 2000 == 0:
            print(f"  Step {step+1}, FM Loss: {loss:.4f}")
    
    # ---------- 训练DDPM ----------
    ddpm_model = NoisePredictor(data_dim=2).to(device)
    ddpm_trainer = DDPMTrainer(ddpm_model, lr=1e-3, device=device)
    
    print("\n训练 DDPM...")
    for step in range(n_train_steps):
        idx = torch.randint(0, len(data), (256,))
        batch = data[idx]
        loss = ddpm_trainer.train_step(batch)
        if (step + 1) % 2000 == 0:
            print(f"  Step {step+1}, DDPM Loss: {loss:.4f}")
    
    # ---------- 采样并对比轨迹 ----------
    fm_sampler = ODESampler(fm_model, device=device)
    
    # FM采样（带轨迹）
    fm_samples, fm_traj = fm_sampler.euler_sample(
        500, data_dim=2, num_steps=100, return_trajectory=True
    )
    
    # DDPM采样（带轨迹）
    ddpm_samples, ddpm_traj = ddpm_trainer.sample(
        500, data_dim=2, return_trajectory=True
    )
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 真实数据
    axes[0, 0].scatter(data[:, 0].cpu(), data[:, 1].cpu(), s=5, alpha=0.3)
    axes[0, 0].set_title("Ground Truth Data")
    axes[0, 0].set_xlim(-4, 4)
    axes[0, 0].set_ylim(-4, 4)
    
    # FM生成样本
    axes[0, 1].scatter(
        fm_samples[:, 0].cpu(), fm_samples[:, 1].cpu(), s=5, alpha=0.3, c="red"
    )
    axes[0, 1].set_title("Flow Matching Samples (100 steps)")
    axes[0, 1].set_xlim(-4, 4)
    axes[0, 1].set_ylim(-4, 4)
    
    # DDPM生成样本
    axes[0, 2].scatter(
        ddpm_samples[:, 0].cpu(), ddpm_samples[:, 1].cpu(), s=5, alpha=0.3, c="green"
    )
    axes[0, 2].set_title("DDPM Samples (50 steps)")
    axes[0, 2].set_xlim(-4, 4)
    axes[0, 2].set_ylim(-4, 4)
    
    # FM轨迹（选取10条）
    for i in range(10):
        traj_i = fm_traj[:, i, :].cpu().numpy()   # [steps, 2]
        axes[1, 0].plot(traj_i[:, 0], traj_i[:, 1], alpha=0.5, linewidth=0.8)
    axes[1, 0].set_title("FM Trajectories (nearly straight)")
    axes[1, 0].set_xlim(-4, 4)
    axes[1, 0].set_ylim(-4, 4)
    
    # DDPM轨迹（选取10条）
    for i in range(10):
        traj_i = ddpm_traj[:, i, :].cpu().numpy()
        axes[1, 1].plot(traj_i[:, 0], traj_i[:, 1], alpha=0.5, linewidth=0.8)
    axes[1, 1].set_title("DDPM Trajectories (curved)")
    axes[1, 1].set_xlim(-4, 4)
    axes[1, 1].set_ylim(-4, 4)
    
    # 少步FM采样对比
    fm_1step = fm_sampler.euler_sample(500, data_dim=2, num_steps=1)
    fm_5step = fm_sampler.euler_sample(500, data_dim=2, num_steps=5)
    axes[1, 2].scatter(
        fm_1step[:, 0].cpu(), fm_1step[:, 1].cpu(),
        s=5, alpha=0.3, c="orange", label="1 step"
    )
    axes[1, 2].scatter(
        fm_5step[:, 0].cpu(), fm_5step[:, 1].cpu(),
        s=5, alpha=0.3, c="blue", label="5 steps"
    )
    axes[1, 2].set_title("FM: 1-step vs 5-step")
    axes[1, 2].set_xlim(-4, 4)
    axes[1, 2].set_ylim(-4, 4)
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig("fm_vs_ddpm_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n对比图已保存到 fm_vs_ddpm_comparison.png")


def train_consistency_model(
    dataset_fn=make_8_gaussians,
    n_samples: int = 2000,
    n_train_steps: int = 15000,
    device: str = "cpu",
) -> None:
    """训练一致性模型并演示一步/多步生成"""
    
    data = dataset_fn(n_samples).to(device)
    
    # 训练一致性模型
    cm = ConsistencyModel(data_dim=2, hidden_dim=256, sigma_data=0.5)
    trainer = ConsistencyTrainer(cm, lr=1e-4, device=device)
    
    print("训练一致性模型...")
    for step in range(n_train_steps):
        idx = torch.randint(0, len(data), (256,))
        batch = data[idx]
        loss = trainer.train_step(batch)
        if (step + 1) % 3000 == 0:
            print(f"  Step {step+1}, CM Loss: {loss:.4f}")
    
    # 生成
    samples_1step = trainer.one_step_generate(500, data_dim=2)
    samples_3step = trainer.multi_step_generate(500, data_dim=2, num_steps=3)
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].scatter(data[:, 0].cpu(), data[:, 1].cpu(), s=5, alpha=0.3)
    axes[0].set_title("Ground Truth")
    axes[0].set_xlim(-4, 4)
    axes[0].set_ylim(-4, 4)
    
    axes[1].scatter(
        samples_1step[:, 0].cpu(), samples_1step[:, 1].cpu(),
        s=5, alpha=0.3, c="red"
    )
    axes[1].set_title("Consistency Model: 1-step")
    axes[1].set_xlim(-4, 4)
    axes[1].set_ylim(-4, 4)
    
    axes[2].scatter(
        samples_3step[:, 0].cpu(), samples_3step[:, 1].cpu(),
        s=5, alpha=0.3, c="blue"
    )
    axes[2].set_title("Consistency Model: 3-step")
    axes[2].set_xlim(-4, 4)
    axes[2].set_ylim(-4, 4)
    
    plt.tight_layout()
    plt.savefig("consistency_model_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n一致性模型结果已保存到 consistency_model_results.png")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 1. 训练FM和DDPM并对比
    train_and_compare(
        dataset_fn=make_8_gaussians,
        n_train_steps=10000,
        device=device,
    )
    
    # 2. 训练一致性模型
    train_consistency_model(
        dataset_fn=make_8_gaussians,
        n_train_steps=15000,
        device=device,
    )
    
    # 3. 快速测试：FM不同步数的采样质量
    print("\n=== FM采样步数 vs 质量 ===")
    fm_model = VelocityNetwork(data_dim=2).to(device)
    fm_trainer = FlowMatchingTrainer(fm_model, lr=1e-3, device=device)
    
    data = make_8_gaussians(2000).to(device)
    for step in range(10000):
        idx = torch.randint(0, len(data), (256,))
        fm_trainer.train_step(data[idx])
    
    sampler = ODESampler(fm_model, device=device)
    for n_steps in [1, 2, 5, 10, 20, 50, 100]:
        samples = sampler.euler_sample(1000, num_steps=n_steps)
        # 简单的质量度量：到最近数据点的平均距离
        dists = torch.cdist(samples, data)  # [1000, 2000]
        min_dists = dists.min(dim=1).values  # [1000]
        print(f"  Steps={n_steps:>3}: mean dist to data = {min_dists.mean():.4f}")
```

---

## 本章小结

| 概念 | 核心要点 |
|------|---------|
| 扩散模型瓶颈 | 弯曲ODE轨迹需要多步求解，采样速度慢 |
| Flow Matching | 学习从噪声到数据的直线速度场，$\mathcal{L}_{CFM} = \|v_\theta(x_t, t) - (x_1 - x_0)\|^2$ |
| 线性插值 | $x_t = (1-t)x_0 + tx_1$，速度 $v = x_1 - x_0$ 为常数 |
| Rectified Flow | 通过Reflow操作进一步拉直轨迹，趋向最优传输 |
| FM vs DDPM | FM轨迹更直，ODE更容易求解，少步采样质量更好 |
| FM与扩散的统一 | 扩散模型是FM在特定 $(\alpha_t, \sigma_t)$ 调度下的特例 |
| SD3/FLUX | 采用Flow Matching + MMDiT，logit-normal时间步采样 |
| 一致性模型 | ODE轨迹上所有点映射到同一终点 $x_0$，实现一步生成 |
| 一致性蒸馏(CD) | 用预训练教师模型引导，2步即接近256步DDPM质量 |
| 一致性训练(CT) | 无需教师模型，直接从数据训练一致性函数 |
| 多步细化 | 一步粗预测→加噪→精化，2-4步显著提升质量 |

---

## 练习题

### 基础题

**练习1**：写出Flow Matching和DDPM在以下方面的对应关系：
- (a) 前向过程（噪声构造方式）
- (b) 预测目标
- (c) 采样方式
- (d) 时间方向约定

**练习2**：解释一致性模型的边界条件 $f(x_\epsilon, \epsilon) = x_\epsilon$ 为什么必须满足。如果违反这个条件会怎样？

### 中级题

**练习3**：在上面的Flow Matching实现中，修改代码以支持**余弦插值路径**：
$$\phi_t(x_0 | x_1) = \cos(\pi t / 2) \cdot x_0 + \sin(\pi t / 2) \cdot x_1$$
- (a) 推导对应的条件速度场
- (b) 修改训练代码
- (c) 实验对比线性路径和余弦路径的效果

**练习4**：一致性蒸馏（CD）需要用教师模型ODE走一步来生成训练数据对。请设计一个完整的CD训练流程，使用之前训练好的Flow Matching模型作为教师。写出代码并分析质量与一致性训练（CT）的差异。

### 提高题

**练习5**：Rectified Flow的Reflow操作声称可以让轨迹越来越直。设计一个实验来验证这一点：
- (a) 定义"轨迹直度"的量化指标（提示：考虑轨迹曲率或端到端距离与总路程的比值）
- (b) 在2D数据集上执行3轮Reflow
- (c) 绘制每轮Reflow后的轨迹可视化和直度指标变化曲线
- (d) 分析Reflow对少步采样质量（1步、2步、5步）的影响

---

## 练习答案

### 练习1

Flow Matching与DDPM的对应关系：

**(a) 前向过程（噪声构造）**：
- DDPM：$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$，其中 $\epsilon \sim \mathcal{N}(0,I)$
- FM：$x_t = (1-t) x_0^{noise} + t \cdot x_1^{data}$，其中 $x_0^{noise} \sim \mathcal{N}(0,I)$

注意方向不同：DDPM从数据加噪到噪声，FM从噪声插值到数据。

**(b) 预测目标**：
- DDPM：预测噪声 $\epsilon_\theta(x_t, t) \approx \epsilon$
- FM：预测速度 $v_\theta(x_t, t) \approx x_1 - x_0 = \text{数据} - \text{噪声}$

两者可以相互转换。在VP调度下，$v = \sqrt{\bar{\alpha}_t}\epsilon - \sqrt{1-\bar{\alpha}_t}x_0$（即 $v$-prediction）。

**(c) 采样方式**：
- DDPM：从 $x_T$（噪声）逆向积分ODE/SDE到 $x_0$（数据），$t: T \to 0$
- FM：从 $x_0$（噪声）正向积分ODE到 $x_1$（数据），$t: 0 \to 1$

**(d) 时间方向约定**：
- DDPM：$t=0$ 是数据，$t=T$ 是噪声
- FM（标准约定）：$t=0$ 是噪声，$t=1$ 是数据

### 练习2

**边界条件 $f(x_\epsilon, \epsilon) = x_\epsilon$ 的必要性**：

当 $t = \epsilon \approx 0$ 时，$x_\epsilon$ 已经非常接近干净数据 $x_0$（几乎没有噪声）。此时一致性函数应该返回这个几乎干净的输入本身，因为不需要任何去噪。

数学上，ODE轨迹在 $t = \epsilon$ 的终点就是 $x_\epsilon$ 自己。一致性条件要求同一轨迹上的所有点映射到同一个终点，因此 $f(x_\epsilon, \epsilon) = x_\epsilon$。

**如果违反**：
1. 模型在接近干净数据时仍然会"修改"它，引入不必要的变形
2. 多步采样中的最后一步会产生偏差，无法收敛到正确的数据
3. 一步生成的输出会有系统性偏移

实现上通过 $c_{skip}(\epsilon) = 1, c_{out}(\epsilon) = 0$ 的参数化来硬编码保证这个条件。

### 练习3

余弦插值路径的Flow Matching：

**(a) 条件速度场推导**：

$$\phi_t = \cos(\pi t/2) \cdot x_0 + \sin(\pi t/2) \cdot x_1$$

$$u_t = \frac{d\phi_t}{dt} = -\frac{\pi}{2}\sin(\pi t/2) \cdot x_0 + \frac{\pi}{2}\cos(\pi t/2) \cdot x_1$$

**(b) 修改后的训练代码**：

```python
class FlowMatchingCosine(FlowMatchingTrainer):
    def compute_loss(self, x_1: torch.Tensor) -> torch.Tensor:
        B = x_1.shape[0]
        x_0 = torch.randn_like(x_1)
        t = torch.rand(B, 1, device=self.device)
        
        # 余弦插值
        cos_t = torch.cos(torch.pi * t / 2)
        sin_t = torch.sin(torch.pi * t / 2)
        x_t = cos_t * x_0 + sin_t * x_1
        
        # 余弦速度场
        target_v = (-torch.pi / 2) * torch.sin(torch.pi * t / 2) * x_0 \
                 + (torch.pi / 2) * torch.cos(torch.pi * t / 2) * x_1
        
        pred_v = self.model(x_t, t)
        return F.mse_loss(pred_v, target_v)
```

**(c) 实验对比**：
- 线性路径：速度恒定，轨迹最直
- 余弦路径：起止阶段速度较慢，中间较快（类似ease-in-out）
- 预期结果：线性路径在少步采样时更好（轨迹更直），余弦路径在多步采样时可能略好（在噪声/数据端分配更多注意力）

### 练习4

一致性蒸馏实现：

```python
class ConsistencyDistillation:
    """使用预训练FM模型作为教师的一致性蒸馏"""
    
    def __init__(
        self,
        student: ConsistencyModel,
        teacher: VelocityNetwork,
        lr: float = 1e-4,
        ema_rate: float = 0.999,
        num_timesteps: int = 150,
        device: str = "cpu",
    ):
        self.student = student.to(device)
        self.teacher = teacher.to(device).eval()
        self.device = device
        self.optimizer = torch.optim.Adam(student.parameters(), lr=lr)
        
        # EMA目标
        self.ema_student = ConsistencyModel(data_dim=2, hidden_dim=256).to(device)
        self.ema_student.load_state_dict(student.state_dict())
        self.ema_rate = ema_rate
        
        self.sigmas = torch.linspace(0.002, 80.0, num_timesteps, device=device)
    
    @torch.no_grad()
    def teacher_ode_step(self, x_t, t_from, t_to):
        """教师模型ODE一步（Euler）"""
        dt = t_to - t_from  # [B, 1]
        v = self.teacher(x_t, t_from)
        return x_t + dt * v
    
    def train_step(self, x_0):
        B = x_0.shape[0]
        n = torch.randint(1, len(self.sigmas), (B,), device=self.device)
        
        t_n = self.sigmas[n][:, None]
        t_n1 = self.sigmas[n - 1][:, None]
        
        noise = torch.randn_like(x_0)
        x_tn = x_0 + t_n * noise
        
        # 教师ODE一步：从t_n到t_{n-1}
        with torch.no_grad():
            # FM教师使用归一化时间[0,1]
            t_from_norm = t_n / 80.0   # 归一化
            t_to_norm = t_n1 / 80.0
            x_tn1_teacher = self.teacher_ode_step(x_tn, t_from_norm, t_to_norm)
        
        # 学生预测
        pred_online = self.student(x_tn, t_n)
        with torch.no_grad():
            pred_target = self.ema_student(x_tn1_teacher, t_n1)
        
        loss = F.mse_loss(pred_online, pred_target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新EMA
        for p, p_ema in zip(self.student.parameters(), self.ema_student.parameters()):
            p_ema.data.mul_(self.ema_rate).add_(p.data, alpha=1 - self.ema_rate)
        
        return loss.item()
```

CD vs CT的差异：CD通常收敛更快、质量更好（因为有教师的精确引导），但需要预训练模型；CT更简单但可能需要更多训练步。

### 练习5

Reflow验证实验：

**(a) 轨迹直度指标**：

$$\text{Straightness} = \frac{\|x_1 - x_0\|^2}{\int_0^1 \|v_\theta(x_t, t)\|^2 dt}$$

对于完全笔直的轨迹（匀速直线运动），此值为1。轨迹越弯，值越小。

近似计算：$\text{Straightness} \approx \frac{\|x_1 - x_0\|^2}{\sum_{k=0}^{K-1} \Delta t \cdot \|v_\theta(x_{t_k}, t_k)\|^2}$

**(b) Reflow实现**：

```python
def reflow(model, sampler, data, n_pairs=5000, n_train=10000, device="cpu"):
    """一轮Reflow操作"""
    # 生成配对：噪声 -> 模型生成的数据
    x_0_noise = torch.randn(n_pairs, 2, device=device)
    t_all = torch.zeros(n_pairs, 1, device=device)
    
    # 从噪声出发，用当前模型生成数据
    with torch.no_grad():
        x_1_gen = sampler.euler_sample(n_pairs, num_steps=100)
    
    # 用新配对重新训练
    new_model = VelocityNetwork(data_dim=2).to(device)
    opt = torch.optim.Adam(new_model.parameters(), lr=1e-3)
    
    for step in range(n_train):
        idx = torch.randint(0, n_pairs, (256,))
        x0_batch = x_0_noise[idx]
        x1_batch = x_1_gen[idx]
        t = torch.rand(256, 1, device=device)
        
        x_t = (1 - t) * x0_batch + t * x1_batch
        target = x1_batch - x0_batch
        pred = new_model(x_t, t)
        
        loss = F.mse_loss(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    
    return new_model
```

**(c)-(d) 预期结果**：
- 每轮Reflow后，Straightness指标应该增加（接近1.0）
- 轨迹可视化从弯曲逐渐变直
- 少步采样质量逐轮改善，尤其是1-2步采样的提升最显著
- 多步采样（50+步）的质量差异较小（因为步数够多时弯曲轨迹也能被精确积分）

---

## 延伸阅读

1. Lipman, Y., et al. (2022). "Flow Matching for Generative Modeling." *ICLR 2023*.
2. Liu, X., Gong, C., & Liu, Q. (2022). "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow." *ICLR 2023*. (Rectified Flow)
3. Song, Y., et al. (2023). "Consistency Models." *ICML 2023*.
4. Esser, P., et al. (2024). "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis." *ICML 2024*. (SD3)
5. Song, Y. & Dhariwal, P. (2023). "Improved Techniques for Training Consistency Models." *arXiv:2310.14189*.
6. Tong, A., et al. (2023). "Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport." *TMLR 2024*. (OT-CFM)
7. Albergo, M. & Vanden-Eijnden, E. (2022). "Building Normalizing Flows with Stochastic Interpolants." *ICLR 2023*.

---

<div align="center">

[⬅️ 第二十一章：DiT：扩散Transformer](21-dit-diffusion-transformer.md) | [📖 目录](../README.md) | [第二十三章：扩散模型的加速与部署 ➡️](../part8-applications/23-acceleration-deployment.md)

</div>
