# 第七章：正向扩散过程与加噪

> **本章导读**：扩散模型的核心思想惊人地简单——如果我们能系统地将数据变成噪声，那么学会"倒放"这个过程就能从噪声中生成数据。本章将严格推导正向扩散过程的每一个数学细节，从马尔科夫链的定义出发，推导出闭合形式的加噪公式 $q(x_t|x_0)$，并进一步推导逆向过程所需的后验分布 $q(x_{t-1}|x_t, x_0)$。这些推导是理解DDPM及其后续所有扩散模型的数学基石。

**前置知识**: 高斯分布与重参数化技巧（第四章），马尔科夫链基础（第五章），KL散度与变分推断（第六章）

**预计学习时间**: 3-4小时

## 学习目标

1. 理解扩散模型的物理直觉，建立"加噪-去噪"的核心思维框架
2. 掌握正向过程的马尔科夫链定义，理解噪声调度 $\{\beta_t\}$ 的设计原则
3. 独立推导闭合形式 $q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$
4. 完整推导后验分布 $q(x_{t-1}|x_t, x_0)$ 的均值和方差
5. 从信息论角度理解扩散过程中信息的不可逆损失

---

## 7.1 扩散模型的直觉

### 7.1.1 物理扩散的启发

想象在一杯清水中滴入一滴墨水。墨水最初集中在一个点（高度有结构的分布），随着时间推移，墨水分子在热运动的驱动下逐渐扩散，最终均匀分布在整杯水中（无结构的均匀分布）。这个过程有几个关键特征：

- **不可逆性**：热力学第二定律告诉我们，孤立系统的熵只增不减。墨水自发扩散后不会自发聚集回来
- **局部性**：每一步扩散只涉及微小的随机位移，是一个马尔科夫过程
- **终态确定**：无论初始分布如何，充分长时间后系统总会达到热平衡（最大熵态）

扩散概率模型（Diffusion Probabilistic Models）将这个物理直觉数学化：

$$\text{结构化数据} \xrightarrow{\text{逐步加噪}} \text{纯高斯噪声}$$

如果我们能学会"逆转"这个过程——从噪声中逐步恢复结构——就获得了一个强大的生成模型。

### 7.1.2 信息被噪声"销毁"的过程

从信息论的视角，正向扩散过程可以理解为信息的逐步销毁。设 $x_0$ 是一张自然图像，它包含丰富的语义信息（物体、纹理、颜色等）。在每一步加噪中：

- **低频信息先被保留**：前几步加噪主要破坏高频细节（边缘、纹理）
- **全局结构后被消除**：随着噪声累积，连物体的大致轮廓和颜色分布也被淹没
- **最终只剩噪声**：当 $t$ 足够大时，$x_t$ 与 $x_0$ 几乎独立，所有原始信息都已丢失

这个"信息销毁"的时间线对模型设计有深刻影响——逆向过程需要先恢复全局结构，再逐步填充细节，这与人类绘画的过程不谋而合。

### 7.1.3 逆向过程：学习的目标

扩散模型的训练目标是学习一个参数化的逆向过程 $p_\theta(x_{t-1}|x_t)$，使其尽可能接近真实的逆向后验 $q(x_{t-1}|x_t)$。关键的数学洞见在于：

- 正向过程 $q(x_t|x_{t-1})$ 是人为设计的，完全已知
- 逆向后验 $q(x_{t-1}|x_t)$ 需要知道整个数据分布 $q(x_0)$，无法直接计算
- 但条件后验 $q(x_{t-1}|x_t, x_0)$ 是可以解析计算的（本章7.5节将推导）
- 这个条件后验为训练提供了监督信号

---

## 7.2 正向过程的数学定义

### 7.2.1 马尔科夫链定义

正向扩散过程定义为一个离散时间马尔科夫链，从数据分布 $x_0 \sim q(x_0)$ 出发，经过 $T$ 步逐渐加噪：

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t;\, \sqrt{1 - \beta_t}\, x_{t-1},\, \beta_t I)$$

其中 $\beta_t \in (0, 1)$ 是第 $t$ 步的噪声强度（noise schedule），$t = 1, 2, \ldots, T$。

展开这个高斯分布的含义：

$$x_t = \sqrt{1 - \beta_t}\, x_{t-1} + \sqrt{\beta_t}\, \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I)$$

每一步做了两件事：
1. **缩放**：将 $x_{t-1}$ 乘以 $\sqrt{1 - \beta_t} < 1$，使信号幅度略微缩小
2. **加噪**：添加方差为 $\beta_t$ 的独立高斯噪声

> **为什么要缩放？** 如果只是简单地叠加噪声 $x_t = x_{t-1} + \sqrt{\beta_t}\epsilon_t$，那么 $x_t$ 的方差会不断增长。通过 $\sqrt{1-\beta_t}$ 的缩放，可以保证如果 $x_{t-1}$ 的方差为 $1$，那么 $x_t$ 的方差也近似为 $1$：
> $$\text{Var}(x_t) = (1-\beta_t)\text{Var}(x_{t-1}) + \beta_t = \text{Var}(x_{t-1})$$
> 这种方差保持性质对数值稳定性至关重要。

### 7.2.2 噪声调度 $\{\beta_t\}_{t=1}^T$

噪声调度决定了正向过程"销毁"信息的速率。典型设置：

**线性调度**（Ho et al., 2020 原始设定）：

$$\beta_t = \beta_{\min} + \frac{t-1}{T-1}(\beta_{\max} - \beta_{\min})$$

其中 $\beta_1 = \beta_{\min} = 10^{-4}$，$\beta_T = \beta_{\max} = 0.02$，$T = 1000$。

设计考量：
- $\beta_t$ 值很小（$10^{-4}$ 到 $0.02$），确保每一步的变化是微小的
- 这使得逆向转移分布 $q(x_{t-1}|x_t)$ 也近似为高斯分布（当步长足够小时，参见Feller 1949）
- $T = 1000$ 步看似很多，但正是这种"微小步长、大量迭代"的策略使得高斯近似成立

**余弦调度**（Nichol & Dhariwal, 2021）：

$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2$$

其中 $s = 0.008$ 是一个小偏移量，防止 $\beta_t$ 在 $t$ 接近 $0$ 时过小。然后从 $\bar{\alpha}_t$ 反推：

$$\beta_t = 1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}$$

余弦调度的优势是使 $\bar{\alpha}_t$ 的衰减更平滑，避免线性调度在中间时间步噪声增长过快的问题。

### 7.2.3 联合分布

整个正向过程的联合分布由马尔科夫性质给出：

$$q(x_{1:T} | x_0) = \prod_{t=1}^{T} q(x_t | x_{t-1})$$

这意味着给定 $x_0$，整条轨迹 $x_1, x_2, \ldots, x_T$ 的联合分布完全由转移核 $\{q(x_t|x_{t-1})\}_{t=1}^T$ 确定。

---

## 7.3 闭合形式推导（关键）

### 7.3.1 核心符号定义

为了简化后续推导，定义：

$$\alpha_t = 1 - \beta_t$$

$$\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s = \prod_{s=1}^{t}(1 - \beta_s)$$

由于每个 $\alpha_s \in (0, 1)$，$\bar{\alpha}_t$ 随 $t$ 单调递减，且 $\bar{\alpha}_0 = 1$。当 $T$ 足够大时，$\bar{\alpha}_T \approx 0$。

直觉解读：
- $\alpha_t$ 衡量第 $t$ 步保留了多少"信号"
- $\bar{\alpha}_t$ 衡量从第 $0$ 步到第 $t$ 步累积保留了多少"信号"
- $1 - \bar{\alpha}_t$ 衡量累积添加了多少"噪声"

### 7.3.2 完整推导

**定理**：正向过程的边际分布具有闭合形式：

$$q(x_t | x_0) = \mathcal{N}(x_t;\, \sqrt{\bar{\alpha}_t}\, x_0,\, (1 - \bar{\alpha}_t) I)$$

**证明**：使用数学归纳法。

**基础情形**（$t = 1$）：

$$q(x_1 | x_0) = \mathcal{N}(x_1;\, \sqrt{1-\beta_1}\, x_0,\, \beta_1 I) = \mathcal{N}(x_1;\, \sqrt{\alpha_1}\, x_0,\, (1-\alpha_1) I)$$

由于 $\bar{\alpha}_1 = \alpha_1$，这与 $\mathcal{N}(x_1; \sqrt{\bar{\alpha}_1}x_0, (1-\bar{\alpha}_1)I)$ 一致。

**归纳步骤**：假设对 $t-1$ 成立，即：

$$q(x_{t-1} | x_0) = \mathcal{N}(x_{t-1};\, \sqrt{\bar{\alpha}_{t-1}}\, x_0,\, (1 - \bar{\alpha}_{t-1}) I)$$

利用重参数化，写出：

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\, x_0 + \sqrt{1 - \bar{\alpha}_{t-1}}\, \epsilon_1, \quad \epsilon_1 \sim \mathcal{N}(0, I)$$

再由正向转移核：

$$x_t = \sqrt{\alpha_t}\, x_{t-1} + \sqrt{1 - \alpha_t}\, \epsilon_2, \quad \epsilon_2 \sim \mathcal{N}(0, I)$$

其中 $\epsilon_1$ 与 $\epsilon_2$ 独立。将第一式代入第二式：

$$x_t = \sqrt{\alpha_t}\left(\sqrt{\bar{\alpha}_{t-1}}\, x_0 + \sqrt{1 - \bar{\alpha}_{t-1}}\, \epsilon_1\right) + \sqrt{1 - \alpha_t}\, \epsilon_2$$

$$= \sqrt{\alpha_t \bar{\alpha}_{t-1}}\, x_0 + \sqrt{\alpha_t(1 - \bar{\alpha}_{t-1})}\, \epsilon_1 + \sqrt{1 - \alpha_t}\, \epsilon_2$$

由于 $\epsilon_1, \epsilon_2$ 都是标准正态且独立，它们的线性组合仍为正态分布。方差为：

$$\alpha_t(1 - \bar{\alpha}_{t-1}) + (1 - \alpha_t) = \alpha_t - \alpha_t\bar{\alpha}_{t-1} + 1 - \alpha_t = 1 - \alpha_t\bar{\alpha}_{t-1} = 1 - \bar{\alpha}_t$$

因此：

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

即 $q(x_t | x_0) = \mathcal{N}(x_t;\, \sqrt{\bar{\alpha}_t}\, x_0,\, (1 - \bar{\alpha}_t) I)$。$\square$

### 7.3.3 重参数化形式

闭合形式最常用的表达方式是重参数化形式：

$$\boxed{x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)}$$

这个公式极其重要，它意味着：

1. **跳步采样**：给定 $x_0$，可以直接计算任意时刻 $t$ 的 $x_t$，无需逐步迭代
2. **高效训练**：训练时只需随机采样 $t$，用上式计算 $x_t$，而不需要模拟整条马尔科夫链
3. **信号-噪声分解**：$x_t$ 是原始信号 $x_0$ 和纯噪声 $\epsilon$ 的加权和，权重分别为 $\sqrt{\bar{\alpha}_t}$ 和 $\sqrt{1-\bar{\alpha}_t}$

注意权重平方和：$\bar{\alpha}_t + (1 - \bar{\alpha}_t) = 1$，这保证了如果 $x_0$ 和 $\epsilon$ 都是标准正态的，$x_t$ 的方差也为 $1$。

### 7.3.4 闭合形式的直觉理解

将闭合形式用信噪比来理解：

| 时间步 $t$ | $\bar{\alpha}_t$ | 信号强度 $\sqrt{\bar{\alpha}_t}$ | 噪声强度 $\sqrt{1-\bar{\alpha}_t}$ | 物理含义 |
|:---:|:---:|:---:|:---:|:---|
| $t = 0$ | $1$ | $1$ | $0$ | 纯净信号 |
| $t$ 较小 | $\approx 1$ | $\approx 1$ | $\approx 0$ | 几乎看不出噪声 |
| $t = T/2$ | $\approx 0.5$ | $\approx 0.71$ | $\approx 0.71$ | 信号和噪声旗鼓相当 |
| $t$ 较大 | $\approx 0$ | $\approx 0$ | $\approx 1$ | 几乎只有噪声 |
| $t = T$ | $\approx 0$ | $\approx 0$ | $\approx 1$ | 纯高斯噪声 |

---

## 7.4 加噪过程的可视化分析

### 7.4.1 不同时间步的图像变化

考虑一张 $64 \times 64$ 的自然图像 $x_0$，在 $T = 1000$ 步的线性调度下：

- $t = 0$（原图）：所有细节清晰可见
- $t = 50$：高频纹理开始模糊，如发丝、布料纹理
- $t = 200$：边缘变得模糊，但主要物体仍可辨认
- $t = 500$：只能看到模糊的色块和大致轮廓
- $t = 800$：几乎看不出原始内容，只有微弱的亮暗区域
- $t = 1000$：视觉上与纯高斯噪声无异

### 7.4.2 $\bar{\alpha}_t$ 的衰减曲线

对于线性调度 $\beta_t$ 从 $10^{-4}$ 到 $0.02$：

$$\bar{\alpha}_t = \prod_{s=1}^{t}(1 - \beta_s)$$

由于 $\ln\bar{\alpha}_t = \sum_{s=1}^{t}\ln(1-\beta_s) \approx -\sum_{s=1}^{t}\beta_s$（当 $\beta_s$ 较小时），线性调度下 $\bar{\alpha}_t$ 近似为：

$$\bar{\alpha}_t \approx \exp\left(-\sum_{s=1}^{t}\beta_s\right) \approx \exp\left(-t\bar{\beta}\right)$$

其中 $\bar{\beta} = \frac{1}{t}\sum_{s=1}^{t}\beta_s$ 是平均噪声强度。这表明 $\bar{\alpha}_t$ 近似呈指数衰减。

### 7.4.3 信噪比（SNR）分析

信噪比定义为信号功率与噪声功率之比：

$$\text{SNR}(t) = \frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t}$$

取对数形式更常用：

$$\log\text{SNR}(t) = \log\bar{\alpha}_t - \log(1 - \bar{\alpha}_t)$$

SNR 的性质：
- $t = 0$ 时 $\text{SNR} = +\infty$（纯信号）
- $t = T$ 时 $\text{SNR} \approx 0$（纯噪声）
- SNR 单调递减

在扩散模型的理论分析中，许多重要结论可以用 SNR 统一表达，例如损失权重、最优方差等。

### 7.4.4 线性调度 vs 余弦调度对比

| 特性 | 线性调度 | 余弦调度 |
|:---|:---|:---|
| $\bar{\alpha}_t$ 衰减 | 近似指数，中间段快 | 近似 $\cos^2$ 型，更平滑 |
| 信息损失分布 | 集中在前 $\sim 200$ 步 | 更均匀分布 |
| 低 SNR 区域 | 较长的"纯噪声"阶段 | 更短 |
| 生成质量 | 对于低分辨率图像足够 | 对高分辨率图像更好 |
| 训练效率 | 大量步骤的梯度贡献小 | 各步贡献更均衡 |

余弦调度的核心优势在于：线性调度下 $\bar{\alpha}_t$ 在中间时间步衰减过快，导致信息在不到一半的时间步内就几乎完全丢失，后续步骤的训练信号很弱。余弦调度通过更平缓的衰减曲线解决了这个问题。

---

## 7.5 后验分布的推导

### 7.5.1 条件后验的重要性

后验分布 $q(x_{t-1}|x_t, x_0)$ 描述了"已知起点 $x_0$ 和当前状态 $x_t$，前一步 $x_{t-1}$ 的分布"。这个分布在DDPM的训练目标推导中扮演核心角色——它是逆向过程 $p_\theta(x_{t-1}|x_t)$ 需要拟合的目标。

### 7.5.2 贝叶斯定理展开

利用贝叶斯定理：

$$q(x_{t-1} | x_t, x_0) = \frac{q(x_t | x_{t-1}, x_0) \cdot q(x_{t-1} | x_0)}{q(x_t | x_0)}$$

由马尔科夫性质，$q(x_t | x_{t-1}, x_0) = q(x_t | x_{t-1})$，因此：

$$q(x_{t-1} | x_t, x_0) = \frac{q(x_t | x_{t-1}) \cdot q(x_{t-1} | x_0)}{q(x_t | x_0)}$$

三个分布都是已知的高斯分布：

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t;\, \sqrt{\alpha_t}\, x_{t-1},\, (1-\alpha_t)I)$$

$$q(x_{t-1} | x_0) = \mathcal{N}(x_{t-1};\, \sqrt{\bar{\alpha}_{t-1}}\, x_0,\, (1-\bar{\alpha}_{t-1})I)$$

$$q(x_t | x_0) = \mathcal{N}(x_t;\, \sqrt{\bar{\alpha}_t}\, x_0,\, (1-\bar{\alpha}_t)I)$$

### 7.5.3 完整推导

取对数并只关注与 $x_{t-1}$ 相关的项（因为我们要求关于 $x_{t-1}$ 的分布）：

$$\log q(x_{t-1} | x_t, x_0) = \log q(x_t | x_{t-1}) + \log q(x_{t-1} | x_0) - \log q(x_t | x_0)$$

逐项展开（省略常数项，用 $\propto$ 表示）：

**第一项** $\log q(x_t | x_{t-1})$：

$$\propto -\frac{1}{2(1-\alpha_t)} \|x_t - \sqrt{\alpha_t} x_{t-1}\|^2$$

$$= -\frac{1}{2(1-\alpha_t)}\left(\|x_t\|^2 - 2\sqrt{\alpha_t} x_t^\top x_{t-1} + \alpha_t \|x_{t-1}\|^2\right)$$

**第二项** $\log q(x_{t-1} | x_0)$：

$$\propto -\frac{1}{2(1-\bar{\alpha}_{t-1})} \|x_{t-1} - \sqrt{\bar{\alpha}_{t-1}} x_0\|^2$$

$$= -\frac{1}{2(1-\bar{\alpha}_{t-1})}\left(\|x_{t-1}\|^2 - 2\sqrt{\bar{\alpha}_{t-1}} x_0^\top x_{t-1} + \bar{\alpha}_{t-1}\|x_0\|^2\right)$$

**第三项** $\log q(x_t | x_0)$ 不含 $x_{t-1}$，视为常数。

合并含 $x_{t-1}$ 的项。先提取 $\|x_{t-1}\|^2$ 的系数（确定方差）：

$$-\frac{1}{2}\left(\frac{\alpha_t}{1-\alpha_t} + \frac{1}{1-\bar{\alpha}_{t-1}}\right)\|x_{t-1}\|^2$$

系数为：

$$\frac{\alpha_t}{1-\alpha_t} + \frac{1}{1-\bar{\alpha}_{t-1}} = \frac{\alpha_t(1-\bar{\alpha}_{t-1}) + (1-\alpha_t)}{(1-\alpha_t)(1-\bar{\alpha}_{t-1})}$$

分子展开：

$$\alpha_t - \alpha_t\bar{\alpha}_{t-1} + 1 - \alpha_t = 1 - \alpha_t\bar{\alpha}_{t-1} = 1 - \bar{\alpha}_t$$

因此精度（方差的倒数）为：

$$\frac{1}{\tilde{\beta}_t} = \frac{1 - \bar{\alpha}_t}{(1-\alpha_t)(1-\bar{\alpha}_{t-1})}$$

**后验方差**：

$$\boxed{\tilde{\beta}_t = \frac{(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \cdot \beta_t = \frac{(1-\bar{\alpha}_{t-1})(1-\alpha_t)}{1 - \bar{\alpha}_t}}$$

接下来提取 $x_{t-1}$ 的一次项系数（确定均值）：

$$\frac{\sqrt{\alpha_t}}{1-\alpha_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}} x_0$$

后验均值 = 一次项系数 $\times$ 后验方差：

$$\tilde{\mu}_t(x_t, x_0) = \tilde{\beta}_t \left(\frac{\sqrt{\alpha_t}}{1-\alpha_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}} x_0\right)$$

代入 $\tilde{\beta}_t$ 并化简：

$$\tilde{\mu}_t(x_t, x_0) = \frac{(1-\bar{\alpha}_{t-1})\beta_t}{1-\bar{\alpha}_t} \cdot \frac{\sqrt{\alpha_t}}{\beta_t} x_t + \frac{(1-\bar{\alpha}_{t-1})\beta_t}{1-\bar{\alpha}_t} \cdot \frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}} x_0$$

$$= \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1-\bar{\alpha}_t} x_0$$

**后验均值**：

$$\boxed{\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1 - \bar{\alpha}_t} x_0}$$

### 7.5.4 完整后验分布

综合以上推导，条件后验为：

$$\boxed{q(x_{t-1} | x_t, x_0) = \mathcal{N}\left(x_{t-1};\, \tilde{\mu}_t(x_t, x_0),\, \tilde{\beta}_t I\right)}$$

其中：

$$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1 - \bar{\alpha}_t} x_0$$

$$\tilde{\beta}_t = \frac{(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \cdot \beta_t$$

### 7.5.5 后验均值的另一种形式

利用 $x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon)$ 代入后验均值公式，可以得到用 $(x_t, \epsilon)$ 表达的形式：

$$\tilde{\mu}_t = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\epsilon\right)$$

这个形式在DDPM的采样算法中极为关键——如果我们能从 $x_t$ 预测出噪声 $\epsilon$，就能计算出后验均值，从而进行采样。

---

## 7.6 信息论视角

### 7.6.1 扩散过程的微分熵变化

对于连续随机变量 $x_t$，微分熵为：

$$h(x_t) = -\int q(x_t) \log q(x_t)\, dx_t$$

在正向扩散过程中，$h(x_t)$ 单调递增。直觉上，加噪使分布更"扩散"，不确定性增大。

当 $t \to T$，$q(x_T) \to \mathcal{N}(0, I)$，此时微分熵达到最大值（在给定方差约束下，高斯分布的微分熵最大）：

$$h(x_T) \approx \frac{d}{2}\ln(2\pi e)$$

其中 $d$ 是数据维度。

### 7.6.2 数据处理不等式

数据处理不等式（Data Processing Inequality）指出，对于马尔科夫链 $x_0 \to x_1 \to \cdots \to x_T$：

$$I(x_0; x_T) \leq I(x_0; x_{T-1}) \leq \cdots \leq I(x_0; x_1) \leq H(x_0)$$

其中 $I(x_0; x_t)$ 是互信息，$H(x_0)$ 是 $x_0$ 的熵。这说明：

- 信息只能在正向过程中丢失，不能增加
- 每一步 $q(x_t|x_{t-1})$ 都可能损失信息
- 这正是逆向过程需要"学习"的：恢复这些丢失的信息

### 7.6.3 互信息 $I(x_0; x_t)$ 的衰减

利用闭合形式 $q(x_t|x_0)$，可以计算（对于高斯情况）：

$$I(x_0; x_t) = h(x_t) - h(x_t | x_0) = h(x_t) - \frac{d}{2}\ln(2\pi e(1-\bar{\alpha}_t))$$

随着 $t$ 增大，$\bar{\alpha}_t \to 0$，条件熵 $h(x_t|x_0) \to \frac{d}{2}\ln(2\pi e)$，而 $h(x_t)$ 也趋向同一值，因此 $I(x_0; x_t) \to 0$。

这从信息论角度严格证明了：正向扩散过程确实在逐步销毁 $x_0$ 中的所有信息。

### 7.6.4 与变分下界的联系

DDPM的训练目标（ELBO）可以用互信息分解来理解。负的ELBO上界可以写为：

$$-\text{ELBO} = D_{KL}(q(x_T|x_0) \| p(x_T)) + \sum_{t=2}^{T} D_{KL}(q(x_{t-1}|x_t,x_0) \| p_\theta(x_{t-1}|x_t)) - \mathbb{E}[\log p_\theta(x_0|x_1)]$$

每一项 $D_{KL}(q(x_{t-1}|x_t,x_0) \| p_\theta(x_{t-1}|x_t))$ 度量的是逆向模型在第 $t$ 步恢复信息的能力。如果所有 KL 项为零，模型完美恢复了所有丢失的信息。

---

## 代码实战

```python
"""
第七章代码实战：正向扩散过程的完整实现与可视化
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from dataclasses import dataclass


# ============================================================
# 1. 噪声调度器
# ============================================================

class LinearNoiseScheduler:
    """线性噪声调度器
    
    beta_t 从 beta_start 线性增长到 beta_end。
    
    参考: Ho et al. "Denoising Diffusion Probabilistic Models" (2020)
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str = "cpu",
    ):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # 线性插值生成 beta 序列
        self.betas = torch.linspace(
            beta_start, beta_end, num_timesteps, device=device
        )  # (T,)
        
        # 计算 alpha 和 alpha_bar
        self.alphas = 1.0 - self.betas  # (T,)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)  # (T,)
        
        # 计算常用的派生量
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)  # (T,)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)  # (T,)
        
        # 信噪比
        self.snr = self.alpha_bars / (1.0 - self.alpha_bars)  # (T,)
        
        # 后验方差（用于第八章的逆向过程）
        alpha_bars_prev = torch.cat(
            [torch.tensor([1.0], device=device), self.alpha_bars[:-1]]
        )  # (T,), alpha_bar_{t-1}，第0个元素为1
        self.posterior_variance = (
            (1.0 - alpha_bars_prev) / (1.0 - self.alpha_bars) * self.betas
        )  # (T,)
        
        # 后验均值系数
        self.posterior_mean_coef1 = (
            torch.sqrt(alpha_bars_prev) * self.betas / (1.0 - self.alpha_bars)
        )  # (T,) — x_0 的系数
        self.posterior_mean_coef2 = (
            torch.sqrt(self.alphas) * (1.0 - alpha_bars_prev) / (1.0 - self.alpha_bars)
        )  # (T,) — x_t 的系数


class CosineNoiseScheduler:
    """余弦噪声调度器
    
    alpha_bar_t 按余弦函数衰减，更平滑。
    
    参考: Nichol & Dhariwal "Improved Denoising Diffusion Probabilistic Models" (2021)
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        s: float = 0.008,
        max_beta: float = 0.999,
        device: str = "cpu",
    ):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # 计算 f(t) = cos^2((t/T + s) / (1 + s) * pi/2)
        steps = torch.arange(num_timesteps + 1, dtype=torch.float64, device=device)
        f = torch.cos(((steps / num_timesteps) + s) / (1 + s) * (np.pi / 2)) ** 2
        
        # alpha_bar_t = f(t) / f(0)
        alpha_bars = f / f[0]
        
        # 从 alpha_bar 反推 beta
        betas = 1.0 - (alpha_bars[1:] / alpha_bars[:-1])
        betas = torch.clamp(betas, min=0.0, max=max_beta)  # 防止数值问题
        
        self.betas = betas.float()  # (T,)
        self.alphas = 1.0 - self.betas  # (T,)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)  # (T,)
        
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)  # (T,)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)  # (T,)
        
        self.snr = self.alpha_bars / (1.0 - self.alpha_bars)  # (T,)
        
        alpha_bars_prev = torch.cat(
            [torch.tensor([1.0], device=device).float(), self.alpha_bars[:-1]]
        )
        self.posterior_variance = (
            (1.0 - alpha_bars_prev) / (1.0 - self.alpha_bars) * self.betas
        )
        self.posterior_mean_coef1 = (
            torch.sqrt(alpha_bars_prev) * self.betas / (1.0 - self.alpha_bars)
        )
        self.posterior_mean_coef2 = (
            torch.sqrt(self.alphas) * (1.0 - alpha_bars_prev) / (1.0 - self.alpha_bars)
        )


# ============================================================
# 2. 正向扩散过程
# ============================================================

class ForwardDiffusion:
    """正向扩散过程
    
    使用闭合形式直接采样 q(x_t | x_0)。
    """
    
    def __init__(self, scheduler: LinearNoiseScheduler | CosineNoiseScheduler):
        self.scheduler = scheduler
    
    def add_noise(
        self,
        x_0: torch.Tensor,       # (B, C, H, W) 或 (B, D)
        t: torch.Tensor,          # (B,) 时间步索引，范围 [0, T-1]
        noise: Optional[torch.Tensor] = None,  # (B, C, H, W) 或 (B, D)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """使用闭合形式 q(x_t|x_0) 加噪
        
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        
        Returns:
            x_t: 加噪后的样本
            noise: 使用的噪声（用于训练时计算损失）
        """
        if noise is None:
            noise = torch.randn_like(x_0)  # (B, C, H, W)
        
        # 取出对应时间步的系数，并调整形状以便广播
        sqrt_alpha_bar = self.scheduler.sqrt_alpha_bars[t]  # (B,)
        sqrt_one_minus_alpha_bar = self.scheduler.sqrt_one_minus_alpha_bars[t]  # (B,)
        
        # 调整维度以便与 x_0 广播
        while sqrt_alpha_bar.dim() < x_0.dim():
            sqrt_alpha_bar = sqrt_alpha_bar.unsqueeze(-1)
            sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.unsqueeze(-1)
        
        # 闭合形式加噪
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise  # (B, C, H, W)
        
        return x_t, noise
    
    def sample_q_posterior(
        self,
        x_t: torch.Tensor,   # (B, C, H, W)
        x_0: torch.Tensor,   # (B, C, H, W)
        t: torch.Tensor,     # (B,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """采样后验分布 q(x_{t-1} | x_t, x_0)
        
        Returns:
            posterior_mean: 后验均值
            posterior_variance: 后验方差
        """
        coef1 = self.scheduler.posterior_mean_coef1[t]  # (B,)
        coef2 = self.scheduler.posterior_mean_coef2[t]  # (B,)
        var = self.scheduler.posterior_variance[t]       # (B,)
        
        while coef1.dim() < x_0.dim():
            coef1 = coef1.unsqueeze(-1)
            coef2 = coef2.unsqueeze(-1)
            var = var.unsqueeze(-1)
        
        posterior_mean = coef1 * x_0 + coef2 * x_t  # (B, C, H, W)
        
        return posterior_mean, var


# ============================================================
# 3. 可视化工具
# ============================================================

def visualize_forward_diffusion(
    x_0: torch.Tensor,                          # (C, H, W) 单张图像
    scheduler: LinearNoiseScheduler,
    timesteps: list[int] = [0, 50, 200, 500, 800, 999],
) -> plt.Figure:
    """可视化不同时间步的加噪结果"""
    diffusion = ForwardDiffusion(scheduler)
    
    fig, axes = plt.subplots(1, len(timesteps), figsize=(3 * len(timesteps), 3))
    
    for i, t_val in enumerate(timesteps):
        if t_val == 0:
            img = x_0
        else:
            t = torch.tensor([t_val])
            x_0_batch = x_0.unsqueeze(0)  # (1, C, H, W)
            x_t, _ = diffusion.add_noise(x_0_batch, t)
            img = x_t.squeeze(0)  # (C, H, W)
        
        # 将张量转换为可显示的格式
        img_np = img.permute(1, 2, 0).clamp(0, 1).numpy()
        if img_np.shape[-1] == 1:
            img_np = img_np.squeeze(-1)
            axes[i].imshow(img_np, cmap="gray")
        else:
            axes[i].imshow(img_np)
        
        alpha_bar = scheduler.alpha_bars[t_val].item() if t_val > 0 else 1.0
        axes[i].set_title(f"t={t_val}\n" + r"$\bar{\alpha}$" + f"={alpha_bar:.3f}")
        axes[i].axis("off")
    
    fig.suptitle("Forward Diffusion Process", fontsize=14)
    plt.tight_layout()
    return fig


def plot_schedule_curves(
    linear_scheduler: LinearNoiseScheduler,
    cosine_scheduler: CosineNoiseScheduler,
) -> plt.Figure:
    """绘制两种调度器的 alpha_bar、beta、SNR 曲线对比"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    T = linear_scheduler.num_timesteps
    t_range = np.arange(T)
    
    # --- alpha_bar 曲线 ---
    axes[0].plot(
        t_range,
        linear_scheduler.alpha_bars.numpy(),
        label="Linear",
        linewidth=2,
    )
    axes[0].plot(
        t_range,
        cosine_scheduler.alpha_bars.numpy(),
        label="Cosine",
        linewidth=2,
        linestyle="--",
    )
    axes[0].set_xlabel("Timestep t")
    axes[0].set_ylabel(r"$\bar{\alpha}_t$")
    axes[0].set_title(r"Cumulative Signal Retention $\bar{\alpha}_t$")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # --- beta 曲线 ---
    axes[1].plot(
        t_range,
        linear_scheduler.betas.numpy(),
        label="Linear",
        linewidth=2,
    )
    axes[1].plot(
        t_range,
        cosine_scheduler.betas.numpy(),
        label="Cosine",
        linewidth=2,
        linestyle="--",
    )
    axes[1].set_xlabel("Timestep t")
    axes[1].set_ylabel(r"$\beta_t$")
    axes[1].set_title(r"Noise Schedule $\beta_t$")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # --- log SNR 曲线 ---
    log_snr_linear = torch.log10(linear_scheduler.snr + 1e-10).numpy()
    log_snr_cosine = torch.log10(cosine_scheduler.snr + 1e-10).numpy()
    
    axes[2].plot(t_range, log_snr_linear, label="Linear", linewidth=2)
    axes[2].plot(
        t_range, log_snr_cosine, label="Cosine", linewidth=2, linestyle="--"
    )
    axes[2].set_xlabel("Timestep t")
    axes[2].set_ylabel(r"$\log_{10}$ SNR")
    axes[2].set_title("Signal-to-Noise Ratio (log scale)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    
    plt.tight_layout()
    return fig


def verify_posterior(
    scheduler: LinearNoiseScheduler,
    num_samples: int = 100000,
    t_val: int = 500,
    dim: int = 2,
) -> None:
    """用蒙特卡洛方法验证后验分布的解析公式"""
    diffusion = ForwardDiffusion(scheduler)
    
    # 固定一个 x_0
    x_0 = torch.randn(1, dim) * 0.5 + 1.0  # (1, D)
    x_0 = x_0.expand(num_samples, -1)  # (N, D)
    
    t = torch.full((num_samples,), t_val, dtype=torch.long)
    t_prev = torch.full((num_samples,), t_val - 1, dtype=torch.long)
    
    # 采样 x_t 和 x_{t-1}
    # 方法1：通过闭合形式独立采样 x_t 和 x_{t-1}，然后筛选（不实际）
    # 方法2：先采样 x_{t-1}，再从 x_{t-1} 采样 x_t
    noise_prev = torch.randn_like(x_0)  # (N, D)
    x_t_minus_1, _ = diffusion.add_noise(x_0, t_prev, noise_prev)  # (N, D)
    
    # 从 q(x_t | x_{t-1}) 采样 x_t
    sqrt_alpha = torch.sqrt(scheduler.alphas[t_val])
    beta = scheduler.betas[t_val]
    noise_step = torch.randn_like(x_0)  # (N, D)
    x_t = sqrt_alpha * x_t_minus_1 + torch.sqrt(beta) * noise_step  # (N, D)
    
    # 计算解析后验
    x_0_single = x_0[0:1]  # (1, D)
    posterior_mean, posterior_var = diffusion.sample_q_posterior(
        x_t[0:1], x_0_single, torch.tensor([t_val])
    )
    
    # 蒙特卡洛估计：给定 x_0 和 x_t，x_{t-1} 的经验统计
    # 我们用联合采样的方法来验证
    print(f"验证后验分布 q(x_{{t-1}} | x_t, x_0) 在 t={t_val}")
    print(f"解析后验均值: {posterior_mean.squeeze().numpy()}")
    print(f"解析后验方差: {posterior_var.squeeze().item():.6f}")
    
    # 通过大量采样估计经验后验
    # 固定 x_0 和 x_t = x_t[0]，采样许多 x_{t-1}
    x_t_fixed = x_t[0:1].expand(num_samples, -1)  # (N, D)
    x_0_fixed = x_0[0:1].expand(num_samples, -1)  # (N, D)
    
    # 从解析后验采样
    post_mean, post_var = diffusion.sample_q_posterior(
        x_t_fixed, x_0_fixed, torch.full((num_samples,), t_val, dtype=torch.long)
    )
    posterior_samples = post_mean + torch.sqrt(post_var) * torch.randn_like(x_0_fixed)
    
    # 验证采样结果的统计量
    empirical_mean = posterior_samples.mean(dim=0)
    empirical_var = posterior_samples.var(dim=0)
    
    print(f"经验采样均值: {empirical_mean.numpy()}")
    print(f"经验采样方差: {empirical_var.mean().item():.6f}")
    print(f"均值误差: {(posterior_mean.squeeze() - empirical_mean).abs().max().item():.6f}")
    print("验证通过!" if (posterior_mean.squeeze() - empirical_mean).abs().max().item() < 0.05 else "验证失败!")


# ============================================================
# 4. 主程序
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    
    # 创建两种调度器
    T = 1000
    linear_sched = LinearNoiseScheduler(num_timesteps=T)
    cosine_sched = CosineNoiseScheduler(num_timesteps=T)
    
    # --- 绘制调度曲线对比 ---
    fig_curves = plot_schedule_curves(linear_sched, cosine_sched)
    plt.savefig("schedule_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("调度曲线已保存为 schedule_comparison.png")
    
    # --- 可视化正向扩散过程 ---
    # 创建一个简单的测试图像（棋盘格）
    img_size = 64
    x = torch.zeros(1, img_size, img_size)
    for i in range(img_size):
        for j in range(img_size):
            if (i // 8 + j // 8) % 2 == 0:
                x[0, i, j] = 1.0
    
    fig_diffusion = visualize_forward_diffusion(x, linear_sched)
    plt.savefig("forward_diffusion.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("正向扩散可视化已保存为 forward_diffusion.png")
    
    # --- 打印关键时间步的统计量 ---
    print("\n=== 关键时间步统计 ===")
    print(f"{'t':>5} | {'alpha_bar':>10} | {'signal':>10} | {'noise':>10} | {'SNR':>10} | {'log SNR':>10}")
    print("-" * 70)
    for t in [0, 50, 100, 200, 500, 800, 999]:
        ab = linear_sched.alpha_bars[t].item()
        sig = np.sqrt(ab)
        noi = np.sqrt(1 - ab)
        snr = ab / (1 - ab) if ab < 1 else float("inf")
        log_snr = np.log10(snr) if snr > 0 and snr != float("inf") else float("inf")
        print(f"{t:>5} | {ab:>10.6f} | {sig:>10.6f} | {noi:>10.6f} | {snr:>10.4f} | {log_snr:>10.4f}")
    
    # --- 验证后验分布 ---
    print("\n=== 后验分布验证 ===")
    verify_posterior(linear_sched, num_samples=100000, t_val=500)
    print()
    verify_posterior(linear_sched, num_samples=100000, t_val=100)
    
    # --- 验证闭合形式的正确性 ---
    print("\n=== 闭合形式验证（逐步 vs 直接） ===")
    diffusion = ForwardDiffusion(linear_sched)
    x_0 = torch.randn(1000, 2)  # (1000, 2)
    t_target = 100
    
    # 方法1：逐步加噪
    x_step = x_0.clone()
    for t in range(t_target):
        noise = torch.randn_like(x_step)
        x_step = torch.sqrt(linear_sched.alphas[t]) * x_step + torch.sqrt(linear_sched.betas[t]) * noise
    
    # 方法2：闭合形式直接加噪
    t_tensor = torch.full((1000,), t_target, dtype=torch.long)
    x_direct, _ = diffusion.add_noise(x_0, t_tensor)
    
    # 比较统计量
    print(f"逐步加噪 - 均值: {x_step.mean(0).numpy()}, 方差: {x_step.var(0).numpy()}")
    print(f"闭合形式 - 均值: {x_direct.mean(0).numpy()}, 方差: {x_direct.var(0).numpy()}")
    print(f"理论方差: alpha_bar * var(x_0) + (1 - alpha_bar) = "
          f"{linear_sched.alpha_bars[t_target].item():.4f} * 1.0 + "
          f"{1 - linear_sched.alpha_bars[t_target].item():.4f} = 1.0")
```

---

## 本章小结

| 概念 | 公式 / 要点 |
|:---|:---|
| 正向转移核 | $q(x_t\|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$ |
| 累积信号保留率 | $\bar{\alpha}_t = \prod_{s=1}^t (1-\beta_s)$ |
| 闭合形式加噪 | $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$ |
| 信噪比 | $\text{SNR}(t) = \bar{\alpha}_t / (1-\bar{\alpha}_t)$ |
| 后验均值 | $\tilde{\mu}_t = \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}x_0$ |
| 后验方差 | $\tilde{\beta}_t = \frac{(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\beta_t$ |
| 噪声调度 | 线性：$\beta_t \in [10^{-4}, 0.02]$；余弦：基于 $\cos^2$ 更平滑 |
| 信息论 | 互信息 $I(x_0;x_t)$ 随 $t$ 单调递减至零 |

---

## 练习题

### 基础题

**练习 7.1**（闭合形式验证）

给定 $\beta_1 = 0.0001$，$\beta_2 = 0.0002$，$\beta_3 = 0.0003$，手动计算 $\bar{\alpha}_3$，以及 $x_3$ 中信号和噪声的权重。验证 $\sqrt{\bar{\alpha}_3}^2 + \sqrt{1-\bar{\alpha}_3}^2 = 1$。

**练习 7.2**（后验方差计算）

使用上述 $\beta$ 值，计算 $t=3$ 时的后验方差 $\tilde{\beta}_3$。与 $\beta_3$ 比较，哪个更大？解释原因。

### 中级题

**练习 7.3**（余弦调度推导）

证明：对于余弦调度，当 $s \to 0$ 时，$\bar{\alpha}_t \approx \cos^2(\frac{\pi t}{2T})$。画出此函数并与 $e^{-ct}$（适当选择 $c$）比较。

**练习 7.4**（方差保持性质）

证明：如果 $x_0 \sim \mathcal{N}(0, I)$，则对所有 $t$，$x_t \sim \mathcal{N}(0, I)$。这个"方差保持"性质对数值稳定性有什么意义？如果去掉 $\sqrt{1-\beta_t}$ 的缩放（即 $x_t = x_{t-1} + \sqrt{\beta_t}\epsilon_t$），$\text{Var}(x_t)$ 如何增长？

### 提高题

**练习 7.5**（信噪比与最优调度）

定义信息效率为 $\eta = \frac{1}{T}\sum_{t=1}^T |I(x_0;x_t) - I(x_0;x_{t-1})|$，其中 $I$ 为互信息。直觉上，最优调度应该让每一步的"信息损失"尽可能均匀。假设 $x_0 \sim \mathcal{N}(0, I)$，推导出此条件下 $\bar{\alpha}_t$ 应满足的关系。提示：利用高斯变量的互信息公式 $I(x_0;x_t) = -\frac{d}{2}\log(1 - \bar{\alpha}_t)$（当 $\text{Var}(x_0) = I$ 时）。

---

## 练习答案

### 练习 7.1 解答

计算各步的 $\alpha$：

$$\alpha_1 = 1 - 0.0001 = 0.9999$$
$$\alpha_2 = 1 - 0.0002 = 0.9998$$
$$\alpha_3 = 1 - 0.0003 = 0.9997$$

累积乘积：

$$\bar{\alpha}_3 = 0.9999 \times 0.9998 \times 0.9997 = 0.9994$$

（精确值：$0.999400059994 \approx 0.9994$）

信号权重：$\sqrt{\bar{\alpha}_3} = \sqrt{0.9994} \approx 0.99970$

噪声权重：$\sqrt{1-\bar{\alpha}_3} = \sqrt{0.0006} \approx 0.02449$

验证：$\bar{\alpha}_3 + (1 - \bar{\alpha}_3) = 0.9994 + 0.0006 = 1$。注意验证的是系数的平方和为1，不是系数本身之和。

### 练习 7.2 解答

需要 $\bar{\alpha}_2 = \alpha_1 \cdot \alpha_2 = 0.9999 \times 0.9998 = 0.99970002$。

后验方差：

$$\tilde{\beta}_3 = \frac{(1 - \bar{\alpha}_2)}{1 - \bar{\alpha}_3} \cdot \beta_3 = \frac{1 - 0.99970002}{1 - 0.9994} \times 0.0003$$

$$= \frac{0.00029998}{0.0006} \times 0.0003 = 0.49997 \times 0.0003 \approx 0.00015$$

$\tilde{\beta}_3 \approx 0.00015 < \beta_3 = 0.0003$。后验方差小于前向噪声方差，因为后验分布利用了 $x_0$ 的额外信息来约束 $x_{t-1}$ 的不确定性。

### 练习 7.3 解答

当 $s \to 0$ 时：

$$f(t) = \cos^2\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right) \to \cos^2\left(\frac{t}{T} \cdot \frac{\pi}{2}\right)$$

$$\bar{\alpha}_t = \frac{f(t)}{f(0)} \to \frac{\cos^2(\pi t / 2T)}{\cos^2(0)} = \cos^2\left(\frac{\pi t}{2T}\right)$$

与指数衰减 $e^{-ct}$ 比较（选 $c$ 使得半衰点相同）：令 $\bar{\alpha}_{T/2} = 0.5$：
- 余弦：$\cos^2(\pi/4) = 0.5$，恰好在 $t = T/2$
- 指数：$e^{-cT/2} = 0.5$，得 $c = \frac{2\ln 2}{T}$

余弦调度在端点附近变化慢、中间变化快；指数衰减始终均匀。余弦调度更好，因为它在 $t \approx 0$（数据几乎无噪声）和 $t \approx T$（几乎纯噪声）附近花更多时间，这些区域的去噪相对容易，模型可以集中学习中间区域。

### 练习 7.4 解答

若 $x_0 \sim \mathcal{N}(0, I)$，则由闭合形式：

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$$

$$\mathbb{E}[x_t] = \sqrt{\bar{\alpha}_t}\mathbb{E}[x_0] + \sqrt{1-\bar{\alpha}_t}\mathbb{E}[\epsilon] = 0$$

$$\text{Var}(x_t) = \bar{\alpha}_t \text{Var}(x_0) + (1-\bar{\alpha}_t)\text{Var}(\epsilon) = \bar{\alpha}_t I + (1-\bar{\alpha}_t) I = I$$

所以 $x_t \sim \mathcal{N}(0, I)$ 对所有 $t$ 成立。

如果去掉缩放，$x_t = x_{t-1} + \sqrt{\beta_t}\epsilon_t$，则：

$$\text{Var}(x_t) = \text{Var}(x_{t-1}) + \beta_t = 1 + \sum_{s=1}^{t}\beta_s$$

方差线性增长，导致 $x_t$ 的数值范围不断扩大，造成数值不稳定。

### 练习 7.5 解答

当 $x_0 \sim \mathcal{N}(0, I)$ 时，$x_t | x_0 \sim \mathcal{N}(\sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$。互信息为：

$$I(x_0; x_t) = h(x_t) - h(x_t|x_0) = \frac{d}{2}\ln(2\pi e) - \frac{d}{2}\ln(2\pi e(1-\bar{\alpha}_t)) = -\frac{d}{2}\ln(1-\bar{\alpha}_t)$$

每一步的信息损失为：

$$\Delta I_t = I(x_0;x_{t-1}) - I(x_0;x_t) = -\frac{d}{2}\ln(1-\bar{\alpha}_{t-1}) + \frac{d}{2}\ln(1-\bar{\alpha}_t) = \frac{d}{2}\ln\frac{1-\bar{\alpha}_t}{1-\bar{\alpha}_{t-1}}$$

要使每步信息损失均匀（$\Delta I_t = C$ 常数），需要：

$$\frac{1-\bar{\alpha}_t}{1-\bar{\alpha}_{t-1}} = e^{2C/d} = \text{const}$$

即 $1 - \bar{\alpha}_t$ 应呈几何级数增长。设 $r = e^{2C/d}$，则：

$$1 - \bar{\alpha}_t = (1-\bar{\alpha}_0) \cdot r^t = \delta \cdot r^t$$

其中 $\delta$ 趋近于0。这意味着 $\bar{\alpha}_t = 1 - \delta r^t$，即 $\log(1-\bar{\alpha}_t)$ 应关于 $t$ 线性递增。这就是"对数SNR线性调度"的理论基础。

---

## 延伸阅读

1. **Ho, J., Jain, A., & Abbeel, P.** (2020). "Denoising Diffusion Probabilistic Models." *NeurIPS 2020.* -- DDPM原始论文，定义了正向过程和线性噪声调度。

2. **Nichol, A. Q., & Dhariwal, P.** (2021). "Improved Denoising Diffusion Probabilistic Models." *ICML 2021.* -- 提出余弦噪声调度和学习方差。

3. **Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., & Ganguli, S.** (2015). "Deep Unsupervised Learning using Nonequilibrium Thermodynamics." *ICML 2015.* -- 扩散模型的开山之作，将非平衡热力学引入生成模型。

4. **Kingma, D. P., Salimans, T., Poole, B., & Ho, J.** (2021). "Variational Diffusion Models." -- 从信噪比角度统一分析扩散模型的噪声调度。

5. **Feller, W.** (1949). "On the Theory of Stochastic Processes, with Particular Reference to Applications." -- 扩散过程的经典数学理论。

---

[上一章：马尔科夫链与随机过程](../part2-math-foundations/06-markov-chains-and-stochastic-processes.md) | [目录](../README.md) | [下一章：逆向去噪过程与DDPM](./08-reverse-denoising-ddpm.md)
