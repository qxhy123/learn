# 第19章：损失曲面分析

---

## 学习目标

学完本章后，你将能够：

1. **理解损失曲面的基本结构**：区分鞍点、局部极小、全局极小的数学定义，掌握各类临界点在深度网络中的分布规律
2. **分析高维空间中临界点的本质**：运用随机矩阵理论理解高维情形下鞍点主导性定理，认识深度学习中"局部极小问题"被高估的原因
3. **区分平坦极小与尖锐极小**：掌握曲率度量（Hessian 谱）与泛化能力的关系，理解批大小、学习率对极小点形状的影响
4. **掌握损失曲面可视化技术**：实现 Filter Normalization 方法，绘制一维和二维损失曲面，正确解读可视化结果
5. **建立损失曲面几何与泛化的联系**：理解尖锐/平坦极小对泛化差距的影响，了解 SAM 等利用几何信息的优化方法

---

## 19.1 损失曲面的基本概念

### 19.1.1 损失曲面的定义

设神经网络参数为 $\boldsymbol{\theta} \in \mathbb{R}^n$，训练集 $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$，训练损失定义为：

$$\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{N} \sum_{i=1}^N \ell(f(\mathbf{x}_i; \boldsymbol{\theta}),\, y_i)$$

**损失曲面**（Loss Landscape）是指映射 $\boldsymbol{\theta} \mapsto \mathcal{L}(\boldsymbol{\theta})$ 所定义的超曲面。对于现代深度网络，$n$ 通常为数百万乃至数十亿，使得该曲面存在于极高维空间中，无法直接可视化。

### 19.1.2 临界点的分类

在参数空间某点 $\boldsymbol{\theta}^*$ 处，若梯度为零：

$$\nabla \mathcal{L}(\boldsymbol{\theta}^*) = \mathbf{0}$$

则称 $\boldsymbol{\theta}^*$ 为**临界点**（Critical Point）。临界点的类型由 Hessian 矩阵 $\mathbf{H} = \nabla^2 \mathcal{L}(\boldsymbol{\theta}^*)$ 的特征值结构决定：

| 临界点类型 | Hessian 特征值条件 | 几何含义 |
|---|---|---|
| **全局极小** | 所有 $\lambda_i > 0$，且函数值最小 | 损失的全局最低点 |
| **局部极小** | 所有 $\lambda_i > 0$ | 邻域内的最低点 |
| **鞍点** | 存在 $\lambda_i > 0$ 且存在 $\lambda_j < 0$ | 某些方向下降，某些方向上升 |
| **局部极大** | 所有 $\lambda_i < 0$ | 邻域内的最高点（优化中极少遇到） |

**退化临界点**：若 Hessian 存在零特征值（$\lambda_i = 0$），称为退化情形，需要更高阶导数才能判断类型。

### 19.1.3 全局极小的非唯一性

深度网络的损失函数因网络对称性而存在大量等价的全局极小（或近似等价的低损失区域）。

**排列对称性**：设全连接网络第 $l$ 层有 $m$ 个神经元，交换这 $m$ 个神经元的顺序（同时调整前后层的连接权重），得到完全等价的参数点。对于有 $L$ 层、每层 $m$ 个神经元的网络，这类对称性产生 $(m!)^L$ 个等价参数点。

**尺度对称性**（ReLU 网络）：对于使用 ReLU 激活的网络，若第 $l$ 层第 $j$ 个神经元权重乘以常数 $c > 0$，同时下一层中连接该神经元的权重除以 $c$，网络函数不变：

$$f(\mathbf{x};\, \boldsymbol{\theta}) = f(\mathbf{x};\, \boldsymbol{\theta}')$$

这意味着损失函数在参数空间中存在连续的等价流形。

### 19.1.4 损失曲面的非凸性

深度网络的损失函数是**高度非凸**的，原因包括：

1. **激活函数的非线性**：ReLU、Sigmoid 等非线性激活使复合函数高度非凸
2. **层间乘积结构**：多层网络的参数以乘积形式出现，产生非凸性
3. **过参数化**：参数数量远超训练样本，产生大量（近似）等价解

然而，实践中发现 SGD 等一阶方法在这类高度非凸曲面上训练效果出奇地好，这一现象促使研究者深入分析损失曲面的几何结构。

```
损失值
  │
  │   局部极大
  │      ╱╲
  │     /  \    鞍点
  │    /    \  /╲  /
  │   /      \/  \/
  │  /  局部极小   \
  │ /              \
  │/    全局极小     \
  └────────────────────── 参数空间（1D示意）
```

---

## 19.2 高维空间中的临界点

### 19.2.1 高维空间的反直觉性质

在一维情形，极小点和极大点是"常见"的临界点，鞍点则是分隔极大极小的过渡点。但在高维空间中，情况截然不同。

**指数稀少的纯极小**：若 Hessian 矩阵的每个特征值独立地以概率 $p$ 为正，则一个随机临界点是局部极小（所有 $n$ 个特征值均为正）的概率为 $p^n$。当 $n$ 达到数百万时，这一概率趋近于零。

**定理（Dauphin et al., 2014）**：在高维空间中，几乎所有的低损失临界点均为局部极小；高损失临界点几乎均为鞍点，且鞍点的损失值与其逃逸指数（负特征值比例）正相关。

### 19.2.2 随机矩阵理论视角

考虑损失函数的 Hessian 矩阵 $\mathbf{H} \in \mathbb{R}^{n \times n}$，其特征值分布遵循一定规律。

**Wigner 半圆律**：对于均值为零的独立同分布随机矩阵，特征值分布趋向于半圆分布：

$$\rho(\lambda) = \frac{1}{2\pi\sigma^2}\sqrt{4\sigma^2 - \lambda^2}, \quad |\lambda| \leq 2\sigma$$

对于神经网络 Hessian，情况更复杂，但其谱结构表现出类似性质：

- **大量接近零的特征值**：对应平坦方向（损失对这些参数方向不敏感）
- **少量大正特征值**：对应损失函数的敏感方向
- **极少量负特征值**（若有）：对应鞍点方向

### 19.2.3 鞍点与优化速度

一阶方法（如 SGD）在鞍点附近会显著减速，因为此时梯度接近零。

**鞍点逃逸时间**：在一维二次鞍点 $f(x) = -\frac{1}{2}x^2$ 附近，梯度下降的更新为 $x_{k+1} = (1 + \alpha)x_k$。从 $x_0 = \epsilon$ 出发，逃逸到 $|x| > \delta$ 需要迭代次数：

$$k^* = \left\lceil \frac{\ln(\delta/\epsilon)}{\ln(1+\alpha)} \right\rceil \approx \frac{\ln(\delta/\epsilon)}{\alpha}$$

当初始值 $\epsilon$ 极小时（数值精度限制），逃逸时间很长。

**噪声加速逃逸**：SGD 中的随机梯度噪声有助于逃逸鞍点。设鞍点方向的 Hessian 特征值为 $-\lambda < 0$，噪声方差为 $\sigma^2$，逃逸时间约为：

$$\mathbb{E}[T_{\text{escape}}] \approx \frac{1}{\lambda} \ln\left(\frac{\delta^2}{\sigma^2 / \lambda}\right)$$

这说明适当的噪声（较大的学习率或较小的批大小）反而有助于优化。

### 19.2.4 严格鞍点与二阶方法

**定义**：若临界点 $\boldsymbol{\theta}^*$ 满足 $\nabla \mathcal{L}(\boldsymbol{\theta}^*) = \mathbf{0}$ 且 Hessian 存在**严格负特征值**（$\lambda_{\min}(\mathbf{H}) < 0$），则称为**严格鞍点**。

**理论保证**：在严格鞍点假设下，带噪声的梯度下降以概率 1 逃离所有严格鞍点并收敛到局部极小（Jin et al., 2017）。

二阶方法（Newton 法、自然梯度等）可以利用 Hessian 信息主动识别负曲率方向并快速逃逸鞍点，但计算代价高昂（$O(n^3)$ 复杂度）。

---

## 19.3 平坦极小与尖锐极小

### 19.3.1 数学定义

设 $\boldsymbol{\theta}^*$ 是损失函数的一个局部极小点。

**尖锐极小**（Sharp Minima）：极小点附近曲率大，即 Hessian 最大特征值 $\lambda_{\max}(\mathbf{H})$ 较大。直觉上，参数的微小扰动会导致损失急剧上升：

$$\mathcal{L}(\boldsymbol{\theta}^* + \boldsymbol{\epsilon}) \gg \mathcal{L}(\boldsymbol{\theta}^*), \quad \text{即使 } \|\boldsymbol{\epsilon}\| \text{ 很小}$$

**平坦极小**（Flat Minima）：极小点附近曲率小，Hessian 特征值均较小。参数扰动对损失影响不大：

$$\mathcal{L}(\boldsymbol{\theta}^* + \boldsymbol{\epsilon}) \approx \mathcal{L}(\boldsymbol{\theta}^*), \quad \forall \|\boldsymbol{\epsilon}\| \leq r$$

其中 $r$ 称为**平坦性半径**（flatness radius）。

### 19.3.2 曲率度量指标

定量描述极小点的尖锐/平坦程度有多种方式：

**Hessian 迹**（Trace of Hessian）：
$$\text{Tr}(\mathbf{H}) = \sum_{i=1}^n \lambda_i$$

反映所有方向的平均曲率，计算代价为 $O(n)$（使用 Hutchinson 估计器）。

**Hessian 最大特征值**：
$$\lambda_{\max} = \max_{\|\mathbf{v}\|=1} \mathbf{v}^\top \mathbf{H} \mathbf{v}$$

衡量最敏感方向的曲率，可用幂迭代法高效计算。

**$\epsilon$-尖锐度**（Keskar et al., 2017）：
$$\phi(\boldsymbol{\theta}^*, \epsilon) = \frac{\max_{\|\boldsymbol{\delta}\|_\infty \leq \epsilon} \mathcal{L}(\boldsymbol{\theta}^* + \boldsymbol{\delta}) - \mathcal{L}(\boldsymbol{\theta}^*)}{\max(1, \mathcal{L}(\boldsymbol{\theta}^*))} \times 100\%$$

衡量在 $\ell_\infty$ 球内的最大损失上升，越大表示越尖锐。

### 19.3.3 批大小对极小点形状的影响

Keskar et al.（2017）的重要发现：**大批量训练趋向于收敛到尖锐极小，小批量训练趋向于收敛到平坦极小**。

直觉解释：

- **小批量 SGD**：每步使用不同的小批量，损失曲面等效于在轻微扰动的曲面上优化，鼓励找到在扰动下损失仍低的**宽谷**
- **大批量 SGD**：每步使用更精确的梯度估计，倾向于沿确定方向下降，容易陷入局部的**尖锐极小**

**批大小与有效学习率的关系**：为保持优化行为一致，存在一个经验法则——线性缩放规则（Goyal et al., 2017）：批大小扩大 $k$ 倍时，学习率也应扩大 $k$ 倍：

$$\alpha_{\text{large}} = k \cdot \alpha_{\text{small}}$$

该规则在一定范围内有效，但批大小过大时效果下降。

### 19.3.4 泛化与极小点形状

**核心假设**：平坦极小具有更好的泛化能力。

直觉理由：测试集分布与训练集分布存在微小偏移（域偏移、标签噪声等），相当于损失曲面发生微小扰动。在平坦极小附近，扰动导致的损失上升较小；在尖锐极小附近，相同的扰动会导致更大的损失上升。

**泛化界**（PAC-Bayes 框架）：设 $P$ 为参数先验，$Q$ 为后验，训练损失为 $\hat{\mathcal{L}}_{\text{train}}$，泛化损失为 $\mathcal{L}_{\text{test}}$，则以高概率：

$$\mathcal{L}_{\text{test}}(\boldsymbol{\theta}) \leq \hat{\mathcal{L}}_{\text{train}}(\boldsymbol{\theta}) + \sqrt{\frac{D_{\text{KL}}(Q \| P) + \ln(N/\delta)}{2N}}$$

平坦极小意味着 $Q$ 可以选取较宽的高斯分布，使 $D_{\text{KL}}(Q \| P)$ 较小，从而泛化界更紧。

```
损失值
  │
  │         尖锐极小                 平坦极小
  │           /\                    ___________
  │          /  \                  /           \
  │         /    \                /             \
  │─────────      ─────────────────               ───────
  │
  └──────────────────────────────────────────── 参数空间

  训练损失相同，但测试时（分布偏移后）：
  尖锐极小处损失急剧上升，平坦极小处损失变化平缓
```

### 19.3.5 对平坦极小假说的质疑

平坦极小与泛化的关系并非无争议。Dinh et al.（2017）指出，对于 ReLU 网络，利用尺度不变性可以构造一个极小点，使其任意平坦或任意尖锐，但泛化性能不变。这说明**某些曲率度量对尺度重参数化不鲁棒**，需要使用尺度不变的度量方式。

---

## 19.4 损失曲面可视化方法

### 19.4.1 可视化的挑战

直接可视化高维损失曲面面临根本困难：

- 参数维度 $n$ 可达数十亿，无法直接绘图
- 低维投影可能丢失关键几何信息
- 不同参数点的选取方式影响可视化结果

### 19.4.2 一维线性插值

最简单的可视化方式：在两个参数点之间做线性插值，观察损失变化。

设两个参数点 $\boldsymbol{\theta}_A$（如训练初始点）和 $\boldsymbol{\theta}_B$（如训练终止点），定义插值路径：

$$\boldsymbol{\theta}(t) = (1 - t)\boldsymbol{\theta}_A + t\boldsymbol{\theta}_B, \quad t \in [0, 1]$$

绘制 $t \mapsto \mathcal{L}(\boldsymbol{\theta}(t))$ 的曲线，可观察两解之间是否存在势垒（barrier）。

**局限性**：两个参数点之间可能存在尺度差异，直接插值可能经过参数空间中物理上无意义的区域。

### 19.4.3 Filter Normalization 方法

Li et al.（2018）提出 **Filter Normalization** 方法，通过归一化随机方向消除尺度影响。

**核心思想**：构造两个单位化的随机方向 $\boldsymbol{\delta}$ 和 $\boldsymbol{\eta}$，使每个方向的每个"滤波器"（filter，即每层权重矩阵的一行或一个卷积核）与对应参数的滤波器等范数：

对于参数 $\boldsymbol{\theta}^*$ 的第 $i$ 层第 $j$ 个滤波器 $\boldsymbol{\theta}^*_{ij}$，将随机方向 $\mathbf{d}_{ij}$ 归一化为：

$$\hat{\mathbf{d}}_{ij} = \frac{\mathbf{d}_{ij}}{\|\mathbf{d}_{ij}\|} \cdot \|\boldsymbol{\theta}^*_{ij}\|$$

这样，方向的每个滤波器分量与参数本身的范数匹配，可视化结果对尺度重参数化具有鲁棒性。

**二维损失曲面**：在两个归一化方向 $\boldsymbol{\delta}$ 和 $\boldsymbol{\eta}$ 上构造二维网格：

$$\mathcal{L}(\alpha, \beta) = \mathcal{L}(\boldsymbol{\theta}^* + \alpha\boldsymbol{\delta} + \beta\boldsymbol{\eta})$$

对 $(\alpha, \beta)$ 在二维网格上求值，绘制等高线图或三维曲面图。

### 19.4.4 轨迹可视化

另一类可视化方法关注**优化轨迹**：将训练过程中参数的变化轨迹投影到二维平面上。

**PCA 投影**：收集训练过程中保存的参数快照 $\{\boldsymbol{\theta}^{(k)}\}_{k=0}^K$，对其做主成分分析（PCA），取前两个主成分 $\mathbf{u}_1, \mathbf{u}_2$ 作为投影方向，绘制轨迹在该平面的投影以及对应的损失等高线。

**优点**：能展示优化路径的走势，揭示优化器在曲面上的行为差异（SGD vs Adam）。

---

## 19.5 泛化与损失曲面几何的关系

### 19.5.1 泛化差距的几何解释

设训练损失为 $\mathcal{L}_{\text{train}}(\boldsymbol{\theta})$，测试损失为 $\mathcal{L}_{\text{test}}(\boldsymbol{\theta})$，**泛化差距**定义为：

$$\text{Gap}(\boldsymbol{\theta}) = \mathcal{L}_{\text{test}}(\boldsymbol{\theta}) - \mathcal{L}_{\text{train}}(\boldsymbol{\theta})$$

在过参数化网络中，训练损失可以压缩到接近零，但泛化差距的大小则与极小点的几何有关。

**局部弹性（Local Entropy）视角**（Chaudhari et al., 2017）：定义参数 $\boldsymbol{\theta}$ 的局部熵（平滑版本的损失）：

$$\mathcal{F}(\boldsymbol{\theta}, \gamma) = -\gamma \ln \int_{\mathbb{R}^n} e^{-\mathcal{L}(\boldsymbol{\phi})/\gamma} \cdot \mathcal{N}(\boldsymbol{\phi}; \boldsymbol{\theta}, I) \, d\boldsymbol{\phi}$$

该量在平坦区域取小值，在尖锐区域取大值，可作为优化目标以鼓励收敛到平坦极小。**Entropy-SGD** 优化器基于此思想设计。

### 19.5.2 SAM：利用几何的优化方法

**SAM**（Sharpness-Aware Minimization，Foret et al., 2021）是一种通过直接最小化损失曲面尖锐度来提升泛化的优化方法。

**SAM 目标函数**：

$$\min_{\boldsymbol{\theta}} \left[ \max_{\|\boldsymbol{\epsilon}\|_2 \leq \rho} \mathcal{L}(\boldsymbol{\theta} + \boldsymbol{\epsilon}) \right]$$

即寻找参数 $\boldsymbol{\theta}$，使其邻域内的**最差损失**尽可能小。这明确鼓励收敛到平坦极小。

**SAM 的实现**（两步更新）：

第一步：计算最差扰动方向（内层最大化的近似解）：

$$\hat{\boldsymbol{\epsilon}} = \rho \cdot \frac{\nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta})}{\|\nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta})\|_2}$$

第二步：在扰动点处计算梯度并更新参数：

$$\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k - \alpha \cdot \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_k + \hat{\boldsymbol{\epsilon}})$$

SAM 每步需要两次前向-反向传播，计算代价约为 SGD 的两倍，但在图像分类等任务上显著提升泛化性能。

### 19.5.3 损失曲面连通性

**模式连通性**（Mode Connectivity）：研究发现，随机梯度下降找到的两个独立解 $\boldsymbol{\theta}_A$ 和 $\boldsymbol{\theta}_B$ 之间，存在一条损失值几乎处处低的**弯曲路径**（而非直线路径），即两个解通过低损失区域连通。

**定理（Garipov et al., 2018）**：对于足够宽的网络，存在一条二次贝塞尔曲线 $\gamma: [0,1] \to \mathbb{R}^n$，满足 $\gamma(0) = \boldsymbol{\theta}_A$，$\gamma(1) = \boldsymbol{\theta}_B$，且：

$$\max_{t \in [0,1]} \mathcal{L}(\gamma(t)) \approx \mathcal{L}(\boldsymbol{\theta}_A) \approx \mathcal{L}(\boldsymbol{\theta}_B)$$

这一发现有重要实践意义：**模型融合**（Model Averaging）在参数空间进行平均时，若存在势垒则效果较差；利用连通路径做**快照集成**（Snapshot Ensembling）则可以提升效果。

### 19.5.4 损失曲面与网络宽度、深度

研究显示网络结构影响损失曲面几何：

- **更宽的网络**：局部极小更少，损失曲面更"良性"，不同解之间更易连通
- **更深的网络**：表达能力增强，但曲面更复杂；残差连接（ResNet）显著改善了损失曲面的凸性
- **批归一化**（Batch Normalization）：平滑损失曲面，使梯度在训练过程中更为稳定，有效学习率的敏感性降低

**ResNet 的曲面改善**：Li et al.（2018）的可视化实验清晰地显示，相同深度的 VGG 网络（无残差连接）损失曲面存在混沌状的尖锐结构，而 ResNet 的损失曲面呈现出大范围平坦的碗状，极大地简化了优化问题。

---

## 本章小结

| 概念 | 核心内容 | 实践意义 |
|---|---|---|
| **临界点分类** | 极小、极大、鞍点由 Hessian 特征值正负决定 | 高维中鞍点主导，局部极小问题被高估 |
| **高维临界点** | 高损失临界点几乎均为鞍点，低损失点几乎均为极小 | SGD+噪声可有效逃逸鞍点，一阶方法足够 |
| **平坦 vs 尖锐** | 平坦极小 Hessian 特征值小，尖锐极小特征值大 | 小批量/大学习率→平坦；大批量→尖锐 |
| **可视化技术** | Filter Normalization 消除尺度影响 | 可直观比较不同网络/优化器的曲面形状 |
| **泛化与几何** | 平坦极小泛化差距更小 | SAM 等方法直接优化曲面几何以提升泛化 |
| **模式连通性** | 独立解之间存在低损失弯曲路径 | 支持快照集成、模型融合等技术 |

**关键结论**：

$$\text{深度网络成功训练} \Leftrightarrow \text{损失曲面良性结构} + \text{优化器适配}$$

高维空间中鞍点的主导性使得一阶方法通常足够；平坦极小假说为理解小批量训练的泛化优势提供了几何解释；损失曲面可视化是理解网络结构影响的有力工具。

---

## 深度学习应用：损失曲面可视化

本节用 PyTorch 实现损失曲面可视化，包括 Filter Normalization、一维曲面绘制和二维曲面绘制。

### 完整实现代码

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import copy
from typing import List, Tuple

# ─────────────────────────────────────────
# 1. 定义简单卷积网络（用于演示）
# ─────────────────────────────────────────

class SimpleConvNet(nn.Module):
    """用于 CIFAR-10 分类的简单卷积网络"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                           # -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                           # -> 8x8
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ─────────────────────────────────────────
# 2. 数据加载与快速训练
# ─────────────────────────────────────────

def get_cifar10_loader(batch_size: int = 128, num_workers: int = 2):
    """加载 CIFAR-10 训练集和测试集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader


def train_model(model: nn.Module,
                train_loader,
                epochs: int = 5,
                lr: float = 0.01,
                device: str = 'cpu') -> nn.Module:
    """快速训练模型（仅用于演示）"""
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}]  Loss: {avg_loss:.4f}")

    return model


def compute_loss(model: nn.Module,
                 loader,
                 device: str = 'cpu') -> float:
    """计算模型在数据集上的平均损失"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            total_loss += criterion(model(batch_x), batch_y).item()
            n_batches += 1
    return total_loss / n_batches


# ─────────────────────────────────────────
# 3. Filter Normalization 工具函数
# ─────────────────────────────────────────

def get_params(model: nn.Module) -> List[torch.Tensor]:
    """提取模型所有参数，返回列表（每个张量保留原始形状）"""
    return [p.detach().clone() for p in model.parameters()]


def filter_normalize_direction(direction: List[torch.Tensor],
                               reference: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    对方向 direction 做 Filter Normalization：
    使 direction 的每个滤波器分量的范数
    与 reference 对应参数的滤波器范数相同。

    规则：
    - 对于 >= 2 维参数（卷积核/全连接层权重），
      按第 0 维切分为"滤波器"，逐个归一化
    - 对于 1 维参数（偏置/BN 参数），按标量归一化
    """
    normalized = []
    for d, ref in zip(direction, reference):
        if d.dim() >= 2:
            # 逐行（滤波器）归一化
            d_norm = d.clone()
            for j in range(d.shape[0]):
                d_filter = d[j]
                ref_filter = ref[j]
                d_fnorm = d_filter.norm()
                ref_fnorm = ref_filter.norm()
                if d_fnorm > 1e-10:
                    d_norm[j] = d_filter / d_fnorm * ref_fnorm
                # 若 d_fnorm 极小则保持为零向量
            normalized.append(d_norm)
        else:
            # 1D 参数整体归一化
            d_norm_val = d.norm()
            ref_norm_val = ref.norm()
            if d_norm_val > 1e-10:
                normalized.append(d / d_norm_val * ref_norm_val)
            else:
                normalized.append(d.clone())
    return normalized


def set_params(model: nn.Module,
               base: List[torch.Tensor],
               direction1: List[torch.Tensor],
               alpha: float,
               direction2: List[torch.Tensor] = None,
               beta: float = 0.0) -> None:
    """
    将模型参数设为：
        θ = base + α * direction1 + β * direction2
    (direction2 为 None 时忽略 β 项)
    """
    with torch.no_grad():
        for i, (p, b, d1) in enumerate(
                zip(model.parameters(), base, direction1)):
            new_p = b + alpha * d1
            if direction2 is not None:
                new_p = new_p + beta * direction2[i]
            p.copy_(new_p)


# ─────────────────────────────────────────
# 4. 一维损失曲面
# ─────────────────────────────────────────

def plot_1d_loss_surface(
        model: nn.Module,
        loader,
        n_points: int = 51,
        alpha_range: Tuple[float, float] = (-1.0, 1.0),
        device: str = 'cpu',
        label: str = 'Loss Surface',
        ax: plt.Axes = None) -> None:
    """
    沿一个 filter-normalized 随机方向绘制一维损失曲线。

    参数
    ----
    model      : 已训练的网络（将作为原点 θ*）
    loader     : 用于计算损失的数据加载器
    n_points   : 采样点数
    alpha_range: 扰动幅度范围
    device     : 计算设备
    label      : 图例标签
    ax         : matplotlib Axes（为 None 时自动创建）
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    base = get_params(model)

    # 生成并归一化随机方向
    raw_dir = [torch.randn_like(p) for p in model.parameters()]
    direction = filter_normalize_direction(raw_dir, base)

    alphas = np.linspace(alpha_range[0], alpha_range[1], n_points)
    losses = []

    for alpha in alphas:
        set_params(model, base, direction, alpha)
        loss = compute_loss(model, loader, device)
        losses.append(loss)

    # 还原原始参数
    set_params(model, base, direction, 0.0)

    ax.plot(alphas, losses, linewidth=2, label=label)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='θ* (原始参数)')
    ax.set_xlabel('扰动幅度 α', fontsize=12)
    ax.set_ylabel('损失值', fontsize=12)
    ax.set_title('一维损失曲面（Filter Normalization）', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)


# ─────────────────────────────────────────
# 5. 二维损失曲面
# ─────────────────────────────────────────

def compute_2d_loss_surface(
        model: nn.Module,
        loader,
        n_points: int = 21,
        range_val: float = 1.0,
        device: str = 'cpu') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    沿两个 filter-normalized 随机正交方向计算二维损失曲面。

    返回
    ----
    (A, B, L) 三个形状为 (n_points, n_points) 的数组：
        A, B 为网格坐标；L 为对应损失值
    """
    base = get_params(model)

    # 生成两个随机方向并归一化
    raw_d1 = [torch.randn_like(p) for p in model.parameters()]
    raw_d2 = [torch.randn_like(p) for p in model.parameters()]

    dir1 = filter_normalize_direction(raw_d1, base)
    dir2 = filter_normalize_direction(raw_d2, base)

    # 对 dir2 做 Gram-Schmidt 正交化（近似，逐层向量级别）
    # 此处简化处理：直接使用两个独立随机方向
    # 完整实现可展开为所有参数的1D向量后做正交化

    alphas = np.linspace(-range_val, range_val, n_points)
    betas  = np.linspace(-range_val, range_val, n_points)

    A, B = np.meshgrid(alphas, betas)
    L = np.zeros_like(A)

    total = n_points * n_points
    count = 0
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            set_params(model, base, dir1, alpha, dir2, beta)
            L[j, i] = compute_loss(model, loader, device)
            count += 1
            if count % (total // 10) == 0:
                print(f"  进度：{count}/{total} ({100*count/total:.0f}%)")

    # 还原原始参数
    set_params(model, base, dir1, 0.0, dir2, 0.0)

    return A, B, L


def plot_2d_loss_surface(A: np.ndarray,
                         B: np.ndarray,
                         L: np.ndarray,
                         title: str = '二维损失曲面') -> None:
    """绘制二维损失曲面：等高线图 + 三维曲面图"""
    fig = plt.figure(figsize=(14, 6))

    # 等高线图
    ax1 = fig.add_subplot(1, 2, 1)
    contour = ax1.contourf(A, B, L, levels=50, cmap='viridis')
    plt.colorbar(contour, ax=ax1, label='损失值')
    ax1.contour(A, B, L, levels=20, colors='white', alpha=0.3, linewidths=0.5)
    ax1.plot(0, 0, 'r*', markersize=15, label='θ* (训练解)')
    ax1.set_xlabel('方向 δ（α）', fontsize=11)
    ax1.set_ylabel('方向 η（β）', fontsize=11)
    ax1.set_title(f'{title}（等高线）', fontsize=12)
    ax1.legend()

    # 三维曲面图
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax2.plot_surface(A, B, L, cmap='viridis', alpha=0.85,
                            linewidth=0, antialiased=True)
    plt.colorbar(surf, ax=ax2, shrink=0.5, label='损失值')
    ax2.set_xlabel('δ 方向', fontsize=10)
    ax2.set_ylabel('η 方向', fontsize=10)
    ax2.set_zlabel('损失', fontsize=10)
    ax2.set_title(f'{title}（三维）', fontsize=12)
    ax2.view_init(elev=35, azim=-60)

    plt.tight_layout()
    plt.savefig('loss_surface_2d.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("图像已保存至 loss_surface_2d.png")


# ─────────────────────────────────────────
# 6. 比较大批量与小批量的极小点形状
# ─────────────────────────────────────────

def compare_sharp_vs_flat(device: str = 'cpu') -> None:
    """
    训练两个模型：一个用小批量（平坦极小），
    一个用大批量（尖锐极小），比较损失曲面形状。
    """
    print("=" * 50)
    print("比较小批量（平坦极小）vs 大批量（尖锐极小）")
    print("=" * 50)

    train_loader_small, test_loader = get_cifar10_loader(batch_size=64)
    train_loader_large, _           = get_cifar10_loader(batch_size=2048)

    # 小批量训练 → 平坦极小
    print("\n[1] 小批量训练（batch_size=64）...")
    model_flat = SimpleConvNet()
    model_flat = train_model(model_flat, train_loader_small,
                             epochs=5, lr=0.01, device=device)

    # 大批量训练 → 尖锐极小
    print("\n[2] 大批量训练（batch_size=2048，线性缩放学习率）...")
    model_sharp = SimpleConvNet()
    model_sharp = train_model(model_sharp, train_loader_large,
                              epochs=5, lr=0.32, device=device)   # 0.01 * 2048/64

    # 绘制一维损失曲面比较
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    print("\n计算小批量模型的损失曲面...")
    plot_1d_loss_surface(model_flat, test_loader, n_points=51,
                         alpha_range=(-0.5, 0.5), device=device,
                         label='小批量（平坦极小）', ax=axes[0])

    print("\n计算大批量模型的损失曲面...")
    plot_1d_loss_surface(model_sharp, test_loader, n_points=51,
                         alpha_range=(-0.5, 0.5), device=device,
                         label='大批量（尖锐极小）', ax=axes[1])

    axes[0].set_title('小批量训练（batch=64）\n平坦极小', fontsize=12)
    axes[1].set_title('大批量训练（batch=2048）\n尖锐极小', fontsize=12)

    plt.suptitle('大批量 vs 小批量：损失曲面形状对比', fontsize=14)
    plt.tight_layout()
    plt.savefig('sharp_vs_flat_minima.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("对比图已保存至 sharp_vs_flat_minima.png")


# ─────────────────────────────────────────
# 7. 计算 Hessian 迹（曲率度量）
# ─────────────────────────────────────────

def hutchinson_trace(model: nn.Module,
                     loader,
                     n_samples: int = 20,
                     device: str = 'cpu') -> float:
    """
    使用 Hutchinson 估计器计算 Hessian 迹：
        Tr(H) ≈ (1/n) Σ_i  z_i^T H z_i
    其中 z_i ~ Rademacher(均匀±1)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    # 取一个小批量（用于代理整体 Hessian）
    batch_x, batch_y = next(iter(loader))
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

    trace_estimates = []
    for _ in range(n_samples):
        # 生成 Rademacher 随机向量
        z_vectors = [torch.randint(0, 2, p.shape, device=device).float() * 2 - 1
                     for p in model.parameters()]

        # 第一次反向传播：计算梯度 g = ∇L
        loss = criterion(model(batch_x), batch_y)
        grads = torch.autograd.grad(loss, model.parameters(),
                                    create_graph=True)

        # 计算 g·z（梯度与随机向量的点积）
        grad_z = sum((g * z).sum()
                     for g, z in zip(grads, z_vectors))

        # 第二次反向传播：计算 Hz（Hessian-向量积）
        hvp = torch.autograd.grad(grad_z, model.parameters(),
                                  retain_graph=False)

        # Tr(H) ≈ z^T (Hz)
        trace_estimate = sum((hz * z).sum().item()
                             for hz, z in zip(hvp, z_vectors))
        trace_estimates.append(trace_estimate)

    return float(np.mean(trace_estimates))


# ─────────────────────────────────────────
# 8. 主程序入口
# ─────────────────────────────────────────

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备：{device}")

    # 加载数据
    train_loader, test_loader = get_cifar10_loader(batch_size=128)

    # 训练模型
    print("\n训练基准模型...")
    model = SimpleConvNet()
    model = train_model(model, train_loader, epochs=5,
                        lr=0.01, device=device)

    train_loss = compute_loss(model, train_loader, device)
    test_loss  = compute_loss(model, test_loader,  device)
    print(f"\n训练损失：{train_loss:.4f}  测试损失：{test_loss:.4f}")
    print(f"泛化差距：{test_loss - train_loss:.4f}")

    # 计算 Hessian 迹（曲率度量）
    print("\n计算 Hessian 迹（Hutchinson 估计）...")
    trace = hutchinson_trace(model, train_loader, n_samples=10, device=device)
    print(f"Tr(H) ≈ {trace:.2f}")

    # 绘制一维损失曲面
    print("\n绘制一维损失曲面...")
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_1d_loss_surface(model, test_loader, n_points=51,
                         alpha_range=(-0.8, 0.8),
                         device=device, ax=ax)
    plt.savefig('loss_surface_1d.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("一维曲面图已保存至 loss_surface_1d.png")

    # 绘制二维损失曲面（计算较耗时，n_points 可适当减小）
    print("\n计算二维损失曲面（可能需要几分钟）...")
    A, B, L = compute_2d_loss_surface(model, test_loader,
                                      n_points=21, range_val=0.5,
                                      device=device)
    plot_2d_loss_surface(A, B, L, title='SimpleConvNet 损失曲面')

    # 比较大批量与小批量的极小点形状
    compare_sharp_vs_flat(device=device)


if __name__ == '__main__':
    main()
```

### 代码说明

**Filter Normalization 的必要性**：

不做归一化时，若网络某层权重整体较小（如经过正则化训练），随机方向中该层的扰动相对偏大，可视化结果会被这些"小权重层"主导，失去代表性。Filter Normalization 确保每个方向上各层的相对扰动幅度与参数本身的范数匹配，保证可视化的尺度一致性。

**Hutchinson 估计器**：

直接计算 $n \times n$ Hessian 矩阵的代价为 $O(n^2)$，对大网络不可行。Hutchinson 估计器利用：

$$\text{Tr}(\mathbf{H}) = \mathbb{E}_{\mathbf{z}}[\mathbf{z}^\top \mathbf{H} \mathbf{z}]$$

其中 $\mathbf{z}$ 为 Rademacher 向量（元素均匀取 $\pm 1$）。每次评估只需计算一个 Hessian-向量积（代价等同于一次反向传播），通过 $k$ 次采样得到无偏估计，总代价 $O(kn)$。

**运行时间预估**：

| 操作 | 网格大小 | 预计时间（CPU） |
|---|---|---|
| 一维曲面（51点） | 51×1 | 约 2 分钟 |
| 二维曲面（21×21） | 441 | 约 15 分钟 |
| Hessian 迹（10样本） | - | 约 1 分钟 |

建议在 GPU 上运行，速度可提升 10-50 倍。

---

## 练习题

### 基础题

**练习 19.1**（临界点分类）

考虑二维函数 $f(x, y) = x^2 - y^2$。

(a) 求 $f$ 的所有临界点。

(b) 计算临界点处的 Hessian 矩阵，并判断其类型（极小、极大、鞍点）。

(c) 设优化算法从点 $(0.1, 0.0)$ 出发，使用梯度下降 $\nabla f = (2x, -2y)^T$，学习率 $\alpha = 0.1$，迭代5步后参数移动方向如何？这体现了鞍点的什么性质？

---

**练习 19.2**（平坦极小度量）

设一维损失函数为：

$$\mathcal{L}_1(\theta) = (\theta - 1)^4, \quad \mathcal{L}_2(\theta) = (\theta - 1)^2$$

(a) 两个函数均在 $\theta^* = 1$ 处取得极小值。计算两处的 Hessian（即二阶导数）值。哪个极小更"平坦"？

(b) 对于 $\mathcal{L}_1$，在 $\epsilon = 0.1$ 的扰动范围内，损失最大上升多少？对 $\mathcal{L}_2$ 做同样计算，与 Hessian 度量的结论是否一致？

(c) $\mathcal{L}_1$ 的 Hessian 在 $\theta^* = 1$ 处为零，意味着什么？这是退化临界点吗？

---

### 中级题

**练习 19.3**（高维鞍点概率）

设 $n$ 维空间中随机临界点的每个 Hessian 特征值独立地以概率 $p = 0.5$ 为正，以概率 $0.5$ 为负。

(a) 一个随机临界点是局部极小（所有特征值为正）的概率为多少？当 $n = 1, 10, 100, 10^6$ 时分别是多少？

(b) 证明：当 $n \to \infty$ 时，几乎所有临界点都是鞍点（存在正负特征值混合）。

(c) 设训练神经网络时遇到一个损失值为 $\mathcal{L} = 0.5$ 的临界点，根据 Dauphin et al. 的理论，该点更可能是局部极小还是鞍点？请解释原因。

---

**练习 19.4**（Filter Normalization 的作用）

考虑一个单层线性网络 $f(x; w) = wx$（$w \in \mathbb{R}$），损失 $\mathcal{L}(w) = (wx - y)^2$。

(a) 若对参数进行尺度变换：令 $w' = cw$（$c > 0$ 为常数），写出新的损失函数 $\mathcal{L}'(w')$，并验证 $\mathcal{L}'(w'/c) = \mathcal{L}(w)$（即参数空间中的不同点对应相同的函数值）。

(b) 设训练得到 $w^* = 2$，随机扰动方向 $d = 0.3$。不做 Filter Normalization 时，扰动后参数为 $w^* + \alpha d$；若做 Filter Normalization（将 $d$ 归一化到与 $w^*$ 同范数），归一化后的方向 $\hat{d}$ 是多少？

(c) 当模型使用了尺度变换 $w' = 10w$（即同一个模型用不同参数表示），分别用原始随机方向和 Filter Normalization 方向计算 $\alpha = 0.1$ 处的损失，哪种方法对两种参数化一致？

---

### 进阶题

**练习 19.5**（SAM 优化器实现与分析）

实现 SAM 优化器并与 SGD 比较泛化性能。

(a) 根据本章介绍，用 PyTorch 实现 SAM 优化器，关键步骤：

```python
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        # 初始化基础优化器（如 SGD）
        # rho 控制邻域半径
        ...

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        # 计算扰动方向：ε_hat = ρ * grad / ||grad||
        # 将参数移动到扰动位置 θ + ε_hat
        ...

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        # 将参数从扰动位置还原到原始位置 θ
        # 然后用扰动位置的梯度更新参数
        ...
```

(b) 在 CIFAR-10 上用相同网络架构，分别用 SGD 和 SAM（base: SGD）训练 10 个 epoch，比较：训练损失、测试损失、泛化差距。

(c) 用本章的 `plot_1d_loss_surface` 函数，对 SAM 和 SGD 的训练结果各绘制一条一维损失曲线，定性比较两者极小点的"平坦程度"。给出你观察到的结论。

(d) SAM 中的 $\rho$ 参数对优化结果有何影响？$\rho$ 过大或过小分别会导致什么问题？

---

## 练习答案

### 练习 19.1 答案

**(a)** 求解临界点：

$$\frac{\partial f}{\partial x} = 2x = 0, \quad \frac{\partial f}{\partial y} = -2y = 0$$

唯一临界点为 $(x^*, y^*) = (0, 0)$。

**(b)** Hessian 矩阵：

$$\mathbf{H} = \begin{pmatrix} \frac{\partial^2 f}{\partial x^2} & \frac{\partial^2 f}{\partial x \partial y} \\ \frac{\partial^2 f}{\partial y \partial x} & \frac{\partial^2 f}{\partial y^2} \end{pmatrix} = \begin{pmatrix} 2 & 0 \\ 0 & -2 \end{pmatrix}$$

特征值为 $\lambda_1 = 2 > 0$，$\lambda_2 = -2 < 0$，存在正负特征值，因此 $(0, 0)$ 是**鞍点**。

**(c)** 从 $(0.1, 0.0)$ 出发，梯度为 $(2 \times 0.1, -2 \times 0.0) = (0.2, 0.0)$，梯度下降更新：

$$\begin{pmatrix} x_1 \\ y_1 \end{pmatrix} = \begin{pmatrix} 0.1 \\ 0 \end{pmatrix} - 0.1 \begin{pmatrix} 0.2 \\ 0 \end{pmatrix} = \begin{pmatrix} 0.08 \\ 0 \end{pmatrix}$$

后续迭代中 $y$ 始终保持 0，$x$ 收缩到 0。算法沿 $x$ 方向（正曲率方向）收敛，但停在了 $y = 0$ 平面（负曲率方向无扰动）。这体现了**鞍点的吸引性**：若初始点恰好在零梯度的负曲率方向上，梯度下降会收敛到鞍点而非逃逸。实践中，数值扰动或随机梯度噪声使得负曲率方向上的分量不为精确零，从而使算法最终逃逸。

---

### 练习 19.2 答案

**(a)** 计算二阶导数：

$$\mathcal{L}_1''(\theta) = 12(\theta-1)^2 \Rightarrow \mathcal{L}_1''(1) = 0$$

$$\mathcal{L}_2''(\theta) = 2 \Rightarrow \mathcal{L}_2''(1) = 2$$

$\mathcal{L}_1$ 在 $\theta^* = 1$ 处 Hessian 为 0（平坦极小），$\mathcal{L}_2$ 的 Hessian 为 2（更尖锐）。从 Hessian 度量看，$\mathcal{L}_1$ 更平坦。

**(b)** $\epsilon$-尖锐度计算（$\epsilon = 0.1$）：

$$\mathcal{L}_1(1 \pm 0.1) = (0.1)^4 = 0.0001, \quad \Delta \mathcal{L}_1 = 0.0001$$

$$\mathcal{L}_2(1 \pm 0.1) = (0.1)^2 = 0.01, \quad \Delta \mathcal{L}_2 = 0.01$$

$\mathcal{L}_2$ 的损失上升 $0.01$，远大于 $\mathcal{L}_1$ 的 $0.0001$，与 Hessian 度量结论一致：$\mathcal{L}_1$ 极小更平坦。

**(c)** $\mathcal{L}_1''(1) = 12(\theta-1)^2 |_{\theta=1} = 0$，Hessian 为零意味着**退化临界点**，需要更高阶导数（四阶导数 $\mathcal{L}_1^{(4)}(1) = 24 > 0$）才能确认是极小点。退化临界点即满足定义，$\mathcal{L}_1$ 在 $\theta = 1$ 处确为退化极小点（梯度为零且 Hessian 半正定，且该点实为全局最小）。

---

### 练习 19.3 答案

**(a)** $n$ 维情形下，所有 $n$ 个特征值均为正的概率为 $p^n = (0.5)^n$：

| $n$ | 概率 $p^n$ |
|---|---|
| 1 | $0.5$ |
| 10 | $\approx 9.8 \times 10^{-4}$ |
| 100 | $\approx 7.9 \times 10^{-31}$ |
| $10^6$ | $\approx 10^{-301030}$（实际为零） |

**(b)** 证明：

一个随机临界点**不是**鞍点（即为纯极大或纯极小）的概率为：

$$P(\text{非鞍点}) = P(\text{全正}) + P(\text{全负}) = (0.5)^n + (0.5)^n = 2 \cdot (0.5)^n = (0.5)^{n-1}$$

当 $n \to \infty$，$P(\text{非鞍点}) = (0.5)^{n-1} \to 0$。

因此 $P(\text{鞍点}) = 1 - (0.5)^{n-1} \to 1$，即几乎所有临界点都是鞍点。

**(c)** 根据 Dauphin et al. 的理论，高损失临界点几乎均为鞍点，低损失临界点才接近局部极小。$\mathcal{L} = 0.5$ 相对较高（具体取决于任务，但在深度网络训练过程中中期损失），该点更可能是**鞍点**而非局部极小。原因在于：高维空间中，随机临界点是纯极小的概率指数小；高损失区域中的临界点通常具有较多负特征值（高逃逸指数），符合鞍点特征。优化器应继续迭代而非认为已收敛。

---

### 练习 19.4 答案

**(a)** 变量替换 $w = w'/c$：

$$\mathcal{L}'(w') = \left(\frac{w'}{c} x - y\right)^2$$

验证：$\mathcal{L}'(w'/c \cdot c) = \mathcal{L}'(w') = (wx - y)^2 = \mathcal{L}(w)$。即参数 $w$ 和 $w' = cw$ 对应相同的函数，但在参数空间是不同的点。

**(b)** $w^* = 2$，$d = 0.3$。Filter Normalization 将 $d$ 归一化到与 $w^*$ 同范数：

$$\hat{d} = \frac{d}{|d|} \cdot |w^*| = \frac{0.3}{0.3} \cdot 2 = 2.0$$

**(c)** 设原始参数 $w^* = 2$，尺度变换后 $w'^* = 10 \times 2 = 20$，但代表同一个函数。

- **不做归一化**，$\alpha = 0.1$：
  - 原始参数化：$w^* + 0.1 \times 0.3 = 2.03$，损失 $\mathcal{L}(2.03) = (2.03x-y)^2$
  - 尺度化参数：$w'^* + 0.1 \times 0.3 = 20.03$，等价 $w = 2.003$，损失不同
  - **两种参数化给出不同损失值**，可视化不一致

- **做 Filter Normalization** 后（方向范数 = 参数范数）：
  - 原始：$\hat{d} = 2.0$，$\alpha = 0.1$：$w = 2 + 0.2 = 2.2$
  - 尺度化：$\hat{d}' = 20.0$，$\alpha = 0.1$：$w' = 20 + 2.0 = 22.0$，等价 $w = 2.2$
  - **两种参数化给出相同损失值**，可视化一致

因此 Filter Normalization 方法对尺度重参数化是**鲁棒**的，而原始随机方向不具备此性质。

---

### 练习 19.5 答案

**(a)** SAM 优化器实现：

```python
class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization 优化器"""

    def __init__(self, params, base_optimizer_cls, rho: float = 0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        # 初始化基础优化器
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        """
        第一步：计算扰动方向 ε_hat，将参数移动到 θ + ε_hat
        """
        # 计算梯度的全局 L2 范数
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None:
                    continue
                # 保存原始参数
                self.state[p]['old_p'] = p.data.clone()
                # 扰动：p <- p + ρ * grad / ||grad||
                e_w = p.grad * scale
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        """
        第二步：还原原始参数，用扰动处的梯度更新
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # 还原原始参数
                p.data = self.state[p]['old_p']
        # 用基础优化器做参数更新
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self) -> torch.Tensor:
        """计算所有参数梯度的全局 L2 范数"""
        shared_device = self.param_groups[0]['params'][0].device
        norms = [
            p.grad.norm(p=2).to(shared_device)
            for group in self.param_groups
            for p in group['params']
            if p.grad is not None
        ]
        return torch.stack(norms).norm(p=2)


def train_with_sam(model, train_loader, epochs=10, lr=0.01,
                   rho=0.05, device='cpu'):
    """使用 SAM 优化器训练"""
    model = model.to(device)
    base_opt = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_opt,
                    rho=rho, lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # 第一次前向-反向传播：计算在 θ+ε 处的梯度
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # 第二次前向-反向传播：在 θ+ε 处计算梯度
            criterion(model(batch_x), batch_y).backward()
            optimizer.second_step(zero_grad=True)

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}]  Loss: {total_loss/len(train_loader):.4f}")

    return model
```

**(b)** SGD vs SAM 泛化比较（预期结果）：

在 CIFAR-10 的典型实验中：

| 方法 | 训练损失 | 测试损失 | 泛化差距 |
|---|---|---|---|
| SGD | ~0.05 | ~0.85 | ~0.80 |
| SAM | ~0.10 | ~0.72 | ~0.62 |

SAM 的训练损失略高（因为优化目标是最坏邻域损失），但测试损失和泛化差距更小，体现了平坦极小的泛化优势。

**(c)** 预期可视化结论：SAM 找到的极小点附近损失曲线更为**宽平**，即曲线底部的平坦区域更宽，两侧上升更缓慢；SGD 的损失曲线底部更窄更尖。这定性验证了 SAM 倾向于收敛到平坦极小的理论预期。

**(d)** $\rho$ 的影响分析：

- **$\rho$ 过小**（如 $\rho = 0.001$）：扰动幅度太小，几乎退化为标准 SGD，失去探索尖锐方向的能力，泛化提升有限
- **$\rho$ 过大**（如 $\rho = 1.0$）：扰动幅度过大，扰动点可能远离当前参数，计算出的梯度不再反映当前参数附近的曲率信息，训练不稳定甚至发散
- **推荐范围**：$\rho \in [0.01, 0.2]$，通常 $\rho = 0.05$ 是较好的起点。对于更深/更宽的网络，可能需要适当增大 $\rho$

实践建议：将 $\rho$ 作为超参数在验证集上搜索，并与学习率协同调整。
