# 第24章：概率图模型

> **定位**：本章是经典概率论与现代深度学习的桥梁。概率图模型将图论的结构化表示与概率论的不确定性推理融为一体，不仅统一了朴素贝叶斯、隐马尔可夫模型、高斯混合模型等经典方法，也为变分自编码器（VAE）等深度生成模型提供了理论基础。

---

## 学习目标

读完本章，你将能够：

1. 理解概率图模型的核心思想——用图结构编码随机变量之间的条件独立性，并利用因子分解大幅降低计算复杂度
2. 掌握贝叶斯网络（有向图）的构建规则、d-分离准则以及精确推断与近似推断方法
3. 理解马尔可夫随机场（无向图）的势函数表示、Gibbs分布及其与贝叶斯网络的联系与区别
4. 推导EM算法的数学原理，并能将其应用于高斯混合模型（GMM）等隐变量模型的参数估计
5. 理解变分推断的核心思想——用ELBO下界将推断问题转化为优化问题，并认识到VAE正是变分推断与深度神经网络的结合

---

## 24.1 图模型概述

### 24.1.1 为什么需要图模型

设想一个包含 $n$ 个二值随机变量 $X_1, X_2, \ldots, X_n$ 的联合分布。完整表示这个分布需要 $2^n - 1$ 个参数——当 $n = 100$ 时，这是一个天文数字，完全无法存储和计算。

**核心洞察**：现实世界中，大多数变量并不直接相互依赖。一个人是否感冒，与遥远城市的股票价格几乎无关。如果能将这些"局部依赖"结构显式地编码进模型，就能用远少于 $2^n$ 的参数表示联合分布。

**概率图模型（Probabilistic Graphical Model, PGM）**正是这一思想的形式化：

$$\text{图} G = (V, E) \quad \text{其中节点} V \text{对应随机变量，边} E \text{编码依赖关系}$$

### 24.1.2 条件独立性

**条件独立性**是图模型的核心概念。若给定 $Z$ 后，$X$ 与 $Y$ 独立，记作：

$$X \perp\!\!\!\perp Y \mid Z$$

等价地：

$$P(X, Y \mid Z) = P(X \mid Z) \cdot P(Y \mid Z)$$

**链式法则的因子分解**：任意联合分布都可以写成：

$$P(X_1, X_2, \ldots, X_n) = \prod_{i=1}^{n} P(X_i \mid X_1, \ldots, X_{i-1})$$

图模型的目标是利用条件独立性简化每个条件因子，使得每个 $X_i$ 只依赖于其**父节点集合** $\text{Pa}(X_i)$（有向图）或**邻居集合**（无向图）。

### 24.1.3 两大类图模型

| 特征 | 贝叶斯网络（有向图）| 马尔可夫随机场（无向图）|
|------|-------------------|------------------------|
| 边的类型 | 有向边（DAG） | 无向边 |
| 因子分解 | 条件概率 $P(X_i \mid \text{Pa}(X_i))$ | 势函数 $\psi_c(X_c)$ |
| 典型应用 | 因果推断、贝叶斯网络 | 图像分割、社交网络 |
| 归一化 | 自动满足 | 需要配分函数 $Z$ |

---

## 24.2 贝叶斯网络（有向图模型）

### 24.2.1 定义与因子分解

**贝叶斯网络**是一个有向无环图（DAG），其中：
- 每个节点 $X_i$ 对应一个随机变量
- 有向边 $X_j \to X_i$ 表示 $X_j$ 是 $X_i$ 的"父节点"

**联合分布的因子分解**：

$$\boxed{P(X_1, X_2, \ldots, X_n) = \prod_{i=1}^{n} P(X_i \mid \text{Pa}(X_i))}$$

其中 $\text{Pa}(X_i)$ 是节点 $X_i$ 的父节点集合（若无父节点则为先验 $P(X_i)$）。

**例：学生成绩模型**

考虑变量：课程难度 $D$、学生智力 $I$、考试成绩 $G$、推荐信 $L$、SAT分数 $S$。

图结构：$D \to G \leftarrow I \to S$，$G \to L$

联合分布：

$$P(D, I, G, L, S) = P(D) \cdot P(I) \cdot P(G \mid D, I) \cdot P(L \mid G) \cdot P(S \mid I)$$

原本需要 $2^5 - 1 = 31$ 个参数，因子分解后仅需少量参数。

### 24.2.2 三种基本连接结构

理解贝叶斯网络中信息"流动"的规律，关键是分析三种基本连接模式：

**1. 链式结构（Chain）**：$X \to Y \to Z$

给定 $Y$ 后，$X$ 与 $Z$ 条件独立：

$$P(X, Z \mid Y) = P(X \mid Y) \cdot P(Z \mid Y) \implies X \perp\!\!\!\perp Z \mid Y$$

**2. 共因结构（Fork）**：$X \leftarrow Y \to Z$

给定 $Y$ 后，$X$ 与 $Z$ 条件独立。$Y$ 是 $X$ 和 $Z$ 的公共原因，观测 $Y$ "阻断"了 $X$ 与 $Z$ 的关联。

**3. 碰撞结构（v-structure / Collider）**：$X \to Y \leftarrow Z$

- 未观测 $Y$：$X$ 与 $Z$ 独立（$X \perp\!\!\!\perp Z$）
- 观测 $Y$（或其后代）：$X$ 与 $Z$ 变得**相关**（解释消去效应）

这是与前两种结构相反的行为，也是 d-分离中最微妙的部分。

### 24.2.3 d-分离准则

**d-分离（directional separation）**是判断贝叶斯网络中条件独立性的通用算法。

**定义**：在有向图中，给定观测集合 $\mathbf{Z}$，若所有连接 $X$ 与 $Y$ 的路径都被"阻断"，则称 $X$ 与 $Y$ 被 $\mathbf{Z}$ d-分离，记作 $(X \perp\!\!\!\perp Y \mid \mathbf{Z})_G$。

**路径被阻断的条件**（路径上存在节点 $m$）：

| 结构类型 | 阻断条件 |
|---------|---------|
| 链 $\cdots \to m \to \cdots$ | $m \in \mathbf{Z}$ |
| 分叉 $\cdots \leftarrow m \to \cdots$ | $m \in \mathbf{Z}$ |
| 碰撞 $\cdots \to m \leftarrow \cdots$ | $m \notin \mathbf{Z}$ 且 $m$ 的后代 $\notin \mathbf{Z}$ |

**定理（Markov性质）**：若图 $G$ 中 $X$ 与 $Y$ 被 $\mathbf{Z}$ d-分离，则在满足图 $G$ 的所有分布中，$X \perp\!\!\!\perp Y \mid \mathbf{Z}$ 成立。

### 24.2.4 精确推断：变量消元法

贝叶斯网络的核心任务是**推断**：给定部分观测，计算其余变量的后验分布。

**变量消元（Variable Elimination）**通过逐步边缘化（积分消除）未观测变量来计算目标概率。

以链式模型 $A \to B \to C$ 为例，计算 $P(A \mid C = c)$：

$$P(A \mid C = c) \propto \sum_B P(A) \cdot P(B \mid A) \cdot P(C = c \mid B)$$

关键技巧：先计算 $\tau(B) = P(C = c \mid B)$，再计算 $\sum_B P(B \mid A) \cdot \tau(B)$，避免重复计算。

对于树形结构，变量消元等价于**置信传播（Belief Propagation）**，复杂度为 $O(n \cdot k^2)$，其中 $k$ 为变量的状态数。对于有环图，需使用**循环置信传播**（近似推断）。

---

## 24.3 马尔可夫随机场（无向图模型）

### 24.3.1 定义与Gibbs分布

**马尔可夫随机场（Markov Random Field, MRF）**，又称**马尔可夫网络**，使用无向图表示变量间的对称依赖关系。

**局部Markov性质**：给定邻居节点集合 $\mathcal{N}(X_i)$，$X_i$ 与其余非邻居节点条件独立：

$$X_i \perp\!\!\!\perp \mathbf{X}_{\text{rest}} \mid \mathbf{X}_{\mathcal{N}(i)}$$

**团（Clique）**：图中完全连接的子集（任意两节点之间都有边）。

**Hammersley-Clifford定理**：满足局部Markov性质的正分布，可以表示为**极大团上势函数的乘积**：

$$\boxed{P(\mathbf{X}) = \frac{1}{Z} \prod_{c \in \mathcal{C}} \psi_c(\mathbf{X}_c)}$$

其中：
- $\psi_c(\mathbf{X}_c) > 0$ 是定义在团 $c$ 上的**势函数（potential function）**
- $Z = \sum_{\mathbf{X}} \prod_c \psi_c(\mathbf{X}_c)$ 是**配分函数（partition function）**，用于归一化
- $\mathcal{C}$ 是所有极大团的集合

**Gibbs分布**：将势函数写成能量函数的指数形式：

$$P(\mathbf{X}) = \frac{1}{Z} \exp\left(-\sum_{c} E_c(\mathbf{X}_c)\right) = \frac{1}{Z} \exp(-E(\mathbf{X}))$$

其中 $E(\mathbf{X}) = \sum_c E_c(\mathbf{X}_c)$ 称为**能量函数**。

### 24.3.2 与贝叶斯网络的比较

**表达能力**：两者的表达能力不完全相同，各有擅长的独立性结构。存在既不能用贝叶斯网络也不能用MRF精确表示的分布。

**将贝叶斯网络转化为MRF（道德化）**：

1. 为每个节点的父节点两两相连（"婚姻化"）
2. 将所有有向边替换为无向边

这个过程称为**道德化（moralization）**，得到的图称为**道德图**。注意：道德化可能引入新的独立性损失。

**配分函数的计算**：MRF最大的挑战是配分函数 $Z$ 的计算。对于离散变量，精确计算需要对所有状态求和，复杂度为指数级。这是MRF推断困难的根本原因，也是为何需要MCMC、变分推断等近似方法。

### 24.3.3 条件随机场

**条件随机场（Conditional Random Field, CRF）**是MRF的判别式变体，直接建模条件分布 $P(\mathbf{Y} \mid \mathbf{X})$：

$$P(\mathbf{Y} \mid \mathbf{X}) = \frac{1}{Z(\mathbf{X})} \exp\left(\sum_c \psi_c(\mathbf{Y}_c, \mathbf{X})\right)$$

CRF在序列标注（如命名实体识别）中取得了重要成果，是联结图模型与深度学习的早期尝试。

---

## 24.4 隐变量模型与EM算法

### 24.4.1 隐变量模型

许多真实问题中，我们观测到数据 $\mathbf{x}$，但生成数据的过程涉及**隐变量（latent variable）** $\mathbf{z}$。

**边缘似然（Evidence）**：

$$P(\mathbf{x} \mid \theta) = \int P(\mathbf{x}, \mathbf{z} \mid \theta) \, d\mathbf{z} = \int P(\mathbf{x} \mid \mathbf{z}, \theta) P(\mathbf{z} \mid \theta) \, d\mathbf{z}$$

直接最大化边缘似然通常很难，因为积分没有解析解。EM算法提供了一种迭代优化方案。

### 24.4.2 EM算法的推导

**目标**：最大化对数边缘似然 $\log P(\mathbf{x} \mid \theta)$。

对于任意关于 $\mathbf{z}$ 的分布 $q(\mathbf{z})$，利用Jensen不等式：

$$\log P(\mathbf{x} \mid \theta) = \log \int P(\mathbf{x}, \mathbf{z} \mid \theta) \, d\mathbf{z}$$

$$= \log \int q(\mathbf{z}) \frac{P(\mathbf{x}, \mathbf{z} \mid \theta)}{q(\mathbf{z})} \, d\mathbf{z}$$

$$\geq \int q(\mathbf{z}) \log \frac{P(\mathbf{x}, \mathbf{z} \mid \theta)}{q(\mathbf{z})} \, d\mathbf{z} \quad \text{（Jensen不等式）}$$

$$= \underbrace{\mathbb{E}_{q(\mathbf{z})}[\log P(\mathbf{x}, \mathbf{z} \mid \theta)]}_{\text{期望完全对数似然}} + \underbrace{H[q(\mathbf{z})]}_{\text{熵}} := \mathcal{L}(q, \theta)$$

等号成立条件：$q(\mathbf{z}) = P(\mathbf{z} \mid \mathbf{x}, \theta)$（当 $q$ 等于后验分布时）。

**EM算法的两步迭代**：

$$\boxed{\text{E步（期望步）}：q^{(t+1)}(\mathbf{z}) = P(\mathbf{z} \mid \mathbf{x}, \theta^{(t)})}$$

$$\boxed{\text{M步（最大化步）}：\theta^{(t+1)} = \arg\max_\theta \mathbb{E}_{q^{(t+1)}(\mathbf{z})}[\log P(\mathbf{x}, \mathbf{z} \mid \theta)]}$$

**单调性保证**：EM算法保证对数似然单调不减：

$$\log P(\mathbf{x} \mid \theta^{(t+1)}) \geq \log P(\mathbf{x} \mid \theta^{(t)})$$

证明：E步使下界 $\mathcal{L}$ 紧（消除了KL散度间隙），M步提升下界，因此总体对数似然不减。

### 24.4.3 高斯混合模型（GMM）的EM算法

**模型定义**：GMM假设数据由 $K$ 个高斯分量混合生成：

$$P(\mathbf{x} \mid \theta) = \sum_{k=1}^{K} \pi_k \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

其中 $\pi_k \geq 0$，$\sum_k \pi_k = 1$，$\theta = \{\pi_k, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k\}_{k=1}^K$。

隐变量 $z_i \in \{1, \ldots, K\}$ 表示第 $i$ 个数据点属于哪个分量，$P(z_i = k) = \pi_k$。

**E步（计算后验"责任"）**：

$$r_{ik} = P(z_i = k \mid \mathbf{x}_i, \theta^{(t)}) = \frac{\pi_k^{(t)} \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k^{(t)}, \boldsymbol{\Sigma}_k^{(t)})}{\sum_{j=1}^K \pi_j^{(t)} \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_j^{(t)}, \boldsymbol{\Sigma}_j^{(t)})}$$

**M步（更新参数）**：

令 $N_k = \sum_{i=1}^N r_{ik}$（分量 $k$ 的有效样本数），则：

$$\boldsymbol{\mu}_k^{(t+1)} = \frac{1}{N_k} \sum_{i=1}^N r_{ik} \mathbf{x}_i$$

$$\boldsymbol{\Sigma}_k^{(t+1)} = \frac{1}{N_k} \sum_{i=1}^N r_{ik} (\mathbf{x}_i - \boldsymbol{\mu}_k^{(t+1)})(\mathbf{x}_i - \boldsymbol{\mu}_k^{(t+1)})^\top$$

$$\pi_k^{(t+1)} = \frac{N_k}{N}$$

**EM vs K-means**：K-means可以看作GMM EM算法的硬分配（hard assignment）版本——每个数据点被强制分配到最近的中心，而非保留软概率权重 $r_{ik}$。

---

## 24.5 变分推断简介

### 24.5.1 推断问题的困难性

后验推断的核心任务是计算：

$$P(\mathbf{z} \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid \mathbf{z}) P(\mathbf{z})}{P(\mathbf{x})}$$

难点在于分母 $P(\mathbf{x}) = \int P(\mathbf{x} \mid \mathbf{z}) P(\mathbf{z}) \, d\mathbf{z}$ 通常无法解析计算。

两大近似推断方法：
- **MCMC（马尔可夫链蒙特卡洛）**：通过采样近似后验，渐进精确但计算慢
- **变分推断（Variational Inference）**：将推断转化为优化，速度快但有近似误差

### 24.5.2 变分推断框架

**核心思想**：在某个"简单"分布族 $\mathcal{Q} = \{q_\phi(\mathbf{z})\}$ 中，寻找最接近真实后验 $P(\mathbf{z} \mid \mathbf{x})$ 的分布：

$$q^*_\phi(\mathbf{z}) = \arg\min_{q \in \mathcal{Q}} \text{KL}(q_\phi(\mathbf{z}) \| P(\mathbf{z} \mid \mathbf{x}))$$

**KL散度展开**：

$$\text{KL}(q_\phi(\mathbf{z}) \| P(\mathbf{z} \mid \mathbf{x})) = \mathbb{E}_{q_\phi}\left[\log \frac{q_\phi(\mathbf{z})}{P(\mathbf{z} \mid \mathbf{x})}\right]$$

$$= \mathbb{E}_{q_\phi}[\log q_\phi(\mathbf{z})] - \mathbb{E}_{q_\phi}[\log P(\mathbf{x}, \mathbf{z})] + \log P(\mathbf{x})$$

由于 $\text{KL} \geq 0$，得到：

$$\log P(\mathbf{x}) \geq \underbrace{\mathbb{E}_{q_\phi}[\log P(\mathbf{x}, \mathbf{z})] - \mathbb{E}_{q_\phi}[\log q_\phi(\mathbf{z})]}_{\text{证据下界（ELBO）}}$$

### 24.5.3 证据下界（ELBO）

**ELBO（Evidence Lower BOund）**是变分推断的核心量：

$$\boxed{\mathcal{L}(\phi) = \mathbb{E}_{q_\phi(\mathbf{z})}[\log P(\mathbf{x}, \mathbf{z})] - \mathbb{E}_{q_\phi(\mathbf{z})}[\log q_\phi(\mathbf{z})]}$$

等价分解：

$$\mathcal{L}(\phi) = \mathbb{E}_{q_\phi(\mathbf{z})}[\log P(\mathbf{x} \mid \mathbf{z})] - \text{KL}(q_\phi(\mathbf{z}) \| P(\mathbf{z}))$$

- 第一项：**重建项**，鼓励 $q_\phi(\mathbf{z})$ 找到能解释观测数据的隐变量值
- 第二项：**正则化项**，鼓励近似后验接近先验，防止过拟合

**最大化ELBO等价于**：
1. 最小化 $\text{KL}(q_\phi \| P(\cdot \mid \mathbf{x}))$（推断目标）
2. 最大化 $\log P(\mathbf{x})$（学习目标）

### 24.5.4 均场近似

**均场（Mean-Field）变分推断**假设 $q_\phi(\mathbf{z})$ 在各维度上完全分解：

$$q_\phi(\mathbf{z}) = \prod_{j} q_j(z_j)$$

在此假设下，可以推导出每个因子的最优形式（坐标上升）：

$$q_j^*(z_j) \propto \exp\left(\mathbb{E}_{q_{-j}}[\log P(\mathbf{x}, \mathbf{z})]\right)$$

其中 $\mathbb{E}_{q_{-j}}$ 表示对除 $z_j$ 外的所有变量取期望。

---

## 本章小结

本章系统介绍了概率图模型的核心理论框架：

1. **图模型的统一视角**：通过图结构编码条件独立性，实现联合分布的高效因子分解，是处理高维概率分布的核心工具

2. **贝叶斯网络**：基于有向无环图，因子分解为条件概率乘积。d-分离准则提供了判断条件独立性的算法，三种基本连接结构（链、分叉、碰撞）揭示了信息流动的本质

3. **马尔可夫随机场**：基于无向图，通过势函数和配分函数表示联合分布。Hammersley-Clifford定理建立了图结构与分布因子分解的等价性

4. **EM算法**：通过引入辅助分布 $q(\mathbf{z})$ 构造对数似然的下界，交替优化以单调提升似然。在GMM上的应用展示了软分配与参数估计的优雅交互

5. **变分推断**：将难以计算的后验推断转化为ELBO最大化问题，均场近似和重参数化技巧使其能与深度学习结合，催生了VAE等现代生成模型

---

## 深度学习应用：VAE、GMM与隐变量模型

### 应用背景

变分自编码器（VAE）是变分推断与深度神经网络的融合：
- **编码器** $q_\phi(\mathbf{z} \mid \mathbf{x})$：用神经网络参数化近似后验（变分推断中的 $q$）
- **解码器** $P_\theta(\mathbf{x} \mid \mathbf{z})$：用神经网络参数化似然函数
- **训练目标**：最大化ELBO $= \mathbb{E}_{q_\phi}[\log P_\theta(\mathbf{x} \mid \mathbf{z})] - \text{KL}(q_\phi(\mathbf{z} \mid \mathbf{x}) \| P(\mathbf{z}))$

**重参数化技巧（Reparameterization Trick）**使得梯度可以通过采样操作反向传播：

$$\mathbf{z} = \boldsymbol{\mu}_\phi(\mathbf{x}) + \boldsymbol{\sigma}_\phi(\mathbf{x}) \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

这样梯度 $\nabla_\phi \mathbb{E}_{q_\phi}[\cdot]$ 可以通过 $\boldsymbol{\epsilon}$ 的固定分布计算，而非通过随机节点。

### 完整PyTorch实现

```python
"""
概率图模型深度学习应用：VAE 与 GMM

包含：
1. 高斯混合模型（GMM）的 EM 算法实现
2. 变分自编码器（VAE）的 PyTorch 实现
3. 两者在 MNIST 数据集上的对比实验
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


# ============================================================
# Part 1: 高斯混合模型 (GMM) 的 EM 算法
# ============================================================

class GaussianMixture:
    """
    高斯混合模型：使用 EM 算法进行参数估计

    模型：P(x) = sum_k pi_k * N(x | mu_k, Sigma_k)
    隐变量：z_i in {1, ..., K}，表示第 i 个点的分量归属
    """

    def __init__(self, n_components: int = 3, n_iter: int = 100,
                 tol: float = 1e-4, random_state: int = 42):
        self.K = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.rng = np.random.RandomState(random_state)

    def fit(self, X: np.ndarray) -> 'GaussianMixture':
        """EM 算法拟合 GMM"""
        N, D = X.shape

        # 初始化参数（随机选取数据点作为均值）
        idx = self.rng.choice(N, self.K, replace=False)
        self.mu = X[idx].copy()                         # (K, D)
        self.Sigma = np.array([np.eye(D)] * self.K)    # (K, D, D)
        self.pi = np.ones(self.K) / self.K              # (K,)

        log_likelihoods = []

        for iteration in range(self.n_iter):
            # ---- E步：计算每个数据点属于每个分量的后验概率 ----
            # r[i, k] = P(z_i = k | x_i, theta)
            r = self._e_step(X)   # (N, K)

            # ---- M步：更新参数 ----
            self._m_step(X, r)

            # 计算对数似然（用于收敛判断）
            log_lik = self._log_likelihood(X)
            log_likelihoods.append(log_lik)

            if iteration > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < self.tol:
                print(f"  GMM 收敛于第 {iteration+1} 次迭代，"
                      f"对数似然 = {log_lik:.4f}")
                break

        self.log_likelihoods_ = log_likelihoods
        return self

    def _e_step(self, X: np.ndarray) -> np.ndarray:
        """
        E步：计算"责任"矩阵

        r[i, k] = pi_k * N(x_i | mu_k, Sigma_k) / sum_j pi_j * N(x_i | mu_j, Sigma_j)
        """
        N = X.shape[0]
        log_r = np.zeros((N, self.K))

        for k in range(self.K):
            log_r[:, k] = np.log(self.pi[k] + 1e-300) + \
                          self._log_gaussian(X, self.mu[k], self.Sigma[k])

        # 数值稳定的 log-sum-exp
        log_r -= log_r.max(axis=1, keepdims=True)
        r = np.exp(log_r)
        r /= r.sum(axis=1, keepdims=True)   # 归一化
        return r

    def _m_step(self, X: np.ndarray, r: np.ndarray):
        """
        M步：用加权样本更新参数

        mu_k = sum_i r_ik * x_i / N_k
        Sigma_k = sum_i r_ik * (x_i - mu_k)(x_i - mu_k)^T / N_k
        pi_k = N_k / N
        """
        N, D = X.shape
        N_k = r.sum(axis=0)   # (K,)，每个分量的有效样本数

        for k in range(self.K):
            # 更新均值
            self.mu[k] = (r[:, k:k+1] * X).sum(axis=0) / N_k[k]

            # 更新协方差矩阵
            diff = X - self.mu[k]                              # (N, D)
            self.Sigma[k] = (r[:, k:k+1] * diff).T @ diff / N_k[k]
            self.Sigma[k] += 1e-6 * np.eye(D)                  # 正则化，防止奇异

        # 更新混合系数
        self.pi = N_k / N

    def _log_gaussian(self, X: np.ndarray, mu: np.ndarray,
                      Sigma: np.ndarray) -> np.ndarray:
        """计算多元高斯分布的对数概率密度"""
        D = X.shape[1]
        diff = X - mu
        sign, log_det = np.linalg.slogdet(Sigma)
        inv_Sigma = np.linalg.inv(Sigma)
        mahal = np.einsum('nd,dd,nd->n', diff, inv_Sigma, diff)   # 马氏距离
        return -0.5 * (D * np.log(2 * np.pi) + log_det + mahal)

    def _log_likelihood(self, X: np.ndarray) -> float:
        """计算数据集的总对数似然"""
        N = X.shape[0]
        log_p = np.zeros((N, self.K))
        for k in range(self.K):
            log_p[:, k] = np.log(self.pi[k] + 1e-300) + \
                          self._log_gaussian(X, self.mu[k], self.Sigma[k])
        # log-sum-exp
        log_max = log_p.max(axis=1, keepdims=True)
        log_p_x = np.log(np.exp(log_p - log_max).sum(axis=1)) + log_max.squeeze()
        return log_p_x.sum()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测每个数据点最可能的分量标签"""
        r = self._e_step(X)
        return r.argmax(axis=1)

    def sample(self, n_samples: int = 100) -> np.ndarray:
        """从 GMM 中生成样本"""
        components = self.rng.choice(self.K, size=n_samples, p=self.pi)
        D = self.mu.shape[1]
        samples = np.zeros((n_samples, D))
        for k in range(self.K):
            mask = (components == k)
            nk = mask.sum()
            if nk > 0:
                samples[mask] = self.rng.multivariate_normal(
                    self.mu[k], self.Sigma[k], nk
                )
        return samples


# ============================================================
# Part 2: 变分自编码器 (VAE)
# ============================================================

class VAEEncoder(nn.Module):
    """
    VAE 编码器：实现近似后验 q_phi(z | x)

    输出：隐变量的均值 mu 和对数方差 log_var
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 512,
                 latent_dim: int = 20):
        super().__init__()
        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var


class VAEDecoder(nn.Module):
    """
    VAE 解码器：实现生成分布 P_theta(x | z)

    对于二值图像，输出 Bernoulli 参数（像素值）
    """

    def __init__(self, latent_dim: int = 20, hidden_dim: int = 512,
                 output_dim: int = 784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),   # 输出 [0,1] 范围的像素概率
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VAE(nn.Module):
    """
    变分自编码器（Variational Autoencoder）

    生成过程（解码）：
        z ~ P(z) = N(0, I)
        x ~ P_theta(x | z) = Bernoulli(f_theta(z))

    推断过程（编码）：
        q_phi(z | x) = N(mu_phi(x), diag(sigma^2_phi(x)))

    训练目标：最大化 ELBO
        L = E_q[log P_theta(x|z)] - KL(q_phi(z|x) || P(z))
          = 重建项 - KL 正则化项
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 512,
                 latent_dim: int = 20):
        super().__init__()
        self.encoder = VAEEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = VAEDecoder(latent_dim, hidden_dim, input_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu: torch.Tensor,
                       log_var: torch.Tensor) -> torch.Tensor:
        """
        重参数化技巧：z = mu + sigma * epsilon, epsilon ~ N(0, I)

        将随机性移到 epsilon，使得梯度可以通过 mu 和 log_var 反向传播
        """
        if self.training:
            std = torch.exp(0.5 * log_var)       # sigma = exp(log_var / 2)
            eps = torch.randn_like(std)            # epsilon ~ N(0, I)
            return mu + std * eps
        else:
            return mu   # 推断时直接使用均值

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播：编码 -> 重参数化采样 -> 解码"""
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var

    def elbo_loss(self, x: torch.Tensor, x_recon: torch.Tensor,
                  mu: torch.Tensor, log_var: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算 ELBO（取负号作为损失函数）

        ELBO = E_q[log P(x|z)] - KL(q(z|x) || P(z))

        重建项（BCE）：E_q[log P(x|z)] ≈ sum_i x_i log x_recon_i + (1-x_i) log (1-x_recon_i)

        KL 项（对角高斯有解析解）：
            KL(N(mu, diag(sigma^2)) || N(0, I))
            = -0.5 * sum_j (1 + log_var_j - mu_j^2 - exp(log_var_j))
        """
        # 重建损失（逐像素二元交叉熵，取均值）
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')

        # KL 散度（解析形式，无需采样）
        # KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # 总损失 = 重建损失 + KL 散度
        total_loss = recon_loss + kl_loss
        return total_loss, recon_loss, kl_loss

    @torch.no_grad()
    def generate(self, n_samples: int = 16,
                 device: str = 'cpu') -> torch.Tensor:
        """从先验 P(z) = N(0, I) 采样并解码"""
        z = torch.randn(n_samples, self.latent_dim).to(device)
        return self.decoder(z)

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """重建输入图像"""
        x_recon, _, _ = self.forward(x)
        return x_recon


# ============================================================
# Part 3: 训练与可视化
# ============================================================

def train_vae(model: VAE, train_loader: DataLoader,
              n_epochs: int = 10, lr: float = 1e-3,
              device: str = 'cpu') -> list:
    """训练 VAE 模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()

    history = []
    for epoch in range(n_epochs):
        total_loss = recon_total = kl_total = 0.0
        n_batches = 0

        for x, _ in train_loader:
            x = x.view(x.size(0), -1).to(device)     # 展平图像 (B, 784)

            optimizer.zero_grad()
            x_recon, mu, log_var = model(x)
            loss, recon_loss, kl_loss = model.elbo_loss(x, x_recon, mu, log_var)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            recon_total += recon_loss.item()
            kl_total += kl_loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_recon = recon_total / n_batches
        avg_kl = kl_total / n_batches
        history.append({'loss': avg_loss, 'recon': avg_recon, 'kl': avg_kl})

        print(f"Epoch [{epoch+1:2d}/{n_epochs}] "
              f"Loss: {avg_loss:.1f} "
              f"(Recon: {avg_recon:.1f}, KL: {avg_kl:.1f})")

    return history


def demo_gmm():
    """GMM EM 算法演示"""
    print("=" * 60)
    print("高斯混合模型 (GMM) - EM 算法演示")
    print("=" * 60)

    # 生成合成数据：3 个高斯分量
    rng = np.random.RandomState(42)
    centers = np.array([[0, 0], [5, 0], [2.5, 4]])
    X_list = [rng.multivariate_normal(c, np.eye(2) * 0.8, 200) for c in centers]
    X = np.vstack(X_list)
    y_true = np.repeat([0, 1, 2], 200)

    print(f"\n数据集：{X.shape[0]} 个点，{len(centers)} 个真实分量")

    # 拟合 GMM
    gmm = GaussianMixture(n_components=3, n_iter=100, tol=1e-6)
    gmm.fit(X)

    # 评估
    y_pred = gmm.predict(X)
    # 简单评估（不考虑标签排列）
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import confusion_matrix
    C = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-C)
    accuracy = C[row_ind, col_ind].sum() / len(y_true)
    print(f"聚类准确率：{accuracy:.3f}")

    print("\n估计的混合权重：", np.round(gmm.pi, 3))
    print("估计的均值：\n", np.round(gmm.mu, 3))

    return gmm, X


def demo_vae():
    """VAE 训练演示"""
    print("\n" + "=" * 60)
    print("变分自编码器 (VAE) - MNIST 演示")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备：{device}")

    # 加载 MNIST 数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True,
                                   transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,
                              num_workers=0)

    # 创建 VAE 模型
    vae = VAE(input_dim=784, hidden_dim=512, latent_dim=20)
    total_params = sum(p.numel() for p in vae.parameters())
    print(f"\nVAE 参数量：{total_params:,}")
    print(f"  编码器：输入(784) -> 隐层(512) -> 隐变量均值+方差(20+20)")
    print(f"  解码器：隐变量(20) -> 隐层(512) -> 输出(784)")

    # 训练
    print("\n开始训练（5 个 epoch，仅作演示）...")
    history = train_vae(vae, train_loader, n_epochs=5, lr=1e-3, device=device)

    return vae, history


def visualize_latent_space(vae: VAE, test_loader: DataLoader,
                           device: str = 'cpu'):
    """
    可视化 2D 隐空间（当 latent_dim=2 时）

    展示 VAE 学到的连续、结构化的隐空间表示
    """
    if vae.latent_dim != 2:
        print("注：隐空间可视化需要 latent_dim=2，当前跳过")
        return

    vae.eval()
    all_mu = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.view(x.size(0), -1).to(device)
            mu, _ = vae.encoder(x)
            all_mu.append(mu.cpu().numpy())
            all_labels.append(y.numpy())

    all_mu = np.concatenate(all_mu)
    all_labels = np.concatenate(all_labels)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(all_mu[:, 0], all_mu[:, 1],
                          c=all_labels, cmap='tab10', alpha=0.3, s=1)
    plt.colorbar(scatter, label='数字类别')
    plt.title('VAE 隐空间可视化（测试集）')
    plt.xlabel('$z_1$')
    plt.ylabel('$z_2$')
    plt.tight_layout()
    plt.savefig('vae_latent_space.png', dpi=150)
    print("隐空间可视化保存至 vae_latent_space.png")


if __name__ == '__main__':
    # 演示 1：GMM EM 算法
    try:
        from scipy.optimize import linear_sum_assignment
        from sklearn.metrics import confusion_matrix
        gmm, X_data = demo_gmm()
    except ImportError:
        print("提示：完整 GMM 演示需要 scipy 和 sklearn")
        # 简化版演示（不计算准确率）
        rng = np.random.RandomState(42)
        X_data = np.vstack([
            rng.multivariate_normal([0, 0], np.eye(2), 200),
            rng.multivariate_normal([5, 0], np.eye(2), 200),
            rng.multivariate_normal([2.5, 4], np.eye(2), 200),
        ])
        gmm = GaussianMixture(n_components=3).fit(X_data)

    # 演示 2：VAE
    print("\n提示：VAE 训练需要 torchvision 和 MNIST 数据集")
    print("如已安装，取消注释以下代码：")
    print("  vae, history = demo_vae()")

    # 快速验证 VAE 前向传播
    print("\n验证 VAE 模型结构：")
    vae_model = VAE(input_dim=784, hidden_dim=256, latent_dim=10)
    x_test = torch.randn(4, 784)
    x_recon, mu, log_var = vae_model(x_test)
    loss, recon, kl = vae_model.elbo_loss(
        torch.sigmoid(x_test), x_recon, mu, log_var
    )
    print(f"  输入形状：{x_test.shape}")
    print(f"  重建形状：{x_recon.shape}")
    print(f"  隐变量均值形状：{mu.shape}")
    print(f"  ELBO = -(重建损失 {recon.item():.1f} + KL散度 {kl.item():.1f})")
    print("  前向传播验证通过！")
```

---

## 练习题

**练习 24.1（贝叶斯网络因子分解）**

考虑以下贝叶斯网络，包含节点 $A, B, C, D, E$，其中边的结构为：
$A \to C$，$B \to C$，$C \to D$，$C \to E$

(1) 写出联合分布 $P(A, B, C, D, E)$ 的因子分解形式。

(2) 利用 d-分离准则，判断以下条件独立性是否成立：
   - $A \perp\!\!\!\perp B$（未观测任何节点）
   - $A \perp\!\!\!\perp B \mid C$
   - $D \perp\!\!\!\perp E \mid C$
   - $A \perp\!\!\!\perp D \mid C$

(3) 计算边缘分布 $P(D)$ 的表达式（以条件概率表示）。

---

**练习 24.2（d-分离的碰撞结构）**

考虑碰撞结构 $X \to Y \leftarrow Z$，其中 $X, Z$ 为二值变量，$P(X = 1) = P(Z = 1) = 0.5$，且 $Y = X \oplus Z$（异或）。

(1) 验证 $X$ 与 $Z$ 在不观测 $Y$ 时独立。

(2) 计算 $P(X = 1 \mid Y = 1)$ 和 $P(X = 1 \mid Y = 1, Z = 0)$，说明观测 $Y$ 后 $X$ 与 $Z$ 变得相关（即"解释消去"效应）。

(3) 用信息论中的互信息 $I(X; Z \mid Y)$ 定量描述这一依赖程度。

---

**练习 24.3（EM 算法推导）**

设观测数据 $\mathbf{x} = \{x_1, \ldots, x_N\}$ 来自参数为 $\theta$ 的隐变量模型，隐变量为 $\mathbf{z} = \{z_1, \ldots, z_N\}$。

(1) 写出对数边缘似然 $\log P(\mathbf{x} \mid \theta)$ 的下界（ELBO）的推导过程，明确使用 Jensen 不等式的步骤。

(2) 证明当 $q(\mathbf{z}) = P(\mathbf{z} \mid \mathbf{x}, \theta)$ 时，下界与对数似然相等（即间隙为零）。

(3) 证明 EM 算法的单调性：$\log P(\mathbf{x} \mid \theta^{(t+1)}) \geq \log P(\mathbf{x} \mid \theta^{(t)})$。

---

**练习 24.4（GMM 的 EM 算法）**

考虑一维 GMM：$P(x) = \pi_1 \mathcal{N}(x \mid \mu_1, \sigma^2) + \pi_2 \mathcal{N}(x \mid \mu_2, \sigma^2)$（两个分量共享方差 $\sigma^2$）。

(1) 写出 E 步：给定当前参数 $\{\pi_k^{(t)}, \mu_k^{(t)}\}$，计算责任 $r_{ik}$。

(2) 写出 M 步：在共享方差 $\sigma^2$ 的约束下，推导参数的更新公式。

(3) 如果两个分量完全重叠（$\mu_1^{(t)} = \mu_2^{(t)}$），EM 算法会如何演化？这说明 GMM 的什么问题？

---

**练习 24.5（VAE 与 ELBO）**

VAE 使用对角高斯近似后验 $q_\phi(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}(\mathbf{z} \mid \boldsymbol{\mu}_\phi(\mathbf{x}), \text{diag}(\boldsymbol{\sigma}^2_\phi(\mathbf{x})))$，先验 $P(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$。

(1) 推导 KL 散度 $\text{KL}(q_\phi(\mathbf{z} \mid \mathbf{x}) \| P(\mathbf{z}))$ 的解析表达式。

(2) 解释"重参数化技巧"的必要性：为什么不能直接对 $\mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})}[\log P_\theta(\mathbf{x} \mid \mathbf{z})]$ 关于 $\phi$ 求梯度？

(3) 描述 VAE 中"KL 坍塌（KL collapse）"现象：当 KL 项变为零时意味着什么？为什么这是一个问题？如何缓解？

---

## 练习答案

### 练习 24.1 解答

**(1) 因子分解**

根据贝叶斯网络因子分解公式，每个节点的概率只依赖于其父节点：

$$P(A, B, C, D, E) = P(A) \cdot P(B) \cdot P(C \mid A, B) \cdot P(D \mid C) \cdot P(E \mid C)$$

**(2) 条件独立性判断**

- **$A \perp\!\!\!\perp B$**：路径 $A \to C \leftarrow B$，中间节点 $C$ 是碰撞节点，且 $C$ 未被观测，路径被阻断。**成立** ✓

- **$A \perp\!\!\!\perp B \mid C$**：观测 $C$ 后，碰撞节点被"激活"，路径 $A \to C \leftarrow B$ 变为畅通。**不成立** ✗（观测子节点成绩会使父节点智力与难度产生关联）

- **$D \perp\!\!\!\perp E \mid C$**：路径 $D \leftarrow C \to E$，中间节点 $C$ 是分叉节点，观测 $C$ 阻断路径。**成立** ✓

- **$A \perp\!\!\!\perp D \mid C$**：路径 $A \to C \to D$，中间节点 $C$ 是链式节点，观测 $C$ 阻断路径。**成立** ✓

**(3) 边缘分布**

$$P(D) = \sum_C P(D \mid C) P(C) = \sum_C P(D \mid C) \sum_A \sum_B P(C \mid A, B) P(A) P(B)$$

---

### 练习 24.2 解答

**(1) 独立性验证**

$P(X, Z) = P(X) P(Z) = 0.25$，对所有 $(x, z)$ 组合均成立，因为 $X$ 和 $Z$ 是独立定义的。$P(X=1, Z=1) = 0.25 = P(X=1) \cdot P(Z=1)$，独立性成立。

**(2) 条件分布**

$Y = X \oplus Z$，所以 $Y = 1$ 当且仅当 $X \neq Z$，即 $(X=1, Z=0)$ 或 $(X=0, Z=1)$：

$$P(Y = 1) = 0.5$$

$$P(X = 1 \mid Y = 1) = \frac{P(X=1, Y=1)}{P(Y=1)} = \frac{P(X=1, Z=0)}{0.5} = \frac{0.25}{0.5} = 0.5$$

$$P(X = 1 \mid Y = 1, Z = 0) = \frac{P(X=1, Z=0, Y=1)}{P(Z=0, Y=1)} = \frac{P(X=1, Z=0)}{P(X=1, Z=0)} = 1$$

因此 $P(X=1 \mid Y=1) = 0.5$，而 $P(X=1 \mid Y=1, Z=0) = 1$：观测 $Y$ 后再观测 $Z$ 会完全确定 $X$，说明 $X$ 与 $Z$ 在给定 $Y$ 时**强相关**。

**(3) 条件互信息**

$$I(X; Z \mid Y) = \sum_{y} P(y) \sum_{x,z} P(x, z \mid y) \log \frac{P(x, z \mid y)}{P(x \mid y) P(z \mid y)} = 1 \text{ bit}$$

（当 $Y=1$ 时，给定 $Z$ 后 $X$ 完全确定；当 $Y=0$ 时同理，条件互信息为 1 比特。）

---

### 练习 24.3 解答

**(1) ELBO 推导**

$$\log P(\mathbf{x} \mid \theta) = \log \int P(\mathbf{x}, \mathbf{z} \mid \theta) \, d\mathbf{z}$$

引入任意分布 $q(\mathbf{z})$：

$$= \log \int q(\mathbf{z}) \frac{P(\mathbf{x}, \mathbf{z} \mid \theta)}{q(\mathbf{z})} \, d\mathbf{z}$$

由 Jensen 不等式（$\log$ 是凹函数，$\log \mathbb{E}[f] \geq \mathbb{E}[\log f]$）：

$$\geq \int q(\mathbf{z}) \log \frac{P(\mathbf{x}, \mathbf{z} \mid \theta)}{q(\mathbf{z})} \, d\mathbf{z} = \mathcal{L}(q, \theta)$$

**(2) 间隙为零的条件**

间隙为：

$$\log P(\mathbf{x} \mid \theta) - \mathcal{L}(q, \theta) = \text{KL}(q(\mathbf{z}) \| P(\mathbf{z} \mid \mathbf{x}, \theta))$$

当 $q(\mathbf{z}) = P(\mathbf{z} \mid \mathbf{x}, \theta)$ 时，$\text{KL} = 0$，间隙为零。

**(3) EM 单调性证明**

E步：令 $q^{(t)}(\mathbf{z}) = P(\mathbf{z} \mid \mathbf{x}, \theta^{(t)})$，此时 $\mathcal{L}(q^{(t)}, \theta^{(t)}) = \log P(\mathbf{x} \mid \theta^{(t)})$

M步：$\theta^{(t+1)} = \arg\max_\theta \mathcal{L}(q^{(t)}, \theta)$，所以 $\mathcal{L}(q^{(t)}, \theta^{(t+1)}) \geq \mathcal{L}(q^{(t)}, \theta^{(t)})$

因此：

$$\log P(\mathbf{x} \mid \theta^{(t+1)}) \geq \mathcal{L}(q^{(t)}, \theta^{(t+1)}) \geq \mathcal{L}(q^{(t)}, \theta^{(t)}) = \log P(\mathbf{x} \mid \theta^{(t)})$$

---

### 练习 24.4 解答

**(1) E步**

$$r_{i1} = \frac{\pi_1^{(t)} \mathcal{N}(x_i \mid \mu_1^{(t)}, \sigma^2)}{\pi_1^{(t)} \mathcal{N}(x_i \mid \mu_1^{(t)}, \sigma^2) + \pi_2^{(t)} \mathcal{N}(x_i \mid \mu_2^{(t)}, \sigma^2)}$$

$r_{i2} = 1 - r_{i1}$

**(2) M步（共享方差）**

令 $N_k = \sum_i r_{ik}$，参数更新为：

$$\mu_k^{(t+1)} = \frac{\sum_i r_{ik} x_i}{N_k}, \quad \pi_k^{(t+1)} = \frac{N_k}{N}$$

共享方差：

$$(\sigma^2)^{(t+1)} = \frac{\sum_k \sum_i r_{ik} (x_i - \mu_k^{(t+1)})^2}{N}$$

**(3) 完全重叠时的行为**

当 $\mu_1 = \mu_2$ 时，两个分量对每个数据点贡献相同，$r_{i1} = \pi_1, r_{i2} = \pi_2$。M步更新后，两个均值都等于全局加权均值：

$$\mu_1^{(t+1)} = \mu_2^{(t+1)} = \frac{1}{N} \sum_i x_i$$

这是**局部极值陷阱**：模型无法分辨两个分量，不会自动打破对称性。这说明 GMM 对初始化敏感，需要良好的参数初始化（如 K-means++ 初始化）来避免收敛到退化解。

---

### 练习 24.5 解答

**(1) KL 散度解析表达式**

设 $q_\phi = \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$，$P(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$，维度为 $J$：

$$\text{KL}(q_\phi \| P) = -\frac{1}{2} \sum_{j=1}^{J} \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

推导：利用高斯分布 KL 散度的公式，两个高斯分布之间的 KL 散度为：

$$\text{KL}(\mathcal{N}(\mu, \Sigma) \| \mathcal{N}(0, I)) = \frac{1}{2}(\text{tr}(\Sigma) + \mu^\top \mu - J - \log |\Sigma|)$$

对于对角协方差，$\text{tr}(\Sigma) = \sum_j \sigma_j^2$，$\log|\Sigma| = \sum_j \log \sigma_j^2$，代入即得上式。

**(2) 重参数化技巧的必要性**

$\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log P_\theta(\mathbf{x}|\mathbf{z})]$ 对 $\phi$ 的梯度为：

$$\nabla_\phi \mathbb{E}_{q_\phi}[\cdot] = \nabla_\phi \int q_\phi(\mathbf{z}|\mathbf{x}) \log P_\theta(\mathbf{x}|\mathbf{z}) \, d\mathbf{z}$$

由于积分的边界依赖于 $\phi$（通过分布 $q_\phi$），无法直接交换梯度和积分顺序。蒙特卡洛估计 $\frac{1}{S}\sum_s \nabla_\phi [\log P_\theta(\mathbf{x}|\mathbf{z}^{(s)})]$ 需要 $\mathbf{z}^{(s)}$ 对 $\phi$ 可微，但直接从 $q_\phi$ 采样是不可微的操作。

重参数化将 $\mathbf{z} = \boldsymbol{\mu}_\phi + \boldsymbol{\sigma}_\phi \odot \boldsymbol{\epsilon}$，$\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$，将随机性从参数中分离出来，使得 $\mathbf{z}$ 对 $\phi$ 可微。

**(3) KL 坍塌**

KL 坍塌指训练过程中 $\text{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| P(\mathbf{z})) \to 0$，即近似后验趋向于先验 $P(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$，不再依赖输入 $\mathbf{x}$。

**意味着**：编码器没有将任何有用信息编入隐变量，解码器完全忽略 $\mathbf{z}$（退化为普通自回归模型）。

**原因**：当解码器过于强大时，它可以直接从输入预测输出而不需要隐变量的帮助，KL 项就"免费"被最小化了。

**缓解方法**：
- **KL 退火**：训练初期给 KL 项加小权重 $\beta < 1$，逐渐增加到 1
- **$\beta$-VAE**：固定使用 $\beta > 1$ 的 KL 权重以增强隐变量的解耦
- **自由比特（Free Bits）**：保证每个隐变量维度至少贡献一定量的 KL 散度
