# 第2章：条件概率与独立性

## 学习目标

学完本章后，你将能够：

- 理解条件概率的定义与直觉含义，正确计算 $P(A|B)$
- 掌握乘法公式与链式法则，分解多事件的联合概率
- 运用全概率公式处理复杂的分情况概率计算
- 理解并应用贝叶斯公式进行"逆向推断"（从结果到原因）
- 区分事件的独立性与互斥性，理解两两独立与相互独立的区别

---

## 2.1 条件概率的定义

### 直觉引入

假设你知道今天**已经下雨**，请问"路上堵车"的概率是多少？这与你不知道天气时的估计不同——额外的信息改变了你的判断。

**条件概率**正是刻画"在已知某事件发生的前提下，另一事件发生的概率"。

### 正式定义

设 $B$ 为样本空间 $\Omega$ 中的一个事件，且 $P(B) > 0$，则在事件 $B$ 发生的条件下，事件 $A$ 发生的**条件概率**定义为：

$$
\boxed{P(A \mid B) = \frac{P(A \cap B)}{P(B)}}
$$

**几何直觉**：将 $B$ 视为"缩小后的新样本空间"，$A \cap B$ 是这个新空间中 $A$ 所占的部分。条件概率就是在新空间中重新归一化后的概率。

$$
\underbrace{P(A \mid B)}_{\text{条件概率}} = \frac{\overbrace{P(A \cap B)}^{A \text{ 与 } B \text{ 同时发生}}}{\underbrace{P(B)}_{\text{归一化因子}}}
$$

### 示例：医疗检测

一种疾病在人群中的患病率为 $1\%$，某检测的灵敏度（真阳性率）为 $99\%$，特异度（真阴性率）为 $95\%$。

设事件：
- $D$：患病，$P(D) = 0.01$
- $+$：检测阳性

已知：$P(+|D) = 0.99$，$P(+|\bar{D}) = 0.05$

问：已知某人检测阳性，他真正患病的概率是多少？

这个问题需要**贝叶斯公式**来解答，我们在 2.4 节详细讨论。

### 条件概率的性质

条件概率 $P(\cdot \mid B)$ 本身也是一个合法的概率测度，满足概率的全部公理：

1. **非负性**：$P(A \mid B) \geq 0$
2. **规范性**：$P(\Omega \mid B) = 1$
3. **可列可加性**：若 $A_1, A_2, \ldots$ 两两互斥，则

$$
P\!\left(\bigcup_{i=1}^{\infty} A_i \,\middle|\, B\right) = \sum_{i=1}^{\infty} P(A_i \mid B)
$$

---

## 2.2 乘法公式

### 基本乘法公式

由条件概率的定义直接变形，得到**乘法公式**（multiplication rule）：

$$
\boxed{P(A \cap B) = P(A \mid B) \cdot P(B) = P(B \mid A) \cdot P(A)}
$$

**含义**：两个事件同时发生的概率，等于其中一个发生的概率乘以在该事件已发生前提下另一个发生的概率。

### 链式法则（Chain Rule）

乘法公式可以推广到多个事件，称为**链式法则**：

$$
\boxed{P(A_1 \cap A_2 \cap \cdots \cap A_n) = P(A_1) \cdot P(A_2 \mid A_1) \cdot P(A_3 \mid A_1 \cap A_2) \cdots P(A_n \mid A_1 \cap \cdots \cap A_{n-1})}
$$

简记为：

$$
P\!\left(\bigcap_{i=1}^{n} A_i\right) = \prod_{i=1}^{n} P\!\left(A_i \,\middle|\, \bigcap_{j=1}^{i-1} A_j\right)
$$

其中 $P(A_1 | \emptyset) \triangleq P(A_1)$。

**注**：链式法则在语言模型中至关重要。一段文本 $w_1 w_2 \cdots w_n$ 的概率被分解为：

$$
P(w_1, w_2, \ldots, w_n) = P(w_1) \cdot P(w_2 \mid w_1) \cdot P(w_3 \mid w_1, w_2) \cdots P(w_n \mid w_1, \ldots, w_{n-1})
$$

这正是自回归语言模型（如 GPT）的核心概率建模思路。

### 示例：抽签问题

袋中有 5 张签，其中 2 张中签。甲先抽，乙后抽（不放回），乙中签的概率是多少？

设 $A$：甲中签，$B$：乙中签。

$$
P(B) = P(B \mid A) P(A) + P(B \mid \bar{A}) P(\bar{A})
$$

$$
= \frac{1}{4} \cdot \frac{2}{5} + \frac{2}{4} \cdot \frac{3}{5} = \frac{2}{20} + \frac{6}{20} = \frac{8}{20} = \frac{2}{5}
$$

乙中签的概率与甲相同！这说明**抽签的公平性**与抽取顺序无关。

---

## 2.3 全概率公式

### 划分的概念

若事件 $B_1, B_2, \ldots, B_n$ 满足：

1. **互斥**：$B_i \cap B_j = \emptyset$（$i \neq j$）
2. **完备**：$B_1 \cup B_2 \cup \cdots \cup B_n = \Omega$
3. **正概率**：$P(B_i) > 0$（$i = 1, \ldots, n$）

则称 $\{B_1, B_2, \ldots, B_n\}$ 为样本空间 $\Omega$ 的一个**完备事件组**（或**划分**）。

### 全概率公式

对样本空间的任意划分 $\{B_1, \ldots, B_n\}$ 和任意事件 $A$：

$$
\boxed{P(A) = \sum_{i=1}^{n} P(A \mid B_i) \cdot P(B_i)}
$$

**直觉**：将复杂事件 $A$ 按"原因"分情况讨论——每种原因 $B_i$ 发生的概率为 $P(B_i)$，在该原因下 $A$ 发生的概率为 $P(A|B_i)$，对所有可能原因求加权平均。

```
         Ω
    ┌────┬────┬────┐
    │ B₁ │ B₂ │ B₃ │
    │ ▓▓▓│░░░░│ ▓▓ │  ← A∩B₁, A∩B₂, A∩B₃
    └────┴────┴────┘
     P(A) = P(A|B₁)P(B₁) + P(A|B₂)P(B₂) + P(A|B₃)P(B₃)
```

### 示例：产品质量检验

工厂有三条生产线，各生产该产品的 $30\%$、$45\%$、$25\%$，次品率分别为 $2\%$、$3\%$、$5\%$。

随机抽取一件产品，是次品的概率？

设 $B_i$：来自第 $i$ 条生产线，$A$：次品。

$$
P(A) = P(A \mid B_1) P(B_1) + P(A \mid B_2) P(B_2) + P(A \mid B_3) P(B_3)
$$

$$
= 0.02 \times 0.30 + 0.03 \times 0.45 + 0.05 \times 0.25
$$

$$
= 0.006 + 0.0135 + 0.0125 = 0.032
$$

次品率约为 $3.2\%$。

---

## 2.4 贝叶斯公式

### 从"果"到"因"的推断

全概率公式计算的是"已知原因，求结果的概率"；而贝叶斯公式解决的是反向问题：**已知结果，推断原因的概率**。

### 贝叶斯公式

设 $\{B_1, \ldots, B_n\}$ 为 $\Omega$ 的划分，$A$ 为任意正概率事件，则：

$$
\boxed{P(B_i \mid A) = \frac{P(A \mid B_i) \cdot P(B_i)}{\displaystyle\sum_{j=1}^{n} P(A \mid B_j) \cdot P(B_j)}}
$$

**三个关键量的统计学命名**：

| 名称 | 符号 | 含义 |
|------|------|------|
| **先验概率**（prior） | $P(B_i)$ | 在观测到 $A$ 之前，对 $B_i$ 的初始判断 |
| **似然**（likelihood） | $P(A \mid B_i)$ | 在 $B_i$ 为真时，观测到 $A$ 的概率 |
| **后验概率**（posterior） | $P(B_i \mid A)$ | 观测到 $A$ 之后，对 $B_i$ 的更新判断 |

$$
\underbrace{P(B_i \mid A)}_{\text{后验}} \propto \underbrace{P(A \mid B_i)}_{\text{似然}} \times \underbrace{P(B_i)}_{\text{先验}}
$$

这一关系常被概括为："**后验 ∝ 似然 × 先验**"。

### 示例：回到医疗检测

接续 2.1 节的问题：

$$
P(D \mid +) = \frac{P(+ \mid D) \cdot P(D)}{P(+ \mid D) \cdot P(D) + P(+ \mid \bar{D}) \cdot P(\bar{D})}
$$

$$
= \frac{0.99 \times 0.01}{0.99 \times 0.01 + 0.05 \times 0.99} = \frac{0.0099}{0.0099 + 0.0495} \approx 0.167
$$

**结论**：即使检测阳性，真正患病的概率仅约 $16.7\%$！

这个反直觉的结果源于**基率（base rate）的稀疏性**——患病率仅 $1\%$，大量假阳性"淹没"了真阳性信号。这一现象称为**基率谬误**（base rate fallacy）。

### 贝叶斯更新（Sequential Updating）

贝叶斯公式支持**序贯更新**：将前一次的后验作为下一次的先验，反复迭代。

$$
P(H \mid \text{数据}_1) \xrightarrow{\text{新数据}_2} P(H \mid \text{数据}_1, \text{数据}_2) \xrightarrow{\text{新数据}_3} \cdots
$$

这是贝叶斯统计的核心思想，也是贝叶斯深度学习的理论基础。

---

## 2.5 事件的独立性

### 独立性的定义

若两个事件 $A$、$B$ 满足：

$$
\boxed{P(A \cap B) = P(A) \cdot P(B)}
$$

则称 $A$ 与 $B$ **相互独立**。

**等价条件**（当 $P(B) > 0$ 时）：

$$
P(A \mid B) = P(A)
$$

即：知道 $B$ 发生与否，不改变 $A$ 发生的概率。

**注意**：独立性与互斥性是两个不同的概念！

| | 互斥（Mutually Exclusive） | 独立（Independent） |
|---|---|---|
| 定义 | $A \cap B = \emptyset$ | $P(A \cap B) = P(A)P(B)$ |
| 含义 | 不能同时发生 | 互不影响 |
| 关系 | 若 $P(A), P(B) > 0$，则互斥必不独立 | 独立的正概率事件必不互斥 |

### 两两独立 vs 相互独立

对于多个事件，独立性存在强弱之分：

**两两独立**（pairwise independence）：任意两个事件独立，即对所有 $i \neq j$：

$$
P(A_i \cap A_j) = P(A_i) P(A_j)
$$

**相互独立**（mutual independence）：对所有子集 $\{i_1, \ldots, i_k\} \subseteq \{1, \ldots, n\}$（$k \geq 2$）：

$$
P(A_{i_1} \cap A_{i_2} \cap \cdots \cap A_{i_k}) = P(A_{i_1}) \cdot P(A_{i_2}) \cdots P(A_{i_k})
$$

**重要**：两两独立 $\not\Rightarrow$ 相互独立！

**反例**（Bernstein）：投掷两枚均匀硬币，令：
- $A$：第一枚正面
- $B$：第二枚正面
- $C$：两枚结果相同

计算验证：$P(A) = P(B) = P(C) = \frac{1}{2}$，$P(A \cap B) = P(A \cap C) = P(B \cap C) = \frac{1}{4}$，故 $A, B, C$ 两两独立。

但 $P(A \cap B \cap C) = \frac{1}{4} \neq \frac{1}{8} = P(A)P(B)P(C)$，故三者**不相互独立**。

### 独立性的实际意义

- 若 $A_1, \ldots, A_n$ 相互独立，则它们同时发生的概率为各自概率之积：

$$
P\!\left(\bigcap_{i=1}^{n} A_i\right) = \prod_{i=1}^{n} P(A_i)
$$

- 独立随机变量的联合分布等于边缘分布之积（将在后续章节深入讨论）。
- 独立性假设大大简化了概率计算，是朴素贝叶斯分类器等算法的核心假设。

---

## 本章小结

| 概念 | 公式 | 记忆要点 |
|------|------|----------|
| 条件概率 | $P(A\|B) = P(A\cap B)/P(B)$ | 缩小样本空间，重新归一化 |
| 乘法公式 | $P(A\cap B) = P(A\|B)P(B)$ | 由条件概率变形而来 |
| 链式法则 | $P(\bigcap A_i) = \prod P(A_i \| A_1\cdots A_{i-1})$ | 语言模型的核心分解 |
| 全概率公式 | $P(A) = \sum P(A\|B_i)P(B_i)$ | 按"原因"加权平均 |
| 贝叶斯公式 | $P(B_i\|A) \propto P(A\|B_i)P(B_i)$ | 后验 ∝ 似然 × 先验 |
| 独立性 | $P(A\cap B) = P(A)P(B)$ | 互不影响，注意区分互斥 |

**核心思维方式**：贝叶斯公式提供了一种**理性更新信念**的框架——面对新证据，以先验为出发点，通过似然调整，得到后验。这是科学推断的数学基础。

---

## 深度学习应用：贝叶斯神经网络与先验后验

### 从参数估计到贝叶斯推断

标准神经网络训练的目标是找到一组**固定的最优参数** $\theta^*$：

$$
\theta^* = \arg\max_\theta \log P(\mathcal{D} \mid \theta)
$$

这是**最大似然估计**（MLE）。然而，这种方法有两个局限：

1. 无法量化参数的**不确定性**
2. 容易在小数据集上**过拟合**

**贝叶斯神经网络**（Bayesian Neural Network, BNN）将参数视为**随机变量**，用概率分布而非点估计来描述参数：

$$
P(\theta \mid \mathcal{D}) = \frac{P(\mathcal{D} \mid \theta) \cdot P(\theta)}{P(\mathcal{D})}
$$

其中：
- $P(\theta)$：**先验分布**——训练前对参数的信念（如 $\theta \sim \mathcal{N}(0, \sigma^2 I)$）
- $P(\mathcal{D} \mid \theta)$：**似然**——参数为 $\theta$ 时观测到数据的概率
- $P(\theta \mid \mathcal{D})$：**后验分布**——训练后对参数的信念
- $P(\mathcal{D})$：**边缘似然**（model evidence）——归一化常数

### 先验与正则化的联系

高斯先验 $P(\theta) = \mathcal{N}(0, \sigma^2 I)$ 对应于 **L2 正则化**（权重衰减）！

**推导**：最大后验估计（MAP）：

$$
\theta^{\text{MAP}} = \arg\max_\theta \log P(\theta \mid \mathcal{D})
$$

$$
= \arg\max_\theta \left[\log P(\mathcal{D} \mid \theta) + \log P(\theta)\right]
$$

$$
= \arg\max_\theta \left[\log P(\mathcal{D} \mid \theta) - \frac{\|\theta\|^2}{2\sigma^2}\right]
$$

$$
= \arg\min_\theta \left[\mathcal{L}(\theta) + \underbrace{\frac{1}{2\sigma^2}}_{\lambda} \|\theta\|^2\right]
$$

这正是带 **L2 正则化**的损失函数！正则化系数 $\lambda = 1/(2\sigma^2)$ 对应先验的精度。

**拉普拉斯先验** $P(\theta) \propto \exp(-|\theta|/b)$ 则对应 **L1 正则化**（稀疏解）。

| 先验分布 | 对应正则化 | 效果 |
|----------|-----------|------|
| 高斯 $\mathcal{N}(0, \sigma^2)$ | L2（权重衰减） | 参数趋近于零，平滑解 |
| 拉普拉斯 $\text{Laplace}(0, b)$ | L1（LASSO） | 参数稀疏化 |
| 均匀分布 | 无正则化 | 等价于 MLE |

### 预测与不确定性量化

贝叶斯预测通过对后验**积分**，得到预测分布：

$$
P(y^* \mid x^*, \mathcal{D}) = \int P(y^* \mid x^*, \theta) \cdot P(\theta \mid \mathcal{D}) \, d\theta
$$

这个积分通常**难以解析计算**（因为神经网络是非线性的），实践中常用近似方法：

- **变分推断**（Variational Inference）：用简单分布 $q(\theta)$ 近似后验
- **马尔可夫链蒙特卡洛**（MCMC）：采样得到后验样本
- **MC Dropout**：训练时和推断时都使用 Dropout，以多次前向传播的方差估计不确定性

### PyTorch 代码示例：MC Dropout 近似贝叶斯推断

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# ─── 1. 定义带 Dropout 的贝叶斯网络 ───────────────────────────────────────────
class BayesianMLP(nn.Module):
    """
    使用 MC Dropout 近似贝叶斯神经网络。
    关键：推断时保持 Dropout 开启（train 模式），多次采样来估计不确定性。
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout_p: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.dropout(self.fc1(x)))
        x = F.relu(self.dropout(self.fc2(x)))
        return self.fc3(x)


def mc_predict(model: BayesianMLP, x: torch.Tensor, n_samples: int = 100) -> tuple:
    """
    MC Dropout 推断：
    - 将模型置于 train 模式以启用 Dropout
    - 多次前向传播，每次 Dropout 随机遮蔽不同神经元
    - 返回预测的均值（期望后验预测）和标准差（不确定性）
    """
    model.train()  # 保持 Dropout 激活
    predictions = []

    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(x)          # 每次调用使用不同的 Dropout mask
            predictions.append(pred)

    predictions = torch.stack(predictions, dim=0)  # shape: [n_samples, batch, output]

    # 后验预测均值：对应贝叶斯积分 ∫ P(y|x,θ)P(θ|D)dθ 的蒙特卡洛近似
    mean = predictions.mean(dim=0)

    # 预测标准差：量化认知不确定性（epistemic uncertainty）
    std = predictions.std(dim=0)

    return mean, std


# ─── 2. 生成合成回归数据 ────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# 训练数据：两段区间，中间有间隙（用于检验间隙处的不确定性）
x_train_1 = torch.linspace(-3, -1, 50).unsqueeze(1)
x_train_2 = torch.linspace(1, 3, 50).unsqueeze(1)
x_train = torch.cat([x_train_1, x_train_2], dim=0)

# 真实函数：y = sin(x) + 噪声
y_train = torch.sin(x_train) + 0.1 * torch.randn_like(x_train)


# ─── 3. 训练模型 ───────────────────────────────────────────────────────────────
model = BayesianMLP(input_dim=1, hidden_dim=64, output_dim=1, dropout_p=0.1)

# L2 正则化（weight_decay）对应高斯先验 P(θ) = N(0, 1/weight_decay * I)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    pred = model(x_train)
    loss = F.mse_loss(pred, y_train)
    loss.backward()
    optimizer.step()

print(f"训练损失: {loss.item():.4f}")


# ─── 4. 预测与不确定性可视化 ───────────────────────────────────────────────────
x_test = torch.linspace(-4, 4, 200).unsqueeze(1)
mean_pred, std_pred = mc_predict(model, x_test, n_samples=200)

mean_np = mean_pred.squeeze().numpy()
std_np = std_pred.squeeze().numpy()
x_np = x_test.squeeze().numpy()

plt.figure(figsize=(10, 5))
plt.scatter(x_train.numpy(), y_train.numpy(), s=10, c='black', zorder=5, label='训练数据')
plt.plot(x_np, np.sin(x_np), 'g--', label='真实函数 sin(x)', linewidth=1.5)
plt.plot(x_np, mean_np, 'b-', label='后验预测均值', linewidth=2)

# 不确定性区间（±2σ ≈ 95% 置信区间）
plt.fill_between(x_np, mean_np - 2 * std_np, mean_np + 2 * std_np,
                 alpha=0.3, color='blue', label='±2σ 不确定性区间')

plt.axvspan(-1, 1, alpha=0.1, color='red', label='数据间隙（高不确定性）')
plt.legend()
plt.title('MC Dropout 贝叶斯神经网络：预测均值与不确定性')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.savefig('bayesian_nn_uncertainty.png', dpi=150)
plt.show()

# ─── 5. 验证贝叶斯性质 ─────────────────────────────────────────────────────────
# 数据区域内的不确定性应低于间隙区域
in_data_mask = (x_np < -1) | (x_np > 1)
gap_mask = (x_np >= -1) & (x_np <= 1)

print(f"数据区域平均不确定性（std）: {std_np[in_data_mask].mean():.4f}")
print(f"数据间隙平均不确定性（std）: {std_np[gap_mask].mean():.4f}")
print("间隙处不确定性更高，符合贝叶斯神经网络的预期行为！")
```

### 变分推断的基本思想

由于后验 $P(\theta|\mathcal{D})$ 通常难以直接计算，**变分推断**通过最小化变分分布 $q_\phi(\theta)$ 与真实后验的 KL 散度来近似：

$$
\phi^* = \arg\min_\phi \text{KL}\left[q_\phi(\theta) \,\|\, P(\theta \mid \mathcal{D})\right]
$$

等价地，最大化**证据下界**（ELBO）：

$$
\mathcal{L}(\phi) = \underbrace{\mathbb{E}_{q_\phi(\theta)}[\log P(\mathcal{D} \mid \theta)]}_{\text{期望对数似然（拟合度）}} - \underbrace{\text{KL}[q_\phi(\theta) \,\|\, P(\theta)]}_{\text{先验与变分分布的差异（正则项）}}
$$

这两项分别对应：**模型对训练数据的拟合**和**参数分布对先验的偏离程度**。贝叶斯框架自然地平衡了拟合与复杂度。

### 认知不确定性 vs 偶然不确定性

贝叶斯神经网络能区分两类不确定性：

- **认知不确定性**（Epistemic Uncertainty）：来源于训练数据不足，可通过收集更多数据减少。对应模型参数的不确定性。
- **偶然不确定性**（Aleatoric Uncertainty）：来源于数据本身的噪声，无法通过更多数据消除。

$$
\underbrace{\text{Var}[y^* \mid x^*, \mathcal{D}]}_{\text{总不确定性}} = \underbrace{\text{Var}_\theta[\mathbb{E}[y^*|\theta]]}_{\text{认知不确定性}} + \underbrace{\mathbb{E}_\theta[\text{Var}[y^*|\theta]]}_{\text{偶然不确定性}}
$$

这种区分在自动驾驶、医疗 AI 等安全关键场景中极为重要。

---

## 练习题

**题 1（基础）** 一个袋子里有 4 个红球和 6 个蓝球，不放回地依次取出两个球。

（a）已知第一个球是红球，第二个球也是红球的概率是多少？

（b）两个球都是红球的概率是多少？

---

**题 2（全概率公式）** 某工厂生产的零件由三台机器完成：机器 A 生产 $50\%$，机器 B 生产 $30\%$，机器 C 生产 $20\%$；各机器的次品率分别为 $1\%$、$2\%$、$3\%$。

（a）随机抽取一个零件，是次品的概率是多少？

（b）已知抽到的是次品，它来自机器 A 的概率是多少？

---

**题 3（贝叶斯定理）** 某病毒检测试剂盒的灵敏度（真阳性率）为 $95\%$，特异度（真阴性率）为 $90\%$。假设该病毒在某地区的感染率为 $0.5\%$。

（a）某人随机检测结果为阳性，他真正感染的概率是多少？

（b）如果某人属于高风险人群，感染率估计为 $10\%$，阳性时真正感染的概率又是多少？

（c）这两个结果说明了什么？对检测策略有什么启示？

---

**题 4（独立性）** 设事件 $A$ 和 $B$ 相互独立，$P(A) = 0.3$，$P(B) = 0.4$。

（a）计算 $P(A \cup B)$。

（b）计算 $P(A \mid B^c)$（$B^c$ 表示 $B$ 的补事件）。

（c）证明：若 $A$ 和 $B$ 独立，则 $A$ 和 $B^c$ 也独立。

---

**题 5（综合应用）** 某 AI 系统对图像进行分类，共有三个类别：猫（Cat）、狗（Dog）、其他（Other），先验概率分别为 $P(C) = 0.3$，$P(D) = 0.5$，$P(O) = 0.2$。

已知该系统给出"置信度高"信号的条件概率为：$P(\text{高} \mid C) = 0.8$，$P(\text{高} \mid D) = 0.7$，$P(\text{高} \mid O) = 0.4$。

（a）随机输入一张图片，系统给出"高置信度"信号的概率是多少？

（b）已知系统给出高置信度信号，图片是猫的概率是多少？

（c）假设系统在高置信度下的分类准确率为 $90\%$（无论类别），这是否改变你对（b）的理解？

---

## 练习答案

<details>
<summary>题 1 详细解答</summary>

设 $R_1$：第一个球是红球，$R_2$：第二个球是红球。

**（a）条件概率 $P(R_2 \mid R_1)$**

已知第一个是红球，袋中剩余 9 个球，其中 3 个红球：

$$
P(R_2 \mid R_1) = \frac{3}{9} = \frac{1}{3}
$$

**（b）联合概率 $P(R_1 \cap R_2)$**

使用乘法公式：

$$
P(R_1 \cap R_2) = P(R_2 \mid R_1) \cdot P(R_1) = \frac{1}{3} \times \frac{4}{10} = \frac{4}{30} = \frac{2}{15} \approx 0.133
$$

也可直接计算：从 10 个球中不放回取 2 个的总方案数为 $\binom{10}{2} = 45$，两个都是红球的方案数为 $\binom{4}{2} = 6$，故概率为 $6/45 = 2/15$。两种方法一致。

</details>

<details>
<summary>题 2 详细解答</summary>

设 $A, B, C$：来自对应机器；$D$：次品。

**（a）全概率公式**

$$
P(D) = P(D \mid A)P(A) + P(D \mid B)P(B) + P(D \mid C)P(C)
$$

$$
= 0.01 \times 0.50 + 0.02 \times 0.30 + 0.03 \times 0.20
$$

$$
= 0.005 + 0.006 + 0.006 = 0.017
$$

次品率为 $1.7\%$。

**（b）贝叶斯公式**

$$
P(A \mid D) = \frac{P(D \mid A) \cdot P(A)}{P(D)} = \frac{0.01 \times 0.50}{0.017} = \frac{0.005}{0.017} \approx 0.294
$$

即已知是次品，来自机器 A 的概率约为 $29.4\%$。

**验证**：$P(B \mid D) = 0.006/0.017 \approx 35.3\%$，$P(C \mid D) = 0.006/0.017 \approx 35.3\%$，三者之和为 $100\%$，正确。

**洞察**：尽管机器 A 产量最大（50\%），但其次品率最低，导致次品来自 A 的概率（29.4\%）低于 A 的产量占比（50\%）。

</details>

<details>
<summary>题 3 详细解答</summary>

设 $I$：感染；$+$：检测阳性。

已知：$P(+ \mid I) = 0.95$，$P(- \mid \bar{I}) = 0.90$，故 $P(+ \mid \bar{I}) = 0.10$。

**（a）感染率 $P(I) = 0.005$**

$$
P(+) = P(+ \mid I)P(I) + P(+ \mid \bar{I})P(\bar{I})
$$

$$
= 0.95 \times 0.005 + 0.10 \times 0.995 = 0.00475 + 0.0995 = 0.10425
$$

$$
P(I \mid +) = \frac{0.95 \times 0.005}{0.10425} = \frac{0.00475}{0.10425} \approx 4.6\%
$$

**（b）高风险人群 $P(I) = 0.10$**

$$
P(+) = 0.95 \times 0.10 + 0.10 \times 0.90 = 0.095 + 0.090 = 0.185
$$

$$
P(I \mid +) = \frac{0.95 \times 0.10}{0.185} = \frac{0.095}{0.185} \approx 51.4\%
$$

**（c）分析与启示**

结论对比：

| 人群 | 先验 $P(I)$ | 阳性后验 $P(I\|+)$ |
|------|------------|------------------|
| 一般人群 | $0.5\%$ | $4.6\%$ |
| 高风险人群 | $10\%$ | $51.4\%$ |

**启示**：
1. **先验（基率）极大地影响后验**：同样的检测，阳性结果对一般人群几乎不说明什么（4.6\%），但对高风险人群意义重大（51.4\%）。
2. **检测策略**：大规模筛查低风险人群会产生大量假阳性，造成资源浪费和心理恐慌；**靶向检测高风险人群**才能使阳性结果真正有诊断价值。
3. 这解释了为什么公共卫生政策通常优先对高风险群体（接触史、症状等）进行检测。

</details>

<details>
<summary>题 4 详细解答</summary>

已知 $A, B$ 独立，$P(A) = 0.3$，$P(B) = 0.4$。

**（a）$P(A \cup B)$**

由容斥原理：

$$
P(A \cup B) = P(A) + P(B) - P(A \cap B)
$$

由独立性，$P(A \cap B) = P(A) \cdot P(B) = 0.3 \times 0.4 = 0.12$，故：

$$
P(A \cup B) = 0.3 + 0.4 - 0.12 = 0.58
$$

**（b）$P(A \mid B^c)$**

由（c）的结论，$A$ 和 $B^c$ 也独立，因此：

$$
P(A \mid B^c) = P(A) = 0.3
$$

**（c）证明 $A$ 与 $B^c$ 独立**

需证 $P(A \cap B^c) = P(A) \cdot P(B^c)$。

$$
P(A \cap B^c) = P(A) - P(A \cap B) \quad \text{（因为 } A = (A\cap B) \cup (A \cap B^c) \text{，互斥）}
$$

$$
= P(A) - P(A) \cdot P(B) \quad \text{（由 } A,B \text{ 独立）}
$$

$$
= P(A)(1 - P(B)) = P(A) \cdot P(B^c)
$$

证毕。这说明若 $A$ 与 $B$ 独立，则 $A$ 与 $B^c$、$A^c$ 与 $B$、$A^c$ 与 $B^c$ 均独立。

</details>

<details>
<summary>题 5 详细解答</summary>

设 $C$：猫，$D$：狗，$O$：其他，$H$：高置信度。

**（a）全概率公式**

$$
P(H) = P(H \mid C)P(C) + P(H \mid D)P(D) + P(H \mid O)P(O)
$$

$$
= 0.8 \times 0.3 + 0.7 \times 0.5 + 0.4 \times 0.2
$$

$$
= 0.24 + 0.35 + 0.08 = 0.67
$$

系统给出高置信度信号的概率为 $67\%$。

**（b）贝叶斯公式**

$$
P(C \mid H) = \frac{P(H \mid C) \cdot P(C)}{P(H)} = \frac{0.8 \times 0.3}{0.67} = \frac{0.24}{0.67} \approx 35.8\%
$$

高置信度信号下，图片是猫的后验概率约为 $35.8\%$。

**（c）讨论**

"系统在高置信度下的分类准确率为 $90\%$"描述的是系统**决策的正确性**，而（b）求的是**在高置信度信号下真正是猫的概率**——这是两个不同的问题。

（b）的结论说明：即使系统发出高置信度信号，也不能断定图片是猫（概率仅 $35.8\%$），还需结合系统的具体分类输出（它说是猫还是狗）才能判断。

若系统在高置信度下还输出了"是猫"，则根据 $90\%$ 的准确率，真正是猫的概率会大幅提升。这说明：**置信度信号和分类标签是两个互补的信息维度**，应结合使用。

这正是贝叶斯框架的力量：精确追踪每一条证据对推断的贡献。

</details>

---

*下一章预告*：[第3章：组合数学基础](./03-combinatorics.md) — 我们将学习排列组合的计数方法，为计算古典概型中的复杂概率问题提供工具。
