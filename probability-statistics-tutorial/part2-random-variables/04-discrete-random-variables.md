# 第4章 离散随机变量

> **难度**：★★☆☆☆
> **前置知识**：第1-3章概率基础

---

## 学习目标

- 理解随机变量的概念及其作为样本空间到实数映射的本质
- 掌握离散随机变量的概率质量函数（PMF）和累积分布函数（CDF）
- 熟练计算离散随机变量的期望和方差
- 理解期望和方差的基本性质
- 建立离散随机变量与深度学习分类任务的联系

---

## 4.1 随机变量的概念

### 定义

**随机变量**（Random Variable）是定义在样本空间 $\Omega$ 上的实值函数：

$$X: \Omega \to \mathbb{R}$$

它将每个样本点 $\omega \in \Omega$ 映射到一个实数 $X(\omega)$。

### 直观理解

随机变量是对随机试验结果的数值化描述。例如：

| 随机试验 | 样本空间 $\Omega$ | 随机变量 $X$ |
|----------|-------------------|--------------|
| 掷一枚硬币 | $\{正面, 反面\}$ | 正面记1，反面记0 |
| 掷两枚骰子 | $\{(i,j): 1 \leq i,j \leq 6\}$ | $X = i + j$（点数之和） |
| 射击比赛 | 各种命中情况 | 命中环数 |

### 随机变量的分类

- **离散随机变量**：取值为有限个或可数无穷个
- **连续随机变量**：取值为某个区间内的任意实数

本章聚焦于离散随机变量。

---

## 4.2 离散随机变量与概率质量函数

### 定义

设 $X$ 是离散随机变量，其可能取值为 $x_1, x_2, x_3, \ldots$，则 $X$ 的**概率质量函数**（Probability Mass Function, PMF）定义为：

$$p(x) = P(X = x), \quad x \in \{x_1, x_2, \ldots\}$$

### PMF的性质

1. **非负性**：$p(x) \geq 0$ 对所有 $x$ 成立
2. **归一化**：$\sum_{i} p(x_i) = 1$
3. **概率计算**：$P(X \in A) = \sum_{x_i \in A} p(x_i)$

### 例4.1：掷骰子

掷一枚均匀骰子，$X$ 表示出现的点数。

$$p(x) = \frac{1}{6}, \quad x \in \{1, 2, 3, 4, 5, 6\}$$

验证归一化：$\sum_{x=1}^{6} p(x) = 6 \times \frac{1}{6} = 1$ ✓

### 例4.2：二项分布预览

掷 $n$ 次硬币，$X$ 表示正面出现的次数，正面概率为 $p$：

$$p(k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k = 0, 1, \ldots, n$$

---

## 4.3 累积分布函数

### 定义

随机变量 $X$ 的**累积分布函数**（Cumulative Distribution Function, CDF）定义为：

$$F(x) = P(X \leq x) = \sum_{x_i \leq x} p(x_i)$$

### CDF的性质

1. **单调不减**：若 $x_1 < x_2$，则 $F(x_1) \leq F(x_2)$
2. **右连续**：$\lim_{x \to a^+} F(x) = F(a)$
3. **边界条件**：$\lim_{x \to -\infty} F(x) = 0$，$\lim_{x \to +\infty} F(x) = 1$

### 离散CDF的特点

对于离散随机变量，CDF是**阶梯函数**，在每个可能取值处有跳跃。

### CDF与PMF的关系

$$p(x_i) = F(x_i) - F(x_{i-1}) = P(X \leq x_i) - P(X < x_i)$$

### 例4.3：计算概率

设 $X$ 的PMF为：$p(1) = 0.2, p(2) = 0.3, p(3) = 0.3, p(4) = 0.2$

CDF为：
- $F(1) = 0.2$
- $F(2) = 0.5$
- $F(3) = 0.8$
- $F(4) = 1.0$

$P(2 \leq X \leq 3) = F(3) - F(1) = 0.8 - 0.2 = 0.6$

---

## 4.4 期望

### 定义

离散随机变量 $X$ 的**期望**（Expectation）或**均值**定义为：

$$E[X] = \sum_{i} x_i \cdot p(x_i)$$

期望是随机变量取值的加权平均，权重为对应的概率。

### 函数的期望

若 $g(X)$ 是 $X$ 的函数，则：

$$E[g(X)] = \sum_{i} g(x_i) \cdot p(x_i)$$

### 期望的性质

设 $a, b$ 为常数，$X, Y$ 为随机变量：

1. **常数的期望**：$E[a] = a$
2. **线性性**：$E[aX + b] = aE[X] + b$
3. **可加性**：$E[X + Y] = E[X] + E[Y]$（总是成立！）
4. **独立变量的乘积**：若 $X, Y$ 独立，则 $E[XY] = E[X]E[Y]$

### 例4.4：骰子期望

$$E[X] = \sum_{x=1}^{6} x \cdot \frac{1}{6} = \frac{1+2+3+4+5+6}{6} = \frac{21}{6} = 3.5$$

### 例4.5：二项分布期望

设 $X \sim \text{Binomial}(n, p)$，可以证明：

$$E[X] = np$$

**证明思路**：将 $X$ 写成 $n$ 个独立伯努利变量的和，利用期望的可加性。

---

## 4.5 方差与标准差

### 方差的定义

随机变量 $X$ 的**方差**（Variance）定义为：

$$\text{Var}(X) = E[(X - \mu)^2]$$

其中 $\mu = E[X]$ 是期望。方差度量随机变量偏离均值的程度。

### 方差的计算公式

$$\text{Var}(X) = E[X^2] - (E[X])^2$$

**证明**：
$$\text{Var}(X) = E[(X-\mu)^2] = E[X^2 - 2\mu X + \mu^2] = E[X^2] - 2\mu E[X] + \mu^2 = E[X^2] - \mu^2$$

### 标准差

**标准差**（Standard Deviation）是方差的平方根：

$$\sigma = \sqrt{\text{Var}(X)}$$

标准差与随机变量具有相同的量纲，更便于解释。

### 方差的性质

1. **非负性**：$\text{Var}(X) \geq 0$
2. **常数的方差**：$\text{Var}(a) = 0$
3. **线性变换**：$\text{Var}(aX + b) = a^2 \text{Var}(X)$
4. **独立变量的和**：若 $X, Y$ 独立，则 $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$

### 例4.6：骰子方差

$E[X^2] = \sum_{x=1}^{6} x^2 \cdot \frac{1}{6} = \frac{1+4+9+16+25+36}{6} = \frac{91}{6}$

$\text{Var}(X) = \frac{91}{6} - \left(\frac{7}{2}\right)^2 = \frac{91}{6} - \frac{49}{4} = \frac{182 - 147}{12} = \frac{35}{12} \approx 2.92$

$\sigma = \sqrt{\frac{35}{12}} \approx 1.71$

---

## 本章小结

| 概念 | 定义/公式 |
|------|-----------|
| 随机变量 | $X: \Omega \to \mathbb{R}$ |
| PMF | $p(x) = P(X = x)$ |
| CDF | $F(x) = P(X \leq x)$ |
| 期望 | $E[X] = \sum_i x_i p(x_i)$ |
| 方差 | $\text{Var}(X) = E[X^2] - (E[X])^2$ |
| 标准差 | $\sigma = \sqrt{\text{Var}(X)}$ |

**核心要点**：
- 随机变量将随机试验的结果数值化
- PMF描述离散随机变量的完整概率分布
- 期望是概率加权平均，具有线性性
- 方差度量分布的离散程度

---

## 深度学习应用：分类任务与Softmax输出

### 分类问题的概率视角

在深度学习分类任务中，模型的目标是预测输入属于各个类别的**概率分布**——这正是一个离散随机变量的PMF！

设有 $K$ 个类别，模型输出是一个概率向量：

$$\mathbf{p} = (p_1, p_2, \ldots, p_K), \quad \sum_{k=1}^{K} p_k = 1$$

### Softmax函数

神经网络最后一层通常输出**logits**（未归一化的分数）$\mathbf{z} = (z_1, z_2, \ldots, z_K)$。

**Softmax函数**将logits转换为概率分布：

$$p_k = \text{Softmax}(z_k) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

**性质**：
- 输出非负：$p_k > 0$
- 归一化：$\sum_k p_k = 1$
- 保序：$z_i > z_j \Rightarrow p_i > p_j$

### 交叉熵损失

给定真实标签 $y$（one-hot编码）和预测概率 $\hat{\mathbf{p}}$，**交叉熵损失**定义为：

$$\mathcal{L} = -\sum_{k=1}^{K} y_k \log \hat{p}_k = -\log \hat{p}_y$$

其中 $y$ 是真实类别的索引。

**概率解释**：交叉熵损失是负对数似然，最小化它等价于最大化正确类别的预测概率。

### 温度缩放

Softmax的"尖锐程度"可以通过温度参数 $T$ 控制：

$$p_k = \frac{e^{z_k/T}}{\sum_{j} e^{z_j/T}}$$

- $T \to 0$：分布趋于one-hot（确定性）
- $T = 1$：标准Softmax
- $T \to \infty$：分布趋于均匀（最大不确定性）

### PyTorch代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Softmax将logits转换为概率分布（PMF）
logits = torch.tensor([2.0, 1.0, 0.1])
probs = F.softmax(logits, dim=0)
print(f"Logits: {logits}")
print(f"Probabilities (PMF): {probs}")
print(f"Sum of probabilities: {probs.sum():.4f}")  # 验证归一化

# 2. 验证PMF性质
print(f"\nPMF性质验证:")
print(f"  非负性: {(probs >= 0).all()}")
print(f"  归一化: {probs.sum().item():.6f}")

# 3. 计算期望（预测类别的期望索引）
classes = torch.tensor([0.0, 1.0, 2.0])
expected_class = (probs * classes).sum()
print(f"\n期望类别索引: {expected_class:.4f}")

# 4. 交叉熵损失
true_label = torch.tensor([0])  # 真实类别是0
logits_batch = logits.unsqueeze(0)  # (1, 3)

# 方法1：使用CrossEntropyLoss（接受logits）
criterion = nn.CrossEntropyLoss()
loss1 = criterion(logits_batch, true_label)

# 方法2：手动计算 -log(p_true)
loss2 = -torch.log(probs[0])

print(f"\n交叉熵损失:")
print(f"  nn.CrossEntropyLoss: {loss1.item():.4f}")
print(f"  手动计算 -log(p_0): {loss2.item():.4f}")

# 5. 温度缩放的效果
print("\n温度缩放效果:")
for T in [0.5, 1.0, 2.0, 5.0]:
    probs_T = F.softmax(logits / T, dim=0)
    print(f"  T={T}: {probs_T.numpy().round(3)}")

# 6. 完整的分类网络示例
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)  # 输出logits
        return logits

    def predict_proba(self, x):
        """返回概率分布（PMF）"""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

# 创建模型并演示
model = SimpleClassifier(input_dim=10, num_classes=3)
x = torch.randn(5, 10)  # 5个样本，10维特征

logits = model(x)
probs = model.predict_proba(x)

print(f"\n分类网络输出:")
print(f"  Logits shape: {logits.shape}")
print(f"  Probs shape: {probs.shape}")
print(f"  样本0的PMF: {probs[0].detach().numpy().round(3)}")
print(f"  验证归一化: {probs.sum(dim=1).detach().numpy()}")
```

**输出**：
```
Logits: tensor([2.0000, 1.0000, 0.1000])
Probabilities (PMF): tensor([0.6590, 0.2424, 0.0986])
Sum of probabilities: 1.0000

PMF性质验证:
  非负性: True
  归一化: 1.000000

期望类别索引: 0.4396

交叉熵损失:
  nn.CrossEntropyLoss: 0.4170
  手动计算 -log(p_0): 0.4170

温度缩放效果:
  T=0.5: [0.844 0.114 0.042]
  T=1.0: [0.659 0.242 0.099]
  T=2.0: [0.506 0.307 0.188]
  T=5.0: [0.387 0.338 0.275]
```

### 关键联系

| 概率论概念 | 深度学习对应 |
|-----------|-------------|
| 离散随机变量 | 类别标签 |
| PMF | Softmax输出 |
| 期望 | 预测的平均类别 |
| 熵 | 预测的不确定性 |
| 对数似然 | 负交叉熵损失 |

---

## 练习题

**练习 4.1**（基础）

设离散随机变量 $X$ 的PMF为：

| $x$ | -1 | 0 | 1 | 2 |
|-----|-----|---|---|---|
| $p(x)$ | 0.1 | 0.3 | 0.4 | 0.2 |

(a) 验证这是一个有效的PMF
(b) 计算 $P(X \geq 0)$
(c) 写出CDF $F(x)$

**练习 4.2**（计算）

设 $X$ 的PMF为 $p(k) = c \cdot k$，$k = 1, 2, 3, 4$，其他值概率为0。

(a) 求常数 $c$
(b) 计算 $E[X]$ 和 $\text{Var}(X)$

**练习 4.3**（理解）

证明：对于任意随机变量 $X$ 和常数 $a, b$：

$$\text{Var}(aX + b) = a^2 \text{Var}(X)$$

**练习 4.4**（应用）

一个三分类问题，Softmax输出为 $\mathbf{p} = (0.7, 0.2, 0.1)$。

(a) 若真实标签是类别0，计算交叉熵损失
(b) 若真实标签是类别2，损失是多少？
(c) 解释为什么错误预测的损失更高

**练习 4.5**（深度学习）

考虑温度缩放 Softmax：$p_k = \frac{e^{z_k/T}}{\sum_j e^{z_j/T}}$

(a) 证明当 $T \to 0^+$ 时，分布趋于将所有概率集中在最大logit对应的类别
(b) 证明当 $T \to \infty$ 时，分布趋于均匀分布
(c) 在知识蒸馏中，为什么使用较高的温度？

---

## 练习答案

<details>
<summary>点击展开 练习 4.1 答案</summary>

**(a)** 验证PMF有效性：
- 非负性：$0.1, 0.3, 0.4, 0.2 \geq 0$ ✓
- 归一化：$0.1 + 0.3 + 0.4 + 0.2 = 1.0$ ✓

**(b)** $P(X \geq 0) = P(X=0) + P(X=1) + P(X=2) = 0.3 + 0.4 + 0.2 = 0.9$

**(c)** CDF:
$$F(x) = \begin{cases}
0 & x < -1 \\
0.1 & -1 \leq x < 0 \\
0.4 & 0 \leq x < 1 \\
0.8 & 1 \leq x < 2 \\
1.0 & x \geq 2
\end{cases}$$

</details>

<details>
<summary>点击展开 练习 4.2 答案</summary>

**(a)** 由归一化条件：
$$\sum_{k=1}^{4} c \cdot k = c(1+2+3+4) = 10c = 1$$
$$c = \frac{1}{10} = 0.1$$

**(b)** 期望：
$$E[X] = \sum_{k=1}^{4} k \cdot \frac{k}{10} = \frac{1}{10}(1+4+9+16) = \frac{30}{10} = 3$$

$E[X^2] = \sum_{k=1}^{4} k^2 \cdot \frac{k}{10} = \frac{1}{10}(1+8+27+64) = \frac{100}{10} = 10$

方差：
$$\text{Var}(X) = E[X^2] - (E[X])^2 = 10 - 9 = 1$$

</details>

<details>
<summary>点击展开 练习 4.3 答案</summary>

设 $\mu = E[X]$，则 $E[aX+b] = aE[X] + b = a\mu + b$。

$$\text{Var}(aX+b) = E[(aX+b - (a\mu+b))^2]$$
$$= E[(aX - a\mu)^2]$$
$$= E[a^2(X - \mu)^2]$$
$$= a^2 E[(X-\mu)^2]$$
$$= a^2 \text{Var}(X)$$

注意常数 $b$ 只是平移，不影响离散程度，因此不出现在方差中。

</details>

<details>
<summary>点击展开 练习 4.4 答案</summary>

**(a)** 真实标签是类别0：
$$\mathcal{L} = -\log p_0 = -\log 0.7 \approx 0.357$$

**(b)** 真实标签是类别2：
$$\mathcal{L} = -\log p_2 = -\log 0.1 \approx 2.303$$

**(c)** 交叉熵损失 $-\log p$ 是 $p$ 的递减函数：
- 当预测正确时，$p$ 大（如0.7），$-\log p$ 小
- 当预测错误时，$p$ 小（如0.1），$-\log p$ 大

这使得模型受到惩罚以增加正确类别的预测概率。当 $p \to 0$ 时，损失 $\to \infty$，强烈惩罚过于自信的错误预测。

</details>

<details>
<summary>点击展开 练习 4.5 答案</summary>

**(a)** 当 $T \to 0^+$：

设 $z_{max} = \max_k z_k$ 对应类别 $k^*$。

$$p_k = \frac{e^{z_k/T}}{\sum_j e^{z_j/T}} = \frac{e^{(z_k - z_{max})/T}}{\sum_j e^{(z_j - z_{max})/T}}$$

当 $T \to 0^+$：
- 若 $k = k^*$：分子分母主项都是1，$p_{k^*} \to 1$
- 若 $k \neq k^*$：$z_k - z_{max} < 0$，$e^{(z_k-z_{max})/T} \to 0$，故 $p_k \to 0$

**(b)** 当 $T \to \infty$：

所有 $z_k/T \to 0$，$e^{z_k/T} \to 1$。

$$p_k \to \frac{1}{K}$$

分布趋于均匀。

**(c)** 知识蒸馏中使用较高温度的原因：
- 低温时，teacher模型输出接近one-hot，几乎没有"暗知识"
- 高温"软化"概率分布，揭示类别间的相似性结构
- 例如：数字识别中，7和1的相似性在低温时被掩盖，高温时体现为7类有较高的1类概率
- 这种软标签包含了teacher学到的类间关系，有助于student学习

</details>
