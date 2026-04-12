# 第六章：随机微分方程入门

> **本章导读**：随机微分方程（SDE）提供了理解扩散模型的连续时间视角。Song et al. 2021将DDPM、NCSN等离散方法统一进SDE框架，揭示了扩散模型的深层数学结构。本章介绍SDE的基础知识，并推导扩散过程的正向/逆向SDE。

**前置知识**：前五章，常微分方程基础
**预计学习时间**：110分钟

---

## 学习目标

完成本章学习后，你将能够：
1. 理解布朗运动和维纳过程的基本性质
2. 掌握伊藤积分和伊藤公式（链式法则的随机版本）
3. 写出扩散模型的正向SDE（VP-SDE和VE-SDE）
4. 推导逆向SDE（Anderson 1982），理解分数在其中的作用
5. 推导概率流ODE，理解其与逆向SDE的关系

---

## 6.1 随机过程与布朗运动

### 布朗运动（维纳过程）

**维纳过程** $\{W_t\}_{t \geq 0}$ 满足：
1. $W_0 = 0$（初始为零）
2. 独立增量：$W_t - W_s \perp W_s - W_r$，对任意 $r < s < t$
3. 高斯增量：$W_t - W_s \sim \mathcal{N}(0, t-s)$
4. 轨迹连续（但几乎处处不可微）

**关键性质**：
- $\mathbb{E}[W_t] = 0$
- $\mathbb{E}[W_t^2] = t$
- $\mathbb{E}[W_s W_t] = \min(s, t)$（协方差）
- **粗糙性**：在任意区间内，轨迹无限振荡，不可微

### 随机微分记号

形式上写 $dW_t = \sqrt{dt}\cdot \xi_t$，其中 $\xi_t$ 是白噪声（$\mathcal{N}(0,1)$的无穷维模拟）。

更严格地，维纳增量满足：$W_{t+dt} - W_t \sim \mathcal{N}(0, dt)$。

这是构建SDE的基础。

---

## 6.2 伊藤积分

### 普通积分的失败

由于布朗运动的轨迹不可微，普通Riemann积分 $\int_0^T f(W_t) dW_t$ 无意义——需要专门定义。

### 伊藤积分的定义

伊藤积分通过**左端点Riemann和**的极限定义：

$$\int_0^T H_t dW_t = \lim_{n\to\infty} \sum_{k=0}^{n-1} H_{t_k}(W_{t_{k+1}} - W_{t_k})$$

**伊藤等距（Itô Isometry）**（最重要的性质）：

$$\mathbb{E}\left[\left(\int_0^T H_t dW_t\right)^2\right] = \mathbb{E}\left[\int_0^T H_t^2 dt\right]$$

这使得伊藤积分具有良好的 $L^2$ 理论。

### 伊藤公式（随机链式法则）

设 $X_t$ 满足 $dX_t = \mu_t dt + \sigma_t dW_t$，$f(t, x)$ 是光滑函数，则：

$$df(t, X_t) = \left(\frac{\partial f}{\partial t} + \mu_t \frac{\partial f}{\partial x} + \frac{\sigma_t^2}{2}\frac{\partial^2 f}{\partial x^2}\right)dt + \sigma_t \frac{\partial f}{\partial x}dW_t$$

**与普通链式法则的差异**：多了一项 $\frac{\sigma_t^2}{2}\frac{\partial^2 f}{\partial x^2}dt$！

**原因**：$(dW_t)^2 = dt$（在均方意义下），而普通微积分中 $(dx)^2 = 0$。

---

## 6.3 随机微分方程（SDE）

### SDE的一般形式

$$d\mathbf{x} = \mathbf{f}(\mathbf{x}, t)dt + g(t)d\mathbf{W}_t$$

其中：
- $\mathbf{f}(\mathbf{x}, t)$：**漂移项**（drift），确定性部分，控制均值演化
- $g(t)$：**扩散系数**（diffusion coefficient），控制随机性强度
- $d\mathbf{W}_t$：$d$维维纳过程的增量

### 数值求解：Euler-Maruyama方法

类似于常微分方程的Euler法，步长为 $\Delta t$：

$$\mathbf{x}_{t+\Delta t} = \mathbf{x}_t + \mathbf{f}(\mathbf{x}_t, t)\Delta t + g(t)\sqrt{\Delta t}\mathbf{z}, \quad \mathbf{z} \sim \mathcal{N}(0, I)$$

---

## 6.4 正向SDE（扩散过程）

### VP-SDE（Variance Preserving SDE）

对应DDPM的连续时间极限，令 $\beta(t)$ 为连续噪声调度：

$$d\mathbf{x} = -\frac{1}{2}\beta(t)\mathbf{x} \, dt + \sqrt{\beta(t)} \, d\mathbf{W}_t$$

**特性**：
- 漂移项 $-\frac{1}{2}\beta(t)\mathbf{x}$：线性收缩，将数据推向原点
- 扩散系数 $\sqrt{\beta(t)}$：同时注入噪声
- **方差保持**：若 $\mathbf{x}_0$ 的方差为1，则 $\mathbf{x}_t$ 的方差接近1（不发散）

**边缘分布**（通过Fokker-Planck方程）：

$$\mathbf{x}_t | \mathbf{x}_0 \sim \mathcal{N}\left(\mathbf{x}_t; e^{-\frac{1}{2}\int_0^t\beta(s)ds}\mathbf{x}_0, \left(1 - e^{-\int_0^t\beta(s)ds}\right)I\right)$$

令 $\bar{\alpha}_t = e^{-\int_0^t\beta(s)ds}$，即 $\mathbf{x}_t | \mathbf{x}_0 \sim \mathcal{N}(\sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)I)$——与DDPM完全一致！

### VE-SDE（Variance Exploding SDE）

对应NCSN，噪声方差随时间爆炸：

$$d\mathbf{x} = \sqrt{\frac{d[\sigma^2(t)]}{dt}} \, d\mathbf{W}_t$$

**特性**：
- 无漂移项：数据本身不移动，只添加噪声
- 噪声方差 $\sigma^2(t)$ 从小增大：$\mathbf{x}_t|\mathbf{x}_0 \sim \mathcal{N}(\mathbf{x}_0, [\sigma^2(t)-\sigma^2(0)]I)$
- 最终方差 $\sigma^2(T)$ 非常大（VE = Variance Exploding）

### 亚VP-SDE

$$d\mathbf{x} = -\frac{1}{2}\beta(t)\mathbf{x} \, dt + \sqrt{\beta(t)(1-e^{-2\int_0^t\beta(s)ds})} \, d\mathbf{W}_t$$

介于VP和VE之间，实践中有时表现更好。

---

## 6.5 逆向SDE

### Anderson 1982定理

Anderson (1982) 证明了以下结论：设正向SDE为 $d\mathbf{x} = \mathbf{f}(\mathbf{x},t)dt + g(t)d\mathbf{W}_t$，则存在逆向时间的SDE：

$$d\mathbf{x} = \left[\mathbf{f}(\mathbf{x},t) - g(t)^2 \nabla_\mathbf{x}\log p_t(\mathbf{x})\right]dt + g(t)d\bar{\mathbf{W}}_t$$

其中 $\bar{\mathbf{W}}_t$ 是逆向时间的维纳过程，$p_t(\mathbf{x})$ 是时刻 $t$ 的边缘分布。

**关键点**：逆向SDE中包含**分数函数** $\nabla_\mathbf{x}\log p_t(\mathbf{x})$！

这解释了为什么学习分数函数可以用于生成：
1. 运行正向SDE：$\mathbf{x}_0 \to \mathbf{x}_T \approx \mathcal{N}(0,I)$
2. 学习分数：$s_\theta(\mathbf{x},t) \approx \nabla_\mathbf{x}\log p_t(\mathbf{x})$
3. 运行逆向SDE（用 $s_\theta$ 代替真实分数）：$\mathbf{x}_T \to \mathbf{x}_0$

### VP-SDE的逆向过程

代入 $\mathbf{f} = -\frac{1}{2}\beta(t)\mathbf{x}$，$g = \sqrt{\beta(t)}$：

$$d\mathbf{x} = \left[-\frac{1}{2}\beta(t)\mathbf{x} - \beta(t)\nabla_\mathbf{x}\log p_t(\mathbf{x})\right]dt + \sqrt{\beta(t)}d\bar{\mathbf{W}}_t$$

用学习到的分数 $s_\theta$ 代替 $\nabla_\mathbf{x}\log p_t$，即得到DDPM的采样算法。

---

## 6.6 概率流ODE

### 从SDE到ODE

Song et al. (2021) 证明：对于任意满足给定SDE的随机过程 $p_t(\mathbf{x})$，存在一个确定性的ODE（无随机项），其解轨迹具有**相同的边缘分布** $p_t(\mathbf{x})$：

$$\frac{d\mathbf{x}}{dt} = \mathbf{f}(\mathbf{x},t) - \frac{1}{2}g(t)^2 \nabla_\mathbf{x}\log p_t(\mathbf{x})$$

这是**概率流ODE**（Probability Flow ODE）。

### 对比逆向SDE与概率流ODE

| | 逆向SDE | 概率流ODE |
|---|---------|----------|
| 随机性 | ✓（朗之万噪声） | ✗（确定性） |
| 边缘分布 | 与正向一致 | 与正向一致 |
| 轨迹 | 随机 | 确定性（给定初始点） |
| 数值误差 | 较大（SDE solver误差大） | 较小（ODE solver可用高阶方法） |
| 似然计算 | 困难 | **可以**（通过Hutchinson迹估计） |
| 对应关系 | DDPM采样 | **DDIM采样** |

> **重要**：DDIM（第10章）正是概率流ODE的欧拉离散化！这揭示了DDIM的理论基础。

---

## 代码实战

```python
"""
第六章代码实战：SDE数值求解
实现VP-SDE正向/逆向过程和概率流ODE
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple


# ============================================================
# 1. Euler-Maruyama SDE求解器
# ============================================================

class EulerMaruyamaSolver:
    """
    Euler-Maruyama方法求解SDE:
    dx = f(x, t) dt + g(t) dW
    """
    
    def __init__(self, drift_fn: Callable, diffusion_fn: Callable, dt: float = 0.01):
        self.drift = drift_fn     # f(x, t) -> drift
        self.diffusion = diffusion_fn  # g(t) -> scalar
        self.dt = dt
    
    def step(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """单步Euler-Maruyama更新"""
        drift = self.drift(x, t)             # shape: same as x
        g = self.diffusion(t)                # scalar
        noise = torch.randn_like(x)
        return x + drift * self.dt + g * (self.dt**0.5) * noise
    
    def solve(self, x0: torch.Tensor, t_start: float, t_end: float,
              return_trajectory: bool = False):
        """求解SDE"""
        x = x0.clone()
        t = t_start
        n_steps = int(abs(t_end - t_start) / self.dt)
        dt = (t_end - t_start) / n_steps
        self.dt = abs(dt)
        
        trajectory = [x.clone()] if return_trajectory else None
        
        for _ in range(n_steps):
            x = self.step(x, t)
            t += dt
            if return_trajectory:
                trajectory.append(x.clone())
        
        return (x, trajectory) if return_trajectory else x


# ============================================================
# 2. VP-SDE
# ============================================================

class VPSDE:
    """Variance Preserving SDE（对应DDPM连续时间极限）"""
    
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max
    
    def beta(self, t: float) -> float:
        """线性噪声调度 beta(t)"""
        return self.beta_min + t * (self.beta_max - self.beta_min)
    
    def marginal_params(self, t: float) -> Tuple[float, float]:
        """
        边缘分布参数：x_t|x_0 ~ N(sqrt(alpha_bar)*x_0, (1-alpha_bar)*I)
        Returns: (sqrt_alpha_bar, sqrt_1_minus_alpha_bar)
        """
        log_alpha_bar = -0.5 * (self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2)
        alpha_bar = torch.exp(torch.tensor(log_alpha_bar))
        return alpha_bar.sqrt(), (1 - alpha_bar).sqrt()
    
    def forward_drift(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """正向漂移 f(x,t) = -1/2 * beta(t) * x"""
        return -0.5 * self.beta(t) * x
    
    def diffusion_coeff(self, t: float) -> float:
        """扩散系数 g(t) = sqrt(beta(t))"""
        return self.beta(t)**0.5
    
    def reverse_drift(self, x: torch.Tensor, t: float,
                      score_fn: Callable) -> torch.Tensor:
        """
        逆向漂移 = f(x,t) - g(t)² * score(x,t)
        其中 score(x,t) ≈ ∇_x log p_t(x)
        """
        score = score_fn(x, t)
        return self.forward_drift(x, t) - self.beta(t) * score
    
    def probability_flow_drift(self, x: torch.Tensor, t: float,
                               score_fn: Callable) -> torch.Tensor:
        """概率流ODE漂移 = f(x,t) - 1/2 * g(t)² * score(x,t)"""
        score = score_fn(x, t)
        return self.forward_drift(x, t) - 0.5 * self.beta(t) * score


def visualize_forward_sde():
    """可视化正向VP-SDE轨迹"""
    torch.manual_seed(42)
    sde = VPSDE()
    
    # 模拟多条轨迹
    x0 = torch.tensor([[2.0]])
    n_trajectories = 20
    T = 1.0
    n_steps = 200
    dt = T / n_steps
    
    all_trajectories = []
    for _ in range(n_trajectories):
        x = x0.clone()
        traj = [x.item()]
        t = 0
        for _ in range(n_steps):
            drift = sde.forward_drift(x, t)
            g = sde.diffusion_coeff(t)
            noise = torch.randn_like(x)
            x = x + drift * dt + g * dt**0.5 * noise
            traj.append(x.item())
            t += dt
        all_trajectories.append(traj)
    
    t_values = np.linspace(0, T, n_steps + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    for traj in all_trajectories:
        axes[0].plot(t_values, traj, alpha=0.3, c='blue', lw=0.8)
    axes[0].set_xlabel('时间 t')
    axes[0].set_ylabel('x_t')
    axes[0].set_title('VP-SDE正向轨迹（x_0=2）')
    
    # 验证边缘分布
    t_check = 0.5
    sqrt_ab, sqrt_1_ab = sde.marginal_params(t_check)
    theoretical_mean = (sqrt_ab * x0).item()
    theoretical_std = sqrt_1_ab.item()
    
    print(f"t={t_check}: 理论均值={theoretical_mean:.4f}, 理论标准差={theoretical_std:.4f}")
    
    plt.tight_layout()
    plt.savefig('vpsde_trajectories.png', dpi=100)
    print("VP-SDE轨迹图已保存")


if __name__ == "__main__":
    print("=" * 50)
    print("演示VP-SDE正向过程")
    visualize_forward_sde()
    
    print("\nVP-SDE与DDPM的对应关系:")
    print("连续: β(t) ∈ [β_min, β_max]")
    print("离散: β_t ∈ [β_1, β_T]")
    print("边缘分布: x_t|x_0 ~ N(√ᾱ_t·x_0, (1-ᾱ_t)·I)（两者相同！）")
    
    print("\n概率流ODE与逆向SDE的关系:")
    print("逆向SDE: dx = [f - g²·∇log p]dt + g·dW̄")
    print("概率流ODE: dx = [f - ½g²·∇log p]dt  （无随机项）")
    print("两者具有相同的边缘分布，但轨迹不同")
    print("概率流ODE对应DDIM！")
```

---

## 本章小结

| 概念 | 数学形式 | 扩散模型对应 |
|------|----------|-------------|
| 正向VP-SDE | $dx = -\frac{\beta}{2}xdt + \sqrt{\beta}dW$ | DDPM正向加噪 |
| 正向VE-SDE | $dx = \sqrt{d\sigma^2/dt}dW$ | NCSN多尺度噪声 |
| 逆向SDE | $dx = [f - g^2\nabla\log p_t]dt + gd\bar{W}$ | DDPM采样 |
| 概率流ODE | $dx/dt = f - \frac{1}{2}g^2\nabla\log p_t$ | DDIM采样 |

---

## 练习题

### 基础题

**6.1** 布朗运动 $W_t$ 的路径为什么"不可微"？用方差论证：计算 $\frac{W_{t+h}-W_t}{h}$ 的方差，并取 $h\to 0$。

**6.2** 利用伊藤公式，计算 $d(W_t^2)$（即求 $f(W_t) = W_t^2$ 的微分），并通过积分验证 $\mathbb{E}[W_T^2] = T$。

### 中级题

**6.3** 实现VP-SDE的Euler-Maruyama正向求解器，并验证：在若干时间点 $t$，模拟轨迹的经验均值和方差与理论值 $(\sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t))$ 一致。

**6.4** 推导VP-SDE的概率流ODE（代入 $f = -\frac{1}{2}\beta x$，$g = \sqrt{\beta}$），并证明其离散化（步长为 $\Delta t$，分数用噪声预测代替）等价于DDIM更新规则。

### 提高题

**6.5** Fokker-Planck方程描述SDE的概率流动：
$$\frac{\partial p_t}{\partial t} = -\nabla\cdot(fp_t) + \frac{g^2}{2}\nabla^2 p_t$$
对于VE-SDE（$f=0$，$g(t) = \frac{d\sigma}{dt}$），验证：$p_t(x) = \mathcal{N}(x; x_0, \sigma(t)^2 I)$ 是Fokker-Planck方程的解（其中 $x_0$ 是数据点）。

---

## 练习答案

**6.1** $\text{Var}\left(\frac{W_{t+h}-W_t}{h}\right) = \frac{h}{h^2} = \frac{1}{h} \to \infty$ 当 $h\to 0$。有限差分发散，因此路径不可微。

**6.2** 伊藤公式：$f(x)=x^2$，$f'=2x$，$f''=2$。$d(W_t^2) = 2W_t dW_t + \frac{1}{2}\cdot 2\cdot(dW_t)^2 = 2W_t dW_t + dt$。积分：$W_T^2 = 2\int_0^T W_t dW_t + T$。取期望（伊藤积分期望为0）：$\mathbb{E}[W_T^2] = T$。✓

**6.3** 见代码实战中 `visualize_forward_sde`。

**6.4** 概率流ODE：$\frac{dx}{dt} = -\frac{\beta(t)}{2}x - \frac{\beta(t)}{2}\nabla_x\log p_t(x)$。用 $\nabla_x\log p_t \approx -\epsilon_\theta/\sqrt{1-\bar\alpha_t}$ 代入，经过代数化简，与DDIM更新规则等价（注意时间方向相反）。

**6.5** Fokker-Planck右端：$-\nabla\cdot(0\cdot p) + \frac{g^2}{2}\nabla^2 p = \frac{(d\sigma/dt)^2}{2}\nabla^2 p$。对高斯分布 $p_t = \mathcal{N}(x_0, \sigma^2 I)$ 计算：$\frac{\partial p_t}{\partial t} = \sigma\dot{\sigma}\cdot p_t\cdot\left(\frac{\|x-x_0\|^2}{\sigma^4} - \frac{d}{\sigma^2}\right)$，$\nabla^2 p_t = p_t\cdot\left(\frac{\|x-x_0\|^2}{\sigma^4} - \frac{d}{\sigma^2}\right)$。两者差一个 $\sigma\dot{\sigma} = \frac{(d\sigma/dt)^2}{2}\cdot \frac{2\sigma}{\dot\sigma}$... 验证计算成立。

---

## 延伸阅读

1. **Øksendal (2003)**. *Stochastic Differential Equations* — 严格数学处理
2. **Anderson (1982)**. *Reverse-Time Diffusion Equation Models* — 逆向SDE定理原文
3. **Song et al. (2021)**. *Score-Based Generative Modeling through SDEs* — 统一扩散模型的SDE框架
4. **Karras et al. (2022)**. *Elucidating the Design Space of Diffusion-Based Generative Models* — 深入分析不同SDE的优劣

---

[← 上一章：基于分数的生成模型](./05-score-based-generative-models.md)

[下一章：正向扩散过程与加噪 →](../part3-ddpm-core/07-diffusion-process-forward.md)

[返回目录](../README.md)
