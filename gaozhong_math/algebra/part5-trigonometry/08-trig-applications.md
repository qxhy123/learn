# 三角函数的应用

> **一例速记**：  
> 周期现象建模的核心模型：$y = A\sin(\omega t + \varphi) + b$  
> - $A$ = 振幅（最大值与最小值之差的一半）  
> - $T = \dfrac{2\pi}{\omega}$ = 周期（完成一次完整振荡的时间）  
> - $\varphi$ = 初相（决定"从哪里开始"）  
> - $b$ = 纵向平移（中心线位置）  
> 实际测量：仰角、俯角 + 解三角形（正弦定理 / 余弦定理）。

---

## 一、振幅、周期、相位的实际意义

### 1.1 标准模型 $y = A\sin(\omega t + \varphi) + b$

在实际问题中，时间通常用 $t$ 表示，$y$ 表示随时间变化的量（高度、温度、电压等）。

| 参数 | 数学含义 | 实际意义 |
|------|----------|----------|
| $A$（振幅） | $y$ 偏离中心线的最大距离 | 变化幅度，$A > 0$ |
| $\omega$（角频率） | 决定周期 $T = \dfrac{2\pi}{\omega}$ | $\omega$ 越大，变化越快 |
| $\varphi$（初相） | $t = 0$ 时的相位 | 决定初始状态 |
| $b$（纵移） | 函数中心线 $y = b$ | 均值水平 |

**最大值与最小值**：

$$y_{\max} = A + b, \qquad y_{\min} = -A + b$$

因此，已知最大值 $M$ 和最小值 $m$，反解参数：

$$A = \frac{M - m}{2}, \qquad b = \frac{M + m}{2}$$

### 1.2 从"已知特征"确定参数的流程

**给定的信息** → **对应参数**：

1. **最大值和最小值** → $A = \dfrac{M-m}{2}$，$b = \dfrac{M+m}{2}$
2. **周期 $T$** → $\omega = \dfrac{2\pi}{T}$
3. **初始条件**（如 $t = 0$ 时 $y = y_0$）→ 代入 $y_0 = A\sin\varphi + b$ 求 $\varphi$（结合其他条件确定 $\varphi$ 的范围）

---

## 二、周期现象建模

### 2.1 潮汐模型

某海湾的潮汐高度 $h$（m）随时间 $t$（h，以某天 $0$ 时为起点）近似满足正弦规律。已知：
- 一天内最高潮高 $6.0$ m，最低潮高 $0.4$ m
- 涨潮退潮一个完整周期约为 $12$ h
- $t = 2$ h 时达到最高潮

**建模过程**：

**第一步：确定振幅和纵移。**

$$A = \frac{6.0 - 0.4}{2} = 2.8, \qquad b = \frac{6.0 + 0.4}{2} = 3.2$$

**第二步：确定角频率。**

$$T = 12 \text{ h}, \qquad \omega = \frac{2\pi}{T} = \frac{2\pi}{12} = \frac{\pi}{6}$$

**第三步：确定初相。**

$t = 2$ h 时 $h$ 取最大值，即 $\sin\!\left(\dfrac{\pi}{6} \cdot 2 + \varphi\right) = 1$，所以：

$$\frac{\pi}{3} + \varphi = \frac{\pi}{2} \implies \varphi = \frac{\pi}{6}$$

**模型**：

$$h(t) = 2.8\sin\!\left(\frac{\pi}{6}t + \frac{\pi}{6}\right) + 3.2$$

**应用**：预测某时刻潮高，或确定潮高超过某阈值的时间段。

### 2.2 气温模型

某城市的月平均气温 $T$（°C）随月份 $t$（$t = 1$ 表示 1 月）近似呈正弦规律。已知：
- 最低气温约 $-8$ °C（1 月），最高气温约 $32$ °C（7 月）
- 周期为 $12$ 个月

**建模**：

$$A = \frac{32 - (-8)}{2} = 20, \qquad b = \frac{32 + (-8)}{2} = 12, \qquad \omega = \frac{2\pi}{12} = \frac{\pi}{6}$$

$t = 7$ 时取最大值：$\dfrac{\pi}{6} \cdot 7 + \varphi = \dfrac{\pi}{2}$，$\varphi = \dfrac{\pi}{2} - \dfrac{7\pi}{6} = -\dfrac{2\pi}{3}$。

$$T(t) = 20\sin\!\left(\frac{\pi}{6}t - \frac{2\pi}{3}\right) + 12$$

**验证**：$t = 1$：$T(1) = 20\sin\!\left(\dfrac{\pi}{6} - \dfrac{2\pi}{3}\right) + 12 = 20\sin\!\left(-\dfrac{\pi}{2}\right) + 12 = -20 + 12 = -8$ ✓；$t = 7$：$T(7) = 20\sin\!\left(\dfrac{7\pi}{6} - \dfrac{2\pi}{3}\right) + 12 = 20\sin\dfrac{\pi}{2} + 12 = 32$ ✓。

### 2.3 声波与电信号

**声波**：声音的振动频率决定音调。频率 $f = 440$ Hz（标准 $A$ 音）的声波模型：

$$y(t) = A\sin(2\pi \cdot 440 \cdot t + \varphi) = A\sin(880\pi t + \varphi)$$

周期 $T = \dfrac{1}{440}$ s $\approx 2.27 \times 10^{-3}$ s（极短，人耳感知为连续声音）。

**交流电**：电压 $u(t) = U_0\sin(\omega t + \varphi)$，其中 $U_0$ 为峰值电压，$\omega = 2\pi f$（$f = 50$ Hz 时 $\omega = 100\pi$）。

---

## 三、实际测量问题

### 3.1 仰角与俯角

**定义**：
- **仰角**：观测者视线向上与水平线的夹角（$0° \sim 90°$）
- **俯角**：观测者视线向下与水平线的夹角（$0° \sim 90°$）

**基本模型**（直角三角形）：

观测者在 $A$ 点，$B$ 在 $A$ 正上方高度 $h$ 处，$C$ 为 $B$ 在地面的投影（$AC = d$），则仰角 $\theta$ 满足 $\tan\theta = \dfrac{h}{d}$。

### 3.2 间接测距（两个观测点）

当无法直接测量目标距离时，可在两个已知距离的点观测目标角度，利用正弦定理解三角形。

**标准模型**：

两点 $A, B$ 相距已知（基线 $c = AB$），分别测目标 $C$ 的方位角 $\alpha$ 和 $\beta$（与 $AB$ 方向的夹角），则三角形 $ABC$ 中：
- $\angle A = \alpha$，$\angle B = \beta$，$\angle C = \pi - \alpha - \beta$
- 由正弦定理：$AC = \dfrac{c\sin\beta}{\sin(\alpha+\beta)}$，$BC = \dfrac{c\sin\alpha}{\sin(\alpha+\beta)}$

### 3.3 测高问题

**场景**：无法直接到达目标底部，利用两次观测求高度。

**步骤**：
1. 设未知量（目标高度 $h$，或某段水平距离）
2. 用仰角/俯角建立直角三角形中的 $\tan$ 关系
3. 联立方程组（两个方程，两个未知量）
4. 解方程，求 $h$

---

## 四、典型例题

### 例 1：建立正弦模型

**题目**：某工厂的用电量 $W$（万度）随时刻 $t$（时，$0 \leq t \leq 24$）近似满足 $W = A\sin(\omega t + \varphi) + b$。已知：在 $t = 2$ 时用电量最小为 $2$ 万度，$t = 14$ 时用电量最大为 $8$ 万度。建立模型并求 $t = 8$ 时的用电量。

**【解答】**

**确定参数**：

$$A = \frac{8-2}{2} = 3, \qquad b = \frac{8+2}{2} = 5$$

从最小值时刻 $t = 2$ 到最大值时刻 $t = 14$，经历半个周期：$\dfrac{T}{2} = 14 - 2 = 12$，$T = 24$，$\omega = \dfrac{2\pi}{24} = \dfrac{\pi}{12}$。

$t = 2$ 时取最小值：$\sin\!\left(\dfrac{\pi}{12} \times 2 + \varphi\right) = -1$，即：

$$\frac{\pi}{6} + \varphi = -\frac{\pi}{2} + 2k\pi$$

取 $k = 0$：$\varphi = -\dfrac{\pi}{2} - \dfrac{\pi}{6} = -\dfrac{2\pi}{3}$。

**模型**：$W = 3\sin\!\left(\dfrac{\pi}{12}t - \dfrac{2\pi}{3}\right) + 5$

**$t = 8$ 时**：

$$W(8) = 3\sin\!\left(\frac{8\pi}{12} - \frac{2\pi}{3}\right) + 5 = 3\sin\!\left(\frac{2\pi}{3} - \frac{2\pi}{3}\right) + 5 = 3\sin 0 + 5 = 5 \text{ 万度}$$

$$\boxed{W(8) = 5 \text{ 万度}}$$

> 验证：$t = 2$：$W = 3\sin\!\left(\dfrac{\pi}{6} - \dfrac{2\pi}{3}\right) + 5 = 3\sin\!\left(-\dfrac{\pi}{2}\right) + 5 = -3 + 5 = 2$ ✓；$t = 14$：$W = 3\sin\!\left(\dfrac{7\pi}{6} - \dfrac{2\pi}{3}\right) + 5 = 3\sin\dfrac{\pi}{2} + 5 = 8$ ✓。

---

### 例 2：仰角测高

**题目**：从地面上两点 $A, B$ 观测同一座建筑物的顶端 $C$。$A$ 在 $B$ 正东方，$AB = 20$ m。在 $A$ 点观测 $C$ 的仰角为 $30°$，在 $B$ 点观测 $C$ 的仰角为 $45°$，$C$ 在 $AB$ 的延长线北侧。求建筑物高度 $CD$（$D$ 为 $C$ 正下方地面点）。

**【解答】**

设 $AD = m$，$BD = n$，$CD = h$。

由 $A$ 点仰角 $30°$：$\tan 30° = \dfrac{h}{m}$，$m = h\sqrt{3}$。

由 $B$ 点仰角 $45°$：$\tan 45° = \dfrac{h}{n}$，$n = h$。

由几何关系（$A$ 在 $B$ 正东，$D$ 在两者北侧）：

$$AB^2 = AD^2 - BD^2 \quad \text{（当 A、B、D 形成直角时不对，需用距离关系）}$$

注意：$A, B, D$ 三点不共线（$D$ 在 $AB$ 延长线的北侧），用平面几何：

$$AB^2 = AD^2 + BD^2 - 2 \cdot AD \cdot BD\cos(\angle ADB) \quad \text{（但需知 }\angle ADB\text{）}$$

若题目假设 $D$ 在 $AB$ 的垂直平分线上（即 $AD = BD$），则 $m = n$，即 $h\sqrt{3} = h$，矛盾。

重新设：$D$ 在 $A$ 正北方（即 $AD \perp AB$），则：

$AD = m = h\sqrt{3}$（由仰角 $30°$），$BD = n = h$（由仰角 $45°$）。

$BD^2 = AD^2 + AB^2$（$D$ 在 $A$ 正北，$B$ 在 $A$ 正西 $20$ m 处）：

$$h^2 = 3h^2 + 400 \quad \Rightarrow \quad -2h^2 = 400 \quad \text{(矛盾)}$$

**正确设法**：$A, B$ 在同一水平线上，$D$ 是建筑底部。假设 $A, D$ 之间水平距离为 $d$，$\angle DAB = \theta$：

$$AD = d, \quad CD = h, \quad \tan 30° = \frac{h}{d}, \quad BD = \sqrt{d^2 + 400 - 2d\cdot 20\cos\theta}$$

此题需更具体的几何条件。以下给出简化版（$A, B, D$ 三点中 $D$ 在 $AB$ 延长线上）：

设 $BD = x$，$AD = x + 20$，则：

$$\tan 30° = \frac{h}{x+20}, \quad \tan 45° = \frac{h}{x}$$

由第二式：$h = x$。代入第一式：

$$\frac{x}{x+20} = \tan 30° = \frac{1}{\sqrt{3}} \implies \sqrt{3}x = x + 20 \implies x(\sqrt{3}-1) = 20$$

$$x = \frac{20}{\sqrt{3}-1} = \frac{20(\sqrt{3}+1)}{2} = 10(\sqrt{3}+1)$$

$$\boxed{h = 10(\sqrt{3}+1) \approx 27.3 \text{ m}}$$

---

### 例 3：求满足条件的时间段

**题目**：利用例 1 中的用电量模型 $W = 3\sin\!\left(\dfrac{\pi}{12}t - \dfrac{2\pi}{3}\right) + 5$，求一天中用电量不少于 $6.5$ 万度的时间段。

**【解答】**

解不等式 $W \geq 6.5$：

$$3\sin\!\left(\frac{\pi}{12}t - \frac{2\pi}{3}\right) + 5 \geq 6.5$$

$$\sin\!\left(\frac{\pi}{12}t - \frac{2\pi}{3}\right) \geq \frac{1}{2}$$

令 $u = \dfrac{\pi}{12}t - \dfrac{2\pi}{3}$，$t \in [0, 24]$ 时 $u \in \left[-\dfrac{2\pi}{3}, \dfrac{4\pi}{3}\right]$。

在 $\left[-\dfrac{2\pi}{3}, \dfrac{4\pi}{3}\right]$ 上解 $\sin u \geq \dfrac{1}{2}$：

正弦在 $\left[\dfrac{\pi}{6}, \dfrac{5\pi}{6}\right]$ 上大于等于 $\dfrac{1}{2}$（在一个周期内），该区间与 $\left[-\dfrac{2\pi}{3}, \dfrac{4\pi}{3}\right]$ 的交集即为 $\left[\dfrac{\pi}{6}, \dfrac{5\pi}{6}\right]$。

回代 $u = \dfrac{\pi}{12}t - \dfrac{2\pi}{3}$：

$$\frac{\pi}{6} \leq \frac{\pi}{12}t - \frac{2\pi}{3} \leq \frac{5\pi}{6}$$

各项加 $\dfrac{2\pi}{3}$：

$$\frac{\pi}{6} + \frac{2\pi}{3} \leq \frac{\pi}{12}t \leq \frac{5\pi}{6} + \frac{2\pi}{3}$$

$$\frac{5\pi}{6} \leq \frac{\pi}{12}t \leq \frac{3\pi}{2}$$

各项乘以 $\dfrac{12}{\pi}$：

$$10 \leq t \leq 18$$

$$\boxed{t \in [10, 18] \text{，即上午 10 时至下午 6 时，用电量不少于 6.5 万度}}$$

---

## 五、易错点

1. **$\varphi$ 的确定要联合 $\sin\varphi$ 和 $\cos\varphi$ 两个信息**：只用一个初始条件可能得到两个 $\varphi$ 候选值，需结合"模型的物理意义"（如题目中哪个时刻是最大值还是最小值）来唯一确定 $\varphi$。
2. **周期换算**：$T = \dfrac{2\pi}{\omega}$，注意 $\omega$ 是弧度频率，而 $f = \dfrac{1}{T}$ 是频率（单位 Hz）；日常用语"一天 24 小时一个周期"是 $T = 24$（小时），$\omega = \dfrac{2\pi}{24} = \dfrac{\pi}{12}$。
3. **建模时 $t$ 的单位**：分钟、小时、月、年——$\omega$ 的值随单位不同而不同，代入时确保单位一致。
4. **仰角问题中直角的位置**：仰角定义中，建筑高度 $CD$ 是**垂直于地面**的，直角在底部 $D$，而不是在观测点。画出清晰的示意图是解题的关键。
5. **解不等式时三角函数的范围**：$\sin u \geq k$ 的解集要在 $u$ 的实际范围内取，不是直接写通解。先确定 $u$ 的范围，再在该范围内找解集，最后回代求 $t$。

---

## 六、思路自测题

**自测 1**　已知某地气温 $T$（°C）满足 $T = A\sin(\omega t + \varphi) + b$，最高温 $38°C$（$t = 7$，7 月），最低温 $-2°C$（$t = 1$，1 月），周期 $12$ 月。确定参数 $A, b, \omega, \varphi$。

> 💡 提示：$A = \dfrac{38-(-2)}{2} = 20$，$b = \dfrac{38+(-2)}{2} = 18$（注意，这里纵移是 $18$，不是 $12$）；$\omega = \dfrac{2\pi}{12} = \dfrac{\pi}{6}$；$t = 7$ 时最大：$\dfrac{\pi}{6}\times 7 + \varphi = \dfrac{\pi}{2}$，$\varphi = \dfrac{\pi}{2} - \dfrac{7\pi}{6} = -\dfrac{2\pi}{3}$。

**自测 2**　电压 $u = 220\sqrt{2}\sin(100\pi t + \varphi)$ V，$t = 0$ 时 $u = 110\sqrt{2}$ V 且电压正在增大，求 $\varphi$（$|\varphi| \leq \dfrac{\pi}{2}$）。

> 💡 提示：$t = 0$：$\sin\varphi = \dfrac{110\sqrt{2}}{220\sqrt{2}} = \dfrac{1}{2}$，$\varphi = \dfrac{\pi}{6}$ 或 $\varphi = \dfrac{5\pi}{6}$（舍去，因为 $\dfrac{5\pi}{6} > \dfrac{\pi}{2}$）。验证增大：$u' = 220\sqrt{2} \cdot 100\pi\cos(100\pi t + \varphi)$，$t = 0$ 时 $\cos\dfrac{\pi}{6} > 0$，确实增大 ✓。故 $\varphi = \dfrac{\pi}{6}$。

**自测 3**　在地面上 $P$ 点测得山顶 $Q$ 的仰角为 $60°$，沿水平地面走 $50$ m 到 $R$ 点，$P, R$ 在同一直线上且远离山脚，测得仰角为 $30°$，求山高 $h$（设山脚在 $P, R$ 连线的延长线上）。

> 💡 提示：设山脚到 $R$ 的水平距离为 $x$，$\tan 30° = \dfrac{h}{x+50}$，$\tan 60° = \dfrac{h}{x}$，由第二式 $x = \dfrac{h}{\sqrt{3}}$，代入第一式：$\dfrac{h}{h/\sqrt{3}+50} = \sqrt{3}$，$h = \sqrt{3}(h/\sqrt{3}+50) = h + 50\sqrt{3}$，矛盾！说明 $P$ 比 $R$ 更靠近山脚。重设 $P$ 到山脚水平距离为 $d$：$\tan 60° = \dfrac{h}{d}$（$P$ 更近），$\tan 30° = \dfrac{h}{d+50}$（$R$ 更远）。则 $d = \dfrac{h}{\sqrt{3}}$，$d+50 = h\sqrt{3}$，相减：$50 = h\sqrt{3} - \dfrac{h}{\sqrt{3}} = \dfrac{2h}{\sqrt{3}}$，$h = 25\sqrt{3}$ m。

**自测 4**　利用模型 $h = 2.8\sin\!\left(\dfrac{\pi}{6}t + \dfrac{\pi}{6}\right) + 3.2$（潮汐高度），求一天中潮高低于 $2$ m 的时间段（$0 \leq t < 24$）。

> 💡 提示：解 $2.8\sin\!\left(\dfrac{\pi}{6}t + \dfrac{\pi}{6}\right) + 3.2 < 2$，即 $\sin\!\left(\dfrac{\pi}{6}t + \dfrac{\pi}{6}\right) < -\dfrac{3}{7}$。设 $u = \dfrac{\pi}{6}t + \dfrac{\pi}{6} \in \left[\dfrac{\pi}{6}, \dfrac{25\pi}{6}\right]$，在此范围内解 $\sin u < -\dfrac{3}{7}$，找 $u$ 的解区间，再回代求 $t$（具体数值需计算 $\arcsin\dfrac{3}{7} \approx 25.4°$）。

---

**回头看一眼"一例速记"**：

> $y = A\sin(\omega t + \varphi) + b$：$A = \dfrac{M-m}{2}$，$b = \dfrac{M+m}{2}$，$T = \dfrac{2\pi}{\omega}$。  
> 测高：画图 + $\tan\theta$ 建方程组 + 解方程。

如果现在你能不看笔记，从"最大值 8，最小值 2，周期 12，$t=8$ 时取最大"直接写出模型——本章，你拿下了。
