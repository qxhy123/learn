# 第17章：三角代换

> 三角代换的本质不是“看到根式就背表”，而是借助单位圆恒等式把根式结构改写成更容易处理的三角结构。

## 学习目标

完成本章学习后，你将能够：

1. 理解三角代换背后的单位圆逻辑
2. 根据根式结构选择正确代换
3. 在代换后正确处理变量范围和符号
4. 把根式问题转化为三角恒等式问题
5. 为积分与解析技巧打下基础

---

## 正文内容

## 17.0 三角函数的导数与积分（前置知识）

三角代换是积分技巧，使用前需要掌握三角函数的导数和积分。

### 导数公式及推导

**核心公式**：

$$(\sin x)' = \cos x, \qquad (\cos x)' = -\sin x$$

**推导** $(\sin x)' = \cos x$（从定义出发）：

$$\frac{d}{dx}\sin x = \lim_{h\to 0}\frac{\sin(x+h)-\sin x}{h}$$

用和差化积：$\sin(x+h)-\sin x = 2\cos\frac{2x+h}{2}\sin\frac{h}{2}$，于是：

$$= \lim_{h\to 0}\frac{2\cos(x+h/2)\sin(h/2)}{h} = \lim_{h\to 0}\cos\left(x+\frac{h}{2}\right)\cdot\frac{\sin(h/2)}{h/2} = \cos x \cdot 1 = \cos x$$

其中用到了基本极限 $\lim_{u\to 0}\frac{\sin u}{u} = 1$（弧度制下成立，这正是第1章强调弧度制的原因）。

$(\cos x)'$ 可由 $\cos x = \sin(\pi/2-x)$ 结合链式法则得到。

**其余四个函数**的导数由商法则推出：

| 函数 | 导数 | 推导方法 |
|------|------|----------|
| $\tan x$ | $\sec^2 x$ | $(\sin x/\cos x)' = (\cos^2x+\sin^2x)/\cos^2x$ |
| $\cot x$ | $-\csc^2 x$ | 类似 |
| $\sec x$ | $\sec x\tan x$ | $(1/\cos x)' = \sin x/\cos^2 x$ |
| $\csc x$ | $-\csc x\cot x$ | 类似 |

### 基本积分公式

导数公式反过来就是积分公式：

| 积分 | 结果 |
|------|------|
| $\int\cos x\,dx$ | $\sin x + C$ |
| $\int\sin x\,dx$ | $-\cos x + C$ |
| $\int\sec^2 x\,dx$ | $\tan x + C$ |
| $\int\csc^2 x\,dx$ | $-\cot x + C$ |
| $\int\sec x\tan x\,dx$ | $\sec x + C$ |
| $\int\tan x\,dx$ | $-\ln|\cos x| + C$ |
| $\int\sec x\,dx$ | $\ln|\sec x+\tan x| + C$ |

### 反三角函数的导数

| 函数 | 导数 | 来源 |
|------|------|------|
| $\arcsin x$ | $\dfrac{1}{\sqrt{1-x^2}}$ | 反函数求导法 |
| $\arccos x$ | $-\dfrac{1}{\sqrt{1-x^2}}$ | $= -(\arcsin x)'$ |
| $\arctan x$ | $\dfrac{1}{1+x^2}$ | 反函数求导法 |

因此：$\int\frac{dx}{\sqrt{1-x^2}} = \arcsin x + C$，$\int\frac{dx}{1+x^2} = \arctan x + C$。

这两个积分在三角代换中频繁出现。

---

## 17.1 为什么三角代换有效

若表达式含有：

- $\sqrt{a^2-x^2}$
- $\sqrt{a^2+x^2}$
- $\sqrt{x^2-a^2}$

它们之所以适合三角代换，是因为：

- $1-\sin^2t=\cos^2t$
- $1+\tan^2t=\sec^2t$

也就是说，根式不是凭空消失，而是被三角恒等式吸收掉了。

---

## 17.2 三类经典代换

### 情形一：$\sqrt{a^2-x^2}$

取

$$
x=a\sin t
$$

则：

$$
\sqrt{a^2-x^2}=a\cos t
$$

### 情形二：$\sqrt{a^2+x^2}$

取

$$
x=a\tan t
$$

则：

$$
\sqrt{a^2+x^2}=a\sec t
$$

### 情形三：$\sqrt{x^2-a^2}$

取

$$
x=a\sec t
$$

则：

$$
\sqrt{x^2-a^2}=a\tan t
$$

---

## 17.3 例题：处理根式

令

$$
x=3\sin t
$$

则：

$$
\sqrt{9-x^2}=\sqrt{9-9\sin^2t}=3\sqrt{1-\sin^2t}=3|\cos t|
$$

若取范围

$$
t\in\left[-\frac\pi2,\frac\pi2\right]
$$

则 $\cos t\ge0$，因此：

$$
\sqrt{9-x^2}=3\cos t
$$

### 为什么范围这么重要

如果不限制 $t$ 的范围，就不能随便把 $|\cos t|$ 写成 $\cos t$。 
这就是三角代换里最常被忽略的细节。

---

### 完整例题：计算 $\int\frac{dx}{\sqrt{9-x^2}}$

**第1步**：识别结构 $\sqrt{a^2-x^2}$，取 $x = 3\sin t$，$dx = 3\cos t\,dt$。

**第2步**：代入并化简：

$$\int\frac{3\cos t\,dt}{\sqrt{9-9\sin^2t}} = \int\frac{3\cos t\,dt}{3\cos t} = \int dt = t + C$$

**第3步**：换回原变量。由 $x = 3\sin t$，得 $t = \arcsin\frac{x}{3}$，因此：

$$\int\frac{dx}{\sqrt{9-x^2}} = \arcsin\frac{x}{3} + C$$

### 完整例题：计算 $\int\frac{dx}{x^2+4}$

**第1步**：识别结构 $a^2+x^2$，取 $x = 2\tan t$，$dx = 2\sec^2 t\,dt$。

**第2步**：代入：

$$\int\frac{2\sec^2t\,dt}{4\tan^2t+4} = \int\frac{2\sec^2t\,dt}{4\sec^2t} = \frac{1}{2}\int dt = \frac{t}{2} + C$$

**第3步**：换回。$t = \arctan\frac{x}{2}$，因此：

$$\int\frac{dx}{x^2+4} = \frac{1}{2}\arctan\frac{x}{2} + C$$

---

## 17.4 三角代换和单位圆的关系

以 $x=a\sin t$ 为例，它的几何意义是：

- 把 $x$ 看成半径为 $a$ 的圆上纵坐标投影
- 则 $\sqrt{a^2-x^2}$ 就是对应横坐标长度

也就是说，三角代换背后其实仍然是单位圆 / 圆的几何结构。

---

## 17.5 常见误区与检查清单

- 是否代换后忘记求 $dx$？
- 是否忽略了参数范围，导致符号出错？
- 是否没有判断该选 $\sin,\tan$ 还是 $\sec$？
- 是否把代换结果停在中间，没有回代到原变量？

---

## 本章小结

| 根式结构 | 推荐代换 |
|----------|----------|
| $\sqrt{a^2-x^2}$ | $x=a\sin t$ |
| $\sqrt{a^2+x^2}$ | $x=a\tan t$ |
| $\sqrt{x^2-a^2}$ | $x=a\sec t$ |

---

## 分级例题精练

> 本节精选 6 道例题，分三档难度：**初中基础 ★** / **高中核心 ★★** / **高阶拓展 ★★★**（本章侧重高中核心与高阶拓展）。每题含【题目】【解】【点评】，建议先自行尝试再看解。

### 例题精练 1（★★ 高中核心）

**题目**：设 $x=2\sin t$，$t\in\left[-\dfrac\pi2,\dfrac\pi2\right]$，把 $\sqrt{4-x^2}$ 化成关于 $t$ 的最简形式，并求 $x=1$ 时 $t$ 的值与对应的 $\sqrt{4-x^2}$。

**解**：

$$
\sqrt{4-x^2}=\sqrt{4-4\sin^2t}=2\sqrt{1-\sin^2t}=2\sqrt{\cos^2t}=2|\cos t|
$$

因为 $t\in\left[-\dfrac\pi2,\dfrac\pi2\right]$ 时 $\cos t\ge0$，所以 $|\cos t|=\cos t$，得

$$
\sqrt{4-x^2}=2\cos t
$$

当 $x=1$ 时，$2\sin t=1$，$\sin t=\dfrac12$，在所取范围内 $t=\dfrac\pi6$。此时

$$
\sqrt{4-x^2}=2\cos\frac\pi6=2\cdot\frac{\sqrt3}{2}=\sqrt3
$$

直接验证：$\sqrt{4-1^2}=\sqrt3$ ✓。

**点评**：化简根式的灵魂在于 $\sqrt{\cos^2t}=|\cos t|$，再靠所选参数范围去掉绝对值。养成「先写绝对值、再用范围定号」的习惯，能根除三角代换里最常见的符号错误。

### 例题精练 2（★★ 高中核心）

**题目**：已知 $x=\sec t$，$t\in\left[0,\dfrac\pi2\right)$，化简 $\dfrac{\sqrt{x^2-1}}{x}$。

**解**：

$$
\sqrt{x^2-1}=\sqrt{\sec^2t-1}=\sqrt{\tan^2t}=|\tan t|
$$

在 $t\in\left[0,\dfrac\pi2\right)$ 上 $\tan t\ge0$，故 $\sqrt{x^2-1}=\tan t$。于是

$$
\frac{\sqrt{x^2-1}}{x}=\frac{\tan t}{\sec t}=\frac{\sin t/\cos t}{1/\cos t}=\sin t
$$

**点评**：这里用到恒等式 $\sec^2t-1=\tan^2t$，它正是 $\sqrt{x^2-a^2}$ 型选 $x=a\sec t$ 的根据。化简时把 $\tan,\sec$ 全部还原为 $\sin,\cos$ 往往能让比式瞬间约分。

### 例题精练 3（★★★ 高阶拓展）

**题目**：计算定积分 $\displaystyle\int_0^{a}\sqrt{a^2-x^2}\,dx$（$a>0$），并解释其几何意义。

**解**：

令 $x=a\sin t$，$dx=a\cos t\,dt$，取 $t\in\left[0,\dfrac\pi2\right]$（此区间上 $\cos t\ge0$）。换限：$x=0\Rightarrow t=0$；$x=a\Rightarrow \sin t=1\Rightarrow t=\dfrac\pi2$。被积式

$$
\sqrt{a^2-x^2}=a\cos t
$$

于是

$$
\int_0^{a}\sqrt{a^2-x^2}\,dx=\int_0^{\pi/2}a\cos t\cdot a\cos t\,dt=a^2\int_0^{\pi/2}\cos^2t\,dt
$$

用降幂公式 $\cos^2t=\dfrac{1+\cos 2t}{2}$：

$$
=a^2\int_0^{\pi/2}\frac{1+\cos 2t}{2}\,dt=\frac{a^2}{2}\left[t+\frac{\sin 2t}{2}\right]_0^{\pi/2}=\frac{a^2}{2}\left(\frac\pi2+0\right)=\frac{\pi a^2}{4}
$$

**点评**：定积分用三角代换时，换元的同时要换上下限，省去最后回代的步骤，更简洁。结果 $\dfrac{\pi a^2}{4}$ 恰是半径 $a$ 的圆在第一象限的四分之一面积——曲线 $y=\sqrt{a^2-x^2}$ 正是上半圆，这印证了代换的几何本质。

### 例题精练 4（★★★ 高阶拓展）

**题目**：计算不定积分 $\displaystyle\int\frac{dx}{(x^2+1)^{3/2}}$。

**解**：

识别 $\sqrt{x^2+1}$ 结构，令 $x=\tan t$，$dx=\sec^2t\,dt$，取 $t\in\left(-\dfrac\pi2,\dfrac\pi2\right)$（此时 $\sec t>0$）。则

$$
x^2+1=\tan^2t+1=\sec^2t,\qquad (x^2+1)^{3/2}=\sec^3t
$$

代入

$$
\int\frac{\sec^2t}{\sec^3t}\,dt=\int\frac{dt}{\sec t}=\int\cos t\,dt=\sin t+C
$$

还原：由 $x=\tan t$，构造直角三角形——对边 $x$、邻边 $1$、斜边 $\sqrt{x^2+1}$，故

$$
\sin t=\frac{x}{\sqrt{x^2+1}}
$$

因此

$$
\int\frac{dx}{(x^2+1)^{3/2}}=\frac{x}{\sqrt{x^2+1}}+C
$$

**点评**：分数次幂 $(x^2+1)^{3/2}$ 同样属于 $\sqrt{x^2+a^2}$ 家族。回代时画一个直角三角形读出 $\sin t$ 比硬套反函数更可靠。可对结果求导验证：$\dfrac{d}{dx}\dfrac{x}{\sqrt{x^2+1}}=\dfrac{(x^2+1)^{1/2}-x\cdot\frac{x}{\sqrt{x^2+1}}}{x^2+1}=\dfrac{1}{(x^2+1)^{3/2}}$ ✓。

### 例题精练 5（★★★ 高阶拓展）

**题目**：计算 $\displaystyle\int\frac{\sqrt{x^2-9}}{x}\,dx$（设 $x>3$）。

**解**：

令 $x=3\sec t$，$t\in\left[0,\dfrac\pi2\right)$（保证 $x>3$ 且 $\tan t\ge0$），则 $dx=3\sec t\tan t\,dt$，且

$$
\sqrt{x^2-9}=\sqrt{9\sec^2t-9}=3|\tan t|=3\tan t
$$

代入：

$$
\int\frac{3\tan t}{3\sec t}\cdot 3\sec t\tan t\,dt=\int 3\tan^2t\,dt=3\int(\sec^2t-1)\,dt=3(\tan t-t)+C
$$

还原：由 $\sec t=\dfrac{x}{3}$，得 $\tan t=\sqrt{\sec^2t-1}=\dfrac{\sqrt{x^2-9}}{3}$，$t=\operatorname{arcsec}\dfrac{x}{3}=\arccos\dfrac{3}{x}$。于是

$$
\int\frac{\sqrt{x^2-9}}{x}\,dx=3\cdot\frac{\sqrt{x^2-9}}{3}-3\arccos\frac{3}{x}+C=\sqrt{x^2-9}-3\arccos\frac{3}{x}+C
$$

**点评**：本题代换后出现 $\tan^2t$，要靠 $\tan^2t=\sec^2t-1$ 才能积分，这是 $\sec$ 代换的常见后续。还原 $t$ 时用 $\arccos\dfrac3x$ 等价于 $\operatorname{arcsec}\dfrac x3$，避免直接写不常用的反正割。

### 例题精练 6（★★★ 高阶拓展）

**题目**：用代数代换（而非传统三角代换）证明 $\displaystyle\int\frac{dx}{\sqrt{1-x^2}}=\arcsin x+C$ 与三角代换给出同一答案；并说明当 $x=\sin t$ 时为何要求 $t\in\left[-\dfrac\pi2,\dfrac\pi2\right]$。

**解**：

三角代换路径：令 $x=\sin t$，$dx=\cos t\,dt$，

$$
\int\frac{\cos t\,dt}{\sqrt{1-\sin^2t}}=\int\frac{\cos t}{|\cos t|}\,dt
$$

要让被积式化为 $\int dt=t+C$，必须 $|\cos t|=\cos t$，即 $\cos t\ge0$，故限定 $t\in\left[-\dfrac\pi2,\dfrac\pi2\right]$。在此区间上 $x=\sin t$ 是双射，其反函数恰为 $t=\arcsin x$（值域正是 $\left[-\dfrac\pi2,\dfrac\pi2\right]$），故得 $\arcsin x+C$。

与导数法对照：已知反三角函数导数 $(\arcsin x)'=\dfrac{1}{\sqrt{1-x^2}}$（见 17.0 节），故 $\displaystyle\int\frac{dx}{\sqrt{1-x^2}}=\arcsin x+C$，两条路径结果一致。

为何必须限定区间：若允许 $t$ 跑到 $\left(\dfrac\pi2,\dfrac{3\pi}{2}\right)$，则 $\cos t<0$，$\sqrt{1-\sin^2t}=-\cos t$，积分会得到 $-t+C$，符号出错；同时 $\sin t$ 在更大区间上不再单值，无法唯一回代。把 $t$ 锁在 $\arcsin$ 的主值区间，既保证 $\cos t\ge0$，又保证回代唯一。

**点评**：这道题把三角代换的两个隐性前提暴露了出来：去绝对值需要 $\cos t\ge0$，唯一回代需要 $\sin t$ 单射。$\arcsin$ 的主值区间 $\left[-\dfrac\pi2,\dfrac\pi2\right]$ 同时满足这两条，这正是教材默认取该区间的深层原因，而非死记硬背。

---

## 练习题

1. 为什么三角代换的本质是把根式交给恒等式处理？
2. 若 $x=5\tan t$，求 $\sqrt{25+x^2}$。 
3. 为什么三角代换一定要注意参数范围？
4. 给出一个会因忽略绝对值而错的例子。 
5. 解释三角代换和单位圆之间的几何联系。
