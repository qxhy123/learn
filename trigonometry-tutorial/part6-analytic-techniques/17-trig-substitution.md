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

## 练习题

1. 为什么三角代换的本质是把根式交给恒等式处理？
2. 若 $x=5\tan t$，求 $\sqrt{25+x^2}$。 
3. 为什么三角代换一定要注意参数范围？
4. 给出一个会因忽略绝对值而错的例子。 
5. 解释三角代换和单位圆之间的几何联系。
