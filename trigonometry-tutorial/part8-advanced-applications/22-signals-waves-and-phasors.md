# 第22章：信号、波与相量

> 当三角函数进入工程系统时，它们最重要的身份不再是“可求值函数”，而是描述振动、波、交流和相位关系的语言。

## 学习目标

完成本章学习后，你将能够：

1. 理解频率、相位、振幅在信号中的意义
2. 理解拍频、调制和相量的基本直觉
3. 把多个正弦波的叠加看成结构问题而不是纯代数问题
4. 解释为什么相量可以简化交流问题
5. 把本章和傅里叶、欧拉公式联系起来

---

## 正文内容

## 22.1 三角函数为什么天然适合描述波

波和振动通常具备两个核心特征：

- 周期性
- 平滑性

因此它们常写成：

$$
A\cos(\omega t+\phi)
$$

其中：

- $A$：振幅
- $\omega$：角频率
- $\phi$：相位

这说明三角函数进入工程后，最重要的是参数解释，而不是孤立求值。

---

## 22.2 相位到底是什么

相位决定的是“起始位置”。

两个频率相同但相位不同的波：

- 峰值出现时间不同
- 叠加结果会不同
- 某些情况下甚至会部分抵消

所以相位不是小修正，而是系统行为的重要组成部分。

---

## 22.3 拍频现象

拍频最常见于两个接近频率波的叠加：

$$
\sin101t+\sin99t
$$

用和差化积：

$$
\sin101t+\sin99t=2\sin100t\cos t
$$

因此：

- $\sin100t$ 表示快速振荡
- $2\cos t$ 表示慢变化包络

这就是拍频结构：快振荡被慢包络调制。

---

## 22.4 相量为什么重要

相量可以把正弦波看成复平面上的旋转向量。 
这样一来：

- 相位差变成角度差
- 叠加问题变成向量相加
- 微分方程问题常常更容易处理

所以相量不是新物理对象，而是一个更高效的表示系统。

---

## 22.5 例题：同频不同相位叠加（完整计算）

**问题**：求 $\cos t + \cos\left(t+\frac{\pi}{3}\right)$ 的合成结果。

**方法一：和差化积**

$$\cos t + \cos\left(t+\frac{\pi}{3}\right) = 2\cos\frac{t+(t+\pi/3)}{2}\cos\frac{t-(t+\pi/3)}{2}$$

$$= 2\cos\left(t+\frac{\pi}{6}\right)\cos\left(-\frac{\pi}{6}\right) = 2\cos\frac{\pi}{6}\cos\left(t+\frac{\pi}{6}\right) = \sqrt{3}\cos\left(t+\frac{\pi}{6}\right)$$

**方法二：相量法**

将两个余弦波视为复平面上的向量：$\mathbf{V}_1 = 1\angle 0°$ 和 $\mathbf{V}_2 = 1\angle 60°$。

$$\mathbf{V}_1 + \mathbf{V}_2 = (1+\cos 60°) + i\sin 60° = \frac{3}{2} + i\frac{\sqrt{3}}{2}$$

合成振幅：$A = \sqrt{(3/2)^2+(\sqrt{3}/2)^2} = \sqrt{3}$

合成相位：$\varphi = \arctan\frac{\sqrt{3}/2}{3/2} = \arctan\frac{1}{\sqrt{3}} = \frac{\pi}{6}$

结果：$\sqrt{3}\cos(t+\pi/6)$。两种方法完全一致。

**要点**：相量法在同频叠加中特别高效——把三角运算转为向量加法。当叠加超过2个波时，相量法的优势更加明显。

---

## 22.6 本章小结

| 主题 | 结论 |
|------|------|
| 波形参数 | 振幅、频率、相位缺一不可 |
| 拍频 | 来自相近频率的叠加 |
| 相量 | 把波形问题转成旋转向量问题 |
| 连接章节 | 与和差化积、欧拉公式、Fourier 紧密相连 |

---

## 分级例题精练

> 本节精选 6 道例题，分三档难度：**初中基础 ★** / **高中核心 ★★** / **高阶拓展 ★★★**（本章侧重高中核心与高阶拓展）。每题含【题目】【解】【点评】，建议先自行尝试再看解。

### 例题精练 1（★★ 高中核心）

**题目**：某交流电压可写成 $u(t)=12\sin\!\left(100\pi t+\dfrac{\pi}{6}\right)$（单位：伏，$t$ 单位：秒）。求它的振幅、角频率、频率、周期与初相位。

**解**：把它和标准形式 $A\sin(\omega t+\phi)$ 逐项对照：

- 振幅 $A=12$ 伏。
- 角频率 $\omega=100\pi\ \text{rad/s}$。
- 频率 $f=\dfrac{\omega}{2\pi}=\dfrac{100\pi}{2\pi}=50\ \text{Hz}$。
- 周期 $T=\dfrac{1}{f}=\dfrac{1}{50}=0.02\ \text{s}$（也可由 $T=\dfrac{2\pi}{\omega}=\dfrac{2\pi}{100\pi}=0.02$ 得到）。
- 初相位 $\phi=\dfrac{\pi}{6}$，即波形相对 $\sin(100\pi t)$ 在时间上**超前**了 $\dfrac{\phi}{\omega}=\dfrac{\pi/6}{100\pi}=\dfrac{1}{600}\ \text{s}$。

**点评**：读参数是信号分析的“第一语言”。注意区分角频率 $\omega$（每秒弧度）与频率 $f$（每秒周期），二者差一个 $2\pi$。我国工频交流电正是 $50\ \text{Hz}$，这道题的数字并非随手编的。

### 例题精练 2（★★ 高中核心）

**题目**：把信号 $s(t)=3\cos(2t)+4\sin(2t)$ 化成单一正弦形式 $A\sin(2t+\phi)$，并求其振幅与相位。

**解**：用辅助角公式。设 $A\sin(2t+\phi)=A\cos\phi\sin 2t+A\sin\phi\cos 2t$，与原式对照（按 $\sin 2t$、$\cos 2t$ 系数）：

$$A\cos\phi=4,\qquad A\sin\phi=3.$$

于是振幅

$$A=\sqrt{4^2+3^2}=\sqrt{25}=5,$$

相位

$$\tan\phi=\frac{A\sin\phi}{A\cos\phi}=\frac{3}{4},\qquad \phi=\arctan\frac34\approx 0.6435\ \text{rad}.$$

由于 $A\cos\phi=4>0$、$A\sin\phi=3>0$，$\phi$ 在第一象限，取值正确。故

$$s(t)=5\sin\!\left(2t+\arctan\tfrac34\right).$$

**点评**：“同频的正弦加余弦”一定能合成单一正弦波，振幅为两系数的平方和开根。这一步是相量法的代数版本——稍后第 4 题会看到它的向量解释。务必用两个分量的**符号**确定 $\phi$ 所在象限，单凭 $\arctan$ 会丢象限信息。

### 例题精练 3（★★ 高中核心）

**题目**：把正弦波 $v(t)=6\cos\!\left(\omega t-\dfrac{\pi}{4}\right)$ 表示成相量（取余弦为基准），并说明该相量在复平面上的位置。

**解**：以 $\cos(\omega t+\phi)\leftrightarrow A\angle\phi$ 的约定，把振幅作为模、初相作为辐角：

$$v(t)=6\cos\!\left(\omega t-\tfrac{\pi}{4}\right)\ \Longleftrightarrow\ \mathbf V=6\angle\!\left(-\tfrac{\pi}{4}\right).$$

写成复数直角坐标：

$$\mathbf V=6\left(\cos\!\left(-\tfrac{\pi}{4}\right)+i\sin\!\left(-\tfrac{\pi}{4}\right)\right)=6\left(\tfrac{\sqrt2}{2}-i\tfrac{\sqrt2}{2}\right)=3\sqrt2-3\sqrt2\,i.$$

它是复平面上模长为 $6$、与正实轴成 $-45^\circ$ 角的向量，位于第四象限。

**点评**：相量把“随时间旋转的波”冻结成一个不随时间变的复数——因为公共因子 $e^{i\omega t}$ 对同频的所有波都一样，可以约去。模就是振幅，辐角就是初相。从此“波的叠加”就变成“向量相加”。

### 例题精练 4（★★★ 高阶拓展）

**题目**：用相量法求同频叠加 $y(t)=5\cos(\omega t)+5\cos\!\left(\omega t+\dfrac{2\pi}{3}\right)+5\cos\!\left(\omega t-\dfrac{2\pi}{3}\right)$。

**解**：三个波振幅都是 $5$，相位分别为 $0,\ \dfrac{2\pi}{3},\ -\dfrac{2\pi}{3}$（互差 $120^\circ$）。对应相量：

$$\mathbf V_1=5\angle 0,\quad \mathbf V_2=5\angle\tfrac{2\pi}{3},\quad \mathbf V_3=5\angle\!\left(-\tfrac{2\pi}{3}\right).$$

化为直角坐标相加。利用 $\cos\dfrac{2\pi}{3}=-\dfrac12$、$\sin\dfrac{2\pi}{3}=\dfrac{\sqrt3}{2}$：

$$\mathbf V_1=5,\quad \mathbf V_2=5\left(-\tfrac12+i\tfrac{\sqrt3}{2}\right)=-\tfrac52+i\tfrac{5\sqrt3}{2},\quad \mathbf V_3=-\tfrac52-i\tfrac{5\sqrt3}{2}.$$

求和：

$$\mathbf V_1+\mathbf V_2+\mathbf V_3=\left(5-\tfrac52-\tfrac52\right)+i\left(\tfrac{5\sqrt3}{2}-\tfrac{5\sqrt3}{2}\right)=0+0i=0.$$

合成相量为零，故 $y(t)\equiv 0$。

**点评**：三个等幅、互差 $120^\circ$ 的同频波叠加恰好抵消——这正是三相交流电平衡时中性线电流为零的数学原因。相量法把一道看似要反复用和差化积的题，压缩成“三个等长向量首尾相接成正三角形，合矢量为零”的一眼结论。

### 例题精练 5（★★★ 高阶拓展）

**题目**：两个相近频率的波叠加为 $f(t)=\cos(2\pi\cdot 440\,t)+\cos(2\pi\cdot 444\,t)$（单位：秒）。分析其拍频结构，并求每秒听到几次“强弱起伏”（拍频）。

**解**：用和差化积 $\cos\alpha+\cos\beta=2\cos\dfrac{\alpha+\beta}{2}\cos\dfrac{\alpha-\beta}{2}$。令 $\alpha=2\pi\cdot 440\,t$、$\beta=2\pi\cdot 444\,t$：

$$f(t)=2\cos\!\Big(2\pi\cdot\tfrac{440+444}{2}\,t\Big)\cos\!\Big(2\pi\cdot\tfrac{440-444}{2}\,t\Big)=2\cos(2\pi\cdot 442\,t)\,\cos(2\pi\cdot(-2)\,t).$$

由于余弦为偶函数，$\cos(2\pi\cdot(-2)t)=\cos(2\pi\cdot 2\,t)$，所以

$$f(t)=2\cos(2\pi\cdot 2\,t)\,\cos(2\pi\cdot 442\,t).$$

结构为：载波 $\cos(2\pi\cdot 442\,t)$ 频率 $442\ \text{Hz}$（快振荡），包络 $2\cos(2\pi\cdot 2\,t)$ 频率 $2\ \text{Hz}$（慢调制）。

注意人耳感知的“强弱”取决于幅度 $|2\cos(2\pi\cdot 2\,t)|$，它在一个包络周期内出现**两次**极大（因为取了绝对值，$\cos$ 的正负峰都对应响），所以拍频为包络频率的两倍：

$$f_{\text{beat}}=2\times 2=4\ \text{Hz}=|444-440|\ \text{Hz}.$$

即每秒听到 $4$ 次强弱起伏。

**点评**：拍频等于两频率之差 $|f_1-f_2|$，而包络（半差频率）只有它的一半——区别在于响度由振幅的**绝对值**决定，正负峰都“响”。调音师正是靠把拍频调到零来对准两根弦的音高。

### 例题精练 6（★★★ 高阶拓展）

**题目**：把信号 $g(t)=\sin\!\left(\omega t+\dfrac{\pi}{6}\right)+\sqrt3\,\cos\!\left(\omega t+\dfrac{\pi}{6}\right)$ 合成为单一正弦波 $A\sin(\omega t+\Phi)$，用相量法给出 $A$ 与 $\Phi$。

**解**：令 $\theta=\omega t+\dfrac{\pi}{6}$，原式为 $\sin\theta+\sqrt3\cos\theta$。以正弦为基准把每项写成相量（$\sin(\theta+\varphi)\leftrightarrow 1\angle\varphi$）：

- $\sin\theta=\sin(\theta+0)\ \Leftrightarrow\ 1\angle 0$；
- $\sqrt3\cos\theta=\sqrt3\sin\!\left(\theta+\dfrac{\pi}{2}\right)\ \Leftrightarrow\ \sqrt3\angle\dfrac{\pi}{2}$（因 $\cos\theta=\sin(\theta+\tfrac{\pi}{2})$）。

相量相加（直角坐标）：

$$\mathbf P=1\angle 0+\sqrt3\angle\tfrac{\pi}{2}=(1+0i)+(0+\sqrt3\,i)=1+\sqrt3\,i.$$

合成振幅与相位：

$$A=\sqrt{1^2+(\sqrt3)^2}=\sqrt4=2,\qquad \tan(\text{辐角})=\frac{\sqrt3}{1}=\sqrt3\ \Rightarrow\ \text{辐角}=\frac{\pi}{3}.$$

所以 $\sin\theta+\sqrt3\cos\theta=2\sin\!\left(\theta+\dfrac{\pi}{3}\right)$。代回 $\theta=\omega t+\dfrac{\pi}{6}$：

$$g(t)=2\sin\!\left(\omega t+\frac{\pi}{6}+\frac{\pi}{3}\right)=2\sin\!\left(\omega t+\frac{\pi}{2}\right)=2\cos(\omega t).$$

故 $A=2$，$\Phi=\dfrac{\pi}{2}$。

**点评**：把余弦先转成“超前 $90^\circ$ 的正弦”，叠加就统一在同一基准下，再做向量加法即可。最终化简出 $2\cos(\omega t)$ 是个漂亮的副产品，说明这一组相位安排恰好把信号还原成纯余弦。这正是相量法的威力：所有同频运算都归结为复平面上的加减。

---

## 练习题

1. 为什么三角函数天然适合描述波和振动？
2. 拍频现象的结构本质是什么？
3. 相位改变会对波形产生什么影响？
4. 为什么相量能让同频叠加问题更容易理解？
5. 设计一道需要同时用到和差化积和相量直觉的题。
