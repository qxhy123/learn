# 附录A：三角函数公式速查表

## 1. 基本关系

$$\sin^2x+\cos^2x=1$$

$$\tan x=\frac{\sin x}{\cos x},\quad \cot x=\frac{\cos x}{\sin x}$$

$$\sec x=\frac{1}{\cos x},\quad \csc x=\frac{1}{\sin x}$$

$$1+\tan^2x=\sec^2x,\quad 1+\cot^2x=\csc^2x$$

---

## 2. 和差公式

$$\sin(\alpha\pm\beta)=\sin\alpha\cos\beta\pm\cos\alpha\sin\beta$$

$$\cos(\alpha\pm\beta)=\cos\alpha\cos\beta\mp\sin\alpha\sin\beta$$

$$\tan(\alpha\pm\beta)=\frac{\tan\alpha\pm\tan\beta}{1\mp\tan\alpha\tan\beta}$$

---

## 3. 倍角与半角

$$\sin2x=2\sin x\cos x$$

$$\cos2x=\cos^2x-\sin^2x=2\cos^2x-1=1-2\sin^2x$$

$$\tan2x=\frac{2\tan x}{1-\tan^2x}$$

**降幂公式**（由倍角公式得）：

$$\sin^2x=\frac{1-\cos2x}{2},\quad \cos^2x=\frac{1+\cos2x}{2}$$

**半角公式**：

$$\sin\frac{x}{2}=\pm\sqrt{\frac{1-\cos x}{2}},\quad \cos\frac{x}{2}=\pm\sqrt{\frac{1+\cos x}{2}}$$

$$\tan\frac{x}{2}=\frac{\sin x}{1+\cos x}=\frac{1-\cos x}{\sin x}$$

**三倍角公式**：

$$\sin3x=3\sin x-4\sin^3x$$

$$\cos3x=4\cos^3x-3\cos x$$

---

## 4. 积化和差与和差化积

**积化和差**：

$$\sin A\cos B=\frac{1}{2}[\sin(A+B)+\sin(A-B)]$$

$$\cos A\cos B=\frac{1}{2}[\cos(A-B)+\cos(A+B)]$$

$$\sin A\sin B=\frac{1}{2}[\cos(A-B)-\cos(A+B)]$$

**和差化积**：

$$\sin A+\sin B=2\sin\frac{A+B}{2}\cos\frac{A-B}{2}$$

$$\sin A-\sin B=2\cos\frac{A+B}{2}\sin\frac{A-B}{2}$$

$$\cos A+\cos B=2\cos\frac{A+B}{2}\cos\frac{A-B}{2}$$

$$\cos A-\cos B=-2\sin\frac{A+B}{2}\sin\frac{A-B}{2}$$

---

## 5. 辅助角公式

$$a\sin x+b\cos x=\sqrt{a^2+b^2}\sin(x+\varphi),\quad \tan\varphi=\frac{b}{a}$$

---

## 6. 万能代换

令 $t=\tan\dfrac{x}{2}$，则：

$$\sin x=\frac{2t}{1+t^2},\quad \cos x=\frac{1-t^2}{1+t^2},\quad dx=\frac{2}{1+t^2}dt$$

---

## 7. 反三角函数

| 函数 | 定义域 | 值域 | 含义 |
|------|--------|------|------|
| $\arcsin x$ | $[-1,1]$ | $[-\pi/2,\pi/2]$ | 正弦为 $x$ 的角 |
| $\arccos x$ | $[-1,1]$ | $[0,\pi]$ | 余弦为 $x$ 的角 |
| $\arctan x$ | $\mathbb{R}$ | $(-\pi/2,\pi/2)$ | 正切为 $x$ 的角 |

**互余关系**：$\arcsin x+\arccos x=\dfrac{\pi}{2}$

**导数**：

$$(\arcsin x)'=\frac{1}{\sqrt{1-x^2}},\quad (\arccos x)'=-\frac{1}{\sqrt{1-x^2}},\quad (\arctan x)'=\frac{1}{1+x^2}$$

---

## 8. 三角形

**正弦定理**：$\dfrac{a}{\sin A}=\dfrac{b}{\sin B}=\dfrac{c}{\sin C}=2R$

**余弦定理**：$c^2=a^2+b^2-2ab\cos C$

**面积公式**：$S=\dfrac{1}{2}ab\sin C=\dfrac{abc}{4R}=rs$（$r$为内切圆半径，$s$为半周长）

**海伦公式**：$S=\sqrt{s(s-a)(s-b)(s-c)}$，$s=\dfrac{a+b+c}{2}$

---

## 9. 导数与积分

| 函数 | 导数 |
|------|------|
| $\sin x$ | $\cos x$ |
| $\cos x$ | $-\sin x$ |
| $\tan x$ | $\sec^2 x$ |
| $\cot x$ | $-\csc^2 x$ |
| $\sec x$ | $\sec x\tan x$ |
| $\csc x$ | $-\csc x\cot x$ |

| 积分 | 结果 |
|------|------|
| $\int\sin x\,dx$ | $-\cos x+C$ |
| $\int\cos x\,dx$ | $\sin x+C$ |
| $\int\tan x\,dx$ | $-\ln|\cos x|+C$ |
| $\int\sec^2x\,dx$ | $\tan x+C$ |
| $\int\csc^2x\,dx$ | $-\cot x+C$ |
| $\int\sec x\,dx$ | $\ln|\sec x+\tan x|+C$ |

---

## 10. 三角代换

| 被积式含有 | 代换 | 区间 |
|-----------|------|------|
| $\sqrt{a^2-x^2}$ | $x=a\sin\theta$ | $\theta\in[-\pi/2,\pi/2]$ |
| $\sqrt{a^2+x^2}$ | $x=a\tan\theta$ | $\theta\in(-\pi/2,\pi/2)$ |
| $\sqrt{x^2-a^2}$ | $x=a\sec\theta$ | $\theta\in[0,\pi/2)\cup(\pi/2,\pi]$ |

---

## 11. 欧拉公式与复数

$$e^{ix}=\cos x+i\sin x$$

$$\cos x=\frac{e^{ix}+e^{-ix}}{2},\quad \sin x=\frac{e^{ix}-e^{-ix}}{2i}$$

**De Moivre 公式**：$(\cos\theta+i\sin\theta)^n=\cos n\theta+i\sin n\theta$

**$n$ 次单位根**：$\omega_k=e^{i2k\pi/n}$，$k=0,1,\ldots,n-1$

---

## 12. 旋转矩阵

$$R(\theta)=\begin{pmatrix}\cos\theta & -\sin\theta \\ \sin\theta & \cos\theta\end{pmatrix}$$

性质：$R(\alpha)R(\beta)=R(\alpha+\beta)$，$R(\theta)^{-1}=R(-\theta)=R(\theta)^T$

---

## 13. 重要极限

$$\lim_{x\to 0}\frac{\sin x}{x}=1,\quad \lim_{x\to 0}\frac{\tan x}{x}=1,\quad \lim_{x\to 0}\frac{1-\cos x}{x^2}=\frac{1}{2}$$
