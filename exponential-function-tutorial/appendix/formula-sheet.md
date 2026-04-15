# 公式速查表

> 这一页只收“高频骨架公式”。使用时请同时注意各条公式的适用前提：底数通常要求 $a>0,\ a\neq1$；对数真数要求为正；时间模型要先统一单位。

## 1. 指数运算律

| 公式 | 含义 |
|------|------|
| $a^m a^n=a^{m+n}$ | 同底数幂相乘，指数相加（同底数，且相关幂在当前范围内有定义） |
| $\dfrac{a^m}{a^n}=a^{m-n}$ | 同底数幂相除，指数相减（分母不能为 0） |
| $(a^m)^n=a^{mn}$ | 幂的乘方 |
| $(ab)^x=a^xb^x$ | 乘积的幂（在当前取值范围内有定义时使用） |
| $\left(\dfrac{a}{b}\right)^x=\dfrac{a^x}{b^x}$ | 商的幂（$b\neq0$） |
| $a^{-n}=\dfrac{1}{a^n}$ | 负指数规则（要求 $a\neq0$） |
| $a^{m/n}=\sqrt[n]{a^m}$ | 分数指数与根式关系（实数范围下尤其要检查定义条件） |

---

## 2. 指数与对数

| 公式 | 含义 |
|------|------|
| $a^x=y \iff x=\log_a y$ | 指数与对数互为反函数（要求 $a>0,\ a\neq1,\ y>0$） |
| $\log_a(xy)=\log_a x+\log_a y$ | 乘法转加法（要求 $x>0,\ y>0$） |
| $\log_a(x^r)=r\log_a x$ | 幂可提到前面（要求真数在当前范围内有定义，初学时优先在 $x>0$ 下使用） |
| $\log_a x=\dfrac{\ln x}{\ln a}$ | 换底公式（要求 $x>0,\ a>0,\ a\neq1$） |
| $a^x=e^{x\ln a}$ | 一般底数转自然底数（要求 $a>0$） |

---

## 3. 极限与增长比较

| 公式 / 结论 | 含义 |
|-------------|------|
| $\displaystyle e=\lim_{n\to\infty}\left(1+\frac1n\right)^n$ | 自然常数定义 |
| $\displaystyle \lim_{n\to\infty}\left(1+\frac{x}{n}\right)^n=e^x$ | 连续增长极限 |
| $\ln x \ll x^\alpha \ll a^x$ | 对数、幂、指数的增长层级 |
| $\displaystyle\lim_{h\to 0}\frac{e^h-1}{h}=1$ | 指数函数在0处的变化率 |
| $\displaystyle\lim_{h\to 0}\frac{a^h-1}{h}=\ln a$ | 一般指数的变化率 |
| $e^x \geq 1+x$（$\forall x$） | 指数函数的切线不等式（凸性） |
| $e^x \approx 1+x$（$|x| \ll 1$） | 一阶近似，误差 $O(x^2)$ |
| $e^x \approx 1+x+\frac{x^2}{2}$（$|x|$ 较小） | 二阶近似，误差 $O(x^3)$ |

---

## 4. 导数与积分

| 公式 | 结果 |
|------|------|
| $\dfrac{d}{dx}e^x=e^x$ | 自然指数函数求导 |
| $\dfrac{d}{dx}a^x=a^x\ln a$ | 一般指数函数求导（$a>0,\ a\neq1$） |
| $\dfrac{d}{dx}e^{g(x)}=e^{g(x)}g'(x)$ | 复合指数求导 |
| $\int e^x\,dx=e^x+C$ | 自然指数函数积分 |
| $\int a^x\,dx=\dfrac{a^x}{\ln a}+C$ | 一般指数函数积分（$a>0,\ a\neq1$） |
| $\int e^{kx}\,dx=\dfrac1k e^{kx}+C$ | 常系数指数积分（$k\neq0$） |

---

## 5. 级数展开

| 公式 | 结果 |
|------|------|
| $e^x=\displaystyle\sum_{n=0}^{\infty}\frac{x^n}{n!}$ | 指数函数幂级数 |
| $e^{-x}=\displaystyle\sum_{n=0}^{\infty}\frac{(-1)^n x^n}{n!}$ | 负指数展开 |

---

## 6. 复指数与双曲函数

| 公式 | 结果 |
|------|------|
| $e^{ix}=\cos x+i\sin x$ | Euler 公式 |
| $z=re^{i\theta}$ | 复数极形式（$r=|z|>0$，$\theta$ 为辐角） |
| $z_k=e^{i2k\pi/n}$ | $n$ 次单位根，位于单位圆上等角分布 |
| $\cos x=\dfrac{e^{ix}+e^{-ix}}{2}$ | 余弦的指数表示 |
| $\sin x=\dfrac{e^{ix}-e^{-ix}}{2i}$ | 正弦的指数表示 |
| $\sinh x=\dfrac{e^x-e^{-x}}{2}$ | 双曲正弦 |
| $\cosh x=\dfrac{e^x+e^{-x}}{2}$ | 双曲余弦 |
| $\cosh^2x-\sinh^2x=1$ | 双曲恒等式 |
| $\tanh x=\dfrac{\sinh x}{\cosh x}=\dfrac{e^x-e^{-x}}{e^x+e^{-x}}$ | 双曲正切 |
| $(\sinh x)'=\cosh x$，$(\cosh x)'=\sinh x$ | 双曲函数导数 |
| $\text{arsinh}\,x=\ln(x+\sqrt{x^2+1})$ | 反双曲正弦 |
| $\text{arcosh}\,x=\ln(x+\sqrt{x^2-1})$ | 反双曲余弦（$x \geq 1$） |
| $\text{artanh}\,x=\frac{1}{2}\ln\frac{1+x}{1-x}$ | 反双曲正切（$|x|<1$） |

---

## 7. 矩阵指数

| 公式 | 含义 |
|------|------|
| $e^A=I+A+\dfrac{A^2}{2!}+\cdots$ | 矩阵指数定义 |
| $\mathbf{x}'=A\mathbf{x}\Rightarrow \mathbf{x}(t)=e^{At}\mathbf{x}(0)$ | 线性系统解形式 |
| $e^{O}=I$ | 零矩阵的指数 |
| $(e^A)^{-1}=e^{-A}$ | 矩阵指数总可逆 |
| $\det(e^A)=e^{\text{tr}(A)}$ | Jacobi 公式 |
| $e^{PAP^{-1}}=Pe^AP^{-1}$ | 相似不变性 |
| $e^{A+B}=e^Ae^B$ **仅当** $AB=BA$ | 注意：一般不成立！ |

---

## 8. 常用模型

| 模型 | 表达式 |
|------|--------|
| 离散增长 | $A_n=A_0(1+r)^n$ |
| 连续增长 | $A(t)=A_0e^{kt}$ |
| 连续复利 | $A=Pe^{rt}$ |
| 半衰期 | $T_{1/2}=\dfrac{\ln2}{|k|}$ |

