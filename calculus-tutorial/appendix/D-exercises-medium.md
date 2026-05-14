# 附录 D：微积分 100 道中档题

> 共 **100 题**，全部为 [中档] 难度（高数中档，约对应考研数学中等难度），覆盖微积分全部 8 个 Part（Ch.4–28）。
> 题型以计算型为主，兼有证明与应用型。部分题目改编自 `kaoyan-problems.md` 中 D/E 级题目，并补充典型 ML 应用题。
> 编号 D.01–D.100 按章节顺序连续。**不附答案**，详解见附录 F。

---

## 极限与连续（D.01–D.12，对应 Ch.4–6）

**D.01** [中档] Ch.5  
求 $\displaystyle\lim_{x\to 0}\frac{\sqrt{1+2x}-\sqrt[3]{1+3x}}{x^2}$（提示：分子各项展到二阶，发现一阶项相消）。

**D.02** [中档] Ch.5  
求 $\displaystyle\lim_{x\to 0}\frac{e^x - e^{\sin x}}{x - \sin x}$（提示：令 $e^{\sin x}$ 提出，利用 $e^u-1\sim u$）。

**D.03** [中档] Ch.5  
求 $\displaystyle\lim_{x\to 0}\left(\frac{1+\tan x}{1+\sin x}\right)^{1/\sin^3 x}$（$1^\infty$ 型，取对数后用 $\tan x-\sin x\sim\tfrac{x^3}{2}$）。

**D.04** [中档] Ch.5  
求 $\displaystyle\lim_{x\to+\infty}\left(\sqrt{x^2+x+1}-\sqrt{x^2-x+1}\right)$（$\infty-\infty$ 型，分子有理化）。

**D.05** [中档] Ch.5  
求 $\displaystyle\lim_{x\to 0}\frac{\ln(1+x)-x}{x^2}$（Taylor 展开到二阶）。

**D.06** [中档] Ch.5  
求 $\displaystyle\lim_{x\to 0^+}x^{\sin x}$（$0^0$ 型；写成 $e^{\sin x\ln x}$，再求指数极限）。

**D.07** [中档] Ch.5  
已知 $\lim_{x\to 0}\dfrac{f(x)}{x^2}=3$，求 $\lim_{x\to 0}\dfrac{f(\sin x)}{x^2}$（等价无穷小替换）。

**D.08** [中档] Ch.5  
求 $\displaystyle\lim_{n\to\infty}\sum_{k=1}^n\frac{1}{n+k}$（识别为 Riemann 和，转化为 $\int_0^1\dfrac{dx}{1+x}$）。

**D.09** [中档] Ch.6  
设 $f(x)=\begin{cases}\dfrac{\ln(1+ax)}{x},& x>0\\ b,& x=0\\ \dfrac{e^x-1}{x},& x<0\end{cases}$ 在 $x=0$ 连续，求 $a,b$。

**D.10** [中档] Ch.6  
讨论 $f(x)=\lim_{n\to\infty}\dfrac{x^{2n}-1}{x^{2n}+1}x$ 的连续性，指出所有间断点并分类。

**D.11** [中档] Ch.6  
设 $f$ 在 $[-1,1]$ 连续，$f(0)=2$。证明存在 $\xi\in[-1,1]$ 使 $f(\xi)=\xi^2+1$（利用零点定理）。

**D.12** [中档] Ch.6  
设 $f(x)=|x-1|\cdot|x+1|$，指出 $f$ 在哪些点不可导，并给出理由。

---

## 微分应用（D.13–D.30，对应 Ch.7–10）

**D.13** [中档] Ch.8  
求 $y=(\sin x)^{\cos x}$ 的导数（对数求导法）。

**D.14** [中档] Ch.8  
由方程 $e^y+xy=e$ 确定的隐函数 $y(x)$ 在 $x=0$ 处，求 $y'(0)$ 与 $y''(0)$。

**D.15** [中档] Ch.8  
设参数方程 $\begin{cases}x=t-\sin t\\ y=1-\cos t\end{cases}$，求 $\dfrac{dy}{dx}$ 与 $\dfrac{d^2y}{dx^2}$（参数方程求导公式）。

**D.16** [中档] Ch.8  
求 $y=\dfrac{1}{x^2-3x+2}$ 的 $n$ 阶导数（先部分分式，再用公式 $\left(\tfrac{1}{x-a}\right)^{(n)}=\tfrac{(-1)^n n!}{(x-a)^{n+1}}$）。

**D.17** [中档] Ch.8  
已知 $f$ 二阶可导，$g(x)=f(\sin x)$，用链式法则写出 $g''(x)$。

**D.18** [中档] Ch.8  
求 $y=\arctan\dfrac{2x}{1-x^2}$ 的导数（提示：利用恒等式 $y=2\arctan x$，$|x|<1$）。

**D.19** [中档] Ch.9  
设 $f$ 在 $[0,1]$ 连续，$(0,1)$ 可导，$f(0)=0,f(1)=1$。证明 $\exists\xi\in(0,1)$ 使 $f'(\xi)=2\xi$（构造 $g=f-x^2$，Rolle 定理）。

**D.20** [中档] Ch.9  
证明当 $x>0$ 时 $\ln(1+x)<x$（构造 $h(x)=x-\ln(1+x)$，验证 $h(0)=0$ 且 $h'>0$）。

**D.21** [中档] Ch.9  
求 $y=\dfrac{x}{1+x^2}$ 的凸凹区间与所有拐点（求 $y''$ 并分析符号）。

**D.22** [中档] Ch.9  
求 $f(x)=e^x\sin x$ 在 $[0,2\pi]$ 上的最大值与最小值（令 $f'=e^x(\sin x+\cos x)=0$）。

**D.23** [中档] Ch.9  
用 L'Hôpital 法则求 $\displaystyle\lim_{x\to1}\frac{x-x^x}{1-x+\ln x}$（令 $t=x-1\to0$，分析两侧阶数）。

**D.24** [中档] Ch.9  
已知 $f(x)=x^3-3ax+1$（$a\in\mathbb{R}$），讨论 $f$ 有三个不同实根时 $a$ 的取值范围。

**D.25** [中档] Ch.9  
用导数方法证明：当 $0<x<\dfrac{\pi}{2}$ 时，$\dfrac{2}{\pi}<\dfrac{\sin x}{x}<1$。

**D.26** [中档] Ch.10  
求 $\displaystyle\lim_{x\to 0}\frac{\sin x - x\cos x}{x^3}$（用 $\sin x$ 与 $\cos x$ 的 Taylor 展开，比较三阶项）。

**D.27** [中档] Ch.10  
求 $\displaystyle\lim_{x\to 0}\frac{(1+x)^{1/x}-e}{x}$（设 $g(x)=\tfrac{\ln(1+x)}{x}=1-\tfrac{x}{2}+o(x)$，展开 $e^{g(x)}$）。

**D.28** [中档] Ch.10  
将 $f(x)=\dfrac{1}{1-x}$ 在 $x=2$ 处展开为幂级数（中心不为原点的 Taylor 展开）。

**D.29** [中档] Ch.10  
求 $f(x)=x\ln x$ 的 $n$ 阶导数（$n\ge 2$；用 Leibniz 公式，$x$ 的 $n\ge2$ 次导为 $0$）。

**D.30** [中档] Ch.10  
设 $f(x)=\arctan x$，利用 $f$ 的幂级数 $\displaystyle\sum_{n=0}^\infty\frac{(-1)^n x^{2n+1}}{2n+1}$ 在 $x=1$ 处的收敛值，写出 $\pi$ 的一个级数表达式（Leibniz 公式）。

---

## 积分技巧（D.31–D.48，对应 Ch.11–14）

**D.31** [中档] Ch.11  
求 $\displaystyle\int\frac{x}{\sqrt{x^2+2x+5}}\,dx$（配方后分拆为凑微分项与标准 $1/\sqrt{u^2+a^2}$ 型）。

**D.32** [中档] Ch.11  
求 $\displaystyle\int e^{2x}\cos x\,dx$（分部积分两次，"循环"后移项）。

**D.33** [中档] Ch.11  
求 $\displaystyle\int\frac{dx}{1+e^x}$（分子加减分母，化为 $1-\tfrac{e^x}{1+e^x}$）。

**D.34** [中档] Ch.11  
求 $\displaystyle\int\frac{\sin x}{1+\sin x}\,dx$（分子加减 $1$，乘共轭化 $\tfrac{1}{1+\sin x}$）。

**D.35** [中档] Ch.11  
求 $\displaystyle\int x\arctan x\,dx$（LIATE 分部，$u=\arctan x$，$dv=x\,dx$）。

**D.36** [中档] Ch.11  
求 $\displaystyle\int\frac{dx}{\sqrt{x}+\sqrt[3]{x}}$（令 $x=t^6$，化为有理函数）。

**D.37** [中档] Ch.11  
求 $\displaystyle\int\frac{\ln(1+x)}{x^2}\,dx$（分部，$u=\ln(1+x)$，$v=-1/x$，再部分分式）。

**D.38** [中档] Ch.11  
求 $\displaystyle\int\frac{dx}{x\sqrt{1-\ln^2 x}}$（令 $u=\ln x$，化为 $\int\tfrac{du}{\sqrt{1-u^2}}$）。

**D.39** [中档] Ch.12  
计算 $\displaystyle\int_0^{\pi/2}\sin^4 x\,dx$（Wallis 公式，$n=4$ 偶）。

**D.40** [中档] Ch.12  
计算 $\displaystyle\int_{-1}^1\frac{x^2}{1+e^x}\,dx$（对称区间 + $e^x$ 技巧：$f(x)+f(-x)=x^2$，得 $I=\tfrac13$）。

**D.41** [中档] Ch.12  
计算 $\displaystyle\int_0^\pi x\sin x\,dx$（分部，$u=x$，$dv=\sin x\,dx$）。

**D.42** [中档] Ch.12  
计算 $\displaystyle\int_0^1\frac{\ln(1+x)}{1+x^2}\,dx$（令 $x=\tan t$，再用对称 $t\to\tfrac\pi4-t$ 的技巧）。

**D.43** [中档] Ch.12  
计算 $\displaystyle\int_0^{\pi/2}\ln\sin x\,dx$（利用 $I=\int_0^{\pi/2}\ln\cos x$ 的对称性，以及 $2I=I-\tfrac\pi2\ln 2$）。

**D.44** [中档] Ch.13  
讨论 $\displaystyle\int_0^{+\infty}\frac{dx}{x^p(1+x)}$ 的收敛性（分别在 $x\to0^+$ 与 $x\to+\infty$ 两端判断，条件为 $0<p<1$）。

**D.45** [中档] Ch.13  
计算 Gauss 积分 $\displaystyle\int_0^{+\infty}e^{-x^2}\,dx$（利用二维极坐标技巧，结果为 $\tfrac{\sqrt\pi}{2}$）。

**D.46** [中档] Ch.14  
求曲线 $y=x^2$ 与 $y=\sqrt x$ 围成区域的面积，以及该区域绕 $x$ 轴旋转所得旋转体的体积。

**D.47** [中档] Ch.14  
求 $y=\ln x$ 在 $[1,e]$ 段绕 $x$ 轴旋转的旋转体体积（用 $\pi\int\ln^2 x\,dx$，分部两次）。

**D.48** [中档] Ch.14  
用变限积分 + L'Hôpital 求 $\displaystyle\lim_{x\to 0}\frac{\int_0^{x^2}f(t)\,dt}{x^4}$，其中 $f$ 连续且 $f(0)=0,f'(0)=2$（结果为 $1$）。

---

## 级数（D.49–D.60，对应 Ch.15–17）

**D.49** [中档] Ch.15  
判断 $\displaystyle\sum_{n=1}^\infty\frac{n!}{n^n}$ 的收敛性（比值法：$a_{n+1}/a_n\to 1/e<1$，收敛）。

**D.50** [中档] Ch.15  
判断 $\displaystyle\sum_{n=1}^\infty\frac{1}{n\ln n}$ 的收敛性（积分判别法：$\int_2^\infty\tfrac{dx}{x\ln x}=\infty$，发散）。

**D.51** [中档] Ch.15  
求 $\displaystyle\sum_{n=1}^\infty\frac{n^2}{2^n}$ 的和（比值法验收敛，再用幂级数逐项求导技巧）。

**D.52** [中档] Ch.15  
讨论 $\displaystyle\sum_{n=1}^\infty(-1)^{n-1}\frac{x^n}{n}$ 在 $x=1$ 与 $x=-1$ 处的收敛性（端点验证），并写出收敛域。

**D.53** [中档] Ch.15  
证明：若 $\displaystyle\sum a_n$ 绝对收敛，则它也条件收敛（即收敛）。并举例说明反命题不成立（用 $\displaystyle\sum\tfrac{(-1)^{n-1}}{n}$）。

**D.54** [中档] Ch.16  
求幂级数 $\displaystyle\sum_{n=0}^\infty\frac{x^n}{n!}$ 的收敛半径与和函数（用比值法，和为 $e^x$）。

**D.55** [中档] Ch.16  
求幂级数 $\displaystyle\sum_{n=1}^\infty\frac{x^n}{n}$ 的收敛域与和函数 $S(x)$（端点单独验；和为 $-\ln(1-x)$，$x\in[-1,1)$）。

**D.56** [中档] Ch.16  
求幂级数 $\displaystyle\sum_{n=0}^\infty(n+1)x^n$ 的收敛半径，并用对 $\tfrac{1}{1-x}$ 逐项求导方法写出其和函数（$|x|<1$）。

**D.57** [中档] Ch.16  
将 $f(x)=\ln(1+x)$ 在 $x=0$ 展为幂级数，并利用 $x=1$ 处的收敛值推出 $\displaystyle\sum_{n=1}^\infty\frac{(-1)^{n-1}}{n}=\ln 2$。

**D.58** [中档] Ch.16  
求 $f(x)=\arctan x$ 的 Maclaurin 级数，收敛域，并推出 $\pi=4\displaystyle\sum_{n=0}^\infty\frac{(-1)^n}{2n+1}$（Leibniz 公式）。

**D.59** [中档] Ch.17  
求 $f(x)=|x|$（$-\pi\le x\le\pi$）的 Fourier 级数，并由此推出 $\displaystyle\sum_{n=0}^\infty\frac{1}{(2n+1)^2}=\frac{\pi^2}{8}$。

**D.60** [中档] Ch.17  
求 $f(x)=x^2$（$-\pi\le x\le\pi$）的 Fourier 级数，并由此推出 $\displaystyle\sum_{n=1}^\infty\frac{1}{n^2}=\frac{\pi^2}{6}$（Parseval 等式或直接代值）。

---

## 多元微积分（D.61–D.80，对应 Ch.18–22）

**D.61** [中档] Ch.18  
设 $z=x^y$，求 $z_x,z_y$；再验证 $xz_x\ln x=yz_y$。

**D.62** [中档] Ch.18  
设 $u=f(x,y,z)$，$z=g(x,y)$，用链式法则写出 $\dfrac{\partial u}{\partial x}$（含直接路径和经由 $z$ 的路径）。

**D.63** [中档] Ch.18  
证明 $z=\sin(x+y)+\cos(x-y)$ 满足波动方程 $z_{xx}-z_{yy}=0$（直接计算二阶偏导）。

**D.64** [中档] Ch.18  
设 $F(x,y,z)=x^2+y^2+z^2-3xyz=0$ 在 $(1,1,1)$ 附近确定 $z=z(x,y)$。求 $z_x,z_y$（隐函数定理：$z_x=-F_x/F_z$）。

**D.65** [中档] Ch.18  
求 $f(x,y)=x^2+y^2-xy+x-y$ 的极值（解 $\nabla f=0$，Hessian 判别）。

**D.66** [中档] Ch.18  
求 $f(x,y)=e^{x+y}$ 在 $(0,0)$ 的二阶 Taylor 展开（展开到 $(x+y)^2$ 项）。

**D.67** [中档] Ch.19  
计算 $\displaystyle\iint_D xy\,dA$，$D=\{0\le x\le 1,\,0\le y\le x\}$（先 $y$ 后 $x$ 积分，结果为 $\tfrac{1}{8}$）。

**D.68** [中档] Ch.19  
用极坐标计算 $\displaystyle\iint_{x^2+y^2\le 1}e^{-(x^2+y^2)}\,dA$（结果为 $\pi(1-e^{-1})$）。

**D.69** [中档] Ch.19  
计算 $\displaystyle\iint_{x^2+y^2\le 4}\sqrt{x^2+y^2}\,dA$（极坐标，结果为 $\tfrac{16\pi}{3}$）。

**D.70** [中档] Ch.19  
求抛物面 $z=x^2+y^2$ 与平面 $z=4$ 围成立体的体积（极坐标，结果为 $8\pi$）。

**D.71** [中档] Ch.19  
交换积分次序：$\displaystyle\int_0^1\!\!\int_x^1 f(x,y)\,dy\,dx$（画出积分区域，交换为先 $x$ 后 $y$）。

**D.72** [中档] Ch.20  
计算曲线积分 $\displaystyle\oint_L(-y\,dx+x\,dy)$，$L:x^2+y^2=1$ 逆时针（Green 公式，结果为 $2\pi$）。

**D.73** [中档] Ch.20  
用 Green 公式计算 $\displaystyle\oint_L(x^2y\,dx+xy^2\,dy)$，$L$ 为 $[0,1]^2$ 的正向边界（$Q_x-P_y=y^2-x^2$，结果为 $0$）。

**D.74** [中档] Ch.21  
求 $f(x,y)=x^2+2y^2$ 在约束 $x+y=1$ 下的最小值（Lagrange 乘子法或代入消元，结果为 $x=\tfrac{2}{3},y=\tfrac{1}{3},f_{\min}=\tfrac{2}{3}$）。

**D.75** [中档] Ch.21  
求 $f(x,y)=x^2-y^2$ 在单位圆 $x^2+y^2=1$ 上的最大值与最小值（参数化 $x=\cos\theta,y=\sin\theta$，$f=\cos2\theta$）。

**D.76** [中档] Ch.22  
计算曲面积分 $\displaystyle\iint_S z\,dS$，$S$ 为单位上半球面（利用 $z\,dS=dA$，球面外法向简化，结果为 $\pi$）。

**D.77** [中档] Ch.22  
用 Gauss 散度定理计算 $\displaystyle\oiint_S(x\,dy\,dz+y\,dz\,dx+z\,dx\,dy)$，$S$ 为单位球面外侧（散度为 $3$，体积 $\tfrac{4\pi}{3}$，结果为 $4\pi$）。

**D.78** [中档] Ch.21  
证明向量场 $\mathbf{F}=(yz,xz,xy)$ 是保守场（$\nabla\times\mathbf{F}=\mathbf{0}$），并求其势函数 $\varphi$（$\varphi=xyz$）。

**D.79** [中档] Ch.22  
计算螺旋线 $\mathbf{r}(t)=(\cos t,\sin t,t)$，$0\le t\le 2\pi$ 的弧长（$|\mathbf{r}'|=\sqrt{2}$，结果为 $2\sqrt{2}\pi$）。

**D.80** [中档] Ch.22  
用方向导数公式求 $f=x^2+y^2+z^2$ 在点 $(1,1,1)$ 沿方向 $\mathbf{l}=(1,2,2)/3$ 的方向导数（$\nabla f\cdot\mathbf{l}$，结果为 $\tfrac{10}{3}$）。

---

## 常微分方程（D.81–D.90，对应 Ch.23–24）

**D.81** [中档] Ch.23  
求可分离方程 $y'=\dfrac{y}{x}+1$（$x>0$）的通解（改写为线性方程 $y'-\tfrac{y}{x}=1$，积分因子 $\mu=1/x$，通解 $y=x\ln x+Cx$）。

**D.82** [中档] Ch.23  
求 $y'=\dfrac{x+y}{x-y}$ 的通解（齐次方程，令 $u=y/x$，分离变量，得隐式解 $\arctan(y/x)-\tfrac12\ln(1+(y/x)^2)=\ln|x|+C$）。

**D.83** [中档] Ch.23  
求一阶线性 $y'+2y=e^{-x}$ 的通解（积分因子 $\mu=e^{2x}$，$(ye^{2x})'=e^x$，通解 $y=e^{-x}+Ce^{-2x}$）。

**D.84** [中档] Ch.23  
设 $f$ 连续且满足 $f(x)=e^x+\displaystyle\int_0^x f(t)\,dt$，求 $f(x)$（两边对 $x$ 求导得 $f'=f+e^x$，一阶线性 ODE，结合初值 $f(0)=1$）。

**D.85** [中档] Ch.24  
求 $y''-3y'+2y=0$ 的通解（特征根 $r=1,2$，通解 $y=C_1e^x+C_2e^{2x}$）。

**D.86** [中档] Ch.24  
求 $y''+4y=0$ 的通解（特征根 $r=\pm2i$，通解 $y=C_1\cos2x+C_2\sin2x$）。

**D.87** [中档] Ch.24  
求 $y''-3y'+2y=e^x$ 的通解（$r=1$ 是特征根，共振，设 $y_p=Axe^x$，代入得 $A=-1$，通解 $y=C_1e^x+C_2e^{2x}-xe^x$）。

**D.88** [中档] Ch.24  
求 $y''-4y'+4y=e^{2x}$ 的通解（$r=2$ 是二重特征根，二次共振，设 $y_p=Ax^2e^{2x}$，代入得 $A=\tfrac12$，通解 $y=(C_1+C_2x)e^{2x}+\tfrac{x^2}{2}e^{2x}$）。

**D.89** [中档] Ch.24  
求 $y''+y=\sec x$（$|x|<\tfrac\pi2$）的通解（常数变易法：$y_p=\cos x\ln\cos x+x\sin x$，通解 $y=C_1\cos x+C_2\sin x+\cos x\ln\cos x+x\sin x$）。

**D.90** [中档] Ch.24  
已知 $y''+p(x)y'+q(x)y=0$ 有一个特解 $y_1=e^x$，用降阶法（令 $y=y_1 v$）写出求第二线性无关特解的方程（不要求求解，只要写出关于 $v'$ 的一阶方程形式）。

---

## AI 微积分（D.91–D.100，对应 Ch.25–28）

**D.91** [中档] Ch.25  
设 $f:\mathbb{R}^n\to\mathbb{R}$ 是凸函数。证明：若 $\nabla f(\mathbf{x}^*)=\mathbf{0}$，则 $\mathbf{x}^*$ 是全局最小值点（利用凸函数一阶条件 $f(\mathbf{y})\ge f(\mathbf{x})+\nabla f(\mathbf{x})^\top(\mathbf{y}-\mathbf{x})$）。

**D.92** [中档] Ch.25  
判断 $f(x,y)=x^2+xy+y^2$ 的凸性：计算 Hessian $H=\begin{pmatrix}2&1\\1&2\end{pmatrix}$ 的特征值，并说明 $f$ 是强凸函数（$H\succ0$）。

**D.93** [中档] Ch.26  
设 $f(\mathbf{x})=\|\mathbf{A}\mathbf{x}-\mathbf{b}\|^2$（$A\in\mathbb{R}^{m\times n}$，$\mathbf{b}\in\mathbb{R}^m$）。计算 $\nabla_\mathbf{x}f$，并写出最优性条件（正规方程 $A^\top A\mathbf{x}=A^\top\mathbf{b}$）。

**D.94** [中档] Ch.26  
设 $f(\mathbf{x})=\sigma(\mathbf{w}^\top\mathbf{x})$，其中 $\sigma(t)=\tfrac{1}{1+e^{-t}}$ 为 sigmoid 函数。计算 $\dfrac{\partial f}{\partial\mathbf{w}}$（链式法则 + $\sigma'(t)=\sigma(t)(1-\sigma(t))$）。

**D.95** [中档] Ch.26  
设损失 $L=-\mathbf{y}^\top\log(\mathbf{p})$（交叉熵），其中 $\mathbf{p}=\mathrm{softmax}(\mathbf{z})$，$p_i=e^{z_i}/\sum e^{z_j}$。计算 $\dfrac{\partial L}{\partial z_i}$（结果为 $p_i-y_i$）。

**D.96** [中档] Ch.27  
设 $X\sim\mathcal{N}(0,1)$，计算 $E[X^4]$（利用正态分布矩公式 $E[X^{2k}]=(2k-1)!!$，结果为 $3$）。

**D.97** [中档] Ch.27  
设 $X,Y$ 独立，$X\sim\mathcal{N}(\mu_1,\sigma_1^2)$，$Y\sim\mathcal{N}(\mu_2,\sigma_2^2)$。证明 $Z=X+Y\sim\mathcal{N}(\mu_1+\mu_2,\sigma_1^2+\sigma_2^2)$（利用矩母函数或特征函数）。

**D.98** [中档] Ch.27  
计算 KL 散度 $D_{\mathrm{KL}}(p\|q)$，其中 $p=\mathcal{N}(\mu_1,\sigma^2)$，$q=\mathcal{N}(\mu_2,\sigma^2)$（同方差，结果为 $\dfrac{(\mu_1-\mu_2)^2}{2\sigma^2}$）。

**D.99** [中档] Ch.28  
设 $W_t$ 是标准布朗运动，计算 $E[W_t^2]$，$E[W_s W_t]$（$s<t$），以及 $\mathrm{Var}(W_t-W_s)$（结果分别为 $t$，$s$，$t-s$）。

**D.100** [中档] Ch.28  
设 Itô 过程 $dX_t=\mu\,dt+\sigma\,dW_t$（常数 $\mu,\sigma$）。用 Itô 公式计算 $d(X_t^2)$，并写出 $E[X_t^2]$（初始 $X_0=0$，结果为 $E[X_t^2]=\mu^2t^2+\sigma^2t$，提示：$E[X_t]=\mu t$）。

---

> **题号 / 分组分布索引**
>
> | 分组 | 章节 | 题号范围 | 题数 |
> |------|------|----------|------|
> | 极限连续 | Ch.4–6 | D.01–D.12 | 12 |
> | 微分应用 | Ch.7–10 | D.13–D.30 | 18 |
> | 积分技巧 | Ch.11–14 | D.31–D.48 | 18 |
> | 级数 | Ch.15–17 | D.49–D.60 | 12 |
> | 多元微积分 | Ch.18–22 | D.61–D.80 | 20 |
> | ODE | Ch.23–24 | D.81–D.90 | 10 |
> | AI 微积分 | Ch.25–28 | D.91–D.100 | 10 |
> | **合计** | | **D.01–D.100** | **100** |
