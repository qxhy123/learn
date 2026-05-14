# 附录 F3a：多元微积分详解（C.51-C.65, D.61-D.80, E.37-E.48）

> 覆盖 Ch.18–22，共 **47 题**：C.51–C.65（基础 15 题）、D.61–D.80（中档 20 题）、E.37–E.48（提升 12 题）。
> 格式：**题目回顾 / 思路 / 解答 / 答案**，紧凑风格，总字数控制在 9500 字以内。

---

## 第一部分：基础题详解（C.51–C.65）

---

**C.51** 求 $f(x,y)=x^3+2x^2y-y^3$ 的偏导数 $f_x$ 与 $f_y$。

**思路** 对 $x$ 求偏导时视 $y$ 为常数，对 $y$ 求偏导时视 $x$ 为常数，逐项用幂次法则。

**解答**

$$f_x = \frac{\partial}{\partial x}(x^3+2x^2y-y^3) = 3x^2+4xy$$

$$f_y = \frac{\partial}{\partial y}(x^3+2x^2y-y^3) = 2x^2-3y^2$$

**答案** $f_x=3x^2+4xy$，$f_y=2x^2-3y^2$。

---

**C.52** 求 $z=e^{xy}$ 的偏导数 $\dfrac{\partial z}{\partial x}$ 与 $\dfrac{\partial z}{\partial y}$。

**思路** 链式法则：$\dfrac{\partial}{\partial x}e^{xy}=e^{xy}\cdot y$，类似地对 $y$。

**解答**

$$\frac{\partial z}{\partial x}=e^{xy}\cdot y=ye^{xy},\quad \frac{\partial z}{\partial y}=e^{xy}\cdot x=xe^{xy}$$

**答案** $z_x=ye^{xy}$，$z_y=xe^{xy}$。

---

**C.53** 求 $f(x,y)=\sin(x+y^2)$ 的混合偏导数 $f_{xy}$。

**思路** 先对 $x$ 求偏导得 $f_x$，再对 $y$ 求偏导；或先 $y$ 后 $x$（二阶混合偏导连续则相等）。

**解答**

$$f_x=\cos(x+y^2)\cdot 1=\cos(x+y^2)$$

$$f_{xy}=\frac{\partial}{\partial y}\cos(x+y^2)=-\sin(x+y^2)\cdot 2y=-2y\sin(x+y^2)$$

**答案** $f_{xy}=-2y\sin(x+y^2)$。

---

**C.54** 设 $z=f(x^2-y^2)$（$f$ 可微），求 $z_x,z_y$，并验证 $yz_x+xz_y=0$。

**思路** 令 $u=x^2-y^2$，链式法则：$z_x=f'(u)\cdot 2x$，$z_y=f'(u)\cdot(-2y)$。

**解答**

$$z_x=2xf'(x^2-y^2),\quad z_y=-2yf'(x^2-y^2)$$

验证：$yz_x+xz_y=y\cdot 2xf'+(x)\cdot(-2y)f'=2xyf'-2xyf'=0$。$\checkmark$

**答案** $z_x=2xf'(x^2-y^2)$，$z_y=-2yf'(x^2-y^2)$，恒等式成立。

---

**C.55** 计算二重积分 $\displaystyle\iint_D 1\,dA$，其中 $D=[0,2]\times[0,3]$。

**思路** 被积函数为 $1$，积分即面积 $= 2\times 3$。

**解答**

$$\iint_D 1\,dA = \int_0^2\!\!\int_0^3 1\,dy\,dx=\int_0^2 3\,dx=6$$

**答案** $6$。

---

**C.56** 计算 $\displaystyle\int_0^1\!\!\int_0^1(x+y)\,dy\,dx$。

**思路** 先对 $y$ 积分（$x$ 视常数），再对 $x$ 积分。

**解答**

$$\int_0^1\!\!\int_0^1(x+y)\,dy\,dx=\int_0^1\left[xy+\frac{y^2}{2}\right]_0^1dx=\int_0^1\!\left(x+\frac{1}{2}\right)dx=\left[\frac{x^2}{2}+\frac{x}{2}\right]_0^1=1$$

**答案** $1$。

---

**C.57** 用极坐标计算 $\displaystyle\iint_{x^2+y^2\le4}1\,dA$。

**思路** 极坐标 $x=r\cos\theta,y=r\sin\theta$，$dA=r\,dr\,d\theta$，区域为 $0\le r\le 2,\,0\le\theta\le 2\pi$。

**解答**

$$\int_0^{2\pi}\!\!\int_0^2 r\,dr\,d\theta=2\pi\cdot\left[\frac{r^2}{2}\right]_0^2=2\pi\cdot 2=4\pi$$

**答案** $4\pi$（半径为 $2$ 的圆面积）。

---

**C.58** 求向量场 $\mathbf{F}=(y,x)$ 的散度 $\nabla\cdot\mathbf{F}$ 与旋度 $\nabla\times\mathbf{F}$（二维）。

**思路** 散度 $=\partial P/\partial x+\partial Q/\partial y$；二维旋度（$z$-分量）$=\partial Q/\partial x-\partial P/\partial y$。

**解答**

$$\nabla\cdot\mathbf{F}=\frac{\partial y}{\partial x}+\frac{\partial x}{\partial y}=0+0=0$$

$$(\nabla\times\mathbf{F})_z=\frac{\partial x}{\partial x}-\frac{\partial y}{\partial y}=1-1=0$$

**答案** 散度 $=0$，旋度 $=0$。

---

**C.59** 计算曲线积分 $\displaystyle\int_C x\,ds$，$C$ 为线段从 $(0,0)$ 到 $(1,0)$。

**思路** 参数化 $x=t,y=0$（$0\le t\le 1$），$ds=\sqrt{1+0}\,dt=dt$。

**解答**

$$\int_C x\,ds=\int_0^1 t\,dt=\frac{1}{2}$$

**答案** $\dfrac{1}{2}$。

---

**C.60** 用 Green 公式计算 $\displaystyle\oint_L x\,dy-y\,dx$，$L$ 为单位圆逆时针。

**思路** Green 公式：$\oint_L P\,dx+Q\,dy=\iint_D(Q_x-P_y)\,dA$。这里 $P=-y,Q=x$，故 $Q_x-P_y=1+1=2$。

**解答**

$$\oint_L x\,dy-y\,dx=\iint_D 2\,dA=2\times\pi(1)^2=2\pi$$

**答案** $2\pi$（等于 $2\times$单位圆面积）。

---

**C.61** 求 $f(x,y)=x^2+2y^2$ 在点 $(1,1)$ 处的梯度 $\nabla f(1,1)$。

**思路** $\nabla f=(f_x,f_y)=(2x,4y)$，代入 $(1,1)$。

**解答**

$$\nabla f=(2x,4y)\big|_{(1,1)}=(2,4)$$

**答案** $\nabla f(1,1)=(2,4)$。

---

**C.62** 求 $f(x,y)=x^2y-y^3$ 的驻点（令 $\nabla f=\mathbf{0}$）。

**思路** 联立 $f_x=2xy=0$ 与 $f_y=x^2-3y^2=0$，分情况讨论。

**解答**

$f_x=2xy=0\Rightarrow x=0$ 或 $y=0$。

- 若 $x=0$：代入 $f_y=0-3y^2=0\Rightarrow y=0$，得 $(0,0)$。
- 若 $y=0$：代入 $f_y=x^2=0\Rightarrow x=0$，仍得 $(0,0)$。

**答案** 唯一驻点 $(0,0)$。

---

**C.63** 用 Lagrange 乘子法求 $f(x,y)=x+y$ 在约束 $x^2+y^2=2$ 下的最大值。

**思路** 令 $\nabla f=\lambda\nabla g$（$g=x^2+y^2-2$），即 $(1,1)=\lambda(2x,2y)$，故 $x=y=\tfrac{1}{2\lambda}$，代入约束。

**解答**

$x=y$，代入 $2x^2=2\Rightarrow x=\pm1$。

- $(1,1)$：$f=2$（最大）。
- $(-1,-1)$：$f=-2$（最小）。

**答案** 最大值 $f=2$，在 $(1,1)$ 处取得。

---

**C.64** 用 Gauss 散度定理计算 $\displaystyle\oiint_S\mathbf{F}\cdot d\mathbf{S}$，$\mathbf{F}=(x,y,z)$，$S$ 为单位球面外侧。

**思路** 散度 $\nabla\cdot\mathbf{F}=1+1+1=3$；Gauss 定理化为体积分，单位球体积 $\tfrac{4\pi}{3}$。

**解答**

$$\oiint_S\mathbf{F}\cdot d\mathbf{S}=\iiint_\Omega 3\,dV=3\cdot\frac{4\pi}{3}=4\pi$$

**答案** $4\pi$。

---

**C.65** 给出 Stokes 定理表述，并验证 $\mathbf{F}=(y,-x,0)$，$C$ 为单位圆的例子。

**思路** Stokes 定理：$\oint_C\mathbf{F}\cdot d\mathbf{r}=\iint_S(\nabla\times\mathbf{F})\cdot d\mathbf{S}$；计算旋度并取平面 $z=0$ 为曲面。

**解答**

旋度：$\nabla\times\mathbf{F}=\det\begin{vmatrix}\mathbf{i}&\mathbf{j}&\mathbf{k}\\\partial_x&\partial_y&\partial_z\\y&-x&0\end{vmatrix}=(0-0)\mathbf{i}-(0-0)\mathbf{j}+(-1-1)\mathbf{k}=(0,0,-2)$。

面积分（单位圆盘，法向朝上 $(0,0,1)$）：$\iint_S(0,0,-2)\cdot(0,0,1)\,dA=-2\pi$。

直接计算：$\oint_C(y,-x,0)\cdot(dx,dy,dz)$，参数 $x=\cos t,y=\sin t$：

$$\int_0^{2\pi}(\sin t(-\sin t)+(-\cos t)\cos t)\,dt=\int_0^{2\pi}-1\,dt=-2\pi\quad\checkmark$$

**答案** Stokes 定理成立，两种方法均得 $-2\pi$。

---

## 第二部分：中档题详解（D.61–D.80）

---

**D.61** 设 $z=x^y$，求 $z_x,z_y$；验证 $xz_x\ln x=yz_y$。

**思路** $x^y=e^{y\ln x}$。对 $x$ 求偏导用链式法则，对 $y$ 用指数对数性质。

**解答**

$$z_x=yx^{y-1},\quad z_y=x^y\ln x$$

验证：$xz_x\ln x=x\cdot yx^{y-1}\ln x=yx^y\ln x$；$yz_y=y\cdot x^y\ln x$。两者相等。$\checkmark$

**答案** $z_x=yx^{y-1}$，$z_y=x^y\ln x$，恒等式成立。

---

**D.62** 设 $u=f(x,y,z)$，$z=g(x,y)$，用链式法则写出 $\dfrac{\partial u}{\partial x}$。

**思路** $x$ 影响 $u$ 有两条路径：直接路径 $u\to x$，以及经由 $z=g(x,y)$ 的间接路径。

**解答**

$$\frac{\partial u}{\partial x}=\frac{\partial f}{\partial x}+\frac{\partial f}{\partial z}\cdot\frac{\partial g}{\partial x}=f_x(x,y,g(x,y))+f_z(x,y,g(x,y))\cdot g_x(x,y)$$

**答案** $\dfrac{\partial u}{\partial x}=f_x+f_z\cdot g_x$（含直接路径与经由 $z$ 的间接路径之和）。

---

**D.63** 证明 $z=\sin(x+y)+\cos(x-y)$ 满足波动方程 $z_{xx}-z_{yy}=0$。

**思路** 直接计算二阶偏导，比较。

**解答**

$$z_x=\cos(x+y)-\sin(x-y),\quad z_{xx}=-\sin(x+y)-\cos(x-y)$$

$$z_y=\cos(x+y)+\sin(x-y),\quad z_{yy}=-\sin(x+y)-\cos(x-y)$$

故 $z_{xx}-z_{yy}=[-\sin(x+y)-\cos(x-y)]-[-\sin(x+y)-\cos(x-y)]=0$。$\checkmark$

**答案** $z_{xx}=z_{yy}$，波动方程成立。

---

**D.64** 设 $F=x^2+y^2+z^2-3xyz=0$ 在 $(1,1,1)$ 附近确定 $z=z(x,y)$，求 $z_x,z_y$。

**思路** 隐函数定理：$z_x=-F_x/F_z$，$z_y=-F_y/F_z$。

**解答**

$$F_x=2x-3yz,\quad F_y=2y-3xz,\quad F_z=2z-3xy$$

在 $(1,1,1)$：$F_x=2-3=-1$，$F_y=-1$，$F_z=-1$。

$$z_x\big|_{(1,1,1)}=-\frac{-1}{-1}=-1,\quad z_y\big|_{(1,1,1)}=-\frac{-1}{-1}=-1$$

**答案** $z_x=-\dfrac{2x-3yz}{2z-3xy}$，$z_y=-\dfrac{2y-3xz}{2z-3xy}$；在 $(1,1,1)$ 处均为 $-1$。

---

**D.65** 求 $f(x,y)=x^2+y^2-xy+x-y$ 的极值。

**思路** 令 $\nabla f=0$，解驻点，用 Hessian 判别。

**解答**

$$f_x=2x-y+1=0,\quad f_y=2y-x-1=0$$

由第一式 $y=2x+1$，代入第二式：$2(2x+1)-x-1=3x+1=0\Rightarrow x=-\tfrac{1}{3},y=\tfrac{1}{3}$。

Hessian：$H=\begin{pmatrix}2&-1\\-1&2\end{pmatrix}$，$\det H=3>0$，$f_{xx}=2>0$，故为极小值。

$$f\!\left(-\tfrac{1}{3},\tfrac{1}{3}\right)=\tfrac{1}{9}+\tfrac{1}{9}+\tfrac{1}{9}-\tfrac{1}{3}-\tfrac{1}{3}=-\tfrac{1}{3}$$

**答案** 极小值 $f=-\tfrac{1}{3}$，在 $\left(-\tfrac{1}{3},\tfrac{1}{3}\right)$ 处取得，无极大值。

---

**D.66** 求 $f(x,y)=e^{x+y}$ 在 $(0,0)$ 的二阶 Taylor 展开。

**思路** $e^{x+y}=e^{(x+y)}$，令 $t=x+y$，展开 $e^t=1+t+t^2/2+\cdots$。

**解答**

$$e^{x+y}=1+(x+y)+\frac{(x+y)^2}{2}+O(\|(x,y)\|^3)$$

$$=1+x+y+\frac{x^2+2xy+y^2}{2}+O(3)$$

**答案** $e^{x+y}\approx 1+x+y+\dfrac{x^2}{2}+xy+\dfrac{y^2}{2}$（二阶展开）。

---

**D.67** 计算 $\displaystyle\iint_D xy\,dA$，$D=\{0\le x\le1,0\le y\le x\}$。

**思路** 先对 $y$ 从 $0$ 到 $x$ 积分，再对 $x$ 从 $0$ 到 $1$ 积分。

**解答**

$$\int_0^1\!\!\int_0^x xy\,dy\,dx=\int_0^1 x\cdot\frac{x^2}{2}\,dx=\frac{1}{2}\int_0^1 x^3\,dx=\frac{1}{2}\cdot\frac{1}{4}=\frac{1}{8}$$

**答案** $\dfrac{1}{8}$。

---

**D.68** 用极坐标计算 $\displaystyle\iint_{x^2+y^2\le1}e^{-(x^2+y^2)}\,dA$。

**思路** 极坐标：$r^2=x^2+y^2$，积分区域 $0\le r\le1$，$0\le\theta\le2\pi$。

**解答**

$$\int_0^{2\pi}\!\!\int_0^1 e^{-r^2}r\,dr\,d\theta=2\pi\int_0^1 re^{-r^2}\,dr=2\pi\cdot\left[-\frac{e^{-r^2}}{2}\right]_0^1=2\pi\cdot\frac{1-e^{-1}}{2}=\pi(1-e^{-1})$$

**答案** $\pi(1-e^{-1})$。

---

**D.69** 计算 $\displaystyle\iint_{x^2+y^2\le4}\sqrt{x^2+y^2}\,dA$。

**思路** 极坐标：$\sqrt{x^2+y^2}=r$，区域 $0\le r\le2$。

**解答**

$$\int_0^{2\pi}\!\!\int_0^2 r\cdot r\,dr\,d\theta=2\pi\int_0^2 r^2\,dr=2\pi\cdot\frac{8}{3}=\frac{16\pi}{3}$$

**答案** $\dfrac{16\pi}{3}$。

---

**D.70** 求抛物面 $z=x^2+y^2$ 与平面 $z=4$ 围成立体的体积。

**思路** 立体为 $0\le z\le4$，底为圆盘 $x^2+y^2\le4$。极坐标计算：$V=\iint_{D}(4-r^2)\,dA$（$D:r\le2$）。

**解答**

$$V=\int_0^{2\pi}\!\!\int_0^2(4-r^2)r\,dr\,d\theta=2\pi\int_0^2(4r-r^3)\,dr=2\pi\left[2r^2-\frac{r^4}{4}\right]_0^2=2\pi(8-4)=8\pi$$

**答案** $8\pi$。

---

**D.71** 交换积分次序：$\displaystyle\int_0^1\!\!\int_x^1 f(x,y)\,dy\,dx$。

**思路** 积分区域 $D=\{0\le x\le1,x\le y\le1\}$，即 $\{0\le y\le1,0\le x\le y\}$。

**解答**

原积分区域 $D$：$0\le x\le1$，$x\le y\le1$，等价描述：$0\le y\le1$，$0\le x\le y$。

$$\int_0^1\!\!\int_x^1 f(x,y)\,dy\,dx=\int_0^1\!\!\int_0^y f(x,y)\,dx\,dy$$

**答案** $\displaystyle\int_0^1\!\!\int_0^y f(x,y)\,dx\,dy$。

---

**D.72** 计算 $\displaystyle\oint_L(-y\,dx+x\,dy)$，$L:x^2+y^2=1$ 逆时针。

**思路** Green 公式，$P=-y,Q=x$，$Q_x-P_y=1+1=2$，$D$ 为单位圆盘。

**解答**

$$\oint_L(-y\,dx+x\,dy)=\iint_D 2\,dA=2\pi$$

**答案** $2\pi$。

---

**D.73** 用 Green 公式计算 $\displaystyle\oint_L(x^2y\,dx+xy^2\,dy)$，$L$ 为 $[0,1]^2$ 的正向边界。

**思路** $P=x^2y,Q=xy^2$，$Q_x-P_y=y^2-x^2$，在单位正方形上积分。

**解答**

$$\iint_{[0,1]^2}(y^2-x^2)\,dA=\int_0^1\!\!\int_0^1(y^2-x^2)\,dx\,dy=\int_0^1\!\left(y^2-\frac{1}{3}\right)dy=\frac{1}{3}-\frac{1}{3}=0$$

**答案** $0$（由对称性，$\iint y^2\,dA=\iint x^2\,dA$，二者相减为零）。

---

**D.74** 求 $f(x,y)=x^2+2y^2$ 在约束 $x+y=1$ 下的最小值。

**思路** 代入消元：$x=1-y$，$f(y)=(1-y)^2+2y^2=3y^2-2y+1$，对 $y$ 求导。

**解答**

$f'(y)=6y-2=0\Rightarrow y=\tfrac{1}{3},x=\tfrac{2}{3}$。

$$f_{\min}=\left(\frac{2}{3}\right)^2+2\left(\frac{1}{3}\right)^2=\frac{4}{9}+\frac{2}{9}=\frac{2}{3}$$

**答案** 最小值 $\dfrac{2}{3}$，在 $\left(\dfrac{2}{3},\dfrac{1}{3}\right)$ 处取得。

---

**D.75** 求 $f(x,y)=x^2-y^2$ 在单位圆 $x^2+y^2=1$ 上的最大值与最小值。

**思路** 参数化 $x=\cos\theta,y=\sin\theta$，$f=\cos^2\theta-\sin^2\theta=\cos2\theta$。

**解答**

$f=\cos2\theta$，$\theta\in[0,2\pi)$，最大值 $1$（$\theta=0,\pi$，即 $(\pm1,0)$），最小值 $-1$（$\theta=\pi/2,3\pi/2$，即 $(0,\pm1)$）。

**答案** 最大值 $1$，最小值 $-1$。

---

**D.76** 计算曲面积分 $\displaystyle\iint_S z\,dS$，$S$ 为单位上半球面。

**思路** 参数化：$x=\sin\phi\cos\theta,y=\sin\phi\sin\theta,z=\cos\phi$（$\phi\in[0,\pi/2],\theta\in[0,2\pi]$），$dS=\sin\phi\,d\phi\,d\theta$。

**解答**

$$\iint_S z\,dS=\int_0^{2\pi}\!\!\int_0^{\pi/2}\cos\phi\cdot\sin\phi\,d\phi\,d\theta=2\pi\int_0^{\pi/2}\sin\phi\cos\phi\,d\phi=2\pi\cdot\frac{1}{2}=\pi$$

**答案** $\pi$。

---

**D.77** 用 Gauss 定理计算 $\displaystyle\oiint_S(x\,dy\,dz+y\,dz\,dx+z\,dx\,dy)$，$S$ 为单位球面外侧。

**思路** 散度 $\nabla\cdot(x,y,z)=3$，Gauss 定理：$\iiint_\Omega 3\,dV=3\cdot\tfrac{4\pi}{3}$。

**解答**

$$\oiint_S=\iiint_{\Omega}3\,dV=3\cdot\frac{4\pi}{3}=4\pi$$

**答案** $4\pi$。

---

**D.78** 证明 $\mathbf{F}=(yz,xz,xy)$ 是保守场，并求势函数 $\varphi$。

**思路** 验证旋度为零；逐步积分求 $\varphi$：$\varphi_x=yz\Rightarrow\varphi=xyz+g(y,z)$，再由 $\varphi_y=xz$ 确定 $g$。

**解答**

旋度：$\nabla\times\mathbf{F}=(x-x,y-y,z-z)=\mathbf{0}$，故为保守场。

积分：$\varphi_x=yz\Rightarrow\varphi=xyz+h(y,z)$；$\varphi_y=xz+h_y=xz\Rightarrow h_y=0$；$\varphi_z=xy+h_z=xy\Rightarrow h_z=0$，故 $h=C$。

**答案** $\varphi=xyz+C$。

---

**D.79** 计算螺旋线 $\mathbf{r}(t)=(\cos t,\sin t,t)$，$0\le t\le2\pi$ 的弧长。

**思路** $|\mathbf{r}'(t)|=|(-\sin t,\cos t,1)|=\sqrt{\sin^2t+\cos^2t+1}=\sqrt{2}$，弧长 $=\int_0^{2\pi}\sqrt{2}\,dt$。

**解答**

$$L=\int_0^{2\pi}\sqrt{2}\,dt=2\sqrt{2}\pi$$

**答案** $2\sqrt{2}\pi$。

---

**D.80** 求 $f=x^2+y^2+z^2$ 在 $(1,1,1)$ 沿 $\mathbf{l}=(1,2,2)/3$ 的方向导数。

**思路** 方向导数 $=\nabla f\cdot\mathbf{l}$，$\nabla f=(2x,2y,2z)$，在 $(1,1,1)$ 处 $=(2,2,2)$。

**解答**

$$D_{\mathbf{l}}f=(2,2,2)\cdot\frac{(1,2,2)}{3}=\frac{2+4+4}{3}=\frac{10}{3}$$

**答案** $\dfrac{10}{3}$。

---

## 第三部分：提升题详解（E.37–E.48）

---

**E.37** 证明 $\displaystyle\oint_L(y\,dx+x\,dy)=0$（$L$ 为任意简单闭曲线）。

**思路（方法一）** Green 定理，$P=y,Q=x$，计算 $Q_x-P_y=1-1=0$，被积为零。

**思路（方法二）** 凑全微分：$y\,dx+x\,dy=d(xy)$，闭曲线起终相同故为零。

**解答**

**Green 定理法：** 设 $D$ 为 $L$ 围成区域（满足 Green 定理条件），

$$\oint_L(y\,dx+x\,dy)=\iint_D\!\left(\frac{\partial(x)}{\partial x}-\frac{\partial(y)}{\partial y}\right)dA=\iint_D(1-1)\,dA=0$$

**全微分法：** 注意 $y\,dx+x\,dy=d(xy)$，故

$$\oint_L d(xy)=[xy]_{\text{起点}}^{\text{终点}}=0$$

（闭合路径起终相同。）

**答案** 无论用 Green 定理还是全微分法，$\oint_L(y\,dx+x\,dy)=0$。$\checkmark$

---

**E.38** 用 Green 定理计算椭圆 $\dfrac{x^2}{a^2}+\dfrac{y^2}{b^2}=1$ 的 $\displaystyle\oint_L x\,dy$，并得出面积 $S=\pi ab$。

**思路** 面积公式 $S=\oint_L x\,dy$（Green 定理：$P=0,Q=x$，$Q_x-P_y=1$）；参数化椭圆后计算。

**解答**

由 Green 定理：$\oint_L x\,dy=\iint_D 1\,dA=S$（椭圆面积）。

参数化：$x=a\cos\theta,y=b\sin\theta$，$\theta:0\to2\pi$，$dy=b\cos\theta\,d\theta$。

$$\oint_L x\,dy=\int_0^{2\pi}a\cos\theta\cdot b\cos\theta\,d\theta=ab\int_0^{2\pi}\cos^2\theta\,d\theta=ab\cdot\pi=\pi ab$$

验证三种公式：

- $\oint_L x\,dy=\pi ab$
- $-\oint_L y\,dx=\int_0^{2\pi}b\sin\theta\cdot a\sin\theta\,d\theta=ab\pi=\pi ab$
- $\tfrac{1}{2}\oint_L(x\,dy-y\,dx)=\pi ab$ $\checkmark$

**答案** 椭圆面积 $S=\pi ab$。

---

**E.39** 计算 $\displaystyle\int_0^1\!\!\int_x^1 e^{y^2}\,dy\,dx$（内层无初等原函数，须换序）。

**思路** 积分区域 $D=\{0\le x\le1,x\le y\le1\}$；交换次序后 $D=\{0\le y\le1,0\le x\le y\}$。

**解答**

$$\int_0^1\!\!\int_x^1 e^{y^2}\,dy\,dx=\int_0^1\!\!\int_0^y e^{y^2}\,dx\,dy=\int_0^1 ye^{y^2}\,dy$$

令 $u=y^2$，$du=2y\,dy$：

$$=\frac{1}{2}\int_0^1 e^u\,du=\frac{e-1}{2}$$

**总结** 内层积分无初等原函数 $\to$ 先画区域 $\to$ 交换积分次序 $\to$ 新内层可积。

**答案** $\dfrac{e-1}{2}$。

---

**E.40** 计算 $\displaystyle\iint_{x^2+y^2\le R^2}e^{-(x^2+y^2)}\,dA$，并推出高斯积分 $\displaystyle\int_{-\infty}^{+\infty}e^{-x^2}\,dx=\sqrt{\pi}$。

**思路** 极坐标变换，计算有限圆盘上的积分后令 $R\to\infty$；用 Fubini 定理分离变量。

**解答**

**步骤 1（有限盘）：**

$$I_R=\int_0^{2\pi}\!\!\int_0^R e^{-r^2}r\,dr\,d\theta=2\pi\cdot\left[-\frac{e^{-r^2}}{2}\right]_0^R=\pi(1-e^{-R^2})$$

**步骤 2（令 $R\to\infty$）：** $\displaystyle\iint_{\mathbb{R}^2}e^{-(x^2+y^2)}\,dA=\pi$。

**步骤 3（Fubini）：**

$$\pi=\iint_{\mathbb{R}^2}e^{-x^2-y^2}\,dA=\left(\int_{-\infty}^{+\infty}e^{-x^2}\,dx\right)^2$$

故 $\displaystyle\int_{-\infty}^{+\infty}e^{-x^2}\,dx=\sqrt{\pi}$，$\displaystyle\int_0^{+\infty}e^{-x^2}\,dx=\dfrac{\sqrt{\pi}}{2}$。

**答案** $\displaystyle\iint_{x^2+y^2\le R^2}e^{-(x^2+y^2)}\,dA=\pi(1-e^{-R^2})$；高斯积分 $=\sqrt{\pi}$。

---

**E.41** 计算 $\displaystyle\iiint_\Omega(x^2+y^2+z^2)\,dV$，$\Omega:x^2+y^2+z^2\le R^2$（球坐标）。

**思路** 球坐标：$x^2+y^2+z^2=\rho^2$，$dV=\rho^2\sin\phi\,d\rho\,d\phi\,d\theta$，$\rho\in[0,R],\phi\in[0,\pi],\theta\in[0,2\pi]$。

**解答**

$$\iiint_\Omega\rho^2\cdot\rho^2\sin\phi\,d\rho\,d\phi\,d\theta=\int_0^{2\pi}d\theta\int_0^\pi\sin\phi\,d\phi\int_0^R\rho^4\,d\rho$$

$$=2\pi\cdot 2\cdot\frac{R^5}{5}=\frac{4\pi R^5}{5}$$

**物理含义：** 均匀球（质量 $M$，密度 $\rho_0=\dfrac{3M}{4\pi R^3}$）的转动惯量 $I=\rho_0\cdot\dfrac{4\pi R^5}{5}\cdot\dfrac{2}{3}=\dfrac{2MR^2}{5}$。

**答案** $\dfrac{4\pi R^5}{5}$。

---

**E.42** 在 $x+y+z=1$（$x,y,z>0$）下求 $f=xyz$ 的最大值，并推出三元 AM-GM。

**思路** Lagrange 乘子法：$\nabla(xyz)=\lambda\nabla(x+y+z-1)$，由对称性得 $x=y=z$。

**解答**

Lagrange 条件：$yz=\lambda,xz=\lambda,xy=\lambda$，三式得 $yz=xz=xy$（均非零），故 $x=y=z$。

代入约束 $3x=1\Rightarrow x=y=z=\tfrac{1}{3}$，$f_{\max}=\tfrac{1}{27}$。

**推论（三元 AM-GM）：** 令 $s=x+y+z$，由上述结论 $xyz\le\left(\tfrac{s}{3}\right)^3$，两边开三次方：

$$\sqrt[3]{xyz}\le\frac{x+y+z}{3}\quad\checkmark$$

**答案** $f_{\max}=\dfrac{1}{27}$，在 $x=y=z=\dfrac{1}{3}$ 处取得；$\sqrt[3]{xyz}\le\dfrac{x+y+z}{3}$。

---

**E.43** 设 $f(x,y)=x^2-y^2+2xy$，$P=(1,-1)$：求梯度与最大方向导数。

**思路** 计算 $f_x=2x+2y,f_y=-2y+2x$，代入 $P$；方向导数 $=\nabla f\cdot\mathbf{l}$，沿梯度方向最大，最大值 $=|\nabla f|$。

**解答**

$$f_x=2x+2y,\quad f_y=2x-2y$$

$$\nabla f(1,-1)=(2(1)+2(-1),\,2(1)-2(-1))=(0,4)$$

方向导数：$D_{\mathbf{l}}f=\nabla f\cdot\mathbf{l}=(0,4)\cdot(\cos\alpha,\sin\alpha)=4\sin\alpha$。

最大方向：$\mathbf{l}=(0,1)$（即 $y$ 轴正向），最大方向导数 $=|\nabla f|=4$。

**梯度下降：** $(x_{n+1},y_{n+1})=(x_n,y_n)-\eta(0,4)$；步长 $\eta$ 过大振荡，过小收敛慢，常用线搜索或固定小步长。

**答案** $\nabla f(1,-1)=(0,4)$，最大方向导数为 $4$，沿 $(0,1)$ 方向。

---

**E.44** 设 $\mathbf{F}=(x,y,z)$，$\Sigma$ 为单位球面外侧，计算 $\displaystyle\iint_\Sigma\mathbf{F}\cdot d\mathbf{S}$。

**思路** Gauss 定理：$\iint_\Sigma\mathbf{F}\cdot d\mathbf{S}=\iiint_\Omega\mathrm{div}\,\mathbf{F}\,dV$；$\mathrm{div}\,\mathbf{F}=3$，$V_{\text{球}}=\tfrac{4\pi}{3}$。

**解答**

$$\iint_\Sigma\mathbf{F}\cdot d\mathbf{S}=\iiint_\Omega 3\,dV=3\cdot\frac{4\pi}{3}=4\pi$$

**直接验证：** 球面上外法向 $\mathbf{n}=(x,y,z)$（单位法向），$\mathbf{F}\cdot\mathbf{n}=x^2+y^2+z^2=1$，通量 $=1\times 4\pi=4\pi$。$\checkmark$

**答案** $4\pi$。

---

**E.45** 设 $\mathbf{F}=(y,z,x)$，$\Gamma$ 为平面 $x+y+z=1$ 与第一卦限坐标面的交线（正向），用 Stokes 定理计算 $\displaystyle\oint_\Gamma\mathbf{F}\cdot d\mathbf{r}$。

**思路** 计算旋度，取三角形曲面 $\Sigma$（$x+y+z=1,x,y,z\ge0$），法向上（$\mathbf{n}=\tfrac{1}{\sqrt{3}}(1,1,1)$）。

**解答**

**旋度：**

$$\nabla\times\mathbf{F}=\begin{vmatrix}\mathbf{i}&\mathbf{j}&\mathbf{k}\\\partial_x&\partial_y&\partial_z\\y&z&x\end{vmatrix}=(0-1)\mathbf{i}-(1-0)\mathbf{j}+(0-1)\mathbf{k}=(-1,-1,-1)$$

**Stokes 定理：**

$$\oint_\Gamma\mathbf{F}\cdot d\mathbf{r}=\iint_\Sigma(-1,-1,-1)\cdot\mathbf{n}\,dS$$

法向 $\mathbf{n}=\tfrac{(1,1,1)}{\sqrt{3}}$，$(-1,-1,-1)\cdot\mathbf{n}=\tfrac{-3}{\sqrt{3}}=-\sqrt{3}$。

三角形面积：三顶点 $(1,0,0),(0,1,0),(0,0,1)$，面积 $=\tfrac{\sqrt{3}}{2}$。

$$\oint_\Gamma\mathbf{F}\cdot d\mathbf{r}=(-\sqrt{3})\cdot\frac{\sqrt{3}}{2}=-\frac{3}{2}$$

**答案** $-\dfrac{3}{2}$。

---

**E.46** 设映射 $x=u\cos v-v\sin u,y=u\sin v+v\cos u$，在 $(u,v)=(1,0)$ 处计算 Jacobi 矩阵及其行列式。

**思路** 分别计算 $x_u,x_v,y_u,y_v$，代入 $(1,0)$，验证 $\det J\ne0$。

**解答**

$$x_u=\cos v-v\cos u,\quad x_v=-u\sin v-\sin u$$

$$y_u=\sin v-v\sin u,\quad y_v=u\cos v+\cos u$$

在 $(u,v)=(1,0)$（$\cos0=1,\sin0=0,\sin1\approx0.841,\cos1\approx0.540$）：

$$x_u=1-0=1,\quad x_v=0-\sin1=-\sin1$$

$$y_u=0-0=0,\quad y_v=\cos1+\cos1=1+\cos1$$

$$J=\begin{pmatrix}1&-\sin1\\0&1+\cos1\end{pmatrix},\quad \det J=1\cdot(1+\cos1)-(-\sin1)\cdot0=1+\cos1\approx1.54\ne0$$

局部逆映射存在（隐函数定理）。

**答案** $\det J=1+\cos1\approx1.54\ne0$，局部逆映射在 $(1,0)$ 附近存在。

---

**E.47** 设 $S$ 为球面 $x^2+y^2+z^2=4$（$z\ge0$，上半球面），计算 $\displaystyle\iint_S z\,dS$。

**思路** 球面参数化：$x=2\sin\phi\cos\theta,y=2\sin\phi\sin\theta,z=2\cos\phi$，$\phi\in[0,\pi/2],\theta\in[0,2\pi]$，$dS=4\sin\phi\,d\phi\,d\theta$。

**解答**

$$\iint_S z\,dS=\int_0^{2\pi}\!\!\int_0^{\pi/2}(2\cos\phi)\cdot4\sin\phi\,d\phi\,d\theta=8\int_0^{2\pi}d\theta\int_0^{\pi/2}\sin\phi\cos\phi\,d\phi$$

$$=8\cdot2\pi\cdot\frac{1}{2}=8\pi$$

**交叉验证（质心公式）：** 上半球面面积 $=2\pi R^2=8\pi$（$R=2$），$\bar{z}=R/2=1$（上半球面质心高度），$\iint_S z\,dS=\bar{z}\cdot\text{Area}=1\cdot8\pi=8\pi$。$\checkmark$

**答案** $8\pi$。

---

**E.48** 计算 $\displaystyle\int_L\frac{x\,dy-y\,dx}{x^2+y^2}$，$L$ 为从 $(1,0)$ 逆时针沿上半圆到 $(-1,0)$。

**思路** 验证 $Q_x=P_y$（$x^2+y^2\ne0$ 时），但区域含奇点原点，不能直接用 Green 定理；参数化直接计算。

**解答**

$P=-\dfrac{y}{x^2+y^2},Q=\dfrac{x}{x^2+y^2}$；验证：

$$Q_x=\frac{y^2-x^2}{(x^2+y^2)^2}=P_y$$（注意 $P_y=-\dfrac{x^2-y^2}{(x^2+y^2)^2}$，需重算）

实际 $Q_x=\dfrac{(x^2+y^2)-x\cdot2x}{(x^2+y^2)^2}=\dfrac{y^2-x^2}{(x^2+y^2)^2}$，

$P_y=\dfrac{-(x^2+y^2)+y\cdot2y}{(x^2+y^2)^2}=\dfrac{y^2-x^2}{(x^2+y^2)^2}$，确实 $Q_x=P_y$。

但 $L$ 内部含原点（奇点），不可直接用 Green。改用参数化：$x=\cos t,y=\sin t$（$t:0\to\pi$）。

$$dx=-\sin t\,dt,\quad dy=\cos t\,dt,\quad x^2+y^2=1$$

$$\int_L\frac{x\,dy-y\,dx}{x^2+y^2}=\int_0^\pi\frac{\cos t\cos t\,dt-\sin t(-\sin t\,dt)}{1}=\int_0^\pi(\cos^2t+\sin^2t)\,dt=\int_0^\pi1\,dt=\pi$$

**说明：** $\dfrac{x\,dy-y\,dx}{x^2+y^2}=d(\arctan\tfrac{y}{x})$ 在去掉原点的单连通区域上成立，但上半圆路径对应 $\arctan(\tan t)$ 从 $0$ 到 $\pi$，增量正好为 $\pi$。

**答案** $\pi$。

---

## 总结

| 专题 | 核心方法 | 典型题 |
|------|---------|--------|
| 偏导数与链式法则 | 视另一变量为常数；复合函数路径图 | C.51–C.54, D.61–D.64 |
| 二重积分 | 先画区域；必要时换极坐标或交换次序 | C.55–C.57, D.67–D.71, E.39 |
| 多元极值与优化 | Hessian 判别；Lagrange 乘子法 | C.62–C.63, D.65, D.74, E.42 |
| 向量微积分 | Green / Gauss / Stokes 三大定理配合散度、旋度计算 | C.58–C.65, D.72–D.73, D.77, E.37–E.38, E.44–E.45 |
| 方向导数与梯度 | $D_{\mathbf{l}}f=\nabla f\cdot\hat{\mathbf{l}}$，最大值 $=|\nabla f|$ | C.61, D.80, E.43 |
| 曲面积分 | 参数化 $\to$ 面积元 $dS=|\mathbf{r}_u\times\mathbf{r}_v|\,du\,dv$ | D.76, E.47 |
| 换序与极坐标 | 先画积分域，找换序等价描述；$x=r\cos\theta$ | D.71, E.39–E.41 |
| 高斯积分推导 | 二维极坐标 $+$ Fubini $\Rightarrow\int e^{-x^2}=\sqrt{\pi}$ | E.40 |
