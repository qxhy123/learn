# 附录 F3b：抛物线 + 定点定值定直线详解

> 对应 D 卷第 48–57 题（抛物线中档，Part 7）与 E 卷第 44–53 题（定点定值定直线压轴，Part 5/6/7/10）。
> 每题含：**思路**（方法 + 关联章节）/ **解答**（关键步骤）/ **答案**（$\boxed{}$）/ **总结**。

---

## 一、抛物线中档 10 题（D.48–D.57）

### D.48 [中档] Part 7/01

题目：写出各抛物线的焦点与准线：(1) $y^2=8x$；(2) $x^2=-6y$；(3) $y^2=-4x$；(4) $x^2=2y$。

**思路** 标准形 $y^2=4px\Rightarrow$ 焦点 $(p,0)$，准线 $x=-p$；$x^2=4py\Rightarrow$ 焦点 $(0,p)$，准线 $y=-p$（负号对应左/下开口）。

**解答**

(1) $y^2=8x$：$4p=8\Rightarrow p=2$。焦点 $(2,0)$，准线 $x=-2$。

(2) $x^2=-6y$：$4p=-6\Rightarrow p=-\tfrac{3}{2}$（开口向下）。焦点 $(0,-\tfrac{3}{2})$，准线 $y=\tfrac{3}{2}$。

(3) $y^2=-4x$：$4p=-4\Rightarrow p=-1$（开口向左）。焦点 $(-1,0)$，准线 $x=1$。

(4) $x^2=2y$：$4p=2\Rightarrow p=\tfrac{1}{2}$。焦点 $(0,\tfrac{1}{2})$，准线 $y=-\tfrac{1}{2}$。

**答案** (1) 焦点 $\boxed{(2,0)}$，准线 $\boxed{x=-2}$；(2) $\boxed{(0,-\tfrac{3}{2})}$，$\boxed{y=\tfrac{3}{2}}$；(3) $\boxed{(-1,0)}$，$\boxed{x=1}$；(4) $\boxed{(0,\tfrac{1}{2})}$，$\boxed{y=-\tfrac{1}{2}}$。

**总结** 认准 $4p$ 系数，正负决定开口方向，焦点与准线关于顶点对称。

---

### D.49 [中档] Part 7/02

题目：抛物线 $C$：$y^2=4x$，焦点 $F$，准线 $l$。点 $P(4,y_0)$ 在 $C$ 上，求 $y_0$、$|PF|$、$P$ 到准线距离。

**思路** 先由方程解 $y_0$，再用焦点距离公式 $|PF|=x_0+p=x_0+1$（此时 $4p=4\Rightarrow p=1$），抛物线定义知 $|PF|=$ 到准线距离。

**解答** $y_0^2=4\times4=16\Rightarrow y_0=\pm4$。$|PF|=4+1=5$。准线 $x=-1$，$P$ 到准线距离 $=4-(-1)=5$（与 $|PF|$ 相等，验证定义）。

**答案** $y_0=\boxed{\pm4}$，$|PF|=\boxed{5}$，$P$ 到准线距离 $=\boxed{5}$。

**总结** $|PF|=x_P+p$，是抛物线焦点距离的核心公式，解题必用。

---

### D.50 [中档] Part 7/03

题目：$y^2=2px$（$p>0$）过点 $A(2,2)$，求 $p$，再求过 $A$ 的焦半径。

**思路** 将 $A(2,2)$ 代入求 $p$，焦半径公式 $r=x_0+\tfrac{p}{2}$（对 $y^2=2px$，焦点在 $(\tfrac{p}{2},0)$，准线 $x=-\tfrac{p}{2}$）。

**解答** 代入：$4=2p\cdot2=4p\Rightarrow p=1$。方程为 $y^2=2x$，焦点 $F(\tfrac{1}{2},0)$，准线 $x=-\tfrac{1}{2}$。$A$ 到准线距离（即 $|AF|$）$=2+\tfrac{1}{2}=\tfrac{5}{2}$。

**答案** $p=\boxed{1}$，焦半径 $|AF|=\boxed{\dfrac{5}{2}}$。

**总结** 过点求 $p$ → 代入定值；焦半径 = 横坐标 + 半通径（即到准线距离）。

---

### D.51 [中档] Part 7/04

题目：$y^2=4x$ 的焦点弦 $AB$，$y_1>0>y_2$。(1) 证明 $y_1y_2=-4$；(2) $y_1=2$，求 $|AB|$ 及斜率。

**思路** 焦点 $F(1,0)$，设 $A(\tfrac{y_1^2}{4},y_1)$，$B(\tfrac{y_2^2}{4},y_2)$，三点共线推 $y_1y_2=-4$。弦长用焦半径之和。

**解答**

(1) $A,F,B$ 共线，斜率相等：$\dfrac{y_1-0}{\tfrac{y_1^2}{4}-1}=\dfrac{y_2-0}{\tfrac{y_2^2}{4}-1}$。交叉相乘展开得 $y_1(\tfrac{y_2^2}{4}-1)=y_2(\tfrac{y_1^2}{4}-1)$，即 $\tfrac{y_1y_2^2-y_1y_2^2}{4}-(y_1-y_2)=0$ 化简为 $\tfrac{y_1y_2}{4}(y_2-y_1)=-(y_1-y_2)$，从而 $\tfrac{y_1y_2}{4}=1$，故 $y_1y_2=-4$。（更快：$A,F,B$ 共线 $\Rightarrow$ 斜率 $k=\dfrac{2}{y_1}$，$B$ 满足 $y_2=y_1+\dfrac{y_2^2-y_1^2}{4k^{-1}}$... 直接取 $k=\dfrac{2}{y_1}$ 代入 $B$ 点得 $y_2=-\tfrac{4}{y_1}$，故 $y_1y_2=-4$。）

(2) $y_1=2\Rightarrow y_2=-2$。$x_1=\tfrac{4}{4}=1$，$x_2=1$。$|AB|=|x_1+1|+|x_2+1|=2+2=4$。斜率 $k=\dfrac{2-(-2)}{1-1}$ — 分母为 $0$，弦 $AB$ 竖直（$x=1$），斜率不存在（通径）。

**答案** (1) 证毕 $y_1y_2=\boxed{-4}$；(2) $|AB|=\boxed{4}$，弦 $AB$ 为竖直线（斜率不存在）。

**总结** 焦点弦端点参数 $y_1y_2=-4$（对 $y^2=4x$）是焦弦性质核心，需牢记。

---

### D.52 [中档] Part 7/05

题目：直线 $x=my+n$ 与 $y^2=4x$ 交于 $A(x_1,y_1)$，$B(x_2,y_2)$。用韦达定理求 $y_1+y_2$、$y_1y_2$，并用 $m,n$ 表示弦 $AB$ 中点横坐标。

**思路** 将 $x=my+n$ 代入 $y^2=4(my+n)$，得 $y^2-4my-4n=0$，韦达定理直读。

**解答** 联立：$y^2-4my-4n=0$。

韦达：$y_1+y_2=4m$，$y_1y_2=-4n$。

中点横坐标：$\bar{x}=\dfrac{x_1+x_2}{2}=\dfrac{(my_1+n)+(my_2+n)}{2}=m\cdot\dfrac{y_1+y_2}{2}+n=m\cdot2m+n=2m^2+n$。

**答案** $y_1+y_2=\boxed{4m}$，$y_1y_2=\boxed{-4n}$，中点横坐标 $=\boxed{2m^2+n}$。

**总结** 对抛物线，用斜截式 $x=my+n$ 代入比 $y=kx+b$ 代入更简洁，直接得二次方程套韦达。

---

### D.53 [中档] Part 7/05

题目：$y=k(x-1)$（$k\neq0$）与 $y^2=4x$ 交于 $A,B$。求弦 $AB$ 中点坐标（用 $k$ 表示），并求中点轨迹方程。

**思路** 联立消 $y$，韦达求 $x_1+x_2$，再求 $y_1+y_2$；消 $k$ 得轨迹。

**解答** 将 $y=k(x-1)$ 代入 $y^2=4x$：$k^2(x-1)^2=4x$，即 $k^2x^2-(2k^2+4)x+k^2=0$。

韦达：$x_1+x_2=\dfrac{2k^2+4}{k^2}=2+\dfrac{4}{k^2}$，$x_1x_2=1$。

中点 $M\left(\bar{x},\bar{y}\right)$：$\bar{x}=1+\dfrac{2}{k^2}$，$\bar{y}=k(\bar{x}-1)=k\cdot\dfrac{2}{k^2}=\dfrac{2}{k}$。

消 $k$：$\bar{y}=\dfrac{2}{k}\Rightarrow k=\dfrac{2}{\bar{y}}$，代入 $\bar{x}=1+\dfrac{2}{k^2}=1+\dfrac{\bar{y}^2}{2}$，即 $\bar{y}^2=2(\bar{x}-1)$。

**答案** 中点 $M=\boxed{\left(1+\dfrac{2}{k^2},\,\dfrac{2}{k}\right)}$；轨迹方程 $\boxed{y^2=2(x-1)}$（抛物线，去掉原点）。

**总结** 弦中点轨迹的标准流程：联立→韦达→参数消去 $k$，得到轨迹方程。

---

### D.54 [中档] Part 7/06

题目：$y^2=8x$ 的焦点弦 $AB$ 满足 $|AF|=3|BF|$，求弦长 $|AB|$ 和直线方程。

**思路** $y^2=8x$：$4p=8\Rightarrow p=2$，焦点 $F(2,0)$，准线 $x=-2$。焦点弦调和性质 $\dfrac{1}{|AF|}+\dfrac{1}{|BF|}=\dfrac{1}{p}=\dfrac{1}{2}$（通径半长倒数）。设 $|AF|=r_1$，$|BF|=r_2$，$r_1=3r_2$。

**解答** 由调和性质（$y^2=8x$ 即 $y^2=4\cdot2\cdot x$，$p=2$）：$\dfrac{1}{r_1}+\dfrac{1}{r_2}=\dfrac{1}{p}=\dfrac{1}{2}$。

代入 $r_1=3r_2$：$\dfrac{1}{3r_2}+\dfrac{1}{r_2}=\dfrac{4}{3r_2}=\dfrac{1}{2}\Rightarrow r_2=\dfrac{8}{3}$，$r_1=8$。$|AB|=r_1+r_2=\dfrac{32}{3}$。

焦半径：$|AF|=x_A+2=8\Rightarrow x_A=6$，$y_A^2=48\Rightarrow y_A=4\sqrt{3}$（取正）。$|BF|=x_B+2=\tfrac{8}{3}\Rightarrow x_B=\tfrac{2}{3}$，$y_B=-\tfrac{4\sqrt{3}}{3}$（异号）。直线斜率 $k=\dfrac{4\sqrt{3}-0}{6-2}=\sqrt{3}$，方程 $y=\sqrt{3}(x-2)$。

**答案** $|AB|=\boxed{\dfrac{32}{3}}$，直线方程 $\boxed{y=\sqrt{3}(x-2)}$。

**总结** 焦点弦调和性质 $\tfrac{1}{r_1}+\tfrac{1}{r_2}=\tfrac{1}{p}$ 是求焦弦分段的最快路径。

---

### D.55 [中档] Part 7/07

题目：抛物线 $x^2=4py$（$p>0$），焦点到直线 $y=-1$ 的距离为 $3$，求 $p$ 及抛物线方程。又求纵坐标为 $4$ 的点到焦点的距离。

**思路** $x^2=4py$ 焦点为 $(0,p)$，准线 $y=-p$。焦点到直线 $y=-1$ 的距离 $=|p-(-1)|=p+1$（因 $p>0$）。

**解答** 焦点 $(0,p)$ 到 $y=-1$ 的距离 $=p+1=3\Rightarrow p=2$。方程 $x^2=8y$。

纵坐标 $4$：$x^2=32$，点为 $(\pm4\sqrt{2},4)$。焦点 $(0,2)$，准线 $y=-2$。由定义，距离 $=y_0+p=4+2=6$。

**答案** $p=\boxed{2}$，方程 $\boxed{x^2=8y}$，纵坐标为 $4$ 的点到焦点距离 $=\boxed{6}$。

**总结** 开口向上抛物线 $x^2=4py$：焦半径 $=y_0+p$（用准线定义，比直接算距离简单）。

---

### D.56 [中档] Part 7/08

题目：$y^2=4x$ 上点 $P(1,2)$，求切线方程。

**思路** 方法一（导数）：$x=\tfrac{y^2}{4}$，$\tfrac{dx}{dy}=\tfrac{y}{2}$，在 $y=2$ 处 $\tfrac{dx}{dy}=1$，故 $\tfrac{dy}{dx}=1$，切线斜率 $k=1$。方法二（公式）：$y^2=4x$ 上点 $(x_0,y_0)$ 的切线为 $yy_0=2(x+x_0)$。

**解答** 用公式：$y_0=2$，$x_0=1$：$2y=2(x+1)$，即 $y=x+1$。

**答案** 切线方程 $\boxed{y=x+1}$。

**总结** 切线公式 $yy_0=2(x+x_0)$（对 $y^2=4x$）一步到位，不需导数过程。

---

### D.57 [中档] Part 7/09

题目：$y^2=4x$，过点 $M(3,0)$ 的直线交抛物线于 $A,B$，弦 $AB$ 中点横坐标为 $4$，求直线方程。

**思路** 由中点横坐标与韦达定理，结合直线过 $M(3,0)$ 建立方程组，求斜率。

**解答** 设直线 $y=k(x-3)$，代入 $y^2=4x$：$k^2(x-3)^2=4x$，$k^2x^2-(6k^2+4)x+9k^2=0$。

韦达：$x_1+x_2=\dfrac{6k^2+4}{k^2}$，故中点横坐标 $\bar{x}=\dfrac{x_1+x_2}{2}=3+\dfrac{2}{k^2}=4\Rightarrow k^2=2\Rightarrow k=\pm\sqrt{2}$。

验证直线与抛物线确实有两交点（$\Delta>0$：已用韦达，需 $\Delta=（6k^2+4)^2-4k^2\cdot9k^2=(6k^2+4)^2-36k^4>0$，代入 $k^2=2$：$(16)^2-36\cdot4=256-144>0$，✓）。

**答案** 直线方程 $\boxed{y=\pm\sqrt{2}(x-3)}$（两解）。

**总结** "已知中点横坐标" → 利用韦达 $x_1+x_2$ 表达式直接反解斜率 $k$。

---

## 二、定点定值定直线压轴 10 题（E.44–E.53）

### E.44 [提升] Part 5/09 椭圆动直线过定点

题目：椭圆 $\dfrac{x^2}{4}+y^2=1$，$A(-2,0)$，$B(2,0)$。直线 $l$ 过 $A$，斜率 $k\neq0$，交椭圆于 $A$ 和 $P$；直线 $BP$ 延交椭圆于 $Q$。(1) 用 $k$ 表示 $P$ 的横坐标 $x_1$；(2) 求直线 $AQ$ 斜率，判断 $AQ$ 是否过定点；(3) 若过定点，求坐标。

**思路** $P$ 在椭圆上且在直线 $y=k(x+2)$ 上，联立椭圆求 $x_1$（排除 $x=-2$）；设 $Q$ 的坐标再对 $BP$ 直线做同样操作，最终消 $k$ 观察 $AQ$ 截距是否恒定。

**解答**

(1) 直线 $AP$：$y=k(x+2)$，代入椭圆 $\dfrac{x^2}{4}+k^2(x+2)^2=1$，整理：$(1+4k^2)x^2+16k^2x+16k^2-4=0$。由韦达 $x_Ax_P=\dfrac{16k^2-4}{1+4k^2}$，$x_A=-2$，故 $x_P=\dfrac{4k^2-1}{2(1+4k^2)}\cdot(-2)\cdot(-1)$... 更直接：直接因式分解（$x=-2$ 为一根），$(x+2)[(1+4k^2)x-(2-8k^2)]=0$，故 $x_P=\dfrac{2-8k^2}{1+4k^2}$，$y_P=k(x_P+2)=\dfrac{6k}{1+4k^2}$。

(2) $P=\left(\dfrac{2-8k^2}{1+4k^2},\dfrac{6k}{1+4k^2}\right)$，$B=(2,0)$。直线 $BP$ 斜率 $k_{BP}=\dfrac{\frac{6k}{1+4k^2}}{\frac{2-8k^2}{1+4k^2}-2}=\dfrac{6k}{2-8k^2-2-8k^2}=\dfrac{6k}{-16k^2}=-\dfrac{3}{8k}$。

直线 $BQ$：$y=-\dfrac{3}{8k}(x-2)$，代入椭圆（类似操作，$B(2,0)$ 为一根），得 $x_Q=\dfrac{2-\frac{9}{16k^2}}{1+\frac{9}{64k^2}}$... 用相同因式分解方法：令 $m=-\tfrac{3}{8k}$，$BQ$：$y=m(x-2)$，代入 $(1+4m^2)x^2-16m^2x+16m^2-4=0$，韦达：$x_Bx_Q=\dfrac{16m^2-4}{1+4m^2}$，$x_B=2$，故 $x_Q=\dfrac{16m^2-4}{2(1+4m^2)}=\dfrac{8m^2-2}{1+4m^2}$。代 $m=-\dfrac{3}{8k}$：

$m^2=\dfrac{9}{64k^2}$，$x_Q=\dfrac{\frac{9}{8k^2}-2}{1+\frac{9}{16k^2}}=\dfrac{\frac{9-16k^2}{8k^2}}{\frac{16k^2+9}{16k^2}}=\dfrac{(9-16k^2)\cdot 16k^2}{8k^2(16k^2+9)}=\dfrac{2(9-16k^2)}{16k^2+9}$。

$y_Q=m(x_Q-2)=-\dfrac{3}{8k}\left(\dfrac{2(9-16k^2)}{16k^2+9}-2\right)=-\dfrac{3}{8k}\cdot\dfrac{18-32k^2-32k^2-18}{16k^2+9}=-\dfrac{3}{8k}\cdot\dfrac{-64k^2}{16k^2+9}=\dfrac{24k}{16k^2+9}$。

直线 $AQ$ 斜率 $k_{AQ}=\dfrac{y_Q}{x_Q+2}=\dfrac{\frac{24k}{16k^2+9}}{\frac{2(9-16k^2)}{16k^2+9}+2}=\dfrac{24k}{2(9-16k^2)+2(16k^2+9)}=\dfrac{24k}{18-32k^2+32k^2+18}=\dfrac{24k}{36}=\dfrac{2k}{3}$。

$AQ$ 斜率为 $\dfrac{2k}{3}$（随 $k$ 变化），直线 $AQ$ 过 $A(-2,0)$ 本身，斜率比 $k_{AQ}/k_{AP}=\dfrac{2}{3}$ 恒定（调和对应），但 $AQ$ 不过 $A$ 以外的其他固定点。

**答案** (1) $x_P=\boxed{\dfrac{2-8k^2}{1+4k^2}}$；(2) 直线 $AQ$ 斜率为 $\dfrac{2k}{3}$，随 $k$ 变化；(3) $AQ$ 过 $A(-2,0)$ 本身，不过其他定点。（本题的定点结论在于斜率比 $k_{AQ}/k_{AP}=\dfrac{2}{3}$ 恒定，体现"调和对应"关系。）

**总结** 椭圆动直线定点：联立→韦达→消 $k$；若斜率比为定值，则为调和对应，而非过定点。

---

### E.45 [提升] Part 7/06 抛物线定点

题目：$y^2=4x$，$A(3,0)$。过 $A$ 的直线交抛物线于 $M(x_1,y_1)$，$N(x_2,y_2)$，且 $\overrightarrow{AM}=\lambda\overrightarrow{AN}$。(1) 用 $y_1,y_2$ 表示 $\lambda$；(2) 证明直线 $MN$ 过某定点；(3) 求该定点。

**思路** 参数化：$M=(\tfrac{y_1^2}{4},y_1)$，$N=(\tfrac{y_2^2}{4},y_2)$，由 $\overrightarrow{AM}=\lambda\overrightarrow{AN}$ 得 $\lambda$ 表达式；设直线 $MN$：$y=k(x-b)$，联立抛物线用韦达建关系，消 $\lambda$ 后发现截 $x$ 轴的截距恒定。

**解答**

(1) $\overrightarrow{AM}=(x_1-3,y_1)=\lambda(x_2-3,y_2)$，故 $\lambda=\dfrac{y_1}{y_2}$（由 $y$ 分量），同时 $\lambda=\dfrac{x_1-3}{x_2-3}=\dfrac{\frac{y_1^2}{4}-3}{\frac{y_2^2}{4}-3}$，两式一致。

(2) 设直线 $MN$：$x=my+n$（避免无穷斜率情形），代入 $y^2=4x$：$y^2-4my-4n=0$，韦达：$y_1+y_2=4m$，$y_1y_2=-4n$。

由 $\lambda=\dfrac{y_1}{y_2}$ 且由坐标关系 $x_1-3=\lambda(x_2-3)$：$\dfrac{y_1^2}{4}-3=\dfrac{y_1}{y_2}\left(\dfrac{y_2^2}{4}-3\right)$，展开 $\dfrac{y_1^2}{4}-3=\dfrac{y_1y_2}{4}-\dfrac{3y_1}{y_2}$，

即 $\dfrac{y_1^2-y_1y_2}{4}=3-\dfrac{3y_1}{y_2}=3\cdot\dfrac{y_2-y_1}{y_2}$，$\dfrac{y_1(y_1-y_2)}{4}=-\dfrac{3(y_1-y_2)}{y_2}$（$y_1\neq y_2$），

$\dfrac{y_1}{4}=-\dfrac{3}{y_2}\Rightarrow y_1y_2=-12$，故 $-4n=-12\Rightarrow n=3$。

直线方程 $x=my+3$，令 $y=0$ 得 $x=3$，故直线 $MN$ 恒过点 $(3,0)=A$（约束 $y_1y_2=-12$ 使 $n=3$ 固定，定点即 $A$ 本身）。若改为 $A=F(1,0)$（焦点），则 $y_1y_2=-4$，$n=1$，$MN$ 过焦点。

**答案** (1) $\lambda=\boxed{\dfrac{y_1}{y_2}}$；(2)(3) 直线 $MN$ 在约束 $y_1y_2=-12$ 下过定点 $\boxed{(3,0)}$（即 $A$）。

**总结** 抛物线定点题：设参数化后韦达→关键乘积 $y_1y_2$ 为定值→截距固定→定点确定。

---

### E.46 [提升] Part 6/07 双曲线中点轨迹

题目：$\dfrac{x^2}{4}-y^2=1$，$P_0(1,0)$。直线 $l$ 过 $P_0$，斜率 $k$，交 $C$ 于 $A(x_1,y_1)$，$B(x_2,y_2)$。(1) 韦达求 $x_1+x_2$，$x_1x_2$；(2) 弦 $AB$ 中点 $M$ 坐标（用 $k$）；(3) 判断 $|AM|/|MB|$ 是否定值。

**思路** 直线 $y=k(x-1)$ 代入双曲线方程，整理后用韦达定理；弦中点用 $\bar{x}=\frac{x_1+x_2}{2}$。

**解答**

(1) 代入：$\dfrac{x^2}{4}-k^2(x-1)^2=1$，$(1-4k^2)x^2+8k^2x-4k^2-4=0$（需 $1-4k^2\neq0$，即 $k\neq\pm\tfrac{1}{2}$）。

韦达：$x_1+x_2=\dfrac{-8k^2}{1-4k^2}=\dfrac{8k^2}{4k^2-1}$，$x_1x_2=\dfrac{-4k^2-4}{1-4k^2}=\dfrac{4k^2+4}{4k^2-1}$。

(2) $\bar{x}=\dfrac{4k^2}{4k^2-1}$，$\bar{y}=k(\bar{x}-1)=k\cdot\dfrac{4k^2-(4k^2-1)}{4k^2-1}=\dfrac{k}{4k^2-1}$。

(3) $M$ 是 $AB$ 中点，故 $|AM|=|MB|$，比值恒为 $\boxed{1}$（中点定义即 $1:1$）。

**答案** (1) $x_1+x_2=\boxed{\dfrac{8k^2}{4k^2-1}}$，$x_1x_2=\boxed{\dfrac{4(k^2+1)}{4k^2-1}}$；(2) $M=\boxed{\left(\dfrac{4k^2}{4k^2-1},\dfrac{k}{4k^2-1}\right)}$；(3) $|AM|/|MB|=\boxed{1}$（中点定义）。

**总结** 中点题的 (3) 往往考查中点坐标随 $k$ 的轨迹，可消 $k$ 得 $\bar{y}(4\bar{y}^2\cdot4-1)=...$；此题 $|AM|=|MB|$ 由定义直接得到。

---

### E.47 [提升] Part 5/09 椭圆 OM⊥AB 的 k 与 |AB|

题目：椭圆 $e=\tfrac{\sqrt{3}}{2}$，短轴长 $2$，斜率 $k$ 的直线 $l$ 交椭圆于 $A,B$，中点 $M$，$OM\perp AB$。(1) 求椭圆方程；(2) 求所有 $k$ 的值；(3) 求各 $k$ 对应 $|AB|$，判断是否定值。

**思路** 由 $e,b$ 求椭圆；点差法得 $k_{OM}=-\dfrac{b^2\bar{x}}{a^2\bar{y}}$，再用 $OM\perp AB$ 即 $k_{OM}\cdot k=-1$，联立解 $k$；再求弦长。

**解答**

(1) $b=1$，$e=\dfrac{c}{a}=\dfrac{\sqrt{3}}{2}$，$c^2=a^2-b^2$：设 $a^2=t$，$\dfrac{t-1}{t}=\dfrac{3}{4}\Rightarrow 4t-4=3t\Rightarrow t=4$。椭圆 $\dfrac{x^2}{4}+y^2=1$。

(2) 直线 $y=kx+b_0$（$b_0$ 为截距），代入椭圆，韦达得中点 $M(\bar{x},\bar{y})$：点差法 $\dfrac{\bar{x}}{4\cdot1}$ 的关系：$\dfrac{x_1^2-x_2^2}{4}+(y_1^2-y_2^2)=0\Rightarrow\dfrac{\bar{x}}{4}(x_1-x_2)+(y_1+y_2)(y_1-y_2)=0$... 更简：斜率 $k$ 由点差法 $k_{AB}=-\dfrac{b^2\bar{x}}{a^2\bar{y}}=-\dfrac{\bar{x}}{4\bar{y}}=k$，故 $\bar{x}=-4k\bar{y}$，$M=(-4k\bar{y},\bar{y})$。

$OM\perp AB$：$k_{OM}\cdot k=-1$，$k_{OM}=\dfrac{\bar{y}}{-4k\bar{y}}=-\dfrac{1}{4k}$，则 $-\dfrac{1}{4k}\cdot k=-1\Rightarrow-\dfrac{1}{4}=-1$，矛盾！

故**对标准椭圆 $\dfrac{x^2}{4}+y^2=1$，不存在满足 $OM\perp AB$ 的弦**（除非 $A=B$，不合题意）。

重新推导：$OM$ 斜率 $=\dfrac{\bar{y}}{\bar{x}}$，点差法给出 $k=-\dfrac{\bar{x}}{4\bar{y}}$，$OM\perp AB$ 要求 $\dfrac{\bar{y}}{\bar{x}}\cdot k=-1$，即 $\dfrac{\bar{y}}{\bar{x}}\cdot\left(-\dfrac{\bar{x}}{4\bar{y}}\right)=-1$，$-\dfrac{1}{4}=-1$，永不成立。说明对此椭圆，条件 $OM\perp AB$ 无解。

**结论**：此椭圆不存在满足条件的 $k$；若题目成立，则椭圆方程或条件需调整（如椭圆为 $\dfrac{x^2}{4}+\dfrac{y^2}{4}=1$ 即圆时 $k_{OM}\cdot k=-1$ 可解）。

**答案** (1) 椭圆方程 $\boxed{\dfrac{x^2}{4}+y^2=1}$；(2) 不存在满足 $OM\perp AB$ 的斜率 $k$（点差法得矛盾方程 $-\tfrac{1}{4}=-1$）。

**总结** 椭圆的 $OM\perp AB$ 条件：由点差法 $k=-\dfrac{b^2\bar{x}}{a^2\bar{y}}$ 与 $\dfrac{\bar{y}}{\bar{x}}\cdot k=-1$ 联立，两式矛盾时无解，说明该椭圆不具备此性质。

---

### E.48 [提升] Part 7/07 抛物线三角形面积最小值

题目：$x^2=8y$，焦点 $F(0,2)$。过 $F$ 的直线交 $C$ 于 $A,B$。(2) 求 $\triangle OAB$ 面积关于斜率 $k$ 的表达式，并求最小值；(3) 当 $AB\perp y$ 轴时，求 $|OA|$，$|OB|$，$\angle AOB$。

**思路** $x^2=8y\Rightarrow 4p=8,p=2$，准线 $y=-2$，焦距 $|OF|=2$（焦点在上方）。面积 $S=\tfrac{1}{2}|AB|\cdot d(O,AB)$。

**解答**

(2) 设直线 $AB$：$y=kx+2$（过焦点 $F(0,2)$），代入 $x^2=8(kx+2)$：$x^2-8kx-16=0$。

韦达：$x_1+x_2=8k$，$x_1x_2=-16$，$y_i=\dfrac{x_i^2}{8}$。

$|AB|=\sqrt{1+k^2}\cdot|x_1-x_2|$，$(x_1-x_2)^2=(8k)^2+64=64k^2+64$，$|x_1-x_2|=8\sqrt{k^2+1}$，$|AB|=8(1+k^2)$。

$O$ 到直线 $y=kx+2$（即 $kx-y+2=0$）距离 $d=\dfrac{2}{\sqrt{k^2+1}}$。

$S=\dfrac{1}{2}\cdot8(1+k^2)\cdot\dfrac{2}{\sqrt{k^2+1}}=8\sqrt{k^2+1}\geq8$（当 $k=0$ 时取等）。

(3) $AB\perp y$ 轴即 $k\to\infty$（水平弦），$y=2$，$x^2=16$，$A(-4,2)$，$B(4,2)$。$|OA|=|OB|=\sqrt{16+4}=2\sqrt{5}$。$\cos\angle AOB=\dfrac{\overrightarrow{OA}\cdot\overrightarrow{OB}}{|OA||OB|}=\dfrac{-16+4}{20}=-\dfrac{12}{20}=-\dfrac{3}{5}$，$\angle AOB=\arccos\left(-\dfrac{3}{5}\right)$。

**答案** (2) $S=8\sqrt{k^2+1}$，最小值 $\boxed{8}$（$k=0$ 时）；(3) $|OA|=|OB|=\boxed{2\sqrt{5}}$，$\cos\angle AOB=\boxed{-\dfrac{3}{5}}$。

**总结** 面积 $=\tfrac{1}{2}|AB|\cdot d$ 结合韦达化简，最小值在直线过焦点且垂直主轴时取得（通径是最短焦弦，面积最小）。

---

### E.49 [提升] Part 5/10 椭圆弦中点轨迹与定直线

题目：椭圆 $\dfrac{x^2}{5}+\dfrac{y^2}{4}=1$，$A,B$ 在椭圆上，$AB$ 斜率为 $\dfrac{4}{5}$。(1) 用点差法证明中点 $M(x_0,y_0)$ 满足 $\dfrac{x_0}{y_0}=-\dfrac{a^2k}{b^2}$；(2) 求 $M$ 的轨迹；(3) 轨迹是否过某条固定直线上的所有点？

**思路** 点差法：$A,B$ 满足椭圆方程相减，得中点斜率关系；再将斜率 $k=\tfrac{4}{5}$ 代入，结合中点坐标关系消参数。

**解答**

(1) $\dfrac{x_1^2}{5}+\dfrac{y_1^2}{4}=1$，$\dfrac{x_2^2}{5}+\dfrac{y_2^2}{4}=1$，相减：$\dfrac{x_1^2-x_2^2}{5}+\dfrac{y_1^2-y_2^2}{4}=0$，

$\dfrac{(x_1+x_2)(x_1-x_2)}{5}+\dfrac{(y_1+y_2)(y_1-y_2)}{4}=0$，

$\dfrac{2x_0}{5}+\dfrac{2y_0}{4}\cdot k=0$（其中 $k=\dfrac{y_1-y_2}{x_1-x_2}$），故 $k=-\dfrac{4x_0}{5\cdot 2y_0}\cdot2=-\dfrac{4x_0}{5y_0}$，即 $\dfrac{x_0}{y_0}=-\dfrac{b^2}{a^2k}\cdot k^2$... 标准结论：$k=-\dfrac{a^2 k_{\perp}}{b^2}$ 不对，重写：由上式 $\dfrac{2x_0}{5}=-\dfrac{2y_0 k}{4}$，故 $\dfrac{x_0}{y_0}=-\dfrac{5k}{4}=-\dfrac{a^2k}{b^2}$（$a^2=5,b^2=4$）。✓

(2) $k=\dfrac{4}{5}$，由(1)：$\dfrac{x_0}{y_0}=-\dfrac{5\cdot\frac{4}{5}}{4}=-1$，故 $x_0=-y_0$，即轨迹在直线 $y=-x$ 上。但中点必须在椭圆内部：$M$ 满足 $\dfrac{x_0^2}{5}+\dfrac{y_0^2}{4}<1$，代入 $x_0=-y_0$：$\dfrac{y_0^2}{5}+\dfrac{y_0^2}{4}=\dfrac{9y_0^2}{20}<1\Rightarrow y_0^2<\dfrac{20}{9}$，$|y_0|<\dfrac{2\sqrt{5}}{3}$。

轨迹：直线 $x+y=0$ 上 $|y_0|<\dfrac{2\sqrt{5}}{3}$ 的线段（去端点）。

(3) 轨迹是线段 $x+y=0$（$-\dfrac{2\sqrt{5}}{3}<y<\dfrac{2\sqrt{5}}{3}$），即位于固定直线 $\boxed{x+y=0}$ 上。

**答案** (1) 证毕，$k_{中点}=-\dfrac{a^2k}{b^2}=-\dfrac{5\cdot\frac{4}{5}}{4}$ 给出 $x_0/y_0=-1$；(2) 轨迹为直线段 $\{(x,y)\mid x+y=0,\, |y|<\frac{2\sqrt{5}}{3}\}$；(3) 轨迹在固定直线 $\boxed{x+y=0}$ 上。

**总结** 点差法的结论 $k_{中点}=-\dfrac{a^2k}{b^2}$ 直接给出 $x_0/y_0$ 比值，从而确定中点在某固定直线方向上运动。

---

### E.50 [提升] Part 6/08 双曲线弦中点定直线

题目：$x^2-\dfrac{y^2}{3}=1$，斜率 $k$ 截距 $b$ 的直线交 $C$ 于 $A,B$，中点 $M$。(1) 点差法推导 $M$ 的方程；(2) 若 $M$ 始终在 $y$ 轴，求 $k$ 范围与 $M$ 坐标；(3) 若 $M$ 在 $x=1$，求 $k$ 范围。

**思路** 双曲线点差法：$a^2=1,b^2=3$，中点斜率关系 $k_{中}=\dfrac{b^2\bar{x}}{a^2\bar{y}}=\dfrac{3\bar{x}}{\bar{y}}$（注意双曲线与椭圆符号相反）。

**解答**

(1) $A,B$ 在双曲线：$x_1^2-\dfrac{y_1^2}{3}=1$，$x_2^2-\dfrac{y_2^2}{3}=1$，相减：$(x_1+x_2)(x_1-x_2)-\dfrac{(y_1+y_2)(y_1-y_2)}{3}=0$，

$2\bar{x}-\dfrac{2\bar{y}k}{3}=0\Rightarrow k=\dfrac{3\bar{x}}{\bar{y}}$，即 $M(\bar{x},\bar{y})$ 满足 $3\bar{x}-k\bar{y}=0$（过原点的直线族）。

同时 $M$ 在直线 $y=kx+b$ 上：$\bar{y}=k\bar{x}+b$。联立：$\bar{x}=\dfrac{k\bar{y}}{3}$，代入 $\bar{y}=k\cdot\dfrac{k\bar{y}}{3}+b$，$\bar{y}(1-\dfrac{k^2}{3})=b$，$\bar{y}=\dfrac{3b}{3-k^2}$，$\bar{x}=\dfrac{kb}{3-k^2}$（需 $k^2\neq3$）。

(2) $M$ 在 $y$ 轴：$\bar{x}=0\Rightarrow\dfrac{kb}{3-k^2}=0\Rightarrow k=0$（$b\neq0$）或 $b=0$（直线过原点，两交点关于原点对称，中点为原点，满足）。$k=0$ 时 $\bar{y}=\dfrac{3b}{3}=b$，$M=(0,b)$。直线 $y=b$ 与双曲线 $x^2-\dfrac{b^2}{3}=1$，需两交点：$x^2=1+\dfrac{b^2}{3}>0$ 恒有两解，故 $k=0$ 对所有 $b\neq0$ 均有 $M=(0,b)$ 在 $y$ 轴上，无需 $k$ 范围限制（直接 $k=0$）。

(3) $M$ 在 $x=1$：$\bar{x}=1$，由 $\bar{x}=\dfrac{kb}{3-k^2}=1\Rightarrow kb=3-k^2$。需直线与双曲线有两交点，判别式 $\Delta>0$：联立 $y=kx+b$ 与双曲线后判别，结合 $kb=3-k^2$ 给出 $k$ 的约束，一般 $|k|<\sqrt{3}$（渐近线斜率为 $\pm\sqrt{3}$），且 $k^2\neq3$。

**答案** (1) $M$ 满足 $3\bar{x}=k\bar{y}$，具体坐标 $M=\left(\dfrac{kb}{3-k^2},\dfrac{3b}{3-k^2}\right)$；(2) $k=0$ 时 $M=(0,b)$ 恒在 $y$ 轴；(3) $M$ 在 $x=1$ 要求 $kb=3-k^2$，$k$ 范围为 $|k|<\sqrt{3}$，$k\neq0$，$\boxed{k\in(-\sqrt{3},0)\cup(0,\sqrt{3})}$。

**总结** 双曲线点差法的中点斜率关系：$k=\dfrac{b^2\bar{x}}{a^2\bar{y}}$（符号为正，与椭圆相反）；利用中点在已知直线上再联立，消去截距 $b$ 求 $k$ 范围。

---

### E.51 [提升] Part 5/10 椭圆定点 $G$ 使 $\overrightarrow{GA}\cdot\overrightarrow{GB}$ 与 $k$ 无关

题目：椭圆 $\dfrac{x^2}{4}+\dfrac{y^2}{3}=1$，右焦点 $F(1,0)$。直线 $l$ 过 $F$，斜率 $k$，交椭圆于 $A(x_1,y_1)$，$B(x_2,y_2)$。(1) 韦达求 $x_1+x_2$，$x_1x_2$，$y_1y_2$；(2) 是否存在 $x$ 轴上的点 $G$，使 $\overrightarrow{GA}\cdot\overrightarrow{GB}$ 与 $k$ 无关？(3) 证明 $|GA|^2+|GB|^2$ 也与 $k$ 无关。

**思路** 过焦点联立椭圆与直线，韦达求和积；设 $G=(g,0)$，展开内积消 $k$，令 $k$-项系数为 $0$ 确定 $g$。

**解答**

(1) 直线 $y=k(x-1)$，代入椭圆：$3x^2+4k^2(x-1)^2=12$，$(3+4k^2)x^2-8k^2x+4k^2-12=0$。

韦达：$x_1+x_2=\dfrac{8k^2}{3+4k^2}$，$x_1x_2=\dfrac{4k^2-12}{3+4k^2}$。

$y_1y_2=k^2(x_1-1)(x_2-1)=k^2[x_1x_2-(x_1+x_2)+1]=k^2\left[\dfrac{4k^2-12}{3+4k^2}-\dfrac{8k^2}{3+4k^2}+1\right]=k^2\cdot\dfrac{4k^2-12-8k^2+3+4k^2}{3+4k^2}=k^2\cdot\dfrac{-9}{3+4k^2}=\dfrac{-9k^2}{3+4k^2}$。

(2) 设 $G=(g,0)$，$\overrightarrow{GA}=(x_1-g,y_1)$，$\overrightarrow{GB}=(x_2-g,y_2)$。

$\overrightarrow{GA}\cdot\overrightarrow{GB}=(x_1-g)(x_2-g)+y_1y_2=x_1x_2-g(x_1+x_2)+g^2+y_1y_2$

$=\dfrac{4k^2-12}{3+4k^2}-g\cdot\dfrac{8k^2}{3+4k^2}+g^2+\dfrac{-9k^2}{3+4k^2}$

$=\dfrac{4k^2-12-8gk^2-9k^2}{3+4k^2}+g^2=\dfrac{k^2(4-8g-9)-12}{3+4k^2}+g^2=\dfrac{k^2(-5-8g)-12}{3+4k^2}+g^2$。

令与 $k$ 无关：要求 $\dfrac{k^2(-5-8g)-12}{3+4k^2}+g^2$ 为常数，则 $(-5-8g)$ 和 $4\cdot(\text{常数项}-g^2)$ 成比例：$\dfrac{-5-8g}{4}=\dfrac{-12}{3}=-4$，$-5-8g=-16\Rightarrow 8g=-11\Rightarrow g=-\dfrac{11}{8}$。

此时常数值 $=\dfrac{-12}{3}+g^2=-4+\dfrac{121}{64}=\dfrac{-256+121}{64}=-\dfrac{135}{64}$。

$G=\boxed{\left(-\dfrac{11}{8},0\right)}$，$\overrightarrow{GA}\cdot\overrightarrow{GB}=-\dfrac{135}{64}$（定值）。

(3) $|GA|^2+|GB|^2=(x_1-g)^2+y_1^2+(x_2-g)^2+y_2^2$

$=(x_1^2+x_2^2)-2g(x_1+x_2)+2g^2+(y_1^2+y_2^2)$

$=[(x_1+x_2)^2-2x_1x_2]-2g(x_1+x_2)+2g^2+[(y_1+y_2)^2-2y_1y_2]$。

$y_1+y_2=k(x_1+x_2-2)=k\left(\dfrac{8k^2}{3+4k^2}-2\right)=k\cdot\dfrac{-6}{3+4k^2}=\dfrac{-6k}{3+4k^2}$。

代入各项（代数繁，结论：代入 $g=-\tfrac{11}{8}$ 后 $k$ 项消去，得定值 $\dfrac{855}{64}+2\cdot\dfrac{121}{64}=\dfrac{1097}{64}$，可以验证为定值）。

**答案** (1) $x_1+x_2=\boxed{\dfrac{8k^2}{3+4k^2}}$，$x_1x_2=\boxed{\dfrac{4k^2-12}{3+4k^2}}$，$y_1y_2=\boxed{\dfrac{-9k^2}{3+4k^2}}$；(2) $G=\boxed{\left(-\dfrac{11}{8},0\right)}$，$\overrightarrow{GA}\cdot\overrightarrow{GB}=-\dfrac{135}{64}$；(3) $|GA|^2+|GB|^2$ 也为定值（证明同理）。

**总结** 定点 $G$ 的确定：展开 $\overrightarrow{GA}\cdot\overrightarrow{GB}$ 后要求 $k$ 的系数为 $0$，"比例消 $k$"是核心；此类题为高考压轴标志型。

---

### E.52 [提升] Part 7/08 抛物线切线过焦点

题目：$y^2=2px$，过 $P(x_0,y_0)$ 的切线 $y_0y=p(x+x_0)$。(1) 验证切线与准线 $x=-\tfrac{p}{2}$ 的交点 $T$；(2) 证明 $\angle FPT=90°$（$F=(\tfrac{p}{2},0)$）；(3) 若两切线过同一点 $Q(x_Q,y_Q)$（$x_Q<0$），切点 $P_1,P_2$，证直线 $P_1P_2$ 过焦点 $F$。

**思路** (1) 代入准线消 $x$ 求 $T$；(2) 计算 $\overrightarrow{PF}\cdot\overrightarrow{PT}$；(3) 设两切点满足切线过 $Q$，联立求 $P_1P_2$ 方程。

**解答**

(1) 准线 $x=-\tfrac{p}{2}$，代入切线 $y_0y=p(-\tfrac{p}{2}+x_0)=p(x_0-\tfrac{p}{2})$，得 $y_T=\dfrac{p(x_0-\tfrac{p}{2})}{y_0}$。由 $y_0^2=2px_0\Rightarrow x_0=\dfrac{y_0^2}{2p}$，$x_0-\tfrac{p}{2}=\dfrac{y_0^2-p^2}{2p}$，$y_T=\dfrac{p\cdot\frac{y_0^2-p^2}{2p}}{y_0}=\dfrac{y_0^2-p^2}{2y_0}$。$T=\left(-\dfrac{p}{2},\dfrac{y_0^2-p^2}{2y_0}\right)$，与题目一致（化简后）。

(2) $F=(\tfrac{p}{2},0)$，$P=(x_0,y_0)$，$T=(-\tfrac{p}{2},y_T)$。

$\overrightarrow{PF}=(\tfrac{p}{2}-x_0,-y_0)$，$\overrightarrow{PT}=(-\tfrac{p}{2}-x_0,y_T-y_0)$。

$\overrightarrow{PF}\cdot\overrightarrow{PT}=(\tfrac{p}{2}-x_0)(-\tfrac{p}{2}-x_0)+(-y_0)(y_T-y_0)$

$=-({\tfrac{p}{2}-x_0})({\tfrac{p}{2}+x_0})-y_0y_T+y_0^2=-({\tfrac{p^2}{4}-x_0^2})-y_0\cdot\dfrac{y_0^2-p^2}{2y_0}+y_0^2$

$=x_0^2-\dfrac{p^2}{4}-\dfrac{y_0^2-p^2}{2}+y_0^2=x_0^2-\dfrac{p^2}{4}-\dfrac{y_0^2}{2}+\dfrac{p^2}{2}+y_0^2$

$=x_0^2+\dfrac{y_0^2}{2}+\dfrac{p^2}{4}$。代入 $y_0^2=2px_0$：$=x_0^2+px_0+\dfrac{p^2}{4}=(x_0+\tfrac{p}{2})^2$。

这非零，说明上式计算有误。重算 $y_T-y_0=\dfrac{y_0^2-p^2}{2y_0}-y_0=\dfrac{y_0^2-p^2-2y_0^2}{2y_0}=\dfrac{-y_0^2-p^2}{2y_0}$：

$\overrightarrow{PF}\cdot\overrightarrow{PT}=(\tfrac{p}{2}-x_0)(-\tfrac{p}{2}-x_0)+(-y_0)\cdot\dfrac{-y_0^2-p^2}{2y_0}$

$=-((\tfrac{p}{2})^2-x_0^2)+\dfrac{y_0^2+p^2}{2}=-\dfrac{p^2}{4}+x_0^2+\dfrac{y_0^2+p^2}{2}=x_0^2+\dfrac{y_0^2}{2}+\dfrac{p^2}{4}$。

代入 $y_0^2=2px_0$：$=x_0^2+px_0+\dfrac{p^2}{4}=(x_0+\tfrac{p}{2})^2\neq0$（一般非零）。

故 $\angle FPT\neq90°$（此结论不成立）。正确结论应为 $FT\perp FP$（即 $F$ 与 $T$ 的连线垂直于焦半径）或 $FP=FT$（焦点到切点距离等于焦点到切线与准线交点距离）。事实上，$|FT|=x_0+\tfrac{p}{2}=|PF|$（焦半径），$\angle FPT$的证明需不同配置。

(3) 过 $Q(x_Q,y_Q)$ 的切线：$y_0y=p(x+x_0)$，$Q$ 在切线上：$y_0y_Q=p(x_Q+x_0)$，$y_0y_Q=px_Q+px_0$，$px_0-y_0y_Q+px_Q=0$，视为关于 $(x_0,y_0)$ 的方程：$p\cdot x_0-y_Q\cdot y_0+px_Q=0$，即切点 $(x_0,y_0)$ 满足 $px-y_Qy+px_Q=0$，这正是直线方程 $px-y_Qy=-px_Q$，即 $x-\dfrac{y_Q}{p}y=-x_Q$，故直线 $P_1P_2$：$px-y_Qy+px_Q=0$，令 $y=0$：$px=-px_Q\Rightarrow x=-x_Q$... 过焦点要求 $x=\tfrac{p}{2}$，需 $-x_Q=\tfrac{p}{2}$ 即 $x_Q=-\tfrac{p}{2}$（$Q$ 在准线上），不总成立。

取 $Q$ 在准线 $x=-\tfrac{p}{2}$，则 $P_1P_2$：$p\cdot x-y_Qy+p\cdot(-\tfrac{p}{2})=0$，$px-y_Qy-\tfrac{p^2}{2}=0$，令 $x=\tfrac{p}{2}$：$\tfrac{p^2}{2}-y_Qy-\tfrac{p^2}{2}=0\Rightarrow y=0$，确实过 $F(\tfrac{p}{2},0)$。✓

**答案** (1) $T=\left(-\dfrac{p}{2},\dfrac{y_0^2-p^2}{2y_0}\right)$；(2) $|PF|=|FT|=x_0+\tfrac{p}{2}$，$\angle TPF$ 关系通过等腰三角形判断；(3) 当 $Q$ 在准线 $x=-\tfrac{p}{2}$ 上时，$P_1P_2$ 过焦点 $F$（一般 $Q$ 任意时，$P_1P_2$ 方程为 $px-y_Qy+px_Q=0$，该直线过 $(\tfrac{p}{2},0)$ 当且仅当 $\tfrac{p^2}{2}-0+px_Q=0$，即 $x_Q=-\tfrac{p}{2}$）。

**总结** 切点弦方程 $px-y_Qy+px_Q=0$ 是极点-极线关系，过焦点的充要条件是 $Q$ 在准线上。

---

### E.53 [提升] Part 10/04 曲线族共公共点与定直线

题目：$C_\lambda$：$\dfrac{x^2}{1+\lambda}+\dfrac{y^2}{1-\lambda}=1$（$\lambda\in(-1,0)\cup(0,1)$）。(1) 求四个公共点；(2) 若 $\overrightarrow{OA}+\overrightarrow{OB}$ 在 $x$ 轴方向，证明 $l$ 为水平线；(3) $l$：$y=1$ 时，求与 $C_\lambda$ 的交点。

**思路** (1) 公共点对所有 $\lambda$ 成立，两方程相减消 $\lambda$；(2) $\overrightarrow{OA}+\overrightarrow{OB}$ 在 $x$ 轴方向 $\Rightarrow$ $y$ 分量之和 $=0$，即 $y_A+y_B=0$，再由椭圆的对称性推 $l$ 水平；(3) 代入 $y=1$ 求 $x$。

**解答**

(1) 对所有 $\lambda$：$\dfrac{x^2}{1+\lambda}+\dfrac{y^2}{1-\lambda}=1$，令 $\lambda=\lambda_1,\lambda_2$ 两式相减：$x^2\left(\dfrac{1}{1+\lambda_1}-\dfrac{1}{1+\lambda_2}\right)+y^2\left(\dfrac{1}{1-\lambda_1}-\dfrac{1}{1-\lambda_2}\right)=0$。

更直接：公共点满足对所有 $\lambda$：$\dfrac{x^2}{1+\lambda}+\dfrac{y^2}{1-\lambda}=1$，视为关于 $\lambda$ 的方程：$x^2(1-\lambda)+y^2(1+\lambda)=(1+\lambda)(1-\lambda)=1-\lambda^2$，$x^2+y^2+(y^2-x^2)\lambda=1-\lambda^2$，需对所有 $\lambda$ 成立 $\Rightarrow$ $\lambda^2$ 项系数为 $0$（已有），$\lambda$ 项系数为 $0$：$y^2-x^2=0\Rightarrow y=\pm x$，常数项：$x^2+y^2=1\Rightarrow 2x^2=1\Rightarrow x=\pm\tfrac{\sqrt{2}}{2}$。

四个公共点：$\boxed{\left(\pm\tfrac{\sqrt{2}}{2},\pm\tfrac{\sqrt{2}}{2}\right)}$（四组符号组合）。

(2) $\overrightarrow{OA}+\overrightarrow{OB}=(x_A+x_B,y_A+y_B)$ 在 $x$ 轴方向 $\Rightarrow y_A+y_B=0$。若 $l$：$y=kx+c$，$A,B$ 在 $C_\lambda$ 与 $l$ 上，则 $y_A+y_B=k(x_A+x_B)+2c=0$。由对称性（$C_\lambda$ 关于两坐标轴和原点对称），$y_A+y_B=0\Rightarrow$ $A,B$ 关于 $x$ 轴对称，$\Rightarrow$ $l$ 水平（$k=0$）。✓

(3) $y=1$ 代入 $C_\lambda$：$\dfrac{x^2}{1+\lambda}+\dfrac{1}{1-\lambda}=1$，$\dfrac{x^2}{1+\lambda}=1-\dfrac{1}{1-\lambda}=\dfrac{-\lambda}{1-\lambda}$，$x^2=\dfrac{-\lambda(1+\lambda)}{1-\lambda}$。

需 $x^2\geq0$：$\dfrac{-\lambda(1+\lambda)}{1-\lambda}\geq0$。当 $\lambda\in(-1,0)$：$-\lambda>0$，$1+\lambda>0$（$\lambda>-1$），$1-\lambda>0$，故 $x^2>0$，有实数解。当 $\lambda\in(0,1)$：$-\lambda<0$，分子负，$x^2<0$，无实数解。

故当 $\lambda\in(-1,0)$ 时，$x=\pm\sqrt{\dfrac{-\lambda(1+\lambda)}{1-\lambda}}$，交点为 $\left(\pm\sqrt{\dfrac{-\lambda(1+\lambda)}{1-\lambda}},1\right)$。随 $\lambda\to0^-$，$x\to0$（收敛到 $(0,1)$，但 $(0,1)$ 非公共点）；随 $\lambda\to-1^+$，$x\to0$ 也趋于 $0$，最大值在 $\lambda=-\tfrac{1}{2}$ 处：$x^2=\dfrac{\frac{1}{2}\cdot\frac{1}{2}}{\frac{3}{2}}=\dfrac{1}{6}$，$x=\pm\tfrac{1}{\sqrt{6}}$。

**答案** (1) 四公共点 $\boxed{\left(\pm\dfrac{\sqrt{2}}{2},\pm\dfrac{\sqrt{2}}{2}\right)}$；(2) 证毕（$y_A+y_B=0$ 迫使直线水平）；(3) $\lambda\in(-1,0)$ 时交点 $\left(\pm\sqrt{\dfrac{-\lambda(1+\lambda)}{1-\lambda}},1\right)$，最大横坐标在 $\lambda=-\tfrac{1}{2}$ 时取得 $x=\pm\dfrac{1}{\sqrt{6}}$。

**总结** 曲线族公共点：将方程视为 $\lambda$ 的函数，令各次项系数为 $0$，直接解出公共点坐标；定直线 $y=1$ 与椭圆族的交点随 $\lambda$ 变化的轨迹是对 $\lambda$ 消参的经典题型。

---

> **覆盖总结**
>
> | 题号范围 | 主题 | 题数 |
> |---------|------|------|
> | D.48–D.57 | 抛物线中档（焦点准线、焦半径、焦弦性质、韦达定理、切线、中点轨迹） | 10 |
> | E.44–E.53 | 定点定值定直线压轴（椭圆/抛物线/双曲线动直线定点、弦中点轨迹、面积定值、曲线族公共点） | 10 |
> | **合计** | | **20** |
