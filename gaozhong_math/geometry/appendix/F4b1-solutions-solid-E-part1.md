# 附录 F4b1：立体几何压轴（E.26-E.37）详解

> 覆盖：**E.26–E.37**（立体几何 + 空间向量综合，共 **12 题**）。
>
> 每题格式：题目回顾 / 思路 / 解答 / 答案 / 总结。
>
> toolkit 引用：
> - → toolkit/11 立体向量：建系 → 坐标 → 法向量 → 三类角 / 三类距离；
> - → toolkit/09 二面角三路：综合法 / 向量法 / 定义法；
> - → toolkit/01 四把剑：数形结合、结构识别、换元、构造。

---

## E.26 [提升] Part 9/05

**题目回顾**：正三棱锥 $P\text{-}ABC$，底面边长 $2$，侧棱长 $\sqrt{6}$，$M$ 是 $AB$ 中点。(1) 建立坐标系；(2) 求二面角 $P\text{-}AB\text{-}C$ 的余弦值；(3) 求 $P$ 到底面的距离并用向量法验证。

**思路**（→ toolkit/11 建系 → 法向量）：以 $M$ 为原点，$MC$ 方向为 $y$ 轴，建立空间直角坐标系。二面角由两平面各求法向量，用数量积公式。

**解答**：

**第一步：建坐标系。** 底面等边三角形边长 $2$，$M$ 为 $AB$ 中点，$MC \perp AB$，$MC = \sqrt{3}$。以 $M$ 为原点，$\overrightarrow{MC}$ 为 $y$ 轴正方向，$\overrightarrow{MA}$ 为 $x$ 轴正方向（$MA=1$），$z$ 轴向上。

各顶点坐标：$M=(0,0,0)$，$A=(-1,0,0)$，$B=(1,0,0)$，$C=(0,\sqrt{3},0)$，

$P$ 在底面的射影为重心 $G=\left(0,\frac{\sqrt{3}}{3},0\right)$，$|PG|=h$，$|PA|=\sqrt{6}$，

$$h=\sqrt{|PA|^2-|GA|^2}=\sqrt{6-\left(\frac{2\sqrt{3}}{3}\right)^2}=\sqrt{6-\frac{4}{3}}=\sqrt{\frac{14}{3}}$$

故 $P=\left(0,\frac{\sqrt{3}}{3},\sqrt{\frac{14}{3}}\right)$。

**第二步：求二面角 $P\text{-}AB\text{-}C$。** 二面角的棱为 $AB$（即 $x$ 轴），需求平面 $PAB$ 与平面 $CAB$ 各自的法向量。

$\overrightarrow{AB}=(2,0,0)$，$\overrightarrow{AM}=(1,0,0)$（方向相同），$\overrightarrow{AC}=(1,\sqrt{3},0)$，$\overrightarrow{AP}=(1,\frac{\sqrt{3}}{3},\sqrt{\frac{14}{3}})$。

平面 $CAB$ 的法向量 $\boldsymbol{n_1}$：垂直于 $\overrightarrow{AB}=(2,0,0)$ 和 $\overrightarrow{AC}=(1,\sqrt{3},0)$，取 $\boldsymbol{n_1}=(0,0,1)$（底面法向量）。

平面 $PAB$ 的法向量 $\boldsymbol{n_2}$：垂直于 $\overrightarrow{AB}=(2,0,0)$ 和 $\overrightarrow{AP}$。由 $\boldsymbol{n_2} \perp \overrightarrow{AB}$，令 $\boldsymbol{n_2}=(0,b,c)$；由 $\boldsymbol{n_2} \perp \overrightarrow{AP}$：$b\cdot\frac{\sqrt{3}}{3}+c\cdot\sqrt{\frac{14}{3}}=0$，取 $b=\sqrt{14},\ c=-\sqrt{3}$，故 $\boldsymbol{n_2}=(0,\sqrt{14},-\sqrt{3})$。

$$\cos\theta = \frac{\boldsymbol{n_1}\cdot\boldsymbol{n_2}}{|\boldsymbol{n_1}||\boldsymbol{n_2}|} = \frac{0+0-\sqrt{3}}{\sqrt{14+3}} = \frac{-\sqrt{3}}{\sqrt{17}}$$

二面角为钝角，余弦值为 $-\dfrac{\sqrt{3}}{\sqrt{17}} = -\dfrac{\sqrt{51}}{17}$。

**第三步：$P$ 到底面的距离。** 已在第一步算出 $h = \sqrt{\frac{14}{3}} = \frac{\sqrt{42}}{3}$。向量法验证：底面法向量 $\boldsymbol{n_1}=(0,0,1)$，$\overrightarrow{GP}=\left(0,0,\frac{\sqrt{42}}{3}\right)$，投影 $= \frac{\sqrt{42}}{3}$。✓

**答案**：$P$ 到底面距离 $\boxed{\dfrac{\sqrt{42}}{3}}$；二面角余弦值 $\boxed{-\dfrac{\sqrt{51}}{17}}$。

**总结**：正三棱锥建系选 $AB$ 中点为原点，重心为射影。二面角法向量必须分别对两个平面内的两个方向向量作叉积（或待定系数）。

---

## E.27 [提升] Part 9/05

**题目回顾**：四棱柱 $ABCD\text{-}A_1B_1C_1D_1$，底面菱形 $AB=2$，$\angle DAB=60°$，$AA_1 \perp$ 底面，$AA_1=\sqrt{3}$。(1) 建坐标系；(2) 求二面角 $A\text{-}BD_1\text{-}C$ 余弦值；(3) 过 $A_1$ 作平面 $BDA_1$ 的垂线，求与底面夹角的余弦值。

**思路**（→ toolkit/11 直棱柱建系）：$AA_1 \perp$ 底面，以 $A$ 为原点建系。菱形 $\angle DAB=60°$，边长 $2$，可算各顶点坐标。

**解答**：

**第一步：建坐标系。** 以 $A$ 为原点，$\overrightarrow{AB}$ 为 $x$ 轴，$z$ 轴沿 $AA_1$。$B=(2,0,0)$，$D=(2\cos 60°, 2\sin 60°,0)=(1,\sqrt{3},0)$，$C=B+D=(3,\sqrt{3},0)$（菱形 $\overrightarrow{BC}=\overrightarrow{AD}$），$A_1=(0,0,\sqrt{3})$，$D_1=(1,\sqrt{3},\sqrt{3})$。

**第二步：二面角 $A\text{-}BD_1\text{-}C$。** 棱为 $BD_1$，$\overrightarrow{BD_1}=D_1-B=(-1,\sqrt{3},\sqrt{3})$。

在平面 $ABD_1$ 内取 $\overrightarrow{BA}=(-2,0,0)$；在平面 $CBD_1$ 内取 $\overrightarrow{BC}=(1,\sqrt{3},0)$。

法向量 $\boldsymbol{n_1}=\overrightarrow{BD_1}\times\overrightarrow{BA}$，法向量 $\boldsymbol{n_2}=\overrightarrow{BD_1}\times\overrightarrow{BC}$（叉积）：

$$\boldsymbol{n_1}=\overrightarrow{BD_1}\times\overrightarrow{BA}=\begin{vmatrix}\boldsymbol{i}&\boldsymbol{j}&\boldsymbol{k}\\-1&\sqrt{3}&\sqrt{3}\\-2&0&0\end{vmatrix}=(0\cdot\sqrt{3}-0\cdot\sqrt{3},\ \sqrt{3}\cdot(-2)-(-1)\cdot 0,\ (-1)\cdot 0-\sqrt{3}\cdot(-2))=(0,-2\sqrt{3},2\sqrt{3})$$

$$\boldsymbol{n_2}=\overrightarrow{BD_1}\times\overrightarrow{BC}=\begin{vmatrix}\boldsymbol{i}&\boldsymbol{j}&\boldsymbol{k}\\-1&\sqrt{3}&\sqrt{3}\\1&\sqrt{3}&0\end{vmatrix}=(\sqrt{3}\cdot 0-\sqrt{3}\cdot\sqrt{3},\ \sqrt{3}\cdot 1-(-1)\cdot 0,\ -\sqrt{3}-\sqrt{3})=(-3,\sqrt{3},-2\sqrt{3})$$

$$\cos\theta=\frac{\boldsymbol{n_1}\cdot\boldsymbol{n_2}}{|\boldsymbol{n_1}||\boldsymbol{n_2}|}=\frac{0+(-2\sqrt{3})(\sqrt{3})+2\sqrt{3}\cdot(-2\sqrt{3})}{\sqrt{0+12+12}\cdot\sqrt{9+3+12}}=\frac{-6-12}{\sqrt{24}\cdot\sqrt{24}}=\frac{-18}{24}=-\frac{3}{4}$$

**第三步：$A_1$ 到面 $BDA_1$ 的垂线与底面夹角。** $A_1$ 在面 $BDA_1$ 内，其所在平面 $BDA_1$ 的法向量已含 $z$ 分量，垂线方向即为该法向量方向。面 $BDA_1$ 的法向量用 $\overrightarrow{A_1B}=(2,0,-\sqrt{3})$，$\overrightarrow{A_1D}=(1,\sqrt{3},-\sqrt{3})$ 叉积：$\boldsymbol{m}=\overrightarrow{A_1B}\times\overrightarrow{A_1D}=(\sqrt{3}\cdot(-\sqrt{3})-(-\sqrt{3})\sqrt{3},\,-(-\sqrt{3})\cdot 1-2(-\sqrt{3}),\,2\sqrt{3}-0)=( 0,3\sqrt{3},2\sqrt{3})$。

垂线与底面夹角 $\varphi$（底面法向量 $=(0,0,1)$）：$\sin\varphi=\dfrac{|\boldsymbol{m}\cdot(0,0,1)|}{|\boldsymbol{m}|}=\dfrac{2\sqrt{3}}{\sqrt{27+12}}=\dfrac{2\sqrt{3}}{\sqrt{39}}$；$\cos\varphi=\sqrt{1-\sin^2\varphi}=\sqrt{1-\frac{12}{39}}=\sqrt{\frac{27}{39}}=\sqrt{\frac{9}{13}}=\dfrac{3}{\sqrt{13}}$。

**答案**：二面角 $A\text{-}BD_1\text{-}C$ 余弦值 $\boxed{-\dfrac{3}{4}}$；垂线与底面夹角余弦值 $\boxed{\dfrac{3}{\sqrt{13}}}$。

**总结**：直四棱柱以 $A$ 为原点，建系写坐标。叉积求法向量是最系统的方式；注意法向量方向影响二面角符号。

---

## E.28 [提升] Part 9/06

**题目回顾**：四棱锥 $P\text{-}ABCD$，底面正方形边长 $2$，$PA \perp$ 底面，$PA=2$。(1) 综合法求二面角 $P\text{-}BC\text{-}D$ 正切值；(2) 向量法验证；(3) 棱 $PB$ 上是否存在点 $E$ 使 $DE \perp$ 平面 $PAB$？

**思路**（→ toolkit/09 综合法 + 向量法双路）：综合法：在 $BC$ 上找垂直点，作二面角的平面角；向量法：建坐标求法向量。存在性用向量正交条件。

**解答**：

**第一步（综合法）：** $A=(0,0,0),B=(2,0,0),C=(2,2,0),D=(0,2,0),P=(0,0,2)$。在 $BC$ 上取点 $M=B$（因 $PA \perp$ 底面，$AB \perp BC$），过 $M$ 在平面 $PBC$ 内作 $MB \perp BC$（已满足），平面 $PBM$（即 $PAB$）内 $MB$ 方向即 $\overrightarrow{BA}=(-2,0,0)$；过 $M$ 在底面内作 $MC \perp BC$（即 $\overrightarrow{BC}=(0,2,0)$）时另取 $B$ 到 $DC$ 方向：作 $\overrightarrow{BD}=(-2,2,0)$，$\overrightarrow{BD}$ 在底面中垂直于 $BC$（因 $BC \perp AB$ 但 $BD$ 不垂直）。

改用向量法更简洁：

**第二步（向量法）：** 平面 $PBC$ 法向量：$\overrightarrow{BP}=(-2,0,2),\overrightarrow{BC}=(0,2,0)$，叉积 $\boldsymbol{n_1}=(0\cdot 2-2\cdot 2,-( -2\cdot 2-(-2)\cdot 0),(-2)\cdot 2-0)=(-4,4,-4)$，化简 $\boldsymbol{n_1}=(-1,1,-1)$。

平面 $BCD$ 法向量：$\overrightarrow{BC}=(0,2,0),\overrightarrow{BD}=(-2,2,0)$，叉积 $\boldsymbol{n_2}=(0\cdot 0-0\cdot 2,-(0\cdot(-2)-0\cdot(-2)),0\cdot 2-2\cdot(-2))=(0,0,4)$，化简 $\boldsymbol{n_2}=(0,0,1)$。

$$\cos\theta=\frac{\boldsymbol{n_1}\cdot\boldsymbol{n_2}}{|\boldsymbol{n_1}||\boldsymbol{n_2}|}=\frac{-1}{\sqrt{3}\cdot 1}=-\frac{1}{\sqrt{3}},\quad \tan\theta = \sqrt{2}$$

二面角 $P\text{-}BC\text{-}D$ 的正切值为 $\sqrt{2}$（二面角为钝角，取补）。实际正切值 $= \tan(\pi-\theta)$，应对内部夹角分析：底面在 $BC$ 下侧，$P$ 在上，内侧二面角为锐角侧，$\tan = \sqrt{2}$。

**第三步（存在性）：** 设 $E$ 在 $PB$ 上，$E=P+t\overrightarrow{PB}=(0,0,2)+t(2,0,-2)=(2t,0,2-2t)$，$t\in(0,1)$。平面 $PAB$ 包含 $\overrightarrow{PA}=(0,0,-2)$ 和 $\overrightarrow{AB}=(2,0,0)$，法向量 $\boldsymbol{n}_{PAB}=(0,1,0)$（$xOz$ 平面法向量）。

$DE \perp$ 平面 $PAB$ 要求 $\overrightarrow{DE} \parallel \boldsymbol{n}_{PAB}=(0,1,0)$，即 $\overrightarrow{DE}$ 只有 $y$ 分量。$\overrightarrow{DE}=E-D=(2t-0,0-2,2-2t)=(2t,-2,2-2t)$，须 $2t=0$ 且 $2-2t=0$，矛盾，**不存在**。

**答案**：二面角正切值 $\boxed{\sqrt{2}}$；棱 $PB$ 上**不存在**这样的点 $E$。

**总结**：综合法对简单情形快捷；存在性用参数化设点后代入正交条件，若方程无解则不存在。

---

## E.29 [提升] Part 9/07

**题目回顾**：矩形 $ABCD$（$AB=2,BC=\sqrt{3}$）沿对角线 $BD$ 折叠，使 $\triangle ABD$ 与 $\triangle CBD$ 所在平面的二面角 $A\text{-}BD\text{-}C=90°$。(1) 建坐标系；(2) $A$ 到平面 $BCD$ 的距离；(3) 直线 $AC$ 与平面 $BCD$ 所成角的正弦值。

**思路**（→ toolkit/11 折叠建系）：$\triangle CBD$ 放底面，$\triangle ABD$ 折起后二面角 $=90°$，设折叠后 $A$ 的坐标，用二面角条件定 $A$。

**解答**：

**第一步：建坐标系。** 保持 $\triangle BCD$ 在底面。矩形 $ABCD$ 中 $AB=2,BC=\sqrt{3}$，对角线 $BD=\sqrt{AB^2+AD^2}=\sqrt{4+3}=\sqrt{7}$。$\triangle BCD$ 中 $BC=\sqrt{3},CD=2,BD=\sqrt{7}$。

以 $B$ 为原点，$\overrightarrow{BD}$ 为 $y$ 轴（$BD=\sqrt{7}$），$x$ 轴在底面内垂直 $BD$。

$D=(0,\sqrt{7},0)$；$C$ 在底面：$\triangle BCD$ 中 $BC=\sqrt{3},BD=\sqrt{7},CD=2$。$C$ 到 $BD$（$y$ 轴）的垂足距离 $= \frac{BC\cdot\sin(\angle CBD)}{1}$，用坐标：$C=(c_x,c_y,0)$，$|BC|=\sqrt{3},|CD|=2$，故 $c_x^2+c_y^2=3,\ (c_x)^2+(c_y-\sqrt{7})^2=4$，相减：$-2\sqrt{7}c_y+7=1$，$c_y=\dfrac{6}{2\sqrt{7}}=\dfrac{3}{\sqrt{7}}$，$c_x=\sqrt{3-\frac{9}{7}}=\sqrt{\frac{12}{7}}=\dfrac{2\sqrt{3}}{\sqrt{7}}$。取 $c_x>0$，$C=\left(\dfrac{2\sqrt{3}}{\sqrt{7}},\dfrac{3}{\sqrt{7}},0\right)$。

折叠后 $A$ 在另一侧。$\triangle ABD$：$AB=2,AD=\sqrt{3},BD=\sqrt{7}$（与 $\triangle BCD$ 全等）。$A=(a_x,a_y,a_z)$，由 $|AB|=2$：$a_x^2+a_y^2+a_z^2=4$；由 $|AD|=\sqrt{3}$：$a_x^2+(a_y-\sqrt{7})^2+a_z^2=3$，相减：$-2\sqrt{7}a_y+7=1$，$a_y=\dfrac{3}{\sqrt{7}}$（同 $C$ 的 $y$ 分量）。

二面角 $A\text{-}BD\text{-}C=90°$：棱 $BD$ 在 $y$ 轴，面 $ABD$ 中 $A$ 向 $BD$ 的垂线方向为 $(a_x,0,a_z)$（因 $a_y=\frac{3}{\sqrt{7}}$ 时垂分量），面 $CBD$ 中 $C$ 向 $BD$ 的垂线方向为 $\left(\frac{2\sqrt{3}}{\sqrt{7}},0,0\right)$（即 $x$ 轴方向）。两垂线垂直：$a_x\cdot\dfrac{2\sqrt{3}}{\sqrt{7}}+a_z\cdot 0=0$，故 $a_x=0$。

由 $a_x=0$：$0+\frac{9}{7}+a_z^2=4$，$a_z^2=4-\frac{9}{7}=\frac{19}{7}$，$a_z=\sqrt{\frac{19}{7}}$（取正）。

$A=\left(0,\dfrac{3}{\sqrt{7}},\sqrt{\dfrac{19}{7}}\right)$。

**第二步：$A$ 到面 $BCD$ 的距离。** 面 $BCD$ 为 $z=0$ 平面，距离 $= |a_z| = \sqrt{\dfrac{19}{7}} = \dfrac{\sqrt{133}}{7}$。

**第三步：直线 $AC$ 与面 $BCD$ 所成角。** $\overrightarrow{AC}=C-A=\left(\dfrac{2\sqrt{3}}{\sqrt{7}}-0,\ \dfrac{3}{\sqrt{7}}-\dfrac{3}{\sqrt{7}},\ 0-\sqrt{\dfrac{19}{7}}\right)=\left(\dfrac{2\sqrt{3}}{\sqrt{7}},0,-\sqrt{\dfrac{19}{7}}\right)$。

$$\sin\varphi = \frac{|\overrightarrow{AC}\cdot\boldsymbol{k}|}{|\overrightarrow{AC}|} = \frac{\sqrt{19/7}}{\sqrt{12/7+19/7}} = \frac{\sqrt{19/7}}{\sqrt{31/7}} = \sqrt{\frac{19}{31}} = \frac{\sqrt{19}}{\sqrt{31}}=\frac{\sqrt{589}}{31}$$

**答案**：$A$ 到面 $BCD$ 距离 $\boxed{\dfrac{\sqrt{133}}{7}}$；$AC$ 与底面所成角 $\sin\varphi = \boxed{\dfrac{\sqrt{589}}{31}}$。

**总结**：折叠题关键是确定折叠后点的坐标：距离条件列方程 + 二面角条件（两半面各取垂线，令内积为 $\cos$ 条件）。

---

## E.30 [提升] Part 8/06

**题目回顾**：三棱锥 $P\text{-}ABC$，底面等边三角形边长 $2$，$PA=PB=PC=t$（$t>\dfrac{2\sqrt{3}}{3}$）。(1) 用 $t$ 表示高 $h$；(2) 体积 $V(t)$；(3) 使 $V$ 最大的 $t$。

**思路**（→ toolkit/11 正三棱锥参数化）：$P$ 在底面重心正上方，$|PG|=h$，$|GA|=\frac{2\sqrt{3}}{3}$，勾股定理给出 $h(t)$。体积 $V=\frac{1}{3}Sh$，$S$ 固定，对 $h$ 或 $t$ 求极值。

**解答**：

**第一步：高 $h$。** 等边三角形边长 $2$，重心 $G$ 距顶点 $|GA|=\dfrac{2}{\sqrt{3}}=\dfrac{2\sqrt{3}}{3}$。

$$h = \sqrt{t^2 - \frac{4}{3}}$$

**第二步：体积。** $S = \dfrac{\sqrt{3}}{4}\cdot 2^2 = \sqrt{3}$。

$$V(t) = \frac{1}{3}\cdot\sqrt{3}\cdot\sqrt{t^2-\frac{4}{3}} = \frac{\sqrt{3}}{3}\sqrt{t^2-\frac{4}{3}}$$

**第三步：最大值。** $V(t)$ 关于 $t$ 单调递增（$t$ 越大高越大），故 $V$ **无有限最大值**（随 $t\to\infty$ 趋向无穷）。

若题目含附加约束"体积最大"需限定范围，则在给定约束（如侧面积等于底面积）下求极值：设 $S_{\text{侧}}=3\cdot\frac{1}{2}\cdot 2\cdot l_m=3l_m$（$l_m$ 为侧面斜高），令 $S_{\text{侧}}=S_{\text{底}}=\sqrt{3}$，$l_m=\frac{\sqrt{3}}{3}$。本题按标准解法，问题实为"$V^2$ 关于 $t^2$ 求导"型：

$$V^2 = \frac{1}{3}\cdot 3\cdot\left(t^2-\frac{4}{3}\right) = t^2-\frac{4}{3}$$

$V^2$ 仍单调增，说明 $V$ 单调递增，无最大值。若题意指"$V$ 在 $t=t_0$ 时对某具体约束最大"，请参照实际题目约束补充。

**答案**：高 $h = \boxed{\sqrt{t^2-\dfrac{4}{3}}}$；体积 $V(t) = \boxed{\dfrac{\sqrt{3}}{3}\sqrt{t^2-\dfrac{4}{3}}}$；$V$ 随 $t$ 单调递增，无有限最大值（在给定定义域 $t>\dfrac{2\sqrt{3}}{3}$ 上无最大值）。

**总结**：正三棱锥含参，高由勾股定理给出；体积为高的单调函数时无极值，若题有额外约束需补充。

---

## E.31 [提升] Part 8/07

**题目回顾**：正三棱柱 $ABC\text{-}A_1B_1C_1$，底面边长 $a$，高 $h$。过 $A_1,B,C$ 作截面。(1) 截面 $\triangle A_1BC$ 面积；(2) 截面与底面所成二面角余弦；(3) $h=\sqrt{3}a$ 时，$C_1$ 到截面距离。

**思路**（→ toolkit/11 坐标化 + 法向量）：建坐标系，$A_1BC$ 三点确定截面，法向量 → 面积（叉积模 / 2）→ 二面角 → 点面距（体积法或投影法）。

**解答**：

**第一步：建坐标系。** $A=(0,0,0),B=(a,0,0),C=\left(\frac{a}{2},\frac{\sqrt{3}a}{2},0\right),A_1=(0,0,h)$。

**第二步：截面面积。** $\overrightarrow{A_1B}=(a,0,-h),\ \overrightarrow{A_1C}=\left(\frac{a}{2},\frac{\sqrt{3}a}{2},-h\right)$。

$$\overrightarrow{A_1B}\times\overrightarrow{A_1C}=\left(\frac{\sqrt{3}ah}{2},-\frac{ah}{2}\cdot 2+\frac{ah}{2},\ \frac{\sqrt{3}a^2}{2}\right)$$

精确计算：$\boldsymbol{n}=(n_x,n_y,n_z)$，

$$n_x=0\cdot(-h)-(-h)\cdot\frac{\sqrt{3}a}{2}=\frac{\sqrt{3}ah}{2},\quad n_y=(-h)\cdot\frac{a}{2}-a\cdot(-h)=\frac{ah}{2},\quad n_z=a\cdot\frac{\sqrt{3}a}{2}-0\cdot\frac{a}{2}=\frac{\sqrt{3}a^2}{2}$$

$$|\boldsymbol{n}|=\sqrt{\frac{3a^2h^2}{4}+\frac{a^2h^2}{4}+\frac{3a^4}{4}}=\frac{a}{2}\sqrt{3h^2+a^2}\cdot\sqrt{1}=\frac{a}{2}\sqrt{3h^2+a^2+a^2}\rightarrow\frac{a}{2}\sqrt{3h^2+a^2}$$

（核查：$\frac{3a^2h^2}{4}+\frac{a^2h^2}{4}=a^2h^2$，$|\boldsymbol{n}|=\sqrt{a^2h^2+\frac{3a^4}{4}}=\frac{a}{2}\sqrt{4h^2+3a^2}$。）

截面面积 $= \dfrac{|\boldsymbol{n}|}{2} = \dfrac{a}{4}\sqrt{4h^2+3a^2}$。

**第三步：二面角。** $\boldsymbol{n}=\left(\frac{\sqrt{3}ah}{2},\frac{ah}{2},\frac{\sqrt{3}a^2}{2}\right)$，底面法向量 $\boldsymbol{k}=(0,0,1)$。

$$\cos\theta=\frac{|\boldsymbol{n}\cdot\boldsymbol{k}|}{|\boldsymbol{n}|}=\frac{\frac{\sqrt{3}a^2}{2}}{\frac{a}{2}\sqrt{4h^2+3a^2}}=\frac{\sqrt{3}a}{\sqrt{4h^2+3a^2}}$$

**第四步（$h=\sqrt{3}a$）：** $|\boldsymbol{n}|=\frac{a}{2}\sqrt{12a^2+3a^2}=\frac{a}{2}\cdot a\sqrt{15}=\frac{a^2\sqrt{15}}{2}$。截面面积 $=\frac{a^2\sqrt{15}}{4}$。

$C_1=({\frac{a}{2},\frac{\sqrt{3}a}{2},\sqrt{3}a})$，截面方程由 $\boldsymbol{n}\cdot(\boldsymbol{r}-A_1)=0$（$A_1=(0,0,\sqrt{3}a)$）：

$$\frac{\sqrt{3}ah}{2}x+\frac{ah}{2}y+\frac{\sqrt{3}a^2}{2}(z-h)=0\Rightarrow\sqrt{3}x+y+\sqrt{3}\cdot a\cdot\frac{z-\sqrt{3}a}{h}\cdot\frac{1}{1}=0$$

代 $h=\sqrt{3}a$：$\sqrt{3}x+y+a(z-\sqrt{3}a)/a=0\to\sqrt{3}x+y+(z-\sqrt{3}a)=0$，即 $\sqrt{3}x+y+z=\sqrt{3}a\cdot 2$… 重新算：$\frac{\sqrt{3}a\sqrt{3}a}{2}x+\frac{a\sqrt{3}a}{2}y+\frac{\sqrt{3}a^2}{2}z=\frac{\sqrt{3}a^2}{2}\cdot\sqrt{3}a$，即 $\frac{3a^2}{2}x+\frac{\sqrt{3}a^2}{2}y+\frac{\sqrt{3}a^2}{2}z=\frac{3a^3}{2}$，除以 $\frac{a^2}{2}$：$3x+\sqrt{3}y+\sqrt{3}z=3a$。

$C_1$ 到面距离：代 $(\frac{a}{2},\frac{\sqrt{3}a}{2},\sqrt{3}a)$：$3\cdot\frac{a}{2}+\sqrt{3}\cdot\frac{\sqrt{3}a}{2}+\sqrt{3}\cdot\sqrt{3}a=\frac{3a}{2}+\frac{3a}{2}+3a=6a$。距离 $= \dfrac{|6a-3a|}{\sqrt{9+3+3}}=\dfrac{3a}{\sqrt{15}}=\dfrac{3a\sqrt{15}}{15}=\dfrac{\sqrt{15}a}{5}$。

**答案**：截面面积 $\boxed{\dfrac{a}{4}\sqrt{4h^2+3a^2}}$；二面角余弦 $\boxed{\dfrac{\sqrt{3}a}{\sqrt{4h^2+3a^2}}}$；$C_1$ 到截面距离（$h=\sqrt{3}a$）$= \boxed{\dfrac{\sqrt{15}a}{5}}$。

**总结**：截面面积 = 叉积模 / 2；二面角由截面法向量与底面法向量夹角决定；点面距用截面方程代点坐标。

---

## E.32 [提升] Part 9/06

**题目回顾**：正方体 $ABCD\text{-}A_1B_1C_1D_1$，棱长 $1$，$E,F$ 分别是 $A_1B_1,CD$ 中点。(1) 建坐标系，求 $E,F$ 坐标；(2) $EF$ 与对角线 $AC_1$ 所成角余弦；(3) $E$ 到底面与到面 $ABB_1A_1$ 距离之比。

**思路**（→ toolkit/11 正方体坐标化）：正方体建标准坐标系，直接写坐标，用方向向量计算线线角，点面距即坐标分量。

**解答**：

**第一步：坐标。** $A=(0,0,0),B=(1,0,0),C=(1,1,0),D=(0,1,0),A_1=(0,0,1),B_1=(1,0,1),C_1=(1,1,1),D_1=(0,1,1)$。$E$ 为 $A_1B_1$ 中点：$E=\left(\frac{1}{2},0,1\right)$；$F$ 为 $CD$ 中点：$F=\left(\frac{1}{2},1,0\right)$。

**第二步：$EF$ 与 $AC_1$ 所成角。** $\overrightarrow{EF}=\left(0,1,-1\right)$，$\overrightarrow{AC_1}=(1,1,1)$。

$$\cos\theta=\frac{|\overrightarrow{EF}\cdot\overrightarrow{AC_1}|}{|\overrightarrow{EF}||\overrightarrow{AC_1}|}=\frac{|0+1-1|}{\sqrt{2}\cdot\sqrt{3}}=0$$

所成角为 $90°$，$EF \perp AC_1$。

**第三步：距离之比。** $E=\left(\frac{1}{2},0,1\right)$：$E$ 到底面（$z=0$）距离 $= z_E = 1$；$E$ 到面 $ABB_1A_1$（$y=0$）距离 $= y_E = 0$。

比值 $= 1:0$，即 $E$ 在面 $ABB_1A_1$ 上（因 $y_E=0$），到该面距离为 $0$。

**答案**：$E=\left(\dfrac{1}{2},0,1\right)$，$F=\left(\dfrac{1}{2},1,0\right)$；$EF$ 与 $AC_1$ 所成角 $\boxed{90°}$（互相垂直）；$E$ 在面 $ABB_1A_1$ 上，距离比 $= 1:0$（$E$ 到底面距离为 $1$，到面 $ABB_1A_1$ 距离为 $\boxed{0}$）。

**总结**：正方体坐标化后，点面距等于对应坐标分量；线线角用方向向量数量积。若 $E$ 恰在某面上，距离为零需特别指出。

---

## E.33 [提升] Part 8/05

**题目回顾**：正方形 $ABCD$ 边长 $2$，沿 $AC$ 折叠，$\triangle ABC$ 折起使折叠后 $\angle B'AC = 90°$（$B'$ 为折后 $B$ 的位置，$\triangle ACD$ 不动）。(1) 建坐标系；(2) 二面角 $B'\text{-}AC\text{-}D$ 余弦；(3) $B'$ 到面 $ACD$ 距离。

**思路**（→ toolkit/11 折叠 + 二面角）：$\triangle ACD$ 放底，$A,C,D$ 坐标固定。折叠后 $B'$ 满足 $|AB'|=|AB|=2$、$|CB'|=|CB|=2$ 及 $\angle B'AC=90°$。

**解答**：

**第一步：建坐标系。** 原正方形 $\triangle ACD$：$AC$ 对角线长 $2\sqrt{2}$，$AC$ 为折叠轴。以 $A=(0,0,0)$，$C=(2\sqrt{2},0,0)$（$x$ 轴沿 $AC$），$D$ 在底面。$AD=2,DC=2$，$D$ 到 $AC$ 距离：$\triangle ACD$ 中 $AC=2\sqrt{2},AD=DC=2$，$D$ 为 $\triangle$ 的顶点，$AD=CD=2$，$D$ 到 $AC$ 中点 $M_{AC}=(\sqrt{2},0,0)$ 距离 $= \sqrt{4-2} = \sqrt{2}$（因 $|DM_{AC}|=\sqrt{AD^2-(AC/2)^2}=\sqrt{4-2}=\sqrt{2}$），故 $D=(\sqrt{2},\sqrt{2},0)$（取 $y>0$）。

折叠前 $B$ 关于 $AC$ 的对称位置（在底面另侧），折叠后 $B'=(b_x,b_y,b_z)$：$|AB'|=2$，$|CB'|=2$，且 $\angle B'AC=90°$（$\overrightarrow{AB'}\perp\overrightarrow{AC}$）。

$\overrightarrow{AB'}=(b_x,b_y,b_z)$，$\overrightarrow{AC}=(2\sqrt{2},0,0)$，垂直条件：$b_x=0$。$|AB'|=2$：$b_y^2+b_z^2=4$。$|CB'|=2$：$(b_x-2\sqrt{2})^2+b_y^2+b_z^2=4$，展开：$8-4\sqrt{2}b_x+4=4$，$b_x=\frac{8}{4\sqrt{2}}=\sqrt{2}$，与 $b_x=0$ 矛盾。

重新理解 $\angle B'AC=90°$：$\angle B'AC$ 指折叠后 $B'$，$A$，$C$ 三点中 $A$ 处的角，即 $\overrightarrow{AB'}$ 与 $\overrightarrow{AC}$ 所成角为 $90°$，即 $\overrightarrow{AB'}\cdot\overrightarrow{AC}=0$，此即 $b_x=0$。但 $|CB'|$ 约束给出 $b_x=\sqrt{2}$，矛盾说明题目实为折叠使二面角 $B'\text{-}AC\text{-}D=90°$（即两半面垂直），非 $\angle B'AC=90°$。

**按二面角 $B'\text{-}AC\text{-}D=90°$ 解：** 折叠前 $B$ 在 $AC$ 另侧，$B$ 到 $AC$ 的垂足为 $AC$ 中点 $M_{AC}$，$BM_{AC}\perp AC$，$|BM_{AC}|=\sqrt{2}$（同 $D$ 到 $AC$）。折叠后 $B'M_{AC}\perp AC$ 且 $|B'M_{AC}|=\sqrt{2}$；二面角条件 $\overrightarrow{M_{AC}B'}\perp\overrightarrow{M_{AC}D}$。$\overrightarrow{M_{AC}D}=(0,\sqrt{2},0)$，故 $\overrightarrow{M_{AC}B'}=(0,b_y',b_z')$，且 $b_y'\cdot\sqrt{2}=0$，$b_y'=0$，$|B'M_{AC}|=|b_z'|=\sqrt{2}$，$B'=(\sqrt{2},0,\sqrt{2})$。

**第二步：二面角 $B'\text{-}AC\text{-}D$（验证）。** 平面 $ACD$：$\overrightarrow{AC}=(2\sqrt{2},0,0)$，$\overrightarrow{AD}=(\sqrt{2},\sqrt{2},0)$，法向量 $\boldsymbol{n_1}=(0,0,1)$（法分量仅 $z$）。平面 $ACB'$：$\overrightarrow{AB'}=(\sqrt{2},0,\sqrt{2})$，法向量 $\boldsymbol{n_2}=\overrightarrow{AC}\times\overrightarrow{AB'}=(2\sqrt{2},0,0)\times(\sqrt{2},0,\sqrt{2})=(0\cdot\sqrt{2}-0\cdot 0,0\cdot\sqrt{2}-2\sqrt{2}\cdot\sqrt{2},2\sqrt{2}\cdot 0-0\cdot\sqrt{2})=(0,-4,0)$，即 $\boldsymbol{n_2}=(0,-1,0)$。

$\cos\angle(\boldsymbol{n_1},\boldsymbol{n_2})=0$，二面角 $=90°$。✓

**第三步：$B'$ 到面 $ACD$。** 面 $ACD$ 在 $z=0$（底面），$B'=(\sqrt{2},0,\sqrt{2})$，距离 $= \sqrt{2}$。

**答案**：$B'=\left(\sqrt{2},0,\sqrt{2}\right)$；二面角 $B'\text{-}AC\text{-}D$ 余弦值 $\boxed{0}$（即 $90°$，与题设吻合）；$B'$ 到面 $ACD$ 距离 $\boxed{\sqrt{2}}$。

**总结**：折叠题用垂足距离不变 + 二面角条件定折后坐标；注意角的含义（顶角 vs 二面角）需仔细辨别。

---

## E.34 [提升] Part 8/06

**题目回顾**：等腰梯形 $ABCD$（$AB\parallel CD$，$AB=4,CD=2$，腰 $AD=BC=\sqrt{5}$）沿 $BD$ 折叠，使 $\triangle ABD \perp \triangle BCD$。(1) 折叠后建坐标系；(2) 直线 $AC$ 与面 $BCD$ 所成角正弦；(3) $A$ 到面 $BCD$ 距离。

**思路**（→ toolkit/11 折叠建系）：$\triangle BCD$ 放底面，$\triangle ABD$ 折起后两面垂直。$BD$ 为公共边，用 $BD$ 长及各边长定坐标。

**解答**：

**第一步：$BD$ 长。** 等腰梯形中 $AB=4,CD=2,AD=BC=\sqrt{5}$，$BD$ 的长：$B=(0,0),D=(x_D,y_D)$，$A=(-4,0)$... 更简：梯形中 $B=(2,0)$（取对称中心原点），$A=(-2,0)$，$D=(-1,h_0),C=(1,h_0)$，$AD=\sqrt{1+h_0^2}=\sqrt{5}$，$h_0=2$。$BD=\sqrt{(2-(-1))^2+4}=\sqrt{13}$。

**第二步：建坐标系。** 保持 $\triangle BCD$ 在底面，以 $B=(2,0)$，$D=(-1,2)$，$C=(1,2)$（平面坐标），映射到三维：$B=(0,0,0)$，沿 $BD$ 方向 $\overrightarrow{BD}=(-3,2,0)$，$|BD|=\sqrt{13}$，$y$ 轴沿 $BD$ 方向。$D=\left(0,\sqrt{13},0\right)$，$C$ 到 $BD$ 的垂足 + 垂直分量：$BC=\sqrt{5}$，$\overrightarrow{BC}$ 到 $\overrightarrow{BD}$ 方向投影 $= \frac{\overrightarrow{BC}\cdot\overrightarrow{BD}}{|BD|}$，$\overrightarrow{BC}=C-B=(1,2,0)\to$ 三维取底面：设 $C=(c_x,c_y,0)$，$|BC|=\sqrt{5}$，$|CD|=2$，$|BD|=\sqrt{13}$，$C$ 在 $BD$ 上的投影 $p=\dfrac{\overrightarrow{BC}\cdot\overrightarrow{BD}}{|BD|^2}$，$\overrightarrow{BD}=D-B$。

取简洁坐标：$B=(0,0,0)$，$D=(0,\sqrt{13},0)$（$y$ 轴），$C$ 在底面 $xOy$：$|BC|^2=c_x^2+c_y^2=5$，$|CD|^2=c_x^2+(c_y-\sqrt{13})^2=4$，相减：$-2\sqrt{13}c_y+13=1$，$c_y=\frac{12}{2\sqrt{13}}=\frac{6}{\sqrt{13}}$，$c_x=\sqrt{5-\frac{36}{13}}=\sqrt{\frac{29}{13}}=\frac{\sqrt{29}}{\sqrt{13}}$（取正）。

折叠后 $A$ 在 $\triangle ABD$ 中，$\triangle ABD \perp \triangle BCD$（面垂直），$A$ 满足 $|AB|=4,|AD|=\sqrt{5}$，$A$ 的 $y$ 分量（沿 $BD$）同上述 $C$ 的计算方式：$|AB|=4$，$\overrightarrow{BA}\cdot\overrightarrow{BD}/|BD|^2=\frac{5\cdot\frac{6}{\sqrt{13}}-5+4\cdot...\,}{}$... 设 $A=(a_x,a_y,a_z)$，$|AB|^2=a_x^2+a_y^2+a_z^2=16$，$|AD|^2=a_x^2+(a_y-\sqrt{13})^2+a_z^2=5$，相减：$-2\sqrt{13}a_y+13=11$，$a_y=\frac{1}{\sqrt{13}}$。

面垂直条件：$\triangle ABD$ 平面的法向量垂直于 $\triangle BCD$ 平面法向量...（即两面法向量互相垂直）。面 $BCD$ 法向量：$\boldsymbol{n}_{BCD}=\overrightarrow{BD}\times\overrightarrow{BC}=(0,\sqrt{13},0)\times(c_x,c_y,0)=(0\cdot 0-0\cdot c_y,\,0\cdot c_x-0\cdot 0,\,0\cdot c_y-\sqrt{13}c_x)=(0,0,-\sqrt{13}c_x)$，即 $\boldsymbol{n}_{BCD}=(0,0,1)$（底面法向量，$z$ 轴）。

面垂直即 $a_z\ne 0$，且 $A$ 所在半面与底面垂直。$A$ 在底面的投影（$z=0$ 分量）$= (a_x,\frac{1}{\sqrt{13}},0)$，$A$ 脱离底面由 $a_z$ 给出。由 $|AB|^2=a_x^2+\frac{1}{13}+a_z^2=16$。$A$ 在 $\triangle ABD$ 内需满足 $BD$ 方向约束：$A$ 对 $BD$（$y$ 轴）的垂分量与底面垂直（面垂直条件），即 $A$ 在底面的投影 $A_0=(a_x,\frac{1}{\sqrt{13}},0)$ 满足 $\overrightarrow{BA_0}\perp$ 面 $BCD$ 的某方向……实际上面垂直条件最简洁理解：$A$ 到 $BD$ 的垂线完全垂直于底面，即 $a_x=0$（$A$ 到 $BD$ 的垂足在 $y$ 轴上，$a_x=0$）。

取 $a_x=0$：$\frac{1}{13}+a_z^2=16$，$a_z=\sqrt{16-\frac{1}{13}}=\sqrt{\frac{207}{13}}=\frac{\sqrt{207}}{\sqrt{13}}=\frac{3\sqrt{23}}{\sqrt{13}}$。$A=\left(0,\frac{1}{\sqrt{13}},\frac{3\sqrt{23}}{\sqrt{13}}\right)$。

**第三步：$AC$ 与底面角。** $\overrightarrow{AC}=C-A=\left(\frac{\sqrt{29}}{\sqrt{13}}-0,\frac{6}{\sqrt{13}}-\frac{1}{\sqrt{13}},-\frac{3\sqrt{23}}{\sqrt{13}}\right)=\frac{1}{\sqrt{13}}\left(\sqrt{29},5,-3\sqrt{23}\right)$。

$$\sin\varphi=\frac{|z\text{-分量}|}{|\overrightarrow{AC}|}=\frac{\frac{3\sqrt{23}}{\sqrt{13}}}{\frac{1}{\sqrt{13}}\sqrt{29+25+207}}=\frac{3\sqrt{23}}{\sqrt{261}}=\frac{3\sqrt{23}}{3\sqrt{29}}=\sqrt{\frac{23}{29}}=\frac{\sqrt{667}}{29}$$

**第四步：$A$ 到底面距离 $= a_z = \dfrac{3\sqrt{23}}{\sqrt{13}}=\dfrac{3\sqrt{299}}{13}$。**

**答案**：$A$ 到面 $BCD$ 距离 $\boxed{\dfrac{3\sqrt{299}}{13}}$；$AC$ 与面 $BCD$ 所成角 $\sin\varphi = \boxed{\sqrt{\dfrac{23}{29}}}$。

**总结**：折叠梯形时，先算 $BD$ 长，再用距离约束确定折后坐标；面垂直条件等价于折边到折轴的垂线完全转向 $z$ 轴方向。

---

## E.35 [提升] Part 8/06

**题目回顾**：矩形 $ABCD$，$AB=2,BC=4$，沿 $EF$（$E\in AB,F\in CD$，$AE=CF=1$）折叠，$\triangle AEF \perp$ 面 $EBCF$。(1) 折叠后建坐标系，求各点坐标；(2) $A$ 到面 $EBCF$ 距离；(3) 四棱锥 $A\text{-}EBCF$ 体积。

**思路**（→ toolkit/11 折叠 + 体积）：$EBCF$ 为矩形（$EF=BC=4$，$EB=1$），放底面建系。$\triangle AEF$ 垂直于底面，$A$ 到 $EF$ 的垂线完全垂直底面。

**解答**：

**第一步：坐标系。** $E=(0,0,0),B=(1,0,0),C=(1,4,0),F=(0,4,0)$（$EBCF$ 为矩形，$EB=1,BC=4$）。$AE=1,AF=EF=4$，$\triangle AEF$ 中 $EF \parallel y$ 轴，$A$ 到 $EF$ 的垂足为 $E$（因 $AE\perp EF$：$AE=1$，$AF=\sqrt{AE^2+EF^2}=\sqrt{1+16}=\sqrt{17}$，不对——$AE=1$，$EF=4$，$\angle AEF$...）。

在原矩形中 $A,E,F$ 满足：$AE=1$（沿 $AB$ 方向），$EF=BC=4$（$EF\parallel AB$ 不对，$EF\parallel BC=4$），$AF=\sqrt{AE^2+EF^2}$（矩形中 $AF$ 为对角线：$AF=\sqrt{1^2+4^2}=\sqrt{17}$）。折叠后 $\triangle AEF$ 绕 $EF$ 翻起，$AE=1$，$A$ 到 $EF$ 垂足 $H$：在矩形中 $A$ 到 $EF$ 的距离 $= AE \sin\angle AEF = 1\cdot\frac{EF\perp}{}$... 矩形中 $A$ 在 $EF$ 上的投影：$\overrightarrow{EF}=(0,4,0)$，$\overrightarrow{EA}=(-1,0,0)$（折前，但折叠绕 $EF$ 转），$A$ 到 $EF$ 距离 $= 1$（垂直分量）。折后 $\triangle AEF\perp$ 底面，$A$ 到 $EF$ 的垂线（长 $1$）转为 $z$ 轴方向，$A=(0,0,1)$（$A$ 的 $EF$ 垂足为 $E$，因 $\overrightarrow{EA}$ 垂直 $EF$，$E=(0,0,0)$）。

验证：$|AE|=1$ ✓，$|AF|=\sqrt{0+16+1}=\sqrt{17}$ ✓，$\triangle AEF\perp$ 底面（$A$ 在 $z$ 轴）✓。

**第二步：$A$ 到面 $EBCF$ 距离。** 面 $EBCF$ 为 $z=0$ 平面，$A=(0,0,1)$，距离 $= 1$。

**第三步：体积。** $EBCF$ 面积 $= 1\times 4=4$，$A$ 到底面距离 $= 1$。

$$V = \frac{1}{3}\times 4\times 1 = \frac{4}{3}$$

**答案**：$A=(0,0,1)$（其余点如第一步）；$A$ 到面 $EBCF$ 距离 $\boxed{1}$；体积 $V = \boxed{\dfrac{4}{3}}$。

**总结**：矩形折叠绕 $EF$ 时，若折起部分垂直底面，则折起顶点的 $z$ 坐标 = 该顶点到折叠轴的距离。

---

## E.36 [提升] Part 9/08

**题目回顾**：三棱锥 $P\text{-}ABC$，$PA=PB=PC=3$，$AB=BC=CA=2\sqrt{2}$。(1) $P$ 到 $\triangle ABC$ 垂足 $H$ 是否为重心；(2) 棱 $PA$ 上是否存在 $M$ 使 $BM\perp PC$；(3) 棱 $BC$ 上是否存在 $N$ 使 $PN\perp$ 面 $ABN$。

**思路**（→ toolkit/11 坐标化 + 存在性验证）：等长等边三棱锥建标准坐标，重心为等边三角形重心。存在性用参数化设点 + 正交条件解方程。

**解答**：

**第一步：建坐标系。** 等边三角形 $\triangle ABC$ 边长 $2\sqrt{2}$，中心（重心）$G$，$|GA|=\dfrac{2\sqrt{2}}{\sqrt{3}}=\dfrac{2\sqrt{6}}{3}$。$A=(0,0,0)$，底面中建系：$G=\left(\frac{\sqrt{2}}{...}\right)$，取简洁：$A=(-\sqrt{2},0,0)$，$B=(\sqrt{2},0,0)$，$C=(0,\sqrt{6},0)$（$|AB|=2\sqrt{2}$，$|AC|=\sqrt{2+6}=2\sqrt{2}$ ✓），$G=\frac{A+B+C}{3}=\left(0,\frac{\sqrt{6}}{3},0\right)$。

$|GA|=\sqrt{2+\frac{6}{9}}=\sqrt{2+\frac{2}{3}}=\sqrt{\frac{8}{3}}=\frac{2\sqrt{2}}{\sqrt{3}}$ ✓。$|PG|=\sqrt{9-\frac{8}{3}}=\sqrt{\frac{19}{3}}$，$P=\left(0,\frac{\sqrt{6}}{3},\sqrt{\frac{19}{3}}\right)$。

(1) $PA=PB=PC$ 等长 → $P$ 在底面等距三顶点 → 射影为外心。等边三角形外心 = 重心，故 $H=G$ 为重心。✓

**(2) $M$ 在 $PA$ 上：** $M=P+t\overrightarrow{PA}=P+t(A-P)$，$t\in(0,1)$。$\overrightarrow{BM}=M-B$，$\overrightarrow{PC}=C-P$，条件 $\overrightarrow{BM}\cdot\overrightarrow{PC}=0$。

$\overrightarrow{PA}=A-P=\left(-\sqrt{2}-0,-\frac{\sqrt{6}}{3},-\sqrt{\frac{19}{3}}\right)$，$M=P+t\overrightarrow{PA}=\left(-t\sqrt{2},(1-t)\frac{\sqrt{6}}{3},(1-t)\sqrt{\frac{19}{3}}\right)$（修正：$P$ 的 $y$ 分量为 $\frac{\sqrt{6}}{3}$，$A$ 的 $y=0$，$M_y=\frac{\sqrt{6}}{3}+t(0-\frac{\sqrt{6}}{3})=\frac{\sqrt{6}}{3}(1-t)$；同理 $M_x=-t\sqrt{2}$，$M_z=(1-t)\sqrt{\frac{19}{3}}$）。

$\overrightarrow{PC}=C-P=\left(0,\sqrt{6}-\frac{\sqrt{6}}{3},-\sqrt{\frac{19}{3}}\right)=\left(0,\frac{2\sqrt{6}}{3},-\sqrt{\frac{19}{3}}\right)$。$\overrightarrow{BM}=M-B=\left(-t\sqrt{2}-\sqrt{2},\frac{\sqrt{6}}{3}(1-t),-\frac{\sqrt{19}}{\sqrt{3}}(1-t)\right)$... 实际 $B=(\sqrt{2},0,0)$：$\overrightarrow{BM}=(-t\sqrt{2}-\sqrt{2},\frac{\sqrt{6}}{3}(1-t),(1-t)\sqrt{\frac{19}{3}})$（$z$ 分量含负号，$B_z=0$，$M_z=(1-t)\sqrt{19/3}$）。

$\overrightarrow{BM}\cdot\overrightarrow{PC}=(-\sqrt{2}(t+1))\cdot 0+\frac{\sqrt{6}}{3}(1-t)\cdot\frac{2\sqrt{6}}{3}+(1-t)\sqrt{\frac{19}{3}}\cdot\left(-\sqrt{\frac{19}{3}}\right)$

$=\frac{4(1-t)}{3}-\frac{19(1-t)}{3}=\frac{(1-t)(4-19)}{3}=\frac{-15(1-t)}{3}=-5(1-t)$

令 $=0$：$t=1$，即 $M=A$，不在棱内部。**不存在**内部点 $M$。

**(3) $N$ 在 $BC$ 上：** 设 $N=B+s\overrightarrow{BC}=(\sqrt{2},0,0)+s(-\sqrt{2},\sqrt{6},0)=(\sqrt{2}(1-s),s\sqrt{6},0)$，$s\in(0,1)$。$PN\perp$ 面 $ABN$ 要求 $\overrightarrow{PN}\perp\overrightarrow{AB}$ 且 $\overrightarrow{PN}\perp\overrightarrow{AN}$（或 $\overrightarrow{BN}$）。

$\overrightarrow{PN}=N-P=\left(\sqrt{2}(1-s)-0,s\sqrt{6}-\frac{\sqrt{6}}{3},-\sqrt{\frac{19}{3}}\right)$，$\overrightarrow{AB}=(2\sqrt{2},0,0)$。

$\overrightarrow{PN}\cdot\overrightarrow{AB}=2\sqrt{2}\cdot\sqrt{2}(1-s)=4(1-s)=0$，故 $s=1$，即 $N=C$，不在棱内部。**不存在**。

**答案**：(1) 垂足 $H$ 是重心（$PA=PB=PC$ 等长且底面等边三角形，外心 $=$ 重心）$\boxed{\text{是}}$；(2) **不存在**这样的 $M$；(3) **不存在**这样的 $N$。

**总结**：等长等边三棱锥（正三棱锥）坐标化后，存在性问题化为线性方程，若解在边界则不存在内部点。

---

## E.37 [提升] Part 9/08

**题目回顾**：正三棱柱 $ABC\text{-}A_1B_1C_1$，底面边长 $2$，高 $2\sqrt{3}$。(1) $AA_1$ 上是否存在 $P$ 使 $BP\perp$ 面 $A_1BC$；(2) $M,N$ 分别是 $BB_1,CC_1$ 中点，面 $AMN$ 与底面二面角余弦；(3) $A_1$ 到面 $AMN$ 距离。

**思路**（→ toolkit/11 正三棱柱坐标化 + 存在性）：建标准坐标系，存在性用 $\overrightarrow{BP}$ 与面 $A_1BC$ 的法向量平行来判断。

**解答**：

**第一步：坐标系。** $A=(0,0,0)$，$B=(2,0,0)$，$C=(1,\sqrt{3},0)$，$A_1=(0,0,2\sqrt{3})$，$B_1=(2,0,2\sqrt{3})$，$C_1=(1,\sqrt{3},2\sqrt{3})$。

**(1) 存在性：** 设 $P=(0,0,h)$，$h\in(0,2\sqrt{3})$。$\overrightarrow{BP}\parallel$ 面 $A_1BC$ 的法向量。面 $A_1BC$ 的法向量：$\overrightarrow{BA_1}=(-2,0,2\sqrt{3})$，$\overrightarrow{BC}=(-1,\sqrt{3},0)$，

$$\boldsymbol{n}=\overrightarrow{BA_1}\times\overrightarrow{BC}=\begin{vmatrix}\boldsymbol{i}&\boldsymbol{j}&\boldsymbol{k}\\-2&0&2\sqrt{3}\\-1&\sqrt{3}&0\end{vmatrix}=(0-2\sqrt{3}\cdot\sqrt{3},-(0-(-2)\cdot 0),(-2\sqrt{3}-0))=(-6,0,-2\sqrt{3})$$

化简：$\boldsymbol{n}=(3,0,\sqrt{3})$（取反方向）。$\overrightarrow{BP}=P-B=(-2,0,h)$。$BP\perp$ 面 $A_1BC$ 要求 $\overrightarrow{BP}\parallel\boldsymbol{n}$：$\dfrac{-2}{3}=\dfrac{0}{0}=\dfrac{h}{\sqrt{3}}$，中间项 $\frac{0}{0}$ 不定，但需 $0\parallel 0$ 分量相容，须第二分量 $\frac{0}{0}$ 成立（自动），由 $\frac{-2}{3}=\frac{h}{\sqrt{3}}$：$h=\frac{-2\sqrt{3}}{3}$，负值不在棱上。**不存在**。

**(2) 面 $AMN$ 与底面二面角：** $M=(2,0,\sqrt{3})$（$BB_1$ 中点），$N=(1,\sqrt{3},\sqrt{3})$（$CC_1$ 中点），$A=(0,0,0)$。

$\overrightarrow{AM}=(2,0,\sqrt{3})$，$\overrightarrow{AN}=(1,\sqrt{3},\sqrt{3})$。法向量：

$$\boldsymbol{n}_{AMN}=\overrightarrow{AM}\times\overrightarrow{AN}=\begin{vmatrix}\boldsymbol{i}&\boldsymbol{j}&\boldsymbol{k}\\2&0&\sqrt{3}\\1&\sqrt{3}&\sqrt{3}\end{vmatrix}=(0\cdot\sqrt{3}-\sqrt{3}\cdot\sqrt{3},\,\sqrt{3}\cdot 1-2\cdot\sqrt{3},\,2\sqrt{3}-0)=(-3,-\sqrt{3},2\sqrt{3})$$

底面法向量 $\boldsymbol{k}=(0,0,1)$。

$$\cos\theta=\frac{|\boldsymbol{n}_{AMN}\cdot\boldsymbol{k}|}{|\boldsymbol{n}_{AMN}|}=\frac{2\sqrt{3}}{\sqrt{9+3+12}}=\frac{2\sqrt{3}}{\sqrt{24}}=\frac{2\sqrt{3}}{2\sqrt{6}}=\frac{\sqrt{3}}{\sqrt{6}}=\frac{1}{\sqrt{2}}=\frac{\sqrt{2}}{2}$$

二面角余弦值 $\dfrac{\sqrt{2}}{2}$，即 $45°$。

**(3) $A_1$ 到面 $AMN$ 距离：** $A_1=(0,0,2\sqrt{3})$，面方程由 $\boldsymbol{n}_{AMN}=(-3,-\sqrt{3},2\sqrt{3})$ 过 $A=(0,0,0)$：$-3x-\sqrt{3}y+2\sqrt{3}z=0$。

代 $A_1$：$0+0+2\sqrt{3}\cdot 2\sqrt{3}=12$。距离 $= \dfrac{|12|}{|\boldsymbol{n}_{AMN}|}=\dfrac{12}{\sqrt{24}}=\dfrac{12}{2\sqrt{6}}=\dfrac{6}{\sqrt{6}}=\sqrt{6}$。

**答案**：(1) $AA_1$ 上**不存在**这样的 $P$；(2) 面 $AMN$ 与底面二面角余弦 $\boxed{\dfrac{\sqrt{2}}{2}}$（即 $45°$）；(3) $A_1$ 到面 $AMN$ 距离 $\boxed{\sqrt{6}}$。

**总结**：正三棱柱坐标化直接。存在性问题：$BP\perp$ 面等价于 $\overrightarrow{BP}\parallel$ 面法向量，列比例方程若解超出范围则不存在。点面距 = 代入面方程数值除以法向量模。

---

> **本册覆盖**：E.26–E.37，共 **12 题**，含二面角综合（E.26–E.29）、含参立体（E.30–E.32）、折叠立体（E.33–E.35）、存在性（E.36–E.37）四大类。
>
> **核心工具**：空间坐标化 → 法向量（叉积）→ 点面距（面方程代点）/ 线面角（方向向量与法向量夹角）/ 二面角（两法向量夹角）。
> 存在性一律参数化设点，代入正交或平行条件解方程，解越界则不存在。
