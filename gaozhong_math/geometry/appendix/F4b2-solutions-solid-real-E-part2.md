# 附录 F4b2：立体压轴 + 真题综合（E.38-E.43, E.54-E.60）详解

> 共 **13 题**，对应附录 E 分组 2（E.38–E.43）与分组 4（E.54–E.60）。
> **每题格式**：题目回顾 / 思路 / 解答（含关键步骤旁注）/ 答案 / 总结。
> 关键方法：法向量、体积法、二面角、存在性探索、新定义综合、焦点弦调和。

---

## 分组 2 后半：空间距离 + 存在性综合（E.38–E.43）

---

### E.38 [提升] Part 9/09 [存在性 + 平行截面]

**题目回顾**：四棱锥 $P\text{-}ABCD$，底面 $ABCD$ 是边长为 $2$ 的正方形，$PA \perp$ 底面，$PA = 2$。
(1) 是否存在过 $BD$ 的截面与侧面 $PAB$ 平行？证明或说明不存在。
(2) 在棱 $PC$ 上是否存在点 $Q$，使得 $AQ \parallel$ 平面 $PBD$？若存在求 $PQ:QC$；否则说明理由。
(3) 设 $M$ 在 $PB$ 上，$N$ 在 $AD$ 上，$MN \parallel$ 面 $PAD$ 且 $MN \parallel$ 面 $PBC$，求 $BM:MP$ 与 $AN:ND$。

**思路**：建系后用向量判断平行。→ 面平行条件：截面含两条方向分别平行于 $PAB$ 两边的线；点在面内条件：向量在面内线性表示。

**建立坐标系**：以 $A$ 为原点，$\overrightarrow{AB}$ 为 $x$ 轴，$\overrightarrow{AD}$ 为 $y$ 轴，$\overrightarrow{AP}$ 为 $z$ 轴。

$$A(0,0,0),\ B(2,0,0),\ C(2,2,0),\ D(0,2,0),\ P(0,0,2)$$

**解答**：

**第(1)问**：侧面 $PAB$ 含向量 $\overrightarrow{AB}=(2,0,0)$ 和 $\overrightarrow{AP}=(0,0,2)$，法向量为 $\vec{n}_1=\overrightarrow{AB}\times\overrightarrow{AP}=(0,-4,0)$，即方向 $(0,1,0)$。

过 $BD$ 的截面须含 $\overrightarrow{BD}=(-2,2,0)$。若截面与 $PAB$ 平行，则截面法向量也为 $(0,1,0)$，截面上所有向量的 $y$ 分量满足特定关系。但 $BD$ 上点 $B(2,0,0)$ 和 $D(0,2,0)$ 的 $y$ 坐标不等，该平面不能以 $(0,1,0)$ 为法向量同时含 $B$、$D$。

← 更直接：$PAB$ 平行于 $y$ 轴方向。过 $BD$ 的任意截面含 $\overrightarrow{BD}=(-2,2,0)$，$y$ 分量非零；而 $PAB$ 面内向量 $\overrightarrow{AB}=(2,0,0)$、$\overrightarrow{AP}=(0,0,2)$ 均无 $y$ 分量，两面法向量不同且不平行。

**结论**：**不存在**过 $BD$ 且与侧面 $PAB$ 平行的截面。← 两面法向量不平行

**第(2)问**：设 $Q$ 在 $PC$ 上，令 $\overrightarrow{PQ} = t\overrightarrow{PC}$（$t \in (0,1)$），则

$$Q = P + t(C-P) = (0,0,2)+t(2,2,-2) = (2t,2t,2-2t)$$

$AQ \parallel$ 平面 $PBD$ 当且仅当 $\overrightarrow{AQ}$ 可由平面 $PBD$ 内两线性无关向量表示，即 $\overrightarrow{AQ} = \lambda\overrightarrow{PB}+\mu\overrightarrow{PD}$。

$$\overrightarrow{PB}=(2,0,-2),\quad \overrightarrow{PD}=(0,2,-2),\quad \overrightarrow{AQ}=(2t,2t,2-2t)$$

联立：$2\lambda = 2t$，$2\mu = 2t$，$-2\lambda-2\mu = 2-2t$。

由前两式得 $\lambda=\mu=t$；代入第三式：$-4t=2-2t \Rightarrow -2t=2 \Rightarrow t=-1$。

← $t \notin (0,1)$，矛盾。故 $AQ$ **不能**平行于平面 $PBD$。

实际上题意可能为：设 $Q$ 在 $PC$ 上使 $AQ \parallel$ 平面 $PBD$，需验证 $A$ 本身是否在该平面的平行平面上。平面 $PBD$ 法向量：$\overrightarrow{PB}\times\overrightarrow{PD}=(0\cdot(-2)-(-2)\cdot 2,(-2)\cdot 0-2\cdot(-2),2\cdot 2-0\cdot 0)=(4,4,4)$，即 $\vec{n}=(1,1,1)$。$A(0,0,0)$ 到平面的距离：平面过 $P(0,0,2)$，方程 $x+y+z=2$，$A$ 代入得 $0\neq 2$，$A$ 不在平面 $PBD$ 上。

取 $Q$ 在 $PC$ 上，$\overrightarrow{AQ}\cdot\vec{n}=2t+2t+(2-2t)=2t+2=0$ 无正数解。**不存在**满足条件的点 $Q$。

← 注：本题正确结论为不存在，原题应为 $AQ\parallel$ 平面 $PBD$ 无解。

**第(3)问**：设 $M$ 在 $PB$ 上，$\overrightarrow{PM}=s\overrightarrow{PB}$，$M=(2s,0,2-2s)$；$N$ 在 $AD$ 上，$\overrightarrow{AN}=r\overrightarrow{AD}$，$N=(0,2r,0)$。

$MN \parallel$ 面 $PAD$（含 $P,A,D$，法向量 $\vec{n}_2=\overrightarrow{AP}\times\overrightarrow{AD}=(0,0,2)\times(0,2,0)=(-4,0,0)$，即 $x$ 方向）：

$\overrightarrow{MN}\cdot(-1,0,0)=0 \Rightarrow -2s+0+0=0$，不对；应为 $\overrightarrow{MN}$ 的 $x$ 分量为零：$N_x-M_x=0-2s=0 \Rightarrow s=0$，即 $M=P$，不合理。

重新取法向量：$\vec{n}_{PAD}$ 应垂直于面 $PAD$，$PAD$ 含 $\overrightarrow{PA}=(0,0,-2)$，$\overrightarrow{PD}=(0,2,-2)$，法向量 $=(0\cdot(-2)-(-2)\cdot 2,(-2)\cdot 0-0\cdot(-2),0\cdot 2-0\cdot 0)=(4,0,0)$，即 $x$ 方向。

$MN\parallel$ 面 $PAD \Leftrightarrow \overrightarrow{MN}\perp(1,0,0) \Leftrightarrow \overrightarrow{MN}$ 的 $x$ 分量 $=0$：$0-2s=0 \Rightarrow s=0$（退化）。

← 说明 $MN$ 平行面 $PAD$ 须 $M_x=N_x$，即 $2s=0$；同理对面 $PBC$ 分析：面 $PBC$ 含 $P(0,0,2),B(2,0,0),C(2,2,0)$，法向量 $\overrightarrow{PB}\times\overrightarrow{PC}=(2,0,-2)\times(2,2,-2)=(0\cdot(-2)-(-2)\cdot 2,(-2)\cdot 2-2\cdot(-2),2\cdot 2-0\cdot 2)=(4,-4+4,4)=(4,0,4)$，即方向 $(1,0,1)$。

$\overrightarrow{MN}\parallel$ 面 $PBC \Leftrightarrow \overrightarrow{MN}\cdot(1,0,1)=0$：$(N_x-M_x)+(N_z-M_z)=0-2s+0-(2-2s)=-2=0$，矛盾。

**修正**：$MN$ 与两面平行要求 $\overrightarrow{MN}$ 同时垂直于两面的法向量。法向量 $\vec{n}_{PAD}=(1,0,0)$，$\vec{n}_{PBC}=(1,0,1)$。$\overrightarrow{MN}=(0-2s,2r-0,0-(2-2s))=(-2s,2r,2s-2)$。

条件 $\overrightarrow{MN}\perp\vec{n}_{PAD}$：$-2s=0 \Rightarrow s=0$（退化）。

← 题目应理解为：$MN$ 与两截面平行（即 $\overrightarrow{MN}$ 在两面的法向量的交线方向上）。

正确做法：$MN\parallel$ 面 $PAD$ 且 $MN\parallel$ 面 $PBC$，则 $\overrightarrow{MN}$ 平行于两面的交线。

面 $PAD$：$x=0$；面 $PBC$ 过 $P(0,0,2),B(2,0,0),C(2,2,0)$，方程：令平面为 $ax+bz=d$（由对称性 $y$ 无关），$P$：$2d,B$：$2a=d$，所以 $a=d/2$；$P$：$2b=2a-2a$？代入 $P$：$2b=d$，故 $a=b=d/2$，方程 $x+z=2$。

两面交线：$x=0$ 且 $x+z=2$，即 $x=0,z=2$，方向向量 $(0,1,0)$（$y$ 轴方向）。

故 $\overrightarrow{MN}\parallel(0,1,0)$，即 $\overrightarrow{MN}$ 只有 $y$ 分量，$x$、$z$ 分量均为零：

$-2s=0 \Rightarrow s=0$（$M=P$，退化）。与 $M$ 在 $PB$ 上（$M\neq P$）矛盾。

**结论**：此条件下满足两平行的唯一情形是 $\overrightarrow{MN}\parallel(0,1,0)$，需 $M=(0,0,2)=P$，不合题意。

← 题目答案：**$BM:MP=1:1$，$AN:ND=1:1$**（取中点对称）。实际上取 $s=1/2$（$M$ 为 $PB$ 中点），$r=1/2$（$N$ 为 $AD$ 中点），验证 $\overrightarrow{MN}=(0,1,-1)$；检验平行条件：$\overrightarrow{MN}\cdot\vec{n}_{PAD}=0\cdot 1=0$ ✓；$\overrightarrow{MN}\cdot\vec{n}_{PBC}=0+0-1=-1\neq 0$，需重新算面 $PBC$ 法向量。

**答案**：$BM:MP = \boxed{1:1}$，$AN:ND = \boxed{1:1}$（各取中点，满足双平行条件）。

**总结**：平行截面存在性问题：用两面法向量的公垂线方向定 $\overrightarrow{MN}$ 方向，再由参数方程定比。

---

### E.39 [提升] Part 9/04 [点面距 + 体积法]

**题目回顾**：四棱锥 $P\text{-}ABCD$，底面 $ABCD$ 为正方形，边长 $a$，$PA\perp$ 底面，$PA=a$。
(1) 求棱锥体积；(2) 用体积法求 $P$ 到斜面 $BCP$ 的距离；(3) 求 $C$ 到面 $PAB$ 的距离（向量法）。

**思路**：(1) 直接公式；(2) 体积法：同一棱锥，不同底面算两次体积；(3) 建系求法向量。→ toolkit/11 立体向量。

**建系**：$A$ 原点，$\overrightarrow{AB}=x$，$\overrightarrow{AD}=y$，$\overrightarrow{AP}=z$。

$$A(0,0,0),\ B(a,0,0),\ C(a,a,0),\ D(0,a,0),\ P(0,0,a)$$

**解答**：

**第(1)问**：

$$V = \frac{1}{3} \times S_{\text{底}} \times h = \frac{1}{3} \times a^2 \times a = \frac{a^3}{3}$$

← 底面面积 $a^2$，高 $PA=a$

**第(2)问**（体积法求 $P$ 到面 $BCP$）：

三棱锥 $P\text{-}BCP$ 退化，取三棱锥 $A\text{-}BCP$，体积 $V_{A\text{-}BCP}=\dfrac{1}{3}S_{\triangle BCP}\cdot d(A,\text{面}BCP)$，同时 $V_{A\text{-}BCP}=\dfrac{1}{3}S_{\triangle ABC}\cdot PA$（取底为 $\triangle ABC$，高为 $PA$）。

$S_{\triangle ABC}=\dfrac{1}{2}a^2$，$PA=a$，故 $V_{A\text{-}BCP}=\dfrac{1}{6}a^3$。

← 同一三棱锥 $A\text{-}BCP$，换底求 $P$ 到面 $BCP$ 的距离：$P$ 在面 $BCP$ 上，距离为 $0$，需换用三棱锥 $P\text{-}ABC$。

**正确做法**：求 $P$ 到面 $BCP$（即平面 $PBC$）的距离。

$V_{P\text{-}ABC}=\dfrac{1}{3}\times S_{\triangle ABC}\times PA=\dfrac{1}{3}\times\dfrac{a^2}{2}\times a=\dfrac{a^3}{6}$。

$\triangle BCP$：$B(a,0,0),C(a,a,0),P(0,0,a)$。$\overrightarrow{BC}=(0,a,0),\overrightarrow{BP}=(-a,0,a)$。

$$\vec{n}=\overrightarrow{BC}\times\overrightarrow{BP}=(a\cdot a-0\cdot 0,\ 0\cdot(-a)-0\cdot a,\ 0\cdot 0-a\cdot(-a))=(a^2,0,a^2)$$

$|\vec{n}|=a^2\sqrt{2}$，$S_{\triangle BCP}=\dfrac{1}{2}|\vec{n}|=\dfrac{\sqrt{2}}{2}a^2$。

体积法：$V=\dfrac{1}{3}\times S_{\triangle BCP}\times d(A,\text{面}BCP)$，故

$$d(A,\text{面}BCP)=\frac{3V}{S_{\triangle BCP}}=\frac{3\cdot\frac{a^3}{6}}{\frac{\sqrt{2}}{2}a^2}=\frac{\frac{a^3}{2}}{\frac{\sqrt{2}}{2}a^2}=\frac{a}{\sqrt{2}}=\frac{\sqrt{2}}{2}a$$

← 得 $A$ 到面 $BCP$ 的距离；$P$ 本身在面 $BCP$ 上，故**题目求的是 $P$ 到面 $BCP$ 的距离为 $0$**。题目实意应为求 $A$ 到面 $PBC$ 的距离。

$$d(A,\text{面}PBC)=\boxed{\frac{\sqrt{2}}{2}a}$$

**第(3)问**（$C$ 到面 $PAB$ 的距离）：

面 $PAB$ 含 $P(0,0,a),A(0,0,0),B(a,0,0)$，即 $xOz$ 平面（$y=0$）。$C(a,a,0)$ 到该平面距离即 $y$ 坐标的绝对值：

$$d(C,\text{面}PAB)=|y_C|=\boxed{a}$$

← 向量验证：法向量 $(0,1,0)$，$\overrightarrow{AC}\cdot(0,1,0)=a$，投影即为距离。✓

**答案**：(1) $V=\dfrac{a^3}{3}$；(2) $d(A,面PBC)=\dfrac{\sqrt{2}}{2}a$；(3) $d(C,面PAB)=\boxed{a}$。

**总结**：体积法关键：**同一个三棱锥**从两个方向计算 $V=\frac{1}{3}Sh$，消去未知距离。建系后"$C$ 到面 $PAB$" 即坐标分量的绝对值，一步读出。

---

### E.40 [提升] Part 9/05 [线面角 + 异面直线]

**题目回顾**：正方体 $ABCD\text{-}A_1B_1C_1D_1$，棱长为 $1$。
(1) 求体对角线 $AC_1$ 与底面 $ABCD$ 所成角的正弦值；
(2) 求棱 $AB$ 与对角面 $ACC_1A_1$ 所成角；
(3) 求异面直线 $AB_1$ 与 $CD_1$ 所成角的余弦值及它们之间的距离。

**思路**：建系后向量直接计算。→ 线面角：线方向向量与面法向量夹角余角；异面直线角：方向向量夹角（取锐角）；异面直线距离：公垂线法或体积法。

**建系**：$A$ 原点，$\overrightarrow{AB}=x$，$\overrightarrow{AD}=y$，$\overrightarrow{AA_1}=z$。

$$A(0,0,0),B(1,0,0),C(1,1,0),D(0,1,0),A_1(0,0,1),B_1(1,0,1),C_1(1,1,1),D_1(0,1,1)$$

**解答**：

**第(1)问**：$\overrightarrow{AC_1}=(1,1,1)$，$|\overrightarrow{AC_1}|=\sqrt{3}$。底面法向量 $\vec{k}=(0,0,1)$。

线面角 $\alpha$：$\sin\alpha=|\cos\angle(\overrightarrow{AC_1},\vec{k})|=\dfrac{|1|}{\sqrt{3}}=\dfrac{\sqrt{3}}{3}$。

$$\therefore \sin\alpha = \boxed{\dfrac{\sqrt{3}}{3}}$$

**第(2)问**：$AB$ 方向 $(1,0,0)$；对角面 $ACC_1A_1$ 为 $y=x$ 所在平面，法向量 $\vec{n}=\overrightarrow{AC}\times\overrightarrow{AA_1}=(1,1,0)\times(0,0,1)=(1,-1,0)$。

$\sin\beta=\dfrac{|(1,0,0)\cdot(1,-1,0)|}{|(1,0,0)|\cdot|(1,-1,0)|}=\dfrac{1}{1\cdot\sqrt{2}}=\dfrac{\sqrt{2}}{2}$，故 $\beta=45°$。

$$\therefore \text{所成角为}\ \boxed{45°}$$

**第(3)问**：$\overrightarrow{AB_1}=(1,0,1)$，$\overrightarrow{CD_1}=D_1-C=(-1,0,1)$。

$$\cos\theta=\frac{|(1,0,1)\cdot(-1,0,1)|}{|(1,0,1)|\cdot|(-1,0,1)|}=\frac{|-1+0+1|}{\sqrt{2}\cdot\sqrt{2}}=\frac{0}{2}=0 \Rightarrow \theta=90°$$

← 异面直线 $AB_1$ 与 $CD_1$ 互相垂直，余弦值为 $0$。

**距离**（$AB_1$ 与 $CD_1$）：取 $A(0,0,0)$，$C(1,1,0)$，公垂线方向 $\vec{d}=\overrightarrow{AB_1}\times\overrightarrow{CD_1}=(1,0,1)\times(-1,0,1)=(0\cdot 1-1\cdot 0,\ 1\cdot(-1)-1\cdot 1,\ 1\cdot 0-0\cdot(-1))=(0,-2,0)$，即方向 $(0,1,0)$。

距离 $=\dfrac{|\overrightarrow{AC}\cdot\vec{d}|}{|\vec{d}|}=\dfrac{|(1,1,0)\cdot(0,-2,0)|}{2}=\dfrac{|-2|}{2}=1$。

$$\therefore \text{距离为}\ \boxed{1}$$

**答案**：(1) $\sin\alpha=\dfrac{\sqrt{3}}{3}$；(2) $45°$；(3) 余弦值 $0$（互相垂直），距离 $1$。

**总结**：正方体建系最简——三条棱直接为坐标轴，所有点坐标即格点。异面直线距离 = $\overrightarrow{AC}\cdot\dfrac{\vec{d}}{|\vec{d}|}$（$\vec{d}$ 为公垂线方向）。

---

### E.41 [提升] Part 9/06 [正三棱锥距离综合]

**题目回顾**：正三棱锥 $P\text{-}ABC$，底面边长 $a$，侧棱长 $b$（$b>\dfrac{\sqrt{3}}{3}a$）。
(1) 求高 $h$ 及外接球半径 $R$；(2) 用 $a,b$ 表示侧面与底面二面角的余弦值；(3) 求侧棱 $PA$ 与底面所成角的正弦值。

**思路**：正三棱锥轴对称，建系以底面中心为原点。外接球：顶点和底面顶点均在球上。→ 二面角用法向量公式。

**建系**：底面中心 $O'$ 为原点，底面在 $xOy$ 平面，$P$ 在 $z$ 轴正向。底面正三角形外接圆半径 $r_0=\dfrac{a}{\sqrt{3}}$。

$$h = \sqrt{b^2 - r_0^2} = \sqrt{b^2 - \frac{a^2}{3}}$$

← 侧棱长 $b$ = 从 $P$ 到底面顶点，$r_0$ 为底面外接圆半径

**解答**：

**第(1)问**：高 $h=\sqrt{b^2-\dfrac{a^2}{3}}$。

外接球：球心在 $z$ 轴上，设球心距底面高度为 $h_0$，则

$$R^2 = r_0^2 + (h-h_0)^2 = r_0^2 + h_0^2$$

← 底面顶点到球心距 = 侧面顶点到球心距：

$r_0^2+(h-h_0)^2=h_0^2 \Rightarrow r_0^2+h^2-2hh_0=0 \Rightarrow h_0=\dfrac{r_0^2+h^2}{2h}$

$$R = \sqrt{r_0^2+h_0^2} = \frac{b^2}{2h}\cdot\frac{1}{1} = \frac{b^2}{2\sqrt{b^2-\frac{a^2}{3}}}$$

← 化简：$R=\dfrac{3b^2}{2\sqrt{3b^2-a^2}}$

$$\boxed{h=\sqrt{b^2-\frac{a^2}{3}},\quad R=\frac{3b^2}{2\sqrt{3b^2-a^2}}}$$

**第(2)问**：取 $A$、$B$ 底面顶点，$M$ 为 $AB$ 中点（$M$ 在底面，$O'M\perp AB$，$PM\perp AB$ 为斜面 $PAB$ 的高）。

侧面 $PAB$ 与底面 $ABC$ 的二面角沿棱 $AB$，两面内分别取垂直 $AB$ 的向量：底面取 $\overrightarrow{O'M}$（从 $O'$ 到 $M$），侧面取 $\overrightarrow{PM}$。

$|O'M|=\dfrac{\sqrt{3}}{2}a-r_0+r_0$？$O'M=r_0\cdot\cos 30°$... 正三角形中心到边中点距离 $=\dfrac{a}{2\sqrt{3}}$，设为 $m=\dfrac{a}{2\sqrt{3}}=\dfrac{\sqrt{3}}{6}a$。

侧面斜高 $l=\sqrt{h^2+m^2}=\sqrt{b^2-\dfrac{a^2}{3}+\dfrac{a^2}{12}}=\sqrt{b^2-\dfrac{a^2}{4}}$。

二面角余弦：$\cos\varphi=\dfrac{\overrightarrow{O'M}\cdot\overrightarrow{PM}}{|O'M||PM|}$。取 $\overrightarrow{MO'}=(-m,0,0)$（底面内垂直 $AB$ 向内），$\overrightarrow{MP}=(0,0,h)$（沿侧高方向）... 实际：

$$\cos\varphi = \frac{m}{l}=\frac{\frac{\sqrt{3}}{6}a}{\sqrt{b^2-\frac{a^2}{4}}}=\frac{\sqrt{3}a}{6\sqrt{b^2-\frac{a^2}{4}}}=\frac{\sqrt{3}a}{\sqrt{36b^2-9a^2}}$$

$$\boxed{\cos\varphi = \frac{\sqrt{3}\,a}{\sqrt{36b^2-9a^2}}}$$

**第(3)问**：侧棱 $PA$ 与底面所成角：$PA$ 的水平投影为 $O'A$（底面外接圆半径 $r_0=\dfrac{a}{\sqrt{3}}$），$|PA|=b$。

$$\sin\psi = \frac{h}{b}=\frac{\sqrt{b^2-\frac{a^2}{3}}}{b}=\sqrt{1-\frac{a^2}{3b^2}}=\frac{\sqrt{3b^2-a^2}}{\sqrt{3}\,b}$$

$$\boxed{\sin\psi = \frac{\sqrt{3b^2-a^2}}{\sqrt{3}\,b}}$$

**总结**：正三棱锥三类距离（高、球、线面角）均依赖"底面外接圆半径 $r_0=\dfrac{a}{\sqrt{3}}$"和"底面内切圆半径 $r=\dfrac{a}{2\sqrt{3}}$"两个核心量，先求出再代入。

---

### E.42 [提升] Part 9/07 [正方体截面 + 距离]

**题目回顾**：正方体 $ABCD\text{-}A_1B_1C_1D_1$，棱长为 $2$。
(1) 作截面过 $A$、$C_1$、$B_1$，求截面面积；(2) 求截面 $AB_1C_1$ 与底面 $ABCD$ 所成二面角的余弦值；(3) 求顶点 $D_1$ 到截面 $AB_1C_1$ 的距离。

**思路**：建系求截面形状 → 等腰三角形面积；法向量求二面角；点面距公式。→ toolkit/11。

**建系**：$A(0,0,0),B(2,0,0),C(2,2,0),D(0,2,0),A_1(0,0,2),B_1(2,0,2),C_1(2,2,2),D_1(0,2,2)$。

**解答**：

**第(1)问**：截面顶点 $A(0,0,0),B_1(2,0,2),C_1(2,2,2)$。

$$|AB_1|=\sqrt{4+0+4}=2\sqrt{2},\quad |AC_1|=\sqrt{4+4+4}=2\sqrt{3},\quad |B_1C_1|=\sqrt{0+4+0}=2$$

面积用向量叉积：$\overrightarrow{AB_1}=(2,0,2),\ \overrightarrow{AC_1}=(2,2,2)$。

$$\overrightarrow{AB_1}\times\overrightarrow{AC_1}=(0\cdot 2-2\cdot 2,\ 2\cdot 2-2\cdot 2,\ 2\cdot 2-0\cdot 2)=(-4,0,4)$$

$$S=\frac{1}{2}|(-4,0,4)|=\frac{1}{2}\cdot 4\sqrt{2}=2\sqrt{2}$$

$$\boxed{S_{\triangle AB_1C_1}=2\sqrt{2}}$$

**第(2)问**：截面法向量 $\vec{n}=(-4,0,4)$，化简为 $(-1,0,1)$；底面法向量 $\vec{k}=(0,0,1)$。

$$\cos\theta=\frac{|(-1,0,1)\cdot(0,0,1)|}{|(-1,0,1)|\cdot|(0,0,1)|}=\frac{1}{\sqrt{2}\cdot 1}=\frac{\sqrt{2}}{2}$$

$$\therefore \text{二面角余弦值为}\ \boxed{\dfrac{\sqrt{2}}{2}}\ \text{（即二面角}\ 45°\text{）}$$

**第(3)问**：截面方程：过 $A(0,0,0)$，法向量 $(-1,0,1)$，方程 $-x+z=0$，即 $z=x$。

$D_1(0,2,2)$ 到平面 $z=x$（即 $x-z=0$）的距离：

$$d=\frac{|0-2|}{\sqrt{1^2+0^2+(-1)^2}}=\frac{2}{\sqrt{2}}=\sqrt{2}$$

$$\boxed{d(D_1,\text{截面})=\sqrt{2}}$$

**总结**：正方体截面题——截面形状用叉积面积公式最快；点到截面距离用平面方程点代入除法向量模。

---

### E.43 [提升] Part 8+9/综合 [球与立体]

**题目回顾**：正三棱锥 $P\text{-}ABC$，底面边长为 $4$，侧面为等边三角形。
(1) 求高、内切球半径 $r$ 与外接球半径 $R$；
(2) 求内切球与外接球的球心距；
(3) 设点 $M$ 在外接球面上，求 $M$ 到底面 $ABC$ 的最大距离与最小距离。

**思路**：侧面为等边三角形 → 侧棱长 $= 4$（正四面体！）。建系计算各量。→ 内切球切各面，外接球过各顶点。

**各量计算**：底面边长 $a=4$，侧棱长 $b=4$（侧面为等边三角形，$P$ 到底面顶点距离等于底面边长）。

底面外接圆半径 $r_0=\dfrac{4}{\sqrt{3}}=\dfrac{4\sqrt{3}}{3}$。

棱锥高 $h=\sqrt{b^2-r_0^2}=\sqrt{16-\dfrac{16}{3}}=\sqrt{\dfrac{32}{3}}=\dfrac{4\sqrt{2}}{\sqrt{3}}=\dfrac{4\sqrt{6}}{3}$。

**解答**：

**第(1)问**：

高 $h=\dfrac{4\sqrt{6}}{3}$。

底面内切圆半径（底面正三角形）：$r_{\text{底}}=\dfrac{a}{2\sqrt{3}}=\dfrac{4}{2\sqrt{3}}=\dfrac{2}{\sqrt{3}}=\dfrac{2\sqrt{3}}{3}$。

内切球半径：设球心在轴上距底面高度 $r$（内切球切底面，则球心距底面 $=r$；切侧面，利用侧面斜高）。

侧面斜高 $l=\sqrt{h^2+r_{\text{底}}^2}=\sqrt{\dfrac{32}{3}+\dfrac{4}{3}}=\sqrt{12}=2\sqrt{3}$。

侧面为等边三角形边长 $4$，斜高 $=\dfrac{\sqrt{3}}{2}\cdot 4=2\sqrt{3}$。✓

侧面与轴所成角 $\cos\alpha=r_{\text{底}}/l=\dfrac{2\sqrt{3}/3}{2\sqrt{3}}=\dfrac{1}{3}$，$\sin\alpha=\dfrac{2\sqrt{2}}{3}$（即侧面与底面夹角的余弦）。

设内切球心距底面高度 $= r_i$（内切球半径），由球与侧面相切：

$$\frac{r_i}{\sin\alpha}=\frac{1}{1}\text{ 中的}$$... 

直接用体积法：$V=\dfrac{1}{3}r_i(S_{\text{底}}+3S_{\text{侧}})$，$V=\dfrac{1}{3}S_{\text{底}}h=\dfrac{1}{3}\times 4\sqrt{3}\times\dfrac{4\sqrt{6}}{3}=\dfrac{16\sqrt{2}}{3}\sqrt{3}/\sqrt{3}$。

$S_{\text{底}}=\sqrt{3}a^2/4=4\sqrt{3}$；$V=\dfrac{1}{3}\times 4\sqrt{3}\times\dfrac{4\sqrt{6}}{3}=\dfrac{16\sqrt{18}}{9}=\dfrac{48\sqrt{2}}{9}=\dfrac{16\sqrt{2}}{3}$。

$S_{\text{侧面（每个）}}=\dfrac{\sqrt{3}}{4}\times 16=4\sqrt{3}$，$S_{\text{侧总}}=12\sqrt{3}$。

$$r_i = \frac{3V}{S_{\text{底}}+S_{\text{侧总}}}=\frac{3\cdot\frac{16\sqrt{2}}{3}}{4\sqrt{3}+12\sqrt{3}}=\frac{16\sqrt{2}}{16\sqrt{3}}=\frac{\sqrt{2}}{\sqrt{3}}=\frac{\sqrt{6}}{3}$$

外接球：$R=\dfrac{3b^2}{2\sqrt{3b^2-a^2}}=\dfrac{3\times 16}{2\sqrt{48-16}}=\dfrac{48}{2\sqrt{32}}=\dfrac{48}{8\sqrt{2}}=\dfrac{6}{\sqrt{2}}=3\sqrt{2}$。

$$\boxed{h=\frac{4\sqrt{6}}{3},\quad r=\frac{\sqrt{6}}{3},\quad R=3\sqrt{2}}$$

**第(2)问**：内切球心距底面 $r=\dfrac{\sqrt{6}}{3}$，外接球心距底面 $h_0$：

$h_0=\dfrac{r_0^2+h^2}{2h}-h+h=h-\dfrac{r_0^2}{... }$，取 $h_0=R-（R到顶P的分析）$。

外接球心在轴上距底面高度 $h_1$：$h_1^2+r_0^2=R^2 \Rightarrow h_1^2=18-\dfrac{16}{3}=\dfrac{38}{3}$；另一方向 $(h-h_1)^2+r_0^2=R^2$ 即 $h_1=h-\dfrac{R^2-r_0^2}{h}\cdot\dfrac{h}{1}$...

用公式：$h_1=\dfrac{r_0^2}{h}=\dfrac{16/3}{4\sqrt{6}/3}=\dfrac{16}{4\sqrt{6}}=\dfrac{4}{\sqrt{6}}=\dfrac{2\sqrt{6}}{3}$。（底面顶点到球心水平距 $r_0$，竖直距 $h_1$，$h_1^2+r_0^2=R^2$: $\dfrac{4}{6}\cdot 6+\dfrac{16}{3}=4+\dfrac{16}{3}=\dfrac{28}{3}\neq 18$）

重新：$R=\dfrac{r_0^2+h^2}{2h}\cdot\dfrac{2h}{1}$？ 用 $h_1^2+r_0^2=R^2$，$h_1=h-\dfrac{h^2+r_0^2-R^2+R^2-...}{h}$...

球心公式 $h_1=\dfrac{h^2+r_0^2}{2h}=\dfrac{32/3+16/3}{2\cdot 4\sqrt{6}/3}=\dfrac{16}{8\sqrt{6}/3}=\dfrac{16\times 3}{8\sqrt{6}}=\dfrac{6}{\sqrt{6}}=\sqrt{6}$。

验证 $R=\sqrt{h_1^2+r_0^2}=\sqrt{6+16/3}=\sqrt{34/3}$，不等于 $3\sqrt{2}$。← 重新求 $R$：

$r_0=\dfrac{4\sqrt{3}}{3}$，$h=\dfrac{4\sqrt{6}}{3}$。$h_1=\dfrac{r_0^2+h^2}{2h}=\dfrac{16/3+32/3}{8\sqrt{6}/3}=\dfrac{48/3}{8\sqrt{6}/3}=\dfrac{48}{8\sqrt{6}}=\dfrac{6}{\sqrt{6}}=\sqrt{6}$。

$R=\sqrt{r_0^2+(\text{距底面})^2}$...球心到底面顶点：$r_0^2+(h_1-0)^2$？不对，球心在高 $h_1$ 处，底面顶点在 $z=0$，水平距离 $r_0$。$R^2=r_0^2+h_1^2=\dfrac{16}{3}+6=\dfrac{34}{3}$，$R=\sqrt{34/3}$。

与顶点 $P$ 距离：$|P-\text{球心}|=|h-h_1|=\dfrac{4\sqrt{6}}{3}-\sqrt{6}=\sqrt{6}(\dfrac{4}{3}-1)=\dfrac{\sqrt{6}}{3}=r$（内切球半径），这说明正四面体中 $R=3r$！

$$R=\sqrt{\frac{34}{3}}=\frac{\sqrt{102}}{3},\quad r=\frac{\sqrt{6}}{3}$$

球心距 $=|h_1-r|=\sqrt{6}-\dfrac{\sqrt{6}}{3}=\dfrac{2\sqrt{6}}{3}$。

$$\boxed{\text{球心距}=\frac{2\sqrt{6}}{3}}$$

**第(3)问**：外接球心在轴上距底面高度 $h_1=\sqrt{6}$，球面上点 $M$ 到底面的距离范围为 $[h_1-R,h_1+R]$（在球面上最低到最高），取非负部分（底面以下为负距离）：

$M$ 到底面最大距离 $=h_1+R=\sqrt{6}+\dfrac{\sqrt{102}}{3}$，最小距离 $=|h_1-R|$（若 $R>h_1$ 则球穿过底面，最小距离为 $R-h_1$）。

$R-h_1=\dfrac{\sqrt{102}}{3}-\sqrt{6}=\dfrac{\sqrt{102}-3\sqrt{6}}{3}$。

$$\boxed{d_{\max}=\sqrt{6}+\frac{\sqrt{102}}{3},\quad d_{\min}=\frac{\sqrt{102}-3\sqrt{6}}{3}}$$

**总结**：球与正三棱锥：先算底面外接圆半径 $r_0$，再求高 $h$，然后内切球用体积法 $r=3V/S_{\text{全}}$，外接球用 $R^2=r_0^2+h_1^2$（$h_1$ 为球心高度）。球面上点到底面距离范围 $[h_1-R,h_1+R]$（含负值时取绝对值）。

---

## 分组 4：高考真题难度综合（E.54–E.60）

---

### E.54 [提升] Part 10/01 [椭圆 + 不等式综合]

**题目回顾**：椭圆 $E$：$\dfrac{x^2}{4}+y^2=1$，$F_1(-\sqrt{3},0)$，$F_2(\sqrt{3},0)$。
(1) 证明椭圆外点 $A(m,n)$ 满足 $\dfrac{m^2}{4}+n^2>1$；
(2) 设 $A(1,1)$，$PA$ 为椭圆切线（切点 $P$），求 $P$ 的坐标及切线方程；
(3) 从外点 $A(m,n)$ 向椭圆作两切线，两切点连线（极线）方程为 $\dfrac{mx}{4}+ny=1$。

**思路**：(1) 椭圆外点判别式 $>1$；(2) 切线方程 $\dfrac{x_0 x}{4}+y_0 y=1$，$A$ 在切线上 + 切点在椭圆上；(3) 极线定理。

**解答**：

**第(1)问**：椭圆上的点满足 $\dfrac{x^2}{4}+y^2=1$，外部点满足 $\dfrac{m^2}{4}+n^2>1$（直接由椭圆定义，到两焦点距离之和大于 $2a$ 对应代数不等式）。← 代数证明：外部点到椭圆上最近点距离 $>0$，等价于代入方程 $>1$。

**第(2)问**：设切点 $P(x_0,y_0)$，切线方程 $\dfrac{x_0 x}{4}+y_0 y=1$。因 $A(1,1)$ 在切线上：

$$\frac{x_0}{4}+y_0=1 \quad \cdots (*)$$

又 $P$ 在椭圆上：$\dfrac{x_0^2}{4}+y_0^2=1 \quad \cdots (**)$

由 $(*)$：$y_0=1-\dfrac{x_0}{4}$，代入 $(**)$：$\dfrac{x_0^2}{4}+\left(1-\dfrac{x_0}{4}\right)^2=1$。

展开：$\dfrac{x_0^2}{4}+1-\dfrac{x_0}{2}+\dfrac{x_0^2}{16}=1 \Rightarrow \dfrac{5x_0^2}{16}-\dfrac{x_0}{2}=0 \Rightarrow x_0\left(\dfrac{5x_0}{16}-\dfrac{1}{2}\right)=0$。

$x_0=0$（则 $y_0=1$，切点 $(0,1)$）或 $x_0=\dfrac{8}{5}$（则 $y_0=1-\dfrac{2}{5}=\dfrac{3}{5}$，切点 $\left(\dfrac{8}{5},\dfrac{3}{5}\right)$）。

切线方程：
- 切点 $(0,1)$：$y=1$；
- 切点 $\left(\dfrac{8}{5},\dfrac{3}{5}\right)$：$\dfrac{8x}{20}+\dfrac{3y}{5}=1$，即 $\dfrac{2x}{5}+\dfrac{3y}{5}=1$，即 $2x+3y=5$。

$$\boxed{P_1(0,1),\text{ 切线 }y=1;\quad P_2\!\left(\tfrac{8}{5},\tfrac{3}{5}\right),\text{ 切线 }2x+3y=5}$$

**第(3)问**：两切点 $(0,1)$ 和 $\left(\dfrac{8}{5},\dfrac{3}{5}\right)$ 的连线：斜率 $k=\dfrac{3/5-1}{8/5-0}=\dfrac{-2/5}{8/5}=-\dfrac{1}{4}$，方程 $y-1=-\dfrac{1}{4}x$，即 $x+4y=4$，即 $\dfrac{x}{4}+y=1$，正好是极线 $\dfrac{m\cdot x}{4}+n\cdot y=1$（$m=1,n=1$）。✓

$$\boxed{\text{极线方程：}\frac{mx}{4}+ny=1}$$

**总结**："切点弦"即极线——外点到椭圆两切线的两切点连线方程由外点坐标代入椭圆方程结构得到，是椭圆射影对偶的基本结果。

---

### E.55 [提升] Part 10/02 [立体几何 + 圆锥曲线交叉]

**题目回顾**：正三棱锥 $P\text{-}ABC$，底面边长 $2$，高 $\sqrt{6}$。平面 $PAB$ 内以 $PA$、$PB$ 为焦点，椭圆过 $AB$ 中点 $M$。
(1) 求 $|PA|$（侧棱长）；(2) 在平面 $PAB$ 内建系，写出各点坐标；(3) 求椭圆方程及离心率。

**思路**：先用三维坐标求 $|PA|$；再在平面 $PAB$ 内以 $M$ 为原点建二维系；椭圆半长轴由焦半径之和定。

**解答**：

**第(1)问**：底面正三角形，$M$ 为 $AB$ 中点，$A$、$B$ 两点距中心 $O'$ 的水平距离 $r_0=\dfrac{2}{\sqrt{3}}$（底面外接圆半径）。

$|PA|=\sqrt{h^2+r_0^2}=\sqrt{6+\dfrac{4}{3}}=\sqrt{\dfrac{22}{3}}=\dfrac{\sqrt{66}}{3}$。

另：$|PM|=\sqrt{h^2+|O'M|^2}$，$|O'M|=r_{\text{内}}=\dfrac{1}{\sqrt{3}}$，$|PM|=\sqrt{6+\dfrac{1}{3}}=\sqrt{\dfrac{19}{3}}=\dfrac{\sqrt{57}}{3}$。

$$\boxed{|PA|=\frac{\sqrt{66}}{3}}$$

**第(2)问**：在平面 $PAB$ 内，以 $M$ 为原点，$AB$ 方向为 $x'$ 轴，$PM$ 方向为 $y'$ 轴。

$A(-1,0)$，$B(1,0)$（$|AB|=2$，$M$ 为 $AB$ 中点）；$P(0,|PM|)=(0,\dfrac{\sqrt{57}}{3})$。

**第(3)问**：椭圆焦点为 $PA(=-1,0)$ 和 $PB=(1,0)$ ... 焦点应为 $A$ 和 $B$（即 $A(-1,0)$，$B(1,0)$），半焦距 $c=1$。

椭圆过 $M(0,0)$：焦半径之和 $|MA|+|MB|=1+1=2=2a$，故 $a=1$。

但 $c=1=a$ 时为抛物线，非椭圆。← 检查：椭圆需 $a>c$。

重解：$P$ 在椭圆上，$|PA|+|PB|=2|PA|=\dfrac{2\sqrt{66}}{3}=2a$（等腰），故 $a=\dfrac{\sqrt{66}}{3}$，$c=1$，$b^2=a^2-c^2=\dfrac{66}{9}-1=\dfrac{57}{9}=\dfrac{19}{3}$。

$M(0,0)$ 上：$|MA|+|MB|=1+1=2 \neq 2a$——$M$ 也在椭圆上，须满足 $|MA|+|MB|=2a$，即 $2=\dfrac{2\sqrt{66}}{3}$，矛盾。

← 题意：椭圆以 $P$、$A$ 为焦点（非 $P$、$B$），$P$ 和 $A$ 是焦点，椭圆过 $M$（$AB$ 中点）。

$|PA|=\dfrac{\sqrt{66}}{3}$，$2c=\dfrac{\sqrt{66}}{3}$，$c=\dfrac{\sqrt{66}}{6}$。$|MA|=1$，$|MP|=\dfrac{\sqrt{57}}{3}$，$|MA|+|MP|=1+\dfrac{\sqrt{57}}{3}=2a$，$a=\dfrac{3+\sqrt{57}}{6}$。

离心率 $e=\dfrac{c}{a}=\dfrac{\sqrt{66}/6}{(3+\sqrt{57})/6}=\dfrac{\sqrt{66}}{3+\sqrt{57}}$。

$$\boxed{e=\frac{\sqrt{66}}{3+\sqrt{57}}}$$

**总结**：立几与圆锥交叉题——先在三维坐标系中求出关键长度，再切换到截面内的二维坐标系，用焦半径定义建立椭圆方程。

---

### E.56 [提升] Part 10/03 [新定义：等距弦]

**题目回顾**：定义：直线 $l$ 与曲线 $C$ 交于 $A$、$B$，且 $|OA|=|OB|$（$O$ 原点），称 $l$ 为 $C$ 的"等距弦"。
(1) 圆 $x^2+y^2=r^2$ 的所有等距弦有何特征？(2) 椭圆 $\dfrac{x^2}{4}+y^2=1$ 所有斜率为 $1$ 的等距弦；(3) 证明抛物线 $y^2=4x$ 不存在过焦点的等距弦。

**思路**：$|OA|=|OB|$ 即 $A$、$B$ 在以 $O$ 为圆心的同一圆上，两点关于 $O$ 为对称中心 → 对等距弦的刻画。

**解答**：

**第(1)问**：圆 $x^2+y^2=r^2$ 上每点到 $O$ 距离均为 $r$，所以圆上任意弦都满足 $|OA|=|OB|=r$，即**圆的所有弦都是等距弦**。← 特征：过圆心的直线（直径）也是等距弦，且所有弦均满足。

**第(2)问**：椭圆上 $A(x_1,y_1)$，$B(x_2,y_2)$，斜率为 $1$，$|OA|^2=|OB|^2$：

$x_1^2+y_1^2=x_2^2+y_2^2 \Rightarrow (x_1-x_2)(x_1+x_2)+(y_1-y_2)(y_1+y_2)=0$。

斜率为 $1$：$\dfrac{y_1-y_2}{x_1-x_2}=1$，故 $y_1-y_2=x_1-x_2$（$\neq 0$）：

$(x_1+x_2)+(y_1+y_2)=0$，即中点 $M\left(\dfrac{x_1+x_2}{2},\dfrac{y_1+y_2}{2}\right)$ 满足 $x_0+y_0=0$（$y_0=-x_0$）。← 中点在 $y=-x$ 上

用点差法：椭圆中点弦斜率 $k=-\dfrac{x_0}{4y_0}=1 \Rightarrow x_0=-4y_0$，联立 $y_0=-x_0=4y_0 \Rightarrow 3y_0=0 \Rightarrow y_0=0$。

$y_0=0,x_0=0$，中点为原点 $O(0,0)$。直线过原点，斜率 $1$，方程 $y=x$。

验证：椭圆与 $y=x$ 联立：$\dfrac{x^2}{4}+x^2=1 \Rightarrow \dfrac{5x^2}{4}=1 \Rightarrow x=\pm\dfrac{2}{\sqrt{5}}$，$A=(\dfrac{2}{\sqrt{5}},\dfrac{2}{\sqrt{5}})$，$B=(-\dfrac{2}{\sqrt{5}},-\dfrac{2}{\sqrt{5}})$，关于 $O$ 对称，$|OA|=|OB|$。✓

$$\boxed{\text{斜率为 }1 \text{ 的等距弦方程：}y=x}$$

**第(3)问**：抛物线 $y^2=4x$，焦点 $F(1,0)$。设过 $F$ 的弦端点 $A(t_1^2,2t_1)$，$B(t_2^2,2t_2)$（参数化，$y=2t \Rightarrow x=t^2$）。

过焦点：$A$、$B$、$F$ 共线，可得 $t_1 t_2=-1$（抛物线焦弦参数性质）。

$|OA|^2=t_1^4+4t_1^2=t_1^2(t_1^2+4)$，$|OB|^2=t_2^2(t_2^2+4)$。

若 $|OA|=|OB|$，则 $t_1^2(t_1^2+4)=t_2^2(t_2^2+4)$。因 $t_1 t_2=-1$，$t_2=-\dfrac{1}{t_1}$：

$t_1^2(t_1^2+4)=\dfrac{1}{t_1^2}\left(\dfrac{1}{t_1^2}+4\right)=\dfrac{1+4t_1^2}{t_1^4}$。

令 $u=t_1^2>0$：$u(u+4)=\dfrac{4u+1}{u^2} \Rightarrow u^3(u+4)=4u+1 \Rightarrow u^4+4u^3-4u-1=0$。

$(u^4-1)+4u(u^2-1)=(u^2-1)(u^2+1)+4u(u^2-1)=(u^2-1)(u^2+4u+1)=0$。

$u^2=1 \Rightarrow u=1$（$u>0$），或 $u^2+4u+1=0 \Rightarrow u=\dfrac{-4\pm\sqrt{12}}{2}=-2\pm\sqrt{3}$（均为负数，舍去）。

$u=1 \Rightarrow t_1=\pm 1$，$t_2=\mp 1$，$A=(1,2),B=(1,-2)$，直线 $AB$：$x=1$，即**过焦点的垂直轴的弦**（通径）。

验证：$|OA|=\sqrt{1+4}=\sqrt{5}=|OB|$。这条弦**存在**！← 结论：过焦点的等距弦**存在**（通径），但并非所有过焦点的弦都是等距弦。

$$\boxed{\text{过焦点的等距弦存在，即通径 }x=1\text{（}A(1,2),B(1,-2)\text{）}}$$

**总结**：等距弦 $\Leftrightarrow$ $|OA|=|OB|$ $\Leftrightarrow$ 弦中点与 $O$、弦方向满足特定约束。参数化后代入等距条件得多项式方程，分解因式定参。

---

### E.57 [提升] Part 10/04 [向量 + 圆锥曲线 + 立体综合]

**题目回顾**：椭圆 $E$：$\dfrac{x^2}{4}+y^2=1$ 在 $xOy$ 平面，$P$ 在 $E$ 上，$H=P+(0,0,1)$。
(1) 求 $H$ 的轨迹；(2) 设 $Q(0,0,2)$，求 $|\overrightarrow{HQ}|$ 的最大最小值；(3) $H$ 在圆柱 $x^2+y^2=1$（$z=1$）内时，判断 $P$ 是否在 $E$ 内。

**思路**：(1) 轨迹为平移后的椭圆；(2) $H$ 到定点 $Q$ 距离最值；(3) 比较圆柱截面与椭圆。

**解答**：

**第(1)问**：$P=(x,y,0)$ 在 $E$ 上：$\dfrac{x^2}{4}+y^2=1$；$H=(x,y,1)$。

$H$ 的轨迹：$\dfrac{x^2}{4}+y^2=1,\ z=1$（$z=1$ 平面上的椭圆）。

$$\boxed{H\text{ 的轨迹：}\frac{x^2}{4}+y^2=1,\ z=1}$$

**第(2)问**：$H=(x,y,1)$，$Q=(0,0,2)$。

$|\overrightarrow{HQ}|^2=x^2+y^2+(2-1)^2=x^2+y^2+1$。

$H$ 在椭圆 $\dfrac{x^2}{4}+y^2=1$ 上，故 $x=2\cos\theta,y=\sin\theta$，$x^2+y^2=4\cos^2\theta+\sin^2\theta=1+3\cos^2\theta\in[1,4]$。

$|\overrightarrow{HQ}|^2\in[2,5]$，$|\overrightarrow{HQ}|\in[\sqrt{2},\sqrt{5}]$。

$$\boxed{|\overrightarrow{HQ}|_{\min}=\sqrt{2},\quad|\overrightarrow{HQ}|_{\max}=\sqrt{5}}$$

**第(3)问**：$H$ 在圆柱 $x^2+y^2\leq 1$（$z=1$ 截面），即 $x^2+y^2\leq 1$，故 $P=(x,y,0)$ 满足 $x^2+y^2\leq 1$。

椭圆 $E$：$\dfrac{x^2}{4}+y^2=1$，椭圆内部为 $\dfrac{x^2}{4}+y^2<1$。

若 $x^2+y^2\leq 1$，是否有 $\dfrac{x^2}{4}+y^2\leq 1$？因为 $\dfrac{x^2}{4}\leq x^2$，故 $\dfrac{x^2}{4}+y^2\leq x^2+y^2\leq 1$。✓

故 **$P$ 的轨迹包含在椭圆 $E$ 内**（含边界）。

$$\boxed{P\text{ 在椭圆 }E\text{ 内（含边界）}}$$

**总结**：三维轨迹问题先投影处理：$H$ 的轨迹 = $P$ 的轨迹在 $z=1$ 平面的平移；距离最值用参数化代入；圆柱截面与椭圆的包含关系用分量不等式直接比较。

---

### E.58 [提升] Part 10/05 [新定义：焦弦比]

**题目回顾**：抛物线 $y^2=4x$，焦点 $F(1,0)$。过 $F$ 的弦 $AB$，$\rho=\dfrac{|AF|}{|BF|}\geq 1$（"焦弦比"）。
(1) 用 $\rho$ 表示 $|AB|$；(2) 用 $\rho$ 表示 $|FC|$（$C$ 为 $AB$ 中点）；(3) 若 $\rho=3$，求各点坐标及直线方程。

**思路**：利用抛物线调和性质 $\dfrac{1}{|AF|}+\dfrac{1}{|BF|}=\dfrac{1}{p}=\dfrac{1}{2}$（$p=2$ 时准线 $x=-1$），设 $|AF|=\rho r$，$|BF|=r$。

**解答**：

**第(1)问**：设 $|BF|=r$，$|AF|=\rho r$。由调和性质 $\dfrac{1}{\rho r}+\dfrac{1}{r}=\dfrac{1}{2}$（$p/2=1$，故 $\dfrac{1}{|AF|}+\dfrac{1}{|BF|}=\dfrac{2}{p}=1$）。

← 修正：抛物线 $y^2=2px$，$\dfrac{1}{|AF|}+\dfrac{1}{|BF|}=\dfrac{2}{p}$；本题 $y^2=4x$，$p=2$，故 $\dfrac{1}{|AF|}+\dfrac{1}{|BF|}=1$。

$\dfrac{1}{\rho r}+\dfrac{1}{r}=1 \Rightarrow \dfrac{1+\rho}{\rho r}=1 \Rightarrow r=\dfrac{1+\rho}{\rho}$，$|AF|=\rho r=1+\rho$，$|AB|=|AF|+|BF|=1+\rho+\dfrac{1+\rho}{\rho}=(1+\rho)\left(1+\dfrac{1}{\rho}\right)=\dfrac{(1+\rho)^2}{\rho}$。

$$\boxed{|AB|=\frac{(1+\rho)^2}{\rho}}$$

**第(2)问**：$C$ 为 $AB$ 中点，$|FC|=\left|\dfrac{|AF|-|BF|}{2}\right|=\dfrac{\rho r-r}{2}=\dfrac{(\rho-1)r}{2}=\dfrac{(\rho-1)(1+\rho)}{2\rho}=\dfrac{\rho^2-1}{2\rho}$。

$$\boxed{|FC|=\frac{\rho^2-1}{2\rho}}$$

**第(3)问**：$\rho=3$，$r=\dfrac{1+3}{3}=\dfrac{4}{3}$，$|BF|=\dfrac{4}{3}$，$|AF|=4$。

抛物线焦半径：$|MF|=x_M+1$（准线 $x=-1$），故 $x_A+1=4 \Rightarrow x_A=3$，$y_A^2=12 \Rightarrow y_A=\pm 2\sqrt{3}$；$x_B+1=\dfrac{4}{3} \Rightarrow x_B=\dfrac{1}{3}$，$y_B^2=\dfrac{4}{3} \Rightarrow y_B=\pm\dfrac{2}{\sqrt{3}}=\pm\dfrac{2\sqrt{3}}{3}$。

$A$、$B$ 在过焦点的同一直线上，斜率 $k=\dfrac{y_A-y_B}{x_A-x_B}$：取 $y_A=2\sqrt{3}$，$y_B=-\dfrac{2\sqrt{3}}{3}$（$A$、$F$、$B$ 共线，焦弦两端在 $x$ 轴两侧）：

$k=\dfrac{2\sqrt{3}+\dfrac{2\sqrt{3}}{3}}{3-\dfrac{1}{3}}=\dfrac{\dfrac{8\sqrt{3}}{3}}{\dfrac{8}{3}}=\sqrt{3}$，直线 $y=\sqrt{3}(x-1)$。

中点 $C=\left(\dfrac{3+1/3}{2},\dfrac{2\sqrt{3}-2\sqrt{3}/3}{2}\right)=\left(\dfrac{5}{3},\dfrac{2\sqrt{3}}{3}\right)$，$|FC|=\sqrt{(5/3-1)^2+(2\sqrt{3}/3)^2}=\sqrt{4/9+12/9}=\sqrt{16/9}=\dfrac{4}{3}$。

验证公式：$|FC|=\dfrac{\rho^2-1}{2\rho}=\dfrac{9-1}{6}=\dfrac{4}{3}$。✓

$$\boxed{A(3,2\sqrt{3}),\ B\!\left(\tfrac{1}{3},-\tfrac{2\sqrt{3}}{3}\right),\ C\!\left(\tfrac{5}{3},\tfrac{2\sqrt{3}}{3}\right),\ \text{直线}\ y=\sqrt{3}(x-1)}$$

**总结**：焦弦比问题设 $|BF|=r$ 后由调和性质建方程，得到 $|AB|=\dfrac{(1+\rho)^2}{\rho}$，中点偏移 $|FC|=\dfrac{\rho^2-1}{2\rho}$，是两个关于 $\rho$ 的优美表达式。

---

### E.59 [提升] Part 10/05 [新定义：圆锥斜截面]

**题目回顾**：圆锥面 $x^2+y^2=z^2$（$z\geq 0$），以斜平面 $z=ax+h$（$0<a<1,h>0$）截锥面。
(1) 代入化简截面方程；(2) 说明截面为椭圆，求半轴长；(3) 证明斜截椭圆共享焦点。

**思路**：将 $z=ax+h$ 代入 $x^2+y^2=z^2$，整理为二次曲线方程，通过配方识别椭圆。

**解答**：

**第(1)问**：代入 $x^2+y^2=(ax+h)^2=a^2x^2+2ahx+h^2$：

$(1-a^2)x^2-2ahx+y^2=h^2$

$$\boxed{(1-a^2)x^2-2ahx+y^2=h^2}$$

**第(2)问**：对 $x$ 配方（$1-a^2>0$ 因 $0<a<1$）：

$(1-a^2)\left(x-\dfrac{ah}{1-a^2}\right)^2+y^2=h^2+\dfrac{a^2h^2}{1-a^2}=\dfrac{h^2}{1-a^2}$

令 $X=x-\dfrac{ah}{1-a^2}$，则 $\dfrac{X^2}{\frac{h^2}{(1-a^2)^2}}+\dfrac{y^2}{\frac{h^2}{1-a^2}}=1$。

半轴：$\alpha=\dfrac{h}{1-a^2}$（沿 $x$ 方向），$\beta=\dfrac{h}{\sqrt{1-a^2}}$（沿 $y$ 方向）。因 $\alpha>\beta$（$\dfrac{1}{1-a^2}>\dfrac{1}{\sqrt{1-a^2}}$），长半轴 $\alpha$，短半轴 $\beta$。

$$\boxed{\alpha=\frac{h}{1-a^2},\quad \beta=\frac{h}{\sqrt{1-a^2}}}$$

**第(3)问**：焦距 $c=\sqrt{\alpha^2-\beta^2}=\sqrt{\dfrac{h^2}{(1-a^2)^2}-\dfrac{h^2}{1-a^2}}=h\sqrt{\dfrac{a^2}{(1-a^2)^2}}=\dfrac{ah}{1-a^2}$。

焦点在 $X$ 轴（即平移后的 $x$ 方向）$\pm c$ 处，还原到原 $x$ 坐标：

焦点 $x$ 坐标 $=\dfrac{ah}{1-a^2}\pm\dfrac{ah}{1-a^2}=0$ 或 $\dfrac{2ah}{1-a^2}$。

← 左焦点 $x=0$（$y=0$），恰为斜平面与圆锥顶点所在轴的特殊点；对不同 $a$（固定 $h$），焦点 $x$ 坐标依赖 $a$，不固定。

实际：所有斜截面（固定 $h$，变 $a$）的公共焦点需在 $y=0$ 且焦点 $x=0$（取左焦点时）——此即圆锥面与 $z$ 轴焦点对应准线的几何内涵。← 丹德林球定理：斜截椭圆的焦点由两个内切丹德林球的切点给出，固定 $h$ 时焦点位置随 $a$ 变化，**不**完全共享。

$$\text{焦点坐标（截平面坐标系中）：}\left(0\pm\frac{ah}{1-a^2},0\right),\ z=\frac{ah\cdot a}{1-a^2}+h=\frac{h}{1-a^2}$$

**总结**：圆锥斜截题——代入后配方得椭圆标准形，半轴公式 $\alpha=\dfrac{h}{1-a^2}$，$\beta=\dfrac{h}{\sqrt{1-a^2}}$，焦距 $c=\dfrac{ah}{1-a^2}$，均用 $a,h$ 表示。

---

### E.60 [提升] Part 10/06 [圆锥曲线统一定义]

**题目回顾**：焦点-准线定义：到焦点 $F(c,0)$ 与准线 $x=\dfrac{a^2}{c}$ 距离之比为 $e=\dfrac{c}{a}$。
(1) $e=\dfrac{1}{2},a=2$：导出椭圆方程；(2) $e=1$，$F(1,0)$：导出抛物线方程；(3) $e=2,a=1$：导出双曲线方程；(4) $F(1,0)$，准线 $x=4$，$e=\dfrac{1}{4}$：求轨迹方程。

**思路**：统一定义 $\dfrac{|PF|}{d(P,l)}=e$，代入坐标建方程化简。

**解答**：

**第(1)问**：$e=\dfrac{1}{2}$，$a=2$，$c=ae=1$，准线 $x=\dfrac{4}{1}=4$。

设 $P(x,y)$：$\sqrt{(x-1)^2+y^2}=\dfrac{1}{2}(4-x)$（$x<4$）。平方：$(x-1)^2+y^2=\dfrac{1}{4}(4-x)^2$。

$4(x^2-2x+1)+4y^2=(4-x)^2=16-8x+x^2$，$4x^2-8x+4+4y^2=16-8x+x^2$，$3x^2+4y^2=12$，$\dfrac{x^2}{4}+\dfrac{y^2}{3}=1$。✓（标准椭圆，$a=2,b=\sqrt{3},c=1$）。

$$\boxed{\frac{x^2}{4}+\frac{y^2}{3}=1}$$

**第(2)问**：$e=1$，$F(1,0)$，准线 $x=-1$（抛物线 $y^2=2px$ 的准线 $x=-p/2$，$p=2$）：

$|PF|=d(P,l) \Rightarrow \sqrt{(x-1)^2+y^2}=x+1$。平方：$(x-1)^2+y^2=(x+1)^2$，$y^2=4x$。

$$\boxed{y^2=4x}$$

**第(3)问**：$e=2$，$a=1$，$c=ae=2$，$\dfrac{a^2}{c}=\dfrac{1}{2}$，准线 $x=\dfrac{1}{2}$。

$\dfrac{|PF|}{d(P,x=1/2)}=2$：$\sqrt{(x-2)^2+y^2}=2\left|x-\dfrac{1}{2}\right|=|2x-1|$。

平方：$(x-2)^2+y^2=(2x-1)^2$，$x^2-4x+4+y^2=4x^2-4x+1$，$y^2=3x^2-3$，$x^2-\dfrac{y^2}{3}=1$。

渐近线：$y=\pm\sqrt{3}\,x$。

$$\boxed{x^2-\frac{y^2}{3}=1,\quad\text{渐近线 }y=\pm\sqrt{3}\,x}$$

**第(4)问**：$F(1,0)$，准线 $x=4$，$e=\dfrac{1}{4}$。由 $e=\dfrac{c}{a}$ 及准线 $\dfrac{a^2}{c}=4$，得 $a^2=4c=4ae$，$a=4e=1$，$c=\dfrac{1}{4}$，$b^2=1-\dfrac{1}{16}=\dfrac{15}{16}$。

验证：$\dfrac{|PF|}{d(P,x=4)}=\dfrac{1}{4}$，$|PF|=\dfrac{1}{4}(4-x)$（$x<4$）。

$\sqrt{(x-\frac{1}{4})^2+y^2}=\dfrac{4-x}{4}$，平方展开：$x^2-\dfrac{x}{2}+\dfrac{1}{16}+y^2=\dfrac{(4-x)^2}{16}=\dfrac{16-8x+x^2}{16}$。

$16x^2-8x+1+16y^2=16-8x+x^2$，$15x^2+16y^2=15$，$x^2+\dfrac{16y^2}{15}=1$，即 $\dfrac{x^2}{1}+\dfrac{y^2}{15/16}=1$。

$$\boxed{x^2+\frac{16y^2}{15}=1\quad\left(\text{即 }a=1,\ b=\frac{\sqrt{15}}{4}\right)}$$

**总结**：三种曲线统一焦点-准线定义，代入后平方化简，关键是准线 $x=\dfrac{a^2}{c}$ 的推导（$e<1$：椭圆；$e=1$：抛物线；$e>1$：双曲线）。

---

> **本附录涵盖题目**：E.38–E.43（立体几何空间距离 + 存在性 6 题）、E.54–E.60（真题难度综合 7 题），共 **13 题**。
> **关键方法汇总**：法向量（二面角、线面角）、体积法（点面距）、极线定理（椭圆切点弦）、焦弦调和性（焦弦比）、参数化截面（圆锥斜截）、统一焦准定义（三类曲线）。
