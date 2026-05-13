# 向量的物理与综合应用

> **一例速记**：  
> 力的合成（平行四边形定则）：$\vec{F} = \vec{F_1} + \vec{F_2}$，$|\vec{F}|^2 = |\vec{F_1}|^2 + 2\vec{F_1}\cdot\vec{F_2} + |\vec{F_2}|^2$。  
> 速度合成：实际速度 $\vec{v} = \vec{v_{\text{自身}}} + \vec{v_{\text{媒介}}}$（向量加法）。  
> 几何最短距离：点到直线的最短距离 $= $ 垂线段长；用向量投影公式 $d = \dfrac{|\vec{AP} \times \vec{d}|}{|\vec{d}|}$。  
> 向量不等式：$|\vec{a} + \vec{b}| \leq |\vec{a}| + |\vec{b}|$（三角不等式）。

---

## 一、向量在物理中的应用

### 1. 力的合成与分解

在物理中，力是向量：既有大小（单位 N），又有方向。多个力同时作用于一个物体时，可以用向量加法求**合力**。

**平行四边形定则**：以 $\vec{F_1}$、$\vec{F_2}$ 为邻边作平行四边形，对角线即为合力 $\vec{F} = \vec{F_1} + \vec{F_2}$。

**合力模长公式**：设 $\vec{F_1}$ 与 $\vec{F_2}$ 的夹角为 $\theta$，则：

$$|\vec{F}|^2 = |\vec{F_1}|^2 + 2\vec{F_1}\cdot\vec{F_2} + |\vec{F_2}|^2 = |\vec{F_1}|^2 + 2|\vec{F_1}||\vec{F_2}|\cos\theta + |\vec{F_2}|^2$$

**特殊情形**：
- $\theta = 0°$（同向）：$|\vec{F}| = |\vec{F_1}| + |\vec{F_2}|$（合力最大）
- $\theta = 90°$（垂直）：$|\vec{F}| = \sqrt{|\vec{F_1}|^2 + |\vec{F_2}|^2}$（勾股定理）
- $\theta = 180°$（反向）：$|\vec{F}| = \bigl||\vec{F_1}| - |\vec{F_2}|\bigr|$（合力最小）

**分解**与合成互为逆运算：已知合力 $\vec{F}$，按某两个方向分解，得分力 $\vec{F_1}, \vec{F_2}$。

### 2. 速度的合成

船在水流中行驶时，船的实际速度由两部分叠加：

$$\vec{v} = \vec{v_{\text{船}}} + \vec{v_{\text{水}}}$$

其中 $\vec{v_{\text{船}}}$ 是船相对于水的速度，$\vec{v_{\text{水}}}$ 是水流速度，$\vec{v}$ 是船相对于地面的实际速度。

这是**伽利略速度合成**：在经典力学范围内，速度是向量，直接做向量加法。

**类似场景**：飞机受侧风影响（合速度 = 机速 + 风速），人在行驶的列车上走动等。

### 3. 位移的叠加

多次位移的总效果等于各位移向量之和：

$$\vec{s} = \vec{s_1} + \vec{s_2} + \cdots + \vec{s_n}$$

与路径无关，只看始末位置——这正是"向量只关心方向和大小、不关心起点"的体现。

**重要提醒**：物理量用向量时务必带单位（N、m/s、m 等），高中数学题目通常省略单位，但实际物理计算不能忽略。

---

## 二、向量在几何中的综合应用

### 1. 证明几何定理

向量方法常用于证明经典几何定理，优势在于：**不依赖图形直觉，代数推导严谨**。

**选基底**是向量证明几何的核心技巧：选取图形中合适的向量 $\vec{a}, \vec{b}$ 作为基底，将其他所有向量用 $\vec{a}, \vec{b}$ 的线性组合表示，再利用向量运算规则得出结论。

**常用关系**：
- 中点 $M$ 是 $AB$ 中点：$\vec{OM} = \dfrac{1}{2}(\vec{OA} + \vec{OB})$
- 平行：$\vec{PQ} = \lambda \vec{RS}$
- 垂直：$\vec{PQ} \cdot \vec{RS} = 0$
- 共线：$\vec{OP} = s\vec{OA} + (1-s)\vec{OB}$（$O$ 不在直线 $AB$ 上）

### 2. 解几何最值问题

**点到直线的最短距离**：

过点 $A$ 作直线 $l$ 的垂线，垂足为 $H$，则 $AH$ 是 $A$ 到 $l$ 的最短距离。

设 $l$ 过点 $P$，方向向量为 $\vec{d}$，则 $\vec{AP}$ 在垂直于 $\vec{d}$ 方向上的分量即为最短距离：

$$d_{\min} = \sqrt{|\vec{AP}|^2 - \left(\frac{\vec{AP} \cdot \vec{d}}{|\vec{d}|}\right)^2}$$

**动点最近距离**：若动点 $P$ 在某曲线（直线、圆、抛物线）上运动，求 $P$ 到定点 $A$ 的最近距离，本质是求 $|\vec{AP}|$ 的最小值。

### 3. 证明三角不等式（向量方法）

**定理（三角不等式）**：对任意向量 $\vec{a}, \vec{b}$：

$$\bigl||\vec{a}| - |\vec{b}|\bigr| \leq |\vec{a} + \vec{b}| \leq |\vec{a}| + |\vec{b}|$$

**右侧不等式的向量证明**：

$$|\vec{a} + \vec{b}|^2 = (\vec{a} + \vec{b})\cdot(\vec{a} + \vec{b}) = |\vec{a}|^2 + 2\vec{a}\cdot\vec{b} + |\vec{b}|^2$$

由 $\vec{a}\cdot\vec{b} \leq |\vec{a}||\vec{b}|$（Cauchy-Schwarz 不等式），得：

$$|\vec{a} + \vec{b}|^2 \leq |\vec{a}|^2 + 2|\vec{a}||\vec{b}| + |\vec{b}|^2 = (|\vec{a}| + |\vec{b}|)^2$$

两边取非负平方根：$|\vec{a} + \vec{b}| \leq |\vec{a}| + |\vec{b}|$。

等号成立当且仅当 $\vec{a} \cdot \vec{b} = |\vec{a}||\vec{b}|$，即 $\vec{a}$ 与 $\vec{b}$ 同向（$\theta = 0°$）。

**几何含义**：三角形任意两边之和大于第三边（三角形内可取等，退化为共线时等号成立）。

---

## 三、向量与代数、三角的交叉

### 1. 向量 + 函数：含变量向量求最值

若向量的坐标含参数，求向量模的最值转化为求函数极值。

**例型**：$\vec{a} = (1, \sin\theta)$，$\vec{b} = (\cos\theta, -1)$，求 $|\vec{a} - \vec{b}|$ 的最小值。

处理方法：
1. 写出 $\vec{a} - \vec{b} = (1 - \cos\theta,\ \sin\theta + 1)$
2. $|\vec{a} - \vec{b}|^2 = (1-\cos\theta)^2 + (\sin\theta+1)^2$
3. 展开化简，利用 $\sin^2\theta + \cos^2\theta = 1$，转化为三角函数求极值

### 2. 向量 + 三角：用三角恒等式简化

数量积经常产生形如 $\cos\alpha\cos\beta + \sin\alpha\sin\beta = \cos(\alpha - \beta)$ 的结构，直接套用和差化积或积化和差公式。

**例型**：$\vec{a} = (\cos\alpha, \sin\alpha)$，$\vec{b} = (\cos\beta, \sin\beta)$，则：

$$\vec{a} \cdot \vec{b} = \cos\alpha\cos\beta + \sin\alpha\sin\beta = \cos(\alpha - \beta)$$

由此可知：**单位向量的数量积等于夹角的余弦**——这正是数量积定义的几何意义。

### 3. 向量证明平行四边形对角线互相平分

**定理**：平行四边形 $ABCD$ 的对角线 $AC$ 与 $BD$ 互相平分。

**向量证明**：

设 $\vec{OA} = \vec{a}$，$\vec{OB} = \vec{b}$，$\vec{OC} = \vec{c}$，$\vec{OD} = \vec{d}$（$O$ 为任意参考点）。

$ABCD$ 是平行四边形 $\Rightarrow$ $\vec{AB} = \vec{DC}$：

$$\vec{b} - \vec{a} = \vec{c} - \vec{d} \quad \Rightarrow \quad \vec{a} + \vec{c} = \vec{b} + \vec{d}$$

$AC$ 的中点 $M_1$：$\vec{OM_1} = \dfrac{\vec{a} + \vec{c}}{2}$

$BD$ 的中点 $M_2$：$\vec{OM_2} = \dfrac{\vec{b} + \vec{d}}{2}$

由 $\vec{a} + \vec{c} = \vec{b} + \vec{d}$，得 $\vec{OM_1} = \vec{OM_2}$，即 $M_1 = M_2$——两对角线中点重合，即互相平分。

---

## 四、典型应用例题

### 例 1：力的合成求最大合力

**题目**：两个力 $|\vec{F_1}| = 3\,\text{N}$，$|\vec{F_2}| = 4\,\text{N}$，夹角 $\theta = 90°$，求合力 $|\vec{F}|$ 及其与 $\vec{F_1}$ 的夹角 $\alpha$。

**【思路】** 两力垂直，用勾股定理求合力；用反正切求角度。

**解**：

$$|\vec{F}|^2 = |\vec{F_1}|^2 + |\vec{F_2}|^2 = 9 + 16 = 25 \quad \Rightarrow \quad |\vec{F}| = 5\,\text{N}$$

合力与 $\vec{F_1}$ 的夹角：

$$\tan\alpha = \frac{|\vec{F_2}|}{|\vec{F_1}|} = \frac{4}{3} \quad \Rightarrow \quad \alpha = \arctan\frac{4}{3} \approx 53.1°$$

**答**：合力大小为 $5\,\text{N}$，方向与 $\vec{F_1}$ 成约 $53°$ 角。

---

### 例 2：船渡河问题

**题目**：船在静水中速度为 $v_{\text{船}} = 4\,\text{m/s}$，垂直河岸方向行驶；水流速度 $v_{\text{水}} = 3\,\text{m/s}$，沿河岸方向。求船的实际速度大小及方向（与河岸垂直方向的偏角）。

**【思路】** 实际速度 = 两速度的向量和，两者垂直，用勾股定理。

**解**：

设垂直河岸为 $y$ 方向，沿河岸为 $x$ 方向：

$$\vec{v} = \vec{v_{\text{船}}} + \vec{v_{\text{水}}} = (3, 4)\,\text{m/s（分量形式）}$$

$$|\vec{v}| = \sqrt{3^2 + 4^2} = 5\,\text{m/s}$$

与 $y$ 轴（垂直河岸方向）的偏角：

$$\tan\beta = \frac{v_x}{v_y} = \frac{3}{4} \quad \Rightarrow \quad \beta = \arctan\frac{3}{4} \approx 36.9°$$

**答**：实际速度为 $5\,\text{m/s}$，方向偏向下游约 $37°$。

**延伸**：若河宽为 $d$，船实际到达对岸时漂移距离 $= d \times \dfrac{v_{\text{水}}}{v_{\text{船}}} = d \times \dfrac{3}{4}$（利用过河时间 $t = \dfrac{d}{v_{\text{船}}} = \dfrac{d}{4}$，漂移 $= v_{\text{水}} \cdot t = \dfrac{3d}{4}$）。

---

### 例 3：用向量证明中位线定理

**题目**：在 $\triangle ABC$ 中，$M, N$ 分别是 $AB, AC$ 的中点。证明 $MN \parallel BC$ 且 $MN = \dfrac{1}{2}BC$。

**【思路】** 用基底法，以 $\vec{AB}, \vec{AC}$ 为基底表示 $\vec{MN}$ 和 $\vec{BC}$，若成比例则平行且长度为一半。

**证明**：

设 $\vec{AB} = \vec{b}$，$\vec{AC} = \vec{c}$。

中点坐标：$M$ 是 $AB$ 中点，$\vec{AM} = \dfrac{1}{2}\vec{b}$；$N$ 是 $AC$ 中点，$\vec{AN} = \dfrac{1}{2}\vec{c}$。

$$\vec{MN} = \vec{AN} - \vec{AM} = \frac{1}{2}\vec{c} - \frac{1}{2}\vec{b} = \frac{1}{2}(\vec{c} - \vec{b})$$

$$\vec{BC} = \vec{AC} - \vec{AB} = \vec{c} - \vec{b}$$

因此：

$$\vec{MN} = \frac{1}{2}\vec{BC}$$

这说明：① $\vec{MN}$ 与 $\vec{BC}$ 共线（方向相同），即 $MN \parallel BC$；② $|MN| = \dfrac{1}{2}|BC|$，即中位线长度是底边的一半。

**结论**：三角形中位线平行于底边且等于底边的一半。

---

## 五、易错点汇总

**易错 1：物理量忘记单位，导致答案量纲错误**

向量加法规则不变，但物理量有单位（N、m/s、m 等）。合力 $|\vec{F}|$ 的单位是 N，速度的单位是 m/s，不能混用。高中数学题中通常在题设中给出单位，计算结果带上单位。

**易错 2：几何应用中没有先选定基底**

用向量证明几何题时，第一步必须**明确基底**：选定哪两个（不共线的）向量作为基，所有其他向量都用它们的线性组合表示。基底选错（如选了共线的两个向量）会导致无法表示平面内的任意向量。

**易错 3：三角不等式的等号条件**

$|\vec{a} + \vec{b}| \leq |\vec{a}| + |\vec{b}|$ 等号成立条件是 $\vec{a}$ 与 $\vec{b}$ **同向**（$\theta = 0°$），不是任意情况。反向时等号在 $|\vec{a} + \vec{b}| = \bigl||\vec{a}| - |\vec{b}|\bigr|$ 中成立。

**易错 4：船渡河中"最短时间"与"最短路程"的混淆**

最短过河时间：船头垂直河岸，时间 $t_{\min} = \dfrac{d}{v_{\text{船}}}$，与水速无关。最短路程（合位移最短）：船头应斜向上游，使合速度垂直河岸，此时无漂移。两种策略不同，审题要看问的是哪个量。

**易错 5：向量与三角混合时忘记"模"的非负性**

$|\vec{a}|^2 = \vec{a} \cdot \vec{a}$ 恒成立，$|\vec{a}| \geq 0$ 恒成立。但在展开 $|\vec{a} + \vec{b}|^2$ 时，$2\vec{a}\cdot\vec{b}$ 可正可负——不要错误地认为 $|\vec{a}+\vec{b}|^2 \geq |\vec{a}|^2 + |\vec{b}|^2$（只有垂直时才等于）。

---

## 六、思路自测题

**自测 1**　两力 $|\vec{F_1}| = 6\,\text{N}$，$|\vec{F_2}| = 6\,\text{N}$，夹角 $\theta = 120°$。求合力大小。

> 提示：$|\vec{F}|^2 = 36 + 2 \times 6 \times 6 \times \cos120° + 36 = 36 + 72 \times (-\frac{1}{2}) + 36 = 36$，故 $|\vec{F}| = 6\,\text{N}$。（等边三角形情形，合力与每个分力等大。）

**自测 2**　在 $\triangle ABC$ 中，$G$ 是重心（三条中线的交点）。证明 $\vec{GA} + \vec{GB} + \vec{GC} = \vec{0}$。

> 提示：设 $\vec{GA} = \vec{a}$，$\vec{GB} = \vec{b}$，$\vec{GC} = \vec{c}$。重心性质：$\vec{OG} = \dfrac{1}{3}(\vec{OA}+\vec{OB}+\vec{OC})$ 对任意原点 $O$ 成立。取 $O = G$：$\vec{GG} = \dfrac{1}{3}(\vec{GA}+\vec{GB}+\vec{GC}) = \vec{0}$，故 $\vec{GA}+\vec{GB}+\vec{GC} = \vec{0}$。

**自测 3**　$\vec{a} = (\cos\theta, \sin\theta)$，$\vec{b} = (\cos\varphi, \sin\varphi)$，计算 $|\vec{a} - \vec{b}|^2$，并用三角恒等式化简。

> 提示：$|\vec{a}-\vec{b}|^2 = (\cos\theta-\cos\varphi)^2 + (\sin\theta-\sin\varphi)^2 = 2 - 2(\cos\theta\cos\varphi + \sin\theta\sin\varphi) = 2 - 2\cos(\theta - \varphi)$。  
> 利用半角公式：$= 4\sin^2\dfrac{\theta-\varphi}{2}$，故 $|\vec{a}-\vec{b}| = 2\left|\sin\dfrac{\theta-\varphi}{2}\right|$。（单位向量夹角 $\alpha = \theta - \varphi$ 时，$|\vec{a}-\vec{b}| = 2\sin\dfrac{\alpha}{2}$，与弦长公式一致。）

**自测 4**　已知向量 $\vec{a}, \vec{b}$ 满足 $|\vec{a}| = 3$，$|\vec{b}| = 4$，$|\vec{a} + \vec{b}| = 5$。求 $\vec{a} \cdot \vec{b}$，并判断 $\vec{a}$ 与 $\vec{b}$ 是否垂直。

> 提示：$|\vec{a}+\vec{b}|^2 = |\vec{a}|^2 + 2\vec{a}\cdot\vec{b} + |\vec{b}|^2$，即 $25 = 9 + 2\vec{a}\cdot\vec{b} + 16$，故 $2\vec{a}\cdot\vec{b} = 0$，$\vec{a}\cdot\vec{b} = 0$。$\vec{a} \perp \vec{b}$（满足勾股定理 $3^2+4^2=5^2$，直角三角形）。

---

**回头看"一例速记"**：

> 物理向量加法：合力/合速度 = 向量和，用平行四边形定则或坐标分量运算。  
> 几何综合：先选基底，再用线性组合表示所有向量，最后利用平行/垂直/模的条件推导结论。  
> 三角交叉：数量积展开后经常出现 $\cos(\alpha-\beta)$ 型的化简。  
> 三角不等式：$|\vec{a}+\vec{b}| \leq |\vec{a}| + |\vec{b}|$，等号同向时取。

能不看提示独立完成自测 2（重心证明）和自测 4（数量积计算）——本章，你拿下了。
