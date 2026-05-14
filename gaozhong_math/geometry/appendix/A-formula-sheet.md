# 附录 A：高中几何公式表（种子）

> 覆盖 Part 1–4、Part 8（平面向量基础、平面向量应用、直线与方程、圆与方程、立体几何初步）共 23 章核心公式与定理。每条公式配适用条件与简短说明，方便考前 1–2 小时通览。

---

## Part 1 平面向量基础

### 1.1 向量基本概念 `(part1/01)`

- **向量**：既有大小又有方向的量，记作 $\vec{a}$，图形表示为有向线段 $\overrightarrow{AB}$（$A$ 为起点，$B$ 为终点）。
- **模（长度）**：$|\vec{a}|$；零向量 $\vec{0}$ 模为 $0$，方向任意。
- **单位向量**：模为 $1$ 的向量，$|\vec{e}| = 1$。
- **相等向量**：模相等且方向相同（与起点无关）；**相反向量**：$-\vec{a}$ 与 $\vec{a}$ 模相等方向相反。
- **共线向量（平行向量）**：方向相同或相反，即两向量在同一直线或平行直线上；记 $\vec{a} \parallel \vec{b}$（$\vec{b} \ne \vec{0}$）。

| 概念 | 条件 | 说明 |
|------|------|------|
| 相等 | 模相同 + 方向相同 | 自由向量，起点无关 |
| 共线 | 方向平行（含同向、反向） | $\vec{0}$ 与任意向量共线 |
| 单位向量 | $|\hat{a}| = 1$ | $\hat{a} = \dfrac{\vec{a}}{|\vec{a}|}$ |

### 1.2 向量加减法 `(part1/02)`

**加法**（三角形法则 / 平行四边形法则）：

$$\overrightarrow{AB} + \overrightarrow{BC} = \overrightarrow{AC} \quad \text{（三角形法则，首尾相接）}$$

$$\vec{a} + \vec{b} = \vec{b} + \vec{a} \quad \text{（交换律）}$$

$$(\vec{a} + \vec{b}) + \vec{c} = \vec{a} + (\vec{b} + \vec{c}) \quad \text{（结合律）}$$

**减法**：

$$\vec{a} - \vec{b} = \vec{a} + (-\vec{b})$$

$$\overrightarrow{OA} - \overrightarrow{OB} = \overrightarrow{BA} \quad \text{（同起点相减，箭头从减数终点指向被减数终点）}$$

**关键变形**：$\overrightarrow{AB} = \overrightarrow{OB} - \overrightarrow{OA}$，常用于坐标计算。

### 1.3 向量数乘 `(part1/03)`

$$\lambda\vec{a}:\quad |\lambda\vec{a}| = |\lambda||\vec{a}|;\quad \lambda > 0 \text{ 同向};\quad \lambda < 0 \text{ 反向};\quad \lambda = 0 \Rightarrow \vec{0}$$

**运算律（3 条）**：

$$(\lambda\mu)\vec{a} = \lambda(\mu\vec{a}) \quad \text{（结合律）}$$

$$(\lambda + \mu)\vec{a} = \lambda\vec{a} + \mu\vec{a} \quad \text{（对标量加法分配）}$$

$$\lambda(\vec{a} + \vec{b}) = \lambda\vec{a} + \lambda\vec{b} \quad \text{（对向量加法分配）}$$

**共线定理**：$\vec{a} \ne \vec{0}$ 时，$\vec{b} \parallel \vec{a} \Leftrightarrow \exists \lambda \in \mathbb{R}$，使得 $\vec{b} = \lambda\vec{a}$。

### 1.4 平面向量基本定理与坐标 `(part1/04)`

**基本定理**：设 $\vec{e_1}, \vec{e_2}$ 不共线（一组基底），则平面内任意向量 $\vec{a}$ 存在唯一实数对 $(\lambda_1, \lambda_2)$，使得

$$\vec{a} = \lambda_1\vec{e_1} + \lambda_2\vec{e_2}$$

**直角坐标表示**（$\vec{e_1} = (1,0),\; \vec{e_2} = (0,1)$）：

| 运算 | 公式 |
|------|------|
| 设 $\vec{a} = (x_1, y_1),\; \vec{b} = (x_2, y_2)$ | — |
| 加法 | $\vec{a} + \vec{b} = (x_1+x_2,\; y_1+y_2)$ |
| 减法 | $\vec{a} - \vec{b} = (x_1-x_2,\; y_1-y_2)$ |
| 数乘 | $\lambda\vec{a} = (\lambda x_1,\; \lambda y_1)$ |
| 模 | $|\vec{a}| = \sqrt{x_1^2 + y_1^2}$ |
| $\overrightarrow{AB}$ | $= (x_B - x_A,\; y_B - y_A)$ |
| 共线判定 | $x_1 y_2 - x_2 y_1 = 0$（且 $\vec{a}, \vec{b}$ 不全为零向量） |

**中点公式**：$A(x_1, y_1)$，$B(x_2, y_2)$ 的中点 $M$：

$$M = \left(\frac{x_1+x_2}{2},\; \frac{y_1+y_2}{2}\right)$$

**重心公式**：三角形 $ABC$ 三顶点坐标 $(x_i, y_i)$，重心 $G$：

$$G = \left(\frac{x_1+x_2+x_3}{3},\; \frac{y_1+y_2+y_3}{3}\right)$$

> **易错提醒（Part 1）**
> 1. $\overrightarrow{OA} - \overrightarrow{OB} = \overrightarrow{BA}$（不是 $\overrightarrow{AB}$）！减法方向从"减数终点"到"被减数终点"。
> 2. 共线判定公式 $x_1 y_2 - x_2 y_1 = 0$ 的前提：$\vec{a}, \vec{b}$ 不同时为零向量；零向量与任意向量共线但不能用此公式直接判定。
> 3. 中点公式要对两端点坐标**求平均**，不要用差值。

---

## Part 2 平面向量应用

### 2.1 数量积 `(part2/01)`

**定义**（代数 × 几何）：

$$\boxed{\vec{a} \cdot \vec{b} = |\vec{a}||\vec{b}|\cos\theta = x_1 x_2 + y_1 y_2} \quad \theta = \angle(\vec{a}, \vec{b}) \in [0, \pi]$$

**三大用途**：

| 用途 | 公式 | 说明 |
|------|------|------|
| 求夹角 | $\cos\theta = \dfrac{\vec{a} \cdot \vec{b}}{|\vec{a}||\vec{b}|}$ | $\theta \in [0, \pi]$，注意分母不为零 |
| 判垂直 | $\vec{a} \perp \vec{b} \Leftrightarrow \vec{a} \cdot \vec{b} = 0$ | 充要条件 |
| 求投影 | $\vec{b}$ 在 $\vec{a}$ 上的投影 $= \dfrac{\vec{a} \cdot \vec{b}}{|\vec{a}|}$ | 有正负（取决于 $\cos\theta$ 正负） |

**模平方与展开**：

$$|\vec{a}|^2 = \vec{a} \cdot \vec{a}$$

$$|\vec{a} + \vec{b}|^2 = |\vec{a}|^2 + 2\vec{a} \cdot \vec{b} + |\vec{b}|^2$$

$$|\vec{a} - \vec{b}|^2 = |\vec{a}|^2 - 2\vec{a} \cdot \vec{b} + |\vec{b}|^2$$

**极化恒等式**：

$$\vec{a} \cdot \vec{b} = \frac{1}{4}\left(|\vec{a}+\vec{b}|^2 - |\vec{a}-\vec{b}|^2\right)$$

**运算律**：

$$\vec{a} \cdot \vec{b} = \vec{b} \cdot \vec{a} \quad \text{（交换律）}$$

$$(\lambda\vec{a}) \cdot \vec{b} = \lambda(\vec{a} \cdot \vec{b}) \quad \text{（数乘结合）}$$

$$\vec{a} \cdot (\vec{b} + \vec{c}) = \vec{a} \cdot \vec{b} + \vec{a} \cdot \vec{c} \quad \text{（分配律）}$$

### 2.2 三点共线与分点公式 `(part2/02)`

**三点共线条件**（向量法）：

$$C \text{ 在线段 } AB \text{ 上（含延长线）} \Leftrightarrow \exists \lambda \in \mathbb{R},\; \overrightarrow{OC} = \lambda\overrightarrow{OA} + (1-\lambda)\overrightarrow{OB}$$

等价条件：$\overrightarrow{AC} = t\,\overrightarrow{AB}$，即 $\overrightarrow{OC} - \overrightarrow{OA} = t(\overrightarrow{OB} - \overrightarrow{OA})$。

**分点公式**（$P$ 分有向线段 $\overrightarrow{AB}$ 为 $\lambda : 1$，即 $\overrightarrow{AP} = \lambda\overrightarrow{PB}$）：

$$\overrightarrow{OP} = \frac{\overrightarrow{OA} + \lambda\overrightarrow{OB}}{1 + \lambda} \quad (\lambda \ne -1)$$

坐标形式：

$$P = \left(\frac{x_A + \lambda x_B}{1+\lambda},\; \frac{y_A + \lambda y_B}{1+\lambda}\right)$$

**特例**：$\lambda = 1$ 时 $P$ 为中点，即 $P = \left(\dfrac{x_A+x_B}{2}, \dfrac{y_A+y_B}{2}\right)$。

**三角形内心、外心、垂心常用结论**：

| 特殊点 | 坐标表达 |
|--------|----------|
| 重心 $G$ | $\overrightarrow{OG} = \dfrac{\overrightarrow{OA}+\overrightarrow{OB}+\overrightarrow{OC}}{3}$ |
| $P$ 在 $\triangle ABC$ 内（面积坐标） | $\overrightarrow{OP} = \alpha\overrightarrow{OA}+\beta\overrightarrow{OB}+\gamma\overrightarrow{OC}$，$\alpha+\beta+\gamma=1$，$\alpha,\beta,\gamma>0$ |

> **易错提醒（Part 2）**
> 1. 数量积不满足结合律：$(\vec{a} \cdot \vec{b}) \cdot \vec{c}$ 无意义（$\vec{a} \cdot \vec{b}$ 是标量，标量不能与向量作点积）。
> 2. $\vec{a} \cdot \vec{b} = 0$ 推出 $\vec{a} \perp \vec{b}$，但必须排除 $\vec{a} = \vec{0}$ 或 $\vec{b} = \vec{0}$。
> 3. 分点公式中 $\lambda$ 是有向比，$\lambda < 0$ 表示外分点；不要混淆 $\overrightarrow{AP}:\overrightarrow{PB}$ 的方向。

---

## Part 3 直线与方程

### 3.1 倾斜角与斜率 `(part3/01)`

$$\alpha \in [0°, 180°);\quad k = \tan\alpha \quad (\alpha \ne 90°)$$

**两点斜率**：

$$k = \frac{y_2 - y_1}{x_2 - x_1} \quad (x_1 \ne x_2)$$

**注意**：$\alpha = 90°$ 时直线竖直，斜率不存在。

### 3.2 直线方程五种形式 `(part3/02)`

| 名称 | 方程形式 | 适用条件 / 选用场景 |
|------|----------|--------------------|
| **点斜式** | $y - y_0 = k(x - x_0)$ | 已知一点 + 斜率（最通用） |
| **斜截式** | $y = kx + b$ | 已知斜率 + $y$ 轴截距；便于求交点 |
| **两点式** | $\dfrac{y - y_1}{y_2 - y_1} = \dfrac{x - x_1}{x_2 - x_1}$ | 已知两点，注意 $x_1 \ne x_2, y_1 \ne y_2$ |
| **截距式** | $\dfrac{x}{a} + \dfrac{y}{b} = 1$ | 已知两截距（$a, b \ne 0$）；注意过原点不适用 |
| **一般式** | $Ax + By + C = 0$（$A, B$ 不全零） | 最通用；竖直线仅此式可表示 |

**一般式转各形式**：斜率 $k = -\dfrac{A}{B}$（$B \ne 0$），$y$ 截距 $= -\dfrac{C}{B}$，$x$ 截距 $= -\dfrac{C}{A}$。

### 3.3 直线位置关系 `(part3/03)`

**斜截式判别**（$y = k_1 x + b_1$ 与 $y = k_2 x + b_2$）：

| 位置 | 条件 |
|------|------|
| 平行 $l_1 \parallel l_2$ | $k_1 = k_2,\; b_1 \ne b_2$ |
| 重合 | $k_1 = k_2,\; b_1 = b_2$ |
| 相交（不垂直） | $k_1 \ne k_2$ |
| 垂直 $l_1 \perp l_2$ | $k_1 \cdot k_2 = -1$ |

**一般式判别**（$A_1 x + B_1 y + C_1 = 0$ 与 $A_2 x + B_2 y + C_2 = 0$）：

| 位置 | 条件 |
|------|------|
| 平行 | $\dfrac{A_1}{A_2} = \dfrac{B_1}{B_2} \ne \dfrac{C_1}{C_2}$ |
| 重合 | $\dfrac{A_1}{A_2} = \dfrac{B_1}{B_2} = \dfrac{C_1}{C_2}$ |
| 相交 | $\dfrac{A_1}{A_2} \ne \dfrac{B_1}{B_2}$（或行列式 $A_1 B_2 - A_2 B_1 \ne 0$） |
| 垂直 | $A_1 A_2 + B_1 B_2 = 0$ |

**含参讨论原则**：先判断斜率是否存在，对竖直线单独讨论。

### 3.4 距离公式 `(part3/04)`

**点到直线的距离**（最常用）：

$$\boxed{d = \frac{|Ax_0 + By_0 + C|}{\sqrt{A^2 + B^2}}} \quad \text{点 } P(x_0, y_0) \text{ 到 } Ax+By+C=0$$

**两平行线间的距离**（$l_1: Ax+By+C_1=0$，$l_2: Ax+By+C_2=0$，$A,B$ 完全相同）：

$$\boxed{d = \frac{|C_1 - C_2|}{\sqrt{A^2 + B^2}}}$$

**注意**：两平行线方程须先化为 $A, B$ 完全相同的形式（不能仅成比例）。

**两点距离**：

$$|AB| = \sqrt{(x_B - x_A)^2 + (y_B - y_A)^2}$$

> **易错提醒（Part 3）**
> 1. 含参直线不能直接假设斜率存在——凡遇"斜率不确定"必须先考虑竖直线（$x = $ 常数）。
> 2. 平行线距离公式要求 $A, B$ 系数完全一致；若题目给的是 $2x+y+1=0$ 与 $4x+2y+3=0$，需先把第二条化为 $2x+y+\dfrac{3}{2}=0$ 再套公式。
> 3. 直线垂直条件 $k_1 k_2 = -1$ 仅在两直线斜率均存在时成立；一条竖直 + 一条水平（斜率为 $0$）时直接判断。

---

## Part 4 圆与方程

### 4.1 圆的方程两种形式 `(part4/01)`

**标准方程**（已知圆心和半径，首选）：

$$\boxed{(x - a)^2 + (y - b)^2 = r^2} \quad \text{圆心 } (a, b),\; \text{半径 } r > 0$$

**一般方程**（展开整理得）：

$$x^2 + y^2 + Dx + Ey + F = 0$$

- 配方变标准：圆心 $\left(-\dfrac{D}{2}, -\dfrac{E}{2}\right)$，半径 $r = \dfrac{\sqrt{D^2 + E^2 - 4F}}{2}$
- **判别条件**：$D^2 + E^2 - 4F > 0$ 表示圆；$= 0$ 为点；$< 0$ 无图形。

**过三点定圆**：将三点坐标代入一般方程，解方程组求 $D, E, F$。

### 4.2 直线与圆的位置关系 `(part4/02)`

设圆心 $C$，半径 $r$；直线 $l$；圆心到 $l$ 的距离为 $d$：

| 位置 | 条件 | 交点个数 |
|------|------|----------|
| 相离 | $d > r$ | $0$ |
| 相切 | $d = r$ | $1$ |
| 相交 | $d < r$ | $2$ |

**弦长公式**（直线与圆相交时）：

$$\boxed{|AB| = 2\sqrt{r^2 - d^2}}$$

弦的半长：$l = \sqrt{r^2 - d^2}$，由圆心、弦端点、弦中点构成直角三角形。

**切线方程**（过圆上一点 $P(x_0, y_0)$，圆为标准方程）：

$$\boxed{(x_0 - a)(x - a) + (y_0 - b)(y - b) = r^2}$$

**切线方程**（过圆外一点，需设切点列方程组联立）：设切线斜率为 $k$，由 $d = r$ 解出 $k$，注意验证竖直切线情形。

### 4.3 圆与圆的位置关系 `(part4/03)`

设两圆圆心 $C_1, C_2$，半径 $r_1, r_2$（$r_1 \ge r_2$），圆心距 $d = |C_1 C_2|$：

| 位置 | 条件 | 公切线数 |
|------|------|----------|
| 外离 | $d > r_1 + r_2$ | $4$ 条 |
| 外切 | $d = r_1 + r_2$ | $3$ 条 |
| 相交 | $|r_1 - r_2| < d < r_1 + r_2$ | $2$ 条 |
| 内切 | $d = |r_1 - r_2|$（$d \ne 0$） | $1$ 条 |
| 内含 | $d < |r_1 - r_2|$ | $0$ 条 |

**公共弦方程**（两圆相交时，两方程相减）：

设 $C_1: x^2+y^2+D_1 x+E_1 y+F_1=0$，$C_2: x^2+y^2+D_2 x+E_2 y+F_2=0$，

公共弦（根轴）方程：$(D_1-D_2)x + (E_1-E_2)y + (F_1-F_2) = 0$。

> **易错提醒（Part 4）**
> 1. 一般方程判圆时 $D^2+E^2-4F > 0$ 不要忘记检验；题目给出一般式时，配方步骤不能省略。
> 2. 直线与圆位置关系优先用距离法（圆心到直线的距离 $d$ 与 $r$ 比较），联立法运算量大且易出错。
> 3. 圆与圆位置判别用圆心距 $d$，注意"内切"条件是 $d = r_1 - r_2$（大减小，且 $d \ne 0$），不要写成 $|r_1 - r_2|$ 后忽略 $d = 0$ 的同心圆情形（同心圆为内含，不是内切）。

---

## Part 8 立体几何初步

### 8.1 公理与推论 `(part8/01)`

**三大公理**：

| 公理 | 内容 |
|------|------|
| 公理 1 | 如果直线上两点在一个平面内，则该直线在此平面内。 |
| 公理 2（三点定面） | 经过不在同一直线上的三点，有且只有一个平面。 |
| 公理 3（平面交线） | 如果两个平面有公共点，则它们有且只有一条公共直线，且这两个平面内所有公共点都在这条直线上。 |

**三条推论**：
1. 经过一条直线和直线外一点，有且只有一个平面。
2. 经过两条相交直线，有且只有一个平面。
3. 经过两条平行直线，有且只有一个平面。

**三视图三规律**：正视图（主视图）、侧视图、俯视图满足：

$$\text{长对正（正视图与俯视图等宽）}\quad \text{高平齐（正视图与侧视图等高）}\quad \text{宽相等（侧视图与俯视图等宽）}$$

### 8.2 异面直线 `(part8/02)`

**定义**：不在任何同一平面内的两条直线（既不平行也不相交）。

**夹角**：$\theta \in \left(0°, 90°\right]$（取锐角或直角，不取钝角）。

| 求法 | 步骤 |
|------|------|
| **综合法** | 平移一直线使两线相交，交角即为所求（取锐角）。 |
| **向量法** | 设两直线方向向量 $\vec{u}, \vec{v}$，$\cos\theta = \dfrac{|\vec{u} \cdot \vec{v}|}{|\vec{u}||\vec{v}|}$（绝对值保证取锐角）。 |

### 8.3 直线与平面的位置关系 `(part8/03)`

**三种位置**：线在面内 / 线平行于面 / 线与面相交（含垂直）。

**线面平行**：

| | 内容 |
|--|------|
| **判定定理** | 若平面外一直线与平面内某直线平行，则该直线平行于该平面。（$l \parallel m \subset \alpha, l \not\subset \alpha \Rightarrow l \parallel \alpha$） |
| **性质定理** | 若直线平行于平面，则直线与平面内任意直线平行或相交（不相交即平行）；直线与平面交线平行。 |

**线面垂直**：

| | 内容 |
|--|------|
| **判定定理** | 若一直线与平面内两条**相交**直线均垂直，则该直线垂直于该平面。（$l \perp m_1, l \perp m_2, m_1 \cap m_2 = P \subset \alpha \Rightarrow l \perp \alpha$） |
| **性质定理** | 若直线垂直于平面，则该直线垂直于平面内所有直线。 |
| **向量判定** | 设 $\vec{n}$ 为平面法向量，直线方向向量为 $\vec{v}$，则 $l \perp \alpha \Leftrightarrow \vec{v} \parallel \vec{n}$（即 $\vec{v} \times \vec{n} = \vec{0}$ 或 $\vec{v} = k\vec{n}$）。 |

**线面角（斜线与平面所成角）**：

斜线在平面上的射影与斜线本身的夹角，范围 $\left(0°, 90°\right)$。

### 8.4 平面与平面的位置关系 `(part8/04)`

**两种位置**：平行 / 相交（含垂直）。

**面面平行**：

| | 内容 |
|--|------|
| **判定定理** | 若一平面内两条相交直线均平行于另一平面，则两平面平行。 |
| **性质定理** | 两平行平面被第三个平面截，交线平行。（$\alpha \parallel \beta, \gamma \cap \alpha = l_1, \gamma \cap \beta = l_2 \Rightarrow l_1 \parallel l_2$） |

**面面垂直**：

| | 内容 |
|--|------|
| **判定定理** | 若一平面经过另一平面的垂线，则两平面垂直。（$l \perp \alpha, l \subset \beta \Rightarrow \alpha \perp \beta$） |
| **性质定理** | 两平面垂直时，在一平面内作另一平面的垂线，则该垂线垂直于交线。 |
| **向量判定** | 法向量 $\vec{n_1}, \vec{n_2}$：$\alpha \parallel \beta \Leftrightarrow \vec{n_1} \parallel \vec{n_2}$；$\alpha \perp \beta \Leftrightarrow \vec{n_1} \perp \vec{n_2}$（即 $\vec{n_1} \cdot \vec{n_2} = 0$）。 |

**二面角**：

- 定义：两个半平面（有公共棱）所成的角，范围 $[0°, 180°]$。
- **作法（3 步）**：① 找棱（公共边）；② 在两个半平面内各作与棱垂直的射线；③ 两射线所成角即二面角的平面角。
- **向量法**：设两平面法向量 $\vec{n_1}, \vec{n_2}$，$\cos\theta = \dfrac{|\vec{n_1} \cdot \vec{n_2}|}{|\vec{n_1}||\vec{n_2}|}$（取绝对值以保证角在 $[0°, 90°]$，需判断与实际角的关系）。

  注意：法向量夹角可能是二面角，也可能是其补角，需结合图形判断。

### 8.5 表面积与体积公式 `(part8/05)`

**柱体**（直棱柱）：底面积 $S$，高 $h$，底面周长 $C$：

| 量 | 公式 |
|----|------|
| 侧面积 | $S_{\text{侧}} = C \cdot h$ |
| 全面积 | $S_{\text{全}} = S_{\text{侧}} + 2S$ |
| 体积 | $V = S \cdot h$ |

**锥体**：底面积 $S$，高 $h$，斜高 $l$（正锥），底面周长 $C$：

| 量 | 公式 |
|----|------|
| 侧面积（正锥） | $S_{\text{侧}} = \dfrac{1}{2} C \cdot l$ |
| 全面积 | $S_{\text{全}} = S_{\text{侧}} + S$ |
| 体积 | $V = \dfrac{1}{3} S \cdot h$ |

**台体**（棱台 / 圆台）：上底面积 $S_1$，下底面积 $S_2$，高 $h$：

$$V_{\text{台}} = \dfrac{h}{3}\left(S_1 + S_2 + \sqrt{S_1 S_2}\right)$$

圆台侧面积（$r_1, r_2$ 为上下底半径，$l$ 为母线长）：$S_{\text{侧}} = \pi(r_1 + r_2)l$。

**球**：半径 $R$：

$$S_{\text{球}} = 4\pi R^2 \qquad V_{\text{球}} = \dfrac{4}{3}\pi R^3$$

### 8.6 球的内切外接经典模型 `(part8/06)`

| 几何体 | 球的类型 | 关键关系 |
|--------|----------|----------|
| 正三棱锥 | 外接球 | 用勾股定理求外接圆半径 $R$，再由高关系求球心位置 |
| 正三棱柱 | 外接球 | 球心在体对称轴上，$R^2 = r^2 + \left(\dfrac{h}{2}\right)^2$（$r$ 为底面外接圆半径） |
| 正四棱柱（边长 $a$，高 $h$） | 外接球 | $R = \dfrac{\sqrt{2a^2 + h^2}}{2}$ |
| 正四面体（边长 $a$） | 外接球 | $R = \dfrac{\sqrt{6}}{4}a$；内切球 $r = \dfrac{\sqrt{6}}{12}a$ |
| 直三棱锥（三侧棱两两垂直，长 $a,b,c$） | 外接球 | $R = \dfrac{\sqrt{a^2+b^2+c^2}}{2}$（直径 = 体对角线） |
| 圆柱内接球 | 球 | $r_{\text{球}} = \dfrac{h}{2} = r_{\text{底}}$（高 = 底直径） |
| 圆锥内切球 | 球 | $r = \dfrac{R \cdot h}{\sqrt{R^2+h^2}+R}$（$R$ 底半径，$h$ 高） |

> **易错提醒（Part 8）**
> 1. 线面垂直判定定理要求平面内**两条相交**直线都与该直线垂直，仅凭一条不够。
> 2. 二面角用向量法求得的角可能是实际二面角的补角——必须结合图形确认角的范围（$< 90°$ 还是 $> 90°$）。
> 3. 台体体积公式 $V = \dfrac{h}{3}(S_1 + S_2 + \sqrt{S_1 S_2})$ 不能与柱体（$V = Sh$）混用；当 $S_1 = 0$ 时退化为锥体，验算一致性。

---

## 综合速查索引

| 公式类别 | 关键词 | 所在节 |
|----------|--------|--------|
| 向量加减法 | 首尾 / 同起点 | 1.2 |
| 向量共线判定 | $x_1 y_2 - x_2 y_1 = 0$ | 1.4 |
| 数量积三用途 | 夹角 / 垂直 / 投影 | 2.1 |
| 三点共线 | $\overrightarrow{OC} = \lambda\overrightarrow{OA}+(1-\lambda)\overrightarrow{OB}$ | 2.2 |
| 直线 5 种形式 | 点斜 / 斜截 / 两点 / 截距 / 一般 | 3.2 |
| 直线垂直 | $k_1 k_2 = -1$ 或 $A_1A_2+B_1B_2=0$ | 3.3 |
| 点到直线距离 | $\dfrac{|Ax_0+By_0+C|}{\sqrt{A^2+B^2}}$ | 3.4 |
| 圆标准方程 | $(x-a)^2+(y-b)^2=r^2$ | 4.1 |
| 弦长公式 | $2\sqrt{r^2-d^2}$ | 4.2 |
| 圆圆 5 位置 | $d$ vs $r_1 \pm r_2$ | 4.3 |
| 线面垂直判定 | ⊥ 两相交直线 | 8.3 |
| 二面角向量法 | 法向量夹角（取绝对值） | 8.4 |
| 球表面积体积 | $4\pi R^2$ / $\dfrac{4}{3}\pi R^3$ | 8.5 |
