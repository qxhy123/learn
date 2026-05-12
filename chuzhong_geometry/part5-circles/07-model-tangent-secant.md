# 切割线模型

> **一图速记**：圆外一点 $P$ 引一条切线 $PA$（$A$ 是切点）和一条割线 $PBC$（依次交圆于 $B, C$，$B$ 在 $P$ 与 $C$ 之间）$\Rightarrow$ $PA^2 = PB \cdot PC$（**切割线定理**）。

![切割线定理](../../figures/svg/model-tangent-secant.svg)

## 一、引入

> 圆 $O$ 外一点 $P$，$PA$ 切圆 $O$ 于点 $A$，过 $P$ 的割线交圆 $O$ 于 $B, C$ 两点（$B$ 在 $P$ 与 $C$ 之间）。已知 $PA = 6$，$PB = 4$，求 $PC$ 的长度。

![切割线引入：PA=6, PB=4, 求 PC](../figures/svg/ex-p5-07-0.svg)

这道题表面只是"圆外一点 + 一条切线 + 一条割线"的简单配置，但它背后藏着初中圆论里最重要的乘积关系之一——**切割线定理**。学完切线判定与切线长定理之后，是时候把"切线 + 割线"组合成一个能算长度的模型了。

## 二、思维路径还原

> "圆外一点引切线 + 割线 —— 这是经典的切割线结构。我隐约记得有一个结论 $PA^2 = PB \cdot PC$，但只记结论不行，必须懂为什么，否则换个图就用不出来。
>
> 怎么证？我先把图画清楚：圆 $O$，外点 $P$，切线 $PA$，割线 $PBC$。要得到长度的乘积关系，自然要往**相似三角形**靠拢。
>
> 我把 $A$ 与 $B$、$A$ 与 $C$ 连起来——出现 $\triangle PAB$ 和 $\triangle PAC$。要证 $PA^2 = PB \cdot PC$，等价于 $\dfrac{PA}{PB} = \dfrac{PC}{PA}$，正好就是这两个三角形对应边的比。
>
> 找相似的依据：$\angle P$ 是这两个三角形的公共角（$\angle APB$ 与 $\angle APC$ 是同一个角，因为 $B, C, P$ 在一条直线上）。共角已经一对了，再差一对角。
>
> 关键就在 $\angle PAB$：$PA$ 是切线、$AB$ 是弦——这是**弦切角**！弦切角定理告诉我们：弦切角等于它所夹弧（$\overarc{AB}$，不含 $C$ 的那段）所对的圆周角。而 $\overarc{AB}$ 所对的圆周角是 $\angle ACB$（顶点 $C$ 在另一段弧上）。
>
> 所以 $\angle PAB = \angle ACB = \angle PCA$。
>
> 两对角分别相等 $\Rightarrow$ AA $\Rightarrow$ $\triangle PAB \sim \triangle PCA$（务必按顶点对应顺序：$P \leftrightarrow P, A \leftrightarrow C, B \leftrightarrow A$）。
>
> 由对应边成比例：$\dfrac{PA}{PC} = \dfrac{PB}{PA}$，交叉相乘即 $PA^2 = PB \cdot PC$。结论被推出来了，不再靠死记。
>
> 回到题目：$6^2 = 4 \cdot PC$，即 $PC = 9$。"

## 三、抽象成模型

把上述推理剥离具体数字，提炼为**切割线模型**：

- **图形特征**：圆 $O$ + 圆外一点 $P$ + 一条切线 $PA$（切点 $A$）+ 一条割线 $PBC$（$B$ 近、$C$ 远）。
- **核心结论**：$PA^2 = PB \cdot PC$。
- **证法骨架**：连 $AB, AC$ $\Rightarrow$ 弦切角 $\angle PAB = \angle ACB$（part5/03 圆周角 + 弦切角定理）+ 共角 $\angle P$ $\Rightarrow$ $\triangle PAB \sim \triangle PCA$ $\Rightarrow$ 比例 $\Rightarrow$ 乘积。
- **本质（圆幂）**：圆外一点 $P$ 关于圆 $O$ 的"幂"恒等于 $PO^2 - r^2$。无论从 $P$ 引哪条切线或割线，都满足 $PA^2 = PB \cdot PC = PO^2 - r^2$，这就是**圆幂定理**的统一视角。

记忆口诀："**切线平方等于割线两段积**"。

## 四、模型变形

切割线只是圆幂定理的"切 + 割"形态。把切线换成另一条割线，或者把外点搬进圆内，公式就变形成姐妹版本：

- **两割线版本（割线-割线定理）**：从圆外一点 $P$ 引**两条割线** $PBC$、$PDE$（$B, D$ 近端，$C, E$ 远端），则
  $$PB \cdot PC = PD \cdot PE.$$
  证法同样靠 AA 相似——$\angle P$ 公共，$\angle PBD = \angle PEC$（圆内接四边形 $BCED$ 对角互补的等价：外角等于内对角）。

- **相交弦版本（点在圆内）**：圆**内**一点 $P$，过 $P$ 的两条弦 $AB$ 与 $CD$ 相交，则
  $$PA \cdot PB = PC \cdot PD.$$
  这就是 sim/04 已经见过的**相交弦定理**，证法是同弧圆周角 $\angle A = \angle D$（$\overarc{BC}$）+ 对顶角 $\Rightarrow \triangle APC \sim \triangle DPB$。

- **极限观点**：当割线绕 $P$ 旋转到与圆相切时，$B$ 与 $C$ 重合于切点 $A$，"两段乘积"退化为"切线长平方"——这就是切割线公式可以从割线-割线公式取极限得到。

**统一**：切割线、割线-割线、相交弦三条等式本质上都是**圆幂定理**在外点切 / 外点割 / 内点割三种位置下的不同表达。

## 五、思考路标

遇到圆 + 线段长度乘积的问题时，按下面的顺序快速识别：

- 看到**圆外一点 + 切线 + 割线** $\Rightarrow$ **切割线定理** $PA^2 = PB \cdot PC$。
- 看到**圆外一点 + 两条割线** $\Rightarrow$ **割线-割线定理** $PB \cdot PC = PD \cdot PE$。
- 看到**圆内一点 + 两弦相交** $\Rightarrow$ **相交弦定理** $PA \cdot PB = PC \cdot PD$。
- 题目出现 $\text{某段}^2 = \text{某段} \cdot \text{某段}$ 形式 $\Rightarrow$ 优先怀疑切割线背景。
- 想证两段乘积相等却没头绪 $\Rightarrow$ 试着找"圆 + 公共顶点 + 相交直线"的圆幂结构。
- 切线长定理（part5/06）和切割线定理常**联合出现**——前者给等长，后者给乘积。

## 六、应用例题

### 例 1（基础切割线）

如引入题：$PA$ 切圆 $O$ 于 $A$，割线 $PBC$ 中 $PA = 6, PB = 4$，求 $PC$。

![切割线例1：同引入题](../figures/svg/ex-p5-07-0.svg)

【思路】识别图形为切割线模型 $\Rightarrow$ $PA^2 = PB \cdot PC$ $\Rightarrow$ $36 = 4 PC$ $\Rightarrow$ $PC = 9$。注意 $BC = PC - PB = 9 - 4 = 5$。

### 例 2（两割线）

从圆外一点 $P$ 引两条割线：第一条依次交圆于 $A$（近）、$B$（远），$PA = 4, PB = 9$；第二条依次交圆于 $C$（近）、$D$（远），$PC = 3$。求 $PD$。

![切割线例2：两割线](../figures/svg/ex-p5-07-2.svg)

【思路】图形为"圆外点 + 两割线" $\Rightarrow$ 割线-割线定理 $PA \cdot PB = PC \cdot PD$ $\Rightarrow$ $4 \times 9 = 3 \times PD$ $\Rightarrow$ $PD = 12$。

### 例 3（相交弦）

圆内一点 $P$，过 $P$ 的弦 $AB$ 被 $P$ 分成 $PA = 2, PB = 6$；另一条过 $P$ 的弦 $CD$ 中 $PC = 3$。求 $PD$。

![切割线例3：圆内相交弦](../figures/svg/ex-p5-07-3.svg)

【思路】图形为"圆内点 + 两弦" $\Rightarrow$ 相交弦定理 $PA \cdot PB = PC \cdot PD$ $\Rightarrow$ $2 \times 6 = 3 \times PD$ $\Rightarrow$ $PD = 4$。

## 七、思路自测题

1. 圆 $O$ 外一点 $P$，$PT$ 切圆于 $T$，割线交圆于 $M, N$（$M$ 近 $P$）。若 $PM = 2, MN = 6$，求 $PT$。
   【提示】先求 $PN = PM + MN = 8$，再用切割线 $PT^2 = PM \cdot PN$。

2. 从 $P$ 引圆的两条割线：一条交圆于 $A, B$，$PA = 3, AB = 5$；另一条交圆于 $C, D$，$PC = 4$。求 $CD$。
   【提示】$PB = 8$，由 $PA \cdot PB = PC \cdot PD$ 求 $PD$，再得 $CD = PD - PC$。

3. 圆内一点 $P$ 处两弦 $AB, CD$ 相交，$PA = 4, PB = 3, PC = 6$，求 $PD$ 与弦 $CD$ 的长度。
   【提示】相交弦 $PA \cdot PB = PC \cdot PD$；$CD = PC + PD$。

4. $PA$ 切圆 $O$ 于 $A$，过 $P$ 的割线过圆心 $O$，交圆于 $M, N$（$M$ 近 $P$）。若 $PA = 6$，圆半径 $r = 4$，求圆心距 $PO$。
   【提示】设 $PO = d$，则 $PM = d - r, PN = d + r$，由切割线 $PA^2 = PM \cdot PN = d^2 - r^2$ $\Rightarrow$ $36 = d^2 - 16$ $\Rightarrow$ $d = \sqrt{52} = 2\sqrt{13}$。这恰好印证了"圆幂 $= d^2 - r^2$"的结论。
