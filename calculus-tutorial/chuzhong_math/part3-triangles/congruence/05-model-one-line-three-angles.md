# 一线三等角（K 字模型）

> **一图速记**：一条直线上出现三个相等的角，则直线两侧的两个三角形**相似**（特殊情况下全等）。形如字母 K。

## 一、引入

> **题目**：直线 $l$ 上顺次有三点 $A, P, B$。在 $l$ 同侧作 $\angle PAC = \angle APD = \angle PBD = \alpha$（$C$ 与 $D$ 在 $l$ 同侧；$AC$、$PD$、$BD$ 分别是从 $A, P, B$ 引出的射线/线段）。求证：$\triangle PAC \sim \triangle BPD$。

（注：标准 K 字配置——$C$ 在线段 $AP$ 同侧的折线端，$D$ 也在 $l$ 同侧；线段 $CP$、$PD$ 在 $P$ 处构成"中间夹角" $\angle CPD$，但只要三个"贴 $l$"的角相等即可。）

请先停下来，自己动笔试一分钟。看到题目里塞了三个"$=\alpha$"的角、又都贴在同一条直线上，先别急着列条件——**这种"三角同侧贴直线"的结构本身就在大喊它的名字：K 字**。

## 二、思维路径还原（解题者的内心独白）

> "看到 $A, P, B$ 三点共线 + 同侧三个相等角——这是教科书里的 K 字模型，立刻反应：左右两侧的三角形相似。
>
> 要证 $\triangle PAC \sim \triangle BPD$，相似只需找两对对应角相等。
>
> 第一对最容易看：$\angle PAC = \alpha$、$\angle PBD = \alpha$（已知 $A$ 处和 $B$ 处贴 $l$ 的角都是 $\alpha$）—— $\angle A = \angle B$，第一对成立。
>
> 第二对找哪里？两个三角形里还各有一个角和顶点 $P$ 有关。
>
> 在 $\triangle PAC$ 里，$\angle APC$ 的位置是 $P$ 向 $C$ 的方向（与 $l$ 同侧）。看直线 $l$ 在 $P$ 处的平角分解：$\angle APC$（向 $C$）+ $\angle CPD$（中间）+ $\angle DPB$（向 $B$，即 $\triangle BPD$ 中的一个角）= $180°$。
>
> 而中间的 $\angle CPD$ 就是题目里给的第三个等于 $\alpha$ 的角！
>
> 所以 $\angle APC + \angle BPD = 180° - \alpha$。
>
> 看 $\triangle PAC$ 内角和：$\angle APC + \angle A + \angle ACP = 180°$，即 $\angle APC = 180° - \alpha - \angle ACP$。
>
> 看 $\triangle BPD$ 内角和：$\angle BPD = 180° - \alpha - \angle BDP$。
>
> 两式相加：$\angle APC + \angle BPD = 360° - 2\alpha - (\angle ACP + \angle BDP)$。
>
> 联合前面 $\angle APC + \angle BPD = 180° - \alpha$，得 $\angle ACP + \angle BDP = 180° - \alpha$。
>
> 这还不够直接……换个更巧的角度：用**三角形外角定理**。
>
> 在 $\triangle PAC$ 中，外角 $\angle CPB = \angle PAC + \angle ACP = \alpha + \angle ACP$（这是 $P$ 处沿 $l$ 向 $B$ 方向的角）。
>
> 同时 $\angle CPB = \angle CPD + \angle DPB = \alpha + \angle DPB$。
>
> 两式相等：$\angle ACP = \angle DPB$ —— 这就是第二对！
>
> 两组角相等 → AA → $\triangle PAC \sim \triangle BPD$。"

把这段内心独白读三遍。会做的人脑子里最关键的两次"跳"是：
1. **平角分解**——把 $P$ 处的 $180°$ 拆成"向 $C$ + 中间 $\alpha$ + 向 $B$"三段，瞬间把三个角串到一起；
2. **外角定理替代内角和**——一旦想到"外角 = 不相邻两内角之和"，第二对相等角直接弹出来，绕开了硬解方程的死胡同。

## 三、抽象成模型

把骨架抽出来，就是**一线三等角（K 字）模型**。

**图形特征**（识别条件，全部满足才算）：

- 一条直线 $l$ 上**顺次有三点** $A, P, B$；
- 在 $l$ 的**同侧**有两条线段（或射线）$CA$、$DB$，分别从 $A$、$B$ 出发；
- $P$ 处也有一条指向 $l$ 同侧的线段 $PC$（或 $PD$）；
- 三个角 $\angle PAC = \angle CPD = \angle DBP = \alpha$，**都贴在直线 $l$ 同侧**。

**核心结论**：

- $\triangle PAC \sim \triangle BPD$（注意对应关系：**$P \leftrightarrow B$、$A \leftrightarrow P$、$C \leftrightarrow D$**）；
- 由相似得对应边成比例：$\dfrac{PA}{BP} = \dfrac{AC}{PD} = \dfrac{PC}{BD}$；
- 当**再加一个等条件** $PA = BP$（即 $P$ 是 $AB$ 的中点）时，相似比为 $1$，**升级为全等** $\triangle PAC \cong \triangle BPD$。

**为什么叫"K 字"**：把直线 $l$ 横着画，三个 $\alpha$ 角同侧贴上去，$AC$、$PD$、$BD$ 三段连起来看，整体外形就像把字母 K 平躺下来。

**证明骨架（一句话版）**：用**三角形外角定理**——$\triangle PAC$ 在 $P$ 处的外角 $\angle CPB$ 一方面 $= \alpha + \angle ACP$，另一方面 $= \alpha + \angle DPB$，相减得 $\angle ACP = \angle DPB$；再加上 $\angle A = \angle B = \alpha$，**AA** 立得。

## 四、模型变形

K 字在中考里几乎从不"原样"出现，永远以下面这些**变形**面孔示人：

- **变形 1（最常见特例）**：$\alpha = 90°$ —— **一线三直角**模型（K 字的"直角版"，中考高频）。三个 $\alpha$ 全变成直角，外角定理那一步直接退化为"两锐角互余"，结论照样成立。
- **变形 2（升级为全等）**：一线三直角（或一般 K 字）**外加 $PA = BP$**——立刻从相似升级到 $\triangle PAC \cong \triangle BPD$。常用于求线段长度或证两段相等。
- **变形 3（嵌入正方形/矩形）**：把正方形的一条边当作 $l$，另两边端点向 $l$ 作垂线（或在边上找点向两邻边做 $90°$）—— **立刻形成一线三直角**。这是中考压轴题最爱的伪装。
- **变形 4（嵌入坐标系）**：动点在 $x$ 轴上滑动，向轴上两定点作直角，又是一线三直角，相似比直接给出坐标方程（part9 函数综合题会反复用到）。
- **本质**：K 字 = "**两侧三角形 + 中间 $\alpha$ 角**共享同一条直线"。三角形外角把"中间那个 $\alpha$"和"两侧 $\alpha$"串成等式——所有变形不过是中间那个 $\alpha$ 取了不同值，或被正方形/坐标系藏起来罢了。

## 五、思考路标（看到 X → 想到 Y）

把下面每一条都嚼到肌肉记忆为止——以后做题不用"想"，是"条件反射"。

- 看到一条直线上有**三个相等角**（同侧）→ **K 字相似**，去找两侧三角形；
- 看到一条直线上有**三个直角** → **一线三直角**（K 字直角特例），直接套相似；
- 看到**正方形/矩形内"一条边上一点向两邻边各引一段构成 $90°$"** → 一线三直角立刻成立；
- 看到 K 字结构 + **一段等条件**（如中点、等腰）→ 相似升级为**全等**；
- 看到 K 字相似要算线段长度 → 写**对应边比例** $\dfrac{PA}{BP} = \dfrac{AC}{PD}$，别写错对应关系；
- 看到动点在轴上滑动 + 两侧直角 → 一线三直角，列比例方程；
- 想用 AA 证相似但**少一个角**，且图中有"三角共线"结构 → 试**三角形外角定理**或"平角分三段"。

## 六、应用例题

下面三例只演示"**怎么用路标识别 + 怎么把模型套上去**"，请你照着把过程补完。

**例 1（基础 K 字相似 + 求线段比）**：$\alpha = 90°$，$PA = 3$，$PB = 4$，$AC = 2$，求 $BD$。

> **路标触发**："一线三直角" → $\triangle PAC \sim \triangle BPD$。**思路**：对应关系 $P\leftrightarrow B$、$A\leftrightarrow P$、$C\leftrightarrow D$，列对应边比例 $\dfrac{PA}{BP} = \dfrac{AC}{PD}$，即 $\dfrac{3}{4} = \dfrac{2}{PD}$，得 $PD = \dfrac{8}{3}$；再由 $\dfrac{PA}{BP} = \dfrac{PC}{BD}$ 并用勾股得到 $PC = \sqrt{PA^2 + AC^2} = \sqrt{13}$，故 $BD = \dfrac{4\sqrt{13}}{3}$。

**例 2（正方形内的一线三直角）**：正方形 $ABCD$ 边长 $4$，$E$ 在 $AB$ 上，$AE = 1$，$F$ 在 $BC$ 上，使得 $\angle DEF = 90°$。求 $BF$。

> **路标触发**："正方形 + 边上一点向两邻边构成 $90°$" → 一线三直角（直线就是 $AB$，三个直角是 $\angle DAE$、$\angle DEF$、$\angle EBF$）。**思路**：$\triangle ADE \sim \triangle BEF$，对应关系 $A\leftrightarrow B$、$D\leftrightarrow E$、$E\leftrightarrow F$。比例 $\dfrac{AD}{BE} = \dfrac{AE}{BF}$，即 $\dfrac{4}{3} = \dfrac{1}{BF}$，解得 $BF = \dfrac{3}{4}$。

**例 3（动点版）**：矩形 $ABCD$ 中，$AB = 6$，$BC = 8$。动点 $P$ 在边 $BC$ 上滑动，过 $P$ 作 $\angle APQ = 90°$，$Q$ 在 $CD$ 上。求证：随着 $P$ 移动，$\triangle ABP \sim \triangle PCQ$ 始终成立；并求当 $BP = 2$ 时 $CQ$ 的值。

> **路标触发**："动点 + 直角 + 矩形边" → 一线三直角（直线是 $BC$，三个直角是 $\angle B$、$\angle APQ$、$\angle C$）。**思路**：前半截照搬 K 字相似的证明（$\angle B = \angle APQ = \angle C = 90°$，外角定理推 $\angle BAP = \angle CPQ$，AA 即得）。$BP = 2$ 时，$PC = 6$；由 $\dfrac{AB}{PC} = \dfrac{BP}{CQ}$，即 $\dfrac{6}{6} = \dfrac{2}{CQ}$，得 $CQ = 2$。

## 七、思路自测题

做下面四题，做之前先在脑中默念一遍第五节的路标。

**自测 1**：直线 $l$ 上顺次有三点 $A, P, B$，$PA = PB = 5$，在 $l$ 同侧作 $\angle PAC = \angle APD = \angle PBD = 60°$，且 $AC = 3$。求 $BD$。

> 💡 提示：K 字 + $PA = PB$ → 升级为**全等** $\triangle PAC \cong \triangle BPD$，对应边 $AC = PD$、$PC = BD$。再在 $\triangle PAC$ 里用余弦定理或直接构造算 $PC$，即得 $BD$。

**自测 2**：正方形 $ABCD$ 中，$E$ 是 $BC$ 中点，$F$ 在 $CD$ 上使 $\angle AEF = 90°$。求 $\dfrac{DF}{FC}$。

> 💡 提示：识别——边 $BC$ 当 $l$，三直角 $\angle B$、$\angle AEF$、$\angle C$。$\triangle ABE \sim \triangle ECF$，设边长为 $2$，$BE = EC = 1$，$AB = 2$，比例算出 $CF = \dfrac{1}{2}$，故 $DF = \dfrac{3}{2}$，比值 $3$。

**自测 3（隐藏的 K 字）**：等边 $\triangle ABC$ 边长 $6$，$D$ 在 $BC$ 上，$BD = 2$。在 $\triangle ABC$ 外、以 $AD$ 为一边作等边 $\triangle ADE$（$E$ 与 $B$ 在 $AD$ 异侧）。过 $D$ 作直线交 $AB$ 于 $M$、交 $AC$ 于 $N$，使 $\angle MDN = 60°$。求证：$\triangle BDM \sim \triangle CND$。

> 💡 提示：题目**没有明说 K 字**，但你应当一眼识别——直线 $l$ 就是 $BC$，三个 $60°$ 的角是 $\angle B$、$\angle MDN$、$\angle C$（等边三角形的两个底角 + 给定的 $\angle MDN$）。三个 $60°$ 同侧贴在 $BC$ 上，正是 $\alpha = 60°$ 的 K 字模型。

**自测 4（一般版）**：直线 $l$ 上顺次有 $A, P, B$，同侧三个相等角 $\angle PAC = \angle APD = \angle PBD = \alpha$，$AC = m$，$PA = a$，$PB = b$。用 $m, a, b$ 表示 $PD$。

> 💡 提示：K 字相似 $\triangle PAC \sim \triangle BPD$，对应边 $\dfrac{PA}{BP} = \dfrac{AC}{PD}$，即 $\dfrac{a}{b} = \dfrac{m}{PD}$，故 $PD = \dfrac{bm}{a}$。这道题印证了"对应边比例由 $PA : PB$ 唯一决定"是模型的一般结论。

---

**回头看一眼"一图速记"**：

> 一条直线上出现三个相等的角，则直线两侧的两个三角形**相似**（特殊情况下全等）。形如字母 K。

如果现在你脑中能秒回这句话，并且能立刻在脑中画出"外角定理一步走"的 AA 证明——那么 K 字模型，你拿下了。
