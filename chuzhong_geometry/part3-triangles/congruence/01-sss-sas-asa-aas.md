# 全等三角形判定：SSS、SAS、ASA、AAS

## 一、图形特征

两个三角形如果能够完全重合（即经过平移、旋转、翻折之后可以一一叠合），它们的**三组对应边分别相等、三组对应角分别相等**。

直观上看，全等三角形的"形状"和"大小"都完全相同，只是位置或姿态可能不同。因此判定两个三角形全等，本质上就是要找到"足以决定一个三角形的最少条件"。一个三角形有 3 条边、3 个角共 6 个元素，但其实不必逐一比较——只要其中三个特定元素对应相等，往往就能锁定全等。

## 二、定义与判定定理

**全等的定义**：如果两个三角形能够完全重合，就称这两个三角形**全等**，记作

$$\triangle ABC \cong \triangle DEF.$$

记号中字母的**顺序非常关键**——它已经表明了对应关系：

$$A \leftrightarrow D,\quad B \leftrightarrow E,\quad C \leftrightarrow F.$$

**全等的性质**：若 $\triangle ABC \cong \triangle DEF$，则

- 对应边相等：$AB = DE,\ BC = EF,\ AC = DF$；
- 对应角相等：$\angle A = \angle D,\ \angle B = \angle E,\ \angle C = \angle F$。

**四大判定定理**：

| 判定 | 内容 | 关键词 |
|------|------|--------|
| **SSS**（边边边）| 三边对应相等 | 三边 |
| **SAS**（边角边）| 两边及其**夹角**对应相等 | 夹角必须在两边之间 |
| **ASA**（角边角）| 两角及其**夹边**对应相等 | 夹边必须在两角之间 |
| **AAS**（角角边）| 两角及其中一角的**对边**对应相等 | 边不在两角之间，但与其中一角相对 |

![全等四判定示意：SSS / SAS / ASA / AAS（已标记的元素是判定中相等的部分）](../../figures/svg/thm-congruence-criteria.svg)

**关键澄清：SSA 一般不成立**。

"两边及其中一边的对角"（非夹角）相等，不能判定全等。反例：取一段长度固定的线段 $AB$ 与一个固定的锐角 $\angle A$，再从 $B$ 处画一条长度固定但比较短的线段——它有可能与对面的射线交于**两个不同的点**，从而得到两个不同的三角形，这就是著名的"两解情形"。同样，直角与钝角下的 SSA 也并非总能判定全等（直角情形下我们另有 **HL** 定理用于直角三角形）。

## 三、判定背后的逻辑

四个判定并不是各自独立的"公理"，它们之间存在推导链条。

1. **SAS 通常作为公理**。两边及其夹角一旦确定，第三条边和另两个角就被完全决定（用余弦定理也可验证：第三边由 $a^2 + b^2 - 2ab\cos C$ 唯一决定）。教材中一般不证 SAS，而把它当作出发点。

2. **ASA 由 SAS + 内角和定理推出**。设两个三角形的两角对应相等，夹边也相等。由三角形内角和为 $180^\circ$，第三个角也对应相等。于是其中一对相等角的两条夹边——夹边本身（已相等）加上由第三角"反推"出的另一条边，配合夹角形成 SAS 结构（演示思路：先用内角和定理补出第三角，再用 SAS）。

3. **AAS 由 ASA + 内角和推出**。给定两角与其中一角的对边，由内角和定理可推出第三角，从而把"对边"重新解释为"另一对角的夹边"，回到 ASA。

4. **SSS 也可由 SAS 配合"三角形三边唯一决定"推出**。直观思路：固定两边，让夹角连续变化，第三边随之单调变化；故第三边一旦给定，夹角随之唯一确定，问题化归为 SAS。这一论证在中学层面用"重合法"完成：把一个三角形翻折后让两边贴合另一个三角形，利用等腰三角形性质证明第三个顶点重合。

可以看到，四个判定背后其实只有一条共同原则：**一个三角形被"够用的"三个独立元素唯一确定**。

## 四、典型应用

### 例 1：四种判定串讲

(1) **SSS**：已知 $AB = DE,\ BC = EF,\ AC = DF$，证 $\triangle ABC \cong \triangle DEF$。
【思路】直接套 SSS，三边对应相等即可。

(2) **SAS**：已知 $AB = DE,\ \angle B = \angle E,\ BC = EF$，证 $\triangle ABC \cong \triangle DEF$。
【思路】$\angle B$ 是 $AB$ 与 $BC$ 的**夹角**，符合 SAS 结构，直接套用。

(3) **ASA**：已知 $\angle A = \angle D,\ AB = DE,\ \angle B = \angle E$，证 $\triangle ABC \cong \triangle DEF$。
【思路】$AB$ 是 $\angle A$ 与 $\angle B$ 的**夹边**，套 ASA。

(4) **AAS**：已知 $\angle A = \angle D,\ \angle B = \angle E,\ BC = EF$，证 $\triangle ABC \cong \triangle DEF$。
【思路】$BC$ 是 $\angle A$ 的**对边**（不是夹边），两角加一对边，套 AAS。

### 例 2：SSA 陷阱

题目：在 $\triangle ABC$ 与 $\triangle DEF$ 中，已知 $AB = DE,\ BC = EF,\ \angle A = \angle D$（注意 $\angle A$ 不是 $AB$ 与 $BC$ 的夹角，而是 $BC$ 的对角）。能否判定两三角形全等？

【思路】**不能**。这正是 SSA 情形。构造反例：取 $\angle A = \angle D = 30^\circ$，$AB = DE = 4$，让 $BC = EF = 2.5$（比从 $B$ 到对边的垂线略长，又比 $AB$ 短）。从 $B$ 出发画半径 $2.5$ 的弧，会与从 $A$ 出发的另一条射线交于两点，对应两个不同形状的三角形——一个锐角、一个钝角——它们都满足上述三组条件，却并不全等。

因此遇到"两边一角"时务必看清那个角是不是**夹角**：是夹角才是 SAS，不是就要谨慎。

### 例 3：等腰三角形中的全等

题目：在 $\triangle ABC$ 中，$AB = AC$，$D, E$ 分别在 $AB, AC$ 上，且 $AD = AE$，连 $BE, CD$ 相交于 $P$。求证：$\triangle ABE \cong \triangle ACD$，并由此得到 $BE = CD$。

【思路】观察 $\triangle ABE$ 与 $\triangle ACD$：
- $AB = AC$（题设，等腰）；
- $\angle A = \angle A$（公共角）；
- $AE = AD$（题设）。

注意：$\angle A$ 恰好是 $AB$ 与 $AE$ 的夹角，也是 $AC$ 与 $AD$ 的夹角。因此满足 **SAS**，得 $\triangle ABE \cong \triangle ACD$。由全等的对应边相等，立即 $BE = CD$。

**进一步**：还可以由该全等导出 $\angle ABE = \angle ACD$，再结合 $\angle DBP = \angle ABC - \angle ABE = \angle ACB - \angle ACD = \angle ECP$，从而进一步研究交点 $P$ 的对称性（例如证 $PB = PC$、$PD = PE$ 等），这是中考几何综合题中的常见套路。

## 五、易错点

1. **SSA 不能用作判定！** 看到"两边和一个角"时，必须确认那个角是不是两边的**夹角**。是夹角才能用 SAS；若是对角，则需另寻条件（如再补一角化为 AAS、或换边化为 SSS）。

2. **"对应"由记号顺序决定**。$\triangle ABC \cong \triangle DEF$ 意味着 $A \leftrightarrow D$、$B \leftrightarrow E$、$C \leftrightarrow F$，对应边和对应角都由此读出。若写成 $\triangle ABC \cong \triangle EFD$，对应关系就完全不同。书写全等结论时，**字母顺序必须严格对应**，否则会导致后续推理出错。

3. **SAS 中的角必须是夹角**。例如题目中给了 $AB = A'B'$、$AC = A'C'$、$\angle B = \angle B'$，乍看是"两边一角"，但 $\angle B$ 并不是 $AB$ 与 $AC$ 的夹角（夹角应是 $\angle A$），所以**不能**直接套 SAS。

4. **AAS 与 ASA 不要混。** AAS 中的边是某一角的**对边**；ASA 中的边是两角的**夹边**。判断时盯住"这条边相对于已知两角的位置"。

5. **公共边、公共角、对顶角往往是"隐藏条件"**，在书写时不要忘记把它们写出来作为一项。

## 六、思路自测题

1. 在 $\triangle ABC$ 与 $\triangle DEF$ 中，$AB = DE$，$\angle A = \angle D$，$AC = DF$。可以判定全等吗？用哪个定理？

2. 在 $\triangle ABC$ 与 $\triangle DEF$ 中，$\angle A = \angle D$，$\angle C = \angle F$，$AB = DE$。可以判定全等吗？$AB$ 在两已知角中扮演什么角色？应用哪个定理？

3. **（非夹角陷阱）** 在 $\triangle ABC$ 与 $\triangle DEF$ 中，$AB = DE$，$BC = EF$，$\angle A = \angle D$。能否判定全等？请说明理由，并尝试给出反例。

4. 已知 $\triangle ABC \cong \triangle DEF$，且 $\angle A = 40^\circ$，$\angle B = 70^\circ$，$AC = 5$。求 $\angle F$ 与 $DF$。

5. 在四边形 $ABCD$ 中，$AB = CD$，$AD = CB$。连 $AC$。证明 $\triangle ABC \cong \triangle CDA$，并指出所用判定。
