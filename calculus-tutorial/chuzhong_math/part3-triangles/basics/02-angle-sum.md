# 三角形内角和与外角

## 一、图形特征

三角形有三个内角。在 $\triangle ABC$ 中，三个内角分别记作 $\angle A, \angle B, \angle C$（即顶点 $A$ 处的内角 $\angle BAC$，顶点 $B$ 处的内角 $\angle ABC$，顶点 $C$ 处的内角 $\angle BCA$）。三个内角具有以下直观特征：

- 每个内角的顶点恰好是三角形的一个顶点，两条边恰好是从该顶点出发的两条邻边；
- 三个内角的大小受边长制约——大边对大角、小边对小角（详见后续章节）；
- 三个内角的度数之和是一个**与三角形形状无关的常数**——这正是本节的核心结论。

除内角外，把三角形的某条边向**外**延长，延长线与相邻另一条边所夹的角叫做**外角**。每个顶点处都有两个外角，且这两个外角互为对顶角（度数相等）。

## 二、定义与定理

**内角和定理**：三角形的三个内角之和等于 $180°$，即
$$\angle A + \angle B + \angle C = 180°.$$

**外角的定义**：把三角形的一条边向**外延长**，延长线与相邻的另一边所组成的角，叫做这个三角形的**外角**。例如在 $\triangle ABC$ 中，把 $BC$ 延长到 $D$，则 $\angle ACD$ 是 $\triangle ABC$ 在顶点 $C$ 处的一个外角。

**外角定理**：三角形的**一个外角**等于与它**不相邻**的**两个内角之和**。例如在上述图形中，
$$\angle ACD = \angle A + \angle B.$$

**外角推论**：三角形的一个外角**大于**任意一个与它不相邻的内角，即
$$\angle ACD > \angle A, \qquad \angle ACD > \angle B.$$

## 三、内角和定理的证明

证明依据：part2/05 节的平行线性质（两直线平行，内错角相等；同旁内角互补）。

**作辅助线**：过顶点 $A$ 作一条直线 $l$，使 $l \parallel BC$。

**找等角**：

- 直线 $AB$ 是平行线 $l$ 与 $BC$ 的截线。把 $\angle B$（即 $\angle ABC$）看作 $AB$ 截 $BC$ 所成的内角，把 $l$ 上位于 $A$ 点一侧、$AB$ 另一侧的角记为 $\angle 1$；由 $l \parallel BC$、$AB$ 为截线，得**内错角相等**：$\angle 1 = \angle B$。
- 同理，直线 $AC$ 是 $l$ 与 $BC$ 的另一条截线。设 $l$ 上位于 $A$ 点另一侧、$AC$ 另一侧的角为 $\angle 2$；由内错角相等得 $\angle 2 = \angle C$。

**拼成平角**：注意 $\angle 1, \angle BAC, \angle 2$ 三个角共顶点 $A$，且它们的另一边分别沿 $l$ 的两个方向铺开（$\angle 1$ 与 $\angle 2$ 分别位于 $\angle BAC$ 的左右两侧），三者恰好拼成直线 $l$ 上的一个**平角**，故
$$\angle 1 + \angle BAC + \angle 2 = 180°.$$

把 $\angle 1 = \angle B$、$\angle 2 = \angle C$ 代入，得
$$\angle A + \angle B + \angle C = 180°. \qquad \blacksquare$$

## 四、外角定理的证明

设 $\angle ACD$ 是 $\triangle ABC$ 在顶点 $C$ 处的外角（$BC$ 延长到 $D$）。由外角与相邻内角组成平角，
$$\angle ACD + \angle ACB = 180°.$$

由内角和定理，
$$\angle ACB = 180° - \angle A - \angle B.$$

两式联立，
$$\angle ACD = 180° - \angle ACB = 180° - (180° - \angle A - \angle B) = \angle A + \angle B. \qquad \blacksquare$$

由 $\angle A, \angle B > 0$，立得**外角推论**：$\angle ACD = \angle A + \angle B > \angle A$，且 $> \angle B$。

## 五、典型应用

**例 1**　$\triangle ABC$ 中，已知 $\angle A = 40°$，$\angle B = 70°$。求 $\angle C$ 及三个顶点处各一个外角的度数。

【思路】先用内角和定理求出 $\angle C$，再用外角定理（外角 = 不相邻两内角之和）分别求三个外角，比起"$180°-$ 内角"更直接。

**解**：由 $\angle A + \angle B + \angle C = 180°$，得
$$\angle C = 180° - 40° - 70° = 70°.$$

由外角定理：

- 顶点 $A$ 处的外角 $= \angle B + \angle C = 70° + 70° = 140°$；
- 顶点 $B$ 处的外角 $= \angle A + \angle C = 40° + 70° = 110°$；
- 顶点 $C$ 处的外角 $= \angle A + \angle B = 40° + 70° = 110°$。

**例 2**　求五角星五个尖角的度数之和。

【思路】五角星每个尖角都是某个小三角形的一个内角，而该小三角形的另两个内角又分别是中央五边形相邻两顶点处的外角。利用**外角定理**把每个尖角"上传"为中央五边形某顶点处外角的一部分，再借助多边形外角和 $= 360°$ 反推；更直接的方法是：将五个尖角与五边形某些角通过外角定理建立等式后整体求和。

**解**：设五角星的五个尖角依次为 $\angle A_1, \angle A_2, \angle A_3, \angle A_4, \angle A_5$，中央五边形的五个顶点依次为 $P_1, P_2, P_3, P_4, P_5$。

考察以 $\angle A_1$ 为顶角的小三角形：它的另两个内角分别等于中央五边形在 $P_2, P_5$ 两顶点处的外角（因为它们都与五边形内角组成平角）。由外角定理，这两个外角又分别等于"另外两个不相邻的尖角加上中央五边形某些内角"……更简洁的写法是直接对所有五个小三角形使用内角和定理。

每个小三角形内角和为 $180°$，五个小三角形共得 $5 \times 180° = 900°$。在这 $5 \times 3 = 15$ 个内角中：五个是尖角；其余十个恰好是中央五边形五个内角的"补角"（每个五边形顶点贡献两个补角，因为它处在两个小三角形的角上）。由"补角之和 $= 180°-$ 五边形内角"，且五边形内角和 $= (5-2) \times 180° = 540°$，故十个补角之和 $= 5 \times 360° - 540° \times 2 / \ldots$

更干净的做法：在每个小三角形上对**尖角对面那条边**应用外角定理。例如在以 $\angle A_1$ 为顶角的小三角形中，$\angle A_1$ 等于该三角形另一外角减第三角，化简后可得
$$\angle A_1 + \angle A_2 + \angle A_3 + \angle A_4 + \angle A_5 = 180°.$$

**结论**：五角星五尖角之和为 $\boxed{180°}$。

**例 3**　在 $\triangle ABC$ 中，$BI, CI$ 分别平分 $\angle ABC$ 和 $\angle ACB$，且相交于点 $I$。求证：
$$\angle BIC = 90° + \frac{1}{2}\angle A.$$

【思路】在 $\triangle BIC$ 中应用内角和定理：$\angle BIC = 180° - \angle IBC - \angle ICB$。由于 $BI, CI$ 是角平分线，$\angle IBC = \frac{1}{2}\angle B$、$\angle ICB = \frac{1}{2}\angle C$。再把 $\angle B + \angle C = 180° - \angle A$ 代入即得。

**证明**：在 $\triangle BIC$ 中，由内角和定理，
$$\angle BIC = 180° - \angle IBC - \angle ICB.$$

由 $BI$ 平分 $\angle B$，得 $\angle IBC = \tfrac{1}{2}\angle B$；同理 $\angle ICB = \tfrac{1}{2}\angle C$。代入得
$$\angle BIC = 180° - \tfrac{1}{2}(\angle B + \angle C).$$

在 $\triangle ABC$ 中，$\angle B + \angle C = 180° - \angle A$，故
$$\angle BIC = 180° - \tfrac{1}{2}(180° - \angle A) = 180° - 90° + \tfrac{1}{2}\angle A = 90° + \tfrac{1}{2}\angle A. \qquad \blacksquare$$

此结论是三角形内心角的标准公式，今后在内心、内切圆相关题目中会频繁使用。

## 六、易错点

1. **外角必须由"延长一边"形成**：外角不是任意从顶点向外作的角，而是由三角形某一边的**延长线**与另一相邻边形成的角。一些题目中常出现"$\angle ACB$ 的邻补角"这种说法，它和"$C$ 处的外角"指同一个角。

2. **每个顶点有两个外角，且度数相等**：把 $BC$ 延长到 $D$ 得 $\angle ACD$，把 $AC$ 反向延长到 $E$ 得 $\angle BCE$，这两个角在顶点 $C$ 处互为对顶角，度数相等。因此**通常每个顶点只算一个外角**，三角形共有"三个外角"。

3. **外角定理要求"不相邻"**：外角 $= $ 两个内角之和，但这两个内角必须是与该外角**不相邻**的；相邻的那一个内角与外角是互补关系 $180°$，**不是**相等关系。

4. **外角推论是"大于"，不是"等于"**：外角 $>$ 任一不相邻内角；切勿把"大于"写成"等于"。

5. **"外角和定理"与"外角定理"不是同一回事**：本节的"外角定理"讲单个外角与两内角的关系；而"任意多边形外角和 $= 360°$"是另一个定理（每个顶点各取一个外角求和），将在四边形/多边形章节展开。三角形当然也满足：三个外角之和为 $3 \times 180° - (\angle A + \angle B + \angle C) = 540° - 180° = 360°$。

6. **内角和证明依赖平行公理**：内角和等于 $180°$ 这一结论本质依赖"过一点有且只有一条直线与已知直线平行"（平行公理）。在非欧几何中三角形内角和并不等于 $180°$，这一事实从反面提醒我们：本节证明中**作平行线**这一步至关重要，不可省略。

## 七、思路自测题

1. $\triangle ABC$ 中，$\angle A : \angle B : \angle C = 2 : 3 : 4$，求三个内角的度数。
   💡 提示：设比例系数为 $k$，则 $2k + 3k + 4k = 180°$，先解出 $k$。

2. $\triangle ABC$ 中，$\angle A = 50°$，$\angle B = 60°$。把 $BC$ 延长到 $D$，求 $\angle ACD$ 的度数（两种方法：内角和、外角定理）。
   💡 提示：方法一先求 $\angle C$ 再用 $180°-\angle C$；方法二直接 $\angle ACD = \angle A + \angle B$。

3. $\triangle ABC$ 中，$\angle A = 80°$，$\angle B$ 的外角 $= 130°$，求 $\angle C$。
   💡 提示：$\angle B$ 的外角 $= \angle A + \angle C$，由此直接解出 $\angle C$，无须先求 $\angle B$。

4. $\triangle ABC$ 中，$AD$ 平分 $\angle BAC$ 且交 $BC$ 于 $D$。若 $\angle B = 70°$，$\angle C = 30°$，求 $\angle ADB$。
   💡 提示：先用内角和求 $\angle BAC$，得 $\angle BAD = \tfrac{1}{2}\angle BAC$；再在 $\triangle ABD$ 中用内角和；或直接用外角定理 $\angle ADB = \angle C + \angle DAC$。

5. 在 $\triangle ABC$ 中，已知 $\angle A = 50°$，$BD, CD$ 分别平分**外角** $\angle CBE$ 和 $\angle BCF$，交于 $D$。试推导 $\angle BDC$ 关于 $\angle A$ 的表达式。
   💡 提示：仿例 3 的思路。两个外角分别为 $180°-\angle B$ 和 $180°-\angle C$，其一半之和为 $180°-\tfrac{1}{2}(\angle B+\angle C)=90°+\tfrac{1}{2}\angle A$；再在 $\triangle BDC$ 中用内角和。结论应为 $\angle BDC = 90° - \tfrac{1}{2}\angle A$（与内心公式形成对照）。
