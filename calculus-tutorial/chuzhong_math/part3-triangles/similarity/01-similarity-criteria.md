# 相似三角形判定

## 一、图形特征

相似三角形是初中几何中继全等之后最重要的一类对应关系。直观上，两个三角形相似意味着它们的**形状完全相同，但大小可以不同**——可以把其中一个看作另一个按照某个比例放大或缩小后的结果。

具体来说，两个三角形相似具有以下两个本质特征：

1. **对应角相等**：三组对应的内角分别相等；
2. **对应边成比例**：三组对应的边的比值相等，这个公共比值称为**相似比**，常记作 $k$。

形状相同的本质，就在于"角不变 + 边等比缩放"。当 $k=1$ 时，两个三角形不仅形状相同，大小也相同，即退化为全等三角形——这表明**全等是相似的一种特殊情况**。

从图形上看，如果把一个三角形以某点为中心做位似变换（放大或缩小若干倍），所得三角形与原三角形必然相似；反之，任何一对相似三角形之间都可以通过"平移 + 旋转 + 缩放"建立对应关系。

## 二、定义与判定

### 1. 定义

若 $\triangle ABC$ 与 $\triangle DEF$ 满足：

$$\angle A = \angle D,\quad \angle B = \angle E,\quad \angle C = \angle F$$

$$\frac{AB}{DE} = \frac{BC}{EF} = \frac{CA}{FD} = k$$

则称 $\triangle ABC$ 与 $\triangle DEF$ **相似**，记作 $\triangle ABC \sim \triangle DEF$，$k$ 称为**相似比**。

> 注意：书写 $\triangle ABC \sim \triangle DEF$ 时，字母的顺序就是对应关系——$A \leftrightarrow D$，$B \leftrightarrow E$，$C \leftrightarrow F$，不能随意调换。

### 2. 三大判定定理

直接按定义验证需要"三角 + 三边"共六项条件，过于繁琐。实际上下列任一组条件已足够：

- **AA（两角对应相等）**：若 $\angle A = \angle D$，$\angle B = \angle E$，则 $\triangle ABC \sim \triangle DEF$。
- **SAS（两边成比例且夹角相等）**：若 $\dfrac{AB}{DE} = \dfrac{AC}{DF}$ 且 $\angle A = \angle D$，则 $\triangle ABC \sim \triangle DEF$。
- **SSS（三边对应成比例）**：若 $\dfrac{AB}{DE} = \dfrac{BC}{EF} = \dfrac{CA}{FD}$，则 $\triangle ABC \sim \triangle DEF$。

### 3. 直角三角形的特殊推论

对于直角三角形，由于已经有一组直角相等，只需再有**一组锐角相等**即可判定相似（这是 AA 判定的直接推论）。此外，直角三角形还有一个独特的判定：

- 若两直角三角形的**斜边与一直角边成比例**，则两三角形相似（HL 型相似，可由 SSS 或勾股定理导出）。

### 4. 相似三角形的性质

设 $\triangle ABC \sim \triangle DEF$，相似比为 $k$，则：

- **对应边之比 = $k$**；
- **对应高、对应中线、对应角平分线之比 = $k$**；
- **周长之比 = $k$**；
- **面积之比 = $k^2$**。

面积比为 $k^2$ 的直观原因：面积是"长 $\times$ 宽"型的二维量，每个线性维度都按 $k$ 缩放，因此面积按 $k^2$ 缩放。

## 三、AA 为何足以

三角形的内角和恒为 $180^\circ$，这是一个非常强的约束。若已知 $\angle A = \angle D$ 且 $\angle B = \angle E$，则：

$$\angle C = 180^\circ - \angle A - \angle B = 180^\circ - \angle D - \angle E = \angle F$$

也就是说**第三角自动相等**，无需额外验证。

进一步，由"三角对应相等"可以推出"三边对应成比例"——这是因为一旦三个内角确定，三角形的"形状"就被唯一确定了，剩下的只是整体大小（比例因子）。因此 AA 实际上隐含了 SSS 中的比例关系，只是把"边的信息"全部交给了角去刻画。

这与全等判定形成鲜明对比：全等需要至少一组边的信息（如 SAS、ASA、SSS、AAS），单靠 AAA 是不够的——因为相同的三个角可以对应任意大小的三角形。**全等关心绝对尺寸，相似只关心形状**，因此相似的判定门槛更低。

## 四、典型应用

### 例 1（基础 AA 求线段）

如图，$\triangle ABC$ 中，$DE \parallel BC$，$D$ 在 $AB$ 上，$E$ 在 $AC$ 上，$AD = 3$，$DB = 2$，$BC = 10$，求 $DE$。

**【思路】** $DE \parallel BC$ 给出一组同位角相等，再加上公共角 $\angle A$，即可用 AA 得 $\triangle ADE \sim \triangle ABC$。相似比由 $\dfrac{AD}{AB} = \dfrac{3}{5}$ 直接读出，从而 $DE = \dfrac{3}{5} BC$。

**解：** 由 $DE \parallel BC$ 得 $\angle ADE = \angle ABC$，又 $\angle A$ 公共，故 $\triangle ADE \sim \triangle ABC$。相似比

$$k = \frac{AD}{AB} = \frac{3}{3+2} = \frac{3}{5}$$

因此 $DE = k \cdot BC = \dfrac{3}{5} \times 10 = 6$。

### 例 2（由相似比求面积，反推线段）

$\triangle ABC \sim \triangle A'B'C'$，已知两者面积之比为 $9 : 16$，且 $AB = 6$，求 $A'B'$。

**【思路】** 面积比 $= k^2$，所以由 $k^2 = \dfrac{9}{16}$ 得 $k = \dfrac{3}{4}$（取正值）。再用相似比的定义 $\dfrac{AB}{A'B'} = k$ 反解 $A'B'$。关键提醒：开方时只取正值，因为相似比是长度比。

**解：** 设相似比 $\dfrac{AB}{A'B'} = k$，则

$$k^2 = \frac{S_{\triangle ABC}}{S_{\triangle A'B'C'}} = \frac{9}{16} \implies k = \frac{3}{4}$$

故 $A'B' = \dfrac{AB}{k} = \dfrac{6}{3/4} = 8$。

### 例 3（相似与全等的关系）

证明：若 $\triangle ABC \sim \triangle DEF$ 且相似比 $k = 1$，则 $\triangle ABC \cong \triangle DEF$。

**【思路】** 全等的标志是"对应角相等 + 对应边相等"。相似已经保证了对应角相等；当 $k=1$ 时，对应边比值为 $1$，等价于对应边相等。因此相似的条件直接升级为全等。

**证明：** 由 $\triangle ABC \sim \triangle DEF$ 知 $\angle A = \angle D$，$\angle B = \angle E$，$\angle C = \angle F$，并且

$$\frac{AB}{DE} = \frac{BC}{EF} = \frac{CA}{FD} = k = 1$$

故 $AB = DE$，$BC = EF$，$CA = FD$。由 SSS 知 $\triangle ABC \cong \triangle DEF$。

这说明**全等是相似在 $k=1$ 时的特例**，相似是更一般的"形状相同"概念。

## 五、易错点

1. **对应关系按字母顺序书写。** 写 $\triangle ABC \sim \triangle DEF$ 就意味着 $A \leftrightarrow D$、$B \leftrightarrow E$、$C \leftrightarrow F$。如果只是"两个三角形相似"但对应关系不同，必须重新排列字母，比如 $\triangle ABC \sim \triangle EFD$ 表示的对应关系完全不同。这一点在写比例式时尤为关键。

2. **SSA 不能判定相似。** 与全等类似，"两边成比例 + 其中一边的对角相等"无法唯一确定三角形形状（存在"摆动"的歧义情形），因此 SSA 既不能判定全等，也不能判定相似。唯一例外是直角三角形的 HL 型，因为直角排除了歧义。

3. **周长比与面积比不要混淆。** 周长是一维量，比值 $= k$；面积是二维量，比值 $= k^2$。常见错误是把面积比也当作 $k$，或者反过来在已知面积比时忘了开方求 $k$。

4. **相似比的方向。** $\triangle ABC \sim \triangle DEF$ 时，相似比 $\dfrac{AB}{DE}$ 与 $\dfrac{DE}{AB}$ 互为倒数。题目中"$\triangle ABC$ 与 $\triangle DEF$ 的相似比"通常指前者比后者，写比例式前要明确方向。

5. **平行不等于相似的全部条件。** $DE \parallel BC$ 只给出一组角相等，必须再配合公共角或另一组角才能用 AA。不要直接由"平行"跳到"相似"而省略论证。

## 六、思路自测题

1. $\triangle ABC$ 中，$D$ 在 $AB$ 上，$E$ 在 $AC$ 上，且 $\angle ADE = \angle ACB$。证明 $\triangle ADE \sim \triangle ACB$，并写出对应边的比例式。

   💡 提示：注意公共角 $\angle A$ 与已知的角相等，用 AA。写比例式时严格按字母顺序：$\dfrac{AD}{AC} = \dfrac{AE}{AB} = \dfrac{DE}{CB}$。

2. 两个相似三角形的相似比为 $2 : 3$，较小三角形的周长为 $24$，求较大三角形的周长。

   💡 提示：周长比 $=$ 相似比 $= 2:3$，所以较大三角形周长 $= 24 \times \dfrac{3}{2}$。

3. 两个相似三角形的面积之比为 $25 : 49$，若较小三角形的某边长为 $10$，求较大三角形对应边长。

   💡 提示：先由面积比开方得相似比 $5:7$，再用相似比反推边长 $10 \times \dfrac{7}{5} = 14$。

4. $\triangle ABC$ 中，$\angle C = 90^\circ$，$CD \perp AB$ 于 $D$。证明 $\triangle ACD \sim \triangle ABC \sim \triangle CBD$。

   💡 提示：在每对三角形中找两组相等的角。$\triangle ACD$ 与 $\triangle ABC$ 共用 $\angle A$ 且都有直角；$\triangle ABC$ 与 $\triangle CBD$ 共用 $\angle B$ 且都有直角。这是"射影定理"的几何来源。

5. 判断命题真假并说明理由：若两个三角形有两组边成比例且有一组角相等，则它们相似。

   💡 提示：命题为**假**。SAS 要求相等的角是两条成比例边的**夹角**，若相等的角不是夹角（即 SSA 型），则无法判定相似。可以尝试构造反例：两边比例相同但摆动角度不同的两个三角形。
