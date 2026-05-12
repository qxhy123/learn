# 共角共边相似模型

> **一图速记**：两个三角形**共一个角**，且夹这个角的两边**成比例** → 相似（SAS 相似的可视化版本）。**关键**：比例方向决定对应关系。

## 一、引入

$\triangle ABC$ 中，$D$ 在 $AB$ 上，$E$ 在 $AC$ 上。已知 $\dfrac{AD}{AC} = \dfrac{AE}{AB}$（注意：这是**交叉**比例，$D$ 配 $C$、$E$ 配 $B$，不是 $D$ 配 $B$）。证 $\triangle ADE \sim \triangle ACB$（注意 $\sim$ 后字母顺序是 $ACB$ 不是 $ABC$）。

这个题目乍一看和 A 字模型一模一样——同一张图，同样两点 $D, E$，同样涉及 $AD, AE, AB, AC$ 四条线段的比例。但**结论的字母顺序变了**：不是 $\triangle ADE \sim \triangle ABC$，而是 $\triangle ADE \sim \triangle ACB$。这种"形似神不似"的微妙差别，正是本节要厘清的核心。

## 二、思维路径还原

> "题里出现 $\triangle ABC$ 和它内部两点 $D, E$，且涉及四条线段 $AD, AE, AB, AC$ 的比例——**共角共边**信号。
>
> 关键：仔细看比例式 $\dfrac{AD}{AC} = \dfrac{AE}{AB}$，与 A 字模型的 $\dfrac{AD}{AB} = \dfrac{AE}{AC}$ **不同**。
>
> A 字模型里，$D$ 在 $AB$ 上，对应到 $AB$ 整体；$E$ 在 $AC$ 上，对应到 $AC$ 整体——直接比例 $\to \triangle ADE \sim \triangle ABC$。
>
> 本题是**交叉**：$AD/AC$ 与 $AE/AB$——暗示对应关系反转。
>
> 共角是 $\angle A$（两个三角形都有它）。SAS 相似要求'共角 + 夹角两边成比例'。
>
> 第一个三角形 $\triangle ADE$ 里夹 $\angle A$ 的两边是 $AD, AE$；
>
> 第二个三角形 $\triangle ACB$（注意字母顺序）里夹 $\angle A$ 的两边是 $AC, AB$。
>
> $\dfrac{AD}{AC} = \dfrac{AE}{AB}$——正好是两组**对应**边的比例！
>
> SAS 相似 $\to \triangle ADE \sim \triangle ACB$（按字母对应：$A \leftrightarrow A, D \leftrightarrow C, E \leftrightarrow B$）。"

## 三、抽象成模型

- **图形特征**：两个三角形（可重叠或一个嵌在另一个内部）**共一个顶角**，且夹这个角的两边**成比例**。
- **结论**：相似（SAS 相似的图形化呈现）。
- **对应关系取决于比例的写法**：
  - **直接**比例 $\dfrac{AD}{AB} = \dfrac{AE}{AC}$ $\to$ $\triangle ADE \sim \triangle ABC$（A 字模型，$D \leftrightarrow B, E \leftrightarrow C$）
  - **交叉**比例 $\dfrac{AD}{AC} = \dfrac{AE}{AB}$ $\to$ $\triangle ADE \sim \triangle ACB$（本节模型，$D \leftrightarrow C, E \leftrightarrow B$）
- **等价表达**：交叉比例 $\dfrac{AD}{AC} = \dfrac{AE}{AB}$ $\Leftrightarrow$ 线段乘积 $AB \cdot AD = AC \cdot AE$。

**核心心法**：分母决定"配谁"。$AD$ 的分母是 $AC$，那么 $D$ 在新三角形里就对应 $C$；$AE$ 的分母是 $AB$，那么 $E$ 就对应 $B$。

## 四、模型变形

- **圆中相交弦定理萌芽**：两弦在圆内相交于 $P$，$PA \cdot PB = PC \cdot PD$——由共角共边相似推出（part5 圆中详证）。
- **割线定理**：圆外一点引两割线，$PA \cdot PB = PC \cdot PD$——同样基于共角共边相似。
- **切割线定理**：圆外一点引一切线一割线，$PT^2 = PA \cdot PB$——共角共边相似的特例。
- **退化情形**：当 $D$ 在 $AB$ 上、$E$ 在 $AC$ 延长线上（交叉变为同向）时仍然适用，因为 $\angle A$ 仍然是两三角形的共角。
- **"翻折"视角**：把 $\triangle ACB$ 沿 $\angle A$ 的角平分线翻折，恰好与 $\triangle ADE$ 在 $\angle A$ 处贴合——直观看出"交叉对应"的本质。

## 五、思考路标

- 看到**共顶角 + 比例线段** $\to$ 共角共边相似。
- 看到**线段乘积** $a \cdot b = c \cdot d$（且四段共一个端点） $\to$ 共角共边。
- 比例**直接型** $\dfrac{AD}{AB} = \dfrac{AE}{AC}$ $\to$ A 字模型。
- 比例**交叉型** $\dfrac{AD}{AC} = \dfrac{AE}{AB}$ $\to$ 本节模型（对应反转）。
- **圆中两弦相交**、**圆外引两割线/切割线** $\to$ 共角共边相似背景。
- 写相似结论时一定**先确定对应顶点**再写字母顺序，切忌习惯性地写成 $\triangle ABC$。

## 六、应用例题

**例 1（基础——交叉比例证相似）**：$\triangle ABC$ 中，$D$ 在 $AB$ 上，$E$ 在 $AC$ 上，$\dfrac{AD}{AC} = \dfrac{AE}{AB}$。证 $\triangle ADE \sim \triangle ACB$。

【思路】共角 $\angle A$ 现成；条件 $\dfrac{AD}{AC} = \dfrac{AE}{AB}$ 即 $\triangle ADE$ 夹 $\angle A$ 两边 $AD, AE$ 与 $\triangle ACB$ 夹 $\angle A$ 两边 $AC, AB$ 对应成比例。SAS 相似直接得证。

**例 2（乘积关系反向使用）**：$\triangle ABC$ 中 $D$ 在 $AB$ 上，$E$ 在 $AC$ 上。若 $\triangle ADE \sim \triangle ACB$，证 $AB \cdot AD = AC \cdot AE$。

【思路】这是例 1 的逆命题。由 $\triangle ADE \sim \triangle ACB$，对应顶点 $A \leftrightarrow A, D \leftrightarrow C, E \leftrightarrow B$，故对应边 $AD \leftrightarrow AC, AE \leftrightarrow AB$，得 $\dfrac{AD}{AC} = \dfrac{AE}{AB}$，交叉相乘即 $AB \cdot AD = AC \cdot AE$。

**例 3（圆中相交弦定理）**：圆 $O$ 中弦 $AB, CD$ 相交于 $P$。连 $AC, BD$。证 $\triangle PAC \sim \triangle PDB$，从而 $PA \cdot PB = PC \cdot PD$。

【思路】$\angle APC = \angle DPB$（对顶角），$\angle PAC = \angle PDB$（同弧 $\overarc{BC}$ 的圆周角，part5 内容预告）$\to$ AA 相似 $\to$ $\dfrac{PA}{PD} = \dfrac{PC}{PB}$ $\to$ $PA \cdot PB = PC \cdot PD$。注意这里"共角"是对顶角 $\angle APC = \angle DPB$，"共边比例"则是结论形式——可见共角共边模型在圆中以"对顶角型"出现，是相交弦定理的几何内核。

## 七、思路自测题

1. $\triangle ABC$ 中 $D$ 在 $AB$ 上，$E$ 在 $AC$ 上，$AD = 2, AB = 8, AE = 3, AC = 12$。判断 $\triangle ADE$ 与 $\triangle ABC$ 的相似关系（直接型还是交叉型？字母如何对应？）。提示：先算 $\dfrac{AD}{AB}$ 和 $\dfrac{AE}{AC}$，再算 $\dfrac{AD}{AC}$ 和 $\dfrac{AE}{AB}$，哪个相等？

2. 已知 $AB \cdot AE = AC \cdot AD$，且 $D, E$ 分别在 $AB, AC$ 上（共顶点 $A$）。判断哪两个三角形相似，并写出正确的字母对应顺序。提示：把乘积化为比例 $\dfrac{AE}{AB} = \dfrac{AD}{AC}$，识别是直接型还是交叉型。

3. 圆 $O$ 外一点 $P$ 引两条割线，分别交圆于 $A, B$ 和 $C, D$（$A, C$ 离 $P$ 较近）。试仿照例 3 找出一对相似三角形，并说明对应关系。提示：共角是 $\angle P$（公共角，不是对顶角），考虑 $\triangle PAC$ 与 $\triangle PDB$ 或 $\triangle PAD$ 与 $\triangle PCB$。

4. $\triangle ABC$ 中，$D$ 在 $AB$ 延长线上，$E$ 在 $AC$ 上，$\dfrac{AD}{AC} = \dfrac{AE}{AB}$。问 $\triangle ADE$ 与 $\triangle ACB$ 是否仍然相似？为什么？提示：$\angle DAE$ 与 $\angle CAB$ 是否还是同一个角？检查共角条件是否保持。
