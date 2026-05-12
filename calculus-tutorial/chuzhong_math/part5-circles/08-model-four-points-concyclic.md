# 四点共圆模型

> **一图速记**：四个点 $A, B, C, D$ 在同一个圆上 $\Leftrightarrow$ 任一对"同侧同弧"角相等（$\angle BAC = \angle BDC$）或任一对对角互补（$\angle A + \angle C = 180^\circ$）。**等角同弧、对角互补——四点共圆**。

## 一、引入

> 四边形 $ABCD$ 中，$\angle BAC = \angle BDC = 35^\circ$。求证：$A, B, C, D$ 四点在同一个圆上；并求 $\angle ABD$ 与 $\angle ACD$ 的关系。

这道题的难点在于"圆"根本没出现在条件里——只有四个点和两个相等的角。但只要我们意识到"看 $BC$ 这条线段被 $A$ 和 $D$ 张出相等的角"恰好是**圆周角定理的逆命题**，圆就被"召唤"出来了。一旦四点共圆，剩下的所有角关系都可以用圆周角、对角互补一口气解决。本节就来系统提炼这个"无圆生圆"的模型。

## 二、思维路径还原

> "题目里没画圆，可是出现了 $\angle BAC = \angle BDC$ —— 顶点 $A, D$ 分别对着同一条线段 $BC$，并且 $A, D$ 在 $BC$ 的同侧。这种"同一条线段 + 同侧两顶点 + 张角相等"的画面，我以前在哪里见过？
>
> 想起来了：part5/03 圆周角定理说"同弧所对圆周角相等"。如果 $A, D$ 都在某个圆上、且都在弦 $BC$ 的同侧，那么 $\angle BAC$ 与 $\angle BDC$ 自然相等——这是**正向**。
>
> 现在条件给的是结论 $\angle BAC = \angle BDC$，我要反推：能否说 $A, B, C, D$ 共圆？这就是圆周角定理的**逆命题**。
>
> 凭直觉应当成立：固定线段 $BC$，从同侧看 $BC$ 成定角的轨迹是一段圆弧（"定弦定角看圆弧"）。所以 $A$ 在以 $BC$ 为弦、张角为 $35^\circ$ 的弧上，$D$ 也在这段弧上 $\Rightarrow$ 两点同弧 $\Rightarrow$ 四点共圆。
>
> 严格证：过 $B, C, D$ 作圆 $\omega$（不共线三点定一圆）。设 $A$ 不在 $\omega$ 上，那么直线 $BA$ 与 $\omega$ 还应交于另一点 $A'$，由圆周角 $\angle BA'C = \angle BDC = \angle BAC$，但 $\angle BA'C$ 与 $\angle BAC$ 是 $\triangle ACA'$ 中一内角与一外角，不可能相等——矛盾。所以 $A$ 也在 $\omega$ 上，四点共圆。
>
> 既然共圆，$\angle ABD$ 与 $\angle ACD$ 又对着同一条弦 $AD$ 且在同侧，所以 $\angle ABD = \angle ACD$。
>
> 这条路径让我意识到一件大事：以后只要看到'两个角相等、且对着同一条线段'，就要立刻反应'四点共圆'，然后把整张图当作圆内接图形来用。"

## 三、抽象成模型

把上面的推理抽象为**四点共圆模型**。给定四个点 $A, B, C, D$，下面任一条件都**等价于**它们共圆：

- **判定 1（同侧等角 / 圆周角逆定理）**：$A, D$ 在直线 $BC$ 同侧，且 $\angle BAC = \angle BDC$ $\Rightarrow$ $A, B, C, D$ 共圆。
- **判定 2（对角互补）**：凸四边形 $ABCD$ 中 $\angle A + \angle C = 180^\circ$（等价地 $\angle B + \angle D = 180^\circ$）$\Rightarrow$ 四点共圆。这是圆内接四边形性质的逆命题。
- **判定 3（外角等于内对角）**：四边形 $ABCD$ 一外角等于其内对角，例如 $\angle DCE = \angle A$（$E$ 在 $BC$ 延长线上）$\Rightarrow$ 四点共圆。本质与判定 2 等价。
- **判定 4（相交弦 / 割线乘积逆定理）**：两条线段 $AC$ 与 $BD$ 相交于 $P$，若 $PA \cdot PC = PB \cdot PD$（$P$ 在两线段内部）或两线段延长后相交于 $P$ 且 $PA \cdot PC = PB \cdot PD$（$P$ 在外部），则 $A, B, C, D$ 共圆。这是 part5/07 圆幂定理的逆命题。

**核心收益**：一旦四点共圆，就可以自由调用圆内接图形的全部工具——圆周角、弦切角、对角互补、托勒密、圆幂——把一个"无圆"几何题瞬间转化为"标准圆题"。

记忆口诀："**等角同弦看共圆，对角互补也共圆，乘积相等还是共圆**"。

## 四、模型变形

四点共圆的"外壳"很多，常见变形如下：

- **直角变形（双直角共斜边）**：若 $\angle BAC = \angle BDC = 90^\circ$，则 $A, D$ 都在以 $BC$ 为直径的圆上 $\Rightarrow$ $A, B, C, D$ 共圆，且 $BC$ 是直径（part5/03 直径所对圆周角 $= 90^\circ$ 的逆用）。这是最常考的特例。
- **垂足共圆**：从一点 $P$ 向 $\triangle ABC$ 三边作垂线，垂足两两连同 $P$ 都形成"双直角共斜边"结构 $\Rightarrow$ 多对四点共圆；这就是**西姆松线 / 九点圆**的雏形。
- **高线交点四点共圆**：$\triangle ABC$ 中，两条高的垂足 $D, E$ 与对应顶点构造的四边形 $BDEC$（$\angle BDC = \angle BEC = 90^\circ$）四点共圆——竞赛级常客。
- **角平分线 + 等角**：若 $\angle ABD = \angle ACD$（$B, C$ 在 $AD$ 同侧），则 $A, B, C, D$ 共圆——把判定 1 的"线段"换成"对角线"即可。
- **圆幂逆用（乘积型）**：题目给出 $PA \cdot PC = PB \cdot PD$ 形式的乘积等式（$P$ 是 $AC, BD$ 交点）$\Rightarrow$ 立刻反应判定 4，把四个端点圈成共圆。

## 五、思考路标

遇到"角 / 长度乘积"型几何题，按下面顺序快速识别四点共圆：

- 看到**两个角相等、对着同一条线段、顶点在该线段同侧** $\Rightarrow$ 判定 1，四点共圆。
- 看到**四边形对角和 $= 180^\circ$**（或一外角 $=$ 内对角）$\Rightarrow$ 判定 2/3，四点共圆。
- 看到**两个直角共用一条斜边** $\Rightarrow$ 直角变形，四点共圆于以该斜边为直径的圆。
- 看到**两线段交于一点且 $PA \cdot PC = PB \cdot PD$** $\Rightarrow$ 判定 4，四点共圆。
- 看到**多个垂足、多个 $90^\circ$** $\Rightarrow$ 怀疑垂足共圆 / 双直角共斜边结构。
- 题目要证**两角相等 / 三点共线 / 线段乘积** 而条件零散 $\Rightarrow$ 尝试先证四点共圆，再借圆把零散条件串成一句话。

**操作要点**：证完共圆后，请显式写出"$\Rightarrow A, B, C, D$ 共圆"，之后所有圆性质（圆周角、弦切角、对角互补、圆幂）都可直接调用，这是阅卷的关键得分点。

## 六、应用例题

### 例 1（同侧等角判定）

如引入题：四边形 $ABCD$ 中 $\angle BAC = \angle BDC = 35^\circ$，且 $A, D$ 在 $BC$ 同侧。求证 $A, B, C, D$ 共圆，并证 $\angle ABD = \angle ACD$。

【思路】识别"同侧等角看同弦 $BC$" $\Rightarrow$ 由判定 1，$A, B, C, D$ 共圆。共圆后，$\angle ABD$ 与 $\angle ACD$ 同为弦 $AD$ 所对、且 $B, C$ 在 $AD$ 同侧 $\Rightarrow$ $\angle ABD = \angle ACD$（同弧圆周角相等）。

### 例 2（双直角共斜边）

$\triangle ABC$ 中，$BD \perp AC$ 于 $D$，$CE \perp AB$ 于 $E$。求证 $B, C, D, E$ 四点共圆，并求 $\angle ADE$ 与 $\angle ABC$ 的关系。

【思路】$\angle BDC = \angle BEC = 90^\circ$，两个直角共斜边 $BC$ $\Rightarrow$ 由判定 1（直角变形），$B, C, D, E$ 共圆于以 $BC$ 为直径的圆。共圆后，$BCDE$ 是圆内接四边形 $\Rightarrow$ $\angle ADE = \angle ABC$（圆内接四边形一外角 $\angle ADE$ 等于内对角 $\angle EBC = \angle ABC$，由判定 3 的正向用法）。

### 例 3（圆幂逆用）

线段 $AC$ 与 $BD$ 相交于 $P$，已知 $PA = 2, PC = 6, PB = 3, PD = 4$。判断 $A, B, C, D$ 是否共圆。

【思路】计算 $PA \cdot PC = 12 = PB \cdot PD$ $\Rightarrow$ 由判定 4（相交弦逆定理），四点共圆。延伸：此时若再连 $AB, CD$，由共圆得 $\angle BAC = \angle BDC$（同弧 $BC$），可以反过来求其他角。

## 七、思路自测题

1. 四边形 $ABCD$ 中 $\angle DAC = \angle DBC = 40^\circ$，求证 $A, B, C, D$ 共圆，并求 $\angle ADB$ 与 $\angle ACB$ 的关系。
   【提示】$A, B$ 在 $DC$ 同侧、对 $DC$ 等角 $\Rightarrow$ 判定 1 共圆；共圆后 $\angle ADB = \angle ACB$（同弧 $AB$）。

2. $\triangle ABC$ 中 $\angle A = 70^\circ$，点 $D$ 在 $BC$ 上，过 $D$ 作 $DE \perp AB$ 于 $E$，$DF \perp AC$ 于 $F$。证明 $A, E, D, F$ 共圆，并求 $\angle EDF$。
   【提示】$\angle AED = \angle AFD = 90^\circ$，双直角共斜边 $AD$ $\Rightarrow$ 四点共圆于以 $AD$ 为直径的圆；由对角互补 $\angle A + \angle EDF = 180^\circ \Rightarrow \angle EDF = 110^\circ$。

3. 凸四边形 $ABCD$ 中 $\angle ABC = 110^\circ$，$\angle ADC = 70^\circ$。判断是否四点共圆，并说明理由。
   【提示】$\angle ABC + \angle ADC = 180^\circ$ $\Rightarrow$ 判定 2 $\Rightarrow$ 共圆。

4. 线段 $AB, CD$ 延长线相交于圆外一点 $P$，$PA = 3, PB = 8, PC = 4, PD = 6$。证明 $A, B, C, D$ 共圆。
   【提示】$PA \cdot PB = 24 = PC \cdot PD$ $\Rightarrow$ 判定 4（割线-割线逆）$\Rightarrow$ 四点共圆。

5. $\triangle ABC$ 的两条高 $BD, CE$ 交于 $H$。求证 $A, E, H, D$ 共圆，并指出该圆的直径。
   【提示】$\angle AEH = \angle ADH = 90^\circ$ $\Rightarrow$ 双直角共斜边 $AH$ $\Rightarrow$ 共圆，且 $AH$ 为直径。这是垂心结构的标志性共圆。
