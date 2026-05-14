# 平面与平面的位置关系及二面角

> **一例速记**：  
> **两平面位置**：① 平行 $\alpha \parallel \beta$（无公共点）② 相交 $\alpha \cap \beta = l$（交线 $l$）。  
> **面面平行判定**：$\alpha$ 内**两相交直线**分别 $\parallel \beta$ → $\alpha \parallel \beta$。  
> **面面垂直判定**：$\alpha$ 内有**一条直线** $\perp \beta$ → $\alpha \perp \beta$。  
> **二面角** $\alpha$-$l$-$\beta$：棱 $l$ 上取点 $O$，在两半平面分别作 $\perp l$ 的射线 $OA, OB$ → $\angle AOB$ 是二面角的平面角，范围 $[0, \pi]$。

---

## 一、引入：正三棱锥中证垂直与求二面角

> **题目**：正三棱锥 $P$-$ABC$，底面是正三角形（边长 $2$），$PA = PB = PC = \sqrt{6}$。$E$ 是 $BC$ 中点。
> (1) 证明：$PE \perp$ 平面 $ABC$ 不对——正三棱锥侧棱不一定垂直底面。正确：**设 $O$ 是底面正三角形中心**，证 $PO \perp$ 平面 $ABC$。
> (2) 求二面角 $P$-$BC$-$A$ 的大小。

请先停下来想一想：这两问的核心都是**在立体图形中找出垂直关系，并把空间问题降到平面**。

**第 1 问的策略**：用线面垂直判定（直线 $\perp$ 平面内两相交线）。我们想证 $PO \perp$ 平面 $ABC$，需要在底面找到两条相交直线分别 $\perp PO$。由对称性，连 $AO, BO, CO$，由正三棱锥 $PA = PB = PC$ 知 $\triangle PAO \cong \triangle PBO \cong \triangle PCO$，进而能推出 $PO \perp AO, PO \perp BO$，且 $AO \cap BO = O$。

**第 2 问的策略**：求二面角，找平面角。棱是 $BC$，所以在 $BC$ 上取一点（如中点 $E$），在两半平面各作 $\perp BC$ 的射线。因 $\triangle ABC$ 等边、$E$ 是中点，故 $AE \perp BC$（半平面 $ABC$ 中的垂线）；由 $PB = PC$，$E$ 是 $BC$ 中点，故 $PE \perp BC$（半平面 $PBC$ 中的垂线）。$\angle PEA$ 即为所求。

下面把内心独白完整还原。

---

## 二、思维路径还原（解题者的内心独白）

> "**第 1 问**：先在底面 $\triangle ABC$ 找'中心 $O$'。正三角形中心 = 重心 = 内心 = 外心 = 垂心，是三条中线的交点；到三顶点距离相等 $AO = BO = CO = \frac{2}{\sqrt{3}}$（边长 $2$ 的正三角形外接圆半径）。
> 
> 要证 $PO \perp$ 平面 $ABC$，找平面内两条相交线与 $PO$ 垂直。试 $AO$ 和 $BO$：
> 
> $|PA|^2 = |PO|^2 + |AO|^2$ 吗？我们已知 $PA = \sqrt{6}$，要先求 $|PO|$。由勾股 $|PO|^2 = |PA|^2 - |AO|^2 = 6 - 4/3 = 14/3$？等等，$|AO|^2 = (2/\sqrt{3})^2 = 4/3$。所以 $|PO| = \sqrt{14/3}$。
> 
> 现在反过来：$|PA|^2 = |PO|^2 + |AO|^2 = 14/3 + 4/3 = 6 = (\sqrt{6})^2$ ✓ → 由勾股逆定理 $PO \perp AO$。同理 $PO \perp BO$。$AO \cap BO = O$，且 $AO, BO \subset$ 平面 $ABC$ → 由线面垂直判定：$PO \perp$ 平面 $ABC$。✓
> 
> **第 2 问**：求二面角 $P$-$BC$-$A$。棱是 $BC$。
> 
> 在棱上取点 $E$（$BC$ 中点）。  
> 半平面 $ABC$ 中作 $EA$：因 $\triangle ABC$ 等边、$E$ 是 $BC$ 中点，$AE \perp BC$。✓  
> 半平面 $PBC$ 中作 $EP$：因 $PB = PC$、$E$ 是 $BC$ 中点，$PE \perp BC$。✓  
> 故 $\angle PEA$ 是二面角 $P$-$BC$-$A$ 的平面角。
> 
> 计算 $\angle PEA$：$|AE| = \sqrt{3}$（正三角形高 = $\sqrt{3}$），$|PE| = \sqrt{|PB|^2 - 1^2} = \sqrt{6 - 1} = \sqrt{5}$，$|PA| = \sqrt{6}$。  
> 由余弦定理 $\cos\angle PEA = \frac{|PE|^2 + |AE|^2 - |PA|^2}{2|PE||AE|} = \frac{5 + 3 - 6}{2\sqrt{15}} = \frac{2}{2\sqrt{15}} = \frac{1}{\sqrt{15}} = \frac{\sqrt{15}}{15}$。  
> 二面角 $= \arccos\frac{\sqrt{15}}{15}$。"

---

## 三、抽象成方法

### 两平面的 2 种位置关系

| 位置 | 公共点 | 判定法 | 性质 |
|---|---|---|---|
| **平行** $\alpha \parallel \beta$ | 无 | 一面内**两相交直线**分别平行另一面 | $\alpha \cap \gamma = a, \beta \cap \gamma = b$ → $a \parallel b$ |
| **相交** $\alpha \cap \beta = l$ | 一直线 | （自动） | — |

### 两平面**垂直**的判定与性质

| 类别 | 内容 |
|---|---|
| **判定** | $\alpha$ 内**一条直线** $l \perp \beta$ → $\alpha \perp \beta$ |
| **性质** | $\alpha \perp \beta$ 且 $\alpha \cap \beta = l$，$\alpha$ 内 $m \perp l$ → $m \perp \beta$ |

### 二面角 $\alpha$-$l$-$\beta$

- **定义**：从一条直线 $l$ 出发的两个半平面 $\alpha, \beta$ 构成的图形。
- **平面角**：在棱 $l$ 上任取一点 $O$，在两半平面分别作 $\perp l$ 的射线 $OA \subset \alpha, OB \subset \beta$，$\angle AOB$ 即为二面角的平面角。
- **范围**：$[0, \pi]$。
- **直二面角**：平面角 = $\frac{\pi}{2}$，对应面面垂直。

### 二面角的 3 种求法

| 方法 | 适用 | 步骤 |
|---|---|---|
| **综合法**（作平面角） | 几何关系清晰、棱与垂线易找 | 棱上取点 → 两半平面各作 $\perp$ 棱 → 用余弦定理 / 几何关系算角 |
| **法向量法**（高效，铺垫 Part 9） | 含坐标 / 用空间向量 | 求两半平面法向量 $\vec{n_1}, \vec{n_2}$ → $\cos\theta = \frac{\vec{n_1}\cdot\vec{n_2}}{|\vec{n_1}||\vec{n_2}|}$（看图判正负） |
| **三垂线** | 棱上有特殊点（如顶点）+ 有现成垂线 | 用三垂线定理找出平面角 |

---

## 四、方法变形

### 变形 1：法向量法求二面角的符号

法向量算出的 $\cos\theta$ 可能与实际二面角差一个负号（取决于法向量方向选取）。**实操规则**：算出 $|\cos\theta|$，然后**看图判定二面角是锐角还是钝角**——锐角取 $+|\cos\theta|$，钝角取 $-|\cos\theta|$。

### 变形 2：面面垂直 ⇒ 线面垂直（性质应用）

若已知 $\alpha \perp \beta, \alpha \cap \beta = l$，要把垂直关系传到具体直线：在 $\alpha$ 内作 $m \perp l$ → 由性质 $m \perp \beta$。这是**把面面垂直转为线面垂直**的关键技巧。

### 变形 3：用面面平行证线线平行

若 $\alpha \parallel \beta$，第三平面 $\gamma$ 同时与 $\alpha, \beta$ 相交，交线为 $a, b$，则 $a \parallel b$。

---

## 五、思考路标（条件反射）

1. 看到"**两平面平行**"判定 → 找一面内**两相交直线**与另一面平行（不能只一条！）。
2. 看到"**两平面垂直**"判定 → 找一面内**一条直线**与另一面垂直。
3. 看到"**二面角**" → 找棱 → 找平面角（两半平面各作垂直棱的射线）。
4. **平面角的位置**：必须在**棱上同一点**作出两条垂线。
5. **棱上特殊点**：中点、垂足、顶点都是常用作图点。
6. 看到"$\alpha \perp \beta$ 求线段"→ 用性质：$\alpha$ 内 $\perp$ 棱的线 $\perp \beta$。
7. **法向量法 vs 综合法**：含坐标 → 法向量；几何对称 → 综合法。
8. **法向量的方向**：决定 $\cos$ 的符号；看图调整。

---

## 六、典型应用

### 例 1：直棱柱中的面面平行

> **题目**：直三棱柱 $ABC$-$A_1B_1C_1$ 中，$D, E$ 分别是 $A_1B_1, B_1C_1$ 中点。证明：平面 $DEC$ $\parallel$ 平面 $A_1BC_1$（注意：此题需先验证两面不重合）。

【思路】用判定：在平面 $DEC$ 中找两条相交直线，分别 $\parallel$ 平面 $A_1BC_1$。

【解】$D, E$ 是 $A_1B_1, B_1C_1$ 中点 → $DE \parallel A_1C_1$（中位线性质）。又 $A_1C_1 \subset$ 平面 $A_1BC_1$，$DE \not\subset$ 平面 $A_1BC_1$ → $DE \parallel$ 平面 $A_1BC_1$。

类似地，由几何关系可证另一条 $\parallel$ 关系，从而面面平行。

### 例 2：长方体中的面面垂直

> **题目**：长方体 $ABCD$-$A_1B_1C_1D_1$ 中，$AB = AD = 2, AA_1 = 3$。证明：平面 $ACC_1A_1 \perp$ 平面 $BDD_1B_1$（$AC$ 与 $BD$ 是底面对角线）。

【解】底面 $ABCD$ 中，$AC$ 与 $BD$ 相交于中心 $O$。**关键**：在长方体中，$AB = AD$ 时 $ABCD$ 是正方形，于是 $AC \perp BD$。

证 $AC \perp$ 平面 $BDD_1B_1$：
- $AC \perp BD$（正方形对角线，刚才证）
- $AC \perp BB_1$（直棱柱性质：$BB_1 \perp$ 底面 $ABCD$，$AC \subset$ 底面）  
- $BD$ 和 $BB_1$ 相交于 $B$，都 $\subset$ 平面 $BDD_1B_1$ → 由线面垂直判定 $AC \perp$ 平面 $BDD_1B_1$。

又 $AC \subset$ 平面 $ACC_1A_1$ → 由面面垂直判定 $\alpha \perp \beta$。✓

### 例 3：求三棱锥的二面角

> **题目**：三棱锥 $P$-$ABC$ 中 $PA \perp$ 平面 $ABC$，$AB \perp BC$，$PA = AB = BC = 1$。求二面角 $A$-$PC$-$B$ 的大小。

【思路】棱是 $PC$。需在棱上取点，两半平面各作 $\perp PC$ 的射线。**用三垂线找平面角**。

【解】在棱 $PC$ 上找 $B$ 的射影：因 $PA \perp$ 底面 → $PA \perp BC$；又 $AB \perp BC$ → $BC \perp$ 平面 $PAB$ → $BC \perp PA$ 平面内的 $PB$（$PB \subset$ 平面 $PAB$）。

设 $B$ 在 $PC$ 上的射影为 $H$。则 $BH \perp PC$。在 $PC$ 上取 $H$，在半平面 $APC$ 中过 $H$ 作 $HA' \perp PC$？由对称（$PA \perp$ 底面），其实更便：直接构造作图。

简化：用法向量法。建系 $A(0,0,0), B(1,0,0), C(1,1,0), P(0,0,1)$。$\vec{PA} = (0,0,-1), \vec{PC} = (1,1,-1), \vec{PB} = (1,0,-1)$。

平面 $APC$ 的法向量 $\vec{n_1}$：由 $\vec{n_1} \cdot \vec{PA} = 0, \vec{n_1} \cdot \vec{PC} = 0$ → 设 $\vec{n_1} = (x, y, z)$，$-z = 0, x + y - z = 0$ → $z = 0, x = -y$ → 取 $\vec{n_1} = (1, -1, 0)$。

平面 $BPC$ 的法向量 $\vec{n_2}$：由 $\vec{n_2} \cdot \vec{PB} = 0, \vec{n_2} \cdot \vec{PC} = 0$ → 设 $\vec{n_2} = (x, y, z)$，$x - z = 0, x + y - z = 0$ → $x = z, y = 0$ → 取 $\vec{n_2} = (1, 0, 1)$。

$\cos\theta = \frac{\vec{n_1} \cdot \vec{n_2}}{|\vec{n_1}||\vec{n_2}|} = \frac{1}{\sqrt{2}\cdot\sqrt{2}} = \frac{1}{2}$。

由图判二面角 $A$-$PC$-$B$ 是锐角（$A, B$ 在 $PC$ 同侧上方），所以二面角 = $\frac{\pi}{3}$。

---

## 七、自测题

**自测 1**　四面体 $ABCD$ 中，$M, N$ 分别是 $AB, CD$ 中点。问：是否一定有 $MN \perp AB$？为什么？

> 💡 提示：不一定。仅当 $\triangle ABC$ 和 $\triangle ABD$ 满足某种对称性（如 $AC = BC$ 且 $AD = BD$）时才有 $MN \perp AB$。一般四面体不满足。

**自测 2**　长方体 $ABCD$-$A_1B_1C_1D_1$ 中 $AB = 1, AD = 2, AA_1 = 2$。求二面角 $A_1$-$BD$-$A$ 的大小。

> 💡 提示：棱是 $BD$。在底面作 $A$ 到 $BD$ 的垂线 $AE \perp BD$，则 $A_1E$ 也 $\perp BD$（三垂线定理）。$\angle A_1EA$ 是平面角。算 $|AE| = \frac{AB \cdot AD}{|BD|} = \frac{2}{\sqrt{5}}$；$\tan\angle A_1EA = \frac{|AA_1|}{|AE|} = \frac{2}{2/\sqrt{5}} = \sqrt{5}$ → 角 = $\arctan\sqrt{5}$。

**自测 3**　$\alpha \perp \beta, \alpha \cap \beta = l$，$P \in \alpha$ 且 $P \notin l$。从 $P$ 引 $\beta$ 的垂线，垂足在哪？

> 💡 提示：在 $\alpha$ 内过 $P$ 作 $PQ \perp l$ → 由"面面垂直性质" $PQ \perp \beta$ → 垂足 $Q$ 在棱 $l$ 上。

**自测 4**　正四棱锥 $P$-$ABCD$ 底面边长为 $2$，侧棱长为 $\sqrt{6}$。求二面角 $P$-$AB$-$C$。

> 💡 提示：棱 $AB$ 中点 $M$；$PM \perp AB$（等腰）、$OM \perp AB$（$O$ 是底面中心，$OM$ 是底面中位线方向 → 沿底面中线方向）。算 $|OM| = 1, |PM| = \sqrt{6-1} = \sqrt{5}, |PO| = \sqrt{(\sqrt{6})^2 - (\sqrt{2})^2} = 2$。$\tan\angle PMO = |PO|/|OM| = 2$ → 角 = $\arctan 2$。

**自测 5**　三棱锥 $P$-$ABC$ 中 $PA = PB = PC$。证明：$P$ 在底面 $ABC$ 上的投影 $O$ 是 $\triangle ABC$ 的外心。

> 💡 提示：设 $PO \perp$ 底面。则 $PO \perp OA, OB, OC$。由勾股 $|OA|^2 = |PA|^2 - |PO|^2$，同理 $|OB|^2 = |PB|^2 - |PO|^2, |OC|^2 = |PC|^2 - |PO|^2$。由 $PA = PB = PC$ 得 $|OA| = |OB| = |OC|$ → $O$ 是 $\triangle ABC$ 外心。

---

**回头看一眼"一例速记"**：

> 两平面位置：平行（无公共点）/ 相交（交线）。  
> 面面平行判定：两相交直线分别平行另一面。  
> 面面垂直判定：一面有直线垂直另一面。  
> 二面角：棱上一点作两半平面的垂直棱射线 → 所成角 = 平面角，范围 $[0, \pi]$。  
> 求二面角 3 法：综合（作平面角）/ 法向量（含坐标）/ 三垂线。

如果现在不看笔记能独立完成例 3 的法向量法 + 自测 4 的综合法——本章，你拿下了。
