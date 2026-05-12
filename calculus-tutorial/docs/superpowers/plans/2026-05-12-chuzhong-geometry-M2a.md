# 中考几何教程 M2a 里程碑实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.

**Goal:** 完成 part3-triangles 的 basics 与 congruence 两个子目录，共 9 章。首次大量启用**模板 D（模型/题型类）**，体现"思路培养"特色。

**Architecture:** basics 用模板 C（概念/定理）；congruence/01-02 用模板 C（判定定理）；congruence/03-06 用**模板 D**（模型类，含"思维路径还原"和"思考路标"两个特色小节）。每文件独立提交。

**Spec 参考:** `docs/superpowers/specs/2026-05-12-chuzhong-geometry-tutorial-design.md`
**前序里程碑:** M0（已完成）、M1（已完成）

---

## 文件结构

```
chuzhong_math/part3-triangles/
├── basics/
│   ├── 01-triangle-inequality.md          # 三边关系
│   ├── 02-angle-sum.md                    # 内角和、外角
│   └── 03-classification.md               # 三角形分类
└── congruence/
    ├── 01-sss-sas-asa-aas.md              # 全等判定四定理（模板 C）
    ├── 02-hl-right-triangle.md            # 直角三角形 HL（模板 C）
    ├── 03-model-hand-in-hand.md           # 手拉手模型（模板 D）
    ├── 04-model-half-angle.md             # 半角模型（模板 D）
    ├── 05-model-one-line-three-angles.md  # 一线三等角（模板 D）
    └── 06-model-double-median.md          # 倍长中线（模板 D）
```

---

## 两种模板回顾

**模板 C（概念/定理类）—— 6 节:**
1. 图形特征 / 2. 定义与核心结论 / 3. 推导或证明 / 4. 典型应用 / 5. 易错点 / 6. 思路自测题

**模板 D（模型/题型类）—— 7 节:**
1. 引入：一道让你卡住的题
2. 思维路径还原（解题者的内心独白）— **本书核心特色，必须完整摊开**
3. 抽象成模型（图形特征 → 结论 → 简洁证明）
4. 模型变形（旋转/镜像/退化情形）
5. 思考路标（看到 X → 想到 Y）— **本书核心特色，列点形式**
6. 应用例题（2-3 题，演示路标怎么用）
7. 思路自测题

**全局风格:**
- LaTeX 全部数学符号 ($\triangle$, $\cong$, $\sim$, $\angle$, $\perp$, $\parallel$)
- 字数 1800-3500 中文字
- 模型节标配"一图速记"（节首用文字描述模型最简形态）
- 提交格式：`docs(chuzhong): add part3/<sub>/<file> (<topic>)`

---

## Task 1: basics/01 三角形三边关系

**File:** `chuzhong_math/part3-triangles/basics/01-triangle-inequality.md`（模板 C）

涵盖：

- **一、图形特征**：三条线段首尾相连构成的封闭图形
- **二、定义与定理**：
  - 三角形：三条不共线线段首尾相连组成的图形
  - 表示：$\triangle ABC$；三边 $a, b, c$（小写对应大写顶点的对边）
  - **三边关系定理**：任意两边之和大于第三边
  - **推论**：任意两边之差小于第三边
- **三、证明三边关系**：用"两点间线段最短"（呼应 part1/01）
  - 在 $\triangle ABC$ 中，$A, C$ 是两点，$B$ 是不在线段 $AC$ 上的点
  - $AB + BC > AC$（折线大于直线）
- **四、典型应用** (3 例):
  - 例 1：三条线段 $3, 5, 7$ 能否构成三角形？$3, 5, 8$ 呢？（前者可，后者退化为线段）
  - 例 2：已知两边为 $4, 9$，求第三边取值范围 $5 < x < 13$
  - 例 3：周长 $20$ 的等腰三角形，腰为整数厘米，求腰可能取值（$6, 7, 8, 9$；需 $2x > 20 - 2x$ 即 $x > 5$，又 $2x < 20$ 即 $x < 10$）
- **五、易错点**：必须**任意两边**之和都 $>$ 第三边；只检查最短两边即可（最短两边和 > 最长边 ⇔ 所有组合满足）
- **六、思路自测题** 4 题

---

## Task 2: basics/02 内角和与外角

**File:** `chuzhong_math/part3-triangles/basics/02-angle-sum.md`（模板 C）

涵盖：

- **一、图形特征**：三角形三内角
- **二、定理**：
  - **三角形内角和定理**：$\angle A + \angle B + \angle C = 180°$
  - **外角定理**：三角形的一个外角等于不相邻两内角之和
  - **外角推论**：三角形一个外角大于任一不相邻内角
- **三、内角和证明**（完整推导，呼应 part2/05 例 3）：
  - 过顶点 $A$ 作 $BC$ 的平行线 $l$
  - $l$ 与 $AB$ 形成内错角 $= \angle B$，与 $AC$ 形成内错角 $= \angle C$
  - 三个角拼成平角 → 和为 $180°$
- **四、外角定理证明**：
  - 设外角 $\angle ACD$ 与内角 $\angle ACB$ 互补
  - $\angle ACD = 180° - \angle ACB = 180° - (180° - \angle A - \angle B) = \angle A + \angle B$
- **五、典型应用** (3 例):
  - 例 1：已知两内角 $40°, 70°$，求第三内角和三个外角
  - 例 2：**思路题** —— 五角星五个尖角之和是多少？（结果 $180°$，思路：把每个尖角"传递"到中央五边形的外角上，再用三角形外角定理）
  - 例 3：在 $\triangle ABC$ 中，$\angle B$ 与 $\angle C$ 的平分线交于 $I$，证 $\angle BIC = 90° + \frac{1}{2}\angle A$（思路：$\angle BIC = 180° - \frac{1}{2}(\angle B + \angle C) = 180° - \frac{1}{2}(180° - \angle A)$）
- **六、易错点**：外角是"延长一边"形成；钝角三角形也有外角和概念
- **七、思路自测题** 5 题

---

## Task 3: basics/03 三角形分类

**File:** `chuzhong_math/part3-triangles/basics/03-classification.md`（模板 C，简短）

涵盖：

- **一、图形特征**：根据边的相等情况 / 角的种类分类
- **二、分类**：
  - **按边**：不等边三角形 / 等腰三角形（两边相等）/ 等边三角形（三边都相等，等腰的特殊情形）
  - **按角**：锐角三角形（三内角全 $<90°$）/ 直角三角形（一内角 $=90°$）/ 钝角三角形（一内角 $>90°$）
  - 综合分类：6 类常见组合（如等腰直角、等腰锐角等）
- **三、关键结论**：
  - 直角三角形两锐角互余（呼应 part1/04）
  - 等边三角形也是三个 $60°$ 的等角三角形
- **四、典型应用** (2 例):
  - 例 1：已知三角形两角 $30°, 60°$，判断形状（直角三角形）
  - 例 2：判断三边 $3, 4, 5$ 构成什么三角形（直角三角形——勾股逆定理，铺垫 right-triangle 章节）
- **五、易错点**：等边是等腰的特殊情形；钝角三角形只有一个钝角
- **六、思路自测题** 3 题

---

## Task 4: congruence/01 全等判定 SSS、SAS、ASA、AAS

**File:** `chuzhong_math/part3-triangles/congruence/01-sss-sas-asa-aas.md`（模板 C）

涵盖：

- **一、图形特征**：两个三角形对应边、对应角相等
- **二、定义与判定定理**：
  - 全等：两三角形完全重合，记 $\triangle ABC \cong \triangle DEF$
  - 性质：对应边相等、对应角相等
  - **判定定理**：
    - **SSS**：三边对应相等 → 全等
    - **SAS**：两边及其**夹角**对应相等 → 全等
    - **ASA**：两角及其**夹边**对应相等 → 全等
    - **AAS**：两角及其**中一角对边**对应相等 → 全等
  - 关键澄清：**SSA 一般不成立**（举反例：等腰三角形的腰、底角和底）
- **三、判定背后的逻辑**：
  - SAS 是公理（不证）
  - ASA 可由 SAS 推出（用平角与内角和）
  - AAS 由 ASA + 内角和推出（第三角自动相等）
  - SSS 也可由 SAS + 三边关系推出（演示思路即可，不要求完整严格证明）
- **四、典型应用** (3 例):
  - 例 1：基础——直接套 SSS / SAS / ASA / AAS 各一题
  - 例 2：**思路题** —— 看到题目里给出"两边一角"，第一反应是判定夹角还是不夹角，决定能否用 SAS。给一道需要先判断的题
  - 例 3：综合——在 $\triangle ABC$ 中，$AB = AC$，$D$ 在 $BC$ 上且 $BD = CE$，连 $AD, AE$（$E$ 在 $BC$ 上 / 描述清楚），证两三角形全等
- **五、易错点**：
  - SSA 不能用作判定！
  - 找对应关系时要"按字母对位"（$\triangle ABC \cong \triangle DEF$ 表示 $A \leftrightarrow D$、$B \leftrightarrow E$、$C \leftrightarrow F$）
  - 用 SAS 时一定确认是**夹**角而非其他角
- **六、思路自测题** 5 题，至少 1 题需要识别"非夹角"陷阱

---

## Task 5: congruence/02 直角三角形 HL

**File:** `chuzhong_math/part3-triangles/congruence/02-hl-right-triangle.md`（模板 C，简短）

涵盖：

- **一、图形特征**：两个直角三角形，斜边与一条直角边相等
- **二、判定 HL**：直角三角形中，**斜边和一条直角边**对应相等 → 两直角三角形全等
- **三、HL 的逻辑地位**：
  - HL 是 SSA 的特例——直角让 SSA 不再有歧义
  - 由勾股定理：已知斜边和一直角边 → 另一直角边唯一确定 → 三边都相等 → 由 SSS 全等
  - 所以 HL 本质是 SSS 的应用，但记作单独定理便于使用
- **四、典型应用** (2 例):
  - 例 1：基础——给两直角三角形斜边相等、一直角边相等，证全等
  - 例 2：**思路题** —— 角平分线定理的逆定理证明（到角两边距离相等的点在角平分线上）。两个直角三角形 $\triangle OPM, \triangle OPN$，$\angle PMO = \angle PNO = 90°$，$PM = PN$，$OP$ 公共（斜边）→ HL → $\triangle OPM \cong \triangle OPN$ → $\angle MOP = \angle NOP$（呼应 part1/03）
- **五、易错点**：HL 仅限**直角**三角形；缺直角条件用 HL 是错的
- **六、思路自测题** 3 题

---

## Task 6: congruence/03 手拉手模型 ⭐️ 模板 D 首篇

**File:** `chuzhong_math/part3-triangles/congruence/03-model-hand-in-hand.md`（**模板 D**）

This is the first model-style chapter — set the gold standard for "思维路径还原" and "思考路标".

Top title: `# 手拉手模型`

**节首"一图速记"**（文字描述）：两个等腰三角形共顶点（顶角相等），两腰端点连线相等，两条连线之间的夹角等于顶角。

### 一、引入：一道让你卡住的题

题目：
> $\triangle OAB$ 和 $\triangle OCD$ 都是等腰三角形，$OA = OB$，$OC = OD$，$\angle AOB = \angle COD = 50°$。求证：$AC = BD$。

读者尝试 → 容易卡在"找不到能用 SAS 的两个三角形"。

### 二、思维路径还原（解题者的内心独白）

完整内心独白（必须详细，至少 6-10 行）：

> "看到 $AC$ 和 $BD$，要证两段相等——典型的全等三角形思路。
> 但是 $AC$ 和 $BD$ 看起来不在'明显的'两个三角形里……
> 让我把要证的两段所在的三角形画出来：$\triangle AOC$（含 $AC$）、$\triangle BOD$（含 $BD$）。试这两个！
> 检查条件：$OA = OB$（已知）、$OC = OD$（已知）——这是两组对应边。
> 还差一组对应角。我看到 $\angle AOC$ 和 $\angle BOD$，它们相等吗？
> $\angle AOC = \angle AOB + \angle BOC$、$\angle BOD = \angle COD + \angle BOC$。
> 而 $\angle AOB = \angle COD = 50°$！所以 $\angle AOC = \angle BOD$（**等量加同一项**）。
> 凑齐了：$OA = OB$、$\angle AOC = \angle BOD$、$OC = OD$ —— SAS！
> 全等 → 对应边 $AC = BD$。"

### 三、抽象成模型

**图形特征**：两个共顶点的等腰三角形（顶角相等）
- $OA = OB$、$OC = OD$、$\angle AOB = \angle COD$

**结论**：
- $\triangle AOC \cong \triangle BOD$（SAS）
- $AC = BD$
- $AC$ 与 $BD$ 的夹角 $=$ 顶角 $\angle AOB$（旋转角）

**简洁证明**：用"等量加等量"凑出 $\angle AOC = \angle BOD$，套 SAS。

### 四、模型变形

- **变形 1**：两等腰退化为两等边三角形 → 旋转角 $= 60°$
- **变形 2**：两等腰退化为两等腰直角 → 旋转角 $= 90°$，$AC \perp BD$
- **变形 3**：内外位置变化（一个三角形在另一个内部 vs 外部）→ 证法相同
- **本质**：手拉手 = "**旋转构造全等**"——把 $\triangle OCD$ 看作 $\triangle OAB$ 绕 $O$ 旋转某个角度的结果

### 五、思考路标（看到 X → 想到 Y）

- 看到两个共顶点等腰三角形 → **手拉手**
- 看到题中出现 $60°$ + 等边三角形 → 手拉手 + 旋转 $60°$
- 看到题中出现 $90°$ + 等腰直角 → 手拉手 + 旋转 $90°$（且 $AC \perp BD$）
- 要证两段线段相等 + 共顶点的等腰结构 → 选含两段线段的两个三角形用 SAS
- 看到"凑角"需要 → 试**等量加同一项**或**等量减同一项**

### 六、应用例题（2-3 题，演示路标怎么用）

- 例 1：等边三角形版 —— 两等边 $\triangle OAB, \triangle OCD$ 共顶点 $O$，证 $AC = BD$ 且 $\angle AC$ 与 $\angle BD$ 夹角 $= 60°$。【思路】套手拉手；旋转角 $= 60°$。
- 例 2：等腰直角版 —— 同结构，证 $AC = BD$ 且 $AC \perp BD$。【思路】夹角等于顶角 $90°$。
- 例 3：综合 —— 给一道更复杂的图，需要先**识别**手拉手结构。例如给出共顶点的等腰但顶角隐含相等（需先证）。

### 七、思路自测题 (4 题，用 `💡 提示：`)

至少 1 题是"看似不是手拉手，但通过添加辅助线（构造等腰）就变成手拉手"。

---

## Task 7: congruence/04 半角模型

**File:** `chuzhong_math/part3-triangles/congruence/04-model-half-angle.md`（**模板 D**）

Top title: `# 半角模型`

**节首"一图速记"**：正方形（或等腰直角三角形）+ 顶点处有一个等于半顶角的角，把分散在两段上的条件**通过旋转**折合为一段。

### 一、引入

题目：
> 正方形 $ABCD$ 中，$E$ 在 $BC$ 上，$F$ 在 $CD$ 上，且 $\angle EAF = 45°$。求证：$EF = BE + DF$。

学生卡在"$EF$ 是一段，$BE + DF$ 是两段，怎么把两段合一段？"

### 二、思维路径还原（详细内心独白）

> "$\angle EAF = 45°$，而正方形 $\angle BAD = 90°$，所以 $\angle EAF$ 是 $\angle BAD$ 的**一半**——这是个信号。
> 我要把 $BE$ 和 $DF$ 拼成 $EF$。它们分别在 $BC$、$CD$ 上——能不能把 $\triangle ADF$ 转到 $\triangle ABF'$ 那里去？
> 由于 $AB = AD$（正方形），把 $\triangle ADF$ 绕 $A$ 顺时针旋转 $90°$，$D$ 落到 $B$，$F$ 落到 $BC$ 延长线上某点 $F'$。
> 旋转后 $AF = AF'$、$DF = BF'$。
> 现在看 $\triangle AEF$ 与 $\triangle AEF'$：$AE$ 公共、$AF = AF'$、$\angle FAE = 45°$、$\angle F'AE = \angle F'AB + \angle BAE = \angle DAF + \angle BAE = 90° - 45° = 45°$。
> SAS → 全等！$EF = EF' = BE + BF' = BE + DF$。"

### 三、抽象成模型

**图形特征**：正方形 $ABCD$（或等腰直角三角形）+ 顶点 $A$ 处一个等于半顶角（$45°$）的角 $\angle EAF$，$E, F$ 分别在两邻边上。

**结论**：$EF = BE + DF$（"两短段之和 = 一长段"）

**证法**：旋转 $\triangle ADF$ 至 $\triangle ABF'$（用 $AB = AD$），凑出 $\triangle AEF \cong \triangle AEF'$（SAS），从而 $EF = EF'$。

### 四、模型变形

- 等腰直角三角形 + 半角 = $22.5°$ 的版本
- 半角在外部的版本（$E$ 在 $BC$ 延长线上）→ 结论变为 $EF = DF - BE$
- 顶角不一定是 $90°$ 也成立（任意等腰顶角 $2\alpha$ + 半角 $\alpha$）

### 五、思考路标

- 看到正方形 + 顶点处 $45°$ → 半角模型
- 看到等腰 + 顶角的一半 → 试旋转，把一半折到另一半
- "两短段之和 = 一长段" 的等式 → 旋转构造全等是首选思路
- 旋转的角度 = 等腰的顶角

### 六、应用例题 (2 例)

- 例 1：基础题与引入题同类型，但 $E, F$ 位置稍变
- 例 2：等腰直角三角形 + 半角，证 $EF^2 = BE^2 + DF^2$（旋转后用勾股定理）

### 七、思路自测题 4 题

---

## Task 8: congruence/05 一线三等角（K 字模型）

**File:** `chuzhong_math/part3-triangles/congruence/05-model-one-line-three-angles.md`（**模板 D**）

Top title: `# 一线三等角（K 字模型）`

**节首"一图速记"**：一条直线上出现三个相等的角，则直线两侧的两个三角形相似（或全等）。形似字母 K。

### 一、引入

题目：
> 直线 $l$ 上有三点 $A, P, B$ 顺次排列。在 $l$ 同侧分别作 $\angle 1 = \angle APB = \angle 2 = \alpha$（其中 $\angle 1$ 在 $A$ 处一条线段 $AC$ 与 $l$ 的夹角，$\angle 2$ 在 $B$ 处一条线段 $BD$ 与 $l$ 的夹角）。证 $\triangle APC \sim \triangle PBD$。

### 二、思维路径还原

> "看到 $A, P, B$ 共线 + 三个相等角——经典 K 字。
> 要证 $\triangle APC \sim \triangle PBD$，相似要找两对对应角相等。
> $\angle PAC = \angle 1 = \alpha$（已知）；$\angle PBD = \angle 2 = \alpha$（已知）—— 第一对角 $\angle A = \angle B$ 成立。
> 还差一对。看 $A$ 处的另一个角：$\angle APC + \angle 1 + \angle APB = 180°$（平角）→ $\angle APC = 180° - \alpha - \alpha = 180° - 2\alpha$？
> 等下，再看 $B$ 处：$\angle BPD = 180° - \angle APB - \angle 2 = 180° - \alpha - \alpha = 180° - 2\alpha$。
> $\angle APC = \angle BPD$（都等于 $180° - 2\alpha$）—— 第二对角！
> 两组角相等 → AA → 相似。"

更直接的证法：
> "其实只需用'外角等于不相邻两内角和'：在 $\triangle APC$ 中，外角 $\angle BPC = \angle PAC + \angle ACP$，即 $\angle BPD + \alpha = \alpha + \angle ACP$（因 $\angle BPC = \angle BPD + \angle DPC = \angle BPD + \angle APB - ...$ 略），就更直接。"

### 三、抽象成模型

**图形特征**：直线 $l$ 上三点 $A, P, B$；同侧两线段 $CA, DB$；$\angle CAP = \angle CPD = \angle DBP = \alpha$（三个角相等，统称 K 字三等角）。

**结论**：$\triangle APC \sim \triangle PBD$。当 $AC = PD$（或 $AP = BP$）时为全等。

**证法**：见上内心独白。

### 四、模型变形

- **特殊情形**：$\alpha = 90°$ —— 一线三直角，即"直线上立两条垂线"形态，K 字相似最常见
- **特殊情形 + 等条件**：$\alpha = 90°$ 且 $AC = PD$ → 全等（题中常用）
- **进阶**：嵌入正方形/矩形中——一条边充当 $l$，另两边端点向边作线

### 五、思考路标

- 看到一直线上有三个相等角 → **K 字相似**
- 看到一直线上有三个直角（即一线三直角）→ **K 字特例**
- 看到正方形中"边上一点向两邻边作垂线" → 一线三直角 → 相似
- 含一线三等角的题目，要找的对应关系是"两侧三角形按字母顺序对应"

### 六、应用例题 (3 例)

- 例 1：基础 K 字相似——$\alpha = 90°$ 版本
- 例 2：正方形边上动点 + 一线三直角，求线段长度
- 例 3：综合——动点在边上滑动，恒有 K 字结构

### 七、思路自测题 4 题

---

## Task 9: congruence/06 倍长中线

**File:** `chuzhong_math/part3-triangles/congruence/06-model-double-median.md`（**模板 D**）

Top title: `# 倍长中线模型`

**节首"一图速记"**：题目出现中线（或中点），延长该中线一倍构造全等三角形，把分散在两边一中线上的条件集中到一个三角形里。

### 一、引入

题目（呼应 toolkit/01 中的"演示题"，但本节更深入）：
> 在 $\triangle ABC$ 中，$D$ 是 $BC$ 中点，$AB = 5$，$AC = 7$。求中线 $AD$ 的取值范围。

### 二、思维路径还原

> "中点 $D$ + 要求 $AD$——典型倍长中线信号。
> 延长 $AD$ 到 $E$ 使 $DE = AD$，连 $CE$。
> 看 $\triangle ABD$ 与 $\triangle ECD$：$BD = CD$（$D$ 是中点）、$\angle ADB = \angle CDE$（对顶角，呼应 part2/01）、$AD = ED$（构造）—— SAS → $\triangle ABD \cong \triangle ECD$。
> 全等 → $CE = AB = 5$。
> 现在 $AE = 2AD$，且在 $\triangle ACE$ 中三边为 $7, 5, 2AD$，用三边关系（呼应 basics/01）：
> $|7-5| < 2AD < 7+5$ → $1 < AD < 6$。"

### 三、抽象成模型

**图形特征**：三角形里有一条中线 $AD$（$D$ 是 $BC$ 中点）。

**操作**：延长 $AD$ 到 $E$ 使 $DE = AD$，连 $CE$（或 $BE$）。

**核心结论**：$\triangle ABD \cong \triangle ECD$（SAS：$BD = CD$、对顶角、$AD = ED$）→ $AB = CE$ 且 $AB \parallel CE$（"等长且平行"，因为内错角相等）。

**等价表述**：构造出"以 $A, B, E, C$ 为顶点的平行四边形"——其实就是中点对称构造。

### 四、模型变形

- **倍长任意中点连线**（不仅是中线）：题目中任何"X 是某段中点"都可考虑倍长
- **结合中位线**：倍长中点也是中位线定理的等价构造
- 用途：①求中线长度范围 ②证两段相等 ③把分散条件集中

### 五、思考路标

- 看到中线（或中点）+ 要求或要证涉及中线 → **倍长中线**
- 看到中点 + 三角形两边 + 一中线 → 倍长后用三边关系（如本节例题）
- 看到中点 + 要证两线段相等但不在同一三角形 → 倍长制造全等
- 倍长中线本质 = "中点的中心对称" = 构造平行四边形

### 六、应用例题 (3 例)

- 例 1：求中线范围（同引入题）
- 例 2：证两边相等——$D$ 是 $BC$ 中点，$AB \perp AD$，证 $AD = \frac{1}{2}BC$？（错例：这是直角三角形斜边中线定理，但思路可以用倍长引出）。改为：$D$ 是 $BC$ 中点，$E$ 在 $AC$ 上，$BD$ 与 $AE$ 关系...（自选合适题）
- 例 3：综合——已知中线长求其他边

### 七、思路自测题 4 题，至少一题需要识别"中点信号"

---

## Task 10: M2a 收尾

- [ ] **Step 1: 一致性扫描**

```bash
cd /Users/yangyang/ai_projs/math/calculus-tutorial
grep -rn 'TBD\|TODO\|XXX\|待补' chuzhong_math/part3-triangles
ls chuzhong_math/part3-triangles/basics chuzhong_math/part3-triangles/congruence
```

期望：grep 无输出；basics 3 个文件、congruence 6 个文件。

- [ ] **Step 2: 更新 README**

将 README 中：
```
- Part 3：三角形（含全等、相似、特殊三角形） *(待写)*
```
改为：
```
- Part 3：三角形 *(进行中)*
  - basics/
    - [01. 三角形三边关系](part3-triangles/basics/01-triangle-inequality.md)
    - [02. 内角和与外角](part3-triangles/basics/02-angle-sum.md)
    - [03. 三角形分类](part3-triangles/basics/03-classification.md)
  - congruence/
    - [01. 全等判定 SSS/SAS/ASA/AAS](part3-triangles/congruence/01-sss-sas-asa-aas.md)
    - [02. 直角三角形 HL](part3-triangles/congruence/02-hl-right-triangle.md)
    - [03. 手拉手模型](part3-triangles/congruence/03-model-hand-in-hand.md) ⭐️
    - [04. 半角模型](part3-triangles/congruence/04-model-half-angle.md) ⭐️
    - [05. 一线三等角（K 字模型）](part3-triangles/congruence/05-model-one-line-three-angles.md) ⭐️
    - [06. 倍长中线模型](part3-triangles/congruence/06-model-double-median.md) ⭐️
```

提交：`docs(chuzhong): update README for M2a (part3 basics + congruence)`

- [ ] **Step 3: 标记完成**

在本计划文件末尾追加 `**M2a 完成于：YYYY-MM-DD**`，提交。
