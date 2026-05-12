# 中考代数教程 AM3 里程碑实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.

**Goal:** 完成 `chuzhong_algebra/` 的 part5（一次方程, 5 章）+ part7（不等式, 5 章），共 10 章。

**Architecture:** 7 章模板 C + 3 章模板 D。每章独立提交。

**Spec 参考:** `docs/superpowers/specs/2026-05-12-chuzhong-algebra-design.md`
**前序里程碑:** AM0、AM1、AM2（已完成）

> 注：AM3 暂时跳过 part6（一元二次方程）—— 单独留给 AM4，因为部分章节（part5/05 含参一次方程组、part7/05 含参不等式）需要先建立"含参讨论"的成熟思路。

---

## 文件结构

```
chuzhong_algebra/
├── part5-linear-eq/                           # 一次方程（组）[初一上 + 初二上方程组]
│   ├── 01-linear-one-variable.md              # 一元一次方程
│   ├── 02-linear-two-variable.md              # 二元一次方程组
│   ├── 03-elimination-methods.md              # 代入/加减消元法
│   ├── 04-linear-eq-tricks.md                 # ⭐ 解一次方程组技巧
│   └── 05-parametric-linear.md                # ⭐ 含参一次方程组
└── part7-inequalities/                        # 不等式 [初二下]
    ├── 01-inequality-basics.md                # 不等式概念与性质
    ├── 02-solve-linear-inequality.md          # 解一元一次不等式
    ├── 03-inequality-system.md                # 一元一次不等式组
    ├── 04-inequality-applications.md          # 不等式应用
    └── 05-parametric-inequality.md            # ⭐ 含参不等式分类讨论
```

---

## 模板规范

模板 C 6 节 (1800-2500 字)、模板 D 7 节 (3000-4000 字)。LaTeX 全部数学符号。模板 D 含"思维路径还原"（12-18 行 `>` 引用）+ "思考路标"（≥6 条）。

---

## Task 1: part5/01 一元一次方程

**File:** `chuzhong_algebra/part5-linear-eq/01-linear-one-variable.md`（模板 C，1800-2500 字）

Title: `# 一元一次方程`

- **概念**：含 1 个未知数、未知数最高次 1 次、整式方程
- **标准形式**：$ax + b = 0$（$a \neq 0$），解为 $x = -\frac{b}{a}$
- **等式性质**（解方程的依据）：
  - 性质 1：等式两边加减同一个数/式，等式仍成立
  - 性质 2：等式两边乘除同一个非零数/式，等式仍成立
- **解方程标准步骤**：
  1. 去分母（乘最小公倍数）
  2. 去括号（注意符号）
  3. 移项（变号）
  4. 合并同类项
  5. 系数化为 1
- **典型应用** 3 例每带【思路】：
  - 例 1：基础 $3x - 7 = 14 - 2x$
  - 例 2：含分数 $\frac{2x-1}{3} - \frac{x+2}{2} = 1$
  - 例 3：含字母系数 $ax + 1 = 2x - 3$（讨论 $a \neq 2$ 与 $a = 2$）
- **易错点**：移项要变号；去分母不能漏乘不含分母项；含字母系数要讨论
- **自测题** 4 题。

Commit: `docs(chuzhong_algebra): add part5/01 (linear equation in one variable)`

---

## Task 2: part5/02 二元一次方程组

**File:** `chuzhong_algebra/part5-linear-eq/02-linear-two-variable.md`（模板 C，1800-2500 字）

Title: `# 二元一次方程组`

- **二元一次方程**：含 2 个未知数、每个最高次为 1
- **方程组**：两个二元一次方程联立
- **方程组的解**：同时满足两个方程的 $(x, y)$
- **图象意义**（呼应 part8 一次函数）：两条直线的交点坐标
- **典型应用** 3 例：
  - 例 1：基础 $\begin{cases} 2x + y = 7 \\ x - y = 2 \end{cases}$
  - 例 2：判断解 $(3, 1)$ 是否是某方程组的解
  - 例 3：方程组的解有几种情况（一组解 / 无解 / 无穷多解，对应两直线相交 / 平行 / 重合）
- **易错点**：解必须同时满足两个方程；解的个数判断
- **自测题** 4 题。

Commit: `docs(chuzhong_algebra): add part5/02 (system of two linear equations)`

---

## Task 3: part5/03 代入/加减消元法

**File:** `chuzhong_algebra/part5-linear-eq/03-elimination-methods.md`（模板 C，2000-2700 字）

Title: `# 代入消元法与加减消元法`

- **代入消元法**：
  - 一个方程整理为 "$x = \text{含 }y$ 的式子" 或 "$y = \text{含 }x$ 的式子"
  - 代入另一个方程化为一元一次方程
  - 适用：某个方程系数为 1 或 -1 时
- **加减消元法**：
  - 两方程相加 / 相减消去某未知数
  - 必要时先乘倍数让系数相同 / 相反
  - 适用：某个未知数系数互为倍数关系时
- **典型应用** 3 例：
  - 例 1：代入法解 $\begin{cases} y = 2x - 1 \\ 3x + 2y = 12 \end{cases}$
  - 例 2：加减法解 $\begin{cases} 2x + 3y = 7 \\ 2x - y = 3 \end{cases}$
  - 例 3：选哪种方法？$\begin{cases} 3x + 5y = 8 \\ 5x - 3y = 2 \end{cases}$（系数相对复杂，加减法 + 倍乘）
- **易错点**：代入后**不要代回原方程**而要代入另一方程；加减时符号；要消的是同一未知数
- **自测题** 4 题。

Commit: `docs(chuzhong_algebra): add part5/03 (substitution and elimination methods)`

---

## Task 4: part5/04 解一次方程组技巧 ⭐（模板 D）

**File:** `chuzhong_algebra/part5-linear-eq/04-linear-eq-tricks.md`（**模板 D**，3000-4000 字）

Title: `# 解一次方程组的技巧`

**一例速记**：
> 解 $\begin{cases} 199x + 201y = 200 \\ 201x + 199y = 200 \end{cases}$。系数复杂但**对称**：两式相加 → $400(x+y) = 400$ → $x+y = 1$；两式相减 → $-2x + 2y = 0$ → $y = x$。最后 $x = y = 0.5$。**整体加减比标准代入快 10 倍**。

### 一、引入题
> 解 $\begin{cases} 2x + 3y = 8 \\ 3x + 2y = 7 \end{cases}$。盲算用加减消元法需乘倍数；用整体加减 5 秒。

### 二、思维路径还原（12-18 行 `>`）
> "$\begin{cases} 2x + 3y = 8 \\ 3x + 2y = 7 \end{cases}$ —— 系数 (2,3) 和 (3,2) **交换对称**！  
> **两式相加**：$5x + 5y = 15$ → $x + y = 3$  
> **两式相减**：$-x + y = 1$ → $y - x = 1$  
> 现在更简单的方程组：$\begin{cases} x + y = 3 \\ y - x = 1 \end{cases}$  
> 两式相加：$2y = 4 \Rightarrow y = 2$；代回 $x = 1$  
> 答：$x = 1, y = 2$  
> 关键反射：见'对称系数'立刻整体加减——比单方程消元快得多。"

### 三、抽象成方法
**4 类常用技巧**：

1. **整体加减**（系数对称）：见 $\begin{cases} ax + by = c \\ bx + ay = d \end{cases}$ 立即加减
2. **整体换元**：见 $\begin{cases} \frac{x}{a} + \frac{y}{b} = 1 \\ ... \end{cases}$ 设 $u = \frac{x}{a}, v = \frac{y}{b}$ 简化
3. **特殊解**（一眼看出）：如 $\begin{cases} x + y = 5 \\ x - y = 3 \end{cases}$ 直接 $x=4, y=1$
4. **错位消元**：高阶方程组中先消最容易消的

### 四、方法变形
- 含三元方程组：$\begin{cases} x + y + z = 6 \\ x + y - z = 4 \\ x - y + z = 2 \end{cases}$ 先两两相加减
- 系数含字母：含参情形（part5/05 详讲）

### 五、思考路标（≥6 条）
- 见系数交换对称 → 整体加减
- 见分数系数 → 整体换元 / 先去分母
- 见数字接近 → 尝试简便（如 $199, 201$）
- 见三元 → 先两两组合消一元
- 优先级：观察对称性 → 找最易消的未知数 → 标准消元
- "解一个方程 vs 解整个方程组"：能直接看出解优先

### 六、应用例题 3 例
- 例 1：对称系数（同思维路径还原题）
- 例 2：分数系数 $\begin{cases} \frac{x+y}{2} = 3 \\ \frac{x-y}{3} = 1 \end{cases}$
- 例 3：含字母系数（铺垫 part5/05）

### 七、自测题 5 题

Commit: `docs(chuzhong_algebra): add part5/04 (linear equation tricks)`

---

## Task 5: part5/05 含参一次方程组 ⭐（模板 D）

**File:** `chuzhong_algebra/part5-linear-eq/05-parametric-linear.md`（**模板 D**，3000-4000 字）

Title: `# 含参一次方程组`

**一例速记**：
> 含参 $\begin{cases} x + y = a \\ x - y = b \end{cases}$ —— 解为 $x = \frac{a+b}{2}, y = \frac{a-b}{2}$。看到含参题想"先解出 $x, y$ 含 $a, b$，再用条件推 $a, b$ 关系"。

### 一、引入题
> 已知方程组 $\begin{cases} 2x + y = 5 \\ x - y = m \end{cases}$ 的解满足 $x + y = 3$，求 $m$。

### 二、思维路径还原（12-18 行 `>`）
> "含参题第一反应：**先把 $x, y$ 用 $m$ 表示**（不要先想 $m$ 是多少）。  
> 解方程组：两式相加 → $3x = 5 + m$ → $x = \frac{5+m}{3}$  
> 代回 → $y = 5 - 2x = \frac{5 - 2m}{3}$  
> 现在用约束 $x + y = 3$：$\frac{5+m}{3} + \frac{5-2m}{3} = 3$  
> $\frac{10 - m}{3} = 3$ → $10 - m = 9$ → $m = 1$  
> 验证：$x = 2, y = 1$，$x+y=3$ ✓  
> 关键：**含参题的'三步走'：① 解出 $x, y$ 含参 → ② 代入额外条件 → ③ 解出参数**。"

### 三、抽象成方法
**含参方程组的 3 步标准流程**：
1. **解出含参解**：把 $x, y$ 用参数表达
2. **代入额外条件**：通常题给的"解满足某关系"或"特殊解"
3. **解出参数**：一元方程求解

**变种 1（求参数取值范围）**：当解满足"正"、"整数"等条件时
**变种 2（无解 / 无穷多解 / 唯一解判定）**：通过系数行列式（初中淡化，但可比较系数比例）
**变种 3（同解方程组）**：两组方程组有相同解，用代入互推

### 四、方法变形
- 含参题的"无解条件"：系数比相等但常数项比不等
- 含参题"有无数解"：系数比与常数项比都相等
- 求"$x, y$ 互为相反数"或"$x = y$"等隐含条件

### 五、思考路标（≥6 条）
- 见含参方程组 → 先解 $x, y$ 含参
- 见"解满足某关系" → 代入条件解参数
- 见"解为正" → $x > 0, y > 0$ 两个不等式
- 见"无解 / 无穷多解" → 检查系数比与常数项比
- 含参题答案：参数本身的值或范围
- 解出后验证（含参题验证更重要）

### 六、应用例题 3 例
- 例 1：解满足额外条件（同引入题）
- 例 2：解为正整数（求参数整数范围）
- 例 3：同解方程组（如 $\begin{cases} ax+by=1 \\ 2x+y=3 \end{cases}$ 与 $\begin{cases} 3x-y=2 \\ ax-by=4 \end{cases}$ 同解，求 $a, b$）

### 七、自测题 5 题

Commit: `docs(chuzhong_algebra): add part5/05 (parametric linear equations)`

---

## Task 6: part7/01 不等式基础

**File:** `chuzhong_algebra/part7-inequalities/01-inequality-basics.md`（模板 C，1800-2500 字）

Title: `# 不等式的概念与基本性质`

- **不等式**：用 $>, <, \geq, \leq, \neq$ 连接的式子
- **不等式的解**：使不等式成立的未知数的值
- **解集**：所有解的集合（用区间 / 数轴表示）
- **三大性质**（解不等式的依据）：
  - 性质 1：两边加减同一个数 / 式 → 不等号方向**不变**
  - 性质 2：两边乘除**正**数 → 不等号方向**不变**
  - 性质 3：两边乘除**负**数 → 不等号方向**改变** ⚠️
- **数轴表示**：用空心圆 `\circ`（不含端点 $<, >$）或实心圆 `\bullet`（含端点 $\leq, \geq$）
- **典型应用** 3 例：
  - 例 1：判断 $3, 5, -1$ 哪些是 $2x + 1 > 4$ 的解
  - 例 2：用不等式表示"$x$ 至少为 5"（$x \geq 5$）"$x$ 不超过 3"（$x \leq 3$）
  - 例 3：含负系数变号 $-2x < 6 \Rightarrow x > -3$（**两边除以 $-2$ 变号**）
- **易错点**：乘除负数变号；解集要画数轴或写区间；"至多/至少"对应 ≤ / ≥
- **自测题** 4 题。

Commit: `docs(chuzhong_algebra): add part7/01 (inequality basics)`

---

## Task 7: part7/02 解一元一次不等式

**File:** `chuzhong_algebra/part7-inequalities/02-solve-linear-inequality.md`（模板 C，1800-2500 字）

Title: `# 解一元一次不等式`

- **步骤**（与解一元一次方程**类似**）：
  1. 去分母
  2. 去括号
  3. 移项（变号）
  4. 合并同类项
  5. 系数化为 1（**注意：乘除负数变号**）
- **解集表示**：
  - 数轴（空心 / 实心圆 + 箭头）
  - 区间 $(a, b)$、$[a, b]$、$(-\infty, a]$ 等
  - 集合 $\{x \mid x > 2\}$
- **典型应用** 3 例：
  - 例 1：基础 $2x - 3 \leq 5 \Rightarrow x \leq 4$
  - 例 2：含分数 $\frac{2x-1}{3} > \frac{x+2}{2}$
  - 例 3：含字母系数 $ax > 2$（讨论 $a > 0, a < 0, a = 0$ 三种情形）
- **易错点**：**乘除负数必须变号**；解集用 $>$ 时数轴是空心圆；含字母系数要讨论
- **自测题** 5 题。

Commit: `docs(chuzhong_algebra): add part7/02 (solve linear inequality)`

---

## Task 8: part7/03 一元一次不等式组

**File:** `chuzhong_algebra/part7-inequalities/03-inequality-system.md`（模板 C，1800-2500 字）

Title: `# 一元一次不等式组`

- **不等式组**：几个一元一次不等式联立
- **解集 = 各不等式解集的交集**
- **求解步骤**：
  1. 分别解每个不等式
  2. 在同一数轴上画出每个解集
  3. 取**公共部分**为最终解集
- **4 种典型情形**（用表格）：

| 情形 | 不等式组 | 解集 |
|---|---|---|
| 同大取大 | $x > a, x > b$（$a < b$） | $x > b$ |
| 同小取小 | $x < a, x < b$（$a < b$） | $x < a$ |
| 大小交叉 | $a < x < b$（$a < b$） | $a < x < b$ |
| 大小逆 | $x > b, x < a$（$a < b$） | 无解 |

- **典型应用** 3 例：
  - 例 1：基础 $\begin{cases} 2x - 1 > 3 \\ 3x - 8 < 1 \end{cases}$（解集 $2 < x < 3$）
  - 例 2：含参 $\begin{cases} x > 2 \\ x < m \end{cases}$ 有解 → $m > 2$
  - 例 3：解集为整数 $\begin{cases} x > 1 \\ x \leq 4 \end{cases}$ 的整数解 $\{2, 3, 4\}$
- **易错点**：解集取**交集**不是并集；含参题判断有解 / 无解 / 整数解
- **自测题** 4 题。

Commit: `docs(chuzhong_algebra): add part7/03 (linear inequality system)`

---

## Task 9: part7/04 不等式应用

**File:** `chuzhong_algebra/part7-inequalities/04-inequality-applications.md`（模板 C，1800-2500 字）

Title: `# 不等式的应用`

- **应用题列不等式的步骤**：
  1. 审题：找出"至少 / 至多 / 不少于 / 不超过"等不等关键词
  2. 设元：与方程应用题一样
  3. 列不等式：用不等关键词翻译
  4. 解不等式
  5. 检验：解是否符合实际意义（如"人数 ≥ 0 且为整数"）
- **关键词对应**：
  - "至少 / 不少于 / 不低于" → $\geq$
  - "至多 / 不超过 / 不大于" → $\leq$
  - "超过 / 大于 / 多于" → $>$
  - "不到 / 小于" → $<$
- **典型应用** 3 例：
  - 例 1：购票题——某景区门票成人 $30$ 元，儿童 $15$ 元，预算 $200$ 元至少买 $10$ 张，求 $x$ 名儿童最少几个
  - 例 2：通讯费——A 套餐 $30$ 元含 $100$ 分钟，超出 $0.3$/分钟；B 套餐 $40$ 元含 $200$ 分钟。求多少分钟时 B 优于 A
  - 例 3：方案优选（多个不等式 + 取整数解）
- **易错点**：不等关键词译法；解集取整数（实际意义）；可能要判断"哪种方案最优"
- **自测题** 4 题。

Commit: `docs(chuzhong_algebra): add part7/04 (inequality applications)`

---

## Task 10: part7/05 含参不等式分类讨论 ⭐（模板 D）

**File:** `chuzhong_algebra/part7-inequalities/05-parametric-inequality.md`（**模板 D**，3000-4000 字）

Title: `# 含参不等式分类讨论`

**一例速记**：
> 解 $kx > 3$（$k$ 为常数）。**三种情形**：① $k > 0$ → $x > \frac{3}{k}$；② $k < 0$ → $x < \frac{3}{k}$（变号！）；③ $k = 0$ → $0 > 3$ 无解。**含参不等式必分情形讨论**。

### 一、引入题
> 解不等式 $kx + 2 < 5$（$k$ 为常数）。

### 二、思维路径还原（12-18 行 `>`）
> "$kx + 2 < 5$ → 移项 $kx < 3$  
> 现在要把 $k$ 除过去，但**不知道 $k$ 符号**！  
> **第一步：分类**——$k > 0, k < 0, k = 0$ 三种情形：  
> **情形 1：$k > 0$** → 两边除以 $k$ 不变号 → $x < \frac{3}{k}$  
> **情形 2：$k < 0$** → 两边除以 $k$ **变号** → $x > \frac{3}{k}$  
> **情形 3：$k = 0$** → $0 < 3$ 永远成立 → $x$ 取**任意实数**  
> 综合三种情形给出完整结论。  
> **关键反射**：含参不等式中**变量系数**为参数时，必按参数正/负/零三种情形讨论。"

### 三、抽象成方法
**含参不等式的 3 类常见题**：

1. **参数为系数**（如 $ax > b$）：按 $a > 0, a < 0, a = 0$ 讨论
2. **解集含参**（如 $x > m$ 中 $m$ 含参）：求满足"解集 / 整数解 / 无解"等条件的参数
3. **不等式组含参**：与单不等式相同的讨论 + 交集分析

### 四、方法变形
- **含参方程的解 + 不等式约束**：综合题，常考"求 $m$ 使方程组的解满足某不等式"
- **绝对值不等式**：$|x - a| < r$ 即 $a - r < x < a + r$（初中淡化）

### 五、思考路标（≥6 条）
- 见参数作系数 → 必讨论参数正负零
- 见参数作常数 → 把含参解集与已知解集对比
- "不等式组有解" → 各不等式解集**有交集**
- "解集为整数" → 列举整数解，反推参数范围
- 系数为负 → 不等号变方向
- 答案形式：参数 $m$ 满足的不等式 / 区间

### 六、应用例题 3 例
- 例 1：解 $(m-1)x > 4$（讨论 $m > 1, m < 1, m = 1$）
- 例 2：求 $m$ 使 $\begin{cases} x > 2 \\ x < m \end{cases}$ 有 3 个整数解
- 例 3：综合（方程组解满足不等式）

### 七、自测题 5 题

Commit: `docs(chuzhong_algebra): add part7/05 (parametric inequality)`

---

## Task 11: AM3 收尾

```bash
cd /Users/yangyang/ai_projs/math
grep -rn 'TBD\|TODO\|XXX\|待补' chuzhong_algebra/part5-linear-eq chuzhong_algebra/part7-inequalities
ls chuzhong_algebra/part5-linear-eq
ls chuzhong_algebra/part7-inequalities
printf '\n---\n**AM3 完成于：YYYY-MM-DD**\n' >> docs/superpowers/plans/2026-05-12-chuzhong-algebra-AM3.md
git add docs/superpowers/plans/2026-05-12-chuzhong-algebra-AM3.md
git commit -m "docs(chuzhong_algebra): mark AM3 milestone complete"
git push origin master
```
