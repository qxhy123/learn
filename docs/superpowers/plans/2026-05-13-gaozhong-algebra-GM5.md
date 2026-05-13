# 高中代数教程 GM5 里程碑实施计划

> **REQUIRED SUB-SKILL**: superpowers:subagent-driven-development

**Goal:** 完成 Part 6 复数（2 章）+ Part 7 统计（3 章）+ Part 8 概率（5 章），共 10 章。

**Architecture:** 7 章模板 C + **3 章模板 D**（条件概率、贝叶斯、独立性）。每章独立提交。

**Spec 参考:** `docs/superpowers/specs/2026-05-13-gaozhong-algebra-design.md`
**前序里程碑:** GM0、GM1、GM2、GM3、GM4（种子已完成）

---

## 文件结构

```
gaozhong_math/algebra/
├── part6-complex-numbers/        # [高一下]
│   ├── 01-complex-basics.md
│   └── 02-complex-operations.md
├── part7-statistics/             # [高一下]
│   ├── 01-sampling.md
│   ├── 02-data-features.md
│   └── 03-correlation.md
└── part8-probability/            # [高一下 + 选必三]
    ├── 01-classical-probability.md
    ├── 02-geometric-probability.md
    ├── 03-independence.md           # ⭐
    ├── 04-conditional-probability.md  # ⭐
    └── 05-bayes-and-total-probability.md  # ⭐
```

模板 C 6 节 2200-3200 字；模板 D 7 节 3500-5000 字。LaTeX 全数学。

---

## Part 6: 复数

### Task 1: part6/01 复数基础（模板 C）

Title: `# 复数的概念与几何意义`

涵盖：
- **虚数单位** $i$，$i^2 = -1$；周期性 $i^{4k}=1, i^{4k+1}=i, i^{4k+2}=-1, i^{4k+3}=-i$
- **复数** $z = a + bi$（$a, b \in \mathbb{R}$，$a$ 实部、$b$ 虚部）
- 复数集合 $\mathbb{C}$，包含关系 $\mathbb{R} \subset \mathbb{C}$
- 复数相等：$a + bi = c + di \Leftrightarrow a = c \land b = d$
- **几何意义**：复平面（横轴实，纵轴虚），$z$ 对应点 $(a, b)$ 与向量
- 共轭复数 $\overline{z} = a - bi$，性质 $z \cdot \overline{z} = |z|^2$
- **模** $|z| = \sqrt{a^2 + b^2}$
- 典型应用 3 例（求实虚部 / 求模 / 几何意义）+ 自测题 4 题
- 易错点：$i^2 = -1$ 不是 $i^2 = 1$；不能直接 $\sqrt{-1} \cdot \sqrt{-1} = \sqrt{1}$

### Task 2: part6/02 复数运算（模板 C）

Title: `# 复数的四则运算`

涵盖：
- **加减法**：$(a+bi) \pm (c+di) = (a \pm c) + (b \pm d) i$（几何意义：向量加减）
- **乘法**：$(a+bi)(c+di) = (ac - bd) + (ad+bc)i$
- **除法**：分母实数化（乘以共轭）：$\frac{a+bi}{c+di} = \frac{(a+bi)(c-di)}{c^2+d^2}$
- 性质：模的乘性 $|z_1 z_2| = |z_1| |z_2|$；共轭的乘性 $\overline{z_1 z_2} = \overline{z_1} \cdot \overline{z_2}$
- 实系数方程根的特性：若 $z$ 是实系数 $f(x)$ 的根则 $\overline{z}$ 也是
- 典型应用 3 例（含 $i$ 高次幂、复数方程、共轭技巧）
- 自测题 4 题 + 易错点

---

## Part 7: 统计

### Task 3: part7/01 抽样（模板 C）

Title: `# 抽样方法`

涵盖：
- **简单随机抽样**：抽签法 / 随机数法
- **分层抽样**：按比例从各层抽（$n_i = N_i \cdot \frac{n}{N}$）
- **系统抽样**：按等距规则
- 三种方法适用场景对比
- 典型应用 3 例 + 自测题 4 题
- 易错点：分层抽样按比例 / 系统抽样起点要随机

### Task 4: part7/02 数据特征（模板 C）

Title: `# 数据的数字特征`

涵盖：
- **集中趋势**：平均数 $\bar{x} = \frac{1}{n}\sum x_i$、中位数、众数
- **离散程度**：方差 $s^2 = \frac{1}{n}\sum(x_i - \bar{x})^2$、标准差 $s$
- 极差、四分位距
- 频率分布直方图（关键：频率 / 组距 = 纵轴）
- **百分位数**：$p$% 分位数 = 至少有 $p$% 数据 $\leq$ 它
- 典型应用 3 例（用频率分布求平均 / 方差含义 / 比较两组数据）+ 自测题 5 题

### Task 5: part7/03 相关性（模板 C）

Title: `# 变量的相关性与线性回归`

涵盖：
- 散点图直观判断相关性
- **相关系数** $r = \frac{\sum(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum(x_i-\bar{x})^2 \sum(y_i-\bar{y})^2}}$
- $r$ 的取值范围 $[-1, 1]$ 与含义
- **最小二乘线性回归方程** $\hat{y} = \hat{b}x + \hat{a}$
  - $\hat{b} = \frac{\sum(x_i-\bar{x})(y_i-\bar{y})}{\sum(x_i-\bar{x})^2}$
  - $\hat{a} = \bar{y} - \hat{b}\bar{x}$
- 残差 $e_i = y_i - \hat{y}_i$ 与决定系数 $R^2$
- 典型应用 3 例 + 自测题 4 题

---

## Part 8: 概率

### Task 6: part8/01 古典概型（模板 C）

Title: `# 古典概型与基本概念`

涵盖：
- 随机事件 / 必然事件 / 不可能事件
- **样本空间** $\Omega$ 与样本点
- 事件运算：互斥、对立、相互独立
- **古典概型** $P(A) = \frac{n(A)}{n(\Omega)}$
- 互斥事件加法：$P(A \cup B) = P(A) + P(B)$（互斥）
- 对立事件 $P(\overline{A}) = 1 - P(A)$
- 典型应用 3 例（摸球 / 投骰子 / 排列计数与概率）
- 易错点：列样本点要"等可能" / 互斥 ≠ 独立
- 自测题 4 题

### Task 7: part8/02 几何概型（模板 C）

Title: `# 几何概型`

涵盖：
- 几何概型定义：$P(A) = \frac{A \text{的几何度量}}{\Omega \text{的几何度量}}$
- 一维（长度）、二维（面积）、三维（体积）
- 典型问题：
  - 在线段上随机取点
  - 在区域内随机取点（如蒲丰投针入门版）
  - 等公交时间问题
- 典型应用 3 例 + 自测题 4 题
- 易错点：选择"均匀分布"的几何度量

### Task 8: part8/03 独立性 ⭐（模板 D）

Title: `# 事件的独立性`

**一例速记**：
> $A, B$ 独立 $\Leftrightarrow P(AB) = P(A) P(B)$  
> 多事件独立要满足所有阶乘积

**引入题**：袋中 3 红 2 白，有放回摸两次，求两次都红的概率。

**思维路径还原**（15-20 行）：识别"有放回 → 独立"→ 用乘积。

**3 类常见题型**：① 有放回 / 多次独立试验 ② 元件可靠性（串并联）③ 检测合格率

**思考路标 ≥8 条**：
- 看到"有放回" / "互不影响" → 独立
- 多个独立事件全发生 → 乘积
- 至少一个发生 → $1 - $ 全不发生
- 独立 ≠ 互斥（互斥是不可同时发生，独立是发生与否无关）

例题 3 例 + 自测题 5 题。

### Task 9: part8/04 条件概率 ⭐（模板 D）

Title: `# 条件概率`

**一例速记**：
> $P(B|A) = \frac{P(AB)}{P(A)}$（$P(A) > 0$）  
> "在 $A$ 已发生的前提下 $B$ 发生的概率"

**引入题**：袋中 5 红 3 白，不放回摸 2 球，已知第 1 球红，求第 2 球红的概率。

**思维路径还原** 15-20 行：从样本空间缩小角度理解。

**3 类常见题型**：① 已知前提求条件概率 ② 全概率公式应用 ③ 在条件下的对立 / 独立

**思考路标 ≥8 条**：
- 看到"在……的条件下" → 条件概率
- 条件概率 ≠ 联合概率
- $P(B|A) \cdot P(A) = P(AB)$
- 若 $A, B$ 独立则 $P(B|A) = P(B)$

例题 3 例 + 自测题 5 题。

### Task 10: part8/05 全概率与贝叶斯 ⭐（模板 D）

Title: `# 全概率公式与贝叶斯公式`

**一例速记**：
> 全概率：$P(B) = \sum_i P(A_i) P(B|A_i)$（其中 $\{A_i\}$ 是 $\Omega$ 的一个划分）  
> 贝叶斯：$P(A_i|B) = \frac{P(A_i) P(B|A_i)}{P(B)}$ —— 由结果倒推原因的概率

**引入题**：两个袋，袋 1 有 5 红 5 白，袋 2 有 3 红 7 白。任选一袋取 1 球，求取到红球的概率；若已知取到红球，求来自袋 1 的概率。

**思维路径还原** 15-20 行：用全概率求 $P(B)$，再用贝叶斯求 $P(A_1|B)$。

**3 类常见题型**：① 链式抽样 ② 检验设备的"误诊" / "漏诊"问题 ③ 决策与信息更新

**思考路标 ≥8 条**：
- 多源 / 分层 → 全概率
- 由果求因 → 贝叶斯
- $P(A_i|B)$ 是后验，$P(A_i)$ 是先验

例题 3 例 + 自测题 5 题。

---

## Task 11: GM5 收尾

```bash
cd /Users/yangyang/ai_projs/math
grep -rn 'TBD\|TODO\|XXX\|待补' gaozhong_math/algebra/part6-complex-numbers gaozhong_math/algebra/part7-statistics gaozhong_math/algebra/part8-probability
ls gaozhong_math/algebra/part6-complex-numbers gaozhong_math/algebra/part7-statistics gaozhong_math/algebra/part8-probability
printf '\n---\n**GM5 完成于：YYYY-MM-DD**\n' >> docs/superpowers/plans/2026-05-13-gaozhong-algebra-GM5.md
git add docs/superpowers/plans/2026-05-13-gaozhong-algebra-GM5.md
git commit -m "docs(gaozhong/algebra): mark GM5 milestone complete"
git push origin master
```

---

## 调度

**3 个 sonnet subagent 并行：**
- A: Part 6 复数（2 章）
- B: Part 7 统计（3 章）
- C: Part 8 概率（5 章：3 ⭐ 模板 D）

每章写完立刻 commit。

---
**GM5 完成于：2026-05-13**
