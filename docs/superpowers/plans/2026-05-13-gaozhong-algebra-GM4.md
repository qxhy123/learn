# 高中代数教程 GM4 里程碑实施计划（种子阶段 appendix）

> **REQUIRED SUB-SKILL**: superpowers:subagent-driven-development

**Goal:** 完成 `gaozhong_math/algebra/appendix/` 子集——公式表、模型图集、高一上 80 题题库、详解。
**Architecture:** 4 个 markdown 文件，覆盖 GM0-GM3 已完成的 42 章内容（Part 1-5，高一上）。每文件独立提交。

**Spec 参考:** `docs/superpowers/specs/2026-05-13-gaozhong-algebra-design.md`
**前序里程碑:** GM0、GM1、GM2、GM3（已完成 42 章 + 12 toolkit）

---

## 文件结构

```
gaozhong_math/algebra/appendix/
├── A-formula-sheet.md      # 公式表（高一上覆盖范围）
├── B-model-atlas.md        # 模型图集（高一上典型套路）
├── C-exercises-seed.md     # 高一上 80 题（按 Part 1-5 分布，按难度分层）
└── F-solutions-seed.md     # 80 题详解
```

字数：A ~ 3000-4500、B ~ 3000-4500、C ~ 2800-3800、F ~ 8000-12000（含完整解析）

---

## Task 1: appendix/A 公式表

**Files:**
- Create: `gaozhong_math/algebra/appendix/A-formula-sheet.md`

Title: `# 附录 A：高一上代数公式表`

**结构（按 Part 顺序整理）：**

### Part 1 集合与逻辑
- 集合三性、子集 / 真子集 / 幂集个数公式 $2^n$ / $2^n - 1$ / $2^n$
- 集合运算：$A \cup B$, $A \cap B$, $\complement_U A$ 的德摩根律
- 全称 / 存在量词与其否定
- 充分必要条件（$p \Rightarrow q$ / $q \Rightarrow p$）

### Part 2 不等式
- 不等式基本性质（同向相加、正数相乘）
- 一元二次不等式 $ax^2+bx+c>0$ 解集（按 $a, \Delta$ 分类）
- 分式不等式 $\frac{f}{g} \gtrless 0$ 等价转化
- 基本不等式：$a+b \ge 2\sqrt{ab}$（$a, b > 0$）、$\frac{a+b}{2} \ge \sqrt{ab} \ge \frac{2ab}{a+b}$、$a^2+b^2 \ge 2ab$
- 三元基本不等式

### Part 3 函数概念
- 函数三要素（定义域 / 值域 / 对应关系）
- 单调性定义（$\forall x_1 < x_2, f(x_1) < f(x_2)$ → 增）
- 奇偶性：$f(-x) = f(x)$ 偶、$f(-x) = -f(x)$ 奇
- 复合函数单调性"同增异减"
- 函数变换（平移 / 翻折 / 对称）

### Part 4 指数对数幂
- 指数运算 5 条法则
- 对数运算 3 法则 + 换底公式
- 指 / 对函数性质表（$a > 1$ vs $0 < a < 1$）
- 幂函数 5 个常见图象

### Part 5 三角函数
- 弧度与角度换算 $180° = \pi$
- 扇形 $l = r\theta$，$S = \frac{1}{2}r^2\theta = \frac{1}{2}lr$
- 同角恒等式：$\sin^2 + \cos^2 = 1$，$\tan = \sin/\cos$
- 诱导公式 6 组（奇变偶不变，符号看象限）
- 和角公式：$\sin(\alpha\pm\beta), \cos(\alpha\pm\beta), \tan(\alpha\pm\beta)$
- 倍角公式：$\sin 2\alpha, \cos 2\alpha$（3 形式），$\tan 2\alpha$
- 半角公式（可选）
- 辅助角 $a\sin x + b\cos x = \sqrt{a^2+b^2}\sin(x+\varphi)$
- 三角函数图象与性质表
- 正弦 / 余弦定理 + 面积公式

**特殊角函数值表**（0, 30°, 45°, 60°, 90°, 120°, 135°, 150°, 180°）

Commit: `docs(gaozhong/algebra): add appendix A (formula sheet for high school year 1)`

---

## Task 2: appendix/B 模型图集

**Files:**
- Create: `gaozhong_math/algebra/appendix/B-model-atlas.md`

Title: `# 附录 B：高一上代数模型图集`

**结构（按 toolkit 12 篇 + Part 1-5 典型模型）：**

### 模型 1：集合 Venn 图与运算
- $A \cup B$, $A \cap B$, $\complement_U A$, 德摩根律可视化
- 子集层级关系（一画即清）

### 模型 2：一元二次不等式三态
- $\Delta > 0$ / $\Delta = 0$ / $\Delta < 0$ 时解集的图象判定
- 抛物线开口 $a > 0$ vs $a < 0$ 双向考虑

### 模型 3：分式不等式标轴法（穿针引线）
- $\frac{(x-1)(x-2)}{(x-3)(x-4)} > 0$ 的"穿针引线"图
- 奇穿偶不穿规则

### 模型 4：基本不等式"和定积大、积定和小"
- $a + b = $ 定值时 $ab$ 最大（取 $a=b$）
- $ab = $ 定值时 $a + b$ 最小

### 模型 5：函数单调性四象限决策
- 复合 $f(g(x))$ 单调性"同增异减"图
- 抽象函数 $f(x_1) - f(x_2)$ 处理图

### 模型 6：奇偶性与图象对称
- 奇 = 中心对称（原点）
- 偶 = 轴对称（$y$ 轴）
- 半奇半偶（分段）情形

### 模型 7：含参一元二次（Part 3 经典）
- 根的分布"两正"/"两负"/"一正一负"判别框图
- 顶点在 / 不在区间内时的极值讨论

### 模型 8：指数 / 对数函数对比图
- $y = a^x$ 与 $y = \log_a x$ 关于 $y = x$ 对称
- $a > 1$ vs $0 < a < 1$ 双对比

### 模型 9：三角函数单位圆
- 终边在四象限 → 三角函数符号"一全二正弦三切四余弦"
- 诱导公式直观图

### 模型 10：$y = A\sin(\omega x + \varphi) + b$ 变换链
- 平移→伸缩→振幅 → $b$ 上下移
- 从 $y = \sin x$ 到 $y = 2\sin(2x - \pi/3) + 1$ 一步步可视化

### 模型 11：辅助角图
- $a\sin x + b\cos x$ → $R\sin(x+\varphi)$ 几何意义
- 用单位圆推 $R = \sqrt{a^2 + b^2}$，$\tan\varphi = b/a$

### 模型 12：解三角形二解情形
- ASS（边边角）情形下解的个数判别
- 钝角 / 锐角 / 直角三种判定

每模型配：
- **触发条件**（什么题考这个模型）
- **图象 / 示意**（可用 ASCII 简图或描述：见 figures/）
- **关键技巧**（1-2 句）
- **典型题型**（1-2 道）

Commit: `docs(gaozhong/algebra): add appendix B (model atlas for high school year 1)`

---

## Task 3: appendix/C 高一上 80 题题库

**Files:**
- Create: `gaozhong_math/algebra/appendix/C-exercises-seed.md`

Title: `# 附录 C：高一上代数 80 题题库`

**分布**（按 Part 章节，三档难度）：

| Part | 基础 | 中档 | 提升 | 小计 |
|---|---|---|---|---|
| Part 1 集合逻辑 | 4 | 4 | 2 | 10 |
| Part 2 不等式 | 4 | 5 | 3 | 12 |
| Part 3 函数概念 | 6 | 8 | 4 | 18 |
| Part 4 指对幂 | 4 | 4 | 2 | 10 |
| Part 5 三角函数 | 8 | 12 | 6 | 26 |
| 综合（跨 Part） | — | 2 | 2 | 4 |
| **合计** | **26** | **35** | **19** | **80** |

**题号约定：** C.01 - C.80，按 Part 顺序连续编号。

**每题包含：**
- 题号
- 来源/难度标签：[基础] / [中档] / [提升]
- 关联章节：Part X / xx 章
- 题面（LaTeX 公式）

**举例：**
```
**C.15** [中档] Part 3/03  
若函数 $f(x) = \frac{x^2 + ax + 1}{x}$ 在区间 $[1, +\infty)$ 上单调递增，求 $a$ 的取值范围。
```

Commit: `docs(gaozhong/algebra): add appendix C (80 problems for high school year 1)`

---

## Task 4: appendix/F 高一上 80 题详解

**Files:**
- Create: `gaozhong_math/algebra/appendix/F-solutions-seed.md`

Title: `# 附录 F：高一上 80 题详解`

**每题解析包含：**
1. 题号回顾（再次列出题面）
2. **思路**：1-3 行说明本题考察什么、套路是什么（**关键！** 这是模板 D 思想在题库的延伸）
3. **解答**：完整推导，每步骤的关键代数变形展示
4. **总结**：1-2 句指出本题的"识题特征"（下次见这类题怎么 1 秒识别）

**特别说明：**
- 答案要"看得懂、学得会"
- 中档以上题目展示完整思维链
- 关键步骤标注（"换元"、"放缩"、"分类"等）
- 对应 toolkit 篇章用 → 引用：如 → toolkit/03 构造法

Commit: `docs(gaozhong/algebra): add appendix F (full solutions for 80 problems)`

---

## Task 5: GM4 收尾

```bash
cd /Users/yangyang/ai_projs/math
grep -rn 'TBD\|TODO\|XXX\|待补' gaozhong_math/algebra/appendix
ls gaozhong_math/algebra/appendix
printf '\n---\n**GM4 完成于：YYYY-MM-DD**\n' >> docs/superpowers/plans/2026-05-13-gaozhong-algebra-GM4.md
git add docs/superpowers/plans/2026-05-13-gaozhong-algebra-GM4.md
git commit -m "docs(gaozhong/algebra): mark GM4 milestone complete (seed phase done)"
git push origin master
```

---

## 调度

**2 个 sonnet subagent 并行：**
- A: appendix A（公式表）+ appendix B（模型图集）
- B: appendix C（80 题题库）+ appendix F（80 题详解）

每文件独立 commit。完成后报告：commit SHA、字数、题号分布。

---

## 种子完成里程碑

GM4 完成后，整个"高中代数种子"宣告就绪：
- 13 文件方法论框架（GM0）
- 42 章正文（GM1-GM3）
- 4 文件附录（GM4）
- 共计 59 文件

适合作为独立的"**高一上代数学习指南**"使用。

下一步（用户决策点）：
- ✅ 看效果好 → 继续 GM5-GM12（Part 6-13 + 完整题库 240 题 + figures ~150 张）
- 调整模板 / 继续 → 重写部分章节
- 暂停 → 种子也已是独立可用的学习材料
