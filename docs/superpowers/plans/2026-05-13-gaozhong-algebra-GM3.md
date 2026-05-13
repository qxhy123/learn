# 高中代数教程 GM3 里程碑实施计划

> **REQUIRED SUB-SKILL**: superpowers:subagent-driven-development

**Goal:** 完成 Part 4 指数对数幂函数（3 章）+ Part 5 三角函数（8 章），共 11 章。

**Architecture:** 8 章模板 C + **3 章模板 D**（对数、同角恒等式、和差倍半角）。每章独立提交。

**Spec 参考:** `docs/superpowers/specs/2026-05-13-gaozhong-algebra-design.md`
**前序里程碑:** GM0、GM1、GM2（已完成）

---

## 文件结构

```
gaozhong_math/algebra/
├── part4-exp-log-power/                  # [高一上]
│   ├── 01-exponential-function.md
│   ├── 02-logarithm-and-log-function.md  # ⭐ 对数运算与对数函数
│   └── 03-power-function.md
└── part5-trigonometry/                   # [高一上 / 高一下]
    ├── 01-radian-measure.md
    ├── 02-trig-on-unit-circle.md
    ├── 03-trig-graphs-and-properties.md
    ├── 04-trig-identities.md             # ⭐ 同角/诱导公式
    ├── 05-sum-difference-formulas.md     # ⭐ 和差倍半角
    ├── 06-trig-equations.md
    ├── 07-solving-triangle.md
    └── 08-trig-applications.md
```

模板 C 6 节 **2200-3200 字**；模板 D 7 节 **3500-5000 字**。LaTeX 全数学。

---

## Part 4: 指数 / 对数 / 幂函数

### Task 1: part4/01 指数函数（模板 C）

Title: `# 指数与指数函数`

涵盖：
- **指数运算**：$a^m \cdot a^n = a^{m+n}$，$(a^m)^n = a^{mn}$，$\sqrt[n]{a^m} = a^{m/n}$
- **零指数 / 负指数**：$a^0 = 1$（$a \ne 0$），$a^{-n} = \frac{1}{a^n}$
- **分数指数幂**：$a^{m/n} = \sqrt[n]{a^m}$（$a > 0$）
- **指数函数**：$y = a^x$（$a > 0, a \ne 1$）
- **性质**（用表格，按 $a > 1$ vs $0 < a < 1$ 分类）：
  - 定义域 $\mathbb{R}$，值域 $(0, +\infty)$
  - $a > 1$：**增**；$0 < a < 1$：**减**
  - 恒过 $(0, 1)$
- **图象特征**：曲线、单调、无渐近线（左侧 / 右侧渐近 $x$ 轴）
- 典型应用 3 例
- 易错点：底数限制 $a > 0, a \ne 1$；指数运算变形
- 自测题 4 题

### Task 2: part4/02 对数与对数函数 ⭐（模板 D）

Title: `# 对数运算与对数函数`

**一例速记**：
> $\log_a b = N$ ⇔ $a^N = b$ —— 对数是指数的反函数  
> 三大运算法则：$\log_a (MN) = \log_a M + \log_a N$、$\log_a \frac{M}{N} = \log_a M - \log_a N$、$\log_a M^n = n \log_a M$  
> **换底公式**：$\log_a b = \frac{\log_c b}{\log_c a}$

**引入题**：计算 $\log_2 3 \cdot \log_3 4 \cdot \log_4 5 \cdot \log_5 8$。

**思维路径还原**（15-20 行）：用换底公式 + 链式约分。

**3 类常见题型**：① 化简（用运算法则）② 求值（含换底）③ 函数性质（按 $a > 1$ vs $0 < a < 1$）

思考路标 ≥8 条。

### Task 3: part4/03 幂函数（模板 C）

Title: `# 幂函数`

涵盖：
- 定义 $y = x^\alpha$（$\alpha$ 是常数）
- 5 个常见幂函数图象：$y = x, y = x^2, y = x^3, y = x^{1/2}, y = x^{-1}$
- 性质表（奇偶性 / 单调性 / 定义域）
- 应用：比较幂的大小
- 典型应用 3 例 + 易错点 + 自测题 4 题

---

## Part 5: 三角函数

### Task 4: part5/01 弧度制（模板 C）

Title: `# 角的弧度制`

- 弧度定义：弧长 = 半径时的圆心角为 1 弧度
- 角度与弧度换算：$180° = \pi$ rad
- 任意角的概念（终边落在哪个象限）
- 终边相同的角：$\alpha + 2k\pi$
- 扇形弧长 $l = r\theta$、面积 $S = \frac{1}{2} r^2 \theta = \frac{1}{2} l r$
- 典型应用 + 易错点 + 自测题 4 题

### Task 5: part5/02 单位圆与三角函数定义（模板 C）

Title: `# 三角函数的定义（单位圆视角）`

- 用**单位圆**定义 $\sin\alpha, \cos\alpha, \tan\alpha$：单位圆上终边点 $(\cos\alpha, \sin\alpha)$
- 各象限符号：**一全二正弦三切四余弦**口诀
- 特殊角三角函数值：$0, \frac{\pi}{6}, \frac{\pi}{4}, \frac{\pi}{3}, \frac{\pi}{2}$ 的 $\sin, \cos, \tan$ 表
- 三角函数的奇偶性、周期性
- 典型应用 3 例 + 自测题 4 题

### Task 6: part5/03 图象与性质（模板 C，2500-3500 字）

Title: `# 三角函数的图象与性质`

- $y = \sin x, y = \cos x, y = \tan x$ 图象（用单位圆推导）
- 性质表（用表格）：定义域、值域、周期、奇偶性、单调区间、对称轴 / 对称中心
- $y = A\sin(\omega x + \varphi) + b$：振幅、周期、相位、平移
- 平移规律
- 典型应用 + 易错点 + 自测题 5 题

### Task 7: part5/04 同角恒等式与诱导公式 ⭐（模板 D）

Title: `# 同角三角恒等式与诱导公式`

**一例速记**：
> 三大同角恒等式：$\sin^2\alpha + \cos^2\alpha = 1$、$\tan\alpha = \frac{\sin\alpha}{\cos\alpha}$、$\sec^2 - \tan^2 = 1$  
> 诱导公式口诀：**奇变偶不变，符号看象限**

**引入题**：已知 $\sin\alpha = \frac{3}{5}$（$\alpha$ 在第二象限），求 $\cos\alpha, \tan\alpha$。

思维路径还原 15-20 行 + 思考路标 ≥8 条。

**诱导公式 6 组**：$\frac{\pi}{2} - \alpha, \pi - \alpha, \pi + \alpha, 2\pi - \alpha, -\alpha, \frac{\pi}{2} + \alpha$ 的三角函数

### Task 8: part5/05 和差倍半角公式 ⭐（模板 D）

Title: `# 三角函数的和差倍半角公式`

**一例速记**：
> 和角公式（必背）：  
> $\sin(\alpha \pm \beta) = \sin\alpha\cos\beta \pm \cos\alpha\sin\beta$  
> $\cos(\alpha \pm \beta) = \cos\alpha\cos\beta \mp \sin\alpha\sin\beta$  
> $\tan(\alpha \pm \beta) = \frac{\tan\alpha \pm \tan\beta}{1 \mp \tan\alpha\tan\beta}$  
> **倍角公式**：$\sin 2\alpha = 2\sin\alpha\cos\alpha$、$\cos 2\alpha = \cos^2\alpha - \sin^2\alpha = 2\cos^2 - 1 = 1 - 2\sin^2$

**引入题**：化简 $\sin 75°$。
**思维路径还原**：$\sin 75° = \sin(45° + 30°)$ → 用和角公式。

思维路径 15-20 行 + 思考路标 ≥8 条 + 5 类常见技巧（辅助角 $a\sin x + b\cos x = \sqrt{a^2+b^2}\sin(x+\varphi)$ 等）

### Task 9: part5/06 三角方程（模板 C）

Title: `# 简单三角方程的求解`

- 基本三角方程 $\sin x = a, \cos x = a, \tan x = a$
- 一般解：$x = (-1)^k \arcsin a + k\pi$ 等
- 三角方程的常见类型：可化为二次的（如 $2\sin^2 x - 3\sin x + 1 = 0$）
- 典型应用 3 例 + 自测题 4 题

### Task 10: part5/07 解三角形（模板 C，2500-3500 字）

Title: `# 解三角形：正弦定理与余弦定理`

- **正弦定理**：$\frac{a}{\sin A} = \frac{b}{\sin B} = \frac{c}{\sin C} = 2R$
- **余弦定理**：$c^2 = a^2 + b^2 - 2ab\cos C$
- **面积公式**：$S = \frac{1}{2}ab\sin C$
- 解三角形的 4 类常见情形（SSS, SAS, AAS, ASS 后者需讨论解的个数）
- 典型应用 3 例（基础 + 含参 + 实际应用如测距）
- 自测题 5 题

### Task 11: part5/08 三角应用（模板 C）

Title: `# 三角函数的应用`

- 振幅 / 周期 / 相位的实际意义
- 周期现象建模（潮汐、温度、声波）
- 实际测量问题（仰角俯角 + 解三角形）
- 典型应用 3 例 + 自测题 4 题

---

## Task 12: GM3 收尾

```bash
cd /Users/yangyang/ai_projs/math
grep -rn 'TBD\|TODO\|XXX\|待补' gaozhong_math/algebra/part4-exp-log-power gaozhong_math/algebra/part5-trigonometry
ls gaozhong_math/algebra/part4-exp-log-power gaozhong_math/algebra/part5-trigonometry
printf '\n---\n**GM3 完成于：YYYY-MM-DD**\n' >> docs/superpowers/plans/2026-05-13-gaozhong-algebra-GM3.md
git add docs/superpowers/plans/2026-05-13-gaozhong-algebra-GM3.md
git commit -m "docs(gaozhong/algebra): mark GM3 milestone complete"
git push origin master
```

---

## 调度

**3 个 sonnet subagent 并行：**
- A: Part 4 三章（01 指数 / 02 ⭐ 对数 / 03 幂）
- B: Part 5 前 4 章（01 弧度 / 02 单位圆 / 03 图象性质 / 04 ⭐ 同角诱导）
- C: Part 5 后 4 章（05 ⭐ 和差倍半角 / 06 三角方程 / 07 解三角形 / 08 应用）

每章写完立刻 commit（避免超时丢失）。

---
**GM3 完成于：2026-05-13**
