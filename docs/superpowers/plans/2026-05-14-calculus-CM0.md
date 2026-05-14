# 微积分教程 CM0 里程碑实施计划

> **REQUIRED SUB-SKILL**: superpowers:subagent-driven-development + rendering-math-figures

**Goal:** 完成 `calculus-tutorial/` 改造的方法论框架——README 扩展、preface 扩展、12 篇 thinking-toolkit、figures 基础设施。

**Architecture:** 不动现有 28 章正文。在根目录新增 thinking-toolkit/、figures/ 目录；扩展 README + preface 加上 toolkit 导航与"模板 D 价值"段。

**Spec 参考:** `docs/superpowers/specs/2026-05-14-calculus-refactor-design.md`
**前序：** 无（这是 CM0）。已有 `gaozhong_math/algebra/` 和 `gaozhong_math/geometry/` 作为风格参照。

---

## 文件结构（CM0 产出 15 个文件 + figures 目录）

```
calculus-tutorial/
├── README.md                       # （扩展，加 toolkit + appendix 导航 + 模板 D 价值）
├── 00-preface.md                   # （扩展，加"模板 D 思维训练"段）
├── thinking-toolkit/               # 新增
│   ├── 01-epsilon-language.md
│   ├── 02-equivalent-infinitesimals.md
│   ├── 03-differentiation-rules.md
│   ├── 04-integration-techniques.md
│   ├── 05-series-convergence.md
│   ├── 06-taylor-and-error.md
│   ├── 07-multivar-chain.md
│   ├── 08-multivar-integration.md
│   ├── 09-ode-classification.md
│   ├── 10-convexity-extrema.md
│   ├── 11-inequality-techniques.md
│   └── 12-calculus-for-ai.md
└── figures/                        # 新增
    ├── render.sh
    ├── STYLE.md
    └── src/{tikz,asy}/ + svg/
```

---

## 调度策略

**Phase 1（基础设施 + 元文件，串行 1 个 subagent）**：
- 拷贝 figures 基础设施 + 扩展 README + 扩展 preface

**Phase 2（toolkit 12 篇，并行 3 sonnet subagent）**：
- A 写 toolkit 01-04（极限 / 等价无穷小 / 求导 / 积分）
- B 写 toolkit 05-08（级数 / Taylor / 多元链式 / 多元积分）
- C 写 toolkit 09-12（ODE / 凸性 / 不等式 / AI 微积分）

每篇 toolkit 字数 **3000-4500**（比初高中略长，因微积分概念密度高）。用 `>` 引用块写演示题思维路径，含思考路标和自测题。

---

## Task 1: Phase 1（figures 基建 + README + preface 扩展）

**Files:**
- Create: `calculus-tutorial/figures/render.sh`（从代数复用）
- Create: `calculus-tutorial/figures/STYLE.md`（从代数复用）
- Create: `calculus-tutorial/figures/src/tikz/.gitkeep`
- Create: `calculus-tutorial/figures/src/asy/.gitkeep`
- Create: `calculus-tutorial/figures/svg/.gitkeep`
- Modify: `calculus-tutorial/README.md`（扩展，**不重写**）
- Modify: `calculus-tutorial/00-preface.md`（扩展，**不重写**）

执行步骤：

- [ ] **Step 1: 创建目录骨架**

```bash
cd /Users/yangyang/ai_projs/math
mkdir -p calculus-tutorial/figures/{src/tikz,src/asy,svg} calculus-tutorial/thinking-toolkit
```

- [ ] **Step 2: 拷贝 figures 基础设施**

```bash
cd /Users/yangyang/ai_projs/math
cp gaozhong_math/algebra/figures/render.sh calculus-tutorial/figures/render.sh
cp gaozhong_math/algebra/figures/STYLE.md calculus-tutorial/figures/STYLE.md
chmod +x calculus-tutorial/figures/render.sh
# 路径替换：把"gaozhong_math/algebra"替换为"calculus-tutorial"
sed -i '' 's|gaozhong_math/algebra|calculus-tutorial|g' calculus-tutorial/figures/render.sh
sed -i '' 's|gaozhong_math/algebra|calculus-tutorial|g' calculus-tutorial/figures/STYLE.md
# render.sh 中如果有"gaozhong/algebra"也替换
sed -i '' 's|gaozhong/algebra|calculus|g' calculus-tutorial/figures/render.sh
sed -i '' 's|gaozhong/algebra|calculus|g' calculus-tutorial/figures/STYLE.md
# 添加 .gitkeep
touch calculus-tutorial/figures/src/tikz/.gitkeep calculus-tutorial/figures/src/asy/.gitkeep calculus-tutorial/figures/svg/.gitkeep
```

- [ ] **Step 3: 验证 render.sh 工作**

```bash
cd /Users/yangyang/ai_projs/math/calculus-tutorial/figures
cat > src/tikz/test-curve.tex <<'EOF'
\documentclass[tikz,border=4pt]{standalone}
\usepackage{ctex}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\begin{document}
\begin{tikzpicture}
\begin{axis}[axis lines=middle, width=6cm, height=4cm, samples=100]
\addplot[blue, thick, domain=-3:3] {x^2};
\end{axis}
\end{tikzpicture}
\end{document}
EOF
./render.sh src/tikz/test-curve.tex
ls svg/test-curve.svg && rm src/tikz/test-curve.tex svg/test-curve.svg
```

Expected: 输出 `✓ svg/test-curve.svg`。

- [ ] **Step 4: 扩展 README.md（在末尾追加段落）**

不动现有 README 内容。在文件末尾追加：

```markdown
---

## 改造升级（2026-05-14）

本教程已按 `gaozhong_math/algebra/` 模式做了"包装式改造"：

### 新增 thinking-toolkit（12 篇）

`thinking-toolkit/` 目录是独立的"微积分思维方法论"小册子：

| 编号 | 主题 |
|---|---|
| 01 | [极限的 ε 语言](./thinking-toolkit/01-epsilon-language.md) |
| 02 | [等价无穷小与小 o](./thinking-toolkit/02-equivalent-infinitesimals.md) |
| 03 | [求导套路系统化](./thinking-toolkit/03-differentiation-rules.md) |
| 04 | [积分技巧反 LIATE](./thinking-toolkit/04-integration-techniques.md) |
| 05 | [级数判敛流程图](./thinking-toolkit/05-series-convergence.md) |
| 06 | [Taylor 展开与误差](./thinking-toolkit/06-taylor-and-error.md) |
| 07 | [多元链式与梯度](./thinking-toolkit/07-multivar-chain.md) |
| 08 | [多元积分变换](./thinking-toolkit/08-multivar-integration.md) |
| 09 | [ODE 类型识别](./thinking-toolkit/09-ode-classification.md) |
| 10 | [凸性、单调与极值](./thinking-toolkit/10-convexity-extrema.md) |
| 11 | [微积分中的不等式](./thinking-toolkit/11-inequality-techniques.md) |
| 12 | [微积分中的 AI 思维](./thinking-toolkit/12-calculus-for-ai.md) |

### 新增模板 D 包装

每个**核心套路型章节**（⭐ 标记，共 15 章）在原正文之外，加上：
- 一例速记：核心公式 + 最常用变形
- 思维路径还原：作者第 1 秒怎么想的内心独白
- 思考路标 ≥8 条：条件反射（看到 X 立刻想 Y）
- 易错点 5 条

⭐ 章节包括：函数极限 / 求导法则 / 导数应用 / Taylor 级数 / 积分技巧 / 数项级数 / 幂级数 / 偏导数 / 重积分 / 向量微积分 / 一阶 ODE / 二阶 ODE / 凸优化 / 矩阵微积分。

### 新增题库（240 题，C/D/E 三档）

`appendix/` 目录扩展了三档题库：
- C 基础 80 题
- D 中档 100 题
- E 提升 60 题（含考研真题级 + AI 应用题）
- F 详解（按主题分 4 文件）

### figures（~80 张）

`figures/` 目录用 TikZ + Asymptote 渲染了 ~80 张关键图（极限示意、Taylor 截断、多元 3D、梯度下降轨迹等）。

---

## 配套教程

- [`chuzhong_algebra/`](../chuzhong_algebra/) — 中考代数
- [`chuzhong_geometry/`](../chuzhong_geometry/) — 中考几何
- [`gaozhong_math/algebra/`](../gaozhong_math/algebra/) — 高中代数（含导数 / 数列 / 概率统计）
- [`gaozhong_math/geometry/`](../gaozhong_math/geometry/) — 高中几何（含解析 / 立体 / 空间向量）

四套教程构成"初中 → 高中 → 微积分（含 AI）"完整数学体系。
```

- [ ] **Step 5: 扩展 00-preface.md（在末尾追加段落）**

不动现有 preface 内容。在文件末尾追加：

```markdown
---

## 关于"模板 D 思维训练"（2026-05-14 改造后新增）

本教程已经按 `gaozhong_math/algebra/` 的成功模式做了一次"包装式升级"——在原有严格推导 + 深度学习应用基础上，**新增**了三类训练价值：

### 1. 一例速记（⭐ 章节）

每章开头给出 3-5 行"核心公式 + 最常用变形 + 题型识别"。
这是给"懒人 / 备考者"的速查；只看这一段也能解 70% 的中档题。

### 2. 思维路径还原（⭐ 章节）

用 15-20 行 `>` 引用块完整还原**作者第 1 秒到最后 1 秒的内心独白**：
- 看到题目第一反应是什么？
- 为什么选这个方法（而不是其它）？
- 哪一步是"关键转折"？
- 算完为什么验证 / 反思？

这是给"想理解高手大脑"的读者：通过"模拟一个专家的思维路径"来训练自己。

### 3. 思考路标（≥8 条）

每章末尾的"看到 X 立刻想 Y"反射列表，让你**形成条件反射**：
- 看到 $\lim_{x\to 0} \dfrac{\sin x}{x}$ → 想 "两个重要极限第 1 个"
- 看到 $\int x e^x \, dx$ → 想 "反 LIATE → 分部积分，$x$ 是多项式型，求导"
- 看到二阶常系数齐次 ODE → 想 "特征方程"
- ...

### 配合 thinking-toolkit 使用

`thinking-toolkit/` 12 篇是**主动学习指南**：建议先读完 toolkit 01-02（极限思维 + 等价无穷小），再开始正文 Part 2。学习一段时间后再读 toolkit 中对应主题的篇章，效果最佳。

### 与初高中教程的衔接

本教程的 Part 1 预备知识完整覆盖了 `gaozhong_math/algebra/` 中的对数 / 指数 / 三角内容。如果你刚结束高三，可以**跳过 Part 1**直接从 Part 2 开始。

---
```

- [ ] **Step 6: 三次串行 commit**

```bash
cd /Users/yangyang/ai_projs/math
git add calculus-tutorial/figures
git commit -m "feat(calculus): set up figure rendering infrastructure"

git add calculus-tutorial/README.md
git commit -m "docs(calculus): extend README with toolkit and appendix navigation"

git add calculus-tutorial/00-preface.md
git commit -m "docs(calculus): extend preface with template D thinking-training section"
```

---

## Task 2-13: Phase 2（12 篇 toolkit，3 sonnet subagent 并行）

每篇 3000-4500 字（比初高中略长）。结构（与初高中 toolkit 一致）：

1. 引言 / 概念
2. N 类情形或方法
3. 演示题（用 `>` 引用，详细内心独白）
4. 思考路标（≥6 条）
5. 典型应用 3 例
6. 自测题 4-5 题

### Toolkit 各篇内容要点

**01-epsilon-language.md** —— 极限的 $\varepsilon$ 语言  
$\varepsilon$-$N$ 数列极限 + $\varepsilon$-$\delta$ 函数极限的双语言。三步证明范式（设 $\varepsilon$ → 找 $N$ 或 $\delta$ → 验证）。反向用法（已知极限求参数范围）。3 类典型证明（求多项式极限 / 含根号 / 数列单调有界）。

**02-equivalent-infinitesimals.md** —— 等价无穷小与小 $o$  
常见等价表（$\sin x \sim x$，$\ln(1+x) \sim x$，$e^x - 1 \sim x$，$(1+x)^a - 1 \sim ax$ 等）。等价替换的**正确用法**（**乘除可换、加减不能直接换**）。小 $o(x^n)$ 记号 + Taylor 展开衔接。错误用法警戒：替换后再做加减是常见陷阱。

**03-differentiation-rules.md** —— 求导套路系统化  
6 大求导规则的"机械化"流程：
- 加减 / 数乘 / 乘积 / 商 / 链式 / 反函数
- 隐函数求导（两边对 $x$ 求导 + 解 $y'$）
- 参数式 / 极坐标式
- 高阶导数 + Leibniz 公式
表格速查：见 $f \cdot g$ → $f'g + fg'$；见 $\dfrac{f}{g}$ → $\dfrac{f'g-fg'}{g^2}$；见复合 → 链式。

**04-integration-techniques.md** —— 积分技巧"反 LIATE"  
分部积分的口诀 LIATE（Logarithmic / Inverse trig / Algebraic / Trig / Exponential）—— **反 LIATE 是错的常见说法**，应纠正：实际上 **LIATE 顺序优先级越高，越优先作为 $u$**（求导后简化）。换元 5 大类型：① 简单换元 $u = g(x)$ ② 三角换元 $\sqrt{a^2-x^2}, \sqrt{a^2+x^2}, \sqrt{x^2-a^2}$ ③ 倒代换 $x = 1/t$ ④ 万能代换 ⑤ 部分分式（多项式有理函数）。

**05-series-convergence.md** —— 级数判敛流程图  
完整决策树：① 通项 $a_n$ 是否 $\to 0$？（否 → 发散）② 正项 vs 交错？③ 比值法 $\lim|a_{n+1}/a_n|$ ④ 根值法 $\lim\sqrt[n]{|a_n|}$ ⑤ 比较法（与 $1/n^p$、$1/n!$、$q^n$ 比）⑥ Leibniz（交错）⑦ 积分判别。每个判别法的适用场景 + 反例。

**06-taylor-and-error.md** —— Taylor 展开与误差  
6 大常见函数 Maclaurin 展开 ($e^x, \sin x, \cos x, \ln(1+x), (1+x)^a, \arctan x$) 完整表。截断后的 Peano 余项 $o(x^n)$ vs Lagrange 余项 $\dfrac{f^{(n+1)}(\xi)}{(n+1)!}x^{n+1}$。误差估计：用 Lagrange 余项的最大值上界。AI 应用：神经网络的局部线性化 / Newton 法迭代收敛速率分析。

**07-multivar-chain.md** —— 多元链式与梯度  
$z = f(u, v), u = u(x, y), v = v(x, y)$ 的链式法则（树形依赖图）。梯度 $\nabla f$ 的几何意义（最快上升方向）。方向导数 $D_{\vec{u}} f = \nabla f \cdot \vec{u}$。Jacobian 矩阵作为线性近似。AI 应用：反向传播是链式法则的递归应用。

**08-multivar-integration.md** —— 多元积分变换  
直角坐标 / 极坐标 / 球坐标的选择：① 圆形 / 圆环区域 → 极坐标 ② 球形区域 → 球坐标 ③ 立方体 → 直角。Jacobian 行列式表（极坐标 $r$，球坐标 $\rho^2\sin\varphi$，柱坐标 $r$）。二重积分换序的标准 3 步。

**09-ode-classification.md** —— ODE 类型识别  
一阶 ODE 5 类：① 可分离 $y' = f(x)g(y)$ ② 齐次 $y' = f(y/x)$ ③ 一阶线性 $y' + p(x)y = q(x)$（积分因子）④ Bernoulli $y' + py = qy^n$ ⑤ 全微分（恰当方程）。二阶常系数齐次 / 非齐次：特征方程 + 待定系数 / 常数变易。完整决策树。

**10-convexity-extrema.md** —— 凸性、单调与极值  
单变量：$f''(x) \geq 0$ ⇔ 凸（开口向上）。多变量：Hessian 半正定 ⇔ 凸。Jensen 不等式 $f(\mathbb{E}X) \leq \mathbb{E}f(X)$（凸函数）。极值充分条件：① 一阶必要 $\nabla f = 0$ ② 二阶充分（Hessian 正定 / 负定）③ KKT 入门。AI 应用：凸优化的全局最小性、SVM、神经网络的局部极小。

**11-inequality-techniques.md** —— 微积分中的不等式  
4 大技巧：① 单调性证不等式（构造 $h(x) = f(x) - g(x)$ + 导数）② 凸性 + Jensen ③ 切线放缩（$\ln(1+x) \leq x$，$e^x \geq 1+x$）④ 微积分式不等式（如 Cauchy-Schwarz 积分版）。Hardy / Hölder / Minkowski 简介。

**12-calculus-for-ai.md** —— 微积分中的 AI 思维  
梯度下降的微积分基础。反向传播 = 多元链式法则的递归。Hessian 与 Newton 法。KL 散度 $\text{KL}(p\|q) = \int p\ln(p/q)$ 与凸性。自动微分（forward / backward mode）。Lagrangian 与对偶。

### Subagent 分组

**Subagent A** 写 01, 02, 03, 04（极限 / 等价 / 求导 / 积分）  
**Subagent B** 写 05, 06, 07, 08（级数 / Taylor / 多元链 / 多元积分）  
**Subagent C** 写 09, 10, 11, 12（ODE / 凸性 / 不等式 / AI 微积分）

每篇独立 commit：

```bash
cd /Users/yangyang/ai_projs/math
git add calculus-tutorial/thinking-toolkit/<XX>-<name>.md
git commit -m "docs(calculus): add toolkit/<XX> (<topic>)"
```

---

## Task 14: CM0 收尾

- [ ] **Step 1: 检查无占位符**

```bash
cd /Users/yangyang/ai_projs/math
grep -rn 'TBD\|TODO\|XXX\|待补' calculus-tutorial/thinking-toolkit calculus-tutorial/README.md calculus-tutorial/00-preface.md
```

Expected: 无输出。

- [ ] **Step 2: 列出 toolkit 目录**

```bash
ls calculus-tutorial/thinking-toolkit/
```

Expected: 12 个 md 文件。

- [ ] **Step 3: 标记完成 + push**

```bash
cd /Users/yangyang/ai_projs/math
printf '\n---\n**CM0 完成于：2026-05-14**\n' >> docs/superpowers/plans/2026-05-14-calculus-CM0.md
git add docs/superpowers/plans/2026-05-14-calculus-CM0.md
git commit -m "docs(calculus): mark CM0 milestone complete"
git push origin master
```

---

## 报告要求

每个 subagent 完成后报告：
- 各 commit SHA
- 各文件字数
- 任何问题

---
**CM0 完成于：2026-05-14**
