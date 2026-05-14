# 概率论教程 PM0 里程碑实施计划

> **REQUIRED SUB-SKILL**: superpowers:subagent-driven-development + rendering-math-figures

**Goal:** 完成 `probability-statistics-tutorial/` 改造的方法论框架——README/preface 扩展、12 篇 thinking-toolkit、figures 基础设施。

**Architecture:** 不动现有 24 章正文。新增 thinking-toolkit/、figures/；扩展 README + preface（追加，**不重写**）。

**Spec:** `docs/superpowers/specs/2026-05-14-probability-refactor-design.md`

---

## 文件结构（PM0 产出 15 文件 + figures 目录）

```
probability-statistics-tutorial/
├── README.md                       # 扩展
├── 00-preface.md                   # 扩展
├── thinking-toolkit/               # 新增
│   ├── 01-probability-axioms.md
│   ├── 02-conditional-bayes.md
│   ├── 03-expectation-variance-moments.md
│   ├── 04-distributions-cheatsheet.md
│   ├── 05-multivariate-covariance.md
│   ├── 06-limit-theorems.md
│   ├── 07-estimator-evaluation.md
│   ├── 08-mle-patterns.md
│   ├── 09-hypothesis-testing-7steps.md
│   ├── 10-information-theory.md
│   ├── 11-monte-carlo-mcmc.md
│   └── 12-probability-for-ai.md
└── figures/
    ├── render.sh
    ├── STYLE.md
    └── src/{tikz,asy}/ + svg/
```

---

## 调度策略

**Phase 1（main agent 直接做）**：
- figures 基建 + README + preface 扩展

**Phase 2（3 sonnet subagent 并行）**：
- A: toolkit 01-04（公理 / 贝叶斯 / 期望矩 / 分布速查）
- B: toolkit 05-08（多元 / 极限 / 估计评价 / MLE）
- C: toolkit 09-12（假设检验 / 信息论 / MCMC / AI 思维）

每篇 3000-4500 字。结构：引言 / N 类方法 / 演示题（`>` 内心独白）/ 思考路标 ≥6 / 应用 3 例 / 自测 4-5 题。

---

## Task 1: Phase 1（figures 基建 + README + preface 扩展）

- [ ] **Step 1: 创建目录骨架**

```bash
cd /Users/yangyang/ai_projs/math
mkdir -p probability-statistics-tutorial/figures/{src/tikz,src/asy,svg} probability-statistics-tutorial/thinking-toolkit
```

- [ ] **Step 2: 拷贝 figures 基建（复用 calculus 的）**

```bash
cd /Users/yangyang/ai_projs/math
cp calculus-tutorial/figures/render.sh probability-statistics-tutorial/figures/render.sh
cp calculus-tutorial/figures/STYLE.md probability-statistics-tutorial/figures/STYLE.md
chmod +x probability-statistics-tutorial/figures/render.sh
sed -i '' 's|calculus-tutorial|probability-statistics-tutorial|g; s|calculus|probability|g' probability-statistics-tutorial/figures/render.sh probability-statistics-tutorial/figures/STYLE.md
touch probability-statistics-tutorial/figures/src/tikz/.gitkeep probability-statistics-tutorial/figures/src/asy/.gitkeep probability-statistics-tutorial/figures/svg/.gitkeep
```

- [ ] **Step 3: 验证 render.sh**

```bash
cd /Users/yangyang/ai_projs/math/probability-statistics-tutorial/figures
cat > src/tikz/test-pdf.tex <<'EOF'
\documentclass[tikz,border=4pt]{standalone}
\usepackage{ctex}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\begin{document}
\begin{tikzpicture}
\begin{axis}[axis lines=middle, width=8cm, height=5cm, samples=200]
\addplot[blue, thick, domain=-4:4] {1/sqrt(2*pi)*exp(-x^2/2)};
\end{axis}
\end{tikzpicture}
\end{document}
EOF
./render.sh src/tikz/test-pdf.tex
ls svg/test-pdf.svg && rm src/tikz/test-pdf.tex svg/test-pdf.svg
```

Expected: `✓ svg/test-pdf.svg`。

- [ ] **Step 4: 扩展 README.md（追加段落）**

不动现有内容，末尾追加：
- 改造升级说明
- thinking-toolkit 12 篇导航表
- 模板 D 包装说明
- 题库 240 题导航
- 配套教程链接（calculus / gaozhong / chuzhong）

- [ ] **Step 5: 扩展 00-preface.md（追加段落）**

不动现有内容，末尾追加"模板 D 思维训练"段：
- 一例速记（⭐ 章节）
- 思维路径还原
- 思考路标 ≥8 条
- 配合 thinking-toolkit 使用建议
- 与 calculus / ML 教程的衔接

- [ ] **Step 6: 3 次串行 commit**

```bash
cd /Users/yangyang/ai_projs/math
git add probability-statistics-tutorial/figures
git commit -m "feat(probability): set up figure rendering infrastructure"

git add probability-statistics-tutorial/README.md
git commit -m "docs(probability): extend README with toolkit and appendix navigation"

git add probability-statistics-tutorial/00-preface.md
git commit -m "docs(probability): extend preface with template D thinking-training"
```

---

## Task 2-13: Phase 2（12 篇 toolkit，3 sonnet subagent 并行）

每篇 3000-4500 字。结构与 calculus toolkit 一致。

### Toolkit 各篇内容要点

**01-probability-axioms.md** —— 概率公理与样本空间  
Kolmogorov 三公理（非负 / 归一 / 可加）；$\sigma$-代数概念入门；事件运算（$\cup, \cap, \overline{A}$）；古典 / 几何 / 频率 / 主观 4 种概型对比；演示题用古典概型 + 几何概型双解一道题（如蒲丰投针）。

**02-conditional-bayes.md** —— 条件概率与贝叶斯思维  
$P(B|A) = P(AB)/P(A)$；链式公式 $P(ABC) = P(A)P(B|A)P(C|AB)$；全概率 + 贝叶斯；独立 vs 互斥辨析（最常混淆）；演示题：医学检验贝叶斯反差经典例（先验 1% + 灵敏度 95% + 特异度 90% → 阳性后患病概率 ~ 9%）。

**03-expectation-variance-moments.md** —— 期望 / 方差 / 矩  
$E(X) = \sum x_i p_i$ 或 $\int x f(x)\,dx$；线性性 $E(aX+b) = aE(X)+b$；$\text{Var} = E(X^2) - (EX)^2$；协方差 $\text{Cov}(X,Y) = E[(X-EX)(Y-EY)]$；相关系数 $\rho \in [-1, 1]$；矩母函数 $M(t) = E(e^{tX})$；演示题：求二项分布 $X \sim B(n, p)$ 的 $E, \text{Var}$（两种方法对比：定义 + MGF）。

**04-distributions-cheatsheet.md** —— 常见分布速查表  
**离散**：Bernoulli($p$) / 二项 $B(n,p)$ / Poisson($\lambda$) / 几何 / 超几何 / 负二项 — 表格列 PMF / $EX$ / $\text{Var}$ / 典型场景；**连续**：均匀 $U(a,b)$ / 指数 $\text{Exp}(\lambda)$ / 正态 $N(\mu, \sigma^2)$ / Gamma / Beta / $\chi^2$ / $t$ / $F$ — 表格 + 关系图（如 $\chi^2_n = \sum Z_i^2$）；演示题：识别"等公交时间" → 指数分布。

**05-multivariate-covariance.md** —— 多元分布与协方差  
联合 PMF / PDF / CDF；边缘分布（积分掉一个）；条件分布；独立判别 $f(x,y) = f_X(x) f_Y(y)$；$\text{Cov}$ 矩阵；多元正态 $N(\boldsymbol\mu, \boldsymbol\Sigma)$；演示题：求二元正态条件分布是一元正态。

**06-limit-theorems.md** —— 极限定理（LLN + CLT）  
**LLN**：$\bar{X}_n \to \mu$（弱 / 强）；**CLT**：$\sqrt{n}(\bar{X}_n - \mu)/\sigma \to N(0, 1)$；**收敛模式 4 种**：a.s. / 概率 / $L^p$ / 分布（强弱关系图）；演示题：用 CLT 估算 $P(|\bar{X}_{100} - \mu| < 0.1)$（含查正态分布表）。

**07-estimator-evaluation.md** —— 估计量评价四标尺  
① 无偏 $E(\hat\theta) = \theta$ ② 一致 $\hat\theta_n \to \theta$（概率）③ 有效（方差小）④ Cramér-Rao 下界 $\text{Var}(\hat\theta) \geq 1/I(\theta)$（信息量 $I$）；演示题：比较样本均值 vs 样本中位数估计正态均值（前者更有效）。

**08-mle-patterns.md** —— MLE 求解套路  
**5 步流程**：① 写似然函数 $L(\theta) = \prod f(x_i; \theta)$ ② 取对数 $\ell = \log L$ ③ 求导 $\partial\ell/\partial\theta = 0$ ④ 解 $\hat\theta$ ⑤ 二阶导验极大；演示题：MLE 求正态分布 $(\mu, \sigma^2)$（双参数）。

**09-hypothesis-testing-7steps.md** —— 假设检验 7 步法  
① 提 $H_0/H_1$ ② 选统计量 ③ 定 $\alpha$（通常 0.05）④ 求拒绝域 ⑤ 算检验统计量 ⑥ 判断（拒绝 / 接受 $H_0$）⑦ 报告 $p$ 值；I/II 类错误 + 检验功效 $1-\beta$；演示题：单样本 $t$ 检验完整 7 步。

**10-information-theory.md** —— 信息论核心三件套  
**熵** $H(X) = -\sum p \log p$（衡量不确定性）；**KL 散度** $D(p\|q) = \sum p \log(p/q)$（非对称 + $\geq 0$）；**互信息** $I(X;Y) = H(X) - H(X|Y) = D(p_{XY}\|p_X p_Y)$；**交叉熵** $H(p,q) = -\sum p\log q = H(p) + D(p\|q)$；ML 应用：分类损失 = 交叉熵；演示题：证 KL ≥ 0（用 Jensen 不等式）。

**11-monte-carlo-mcmc.md** —— Monte Carlo + MCMC  
**MC**：用大量样本估计期望 $E_p[f(X)] \approx \frac{1}{n}\sum f(X_i)$；**拒绝采样** + **重要性采样**；**MCMC**：Metropolis-Hastings + Gibbs 采样；收敛性（平稳分布）；演示题：用 Metropolis 采样标准正态分布（手算 acceptance ratio）。

**12-probability-for-ai.md** —— 概率论中的 AI 思维  
**生成式 AI 全景**：VAE（变分推断）/ 扩散模型（Denoising score matching）/ Normalizing Flow（可逆变换 + Jacobian）/ GAN（min-max 博弈）/ 贝叶斯神经网络（参数后验）；**核心数学**：KL 散度 / ELBO / 重参数化技巧；演示题：从 VAE 的 ELBO 推导出"重建损失 + KL 正则"两项。

### Subagent 分组

**Subagent A** 写 01, 02, 03, 04（基础 4 篇）  
**Subagent B** 写 05, 06, 07, 08（中间 4 篇）  
**Subagent C** 写 09, 10, 11, 12（应用 4 篇）

每篇独立 commit：
```bash
cd /Users/yangyang/ai_projs/math
git add probability-statistics-tutorial/thinking-toolkit/<XX>-<name>.md
git commit -m "docs(probability): add toolkit/<XX> (<topic>)"
```

---

## Task 14: PM0 收尾

- [ ] **Step 1: 检查无占位符**
```bash
grep -rn 'TBD\|TODO\|XXX\|待补' probability-statistics-tutorial/thinking-toolkit probability-statistics-tutorial/README.md probability-statistics-tutorial/00-preface.md
```

Expected: 无输出。

- [ ] **Step 2: 列出 toolkit**
```bash
ls probability-statistics-tutorial/thinking-toolkit/
```

Expected: 12 个 md。

- [ ] **Step 3: 标记完成 + push**
```bash
printf '\n---\n**PM0 完成于：2026-05-14**\n' >> docs/superpowers/plans/2026-05-14-probability-PM0.md
git add docs/superpowers/plans/2026-05-14-probability-PM0.md
git commit -m "docs(probability): mark PM0 milestone complete"
git push origin master
```

---

## 报告要求

每个 subagent 完成后报告：commit SHA / 字数 / 问题。

---
**PM0 完成于：2026-05-14**
