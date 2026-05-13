# 高中代数教程 GM0 里程碑实施计划

> **REQUIRED SUB-SKILL**: superpowers:subagent-driven-development + rendering-math-figures

**Goal:** 完成 `gaozhong_math/algebra/` 的方法论框架——README、序言、12 篇思维工具箱、figures 基础设施。

**Architecture:** 纯 markdown 文档项目。每文件独立提交。figures 基础设施从 `~/.claude/skills/rendering-math-figures/templates/` 拷贝。

**Spec 参考:** `docs/superpowers/specs/2026-05-13-gaozhong-algebra-design.md`
**前序：** 无（这是 GM0）。但已有 `chuzhong_algebra/` 作为风格参照。

---

## 文件结构（GM0 产出 15 个文件 + figures 目录）

```
gaozhong_math/algebra/
├── README.md
├── 00-preface.md
├── thinking-toolkit/
│   ├── 01-structure-recognition.md
│   ├── 02-substitution-and-simplification.md
│   ├── 03-construction.md
│   ├── 04-number-shape-deep.md
│   ├── 05-parameter-strategy.md
│   ├── 06-abstract-function.md
│   ├── 07-composite-function.md
│   ├── 08-monotonicity-extrema.md
│   ├── 09-symmetry-and-periodicity.md
│   ├── 10-limit-and-bounds.md
│   ├── 11-classification-and-case-split.md
│   └── 12-checklist-before-solving.md
└── figures/
    ├── render.sh
    ├── STYLE.md
    └── src/{tikz,asy}/ + svg/
```

---

## 调度策略

**Phase 1（基础设施 + 元文件，串行 1 个 subagent）**：
- 拷贝 figures 基础设施 + README + 00-preface

**Phase 2（toolkit 12 篇，并行 3 subagent）**：
- A 写 toolkit 01-04
- B 写 toolkit 05-08
- C 写 toolkit 09-12

每篇 toolkit 字数 2500-3500（与 chuzhong 一致），用 `>` 引用块写思维路径，含思考路标和自测题。

---

## Task 1: Phase 1（基础设施 + README + preface）

执行步骤：
1. `mkdir -p gaozhong_math/algebra/figures/{src/tikz,src/asy,svg} gaozhong_math/algebra/thinking-toolkit`
2. 拷贝 `~/.claude/skills/rendering-math-figures/templates/render.sh` 与 `STYLE.md` 到 `gaozhong_math/algebra/figures/`，并将注释里的 `<project>` 改为 `gaozhong_math/algebra`
3. 写 `README.md` 1800-2500 字
4. 写 `00-preface.md` 2000-2800 字

**README 内容要点**：
- 顶部 banner："高中代数教程——人教 A 版 2019 新课标"
- 配套：`chuzhong_algebra/` 中考代数（衔接）、未来的 `gaozhong_math/geometry/`（向量/解析/立体几何）
- 这本书写给谁（高一新生 / 备战高考 / 自学高中代数的成年读者）
- 与普通教辅的不同（强调高级思维：构造法、抽象函数、含参讨论、数形结合深化）
- 目录：完整列出 13 个 part 的章节，每章后标 ⭐ 表示模板 D，附年级标签
- 怎么用这本书（按高一 → 高二 → 高三复习路线 + 思维工具箱先读）
- 章节结构（模板 C/D 简述）
- 关于练习题（240 题三档 + 详解）
- 关于图（TikZ + Asymptote）

**00-preface 内容要点**：
1. **什么是"高中代数思维"**：相比初中升级了三层
   - 从"算"升到"推"：每个结论要严格证明
   - 从"具体"升到"抽象"：抽象函数、参数族、含参讨论
   - 从"形数分离"升到"形数互译深化"：导数与几何意义、函数图象与方程根
2. **为什么很多人高中代数难**（4 个误区）
3. **这本书的解法**（模板 D 思维路径还原 + toolkit 12 篇 + 套路图集）
4. **怎么读这本书最有效**（先读 toolkit 4 篇核心：结构识别 / 构造法 / 数形结合 / 分类讨论）
5. **结尾**：高考代数不靠刷题量，靠"看到题就识别套路"

Commit:
- `feat(gaozhong/algebra): set up figure rendering infrastructure`
- `docs(gaozhong/algebra): add tutorial README`
- `docs(gaozhong/algebra): add preface on high school algebraic thinking`

---

## Task 2-13: Phase 2（12 篇 toolkit，3 subagent 并行）

每篇 2500-3500 字。结构（与 chuzhong 一致）：
1. 引言 / 概念
2. N 类情形或方法
3. 演示题（用 `>` 引用，详细内心独白）
4. 思考路标（≥6 条）
5. 典型应用 3 例
6. 自测题 4-5 题

### Toolkit 各篇内容要点

**01-structure-recognition.md** —— 高级结构识别  
扩充初中"结构识别"到高中：见 $\frac{a}{b}+\frac{b}{a}$ → 基本不等式；见 $\sin^2+\cos^2$ → 同角恒等；见 $a^n+b^n$ → 二项式或贝努利；见 $f(x_1) - f(x_2)$ → 中值 / 导数符号。表格列 10 种典型结构。

**02-substitution-and-simplification.md** —— 换元与化简  
比初中深：三角换元（$x = \sin\theta$）、对数换元、复合函数内层换元、含参 $t = $ 表达式整体换元。

**03-construction.md** ⭐ NEW（高中独有）—— 构造法  
高中代数标志性思想。构造函数证不等式（如证 $\ln(1+x) < x$ 用 $f(x) = x - \ln(1+x)$）、构造数列、构造对称式、构造辅助函数。3 类经典构造演示。

**04-number-shape-deep.md** —— 数形结合（高中深化）  
比初中加深：函数图象 + 导数几何意义（切线斜率）、解的几何意义（图象交点）、不等式恒成立 = 图象上方关系、参数最值 = 图象动起来。

**05-parameter-strategy.md** —— 含参讨论与参数分离  
高中含参讨论的标准框架：①分类讨论参数范围 ②参数分离 $a > f(x)$ → $a > \max f$ ③主参反换（把 $x$ 视为已知、$a$ 视为变量）。

**06-abstract-function.md** ⭐ NEW —— 抽象函数  
不给具体解析式，只给函数方程 $f(x+y) = f(x) + f(y)$ 或 $f(xy) = f(x)f(y)$，推导性质。这是高考送分（也送命）专题。

**07-composite-function.md** —— 复合函数  
$f \circ g$ 单调性"同增异减"、定义域取交集、值域用内函数像、含参复合 = 双重含参。

**08-monotonicity-extrema.md** —— 单调性与极值思想  
高中所有函数题的核心。从初等定义（$f(x_1) < f(x_2)$）到导数判别，最后用单调性证不等式 / 解方程。

**09-symmetry-and-periodicity.md** —— 对称性与周期性  
$f(a+x) = f(a-x)$ → 对称轴 $x=a$；$f(x+T) = f(x)$ → 周期 $T$。组合规律：两条对称轴 → 周期；对称中心 + 对称轴 → 周期 = 4×距离。

**10-limit-and-bounds.md** —— 极限与放缩  
高中"极限"非严格定义，但有大量放缩：$\sin x < x$（$x>0$）、$\ln(1+x) < x$、$e^x > 1+x$、$\frac{1}{n(n+1)} < \frac{1}{n^2}$ 等经典放缩。

**11-classification-and-case-split.md** —— 分类讨论框架  
何时必须分类（含参 / 绝对值 / 奇偶 / 单调区间端点位置 / 函数零点位置）+ 分类的 5 条规则（不重不漏）+ 标准书写格式。

**12-checklist-before-solving.md** —— 开题清单（高中版）  
读题 → 识结构 → 选方法 → 估难度 → 列方程或函数 → 验证；与 chuzhong 的 7 问相比深化为 8 问。

### Subagent 分组

**Subagent A** 写 01, 02, 03, 04（结构 / 换元 / 构造 / 数形）  
**Subagent B** 写 05, 06, 07, 08（参数 / 抽象 / 复合 / 单调极值）  
**Subagent C** 写 09, 10, 11, 12（对称周期 / 放缩 / 分类 / 清单）

每个 subagent 写 4 篇，每篇独立 commit。

---

## Task 14: GM0 收尾

```bash
cd /Users/yangyang/ai_projs/math
grep -rn 'TBD\|TODO\|XXX\|待补' gaozhong_math/algebra
ls gaozhong_math/algebra/thinking-toolkit
printf '\n---\n**GM0 完成于：YYYY-MM-DD**\n' >> docs/superpowers/plans/2026-05-13-gaozhong-algebra-GM0.md
git add docs/superpowers/plans/2026-05-13-gaozhong-algebra-GM0.md
git commit -m "docs(gaozhong/algebra): mark GM0 milestone complete"
git push origin master
```

---

## 报告要求

每个 subagent 完成后报告：
- 各 commit SHA
- 各文件字数
- 任何需要后续修正的问题（如术语不一致、缺失定义等）
