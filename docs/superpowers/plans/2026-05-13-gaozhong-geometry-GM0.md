# 高中几何教程 GM0 里程碑实施计划

> **REQUIRED SUB-SKILL**: superpowers:subagent-driven-development + rendering-math-figures

**Goal:** 完成 `gaozhong_math/geometry/` 的方法论框架——README、序言、12 篇几何思维工具箱、figures 基础设施。

**Architecture:** 纯 markdown 文档项目。每文件独立提交。figures 基础设施从 `~/.claude/skills/rendering-math-figures/templates/` 拷贝。toolkit 12 篇全部重写为**几何思维**（与代数 toolkit 不同）。

**Spec 参考:** `docs/superpowers/specs/2026-05-13-gaozhong-geometry-design.md`
**前序：** 无（这是 GM0）。已有 `gaozhong_math/algebra/` 和 `chuzhong_geometry/` 作为风格参照。

---

## 文件结构（GM0 产出 15 个文件 + figures 目录）

```
gaozhong_math/geometry/
├── README.md
├── 00-preface.md
├── thinking-toolkit/
│   ├── 01-four-swords.md              # 综合 / 解析 / 向量 / 三角四把剑
│   ├── 02-auxiliary-lines.md          # 辅助线思维
│   ├── 03-number-shape-deep-v2.md     # 数形结合（几何深化）
│   ├── 04-parametrization.md          # 参数化思想
│   ├── 05-invariants-and-symmetry.md  # 不变量与对称
│   ├── 06-position-classification.md  # 位置关系的分类讨论
│   ├── 07-geometric-extrema.md        # 几何最值 5 种方法
│   ├── 08-vector-two-faces.md         # 向量的两副面孔
│   ├── 09-spatial-imagination.md      # 空间几何想象力
│   ├── 10-line-conic-synthesis.md     # 直线圆锥曲线综合
│   ├── 11-vector-in-solid.md          # 立体几何向量法
│   └── 12-checklist-before-solving.md # 几何开题清单
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

每篇 toolkit 字数 **2500-3500**（与代数一致），用 `>` 引用块写思维路径，含思考路标和自测题。

---

## Task 1: Phase 1（基础设施 + README + preface）

**Files:**
- Create: `gaozhong_math/geometry/figures/render.sh`（从代数复用）
- Create: `gaozhong_math/geometry/figures/STYLE.md`（从代数复用）
- Create: `gaozhong_math/geometry/figures/src/tikz/.gitkeep`
- Create: `gaozhong_math/geometry/figures/src/asy/.gitkeep`
- Create: `gaozhong_math/geometry/figures/svg/.gitkeep`
- Create: `gaozhong_math/geometry/README.md`
- Create: `gaozhong_math/geometry/00-preface.md`

执行步骤（由 1 个 subagent 完成或 main agent 直接做）：

- [ ] **Step 1: 创建目录骨架**

```bash
cd /Users/yangyang/ai_projs/math
mkdir -p gaozhong_math/geometry/figures/{src/tikz,src/asy,svg} gaozhong_math/geometry/thinking-toolkit
```

- [ ] **Step 2: 拷贝 figures 基础设施（render.sh + STYLE.md）**

```bash
cd /Users/yangyang/ai_projs/math
cp gaozhong_math/algebra/figures/render.sh gaozhong_math/geometry/figures/render.sh
cp gaozhong_math/algebra/figures/STYLE.md gaozhong_math/geometry/figures/STYLE.md
chmod +x gaozhong_math/geometry/figures/render.sh
# 在 render.sh 和 STYLE.md 中把 "gaozhong/algebra" 替换为 "gaozhong/geometry"
sed -i '' 's|gaozhong_math/algebra|gaozhong_math/geometry|g' gaozhong_math/geometry/figures/render.sh
sed -i '' 's|gaozhong_math/algebra|gaozhong_math/geometry|g' gaozhong_math/geometry/figures/STYLE.md
# 添加 .gitkeep 让空目录入版本控制
touch gaozhong_math/geometry/figures/src/tikz/.gitkeep gaozhong_math/geometry/figures/src/asy/.gitkeep gaozhong_math/geometry/figures/svg/.gitkeep
```

- [ ] **Step 3: 验证 render.sh 可执行（写一个 test.tex 渲染）**

```bash
cd /Users/yangyang/ai_projs/math/gaozhong_math/geometry/figures
cat > src/tikz/test.tex <<'EOF'
\documentclass[tikz,border=4pt]{standalone}
\usepackage{ctex}
\begin{document}
\begin{tikzpicture}
\draw[->] (0,0) -- (3,0) node[right] {$x$};
\draw[->] (0,0) -- (0,3) node[above] {$y$};
\draw[blue, thick] (0,0) -- (2,2);
\node[above right] at (2,2) {$\vec{v}$};
\end{tikzpicture}
\end{document}
EOF
./render.sh src/tikz/test.tex
ls svg/test.svg && rm src/tikz/test.tex svg/test.svg
```

Expected: 输出 `✓ svg/test.svg`，确认工具链可用。

- [ ] **Step 4: 写 README.md（1800-2500 字）**

内容要点：
- 顶部 banner："高中几何教程——人教 A 版 2019 新课标"
- 配套：`gaozhong_math/algebra/` 代数（兄弟教程）、`chuzhong_geometry/` 初中几何（衔接）
- 这本书写给谁（高一下新生 / 备战高考 / 自学高中几何的成年读者）
- 与普通教辅的不同（强调几何思维：四把剑 / 辅助线 / 参数化 / 向量法）
- 目录：完整列出 10 个 part 的章节，每章后标 ⭐ 表示模板 D，附年级标签
- 怎么用这本书（高一下 → 高二上 → 高三复习路线 + thinking-toolkit 先读）
- 章节结构（模板 C/D 简述）
- 关于练习题（240 题三档 + 详解）
- 关于图（TikZ + Asymptote，几何强配图）

- [ ] **Step 5: 写 00-preface.md（2000-2800 字）**

内容要点：
1. **什么是"高中几何思维"**：相比初中升级了三层
   - 从"作图证明"升到"四种方法切换"（综合 / 解析 / 向量 / 三角）
   - 从"具体图形"升到"抽象关系"（圆锥曲线 / 空间向量）
   - 从"二维直观"升到"三维想象 + 代数化"
2. **为什么很多人高中几何难**（4 个误区）
   - 只会综合法，不会向量法
   - 圆锥曲线题"设而不求"不熟
   - 立体几何空间想象薄弱
   - 解析几何运算量大不敢碰
3. **这本书的解法**（模板 D 思维路径还原 + toolkit 12 篇 + 套路图集）
4. **怎么读这本书最有效**（先读 toolkit 4 篇核心：四把剑 / 辅助线 / 参数化 / 向量两面）
5. **结尾**：高考几何不靠刷图量，靠"看到题就切对方法"

- [ ] **Step 6: 串行三次 commit**

```bash
cd /Users/yangyang/ai_projs/math
git add gaozhong_math/geometry/figures
git commit -m "feat(gaozhong/geometry): set up figure rendering infrastructure"

git add gaozhong_math/geometry/README.md
git commit -m "docs(gaozhong/geometry): add tutorial README"

git add gaozhong_math/geometry/00-preface.md
git commit -m "docs(gaozhong/geometry): add preface on high school geometric thinking"
```

---

## Task 2-13: Phase 2（12 篇 toolkit，3 subagent 并行）

每篇 2500-3500 字。结构（与代数一致）：

1. 引言 / 概念
2. N 类情形或方法
3. 演示题（用 `>` 引用，详细内心独白）
4. 思考路标（≥6 条）
5. 典型应用 3 例
6. 自测题 4-5 题

### Toolkit 各篇内容要点

**01-four-swords.md** —— 几何题的四把剑  
高中几何标志性思想：综合法（纯几何推理）、解析法（建系坐标计算）、向量法（向量运算）、三角法（三角恒等变换）。每种方法的适用场景、优劣对比、何时切换。3 类典型题展示"为什么这道题用 X 法最佳"。

**02-auxiliary-lines.md** —— 辅助线思维  
几何老话题在高中的延伸：延长 / 截取 / 平移 / 旋转 / 中位线 / 对称变换。10 种典型辅助线及对应触发条件表。3 道经典演示题（一三角形 / 一圆 / 一四边形）。

**03-number-shape-deep-v2.md** —— 数形结合（几何深化）  
比初中数形结合更深：已知关系式 → 找几何意义；已知几何 → 找代数关系。例：$|\vec{a} + \vec{b}|^2 = |\vec{a}|^2 + |\vec{b}|^2$ 几何上 = 直角；$(x-1)^2 + (y-2)^2 = 4$ → 圆。

**04-parametrization.md** —— 参数化思想  
解析几何 + 立体几何核心套路：把几何对象转代数运算。① 直线参数化 $(x_0 + at, y_0 + bt)$ ② 圆参数化 $(r\cos\theta, r\sin\theta)$ ③ 椭圆参数化 $(a\cos\theta, b\sin\theta)$ ④ 平移参数化 ⑤ 点选 $(t, f(t))$。

**05-invariants-and-symmetry.md** —— 不变量与对称  
对称 / 旋转 / 平移下的几何不变量：长度、夹角、面积、向量内积、对称中心、对称轴。识别图形的对称性可大幅简化计算。

**06-position-classification.md** —— 位置关系的分类讨论  
直线 vs 直线 / 直线 vs 圆 / 圆 vs 圆 / 直线 vs 圆锥曲线 / 平面 vs 平面 等位置关系的判别 + 分类讨论（含 $\Delta$ / 距离比较 / 系数关系）。

**07-geometric-extrema.md** —— 几何最值 5 种方法  
① 参数化 + 求导 ② 数形结合（切线 / 圆心距）③ 反射法（光路最短）④ 极化（向量恒等式）⑤ 函数最值（转代数）。5 类经典最值题对比。

**08-vector-two-faces.md** —— 向量的两副面孔  
向量既是几何对象（有大小有方向、几何意义清晰）又是代数对象（可加减点乘、坐标运算便捷）。何时用几何意义、何时建系算代数。3 道演示题。

**09-spatial-imagination.md** —— 空间几何想象力  
三视图 → 三维还原；截面（如正方体截面有 8 种）；折叠展开（如长方体表面最短路径）；空间位置关系的直观判断（异面、垂直、平行）。

**10-line-conic-synthesis.md** —— 直线圆锥曲线综合  
高考压轴标配：① 韦达定理（$x_1 + x_2, x_1 x_2$ 不求根直接用）② 设而不求 ③ 中点弦（点差法）④ 弦长公式 ⑤ 含参讨论（直线斜率不存在的情形）。

**11-vector-in-solid.md** —— 立体几何向量法  
建系标准流程：①找正交三边 ②设单位长 ③求关键点坐标 ④用公式：线线角 = $|\cos\langle\vec{a},\vec{b}\rangle|$；线面角 = $|\sin\theta| = |\cos\langle\vec{l}, \vec{n}\rangle|$；二面角 = $\cos\langle\vec{n_1}, \vec{n_2}\rangle$ 与图形位置组合判正负。

**12-checklist-before-solving.md** —— 几何开题清单  
读题 → 识图形 → 选方法（4 把剑）→ 设变量 → 估难度 → 列方程 → 验范围（与代数 toolkit/12 平行，强调几何独特步骤"识图 + 选法"）。8 问清单。

### Subagent 分组

**Subagent A** 写 01, 02, 03, 04（四把剑 / 辅助线 / 数形 / 参数化）  
**Subagent B** 写 05, 06, 07, 08（不变对称 / 位置分类 / 几何最值 / 向量两面）  
**Subagent C** 写 09, 10, 11, 12（空间想象 / 直线圆锥 / 立体向量 / 清单）

每个 subagent 写 4 篇，每篇独立 commit：

```bash
cd /Users/yangyang/ai_projs/math
git add gaozhong_math/geometry/thinking-toolkit/<XX>-<name>.md
git commit -m "docs(gaozhong/geometry): add toolkit/<XX> (<topic>)"
```

---

## Task 14: GM0 收尾

- [ ] **Step 1: 检查无占位符**

```bash
cd /Users/yangyang/ai_projs/math
grep -rn 'TBD\|TODO\|XXX\|待补' gaozhong_math/geometry
```

Expected: 无输出（无占位符）。

- [ ] **Step 2: 列出 thinking-toolkit 目录**

```bash
ls gaozhong_math/geometry/thinking-toolkit
```

Expected: 12 个 md 文件。

- [ ] **Step 3: 在计划文档末尾追加完成标记并 commit / push**

```bash
cd /Users/yangyang/ai_projs/math
printf '\n---\n**GM0 完成于：2026-05-13**\n' >> docs/superpowers/plans/2026-05-13-gaozhong-geometry-GM0.md
git add docs/superpowers/plans/2026-05-13-gaozhong-geometry-GM0.md
git commit -m "docs(gaozhong/geometry): mark GM0 milestone complete"
git push origin master
```

---

## 报告要求

每个 subagent 完成后报告：
- 各 commit SHA
- 各文件字数
- 任何需要后续修正的问题（如术语不一致、缺失定义等）
