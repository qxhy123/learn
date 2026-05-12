# 中考代数教程 AM0 里程碑实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development.

**Goal:** 完成 `chuzhong_algebra/` AM0 里程碑——总入口 README、序言、思维工具箱 10 篇，并拷贝/适配 figures 基础设施。代数教程方法论框架立起来。

**Architecture:** 纯 markdown 文档项目（与 `chuzhong_geometry/` 平行）。每个文件单独成节、独立提交。图基础设施从几何拷过来并加代数特有约定。

**Tech Stack:** Markdown + LaTeX（`$...$` / `$$...$$`）+ TikZ/Asymptote（少量，多数 toolkit 章节无图）。渲染脚本复用 `chuzhong_geometry/figures/render.sh`。

**Spec 参考：** `docs/superpowers/specs/2026-05-12-chuzhong-algebra-design.md`

**前序：** `chuzhong_geometry/` 已完成，作为风格参照。

---

## 文件结构

AM0 产出 13 个 markdown 文件 + 1 个 figures 子目录：

| # | 路径 | 职责 |
|---|---|---|
| - | `chuzhong_algebra/figures/` | 拷贝 render.sh + STYLE.md + 建 src/ svg/ 子目录 |
| 1 | `chuzhong_algebra/README.md` | 教程总入口、年级标签、目录、使用建议 |
| 2 | `chuzhong_algebra/00-preface.md` | 序言：什么是"代数思维" |
| 3 | `chuzhong_algebra/thinking-toolkit/01-when-to-substitute.md` | 何时换元 |
| 4 | `chuzhong_algebra/thinking-toolkit/02-when-to-complete-square.md` | 何时配方 |
| 5 | `chuzhong_algebra/thinking-toolkit/03-discriminant-and-vieta.md` | 判别式 + 韦达 |
| 6 | `chuzhong_algebra/thinking-toolkit/04-setting-up-equations.md` | 设元的艺术 |
| 7 | `chuzhong_algebra/thinking-toolkit/05-integral-thinking.md` | 整体思想 |
| 8 | `chuzhong_algebra/thinking-toolkit/06-counting-twice.md` | 算两次思想（NEW） |
| 9 | `chuzhong_algebra/thinking-toolkit/07-structure-recognition.md` | 结构识别（NEW） |
| 10 | `chuzhong_algebra/thinking-toolkit/08-parameter-strategy.md` | 参数与含参讨论（NEW） |
| 11 | `chuzhong_algebra/thinking-toolkit/09-number-shape-combination.md` | 数形结合（NEW） |
| 12 | `chuzhong_algebra/thinking-toolkit/10-checklist-before-solving.md` | 代数题开题清单 |
| 13 | (即上面 README，列表用) | |

---

## Task 1: figures 基础设施

**Files:**
- Create dirs: `chuzhong_algebra/figures/{src/tikz,src/asy,svg}`
- Copy: `chuzhong_geometry/figures/render.sh` → `chuzhong_algebra/figures/render.sh`
- Modify: 上述 render.sh 第 5 行注释里 `chuzhong_geometry/figures` → `chuzhong_algebra/figures`
- Create: `chuzhong_algebra/figures/STYLE.md`（基于几何 STYLE.md，加代数补充）

- [ ] **Step 1: 建目录 + 拷贝 render.sh**

```bash
cd /Users/yangyang/ai_projs/math
mkdir -p chuzhong_algebra/figures/src/tikz chuzhong_algebra/figures/src/asy chuzhong_algebra/figures/svg
cp chuzhong_geometry/figures/render.sh chuzhong_algebra/figures/render.sh
chmod +x chuzhong_algebra/figures/render.sh
sed -i '' 's|chuzhong_geometry/figures|chuzhong_algebra/figures|g' chuzhong_algebra/figures/render.sh
```

- [ ] **Step 2: 创建代数版 STYLE.md**

文件路径：`chuzhong_algebra/figures/STYLE.md`

内容必须包含（基于 `chuzhong_geometry/figures/STYLE.md`，加代数补充）：

1. **命名规则**：同几何（`thm-`, `ex-<id>-<n>`, `q-<level>-NN`），但 ID 用 `p1..p14`
2. **TikZ 通用模板**：与几何一致（含 `\usepackage{ctex}` + `\usetikzlibrary{calc, angles, quotes, decorations.markings, arrows.meta}`，新增 `pgfplots` 用于函数图）
3. **Asymptote 通用模板**：与几何一致
4. **颜色与线型**：同几何，新增"函数曲线主色 = 蓝色"约定
5. **代数特有补充**：
   - **函数图象**：坐标轴用 `\draw[->]` 带箭头；x 轴标签 `$x$` 在右端、y 轴标签 `$y$` 在上端；原点标 `$O$`
   - **函数图象关键点**：用 `\fill` 实心圆点 + 坐标标签（如 $A(2, 3)$）
   - **数轴**：水平线 + 箭头 + 标记关键点（用 `\bullet` 或 `\circ` 区分开/闭端点）
   - **统计图表**：颜色用浅色填充（不超过 4 种颜色：cyan!30, orange!30, green!30, pink!30）
   - **概率树状图**：上下分层，分支用箭头连接，概率值标在分支边上
6. **图风格示例**：附几张样例图描述（不要求源码，引用 chuzhong_geometry/figures/sample-*.svg 路径即可）

字数：1500-2200 字

- [ ] **Step 3: 提交**

```bash
cd /Users/yangyang/ai_projs/math
git add chuzhong_algebra/figures
git commit -m "feat(chuzhong_algebra): set up figure rendering infrastructure (copy from geometry + algebra additions)"
```

---

## Task 2: README.md

**File:** `chuzhong_algebra/README.md`

- [ ] **Step 1: 创建文件**

文件内容必须包含以下小节（结构参考 `chuzhong_geometry/README.md`，但内容针对代数）：

**顶部 banner：**
```markdown
# 中考代数教程

一份强调**思路培养**的中考代数（含统计与概率）教程。不仅讲"怎么算"，更讲"为什么这么变形 / 看到 X 应该立刻想 Y"。

**配套姊妹篇**：[`chuzhong_geometry/`](../chuzhong_geometry/) 中考几何教程
```

**章节小标题与内容要点：**

1. **这本书写给谁**（3 类读者：零基础 / 中等 / 中考冲刺；与几何教程文字对应）
2. **与普通教辅的不同**（强调思路：结构识别、整体思想、算两次；模板 D 类章节有"思维路径还原+思考路标"）
3. **目录**（参照 spec §2 的完整目录，**每章带年级标签**——见下"年级标签说明"）
4. **怎么用这本书**（零基础读者按 part1→ part14 顺序；中等水平先读 thinking-toolkit 10 篇 + 按主题切入；中考冲刺直接 part13 应用题 + part14 综合 + 附录 E）
5. **章节内部结构**（说明 C/D 两种模板，特别说明 D 类的"一例速记 + 思维路径还原 + 思考路标"是本书特色）
6. **关于练习题**（节内 + 节末自测 + 附录 180 题）
7. **关于图**（TikZ/Asymptote；函数图为主，公式可视化）
8. **反馈**

**年级标签说明（重要）：**

参照 spec §2 + `chuzhong_geometry/README.md` 的年级标签风格。基于北师大版 2024 新课标 / 陕西适用。具体每章年级标签：

| Part | 年级 |
|---|---|
| Part 1 数（含根式） | 初一上 + 初二上根式 |
| Part 2 整式 | 初一下 |
| Part 3 因式分解 | 初二下 |
| Part 4 分式 | 初二下 |
| Part 5 一次方程（组） | 初一上 + 初二上方程组 |
| Part 6 一元二次方程 | 初三上 |
| Part 7 不等式 | 初二下 |
| Part 8 一次函数 | 初二上 |
| Part 9 反比例函数 | 初三上 |
| Part 10 二次函数 | 初三下 |
| Part 11 统计 | 初二上 + 初三上 |
| Part 12 概率 | 初三上 |
| Part 13 应用题 | 初三复习 |
| Part 14 综合 | 初三复习 |

每个 part 标题后用 `[初X上/下 / Y上 第N章]` 格式（与几何教程一致）。

⚠️ 在每章链接后加 `⭐️` 标记表示模板 D 章（共 32 个 ⭐️）。

字数控制：1500-2500 字。

- [ ] **Step 2: 通读检查**

- [ ] 所有相对链接（`thinking-toolkit/01-when-to-substitute.md` 等）路径正确
- [ ] 三类读者使用建议清晰
- [ ] 两种章节模板的描述与 spec 一致
- [ ] 14 个 part 全列出，附年级标签
- [ ] ⭐ 标记的 32 个 D 章节标对
- [ ] 与几何 README 风格一致

- [ ] **Step 3: 提交**

```bash
git add chuzhong_algebra/README.md
git commit -m "docs(chuzhong_algebra): add tutorial README with TOC and grade tags"
```

---

## Task 3: 00-preface.md

**File:** `chuzhong_algebra/00-preface.md`

- [ ] **Step 1: 创建文件**

序言要回答三个问题：什么是代数思维 / 为什么很多人代数差 / 这本书会怎么帮。

文件必须包含以下小节，每节内容要点必须全部覆盖：

**一、什么是"代数思维"**

- 代数思维 ≠ 死记公式、机械计算
- 关键三层：
  1. **结构识别**——看到一个式子 / 方程 / 函数，先识别它的"结构"（如 $x^2 + 6x$ 是"二次项 + 一次项"，可配方；$\sqrt{a} + \sqrt{b}$ 是"和根式"可有理化；$ax^2 + bx + c = 0$ 是一元二次方程可用判别式/韦达/求根三选一）
  2. **变形策略**——根据结构选择合适的代数变形（配方、换元、因式分解、整体代入、参数分离等）
  3. **算两次/数形结合/整体观**——更高层次的代数思维（一量两式生方程；式 ↔ 图象互译；把复杂表达式当一个整体处理）
- 用一个对比：同一个题 $x^2 + 4x = 5$
  - 新手：移项 → 求根公式 → 判别式 ...（机械化）
  - 高手：看到结构 → 立刻想 "$x^2+4x = (x+2)^2-4$"（配方）→ 得 $(x+2)^2 = 9$ → 直接看出 $x+2=\pm 3$ → 解出（30 秒）

**二、为什么很多人代数学不好**

列出 4 个常见误区，每条 1-2 句话：
1. 把代数当死记：背公式不理解推导，遇到变形就懵
2. 不识别结构：拿到题就硬算，没看式子在"说"什么
3. 设元乱设：应用题设元随意，导致方程难列难解
4. 思路单一：只会一种方法（如解二次方程只会求根公式），不会因题而选方法

**三、这本书会怎么帮你**

破解上面 4 个误区：
1. 每个核心方法章节都有「思维路径还原」——把高手脑子里发生的事完整摊开
2. 思维工具箱 10 篇专讲方法论（**特别推荐**：算两次、结构识别、整体思想、数形结合 4 篇是高考也能用的代数思想）
3. 设元艺术专章（toolkit 04 + part13/01）系统讲设元策略
4. 每个核心运算章节给出 **2-3 种方法对比**（如二次方程：直接开平方 / 因式分解 / 配方 / 求根公式四种各自的"适用情形"）

**四、怎么读这本书最有效**

- 先读思维工具箱 10 篇（先建立方法论）—— **特别强调 06/07/08/09 这 4 篇是代数核心思维**
- 再选一个具体 part，按"运算章 → 模型/技巧章 → 应用章"的顺序
- D 类（⭐）章节务必先尝试"思维路径还原"，再看官方解答
- 学完一个方法后，回头看附录 B 模型图集（套路图集），能"看式子说方法"才算真正掌握
- 配套使用：代数化简卡壳时，对照几何教程同时复习——很多综合题需要"代数计算 + 几何图形识别"双线作战

**结尾**：一句鼓励——"代数不是死记，是看清结构、选对变形"。

字数：1800-2800 字。

- [ ] **Step 2: 通读检查**

- [ ] 三层代数思维（结构识别 / 变形策略 / 高层思想）的提法与后续 toolkit 章节一致
- [ ] 提到的具体例子（$x^2 + 4x = 5$ 配方）正确无误
- [ ] 与 README 中"两种章节模板"的描述不冲突
- [ ] 没有具体几何题或微积分内容

- [ ] **Step 3: 提交**

```bash
git add chuzhong_algebra/00-preface.md
git commit -m "docs(chuzhong_algebra): add preface on algebraic thinking"
```

---

## Task 4: toolkit/01-when-to-substitute.md（何时换元）

**File:** `chuzhong_algebra/thinking-toolkit/01-when-to-substitute.md`

- [ ] **Step 1: 创建目录与文件**

```bash
mkdir -p chuzhong_algebra/thinking-toolkit
```

文件必须按以下结构组织（与几何 toolkit 风格一致）：

**一、换元不是高考专属**
- 反驳"换元是高考的事"——初中已经在用（解高次方程、分式方程、根式方程时换元降次）
- 换元 = 给复杂表达式起个新名字（如 $y = x^2$、$t = \sqrt{x-1}$、$u = \frac{a}{b}$）后求解

**二、识别"该换元"的 5 类信号**

每类必须给出：信号特征 + 一个示例（公式形式 + 一句话说明）+ 换元做法。

1. **方程出现高次 + 含 $x^2$（或 $x^k$）的重复结构** → 如 $x^4 - 5x^2 + 4 = 0$，令 $y = x^2$ 化为 $y^2 - 5y + 4 = 0$
2. **方程含根号** → 如 $x + \sqrt{x-1} = 3$，令 $t = \sqrt{x-1}$，化为 $t^2 + t - 2 = 0$
3. **分式方程含相同分式结构** → 如 $\frac{x^2+1}{x} + \frac{x}{x^2+1} = 2.5$，令 $u = \frac{x^2+1}{x}$，化为 $u + \frac{1}{u} = 2.5$
4. **多元方程组含对称组合** → 如 $\begin{cases} x+y = 5 \\ xy = 6 \end{cases}$，令 $s = x+y, p = xy$（韦达对称结构）
5. **代数式含明显的重复块** → 如 $(x^2+1)^2 - 2(x^2+1) - 8$，令 $y = x^2+1$ 化简

**三、换元的三步操作**
- 第一步：识别**整体**（哪一块要换元）
- 第二步：用新字母 $y$ / $t$ / $u$ 替换原整体
- 第三步：求出新方程的解后**回代**还原原变量

**四、一个完整的换元演示**

例：求解 $x^4 - 13x^2 + 36 = 0$
- 看到 $x^4 = (x^2)^2$ → 信号 1，令 $y = x^2$
- 原式变 $y^2 - 13y + 36 = 0$ → 因式分解 $(y-4)(y-9) = 0$
- 解 $y_1 = 4, y_2 = 9$
- 回代：$x^2 = 4$ → $x = \pm 2$；$x^2 = 9$ → $x = \pm 3$
- 答：$x = \pm 2, \pm 3$

**五、易错点**
- 换元后**必须回代**，不能停在 $y$ 的解
- 注意换元后**变量的取值范围**（如 $y = x^2 \geq 0$，求出 $y < 0$ 的解要舍）
- 换元前后表达式的等价性（含根号换元如 $t = \sqrt{x-1}$，必有 $t \geq 0$）

**六、自检 checklist**
- 题目里是否有"重复的结构 / 块"？
- 给这个结构起个新字母后，方程是否变简单？
- 新方程的解是否能回代回原方程？
- 新变量是否有附加范围限制？

字数：1800-2500 字。

- [ ] **Step 2: 提交**

```bash
git add chuzhong_algebra/thinking-toolkit/01-when-to-substitute.md
git commit -m "docs(chuzhong_algebra): add thinking-toolkit 01 (when to substitute)"
```

---

## Task 5: toolkit/02-when-to-complete-square.md（何时配方）

**File:** `chuzhong_algebra/thinking-toolkit/02-when-to-complete-square.md`

- [ ] **Step 1: 创建文件**

结构：

**一、配方是代数的"瑞士军刀"**
- 配方 = 把 $ax^2+bx+c$ 写成 $a(x+h)^2+k$ 形式
- 用处：解二次方程、求二次函数顶点、求最值、判别表达式正负

**二、识别"该配方"的 4 类信号**

1. **解一元二次方程** + 一次项系数偶数 → 配方比求根公式快（如 $x^2 + 6x - 7 = 0$）
2. **求二次函数最值 / 顶点** → $y = ax^2 + bx + c \to y = a(x+\frac{b}{2a})^2 + (c - \frac{b^2}{4a})$
3. **判别二次式 $ax^2+bx+c$ 的正负** → 配方后看 $a$ 与 $k$ 的符号
4. **证明不等式** → 如证 $x^2 - 4x + 5 > 0$，配方得 $(x-2)^2 + 1 > 0$

**三、配方公式与三步操作**

$$ax^2 + bx + c = a\left(x + \frac{b}{2a}\right)^2 + c - \frac{b^2}{4a}$$

操作步骤（最常用 $a=1$ 情形）：
- 第一步：提一次项系数 → $x^2 + 6x - 7 = x^2 + 6x \cdot 1 - 7$
- 第二步：加减"一次项系数一半的平方" → $(x^2 + 6x + 9) - 9 - 7 = (x+3)^2 - 16$
- 第三步：解或化简

**四、典型应用**（3 例每带【思路】）：
- 例 1：解 $x^2 - 10x + 21 = 0$（配方法）
- 例 2：求 $y = 2x^2 - 8x + 5$ 的最值（顶点式）
- 例 3：证明 $x^2 + y^2 - 2x - 4y + 5 \geq 0$（两次配方）

**五、易错点**
- 一次项系数为奇数时配方出现 $\frac{1}{2}$，多数题应改用求根公式
- 含 $a \neq 1$ 时务必先提系数再配
- 配方完不能漏掉"加上的项要减回去"

**六、自测题** 4 题。

字数：1800-2500 字。

- [ ] **Step 2: 提交**

```bash
git add chuzhong_algebra/thinking-toolkit/02-when-to-complete-square.md
git commit -m "docs(chuzhong_algebra): add thinking-toolkit 02 (when to complete the square)"
```

---

## Task 6: toolkit/03-discriminant-and-vieta.md（判别式 + 韦达）

**File:** `chuzhong_algebra/thinking-toolkit/03-discriminant-and-vieta.md`

- [ ] **Step 1: 创建文件**

结构：

**一、判别式与韦达定理是一元二次方程的"双子星"**
- 判别式 $\Delta = b^2 - 4ac$ 判断**根的个数与性质**
- 韦达定理：根与系数关系 $x_1 + x_2 = -\frac{b}{a}$、$x_1 x_2 = \frac{c}{a}$，不解方程**直接用根**

**二、判别式三种情形**（用表格）：

| $\Delta$ | 根的情况 | 几何意义（$y = ax^2+bx+c$ 图象与 x 轴） |
|---|---|---|
| $\Delta > 0$ | 两个不相等实根 | 与 x 轴有两个交点 |
| $\Delta = 0$ | 两个相等实根 | 与 x 轴有一个切点 |
| $\Delta < 0$ | 无实根 | 与 x 轴无交点 |

**三、韦达定理与常用变形**

$x_1 + x_2 = -\frac{b}{a}$、$x_1 x_2 = \frac{c}{a}$

衍生公式：
- $x_1^2 + x_2^2 = (x_1+x_2)^2 - 2x_1 x_2$
- $\frac{1}{x_1} + \frac{1}{x_2} = \frac{x_1+x_2}{x_1 x_2}$
- $(x_1-x_2)^2 = (x_1+x_2)^2 - 4 x_1 x_2$

**四、识别"用判别式 / 韦达"的信号**

- **用判别式**：题问"几个根 / 根的存在性 / 求参数使方程有根"
- **用韦达**：题给"两根满足某关系"或"求两根的对称表达式"，**且不要求求出根本身**

**五、典型应用**（3 例）：
- 例 1：判别 $2x^2 - 3x + 1 = 0$ 的根的情况
- 例 2：已知 $x^2 + 3x - 5 = 0$ 两根 $x_1, x_2$，求 $x_1^2 + x_2^2$（用韦达，不解方程）
- 例 3：方程 $x^2 - mx + 4 = 0$ 有两个不相等实根，求 $m$ 范围（判别式 $> 0$，$m^2 - 16 > 0$，$m > 4$ 或 $m < -4$）

**六、易错点**
- 韦达定理前提：**$a \neq 0$**，且**方程是一元二次方程**
- 韦达定理给的是两根的"对称表达"，单独求 $x_1$ 或 $x_2$ 还得解方程
- 含参方程用韦达后，结果可能含参，注意分类讨论

**七、自测题** 4 题。

字数：2000-2700 字。

- [ ] **Step 2: 提交**

```bash
git add chuzhong_algebra/thinking-toolkit/03-discriminant-and-vieta.md
git commit -m "docs(chuzhong_algebra): add thinking-toolkit 03 (discriminant and Vieta)"
```

---

## Task 7: toolkit/04-setting-up-equations.md（设元的艺术）

**File:** `chuzhong_algebra/thinking-toolkit/04-setting-up-equations.md`

结构：

**一、设元是应用题的第一步**
- 列方程 = 把文字描述翻译成数学语言；设元 = 给"未知"起名
- 设得好：方程简单、关系清晰；设得差：方程复杂、计算麻烦

**二、设元的 3 类策略**

1. **直接设**：题目问什么就设什么（如"求速度"→ 设速度为 $v$）
2. **间接设**：直接设导致方程难列时，设一个中间量（如"求两人速度之比"→ 设两人速度分别 $3k, 5k$）
3. **多设少减**：题目含 2-3 个未知，全部设出后通过等量关系减元

**三、设元的 4 条原则**

1. 设元要明确单位（"设速度为 $v$ 千米/时"，不是裸字母）
2. 多个未知时，**尽量减少元数**（能间接表达就不另设）
3. **取值范围**——设完元立刻想"$v > 0$"、"$x$ 为正整数"等限制
4. 设元后**先验证**：题给条件能否全部用 $v$ 等表达？不能则换设法

**四、设元的"思维路径还原"** —— 演示一道行程题

题目：A、B 两地相距 100 km，甲从 A 出发去 B，速度 50 km/h；乙从 B 出发去 A，速度 40 km/h。两人同时出发，何时相遇？

> "题目问'何时相遇'——直接设：设 $t$ 小时后相遇。
> 验证：能否用 $t$ 表达条件？甲走的路程 $= 50t$，乙走的路程 $= 40t$。
> 相遇 = 两人路程和 $=$ 总距离：$50t + 40t = 100$。
> 解出 $t = \frac{10}{9}$ 小时。$t > 0$ 合理。
> 这是最直接的设元法，方程 1 个未知、1 步成立。"

**五、设元的 4 条进阶技巧**

1. 比例题用 $k$（如"甲乙速度比 3:5"→ 设 $3k, 5k$）
2. 几何题设线段长度（不一定是题目问的量）
3. 含参题先设参数，再用约束推参数关系
4. 方案题（如调配）设最大未知量，其他用减法表达

**六、典型应用** 3 例：直接设 / 间接设 / 多设少减。

**七、自测题** 4 题。

字数：2000-2700 字。

- [ ] **Step 1: 创建文件**（按上述结构写）
- [ ] **Step 2: 提交**

```bash
git add chuzhong_algebra/thinking-toolkit/04-setting-up-equations.md
git commit -m "docs(chuzhong_algebra): add thinking-toolkit 04 (art of setting variables)"
```

---

## Task 8: toolkit/05-integral-thinking.md（整体思想）

**File:** `chuzhong_algebra/thinking-toolkit/05-integral-thinking.md`

结构：

**一、整体思想——把"一团"看作"一个"**
- 整体 = 不去拆开看，把复杂表达式当一个变量处理
- 与换元的区别：换元是给整体起新名字；整体思想是不起名字直接用（更高级）

**二、整体思想的 4 类应用**

1. **整体代入**：求 $x^2 + 3x + 1$，但题目给的是 $x^2 + 3x = 5$ → 直接代入 $= 5 + 1 = 6$
2. **整体平移**：函数 $y = (x-2)^2 + 1$ 由 $y = x^2$ 整体右移 2 上移 1（不拆开）
3. **整体平方**：已知 $x + \frac{1}{x} = 3$，求 $x^2 + \frac{1}{x^2}$ → $(x + \frac{1}{x})^2 - 2 = 9 - 2 = 7$
4. **整体消元**：方程组 $\begin{cases} x+y+z = 6 \\ 2(x+y) = z \end{cases}$ → 把 $x+y$ 看作一个整体

**三、识别"用整体"的信号**
- 题目给的条件是"和 / 积 / 差的整体"（如 $x_1 + x_2$、$x_1 x_2$、$x + \frac{1}{x}$）而非单独的 $x_1, x_2$
- 题目要求的也是"和 / 积 / 差"形式
- 直接解出 $x_1, x_2$ 困难 → 整体思想绕开

**四、整体思想的演示题**

题目：已知 $x^2 - 3x + 1 = 0$，求 $\frac{x^4 + 1}{x^2}$。

> "$x^2 - 3x + 1 = 0$ → $x^2 + 1 = 3x$（移项整体化）
> $\frac{x^4 + 1}{x^2} = \frac{(x^2)^2 + 1}{x^2} = x^2 + \frac{1}{x^2}$（除分）
> 现在用整体：从 $x^2 + 1 = 3x$ 除以 $x$（$x \neq 0$）得 $x + \frac{1}{x} = 3$
> 平方：$x^2 + 2 + \frac{1}{x^2} = 9$ → $x^2 + \frac{1}{x^2} = 7$
> 答：7"

**五、思考路标**
- 见 $x + \frac{1}{x}$ 这类整体 → 想平方变形
- 见两根之"和"和"积" → 想韦达 + 整体代入
- 见整个表达式作为已知 → 不拆分直接用
- 函数图象题 → 把"整体平移 / 反射"作为变换看

**六、典型应用** 3 例 + **七、自测题** 4 题。

字数：2000-2700 字。

- [ ] **Step 1: 创建文件**
- [ ] **Step 2: 提交**

```bash
git add chuzhong_algebra/thinking-toolkit/05-integral-thinking.md
git commit -m "docs(chuzhong_algebra): add thinking-toolkit 05 (integral thinking)"
```

---

## Task 9: toolkit/06-counting-twice.md（算两次思想 NEW）

**File:** `chuzhong_algebra/thinking-toolkit/06-counting-twice.md`

结构：

**一、什么是"算两次"**
- 算两次 = **同一个量用两种不同方式表达**，两种结果相等 → 得到方程
- 这是列方程**最基础**的原理，应用题、几何代数综合都靠它

**二、算两次的 4 类常见场景**

1. **行程问题**：总路程 = 甲走 + 乙走（两种方式表达总路程） → 方程
2. **几何面积**：用底高一次、用海伦公式或三角函数一次（两种方式算面积） → 方程
3. **数列与统计**：总人数用各组之和、用平均数 × 总数（两种方式） → 方程
4. **代数表达式**：同一表达式两种因式分解（如展开 vs 公式）→ 推出恒等式

**三、算两次的核心套路**

**步骤**：
1. 找一个"可两种方式表达"的量 $Q$（路程、面积、人数、表达式值）
2. 写出两种表达式 $f_1(\text{未知}) = Q$、$f_2(\text{未知}) = Q$
3. 令 $f_1 = f_2$ 得方程
4. 解之

**四、算两次的演示题**

题目：直角 $\triangle ABC$，$\angle C = 90°$，$AC = 6$，$BC = 8$。求 C 到斜边 $AB$ 的距离。

> "C 到 AB 的距离 = △ABC 的边 $AB$ 上的高 $h$
> **方法 1 算面积**：$S = \frac{1}{2} \cdot AC \cdot BC = \frac{1}{2} \cdot 6 \cdot 8 = 24$
> **方法 2 算面积**：$S = \frac{1}{2} \cdot AB \cdot h$，其中 $AB = \sqrt{6^2 + 8^2} = 10$，所以 $S = 5h$
> **算两次相等**：$24 = 5h$ → $h = 4.8$
> 答：距离 $= 4.8$"

**五、思考路标**
- 见"求高 / 距离" + 直角三角形 → **面积算两次**
- 见行程题 → 总路程或时间算两次
- 见"统计 / 平均数" → 总和算两次（一次按组、一次按总）
- 几何代数综合中"求长度" → 找一个能两种方式表达的几何量

**六、典型应用** 3 例 + **七、自测题** 4 题。

字数：2000-2700 字。

- [ ] **Step 1: 创建文件**
- [ ] **Step 2: 提交**

```bash
git add chuzhong_algebra/thinking-toolkit/06-counting-twice.md
git commit -m "docs(chuzhong_algebra): add thinking-toolkit 06 (counting twice principle)"
```

---

## Task 10: toolkit/07-structure-recognition.md（结构识别 NEW）

**File:** `chuzhong_algebra/thinking-toolkit/07-structure-recognition.md`

结构：

**一、结构识别——代数思维的第一步**
- 看到一个式子先问："它是什么结构？" 而不是"我能算什么？"
- 结构决定方法——同一题不同结构识别 → 不同解法 → 难易差很多

**二、必背 8 种代数结构与对应方法**

每个结构给：**特征 / 标志 / 应对方法**（用表格）：

| 结构 | 特征 | 标志 | 应对方法 |
|---|---|---|---|
| **二次三项式** | $ax^2+bx+c$ | 三项含 $x^2$、$x$、常数 | 配方 / 十字相乘 / 求根公式 |
| **完全平方** | $a^2 \pm 2ab + b^2$ | 两端是平方、中间是 2 倍乘积 | 写成 $(a \pm b)^2$ |
| **平方差** | $a^2 - b^2$ | 两个平方相减 | $(a+b)(a-b)$ |
| **和根式** | $\sqrt{a} \pm \sqrt{b}$ | 含两个根号相加减 | 有理化（乘共轭）|
| **倒数和** | $x + \frac{1}{x}$ | 一个量 + 它的倒数 | 平方变形 |
| **对称式** | $f(x, y) = f(y, x)$ | $x, y$ 对换式子不变 | 换元 $s = x+y, p = xy$ |
| **分式方程** | 分母含未知 | 见 $\frac{}{x}$ 等 | 去分母 + 检验增根 |
| **绝对值** | $\|f(x)\|$ | 含 $\|\cdot\|$ | 分类讨论（按 $f \geq 0$ vs $f < 0$）|

**三、结构识别口诀**

```
见二次三项式，配方/十字/公式三选一；
见两平方相减，立刻平方差；
见根号相加减，有理化；
见倒数和，平方变形；
见对称，s/p 换元；
见分式方程，去分母 + 验根；
见绝对值，分类讨论。
```

**四、演示：同一题，两种结构识别**

题目：求 $\sqrt{12+\sqrt{140}}$ 的值（要求化为简单形式）。

**新手**：盲算
**高手**：结构识别 → "$12 = a + b$, $140 = 4ab$ → $a, b$ 是 $t^2 - 12t + 35 = 0$ 的根 → $t = 5, 7$ → $12 + \sqrt{140} = (\sqrt{5} + \sqrt{7})^2$ → 答 $\sqrt{5} + \sqrt{7}$"

**五、思考路标**（至少 8 条）
- 见 $x^2 + px$ 型 → 想配方加 $\frac{p^2}{4}$
- 见 $a^2 - b^2$ → 想平方差因式分解
- 见 $\sqrt{a \pm b}$（嵌套根号）→ 想 $(\sqrt{m} \pm \sqrt{n})^2$ 还原
- 见分母含未知 → 想分式方程 + 验根
- 见对称式 → 想 $s = x+y, p = xy$ 换元
- 见 $|x-a|$ → 想分类讨论
- 见高次方程 → 想换元降次或因式分解
- 见根号 + 等式 → 想平方两边（注意增根）

**六、典型应用** 3 例 + **七、自测题** 5 题。

字数：2200-3000 字。

- [ ] **Step 1: 创建文件**
- [ ] **Step 2: 提交**

```bash
git add chuzhong_algebra/thinking-toolkit/07-structure-recognition.md
git commit -m "docs(chuzhong_algebra): add thinking-toolkit 07 (structure recognition)"
```

---

## Task 11: toolkit/08-parameter-strategy.md（参数策略 NEW）

**File:** `chuzhong_algebra/thinking-toolkit/08-parameter-strategy.md`

结构：

**一、含参问题——中考代数压轴必考**
- 参数 = 题目里以字母（$m, k, a$ 等）出现的"未确定"量
- 含参题：求参数取值范围 / 分类讨论 / 参数分离求最值

**二、含参问题的 3 类策略**

1. **直接代入策略**：先把参数当作已知数处理，列出"含参解" 后再讨论参数
2. **参数分离策略**：把方程 / 不等式分离为"参数 = 表达式"或"参数 ≥ 表达式"形式，再用图象/最值法
3. **分类讨论策略**：参数影响方程类型（一次/二次）、判别式正负、根的位置时分情形讨论

**三、参数分离的核心套路**

例：求 $kx + 2 = 3x - 1$ 有解时 $k$ 的范围
- 移项整理：$(k-3)x = -3$
- **若 $k \neq 3$**：$x = \frac{-3}{k-3}$，方程有唯一解 → 任意 $k \neq 3$
- **若 $k = 3$**：$0 \cdot x = -3$，无解
- 答：$k \neq 3$

**四、含参一元二次方程的 4 个讨论维度**

1. **是否是二次方程**：$ax^2 + bx + c = 0$，先讨论 $a \neq 0$ 与 $a = 0$
2. **判别式正负**：$\Delta > 0 / = 0 / < 0$
3. **根的范围**：用韦达定理 $x_1 x_2$、$x_1 + x_2$、$\Delta$ 三者结合
4. **根的实际意义**：如"求 $k$ 使方程有两正根" → $\Delta > 0$ + $x_1 + x_2 > 0$ + $x_1 x_2 > 0$

**五、含参不等式的演示**

题目：解不等式 $kx > 3 - x$（$k$ 为常数）。

> "整理：$(k+1)x > 3$
> **若 $k+1 > 0$（即 $k > -1$）**：$x > \frac{3}{k+1}$
> **若 $k+1 < 0$（即 $k < -1$）**：$x < \frac{3}{k+1}$（不等号反转）
> **若 $k = -1$**：$0 > 3$，无解
> 综合：分类讨论给出三种结论。"

**六、思考路标**
- 见含参一次方程 → 讨论"参数系数是否为 0"
- 见含参不等式 → 讨论"参数系数符号"
- 见含参二次方程 → 4 个维度（是否二次 / 判别式 / 韦达 / 根的位置）依次讨论
- 见参数最值题 → 先尝试参数分离 + 函数最值

**七、典型应用** 3 例 + **八、自测题** 4 题。

字数：2200-3000 字。

- [ ] **Step 1: 创建文件**
- [ ] **Step 2: 提交**

```bash
git add chuzhong_algebra/thinking-toolkit/08-parameter-strategy.md
git commit -m "docs(chuzhong_algebra): add thinking-toolkit 08 (parameter strategy)"
```

---

## Task 12: toolkit/09-number-shape-combination.md（数形结合 NEW）

**File:** `chuzhong_algebra/thinking-toolkit/09-number-shape-combination.md`

结构：

**一、数形结合——代数的最高境界**
- 数 = 代数式、方程、不等式
- 形 = 函数图象、数轴、几何图形
- 互译 = 把抽象的"数"变成直观的"形"，反之亦然

**二、数 → 形 的 4 类常用**

1. **方程 → 图象交点**：方程 $f(x) = g(x)$ 的解 = 函数 $y = f(x)$ 与 $y = g(x)$ 图象的交点横坐标
2. **不等式 → 图象比较**：不等式 $f(x) > g(x)$ 的解集 = 图象 $f$ 在 $g$ 上方的 $x$ 区间
3. **绝对值 → 数轴距离**：$|x - a| = $ 数轴上 $x$ 到 $a$ 的距离
4. **二次三项式正负 → 抛物线开口与判别式**

**三、形 → 数 的 4 类常用**

1. **图象题求未知** → 设解析式 + 待定系数法
2. **几何题求长度** → 设坐标 + 距离公式
3. **动态几何 → 函数关系**（如"动点位置 → 面积函数"）
4. **图象交点存在性 → 联立方程判别式**

**四、演示：数 → 形**

题目：求方程 $x^2 - 4 = |x|$ 的解的个数。

> "纯代数解：分类讨论 $x \geq 0$ 和 $x < 0$，分两个二次方程解 —— 4 个候选根筛 ...
> **数形结合**：画 $y = x^2 - 4$（抛物线）和 $y = |x|$（折线 V 形）
> 看图：抛物线和 V 形折线交点 = 解
> 抛物线开口向上、顶点 $(0, -4)$；V 形顶点 $(0, 0)$
> 显然 V 形在抛物线**上方**直到某个 $x$ 之后才相交
> 数图分析得：恰好 2 个交点（左右对称各一个）
> 答：2 个解"

**五、演示：形 → 数**

题目：抛物线 $y = x^2 + bx + c$ 经过 $(1, 0)$ 和 $(3, 0)$，求 $b, c$。

> "**形 → 数**：抛物线与 x 轴交点 = 方程 $x^2 + bx + c = 0$ 的根
> 由韦达：$x_1 + x_2 = -b$，$x_1 x_2 = c$
> 代入：$1 + 3 = -b$ → $b = -4$；$1 \cdot 3 = c$ → $c = 3$
> 答：$b = -4, c = 3$"

**六、思考路标**
- 解方程困难时 → 想图象交点
- 解不等式带绝对值 / 分式 → 数轴 / 图象法
- 求函数最值 → 找顶点（二次）/ 端点（一次）
- 几何与代数综合 → 设坐标桥接
- 见 $|x \pm a|$ → 数轴距离意义

**七、典型应用** 3 例 + **八、自测题** 4 题。

字数：2200-3000 字。

- [ ] **Step 1: 创建文件**
- [ ] **Step 2: 提交**

```bash
git add chuzhong_algebra/thinking-toolkit/09-number-shape-combination.md
git commit -m "docs(chuzhong_algebra): add thinking-toolkit 09 (number-shape combination)"
```

---

## Task 13: toolkit/10-checklist-before-solving.md（开题清单）

**File:** `chuzhong_algebra/thinking-toolkit/10-checklist-before-solving.md`

结构：

**一、为什么需要"开题清单"**
- 代数题卡壳的真正原因：开始算太早，没识别结构
- 5 分钟思考远胜 30 分钟乱算

**二、开题 7 问**

按顺序，每问：**为什么问 + 怎么问 + 典型例子**：

**Q1. 这是什么类型的题？方程？不等式？函数？应用？化简？**
- 类型决定大方向

**Q2. 题目里的代数式是什么结构？**（用 toolkit/07）
- 二次三项式？完全平方？平方差？倒数和？

**Q3. 题目里有几个未知？几个等量关系？**
- 等量关系数 ≥ 未知数 → 能列方程 / 方程组

**Q4. 设元应该设谁？直接设还是间接设？**（用 toolkit/04）

**Q5. 是否需要先化简代数式再处理？**
- 因式分解 / 通分 / 有理化等预处理

**Q6. 用哪种方法解？有几种备选？**
- 一元二次方程：直接开平方 / 因式分解 / 配方 / 求根公式
- 选最快的（如 $x^2 = 4$ 用直接开平方而非求根公式）

**Q7. 答案需要哪种形式？是否要分类讨论 / 检验？**
- 增根（分式方程）/ 范围（不等式取整数解？）/ 实际意义（速度 $> 0$）

**三、清单使用示范** 

给一道中档题，按 Q1-Q7 完整走一遍演示。

**四、把清单内化**
- 训练 2-3 周后过一遍清单只需 30 秒
- 救回 80% 卡题

字数：2200-3000 字。

- [ ] **Step 1: 创建文件**
- [ ] **Step 2: 提交**

```bash
git add chuzhong_algebra/thinking-toolkit/10-checklist-before-solving.md
git commit -m "docs(chuzhong_algebra): add thinking-toolkit 10 (checklist before solving)"
```

---

## Task 14: AM0 收尾

- [ ] **Step 1: 一致性扫描**

```bash
cd /Users/yangyang/ai_projs/math
grep -rn 'TBD\|TODO\|XXX\|待补' chuzhong_algebra/
ls chuzhong_algebra/
ls chuzhong_algebra/thinking-toolkit/
ls chuzhong_algebra/figures/
```

期望：grep 无输出；根目录有 README + 00-preface；thinking-toolkit 下 10 个 .md；figures 下 render.sh + STYLE.md + src/ + svg/。

- [ ] **Step 2: 术语一致性检查**

- 各 toolkit 章节互相引用术语一致（"算两次"、"结构识别"等）
- 与 spec 中的命名一致
- 与 `chuzhong_geometry/thinking-toolkit/` 的术语**不冲突**（如"思维路径还原"用法相同）

- [ ] **Step 3: 标记里程碑完成**

在本计划文件末尾追加 `**AM0 完成于：YYYY-MM-DD**`，提交：

```bash
git add docs/superpowers/plans/2026-05-12-chuzhong-algebra-AM0.md
git commit -m "docs(chuzhong_algebra): mark AM0 milestone complete"
```

---

## 自检总结

- **Spec 覆盖**：AM0 在 spec §6 里程碑表中明确列出"README + 00-preface + thinking-toolkit 10 篇 + figures 基础设施 = 13 文件"，本计划 Task 1-13 一一对应 ✓
- **占位扫描**：无 TBD / 模糊描述；每个文件的小节结构和内容要点都已具体化 ✓
- **类型一致**：thinking-toolkit 各章节互相引用术语一致 ✓
- **粒度**：每个 Task = 一个文件 = 一次提交 ✓
- **基础设施**：figures 拷贝并加代数补充约定 ✓
