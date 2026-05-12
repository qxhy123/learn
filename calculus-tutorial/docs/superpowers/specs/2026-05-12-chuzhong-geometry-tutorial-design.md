# 中考几何教程 设计文档

**创建日期**：2026-05-12
**目标目录**：`chuzhong_math/`
**状态**：设计已确认，待生成实施计划

---

## 1. 定位与风格

- **读者**：综合型——从零基础到中考冲刺均可使用
- **核心特色**：**思路培养** > 知识陈列。不仅讲"是什么"，更讲"怎么想到的"
- **风格**：与同仓库 `calculus-tutorial` 保持一致：纯 markdown + LaTeX 公式，注重推导过程，可版本控制
- **语言**：中文为主，关键术语保留英文/拉丁字母

---

## 2. 整体目录结构（精细版，按模型/题型粒度拆分）

```
chuzhong_math/
├── README.md                          # 教程总览、使用说明、学习路径建议
├── 00-preface.md                      # 写给读者：什么是"几何思维"
│
├── part1-foundations/                 # 图形初步
│   ├── 01-point-line-plane.md
│   ├── 02-angles.md
│   ├── 03-angle-bisector.md
│   └── figures/
│
├── part2-parallel-lines/              # 相交线与平行线
│   ├── 01-vertical-and-adjacent-angles.md
│   ├── 02-parallel-criteria.md
│   ├── 03-three-line-eight-angles.md
│   └── figures/
│
├── part3-triangles/                   # 三角形（重点 part，约占总量 1/4）
│   ├── basics/
│   │   ├── 01-triangle-inequality.md
│   │   ├── 02-angle-sum.md
│   │   └── ...
│   ├── congruence/
│   │   ├── 01-sss-sas-asa.md
│   │   ├── 02-model-hand-in-hand.md          # 手拉手模型
│   │   ├── 03-model-half-angle.md            # 半角模型
│   │   ├── 04-model-one-line-three-angles.md # 一线三等角
│   │   └── ...
│   ├── similarity/
│   │   ├── 01-similarity-criteria.md
│   │   ├── 02-model-A-and-X.md                # A 字 / X 字 / 8 字模型
│   │   ├── 03-model-shared-angle.md           # 共角共边模型
│   │   └── ...
│   ├── right-triangle/
│   │   ├── 01-pythagoras.md
│   │   ├── 02-pythagoras-converse.md
│   │   └── ...
│   ├── special/
│   │   ├── 01-isosceles.md
│   │   ├── 02-equilateral.md
│   │   └── 03-four-centers.md                 # 四心初步
│   └── figures/
│
├── part4-quadrilaterals/              # 四边形
│   ├── 01-polygon-angles.md
│   ├── 02-parallelogram.md
│   ├── 03-rectangle.md
│   ├── 04-rhombus.md
│   ├── 05-square.md
│   ├── 06-trapezoid.md
│   ├── 07-midsegment.md
│   ├── 08-model-midpoint-quadrilateral.md     # 中点四边形等模型
│   └── figures/
│
├── part5-circles/                     # 圆
│   ├── 01-circle-basics.md
│   ├── 02-perpendicular-chord.md              # 垂径定理
│   ├── 03-inscribed-angle.md                  # 圆周角
│   ├── 04-position-relations.md
│   ├── 05-tangent-criteria.md
│   ├── 06-tangent-length.md
│   ├── 07-model-tangent-secant.md
│   ├── 08-model-four-points-concyclic.md      # 四点共圆
│   └── figures/
│
├── part6-transformations/             # 图形变换
│   ├── 01-axial-symmetry.md
│   ├── 02-translation.md
│   ├── 03-rotation.md
│   ├── 04-central-symmetry.md
│   ├── 05-homothety.md
│   ├── 06-model-fold.md                       # 翻折模型
│   ├── 07-model-rotation-tricks.md            # 旋转构造全等
│   └── figures/
│
├── part7-solving-right-triangle/      # 解直角三角形
│   ├── 01-trig-ratios.md
│   ├── 02-special-angles.md
│   ├── 03-applications.md
│   └── figures/
│
├── part8-views-and-projections/       # 视图与投影
│   ├── 01-three-views.md
│   ├── 02-projections.md
│   ├── 03-unfoldings.md
│   └── figures/
│
├── part9-coordinate-geometry/         # 坐标系中的几何
│   ├── 01-coords-and-transforms.md
│   ├── 02-distance-formula.md
│   ├── 03-coord-geometry-models.md
│   └── figures/
│
├── part10-comprehensive/              # 中考综合题专题
│   ├── 01-moving-point-on-line.md             # 线上动点
│   ├── 02-moving-point-on-parabola.md         # 抛物线动点
│   ├── 03-geometry-max-min.md                 # 将军饮马 / 胡不归 / 阿氏圆
│   ├── 04-existence-problems.md               # 存在性问题
│   ├── 05-fold-comprehensive.md
│   ├── 06-rotation-comprehensive.md
│   └── figures/
│
├── thinking-toolkit/                  # 思维工具箱（独立小册子，可单独阅读）
│   ├── 01-when-to-draw-auxiliary-line.md      # 何时该作辅助线
│   ├── 02-auxiliary-line-catalog.md           # 常见辅助线总目录
│   ├── 03-model-recognition.md                # 模型识别速查
│   ├── 04-conversion-techniques.md            # 转化思想（动→静、形→数）
│   └── 05-checklist-before-solving.md         # 看到题先问自己什么
│
└── appendix/
    ├── A-formula-sheet.md             # 公式定理速查
    ├── B-model-atlas.md               # 全部模型图集（一图一结论）
    ├── C-exercises-basic.md           # 基础题库（约 100 题）
    ├── D-exercises-medium.md          # 中档题库（约 100 题）
    ├── E-exercises-advanced.md        # 压轴/真题（约 50 题）
    └── F-solutions.md                 # 全部题库详解（带模型/思路标签）
```

**规模估算**：约 70-80 个 markdown 文件。

---

## 3. 单节内部结构（两种模板）

### 模板 C：概念/定理类章节
适用于：定义、基本定理、性质类内容（如"垂径定理"、"圆周角定理"、"勾股定理"）

```markdown
# 章节标题

## 一、图形特征：怎么一眼认出
（一句话 + 示意图：见到 X 就该想到 Y）

## 二、结论是什么
（精确陈述定理）

## 三、为什么成立（证明）
（完整推导，不跳步）

## 四、典型应用
例 1 …… 【思路】小节演示分析过程
例 2 ……

## 五、易错点 & 反例

## 六、思路自测题（3-5 题，仅给提示不给完整解）
```

### 模板 D：模型/题型类章节
适用于：解题模型、综合题型（如"手拉手模型"、"一线三等角"、"将军饮马"）

```markdown
# 章节标题

## 一、引入：一道让你卡住的题
（先抛题，让读者尝试。提示卡点。）

## 二、思维路径还原（解题者的内心独白）
> 看到两个等腰三角形共顶点……
> 我想找全等或相似……
> 发现 ∠1 + ∠公共 = ∠2 + ∠公共，所以 ∠1 = ∠2……
> 加上两组对应边相等 → SAS → 全等！

（完整摊开"为什么这么想"）

## 三、抽象成模型
图形特征 → 结论 → 证明（简洁版）

## 四、模型变形
（旋转、镜像、退化情形）

## 五、思考路标（看到什么 → 想到什么）
- 看到两个共顶点等腰三角形 → 想手拉手 → 找旋转全等
- 看到 60° 角 + 等边三角形 → 想手拉手 + 旋转 60°
- ……

## 六、应用例题（2-3 题，演示路标怎么用）

## 七、思路自测题
```

**关键设计**：模板 D 的"思维路径还原"和"思考路标"两个小节是本教程区别于普通教辅的核心。

---

## 4. 图、公式、约定

### 图的处理
- **TikZ 代码块**：嵌入 markdown，作为图的源码。支持 LaTeX 渲染器直接出图。优先方式
- **预渲染图**：复杂图形（圆中复合图、动点轨迹等）放 `figures/` 子目录
  - 优先格式 `.svg`，必要时 `.png`
  - 命名规范：`partN-XX-figureM-描述.svg`
  - 每个 part 目录下自带 `figures/` 子文件夹

### 公式
- 行内：`$...$`
- 行间：`$$...$$`
- 与微积分教程统一

### 标注约定
- 点用大写字母：A, B, C
- 角：∠ABC
- 线段：粗体或 $\overline{AB}$
- 辅助线：TikZ 中用 `dashed` 样式

### 模型节标配
- 节首一张"一图速记"：最精简形态的模型图，便于快速查阅

---

## 5. 练习题系统

- **节内例题**：完整解答 + 【思路】小节演示分析过程
- **节末"思路自测题"**：3-5 题，仅给提示不给完整解
  - 提示格式：`💡 提示：注意 BD = BC 这个条件，想想能否构造……`
- **Appendix 题库**：
  - `C-exercises-basic.md`：基础巩固，约 100 题
  - `D-exercises-medium.md`：中档应用，约 100 题
  - `E-exercises-advanced.md`：压轴 + 中考真题，约 50 题
  - `F-solutions.md`：详解，每题标注所用模型/思路标签（便于反向检索）
- **难度标记**：`☆` / `☆☆` / `☆☆☆`

---

## 6. 实施路线（里程碑划分）

为避免一次性铺开过大，按 part 分里程碑交付。每个里程碑可单独走 `writing-plans` → 执行流程。

| 里程碑 | 内容 | 预计文件数 |
|---|---|---|
| M0 | `README.md` + `00-preface.md` + `thinking-toolkit/` 5 篇 | 7 |
| M1 | part1 + part2 | ~8 |
| M2 | **part3 三角形（重点）** | ~20 |
| M3 | part4 + part5 | ~16 |
| M4 | part6 + part7 | ~10 |
| M5 | part8 + part9 | ~6 |
| M6 | part10 中考综合专题 | ~6 |
| M7 | appendix 题库与详解 | 6 |

**本设计文档只产出 spec。**后续每个里程碑独立生成实施计划与执行。

---

## 7. 不在本设计范围内（YAGNI）

- 不做交互式 web 版（纯 markdown 即可）
- 不做视频/动画
- 不做习题自动判分
- 不覆盖高中几何内容（向量、解析几何拓展、立体几何等）
- 不做英文版

---

## 8. 成功标准

- 任意一个章节，读者读完后能复述"看到什么样的图形 → 应该往哪个方向想"
- 思维工具箱 5 篇可作为独立的"几何方法论"小册子单独成立
- 中考综合专题（part10）能覆盖近 5 年主流省市中考压轴题型
- appendix 题库每题都能反查到对应的模型/思路标签
