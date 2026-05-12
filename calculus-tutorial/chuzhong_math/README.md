# 中考几何教程

一份强调**思路培养**的中考几何教程。不仅讲"是什么"，更讲"怎么想到的"。

## 这本书写给谁

- 刚接触几何、想从零打地基的初学者
- 已学过课本、但一遇到综合题就卡壳的中等水平学习者
- 准备中考冲刺、想系统拆解压轴题型的考生

## 与普通教辅的不同

普通教辅把定理罗列出来 → 配例题 → 配习题。本书额外强调：

1. **看到一道题，脑子里第一秒应该想什么** —— 把解题者的"内心独白"完整写出来
2. **图形特征 → 模型 → 思路** 的反射弧训练
3. **思维工具箱**：辅助线、模型识别、转化思想，作为可独立阅读的方法论小册子

## 目录

### 方法论（先读这部分）
- [序言：什么是几何思维](00-preface.md)
- [思维工具箱](thinking-toolkit/)
  - [01. 何时该作辅助线](thinking-toolkit/01-when-to-draw-auxiliary-line.md)
  - [02. 常见辅助线总目录](thinking-toolkit/02-auxiliary-line-catalog.md)
  - [03. 模型识别速查](thinking-toolkit/03-model-recognition.md)
  - [04. 转化思想](thinking-toolkit/04-conversion-techniques.md)
  - [05. 开题前的自检清单](thinking-toolkit/05-checklist-before-solving.md)

### 知识章节（按需阅读）
- Part 1：图形初步 ✓
  - [01. 点、线、面、体](part1-foundations/01-point-line-plane.md)
  - [02. 角](part1-foundations/02-angles.md)
  - [03. 角平分线](part1-foundations/03-angle-bisector.md)
  - [04. 余角与补角](part1-foundations/04-complementary-supplementary.md)
- Part 2：相交线与平行线 ✓
  - [01. 对顶角与邻补角](part2-parallel-lines/01-vertical-and-adjacent-angles.md)
  - [02. 垂线、点到直线的距离](part2-parallel-lines/02-perpendicular-line.md)
  - [03. 三线八角](part2-parallel-lines/03-three-line-eight-angles.md)
  - [04. 平行线的判定](part2-parallel-lines/04-parallel-criteria.md)
  - [05. 平行线的性质与应用](part2-parallel-lines/05-parallel-properties.md)
- Part 3：三角形 *(进行中)*
  - basics/
    - [01. 三角形三边关系](part3-triangles/basics/01-triangle-inequality.md)
    - [02. 内角和与外角](part3-triangles/basics/02-angle-sum.md)
    - [03. 三角形的分类](part3-triangles/basics/03-classification.md)
  - congruence/
    - [01. 全等判定 SSS/SAS/ASA/AAS](part3-triangles/congruence/01-sss-sas-asa-aas.md)
    - [02. 直角三角形 HL](part3-triangles/congruence/02-hl-right-triangle.md)
    - [03. 手拉手模型](part3-triangles/congruence/03-model-hand-in-hand.md) ⭐️
    - [04. 半角模型](part3-triangles/congruence/04-model-half-angle.md) ⭐️
    - [05. 一线三等角（K 字模型）](part3-triangles/congruence/05-model-one-line-three-angles.md) ⭐️
    - [06. 倍长中线模型](part3-triangles/congruence/06-model-double-median.md) ⭐️
- Part 4：四边形 *(待写)*
- Part 5：圆 *(待写)*
- Part 6：图形变换 *(待写)*
- Part 7：解直角三角形 *(待写)*
- Part 8：视图与投影 *(待写)*
- Part 9：坐标系中的几何 *(待写)*
- Part 10：中考综合题专题 *(待写)*

### 附录
- A. 公式定理速查 *(待写)*
- B. 模型图集 *(待写)*
- C/D/E. 分级题库 *(待写)*
- F. 题库详解 *(待写)*

## 怎么用这本书

**零基础读者**：按 Part 1 → Part 10 顺序阅读。每读完一个 Part，回到"思维工具箱"对应章节回顾一遍。

**已学过课本的读者**：可以先通读"思维工具箱"5 篇，建立方法论框架；再按需翻阅模型/题型类章节（每个 Part 下带 `model-` 前缀的文件）。

**中考冲刺**：直接看 Part 10「中考综合题专题」，遇到不熟的模型再回查对应章节。题库练 Appendix C → D → E。

## 章节内部结构

本书章节分两种模板：

- **概念/定理类**（如垂径定理、勾股定理）：图形特征 → 结论 → 证明 → 应用 → 易错点 → 自测题
- **模型/题型类**（如手拉手模型、一线三等角）：引入题 → **思维路径还原** → 抽象成模型 → 变形 → **思考路标** → 应用 → 自测题

模型类章节的"思维路径还原"和"思考路标"是本书的特色——它们把高手解题时脑中发生的事完整写出来。

## 关于练习题

- **节内例题**：完整解答，附【思路】小节
- **节末"思路自测题"**：只给提示不给完整解，逼你自己走一遍思维路径
- **附录题库**：基础 / 中档 / 压轴三档，每题在详解中标注所用模型/思路标签

## 关于图

本书的图以两种方式呈现：
- **TikZ 源码**（嵌入 markdown 代码块，可被 LaTeX 渲染器直接出图）
- **预渲染 SVG**（复杂图形，存放于各 part 的 `figures/` 子目录）

## 反馈

本教程是开源项目，欢迎指出错误或提出改进建议。
