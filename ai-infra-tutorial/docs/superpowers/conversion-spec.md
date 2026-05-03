# AI Infra 教程 HTML 转换规范

适用对象：所有 Batch 4 章节 HTML 转换 subagent。

## 1. 文件路径与命名

- 输入 markdown：例如 `part3-training-infra/09-model-pipeline-parallel.md`
- 输出 HTML：`html/part3/09-model-pipeline-parallel.html`
- Part 0 输出：`html/part0/0a-cpu-microarchitecture.html`
- 附录输出：`html/appendix/glossary.html`

## 2. 必须出现的结构元素

每个章节 HTML 必须按下列顺序包含：

1. `<!DOCTYPE html>` + `<html lang="zh-CN">`
2. `<head>` 含 charset / viewport / title（"第 N 章 · 标题 — AI Infra 教程"）/ link assets/style.css
3. `<body class="has-sidebar">`
4. `<iframe id="sidebar" src="../sidebar.html?current=<chapter-id>" class="sidebar-frame" loading="eager"></iframe>`
5. `<main class="page">`
6. `<nav class="topnav"></nav>`（由 nav.js 自动注入 prev/next）
7. `<section class="hero">`：
   - `<h1>第 N 章 · 标题</h1>`
   - `<p class="sub">` 用第一性原理框架表达本章主旨（不是"为什么 AI 工程师要懂..."）
   - `<div class="chips">` 至少 3 个标签 chip
   - `<div class="note"><strong>不可化简的问题：</strong>...</div>`
   - `<div class="success"><strong>本章学习地图：</strong>拆 → 推 → 绘 → 导（详见 §1）</div>`
8. `<section class="toc">` 本章目录列表
9. `<section class="section" id="s1">` 第一性原理拆解段落，必含：
   - `<h2>1. 第一性原理拆解：[本章主题]</h2>`
   - `<h3>拆 — 不可化简的问题</h3>` + 段落
   - `<h3>推 — 从这个问题如何推导出每个机制</h3>` + 段落
   - `<h3>绘 — 因果链路</h3>` + `<pre class="mermaid">mindmap...</pre>`
   - `<h3>导 — 读完本章你应该能回答</h3>` + `<ol>` 5-7 个第一性问题
   - 全段不少于 800 字
10. `<section class="section" id="sN">` 后续各节
11. `<section class="refbox">` 参考资料 + 学习路线 + 延伸阅读
12. `<nav class="bottomnav"></nav>`（由 nav.js 自动注入 prev/up/next）
13. `<footer class="footer">`
14. 末尾三个 script：
    - `<script src="../assets/tutorial-data.js"></script>`
    - `<script src="../assets/mermaid.min.js"></script>`
    - `<script src="../assets/nav.js"></script>`
    - `<script>mermaid.initialize({ startOnLoad: true, theme: 'neutral' });</script>`

## 3. 风格 token

- Card：`<div class="card">`
- Grid：`.grid-2 / .grid-3 / .grid-4`
- Callout：`.note`（蓝/解释）/ `.warn`（橙/边界提醒）/ `.success`（绿/最佳实践）/ `.danger`（红/严重风险）
- 表格：直出 `<table>`，让 style.css 处理样式
- 代码：`<pre><code>...</code></pre>`，行内 `<code>`
- 行内：`.kbd .chip .mini .caption`

## 4. mermaid 用法

- 包成 `<pre class="mermaid">...</pre>`
- 推荐场景：
  - flowchart 流程
  - stateDiagram-v2 状态机（如 MESI / Checkpoint 状态）
  - sequenceDiagram 时序（如 NCCL AllReduce / K8s 调度）
  - mindmap 思维导图（每章 §1 必有 1 张）
  - architecture-beta / C4Context 系统架构
- 每章 4-8 个 mermaid 图（含 §1 必备 mindmap）

## 5. 手工 SVG 用法

- 当 mermaid 表达力不够（菱形决策、四象限对比、视觉隐喻图）
- 包成：

```html
<section class="figure" id="sN">
  <h3>图：标题</h3>
  <svg viewBox="0 0 1100 650" width="100%" role="img" aria-label="...">
    ...（参考 nm.html 风格）
  </svg>
  <div class="caption">说明</div>
</section>
```

## 6. 内容质量底线

- hero 必须有"不可化简的问题" note + "本章学习地图" success 两个 callout
- §1 必为第一性原理拆解段落（≥800 字 + mindmap + 5-7 题 checklist）
- 全章 ≥4 个 mermaid 或手工 SVG（含 §1 mindmap）
- ≥3 个表格
- ≥5 个 callout（混合类型）
- refbox 含"学习路线"+"延伸阅读"

## 7. 跨章引用

- markdown 跨章链接 `[第 9 章](../part3-training-infra/09-...)` → `<a href="../part3/09-model-pipeline-parallel.html">第 9 章</a>`
- 章内 `§9.3` → `<a href="#s3">§9.3</a>`
- Part 0 章节链接：`<a href="../part0/0a-cpu-microarchitecture.html">§0a</a>`

## 8. 风格参照文件

参考 `/Users/yangyang/ai_projs/math/commands_tutorial/tutorials/nm.html` 全文，特别是：
- hero 区结构
- callout 颜色和文案
- 表格样式
- 手工 SVG 流程图风格
- refbox 段落结构

## 9. 不要做的事

- 不要内联 CSS（使用 `../assets/style.css`）
- 不要硬编码 prev/next 链接（由 nav.js 注入）
- 不要重复 sidebar HTML（由 iframe 加载）
- 不要把 markdown 标题字面翻译，要根据内容判断 section 划分（按现有 markdown ## 章节切分通常即可）
- 不要省略第一性原理 §1 段落
