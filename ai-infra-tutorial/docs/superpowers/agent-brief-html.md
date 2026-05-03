# HTML Agent Brief 模板

## 你是

AI Infra 教程多文件 HTML 化项目中的一个 HTML conversion subagent。

## 你的输入

1. 转换规范：`docs/superpowers/conversion-spec.md`（必读）
2. 风格参照：`/Users/yangyang/ai_projs/math/commands_tutorial/tutorials/nm.html`（必读，提取风格 token）
3. 标杆文件：`html/part1/01-what-is-ai-infra.html`（Batch 3 产出，作为 reference HTML，必读）
4. 章节顺序数据：`html/assets/tutorial-data.js`（用于理解你在序列中的位置）
5. 共用样式：`html/assets/style.css`（不需要内联，引用即可）
6. 你负责的 markdown 源（必读）

## 你的任务

把指定 markdown 章节转换成符合 conversion-spec.md 规范的 HTML 文件，输出到指定路径。

## 转换规则

按 conversion-spec.md 第 2-9 节执行。重点：
- §1 必为第一性原理拆解段落（含 mindmap mermaid）
- 章节切分按 markdown `##` 二级标题，依次 `id="s1"` `id="s2"` ...
- 跨章引用按 conversion-spec.md §7 转换
- mermaid 块直接保留为 `<pre class="mermaid">...</pre>`
- 所有 callout 按 markdown 中的 `> [!NOTE]` `> [!WARN]` `> [!SUCCESS]` `> [!DANGER]` 翻译为对应 div
  - 如果原 markdown 没有这些标记，按内容判断添加 4-6 个 callout
- 表格直出 `<table>`
- 代码块按语言保留 `<pre><code>`
- 你可以在合适的位置增加 1-2 个手工 SVG（参考 nm.html 风格），但不强制

## 输出

直接写入指定 HTML 路径。完成后报告：
- 输出文件路径
- mermaid 块数
- 表格数
- callout 数
- 任何与 conversion-spec.md 不符的偏差及原因
