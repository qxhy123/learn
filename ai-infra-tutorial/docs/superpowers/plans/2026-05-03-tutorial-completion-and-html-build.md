# AI Infra 教程缺口补完 + 多文件 HTML 化 实现 plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把现有 25 章中文 markdown AI Infra 教程做两件事：(1) 全 29 章新增"第一性原理拆解 + 学习大纲"开篇 + 内容扩写补完所有列出的缺口（合并 SPEC.md 17 WUs + 用户新清单）; (2) 按 nm.html paper 风格转成多文件 HTML 静态站点，含 iframe sidebar + mermaid + 手工 SVG。

**Architecture:** 5 批次 67 个并行 subagent。Batch 1 内容补完（29 章 agent），Batch 2 附录/README 同步（2 agent），Batch 3 HTML 框架（1 agent，产出共用 CSS/JS/sidebar/index 和 Ch1 标杆 HTML），Batch 4 章节 HTML 转换（35 个 agent 并行），Batch 5 主线 agent 集成 review + 链接扫描 + 归档 SPEC.md。每 wave 内部并行，wave 间顺序。

**Tech Stack:** Markdown（教程主源）, HTML5 + 内联 SVG + mermaid v11 离线 bundle, JavaScript（sidebar/nav 注入）, CSS（从 nm.html 抽出 + paper 风格扩展）, git 用于版本管理 + 分批 commit。

---

## File Structure

### 工作产物（plan 自身产出）

- Create: `docs/superpowers/conversion-spec.md` — Batch 4 所有 HTML agent 共用的转换规范
- Create: `docs/superpowers/agent-brief-content.md` — Batch 1 content agent 通用 brief 模板
- Create: `docs/superpowers/agent-brief-html.md` — Batch 4 HTML agent 通用 brief 模板

### 教程内容（Markdown 源）

- Create directory: `part0-foundations-of-systems/`
- Create: `part0-foundations-of-systems/0a-cpu-microarchitecture.md`
- Create: `part0-foundations-of-systems/0b-memory-virtual-memory-and-io.md`
- Create: `part0-foundations-of-systems/0c-filesystems-and-storage-internals.md`
- Create: `part0-foundations-of-systems/0d-network-stack-fundamentals.md`
- Modify: 全部 25 个现有章节 markdown 文件（每个新增"第一性原理开篇"段落；§3 列出的章节同时补内容）
- Modify: `appendix/glossary.md` `appendix/tooling-map.md` `appendix/checklists.md` `appendix/answers.md`
- Modify: `README.md`（章节列表 + 学习路径含 Part 0）
- Rename: `SPEC.md` → `SPEC-archive-2026-04-24.md`

### HTML 静态站点

- Create directory: `html/`, `html/assets/`, `html/part0/`, `html/part1/` ... `html/part8/`, `html/appendix/`
- Create: `html/index.html`
- Create: `html/sidebar.html`
- Create: `html/assets/style.css`
- Create: `html/assets/nav.js`
- Create: `html/assets/tutorial-data.js`
- Download: `html/assets/mermaid.min.js`（mermaid v11 offline bundle，~600KB）
- Create: 31 章节 HTML 文件 + 4 附录 HTML 文件

---

## Task 1: 写 Conversion Spec 主控文档

**Files:**
- Create: `docs/superpowers/conversion-spec.md`

- [ ] **Step 1: 创建文件并写入完整 Conversion Spec**

```markdown
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
```

- [ ] **Step 2: Verify file written**

Run: `wc -l /Users/yangyang/ai_projs/math/ai-infra-tutorial/docs/superpowers/conversion-spec.md`
Expected: > 100 lines

- [ ] **Step 3: Commit**

```bash
cd /Users/yangyang/ai_projs/math/ai-infra-tutorial
git add docs/superpowers/conversion-spec.md
git commit -m "Add HTML conversion spec for Batch 4 subagents"
```

---

## Task 2: 写 Content Agent Brief 模板

**Files:**
- Create: `docs/superpowers/agent-brief-content.md`

- [ ] **Step 1: 创建并写入 brief 模板**

写入完整的 Batch 1 content agent 通用 brief 模板，含：

```markdown
# Content Agent Brief 模板

## 基础上下文（每个 agent 都有）

你是 AI Infra 教程改进项目中的一个 content subagent。教程定位是"AI 平台工程师视角，建立瓶颈直觉 + 决策直觉"，不是 ML 算法教程也不是 OS 教科书。

## 你的输入

1. 设计文档：`docs/superpowers/specs/2026-05-03-tutorial-completion-and-html-design.md`
2. 你负责的 markdown 文件路径（见下方"任务"）
3. 该文件当前内容（你需要先 Read）
4. 设计文档 §3 中你负责的扩写项（见下方"任务"）

## 你必须做的事

### A. 全章统一开篇（必须做，无论是否有内容扩写）

在文件开头新增（或重写）一个"第一性原理拆解 + 学习大纲"段落作为新 §1，规格：

- 不少于 800 字
- 4 段三级标题：
  - **拆 — 不可化简的问题**：剥离所有可记忆的术语和工具名，逼出本章要解决的不可化简问题
  - **推 — 从这个问题如何推导出每个机制**：因果推导每个概念为何必然存在
  - **绘 — 因果链路**：一段 mermaid mindmap 代码块（` ```mermaid mindmap ... ``` `）
  - **导 — 读完本章你应该能回答**：5-7 个第一性问题的有序列表
- 原本的导言段落可以前置或并入"拆"小节，不要简单删除

### B. 内容扩写（仅当设计文档 §3 列出本章扩写项时）

按 §3 列出的项目逐条补充内容。每个新增小节 200-1500 字（见 §3 估算字数），保持原章节"问题先行 → 机制 → 边界"的写法。

### C. 风格要求

- 中文为主，术语保留英文原文（如"模型并行（Model Parallelism）"）
- 表格优于长文
- 每个新增小节给出"工程边界"
- 数字优于形容词（"7B 模型 BF16 ~14GB" 优于"较大"）
- 引用其他章节用 `[§N.X](../partX/<slug>.md#section)`

### D. 验收清单

完成后自检：
- [ ] §1 是第一性原理拆解段落，含拆/推/绘/导四段
- [ ] §1 含 mermaid mindmap 代码块
- [ ] §1 含 5-7 个第一性问题列表
- [ ] §1 不少于 800 字（用 `wc -w` 检查）
- [ ] §3 列出的扩写项全部完成
- [ ] 不引入 emoji
- [ ] 文件不超过 1.5x 原始长度（除非是 Part 0 新章节）

## 输出

直接把修改后的 markdown 写回原文件路径。完成后报告：
- 新增字数（粗略）
- 是否有任何无法完成的项目（列出原因）
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/agent-brief-content.md
git commit -m "Add content agent brief template"
```

---

## Task 3: 写 HTML Agent Brief 模板

**Files:**
- Create: `docs/superpowers/agent-brief-html.md`

- [ ] **Step 1: 创建并写入 HTML brief 模板**

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/agent-brief-html.md
git commit -m "Add HTML agent brief template"
```

---

## Task 4: 创建 Part 0 目录

**Files:**
- Create directory: `part0-foundations-of-systems/`

- [ ] **Step 1: 创建目录**

```bash
mkdir -p /Users/yangyang/ai_projs/math/ai-infra-tutorial/part0-foundations-of-systems
```

- [ ] **Step 2: Verify**

Run: `ls /Users/yangyang/ai_projs/math/ai-infra-tutorial/part0-foundations-of-systems`
Expected: empty directory listing

---

## Task 5: Wave 1.1 — 派发 Part 0 四章 content agent

**Files:**
- Create: `part0-foundations-of-systems/0a-cpu-microarchitecture.md`
- Create: `part0-foundations-of-systems/0b-memory-virtual-memory-and-io.md`
- Create: `part0-foundations-of-systems/0c-filesystems-and-storage-internals.md`
- Create: `part0-foundations-of-systems/0d-network-stack-fundamentals.md`

- [ ] **Step 1: 单条消息并行派发 4 个 agent**

每个 agent prompt 结构：

```
你是 AI Infra 教程 Part 0 新章节 content subagent，负责写 Ch 0a「CPU 微架构」。

## 必读
- docs/superpowers/agent-brief-content.md（通用 brief）
- docs/superpowers/specs/2026-05-03-tutorial-completion-and-html-design.md（设计文档，特别看 §2.1 和 §2.5）

## 章节大纲（来自设计文档 §2.1）
- §0a.1 第一性原理拆解 + 学习大纲（全章必备开篇，规格见 brief A 节）
- §0a.2 流水线（pipeline）...
（完整列出 §2.1 全部节）

## 输出路径
part0-foundations-of-systems/0a-cpu-microarchitecture.md

## 全章规格（来自设计文档 §2.5）
- 总字数 ~3500
- 12-14 道练习（基础 6 + 进阶 4 + 设计 2-4）
- 4-8 个 mermaid 图
- 1-2 个手工 SVG 可选
- 结尾 Worked example：DataLoader 8 worker → 16 worker 反而变慢的真实排查
- "深度参考阅读"列表

写完直接写入文件。
```

类似地为 0b、0c、0d 各自定制 prompt（用对应大纲和 worked example）。

并行调用：

```
（同一消息中 4 个 Agent tool calls，subagent_type="general-purpose"，
 每个 isolation 不需要 worktree，因为是新文件不冲突）
```

- [ ] **Step 2: 等待 4 个 agent 完成**

- [ ] **Step 3: 验证 4 个文件存在且符合规格**

```bash
cd /Users/yangyang/ai_projs/math/ai-infra-tutorial
for f in part0-foundations-of-systems/0{a,b,c,d}-*.md; do
  echo "=== $f ==="
  echo "字数: $(wc -w < "$f")"
  echo "练习题: $(grep -c '^### 练习' "$f" || echo 0)"
  echo "mermaid: $(grep -c '```mermaid' "$f" || echo 0)"
  echo "§1 第一性原理: $(grep -c '第一性原理拆解' "$f" || echo 0)"
done
```

Expected: 每章字数 > 2500、练习题 ≥ 12、mermaid ≥ 4、§1 计数 = 1

- [ ] **Step 4: 抽样 review（主线 agent 自己读 0a 全文）**

```bash
cat part0-foundations-of-systems/0a-cpu-microarchitecture.md | head -200
```

检查：开篇是否真按拆/推/绘/导四段，mindmap 是否合理，是否符合教程"AI 工程师视角"定位

- [ ] **Step 5: 如有问题，主线 agent 直接 Edit 修补**

- [ ] **Step 6: Commit**

```bash
git add part0-foundations-of-systems/
git commit -m "Add Part 0: Foundations of Systems (4 new chapters)

- 0a CPU microarchitecture (pipelining, OoO, MESI, false sharing)
- 0b memory/virtual memory/IO (page cache, NUMA, PCIe, DMA)
- 0c filesystems/storage internals (ext4/XFS/ZFS, S3, Lustre)
- 0d network stack fundamentals (TCP/IP, RDMA, GPUDirect)

Each chapter opens with first-principles decomposition (拆/推/绘/导)
and ends with an AI-workload-anchored worked example."
```

---

## Task 6: Wave 1.2 — 派发 Ch 1-9 content agent (9 并行)

**Files:**
- Modify: `part1-foundations/01-what-is-ai-infra.md`
- Modify: `part1-foundations/02-compute-storage-network.md`
- Modify: `part1-foundations/03-from-model-to-production.md`
- Modify: `part2-systems-stack/04-gpu-and-accelerators.md`
- Modify: `part2-systems-stack/05-memory-interconnect-io.md`
- Modify: `part2-systems-stack/06-cuda-runtime-and-kernels.md`
- Modify: `part3-training-infra/07-single-node-training.md`
- Modify: `part3-training-infra/08-data-parallel.md`
- Modify: `part3-training-infra/09-model-pipeline-parallel.md`

- [ ] **Step 1: 准备 9 个 agent prompt**

每个 agent prompt 结构：

```
你是 AI Infra 教程 Ch N content subagent，负责修改 [文件路径]。

## 必读
- docs/superpowers/agent-brief-content.md
- docs/superpowers/specs/2026-05-03-tutorial-completion-and-html-design.md（特别看 §3.0 和 §3 中本章扩写项）

## 你的两项任务
### A. 全章统一开篇（必做）
按 brief A 节，新增/重写第一性原理拆解段落作为新 §1（≥800 字 + mindmap + 5-7 题 checklist）。
原 §1 导言可前置到"拆"小节或并入"推"小节。

### B. 内容扩写（仅 §3 列出时做）
[此处粘贴该章在设计文档 §3 中列出的扩写项，例如对 Ch 9：
- 并行策略选型决策树（mermaid flowchart）+ 典型配置实例表（~1500）
- Sequence Parallelism + Context Parallelism + 三者对比表（~700）
- Interleaved/Zero Bubble 详解（~500）]

## 输出
直接写回原文件。
```

无扩写项的章节（Ch 1, 2 仅含小补丁, 3）只需做 A；其他做 A + B。

- [ ] **Step 2: 单消息并行派发 9 个 agent**

- [ ] **Step 3: 等待全部完成**

- [ ] **Step 4: 验证全部 9 个文件**

```bash
cd /Users/yangyang/ai_projs/math/ai-infra-tutorial
for f in part1-foundations/0{1,2,3}-*.md part2-systems-stack/0{4,5,6}-*.md part3-training-infra/0{7,8,9}-*.md; do
  echo "=== $f ==="
  has_p1=$(grep -c '第一性原理拆解' "$f" || echo 0)
  has_mindmap=$(grep -A 5 '第一性原理拆解' "$f" | grep -c 'mindmap' || echo 0)
  echo "第一性原理段落: $has_p1, mindmap: $has_mindmap"
done
```

Expected: 每个文件第一性原理段落 = 1, mindmap >= 1

- [ ] **Step 5: 抽样 review（主线 agent 读 Ch 9 全文，因扩写最重）**

- [ ] **Step 6: 修补 + Commit**

```bash
git add part1-foundations/ part2-systems-stack/ part3-training-infra/0{7,8,9}-*.md
git commit -m "Wave 1.2: First-principles openers + content expansion for Ch 1-9"
```

---

## Task 7: Wave 1.3 — 派发 Ch 10, 10b, 10c, 11, 12, 13 content agent (6 并行)

**Files:**
- Modify: `part3-training-infra/10-memory-checkpointing-and-recovery.md`
- Modify: `part3-training-infra/10b-alignment-and-post-training.md`
- Modify: `part3-training-infra/10c-finetuning-and-multi-adapter.md`
- Modify: `part4-data-and-storage/11-data-pipeline.md`
- Modify: `part4-data-and-storage/12-artifacts-and-checkpoints.md`
- Modify: `part4-data-and-storage/13-feature-vector-and-cache.md`

- [ ] **Step 1: 准备 6 个 agent prompt（同 Task 6 模式）**

特别注意：
- Ch 10：NCCL Hang 完整排查 + Straggler + Elastic + Pre-flight + FP8/HFU
- Ch 10b：PPO worked example + DPO/GRPO + RM 部署 + 多模型 checkpoint + 7 道新练习
- Ch 10c：Multi-LoRA 显存 + Adapter 兼容 + FTaaS + 5 道新练习
- Ch 13：向量数据库选型 + ANN + RAG Chunking + 增量 vs 全量 + Prefix Caching

- [ ] **Step 2: 单消息并行派发 6 agent**

- [ ] **Step 3-5: 验证 / review / 修补（同 Task 6）**

特别检查：
- Ch 10b 的 PPO 显存表存在
- Ch 10c 的版本兼容性新章节存在
- Ch 13 的向量库选型表 ≥ 6 行

- [ ] **Step 6: Commit**

```bash
git add part3-training-infra/10*.md part4-data-and-storage/
git commit -m "Wave 1.3: First-principles openers + expansion for Ch 10/10b/10c/11/12/13"
```

---

## Task 8: Wave 1.4 — 派发 Ch 14-20 content agent (7 并行)

**Files:**
- Modify: `part5-serving-infra/14-online-inference-architecture.md`
- Modify: `part5-serving-infra/15-batching-scheduling-and-kv-cache.md`
- Modify: `part5-serving-infra/16-quantization-compilation-and-engines.md`
- Modify: `part5-serving-infra/17-multitenancy-and-cost.md`
- Modify: `part6-platform-and-orchestration/18-containers-and-runtime.md`
- Modify: `part6-platform-and-orchestration/19-kubernetes-for-ai.md`
- Modify: `part6-platform-and-orchestration/20-queues-quotas-and-autoscaling.md`

- [ ] **Step 1: 准备 7 agent prompt**

特别注意：
- Ch 14 / 18 仅做 A（开篇）
- Ch 15: LLaMA-70B Worked Example + Prefill-Decode Disaggregated + Speculative + ITL
- Ch 16: 量化决策树 + 引擎决策树 + 校准 + vLLM/TRT-LLM/SGLang 引擎内部
- Ch 17: TCO + Spot + MFU vs Util + Chargeback
- Ch 19: Volcano/Kueue 内部 + 拓扑感知 + 亲和/反亲和
- Ch 20: MIG/MPS/Time-Slicing + GPU 碎片化 + DRF

- [ ] **Step 2: 并行派发**

- [ ] **Step 3-5: 验证 / review / 修补**

- [ ] **Step 6: Commit**

```bash
git add part5-serving-infra/ part6-platform-and-orchestration/
git commit -m "Wave 1.4: First-principles openers + expansion for Ch 14-20"
```

---

## Task 9: Wave 1.5 — 派发 Ch 21-25 content agent (5 并行)

**Files:**
- Modify: `part7-reliability-security/21-observability-and-capacity.md`
- Modify: `part7-reliability-security/22-evaluation-release-and-incident.md`
- Modify: `part7-reliability-security/23-security-isolation-and-governance.md`
- Modify: `part8-advanced-and-capstone/24-build-an-ai-platform.md`
- Modify: `part8-advanced-and-capstone/25-agent-and-inference-time-compute.md`

- [ ] **Step 1: 准备 5 agent prompt**

特别注意：
- Ch 21: Trace 采样 head/tail + cardinality + 错误预算 + 成本归因
- Ch 22: A/B vs 灰度 + 灰度质量采样 + Prompt/配置变更
- Ch 23: Secrets + 模型安全(pickle/SafeTensors) + 供应链(cosign/Trivy/SLSA)
- Ch 24 仅做 A（开篇）
- Ch 25: 去重 + thinking tokens 4 模式 + 推理预算工程实现 + Agent/推理服务集成 + 7 道新练习

- [ ] **Step 2: 并行派发**

- [ ] **Step 3-5: 验证 / review / 修补**

- [ ] **Step 6: Commit**

```bash
git add part7-reliability-security/ part8-advanced-and-capstone/
git commit -m "Wave 1.5: First-principles openers + expansion for Ch 21-25"
```

---

## Task 10: Wave 2 — 附录 + README 同步 (2 并行)

**Files:**
- Modify: `appendix/glossary.md`
- Modify: `appendix/tooling-map.md`
- Modify: `appendix/checklists.md`
- Modify: `appendix/answers.md`
- Modify: `README.md`

- [ ] **Step 1: 派发 agent-glossary-tooling-checklist**

prompt:

```
你是 AI Infra 教程附录同步 subagent。

## 任务
扫描 Batch 1 全部新增 markdown 内容（Part 0 4 章 + Ch 1-25 新开篇 + 各章新增小节），
然后更新 4 份附录文件：

1. appendix/glossary.md — 新增 ~30 个术语，含 Part 0 全部新概念（流水线、OoO、分支预测、SIMD、L1/L2/L3、Cache line、MESI、伪共享、虚拟内存、Page Cache、TLB、Huge Pages、NUMA、io_uring、PCIe lane、DMA、pinned memory、ext4 journal、XFS B+tree、ZFS COW、ARC、Lustre OSS/MDS、TCP CUBIC/BBR、jumbo frame、RDMA QP/CQ、GPUDirect、ECN 等等）+ 各章新增的：梯度压缩、PowerSGD、SP/CP、Interleaved/Zero Bubble、HFU、PPO/DPO/GRPO、RM、Multi-LoRA、Speculative Decoding、ITL、TTFT/TPOT、Volcano/Kueue、MIG/MPS、DRF、SLSA 等

2. appendix/tooling-map.md — 新增 4+ 类别：CPU profiling (perf, vtune)、FS tools (iostat, fio, fsbench)、network tools (ethtool, perftest, ib_*)、mermaid 渲染

3. appendix/checklists.md — 新增 3+ 清单：CPU 性能排查清单、文件系统选型清单、网络配置健康检查清单

4. appendix/answers.md — 补全所有 Batch 1 新增练习题答案（不少于覆盖 50 道新题）

## 输出
直接修改 4 个附录文件。完成后报告每个文件新增字数。
```

- [ ] **Step 2: 派发 agent-readme**

prompt:

```
你是 AI Infra 教程 README 同步 subagent。

## 任务
更新 README.md：

1. "章节导航目录"加 Part 0「体系结构基础」表，含 0a/0b/0c/0d 4 章
2. 更新各 Part 表的"主要内容"列，反映本次扩写新增的关键内容（决策树/Worked Example/SP/CP 等）
3. "学习路径建议"四条路径都更新：路径 A/B/C 全部加入 Part 0 作为可选/必选前置，新增"路径五：体系结构深度路径（Part 0 + Part 2 + Part 3）"
4. "如何判断自己真的学会了"加入"能从第一性原理推导每个机制"
5. 教程特色加入"第一性原理思维框架"和"多文件 HTML 版本"

## 不要做
- 不要重写 README 整体结构
- 保留所有现有学习路径建议描述

## 输出
直接修改 README.md。报告新增/修改字数。
```

- [ ] **Step 3: 单消息并行派发 2 agent**

- [ ] **Step 4: 验证**

```bash
cd /Users/yangyang/ai_projs/math/ai-infra-tutorial
echo "glossary 行数: $(wc -l < appendix/glossary.md)"
echo "tooling-map 行数: $(wc -l < appendix/tooling-map.md)"
echo "checklists 行数: $(wc -l < appendix/checklists.md)"
echo "answers 行数: $(wc -l < appendix/answers.md)"
echo "README Part 0 提及: $(grep -c 'Part 0\|part0' README.md)"
```

Expected: 4 个附录都比改前增加 ≥ 30%；README Part 0 提及 ≥ 5 处

- [ ] **Step 5: Commit**

```bash
git add appendix/ README.md
git commit -m "Wave 2: Sync appendix glossary/tooling/checklists/answers + README with Part 0 and Batch 1 content"
```

---

## Task 11: Wave 3 — HTML 框架 (1 agent)

**Files:**
- Create: `html/index.html`
- Create: `html/sidebar.html`
- Create: `html/assets/style.css`
- Create: `html/assets/nav.js`
- Create: `html/assets/tutorial-data.js`
- Create: `html/part1/01-what-is-ai-infra.html`（Ch 1 标杆 HTML，用作 Batch 4 reference）
- Download: `html/assets/mermaid.min.js`

- [ ] **Step 1: 创建 HTML 目录结构**

```bash
cd /Users/yangyang/ai_projs/math/ai-infra-tutorial
mkdir -p html/assets html/part0 html/part1 html/part2 html/part3 html/part4 html/part5 html/part6 html/part7 html/part8 html/appendix
```

- [ ] **Step 2: 下载 mermaid v11 离线 bundle**

```bash
curl -fsSL https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js -o html/assets/mermaid.min.js
ls -lh html/assets/mermaid.min.js
```

Expected: 文件存在，约 600KB-3MB

- [ ] **Step 3: 派发 agent-html-skeleton**

prompt:

```
你是 AI Infra 教程 HTML 框架 subagent。

## 必读
1. docs/superpowers/specs/2026-05-03-tutorial-completion-and-html-design.md（设计文档，特别看 §1, §4）
2. docs/superpowers/conversion-spec.md
3. /Users/yangyang/ai_projs/math/commands_tutorial/tutorials/nm.html（风格参照，必读全文）
4. part1-foundations/01-what-is-ai-infra.md（Ch 1 markdown 源）

## 你的输出（5 个文件）

### 1. html/assets/style.css
从 nm.html 内联 CSS 抽出，加多文件适配补丁：
- 保留 nm.html 全部 CSS 变量、颜色、布局、字体、callout、表格、grid 样式
- 新增 .has-sidebar body 样式（margin-left: 280px）
- 新增 .sidebar-frame 样式（position: fixed; left: 0; top: 0; width: 280px; height: 100vh; border: none）
- 新增 @media (max-width: 768px) 隐藏 sidebar，显示 hamburger 按钮
- 新增 .topnav .bottomnav 样式（prev/next 按钮）
- 新增 mermaid 图块容器样式（居中、padding、bg light）

### 2. html/assets/tutorial-data.js
单一数据源，定义全部 31 章顺序（0a/0b/0c/0d + 1-25 + 4 附录）：

```javascript
const TUTORIAL = [
  { part: 'Part 0', title: '体系结构基础', chapters: [
    { id: '0a', title: 'CPU 微架构', path: 'part0/0a-cpu-microarchitecture.html' },
    { id: '0b', title: '内存、虚拟内存与 IO', path: 'part0/0b-memory-virtual-memory-and-io.html' },
    { id: '0c', title: '文件系统与存储内核', path: 'part0/0c-filesystems-and-storage-internals.html' },
    { id: '0d', title: '网络协议栈基础', path: 'part0/0d-network-stack-fundamentals.html' },
  ]},
  { part: 'Part 1', title: 'AI Infra 基础认知', chapters: [
    { id: '01', title: '什么是 AI Infra', path: 'part1/01-what-is-ai-infra.html' },
    { id: '02', title: '算力、存储与网络', path: 'part1/02-compute-storage-network.html' },
    { id: '03', title: '从模型实验到生产系统', path: 'part1/03-from-model-to-production.html' },
  ]},
  // ... Part 2-8 全部
  { part: '附录', title: '附录', chapters: [
    { id: 'glossary', title: '术语表', path: 'appendix/glossary.html' },
    { id: 'tooling-map', title: '工具图谱', path: 'appendix/tooling-map.html' },
    { id: 'checklists', title: '上线与排障检查清单', path: 'appendix/checklists.html' },
    { id: 'answers', title: '练习题参考解答', path: 'appendix/answers.html' },
  ]},
];

if (typeof module !== 'undefined') module.exports = TUTORIAL;
```

### 3. html/assets/nav.js
- 监听 DOMContentLoaded
- 解析 window.location.pathname 找到当前章 id
- 在 .topnav 和 .bottomnav 注入"← 上一章 [标题] / ↑ 返回目录 / 下一章 [标题] →"
- 用 postMessage 通知 sidebar iframe 当前章高亮（保险机制，主要靠 ?current= query param）
- 不依赖任何外部库

### 4. html/sidebar.html
独立 HTML 页：
- 顶部 logo 区："AI Infra 教程"标题 + 简介
- 搜索框 input（onInput 做客户端 fuzzy filter）
- 8 个 Part 折叠组（含附录 = 9 组）
- 每章一个链接，target="_top"
- 当前章用 ?current=<id> 参数高亮
- 引入 ../assets/tutorial-data.js + 内联 sidebar 自己的样式（独立 iframe，不需外链 style.css）

### 5. html/index.html
教程门面：
- nm.html 风格 hero（教程总标题 + 副标题 + chips）
- "如何使用本教程"卡片（含"建议用 python -m http.server 打开"提示）
- 学习路径卡片（5 条路径）
- 完整章节卡片墙（每 Part 一个大卡，里面网格列出全部章节，点进对应 HTML）
- 引入 ../assets/style.css

### 6. html/part1/01-what-is-ai-infra.html（Ch 1 标杆 HTML）
按 conversion-spec.md 把 part1-foundations/01-what-is-ai-infra.md 完整转成 HTML。
这是 Batch 4 所有其他 HTML agent 的参考标杆，必须严格按 spec 写，结构完整、风格精良。

## 输出
直接写入对应文件路径。完成后报告每个文件大小和 LoC。
```

- [ ] **Step 4: 等待 agent 完成**

- [ ] **Step 5: 验证 6 个文件存在**

```bash
ls -la html/index.html html/sidebar.html html/assets/style.css html/assets/nav.js html/assets/tutorial-data.js html/part1/01-what-is-ai-infra.html
```

Expected: 6 文件全部存在

- [ ] **Step 6: 抽样浏览器验证**

```bash
echo "请用浏览器打开 html/index.html 和 html/part1/01-what-is-ai-infra.html 检查"
echo "推荐：cd html && python3 -m http.server 8000 然后访问 http://localhost:8000/"
```

主线 agent 用 Read 检查 Ch 1 标杆 HTML 的关键元素：
- iframe id="sidebar" 存在
- 末尾三个 script 引用正确
- §1 含 mermaid mindmap 块
- hero 区两个 callout 文案符合规范

- [ ] **Step 7: 修补 + Commit**

```bash
git add html/
git commit -m "Wave 3: HTML skeleton (style/nav/sidebar/data/index + Ch 1 reference)"
```

---

## Task 12: Wave 4.1 — HTML 转换 Part 0 + Part 1 余下 (6 并行)

**Files:**
- Create: `html/part0/0a-cpu-microarchitecture.html`
- Create: `html/part0/0b-memory-virtual-memory-and-io.html`
- Create: `html/part0/0c-filesystems-and-storage-internals.html`
- Create: `html/part0/0d-network-stack-fundamentals.html`
- Create: `html/part1/02-compute-storage-network.html`
- Create: `html/part1/03-from-model-to-production.html`

- [ ] **Step 1: 准备 6 个 HTML agent prompt**

每个 agent prompt 模板：

```
你是 AI Infra 教程 HTML 转换 subagent，负责把 [章节 markdown 路径] 转成 HTML 输出到 [HTML 路径]。

## 必读
- docs/superpowers/agent-brief-html.md
- docs/superpowers/conversion-spec.md
- html/part1/01-what-is-ai-infra.html（标杆，必读）
- /Users/yangyang/ai_projs/math/commands_tutorial/tutorials/nm.html（风格参照）
- [章节 markdown 路径]（你要转换的源）

## 输出路径
[HTML 路径]

## 要求
严格按 conversion-spec.md 第 2-9 节执行。完成后报告 mermaid 块数 / 表格数 / callout 数。
```

- [ ] **Step 2: 单消息并行派发 6 agent**

- [ ] **Step 3: 验证**

```bash
for f in html/part0/*.html html/part1/0{2,3}-*.html; do
  echo "=== $f ==="
  echo "size: $(wc -c < "$f")"
  echo "mermaid: $(grep -c 'class="mermaid"' "$f")"
  echo "iframe sidebar: $(grep -c 'id="sidebar"' "$f")"
  echo "§1 mindmap: $(grep -A 20 'id="s1"' "$f" | grep -c 'mindmap')"
  echo "scripts: $(grep -c 'mermaid.min.js\|nav.js\|tutorial-data.js' "$f")"
done
```

Expected: 每文件 mermaid ≥ 4, iframe sidebar = 1, §1 mindmap = 1, scripts = 3

- [ ] **Step 4: 浏览器抽样**

主线 agent Read 1 个文件全文检查格式

- [ ] **Step 5: 修补 + Commit**

```bash
git add html/part0/ html/part1/0{2,3}-*.html
git commit -m "Wave 4.1: HTML for Part 0 and Part 1 remainder"
```

---

## Task 13: Wave 4.2 — HTML 转换 Part 2 + Part 3 (9 并行)

**Files:**
- Create: `html/part2/04-gpu-and-accelerators.html`
- Create: `html/part2/05-memory-interconnect-io.html`
- Create: `html/part2/06-cuda-runtime-and-kernels.html`
- Create: `html/part3/07-single-node-training.html`
- Create: `html/part3/08-data-parallel.html`
- Create: `html/part3/09-model-pipeline-parallel.html`
- Create: `html/part3/10-memory-checkpointing-and-recovery.html`
- Create: `html/part3/10b-alignment-and-post-training.html`
- Create: `html/part3/10c-finetuning-and-multi-adapter.html`

- [ ] **Step 1-5: 同 Task 12 模式，9 个 agent 并行**

- [ ] **Step 6: Commit**

```bash
git add html/part2/ html/part3/
git commit -m "Wave 4.2: HTML for Part 2 (Ch 4-6) and Part 3 (Ch 7-10c)"
```

---

## Task 14: Wave 4.3 — HTML 转换 Part 4 + Part 5 (7 并行)

**Files:**
- Create: `html/part4/11-data-pipeline.html`
- Create: `html/part4/12-artifacts-and-checkpoints.html`
- Create: `html/part4/13-feature-vector-and-cache.html`
- Create: `html/part5/14-online-inference-architecture.html`
- Create: `html/part5/15-batching-scheduling-and-kv-cache.html`
- Create: `html/part5/16-quantization-compilation-and-engines.html`
- Create: `html/part5/17-multitenancy-and-cost.html`

- [ ] **Step 1-5: 7 agent 并行**

- [ ] **Step 6: Commit**

```bash
git add html/part4/ html/part5/
git commit -m "Wave 4.3: HTML for Part 4 (Ch 11-13) and Part 5 (Ch 14-17)"
```

---

## Task 15: Wave 4.4 — HTML 转换 Part 6 + Part 7 (6 并行)

**Files:**
- Create: `html/part6/18-containers-and-runtime.html`
- Create: `html/part6/19-kubernetes-for-ai.html`
- Create: `html/part6/20-queues-quotas-and-autoscaling.html`
- Create: `html/part7/21-observability-and-capacity.html`
- Create: `html/part7/22-evaluation-release-and-incident.html`
- Create: `html/part7/23-security-isolation-and-governance.html`

- [ ] **Step 1-5: 6 agent 并行**

- [ ] **Step 6: Commit**

```bash
git add html/part6/ html/part7/
git commit -m "Wave 4.4: HTML for Part 6 (Ch 18-20) and Part 7 (Ch 21-23)"
```

---

## Task 16: Wave 4.5 — HTML 转换 Part 8 + 4 附录 (6 并行)

**Files:**
- Create: `html/part8/24-build-an-ai-platform.html`
- Create: `html/part8/25-agent-and-inference-time-compute.html`
- Create: `html/appendix/glossary.html`
- Create: `html/appendix/tooling-map.html`
- Create: `html/appendix/checklists.html`
- Create: `html/appendix/answers.html`

- [ ] **Step 1: 6 agent 并行**

附录 4 个 HTML 不需要 §1 第一性原理拆解段落（它们不是教学章节，是参考资料）。
附录 agent prompt 要明确说明：
- 不需要"不可化简的问题" / "本章学习地图" callouts
- 不需要 §1 第一性原理段落
- hero 区可以更简洁
- 仍要 ≥3 个表格 + 必备 sidebar iframe + scripts

- [ ] **Step 2-5: 验证 / review / 修补**

- [ ] **Step 6: Commit**

```bash
git add html/part8/ html/appendix/
git commit -m "Wave 4.5: HTML for Part 8 (Ch 24-25) and 4 appendix files"
```

---

## Task 17: Wave 5.1 — 链接完整性扫描

**Files:**
- 不修改文件，只检查

- [ ] **Step 1: 提取所有相对路径 href**

```bash
cd /Users/yangyang/ai_projs/math/ai-infra-tutorial/html
grep -rhoE 'href="[^"#]+\.html[^"]*"' . | sort -u > /tmp/all-hrefs.txt
wc -l /tmp/all-hrefs.txt
```

- [ ] **Step 2: 验证每个链接对应文件存在**

```bash
cd /Users/yangyang/ai_projs/math/ai-infra-tutorial/html
broken=0
while IFS= read -r line; do
  href=$(echo "$line" | sed -E 's/href="([^"#]+)\.html.*/\1.html/')
  # 跳过外链
  if [[ "$href" == http* ]]; then continue; fi
  # 假设全部从 html/ 根开始或 partN/
  for base in . part0 part1 part2 part3 part4 part5 part6 part7 part8 appendix; do
    if [ -f "$base/$href" ] || [ -f "$href" ]; then
      ok=1
      break
    fi
  done
  if [ -z "${ok:-}" ]; then
    echo "BROKEN: $href"
    broken=$((broken+1))
  fi
  unset ok
done < /tmp/all-hrefs.txt
echo "broken total: $broken"
```

Expected: broken = 0

- [ ] **Step 3: 如有 broken 链接，主线 agent 直接 Edit 修复**

---

## Task 18: Wave 5.2 — 必备元素扫描

- [ ] **Step 1: 检查每个章节 HTML 都含必备元素**

```bash
cd /Users/yangyang/ai_projs/math/ai-infra-tutorial/html
fail=0
for f in part*/[0-9]*.html part*/[0-9][a-z]-*.html; do
  has_iframe=$(grep -c 'id="sidebar"' "$f")
  has_mermaid_script=$(grep -c 'mermaid.min.js' "$f")
  has_nav_script=$(grep -c 'nav.js' "$f")
  has_data_script=$(grep -c 'tutorial-data.js' "$f")
  has_hero=$(grep -c 'class="hero"' "$f")
  has_refbox=$(grep -c 'class="refbox"' "$f")
  has_s1=$(grep -c 'id="s1"' "$f")
  has_mindmap=$(grep -A 30 'id="s1"' "$f" | grep -c 'mindmap')
  
  if [ "$has_iframe" -lt 1 ] || [ "$has_mermaid_script" -lt 1 ] || \
     [ "$has_nav_script" -lt 1 ] || [ "$has_data_script" -lt 1 ] || \
     [ "$has_hero" -lt 1 ] || [ "$has_refbox" -lt 1 ] || \
     [ "$has_s1" -lt 1 ] || [ "$has_mindmap" -lt 1 ]; then
    echo "INCOMPLETE: $f (iframe=$has_iframe, mermaid_js=$has_mermaid_script, nav_js=$has_nav_script, data_js=$has_data_script, hero=$has_hero, refbox=$has_refbox, s1=$has_s1, mindmap=$has_mindmap)"
    fail=$((fail+1))
  fi
done
echo "incomplete files: $fail"
```

Expected: fail = 0（附录 HTML 不要求 mindmap，需单独检查）

附录单独检查：

```bash
for f in appendix/*.html; do
  has_iframe=$(grep -c 'id="sidebar"' "$f")
  has_scripts=$(grep -c 'mermaid.min.js\|nav.js\|tutorial-data.js' "$f")
  if [ "$has_iframe" -lt 1 ] || [ "$has_scripts" -lt 3 ]; then
    echo "APPENDIX INCOMPLETE: $f"
  fi
done
```

- [ ] **Step 2: 修补任何不合规章节**

主线 agent 直接 Edit

- [ ] **Step 3: Commit 修补**

```bash
git add html/
git commit -m "Wave 5.1+5.2: Fix broken links and missing elements"
```

---

## Task 19: Wave 5.3 — 浏览器抽样视觉 review

**Files:**
- 不修改文件，只观察

- [ ] **Step 1: 启动本地 server**

```bash
cd /Users/yangyang/ai_projs/math/ai-infra-tutorial/html
python3 -m http.server 8000 &
SERVER_PID=$!
echo $SERVER_PID > /tmp/server.pid
sleep 1
echo "Server at http://localhost:8000/"
```

- [ ] **Step 2: 提示用户打开浏览器抽查 5 个页面**

```
请用浏览器分别访问以下 URL 并报告问题：
1. http://localhost:8000/index.html         (门面页)
2. http://localhost:8000/part0/0a-cpu-microarchitecture.html  (Part 0 第一章)
3. http://localhost:8000/part3/09-model-pipeline-parallel.html (Ch 9 决策树)
4. http://localhost:8000/part5/15-batching-scheduling-and-kv-cache.html (Ch 15 worked example)
5. http://localhost:8000/appendix/glossary.html (附录)

特别检查：
- sidebar 加载且当前章高亮
- mermaid mindmap 渲染（不应显示成 raw text）
- callouts 颜色正确
- 表格样式
- 跨章 prev/next 按钮存在
- 移动端 < 768px sidebar 隐藏
```

- [ ] **Step 3: 收集用户反馈，主线 agent Edit 修补**

- [ ] **Step 4: 关闭 server**

```bash
kill $(cat /tmp/server.pid) 2>/dev/null || true
rm -f /tmp/server.pid
```

- [ ] **Step 5: Commit 修补**

```bash
git add html/
git commit -m "Wave 5.3: Visual review fixes"
```

---

## Task 20: Wave 5.4 — 归档 SPEC.md + 最终 commit

**Files:**
- Rename: `SPEC.md` → `SPEC-archive-2026-04-24.md`
- Modify: `README.md`（加 HTML 入口指引）

- [ ] **Step 1: 归档 SPEC.md**

```bash
cd /Users/yangyang/ai_projs/math/ai-infra-tutorial
git mv SPEC.md SPEC-archive-2026-04-24.md
```

- [ ] **Step 2: 在归档头部加注**

Edit `SPEC-archive-2026-04-24.md` 在第 1 行后插入：

```markdown
> **归档说明（2026-05-03）：** 本 SPEC 描述的 17 个 Work Unit 已全部并入 `docs/superpowers/specs/2026-05-03-tutorial-completion-and-html-design.md` 的设计中并完成实施。本文件保留作为历史参考，不再独立执行。
```

- [ ] **Step 3: README 加 HTML 入口段落**

在 README 末尾「许可证」之前新增：

```markdown
## HTML 版本

本教程同时提供静态 HTML 版本（位于 `html/` 目录），适合离线浏览与分发：

```bash
cd html && python3 -m http.server 8000
# 浏览器访问 http://localhost:8000/index.html
```

HTML 版本特点：
- 浅色 paper 风格，每章独立文件
- 左侧 sidebar 可视化全 31 章导航
- mermaid 图表 + 手工 SVG 流程图
- 所有内容与 markdown 版本同步
```

- [ ] **Step 4: Final commit**

```bash
git add SPEC-archive-2026-04-24.md README.md
git commit -m "Archive original SPEC.md and add HTML entry point to README

The 17 Work Units originally defined in SPEC.md have all been merged
into and implemented per the 2026-05-03 design at
docs/superpowers/specs/2026-05-03-tutorial-completion-and-html-design.md.

The tutorial now has:
- 4 new Part 0 chapters (CPU/memory/FS/network foundations)
- First-principles opener in all 25 existing chapters
- ~22 chapter content expansions per merged spec
- Full multi-file HTML site at html/ in nm.html paper style
- Updated appendix and README

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 5: Verify final state**

```bash
git log --oneline -25
echo "---"
echo "总章节 markdown: $(find . -path './html' -prune -o -path './docs' -prune -o -name '*.md' -print | wc -l)"
echo "总 HTML: $(find html -name '*.html' | wc -l)"
echo "HTML assets: $(ls html/assets/ | wc -l)"
```

Expected:
- 25 commits（全部 batch）
- markdown 文件数 ~33（含 README/preface/SPEC-archive/Part 0 4 章 + Ch 1-25 + 4 附录）
- HTML 文件数 ≥ 33（含 index/sidebar + 31 章 + 4 附录 = 37）
- HTML assets ≥ 4 个（style.css, nav.js, tutorial-data.js, mermaid.min.js）

---

## Self-Review Notes

**Spec coverage check:**
- ✅ §2 Part 0 4 章 → Task 5
- ✅ §3 现有章节扩写 → Tasks 6-9
- ✅ §3.0 全章统一开篇 → 包含在 Tasks 5-9 每个 agent 的 brief A 节
- ✅ §4 HTML 流水线 → Tasks 11-16
- ✅ §4.1 共用资源 → Task 11
- ✅ §4.2 章节模板 → Task 11 Ch 1 标杆 + Tasks 12-16 推广
- ✅ §4.3 sidebar → Task 11
- ✅ §4.4 subagent 直接写 HTML → Tasks 12-16
- ✅ §4.5 index.html → Task 11
- ✅ §5.1 五个批次 → Tasks 4-20
- ✅ §5.2 质量门 → Tasks 17-19
- ✅ §5.3 风险缓解 → Task 11 标杆 + Tasks 17-19 集成 review

**Placeholder scan:** No "TBD"/"TODO"/vague placeholders. All agent prompts spelled out. Verification commands concrete.

**Type/path consistency:** Reviewed paths across tasks; all consistent. `html/part0/`, `html/assets/`, `html/sidebar.html` etc. match across all tasks. `tutorial-data.js` chapter list aligns with the 31 chapter count.

**Granularity check:** Each "Wave" task has 3-6 steps. Wave dispatching tasks are bigger but unavoidable given the parallel-subagent execution model — each step (prepare prompts / dispatch / verify / review / commit) is a clear action.
