# AI Infra Interview Questions Chapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Chapter 26 as an interview-ready AI Infra question bank that supports candidate prep, self-test, and interviewer evaluation, then wire it into the tutorial navigation and published HTML mirror.

**Architecture:** Keep Markdown as the canonical source for the new chapter. Mirror the chapter into one static HTML page using the existing `html/part8/24` and `25` pages as format references, and wire navigation through `README.md` plus `html/assets/tutorial-data.js`. There is no repo-local build pipeline for this tutorial, so the HTML page is a hand-authored publish artifact rather than a generated output.

**Tech Stack:** Markdown, HTML5, existing tutorial CSS/JS assets (`html/assets/nav.js`, `html/assets/tutorial-data.js`), shell, `rg`, `git`, browser verification.

---

## Source References

- Spec: `docs/superpowers/specs/2026-05-07-ai-infra-interview-questions-design.md`
- Tutorial source root: `README.md`, `part8-advanced-and-capstone/`, `html/`
- Reference chapter HTML: `html/part8/25-agent-and-inference-time-compute.html`
- Navigation data: `html/assets/tutorial-data.js`

## File Structure

Read:

- `docs/superpowers/specs/2026-05-07-ai-infra-interview-questions-design.md`
- `README.md`
- `html/assets/tutorial-data.js`
- `html/part8/24-build-an-ai-platform.html`
- `html/part8/25-agent-and-inference-time-compute.html`
- `html/assets/nav.js`
- `html/sidebar.html`

Create:

- `part8-advanced-and-capstone/26-ai-infra-interview-questions.md`
- `html/part8/26-ai-infra-interview-questions.html`

Modify:

- `README.md`
- `html/assets/tutorial-data.js`

Do not touch:

- `html/sidebar.html`
- `html/assets/nav.js`
- `appendix/answers.md`
- any unrelated work under `code/mini-vllm/`

---

## Task 1: Write the Chapter 26 Markdown Source

**Files:**
- Create: `part8-advanced-and-capstone/26-ai-infra-interview-questions.md`

- [ ] **Step 1: Create the chapter scaffold and opening sections**

Write the full chapter header and opening block first, using the same tutorial style as chapters 24 and 25, but adapted to an interview bank. The opening must include:

- `# 第26章：AI Infra 面试题、自测与面试官题库`
- a short opening quote
- a related chapters line pointing to chapters 24 and 25
- `## 1. 第一性原理拆解 + 学习大纲`
- `## 2. 学习目标`
- `## 3. 使用方式`
- `## 4. 题目格式约定`

Use this exact question block shape once, then repeat it across all questions:

```markdown
### 26.1.1 AI Infra 面试到底在考什么

**问题**
说明 AI Infra 面试的核心考点，不要只列组件名，要区分资源、链路、故障和治理。

**考察点**
- 是否能从系统而不是名词出发
- 是否能讲清资源和约束
- 是否能把诊断、设计和治理串起来

**回答框架**
- 先定义 AI Infra 面试考什么
- 再给出四条判断线
- 最后举一个训练或推理的例子

**追问**
- 为什么“会用组件”不等于“会做系统”？
- 训练岗和推理岗的回答侧重点有什么不同？

**评分要点**
- 及格：能讲出资源和系统边界
- 良好：能给出一条完整的判断路径
- 优秀：能结合真实故障或设计案例展开
```

The opening section should make one clear point: AI Infra interviews test system judgment, not memorized component names.

- [ ] **Step 2: Write sections 26.1 and 26.2**

Fill the first half of the question bank with these exact topic buckets and counts:

- `26.1 AI Infra 基础认知与系统分层` - 10 questions
- `26.2 硬件、GPU、内存、网络与存储基础` - 12 questions

Each question in these sections must keep the same repeated structure:

- `问题`
- `考察点`
- `回答框架`
- `追问`
- `评分要点`

Keep the prompts concrete and interview-like. They should cover system layering, resource bottlenecks, GPU selection, memory hierarchy, NUMA/PCIe, networking, and storage trade-offs rather than generic theory.

- [ ] **Step 3: Verify the first half before moving on**

Run:

```bash
rg -n '^### 26\.[12]\.' part8-advanced-and-capstone/26-ai-infra-interview-questions.md | wc -l
```

Expected: `22`

Run:

```bash
rg -n '^\*\*问题\*\*|^\*\*考察点\*\*|^\*\*回答框架\*\*|^\*\*追问\*\*|^\*\*评分要点\*\*' part8-advanced-and-capstone/26-ai-infra-interview-questions.md
```

Expected: the five labels appear in every written question block, with no missing section labels.

- [ ] **Step 4: Write sections 26.3 and 26.4**

Continue the same file with:

- `26.3 训练基础设施与分布式训练` - 12 questions
- `26.4 数据、制品、Checkpoint 与 Registry` - 10 questions

These questions should cover:

- single-node training bottlenecks
- data parallel scaling limits
- model / pipeline parallel trade-offs
- checkpoint sharding and restore semantics
- registry metadata and release units
- supply-chain and signing constraints for model artifacts

- [ ] **Step 5: Verify the midpoint**

Run:

```bash
rg -n '^### 26\.[1-4]\.' part8-advanced-and-capstone/26-ai-infra-interview-questions.md | wc -l
```

Expected: `44`

Run:

```bash
rg -n '^## .*Mock Interview' part8-advanced-and-capstone/26-ai-infra-interview-questions.md
```

Expected: no mock interview packs yet.

- [ ] **Step 6: Write sections 26.5 and 26.6**

Continue with:

- `26.5 推理服务、KV Cache、Batching 与推理引擎` - 14 questions
- `26.6 Kubernetes、调度、队列、配额与平台化` - 12 questions

These questions should cover:

- prefill/decode separation
- batching and cache lifecycle
- serving engine selection and trade-offs
- queue isolation and priority rules
- GPU scheduling and fragmentation
- autoscaling, quotas, and multi-tenant platformization

- [ ] **Step 7: Write sections 26.7 and 26.8 plus mock interview packs**

Finish the chapter with:

- `26.7 可观测性、发布、安全、成本与多租户治理` - 12 questions
- `26.8 综合系统设计与故障排查 Case` - 8 questions
- 3 to 5 mock interview packs

The mock interview packs should be explicit and role-oriented:

- Inference platform engineer, 60 minutes
- Training infrastructure engineer, 60 minutes
- AI platform engineer, 60 minutes
- Reliability and troubleshooting round, 45 minutes
- AI Infra tech lead system design, 90 minutes

Each pack should include warm-up questions, deep-dive questions, one case prompt, and what a strong interviewer should listen for.

- [ ] **Step 8: Verify the final chapter and commit**

Run:

```bash
rg -n '^### 26\.' part8-advanced-and-capstone/26-ai-infra-interview-questions.md | wc -l
```

Expected: a count in the `80-100` range, with the target around `90`.

Run:

```bash
rg -n '^## .*Mock Interview|^### 26\.7|^### 26\.8' part8-advanced-and-capstone/26-ai-infra-interview-questions.md
```

Expected: the governance and case sections plus 3-5 mock interview packs are present.

Then commit only the new chapter file:

```bash
git add part8-advanced-and-capstone/26-ai-infra-interview-questions.md
git commit -m "docs: add AI infra interview questions chapter"
```

---

## Task 2: Update README Navigation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Insert the new Chapter 26 row in Part 8**

Add one row directly after Chapter 25:

```markdown
| 第26章 | [AI Infra 面试题、自测与面试官题库](./part8-advanced-and-capstone/26-ai-infra-interview-questions.md) | 高频题、追问、评分要点、模拟面试组合 | 面试表达与综合系统判断 |
```

Keep the rest of the table unchanged.

- [ ] **Step 2: Verify the README link**

Run:

```bash
rg -n '第26章|26-ai-infra-interview-questions' README.md
```

Expected: one new Part 8 row that points to the new Markdown file.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add chapter 26 to tutorial navigation"
```

---

## Task 3: Update Tutorial Navigation Data

**Files:**
- Modify: `html/assets/tutorial-data.js`

- [ ] **Step 1: Insert the Chapter 26 entry after Chapter 25**

Add this object immediately after the existing `25` entry in the Part 8 group:

```js
      {
        "id": "26",
        "title": "第 26 章 · AI Infra 面试题、自测与面试官题库",
        "path": "part8/26-ai-infra-interview-questions.html"
      }
```

Do not rename the Part 8 group or reorder earlier chapters.

- [ ] **Step 2: Verify the navigation data**

Run:

```bash
rg -n '"id": "26"|"26-ai-infra-interview-questions"' html/assets/tutorial-data.js
```

Expected: the new chapter entry appears once, in the Part 8 array after Chapter 25.

- [ ] **Step 3: Commit**

```bash
git add html/assets/tutorial-data.js
git commit -m "docs: add chapter 26 to tutorial data"
```

---

## Task 4: Create the HTML Mirror

**Files:**
- Create: `html/part8/26-ai-infra-interview-questions.html`

- [ ] **Step 1: Copy the existing chapter page structure and adapt it for Chapter 26**

Use `html/part8/25-agent-and-inference-time-compute.html` as the template. Replace:

- `<title>`
- the sidebar `current=26`
- the hero `<h1>`
- the hero subtitle
- the chip labels
- the note and success callouts
- the table of contents
- every section heading and section body so they mirror the Markdown source

The hero should read like an interview-bank page, not a generic tutorial chapter. A good starting shape is:

```html
<title>第 26 章 · AI Infra 面试题、自测与面试官题库 — AI Infra 教程</title>
<iframe id="sidebar" src="../sidebar.html?current=26" class="sidebar-frame" loading="eager"></iframe>
...
<h1>第 26 章 · AI Infra 面试题、自测与面试官题库</h1>
<p class="sub">把全教程收束成面试高频题、自测题和面试官题库，重点考察资源、链路、故障和治理的系统判断。</p>
```

- [ ] **Step 2: Keep the existing static site footer contract**

The page must keep the same footer script pattern as other chapter pages:

- `../assets/tutorial-data.js`
- `../assets/mermaid.min.js`
- `../assets/nav.js`
- `mermaid.initialize({ startOnLoad: true, theme: 'neutral' });`

Do not add a new build tool or alter `html/sidebar.html`.

- [ ] **Step 3: Verify the HTML mirror**

Run:

```bash
rg -n 'sidebar.html\?current=26|tutorial-data.js|nav.js|第 26 章 · AI Infra 面试题、自测与面试官题库' html/part8/26-ai-infra-interview-questions.html
```

Expected: the sidebar link, assets, and page title/headings are present.

Then open the page in the browser and confirm:

- the sidebar highlights Chapter 26
- prev/next navigation renders
- the page body matches the Markdown structure
- no raw Markdown markers remain in the HTML

- [ ] **Step 4: Commit**

```bash
git add html/part8/26-ai-infra-interview-questions.html
git commit -m "docs: add HTML mirror for chapter 26"
```

---

## Task 5: Final Cross-File Integrity Check

**Files:**
- Read-only verification across the changed files

- [ ] **Step 1: Run a diff sanity check**

Run:

```bash
git diff --check -- \
  part8-advanced-and-capstone/26-ai-infra-interview-questions.md \
  README.md \
  html/assets/tutorial-data.js \
  html/part8/26-ai-infra-interview-questions.html
```

Expected: no whitespace or patch formatting errors.

- [ ] **Step 2: Run a cross-file reference scan**

Run:

```bash
rg -n '26-ai-infra-interview-questions|第26章|AI Infra 面试题、自测与面试官题库' \
  part8-advanced-and-capstone/26-ai-infra-interview-questions.md \
  README.md \
  html/assets/tutorial-data.js \
  html/part8/26-ai-infra-interview-questions.html
```

Expected: all four files reference the new chapter consistently.

- [ ] **Step 3: Close with a final commit only if verification required fixes**

If verification found any issue, fix it in the owning file and commit the fix with a narrow message. If no fix was needed, do not create a redundant commit.

