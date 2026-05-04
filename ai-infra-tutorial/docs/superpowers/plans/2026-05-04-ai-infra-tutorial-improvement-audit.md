# AI Infra Tutorial Improvement Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a senior-engineer improvement spec for the current Markdown AI Infra tutorial source.

**Architecture:** Treat the tutorial audit as a document-production pipeline: inventory the Markdown corpus, read and score chapters by engineering depth, compress raw findings into actionable rewrite tasks, then publish a self-contained improvement spec. Generated HTML is excluded from the audit source of truth.

**Tech Stack:** Markdown, shell, `rg`, `wc`, `git diff --check`.

---

## File Structure

- Read: `docs/superpowers/specs/2026-05-04-ai-infra-tutorial-improvement-audit-design.md`
- Read: `README.md`
- Read: `00-preface.md`
- Read: `part0-foundations-of-systems/*.md`
- Read: `part1-foundations/*.md`
- Read: `part2-systems-stack/*.md`
- Read: `part3-training-infra/*.md`
- Read: `part4-data-and-storage/*.md`
- Read: `part5-serving-infra/*.md`
- Read: `part6-platform-and-orchestration/*.md`
- Read: `part7-reliability-security/*.md`
- Read: `part8-advanced-and-capstone/*.md`
- Read: `appendix/*.md`
- Create: `docs/superpowers/specs/2026-05-04-ai-infra-tutorial-improvement-spec.md`

The final spec is the only planned file creation. Do not edit tutorial body Markdown in this plan.

## Task 1: Build The Markdown Inventory

**Files:**
- Read: all Markdown source files listed in File Structure
- Create: `docs/superpowers/specs/2026-05-04-ai-infra-tutorial-improvement-spec.md`

- [ ] **Step 1: Confirm the audit target set**

Run:

```bash
rg --files \
  README.md 00-preface.md \
  part0-foundations-of-systems part1-foundations part2-systems-stack \
  part3-training-infra part4-data-and-storage part5-serving-infra \
  part6-platform-and-orchestration part7-reliability-security \
  part8-advanced-and-capstone appendix \
  -g '*.md' | sort
```

Expected: the command lists only Markdown source files under the tutorial content directories and does not include `html/` or `docs/superpowers/`.

- [ ] **Step 2: Measure chapter size distribution**

Run:

```bash
wc -w $(rg --files \
  README.md 00-preface.md \
  part0-foundations-of-systems part1-foundations part2-systems-stack \
  part3-training-infra part4-data-and-storage part5-serving-infra \
  part6-platform-and-orchestration part7-reliability-security \
  part8-advanced-and-capstone appendix \
  -g '*.md' | sort) | sort -n
```

Expected: output shows each source file's word count and a total. Use it to identify suspiciously thin overview chapters and unusually large chapters that may need split, summary, or navigation improvement.

- [ ] **Step 3: Map structural signals**

Run:

```bash
rg -n "^(#|##|###) |Worked Example|Mini case|案例|SOP|Runbook|runbook|Checklist|checklist|练习|自测|公式|决策树|验收|回滚|容量|成本|故障|排障" \
  README.md 00-preface.md \
  part0-foundations-of-systems part1-foundations part2-systems-stack \
  part3-training-infra part4-data-and-storage part5-serving-infra \
  part6-platform-and-orchestration part7-reliability-security \
  part8-advanced-and-capstone appendix \
  -g '*.md'
```

Expected: output gives a rough coverage map for examples, SOPs, checklists, exercises, capacity, cost, and failure content. Use this as evidence when deciding which chapters need stronger engineering closure.

- [ ] **Step 4: Create the spec skeleton**

Create `docs/superpowers/specs/2026-05-04-ai-infra-tutorial-improvement-spec.md` with this structure:

```markdown
# AI Infra Tutorial Improvement Spec

> Date: 2026-05-04
> Source of truth: current working-tree Markdown source
> Output focus: actionable rewrite and strengthening blueprint

## 1. Overall Diagnosis

## 2. Improvement Principles

## 3. Part-Level Blueprint

### 3.1 Part 0: Foundations Of Systems

### 3.2 Part 1: AI Infra Foundations

### 3.3 Part 2: Hardware And Systems Stack

### 3.4 Part 3: Training Infrastructure

### 3.5 Part 4: Data And Storage Infrastructure

### 3.6 Part 5: Serving Infrastructure

### 3.7 Part 6: Platform And Orchestration

### 3.8 Part 7: Reliability, Security, And Governance

### 3.9 Part 8: Advanced And Capstone

### 3.10 Appendix

## 4. Chapter-Level Actions

## 5. Cross-Cutting Capability Lines

## 6. Priority And Roadmap

## 7. Acceptance Criteria
```

Do not leave any section empty by the end of Task 6.

## Task 2: Audit Global Structure And Teaching Contract

**Files:**
- Read: `README.md`
- Read: `00-preface.md`
- Read: top-level overview chapters in each part
- Modify: `docs/superpowers/specs/2026-05-04-ai-infra-tutorial-improvement-spec.md`

- [ ] **Step 1: Read the declared tutorial promise**

Read `README.md` and `00-preface.md`. Extract the tutorial's promised reader outcomes: system intuition, bottleneck intuition, platform intuition, diagnostic ability, and design ability.

- [ ] **Step 2: Compare the promise with chapter organization**

Check whether each part has a clear job:

```text
Part 0: systems substrate
Part 1: global mental model
Part 2: hardware/runtime constraints
Part 3: training systems
Part 4: data/artifact/vector systems
Part 5: online inference
Part 6: orchestration/platform resource control
Part 7: reliability/security/governance
Part 8: integration and emerging patterns
Appendix: operational reference
```

Record mismatches where a part's chapters do not clearly produce the stated capability.

- [ ] **Step 3: Write `Overall Diagnosis`**

Populate `## 1. Overall Diagnosis` with:

```markdown
- Strengths to preserve:
  - `<chapter path>` already connects mechanism to production operation through `<specific evidence>`.
- Structural weaknesses:
  - `<chapter path>` promises `<capability>` but lacks `<mechanism/path/failure/cost evidence>`.
- Repeated shallow patterns:
  - `<chapters>` list `<tools/concepts>` without `<decision rule/workload boundary/verification path>`.
- Missing cross-cutting lines:
  - `<capability line>` appears in `<chapter>` but is not carried into `<dependent chapter>`.
- Granularity and ordering issues:
  - `<chapter or part>` should be `<split/merged/reordered/cross-linked>` because `<reader workflow impact>`.
```

Each bullet must cite at least one concrete part or chapter as evidence.

- [ ] **Step 4: Write `Improvement Principles`**

Populate `## 2. Improvement Principles` with chapter quality rules:

```markdown
Every strengthened chapter must include:

1. A boundary section: what the topic is, what it is not, and which adjacent layer owns nearby problems.
2. A path section: control path, data path, state path, or failure path, depending on the topic.
3. An evidence section: metrics, logs, traces, commands, configs, events, or admission checks.
4. A model section: capacity, performance, reliability, or cost when relevant.
5. A failure section: symptoms, root causes, mitigations, rollback or recovery behavior.
6. A worked example with numbers, diagnosis, action, and verification.

Weak patterns to remove:

- Advice without a decision rule.
- Tool lists without selection criteria.
- Performance claims without workload, hardware, and benchmark boundaries.
- Architecture diagrams without state ownership or failure behavior.
- Exercises that ask only for definitions where the chapter claims to teach design.
```

Adjust wording to fit findings, but keep the rules concrete.

## Task 3: Audit Part-Level Blueprints

**Files:**
- Read: all tutorial Markdown source directories
- Modify: `docs/superpowers/specs/2026-05-04-ai-infra-tutorial-improvement-spec.md`

- [ ] **Step 1: Review each part for its capability output**

For each part, answer:

```text
What should be kept?
What should be reinforced?
What should be trimmed or merged?
What capability should a reader have after completing the part?
Which cross-part dependencies must be explicit?
```

- [ ] **Step 2: Populate `Part-Level Blueprint`**

For each subsection under `## 3. Part-Level Blueprint`, write this form:

```markdown
**Keep:**
- `<specific existing chapter trait>` because it already helps readers build `<capability>`.

**Reinforce:**
- Add `<boundary/model/failure/example/artifact>` so readers can make `<engineering decision>`.

**Trim Or Merge:**
- Move or condense `<specific repeated or low-value material>` into `<target chapter or appendix>`.

**Reader Capability After This Part:**
- Reader can `<diagnose/design/operate>` `<specific AI infra workflow>` using `<evidence or model>`.

**Required Cross-Part Links:**
- Link `<source chapter>` to `<target chapter>` at the point where `<dependency>` becomes necessary.
```

Expected: all parts 0-8 and Appendix have non-empty content.

- [ ] **Step 3: Check for duplicate or disconnected material**

Run:

```bash
rg -n "第一性原理|SOP|Checklist|Worked Example|capacity|容量|成本|回滚|验收|多租户|lineage|血缘|评测|发布" \
  part0-foundations-of-systems part1-foundations part2-systems-stack \
  part3-training-infra part4-data-and-storage part5-serving-infra \
  part6-platform-and-orchestration part7-reliability-security \
  part8-advanced-and-capstone appendix \
  -g '*.md'
```

Expected: repeated themes are visible. Use the output to decide whether the spec should recommend cross-references, consolidation, or appendix-level unification.

## Task 4: Audit Chapter-Level Actions

**Files:**
- Read: all tutorial Markdown source files
- Modify: `docs/superpowers/specs/2026-05-04-ai-infra-tutorial-improvement-spec.md`

- [ ] **Step 1: Read chapters by part and capture only high-value actions**

For each chapter, produce 3-6 rewrite actions. Use this exact action shape:

```markdown
### `<path>`

1. **Action title**
   - Reason: `<specific gap and why it matters to senior AI infra readers>`.
   - Add or rewrite: `<named section/table/formula/example/checklist/cross-reference>`.
   - Acceptance signal: `<concrete evidence that the rewrite is stronger>`.
```

If a chapter is already strong, still include actions for integration, pruning, or cross-linking if useful. Do not write sentence-level copy edits unless they affect technical correctness.

- [ ] **Step 2: Ensure actions are implementable**

Every action must name a concrete artifact to add or rewrite, such as:

```text
boundary section
capacity formula
symptom-evidence-root-cause-action table
worked example
decision tree
config example
preflight checklist
rollback protocol
cross-reference
appendix answer update
```

- [ ] **Step 3: Populate `Chapter-Level Actions`**

Add chapter actions under `## 4. Chapter-Level Actions`, grouped by part. Use exact Markdown file paths as headings so future workers can turn actions into edit tasks.

- [ ] **Step 4: Check coverage**

Run:

```bash
for f in $(rg --files \
  README.md 00-preface.md \
  part0-foundations-of-systems part1-foundations part2-systems-stack \
  part3-training-infra part4-data-and-storage part5-serving-infra \
  part6-platform-and-orchestration part7-reliability-security \
  part8-advanced-and-capstone appendix \
  -g '*.md' | sort); do
  rg -q "### \`$f\`" docs/superpowers/specs/2026-05-04-ai-infra-tutorial-improvement-spec.md || echo "missing $f"
done
```

Expected: no output. If any file is missing, add a chapter-level action entry for it or explicitly explain under the appropriate part why it is excluded.

## Task 5: Define Cross-Cutting Capability Lines

**Files:**
- Read: all tutorial Markdown source files
- Modify: `docs/superpowers/specs/2026-05-04-ai-infra-tutorial-improvement-spec.md`

- [ ] **Step 1: Identify horizontal engineering threads**

Review the chapter-level actions and identify where these lines should appear:

```text
capacity planning
performance diagnosis
reliability and recovery
multi-tenancy, fairness, and cost
security, supply chain, and governance
data and artifact lineage
evaluation and release safety
first-principles-to-SOP translation
```

- [ ] **Step 2: Populate `Cross-Cutting Capability Lines`**

For each line, write:

```markdown
### Capability Line Name

- Current issue: `<where this engineering thread appears but does not yet become actionable>`.
- Chapters that should carry it: `<exact Markdown paths>`.
- Artifact to add: `<formula/runbook/checklist/table/example/appendix update>`.
- Acceptance signal: `<how a reader can use the artifact to decide or diagnose something>`.
```

Expected: each line names concrete chapter paths and artifacts, not just abstract themes.

- [ ] **Step 3: Add appendix integration requirements**

For each capability line, state whether it needs an appendix update:

```text
glossary term
tooling-map entry
checklist entry
exercise answer update
```

Expected: appendix actions are reflected both in `## 4. Chapter-Level Actions` and `## 5. Cross-Cutting Capability Lines`.

## Task 6: Prioritize Roadmap And Acceptance Criteria

**Files:**
- Modify: `docs/superpowers/specs/2026-05-04-ai-infra-tutorial-improvement-spec.md`

- [ ] **Step 1: Assign P0/P1/P2 priorities**

Classify recommendations:

```markdown
### P0

- `<priority item>`: affects `<chapters>` because `<credibility or correctness risk>`.

### P1

- `<priority item>`: improves `<engineering usefulness>` in `<chapters>`.

### P2

- `<priority item>`: improves `<consistency, appendix coverage, links, exercises, or references>`.
```

Use these definitions:

```text
P0: materially affects credibility for senior AI infra readers.
P1: significantly improves engineering usefulness.
P2: polish, consistency, appendix strengthening, or optional deeper reference.
```

- [ ] **Step 2: Group work into implementation waves**

Add a roadmap shaped like:

```markdown
### Wave 1: Foundation And Contract Repair

- Scope: `<coherent audit-driven rewrite theme>`.
- Chapters: `<exact Markdown paths>`.
- Done when: `<observable acceptance condition>`.

### Wave 2: Training And Serving Engineering Depth

- Scope: `<coherent audit-driven rewrite theme>`.
- Chapters: `<exact Markdown paths>`.
- Done when: `<observable acceptance condition>`.

### Wave 3: Platform, Reliability, Security, And Cost

- Scope: `<coherent audit-driven rewrite theme>`.
- Chapters: `<exact Markdown paths>`.
- Done when: `<observable acceptance condition>`.

### Wave 4: Appendix, Exercises, And Cross-Link Polish

- Scope: `<coherent audit-driven rewrite theme>`.
- Chapters: `<exact Markdown paths>`.
- Done when: `<observable acceptance condition>`.
```

Adjust wave names if the audit findings justify a different grouping, but keep each wave coherent.

- [ ] **Step 3: Populate `Acceptance Criteria`**

Write checks that future rewrites can run manually and with shell commands:

```markdown
- No empty spec sections.
- Every source Markdown file has a chapter-level action or explicit exclusion.
- Every P0 item maps to at least one implementation wave.
- Every chapter rewrite action includes reason, concrete edit, and acceptance signal.
- Rewritten chapters should pass `rg -n "T[O]DO|T[B]D|F[I]XME|待[补]|后续[补]|这里不[展]开" <changed-files>`.
- Rewritten chapters should pass `git diff --check -- <changed-files>`.
- Cross-references should point to existing Markdown files.
- Appendix answers must be updated when exercises change.
```

## Task 7: Self-Review And Commit

**Files:**
- Modify: `docs/superpowers/specs/2026-05-04-ai-infra-tutorial-improvement-spec.md`

- [ ] **Step 1: Run placeholder scan**

Run:

```bash
rg -n "T[O]DO|T[B]D|F[I]XME|待[定]|待[补]|后续[补]|这里不[展]开|PLACE[H]OLDER|\\.\\.\\." \
  docs/superpowers/specs/2026-05-04-ai-infra-tutorial-improvement-spec.md
```

Expected: no output. If output appears, replace placeholders with concrete content.

- [ ] **Step 2: Check required sections exist**

Run:

```bash
rg -n "^## |^### " docs/superpowers/specs/2026-05-04-ai-infra-tutorial-improvement-spec.md
```

Expected: output includes all major sections from Task 1 and part/chapter/capability/roadmap subsections.

- [ ] **Step 3: Check chapter coverage**

Run the coverage command from Task 4 Step 4 again.

Expected: no missing files.

- [ ] **Step 4: Check Markdown whitespace**

Run:

```bash
git diff --check -- docs/superpowers/specs/2026-05-04-ai-infra-tutorial-improvement-spec.md
```

Expected: no output.

- [ ] **Step 5: Commit only the improvement spec**

Run:

```bash
git add docs/superpowers/specs/2026-05-04-ai-infra-tutorial-improvement-spec.md
git commit -m "Add AI infra tutorial improvement spec"
```

Expected: commit succeeds and includes only the improvement spec.
