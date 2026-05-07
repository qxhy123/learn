# AI Infra Interview Questions Chapter Design

## Context

The AI Infra tutorial is a Markdown-first tutorial. `README.md`, `00-preface.md`, `part*/`, and `appendix/` are the source of truth; `html/` is generated output. The tutorial currently ends with Part 8:

- Chapter 24: building an AI platform
- Chapter 25: agent and inference-time compute infrastructure

The requested addition is a new chapter about AI Infra interview questions. It should serve three audiences at once:

1. Job candidates preparing for AI Infra roles.
2. Learners who want a comprehensive self-test after studying the tutorial.
3. Interviewers who need question prompts, follow-up paths, and scoring rubrics.

## Goals

Add a new Part 8 chapter, Chapter 26, that turns the full tutorial into an interview-ready question bank. The chapter should help readers practice not only definitions, but also system diagnosis, architecture reasoning, trade-off analysis, and clear technical communication.

The chapter should:

- Contain about 80-100 questions, with a target of about 90 questions.
- Cover the full tutorial: foundations, hardware, training, data, inference, platform, reliability, security, cost, and system design.
- Use a consistent question format so it works for candidates, self-test, and interviewers.
- Include interviewer-oriented follow-up questions and scoring criteria.
- Include several mock interview packs for common role directions.

## Non-Goals

- Do not move detailed answers into a new appendix in the first version.
- Do not split the first version into `26a`, `26b`, or multiple subchapters.
- Do not replace existing per-chapter exercises or Appendix D.
- Do not hand-edit `html/` as the source of truth; HTML can be regenerated later through the existing build/conversion flow.
- Do not introduce unrelated refactors to tutorial structure.

## Proposed Placement

Create:

```text
part8-advanced-and-capstone/26-ai-infra-interview-questions.md
```

Update the Part 8 table in `README.md` to add:

```text
第26章 | AI Infra 面试题、自测与面试官题库 | 高频题、追问、评分要点、模拟面试组合 | 面试表达与综合系统判断
```

The generated HTML target, when the HTML build is run, should be:

```text
html/part8/26-ai-infra-interview-questions.html
```

and the navigation data should gain the corresponding Part 8 entry.

## Chapter Structure

The chapter should follow the tutorial's existing style while adapting it to a question-bank format.

### Opening

Use the existing chapter template:

- Title and chapter quote.
- Related chapters section linking back to the full tutorial map.
- "第一性原理拆解 + 学习大纲".
- "学习目标".

The core first-principles message:

> AI Infra interviews do not primarily test whether someone can recite component names. They test whether the candidate can reason through resources, data/request flow, failure modes, and governance constraints under realistic trade-offs.

### Usage Guide

Add a short guide for three usage modes:

- Candidate mode: use questions to practice structured answers.
- Self-test mode: score answers against rubric and identify weak tutorial sections.
- Interviewer mode: use follow-ups and scoring criteria to evaluate depth.

### Question Format

Each question should use a compact, repeated structure:

```markdown
### 26.x.y Question Title

**问题**

...

**考察点**

- ...

**回答框架**

- ...

**追问**

- ...

**评分要点**

- 及格：
- 良好：
- 优秀：
```

This keeps the chapter useful without pushing full long-form answers into Appendix D.

## Topic Coverage

Target about 90 questions across 8 sections:

1. AI Infra foundations and system layering: about 10 questions.
2. Hardware, GPU, memory, network, and storage fundamentals: about 12 questions.
3. Training infrastructure and distributed training: about 12 questions.
4. Data, artifacts, checkpoints, and registry: about 10 questions.
5. Inference serving, KV Cache, batching, and inference engines: about 14 questions.
6. Kubernetes, scheduling, queues, quotas, and platformization: about 12 questions.
7. Observability, release, security, cost, and multi-tenancy governance: about 12 questions.
8. Integrated system design and troubleshooting cases: about 8 questions.

The count can vary slightly during writing as long as the final result stays in the 80-100 question range and every major tutorial part is represented.

## Mock Interview Packs

End the chapter with 3-5 mock interview packs. Recommended packs:

1. Inference platform engineer, 60 minutes.
2. Training infrastructure engineer, 60 minutes.
3. AI platform engineer, 60 minutes.
4. Reliability and troubleshooting round, 45 minutes.
5. AI Infra tech lead system design, 90 minutes.

Each pack should list:

- Warm-up questions.
- Deep-dive questions.
- System design or troubleshooting case.
- What a strong interviewer should listen for.

## Integration Points

Implementation should update Markdown source first:

- Add the new Chapter 26 source file.
- Update `README.md` Part 8 chapter table.

If the current repo has a documented or discoverable HTML generation flow, implementation can then generate/update:

- `html/part8/26-ai-infra-interview-questions.html`
- `html/assets/tutorial-data.js`
- any sidebar or index generated from tutorial data

If no reliable build flow is discoverable, leave HTML generation out of the first implementation and clearly report that only Markdown source was updated.

## Testing And Verification

At minimum:

- Verify the new Markdown file exists.
- Verify the README link points to an existing Markdown file.
- Run a link check or targeted `rg`/script-based check for the new chapter references if no full build exists.
- If HTML is generated, open/check the generated file and navigation entry.

## Risks And Mitigations

Risk: The chapter becomes too long and hard to scan.

Mitigation: Keep each question compact and use the repeated `问题 / 考察点 / 回答框架 / 追问 / 评分要点` format.

Risk: It duplicates existing exercises.

Mitigation: Position it as interview-oriented synthesis. Existing exercises can test chapter learning; Chapter 26 should test cross-chapter reasoning and verbal/system-design expression.

Risk: It becomes only a candidate cram sheet and loses interviewer value.

Mitigation: Require follow-ups and scoring criteria for each question, plus mock interview packs.

Risk: It becomes only high-level and misses infra depth.

Mitigation: Include concrete sections for GPU/memory/network/storage, distributed training, serving internals, Kubernetes scheduling, observability, security, and cost governance.

## Acceptance Criteria

- A Chapter 26 design exists as a single Part 8 chapter.
- The chapter contains 80-100 interview questions.
- Each question includes question prompt, assessment focus, answer framework, follow-ups, and scoring criteria.
- The chapter supports candidate preparation, self-testing, and interviewer evaluation.
- README navigation includes the new Chapter 26 Markdown link.
- Implementation avoids unrelated code or tutorial refactors.
