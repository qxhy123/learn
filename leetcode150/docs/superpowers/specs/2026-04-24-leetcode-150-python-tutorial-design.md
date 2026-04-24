# LeetCode Top Interview 150 Python Tutorial Design

Date: 2026-04-24
Status: Approved design

## Goal

Create a repository-style tutorial for LeetCode's Top Interview 150 list. The tutorial is English-first, Python-based, and aimed at advanced systematic learners who want reusable algorithm patterns, runnable examples, and a maintainable study route.

The first version covers the complete 150-problem scaffold: canonical metadata, generated tutorial pages, generated solution placeholders, generated pytest files with official examples, and two study indexes. It does not attempt to fully solve all 150 problems in the first pass.

## Audience

The target learner already knows basic data structures and wants to build a transferable pattern library for interviews. The tutorial should emphasize pattern recognition, invariants, complexity, pitfalls, and implementation discipline rather than only single-problem answers.

## Repository Structure

```text
README.md
pyproject.toml
data/
  top_interview_150.yaml
docs/
  official-order.md
  pattern-roadmap.md
  problems/
    <pattern>/
      pNNN_slug.md
  superpowers/
    specs/
      2026-04-24-leetcode-150-python-tutorial-design.md
scripts/
  generate_scaffold.py
  validate_metadata.py
solutions/
  <pattern>/
    pNNN_slug.py
tests/
  <pattern>/
    test_pNNN_slug.py
```

`data/top_interview_150.yaml` is the canonical source of truth for problem metadata. Generated docs, solution files, tests, and indexes should be deterministic outputs of that metadata. Human-written solution explanations can be added later without changing the core structure.

## Problem Metadata

Each problem entry in `data/top_interview_150.yaml` contains:

- `number`: LeetCode problem number.
- `title`: English title.
- `slug`: URL and file-safe slug.
- `difficulty`: `Easy`, `Medium`, or `Hard`.
- `official_group`: original Top Interview 150 group.
- `pattern_group`: tutorial route group.
- `patterns`: focused tags that capture reusable techniques.
- `leetcode_url`: canonical LeetCode problem URL.
- `solution_path`: generated solution path.
- `doc_path`: generated tutorial path.
- `test_path`: generated pytest path.
- `examples`: normalized example inputs and outputs used by generated tests.
- `constraints_summary`: short constraint notes when they influence the intended approach.

Example data should be stored as normalized Python-call data instead of raw problem-statement prose. This keeps generated tests simple and avoids statement parsing.

## Study Indexes

The tutorial provides two complementary routes:

1. `docs/official-order.md` preserves the LeetCode Top Interview 150 order and official grouping.
2. `docs/pattern-roadmap.md` reorganizes the same problems by reusable algorithm pattern.

The official-order index helps learners compare against LeetCode's list. The pattern roadmap helps learners build a systematic mental model across arrays, two pointers, sliding window, stack, binary search, trees, graphs, dynamic programming, and related categories.

## Tutorial Page Template

Each problem page under `docs/problems/<pattern>/pNNN_slug.md` uses this English-first structure:

- Problem: title, difficulty, LeetCode link, official group, and pattern group.
- Core Pattern: the main reusable idea as a transferable rule.
- When To Use It: signals that the pattern applies in interviews.
- Approach: step-by-step reasoning.
- Correctness Sketch: concise explanation of why the approach works.
- Complexity: time and space complexity.
- Common Pitfalls: edge cases and implementation mistakes.
- Implementation: link to the Python solution file.
- Tests: link to the pytest file and official examples covered.
- Follow-up Practice: optional similar problems.

Generated pages may contain explicit `TODO` lines for instructional sections. TODOs should be searchable and specific, such as `TODO: Explain the reusable invariant for this problem before filling the implementation.` Avoid vague placeholders.

## Solution Files

Each generated solution file uses a LeetCode-like interface:

- Path: `solutions/<pattern>/pNNN_slug.py`.
- Class: `Solution`.
- Method: the LeetCode method name when known.
- Initial body: `raise NotImplementedError`.
- Docstring: short pointer back to the tutorial document.

The stable interface lets tests and docs reference a predictable module from the first version onward.

## Test Files

Each generated test file uses `pytest`:

- Path: `tests/<pattern>/test_pNNN_slug.py`.
- Imports the matching `Solution` class.
- Encodes official examples from metadata.
- Starts skipped with `pytest.mark.skip(reason="Solution not implemented yet")` until the solution is filled.
- Handles multiple valid outputs by normalization or accepted answer sets.
- Handles mutable-input problems by asserting both return values and expected mutations when LeetCode requires mutation.

The initial full test suite should be runnable with `pytest` and should report skipped tests rather than failures for unimplemented solutions.

## Generation Scripts

`scripts/generate_scaffold.py` reads `data/top_interview_150.yaml` and creates missing tutorial docs, solution files, test files, and index pages. It should not overwrite human-written completed content by default. If destructive regeneration is ever needed, it should require an explicit `--force` flag.

`scripts/validate_metadata.py` checks required fields, unique problem numbers, unique slugs, unique paths, valid difficulties, valid groups, present examples, and path naming conventions.

## Maintenance Flow

To complete one problem:

1. Fill the solution method in `solutions/<pattern>/pNNN_slug.py`.
2. Remove or narrow the skip marker in `tests/<pattern>/test_pNNN_slug.py`.
3. Replace tutorial TODO sections with the final explanation.
4. Run the problem-specific pytest file.
5. Optionally run the full test suite.

This flow keeps each problem independently reviewable and allows the tutorial to grow incrementally.

## External Data Policy

Before generating the official 150-problem metadata, verify the current official LeetCode Top Interview 150 list. Do not rely only on memory because the official list can change over time.

## Non-Goals For First Version

- Fully solving all 150 problems.
- Building a web UI or search interface.
- Adding non-Python implementations.
- Adding new runtime dependencies beyond what is necessary for testing and scaffold generation.
- Copying full LeetCode problem statements into the repository.

## Acceptance Criteria

The first implementation plan should produce:

- A valid Python project skeleton.
- Canonical metadata for the verified Top Interview 150 list.
- Generated docs, solutions, tests, and indexes for all 150 problems.
- `pytest` runnable from the repo root with unimplemented problems skipped.
- Metadata validation that catches missing or inconsistent scaffold fields.
- README instructions for studying, testing, and maintaining the tutorial.

## Risks And Mitigations

- Official list drift: verify the official source before metadata generation.
- Excessive generated files: make generation deterministic and avoid overwrites by default.
- Test ambiguity: encode normalized examples and special-case multiple valid outputs or mutation requirements.
- Empty tutorial feel: make TODOs specific and structured so unfinished sections still guide future authors.
- Maintenance burden: keep metadata as the single source of truth and use scripts to regenerate predictable scaffolding.
