# Two Pointers Content Completion Design

Date: 2026-04-24
Status: Approved design

## Goal

Complete the first real content batch for the LeetCode Top Interview 150 Python tutorial by fully implementing and documenting the `Two Pointers` topic.

This batch turns five scaffolded problems into production-quality tutorial entries with final English explanations, recommended Python solutions, active pytest coverage, and reader-facing completion status.

## Scope

Complete exactly these five problems:

- `125. Valid Palindrome`
- `392. Is Subsequence`
- `167. Two Sum II - Input Array Is Sorted`
- `11. Container With Most Water`
- `15. 3Sum`

A problem is complete when it has final tutorial content, one recommended Python implementation, unskipped tests with meaningful edge coverage, and no instructional scaffold TODOs.

## Non-Goals

- Completing any topic outside `Two Pointers`.
- Implementing multiple code variants per problem.
- Building a web UI or progress dashboard.
- Regenerating the full scaffold with `--force`.
- Copying full LeetCode problem statements into the docs.

## Completion Metadata

Add lightweight completion metadata only to the five `Two Pointers` entries in `data/top_interview_150.yaml`:

```yaml
status: complete
completed_at: 2026-04-24
```

Unfinished problems can continue omitting `status`. This keeps the current metadata simple while allowing README and roadmap text to describe completed work honestly.

## Tutorial Content Template

Each touched doc under `docs/problems/two_pointers/` should replace scaffold TODOs with detailed English tutorial content.

Required sections:

- Problem metadata: title, difficulty, LeetCode link, official group, pattern group, and tags.
- Core Pattern: a transferable two-pointer rule.
- Why Two Pointers Fits: the input property or monotonic tradeoff that enables the method.
- Recommended Approach: step-by-step algorithm.
- Alternative Approaches: one or two useful contrasts without implementing them.
- Correctness Sketch: proof via invariant, symmetry, ordering, or dominance.
- Trace: representative example with pointer movement.
- Complexity: exact time and space complexity.
- Common Pitfalls: specific mistakes for the problem.
- Implementation Notes: link to the solution and any key helper logic.
- Tests: themes covered by the test file.
- Interview Script: concise explanation a learner can say aloud.
- Review Questions: three to five self-check questions.
- Follow-up Practice: related variants or next problems.

Docs must remain English-first and should teach pattern transfer rather than restating the full LeetCode prompt.

## Solution Strategy

Each solution file keeps the existing `class Solution` and LeetCode method name, but replaces `NotImplementedError` with one recommended implementation.

- `Valid Palindrome`: scan from both ends, skip non-alphanumeric characters, compare lowercase characters.
- `Is Subsequence`: scan `s` and `t` with two pointers; advance the `s` pointer only on a match.
- `Two Sum II`: use left/right pointers on sorted numbers and return 1-indexed positions.
- `Container With Most Water`: compute area between left/right bars, update max area, move the shorter side.
- `3Sum`: sort, fix one index, scan the suffix with two pointers, skip duplicates at every level.

The docs may discuss brute force, hash-based alternatives, filtered-string palindrome, binary-search subsequence checks, or hash-set `3Sum`, but the code should expose only the recommended method.

Touched solution files should use Python 3.11 built-in generics such as `list[int]` instead of relying on unimported `List` annotations.

## Test Strategy

Remove the module-level skip from the five `tests/two_pointers/` files and expand them into active tests.

Coverage expectations:

### Valid Palindrome

- Official true/false examples.
- Empty or effectively empty string.
- Mixed punctuation and case.
- False case after filtering.

### Is Subsequence

- Official true/false examples.
- Empty `s`.
- Empty `t`.
- Repeated characters.
- Order-sensitive false case.

### Two Sum II

- Official examples.
- Negative numbers.
- Minimal two-element input.
- Duplicate values.

### Container With Most Water

- Official examples.
- Two bars only.
- Monotonic increasing and decreasing heights.
- Equal heights.

### 3Sum

- Official examples.
- Empty and no-solution inputs.
- All zeros.
- Duplicate-heavy cases.
- Normalized comparison that sorts each triplet and sorts the triplet list.

Verification should run each touched test while developing, then `pytest tests/two_pointers -q`, then full `pytest`.

## README And Roadmap Updates

Update reader-facing progress without introducing a complex status system.

`README.md` should include a small completed-topic section explaining that completed topics have final docs, implemented solutions, and active tests. Mark `Two Pointers` complete.

`docs/pattern-roadmap.md` should mark each `Two Pointers` problem complete and include this recommended learning order:

1. `Valid Palindrome`
2. `Is Subsequence`
3. `Two Sum II - Input Array Is Sorted`
4. `Container With Most Water`
5. `3Sum`

Do not run full scaffold regeneration with `--force` as part of this batch, because completed human-written files should not be overwritten.

## Acceptance Criteria

- All five `Two Pointers` docs have no scaffold TODOs.
- All five `Two Pointers` solution files contain working recommended implementations.
- All five `Two Pointers` test files are active and pass.
- `3Sum` tests normalize output order.
- `data/top_interview_150.yaml` marks only the five touched problems as complete.
- `README.md` and `docs/pattern-roadmap.md` clearly show `Two Pointers` completion status.
- `python scripts/validate_metadata.py data/top_interview_150.yaml` passes.
- `pytest tests/two_pointers -q` passes.
- Full `python -m pytest` passes with the remaining unfinished topics skipped.

## Risks And Mitigations

- Large doc edits can become inconsistent: use the same section order for all five problem pages.
- `3Sum` has flexible output ordering: normalize triplets in tests.
- LeetCode-style type hints may reference undefined `List`: use built-in generics in touched files.
- Generator can overwrite hand-written content if forced: do not use `--force` for this batch.
- Edge cases can outgrow the first pass: include broad but focused tests rather than exhaustive fuzzing.
