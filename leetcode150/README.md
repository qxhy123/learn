# LeetCode Top Interview 150 Python Tutorial

This repository is an English-first, Python-based tutorial for LeetCode's Top Interview 150 list. It is organized for advanced systematic learners who want reusable patterns, runnable examples, and a maintainable study workflow.

## Study Routes

- Official order: `docs/official-order.md`
- Pattern roadmap: `docs/pattern-roadmap.md`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

## Validate Metadata

```bash
python scripts/validate_metadata.py data/top_interview_150.yaml
```

## Generate Scaffold

```bash
python scripts/generate_scaffold.py data/top_interview_150.yaml --root .
```

The generator creates missing docs, solution files, tests, and indexes. It does not overwrite existing files unless `--force` is passed.

## Run Tests

```bash
pytest
```

Unimplemented problem tests are skipped until their matching solution is completed.

## Complete One Problem

1. Implement the method in `solutions/<pattern>/pNNN_slug.py`.
2. Remove the skip marker from `tests/<pattern>/test_pNNN_slug.py`.
3. Replace the problem doc's instructional TODO sections with final teaching content.
4. Run the problem-specific test file.
5. Run `pytest` before committing.
