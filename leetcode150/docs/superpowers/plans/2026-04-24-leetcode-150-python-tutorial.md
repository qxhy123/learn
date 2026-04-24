# LeetCode 150 Python Tutorial Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a metadata-driven Python tutorial scaffold for LeetCode Top Interview 150 with English docs, solution placeholders, pytest examples, and dual study indexes.

**Architecture:** `data/top_interview_150.yaml` is the single source of truth. Validation code checks metadata integrity, and generation code creates deterministic docs, solution placeholders, tests, and indexes without overwriting human-authored files by default.

**Tech Stack:** Python 3.11+, `pytest`, standard-library scaffold scripts, YAML metadata parsed via PyYAML.

---

## File Structure

- Create `README.md`: user-facing tutorial overview, study paths, setup, generation, and test commands.
- Create `pyproject.toml`: package metadata, pytest configuration, and script dependency declaration.
- Create `data/top_interview_150.yaml`: canonical verified Top Interview 150 metadata.
- Create `scripts/validate_metadata.py`: metadata schema and path validation CLI.
- Create `scripts/generate_scaffold.py`: deterministic scaffold generator CLI.
- Create `tests/scripts/test_validate_metadata.py`: focused validator tests using temporary metadata files.
- Create `tests/scripts/test_generate_scaffold.py`: generator behavior tests using temporary output roots.
- Generate `docs/official-order.md`: official LeetCode grouping index.
- Generate `docs/pattern-roadmap.md`: pattern-based learning roadmap.
- Generate `docs/problems/<pattern>/pNNN_slug.md`: one tutorial template per problem.
- Generate `solutions/<pattern>/pNNN_slug.py`: one LeetCode-like Python solution placeholder per problem.
- Generate `tests/<pattern>/test_pNNN_slug.py`: one skipped pytest file per problem with official examples.

---

### Task 1: Project Baseline

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `.gitignore`

- [ ] **Step 1: Create Python project configuration**

Write `pyproject.toml`:

```toml
[project]
name = "leetcode150"
version = "0.1.0"
description = "A metadata-driven Python tutorial scaffold for LeetCode Top Interview 150."
requires-python = ">=3.11"
dependencies = [
  "PyYAML>=6.0.1",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
addopts = "-ra"
```

- [ ] **Step 2: Create root ignore rules**

Write `.gitignore`:

```gitignore
__pycache__/
*.py[cod]
.pytest_cache/
.venv/
.DS_Store
```

- [ ] **Step 3: Create initial README**

Write `README.md`:

```markdown
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
```

- [ ] **Step 4: Run baseline validation**

Run:

```bash
python -m pytest --collect-only
```

Expected: pytest starts successfully and reports no collected tests or only existing tests from later tasks.

- [ ] **Step 5: Commit baseline**

Run:

```bash
git add pyproject.toml README.md .gitignore
git commit -m "Prepare the Python tutorial project baseline" -m "The tutorial scaffold needs a minimal Python project shape before metadata validation and generation can be added.\n\nConfidence: high\nScope-risk: narrow\nTested: python -m pytest --collect-only\nNot-tested: Full scaffold generation not implemented yet"
```

---

### Task 2: Metadata Validator Tests

**Files:**
- Create: `tests/scripts/test_validate_metadata.py`
- Create: `scripts/__init__.py`

- [ ] **Step 1: Create scripts package marker**

Write `scripts/__init__.py`:

```python
"""Repository maintenance scripts for the LeetCode 150 tutorial."""
```

- [ ] **Step 2: Write validator tests**

Write `tests/scripts/test_validate_metadata.py`:

```python
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from scripts.validate_metadata import MetadataError, load_metadata, validate_metadata


def write_yaml(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")


def valid_problem() -> dict:
    return {
        "number": 1,
        "title": "Two Sum",
        "slug": "two-sum",
        "difficulty": "Easy",
        "official_group": "Array / String",
        "pattern_group": "Hash Table",
        "patterns": ["hash-map", "complement-lookup"],
        "leetcode_url": "https://leetcode.com/problems/two-sum/",
        "method_name": "twoSum",
        "signature": "def twoSum(self, nums: list[int], target: int) -> list[int]",
        "solution_path": "solutions/hash_table/p001_two_sum.py",
        "doc_path": "docs/problems/hash_table/p001_two_sum.md",
        "test_path": "tests/hash_table/test_p001_two_sum.py",
        "examples": [
            {"input": {"nums": [2, 7, 11, 15], "target": 9}, "output": [0, 1]},
        ],
        "constraints_summary": "Exactly one valid answer exists.",
    }


def test_load_metadata_reads_problem_list(tmp_path: Path) -> None:
    metadata_path = tmp_path / "problems.yaml"
    write_yaml(metadata_path, yaml.safe_dump({"problems": [valid_problem()]}, sort_keys=False))

    metadata = load_metadata(metadata_path)

    assert metadata["problems"][0]["slug"] == "two-sum"


def test_validate_metadata_accepts_valid_problem() -> None:
    errors = validate_metadata({"problems": [valid_problem()]})

    assert errors == []


def test_validate_metadata_rejects_duplicate_numbers() -> None:
    first = valid_problem()
    second = valid_problem() | {"slug": "two-sum-duplicate"}

    errors = validate_metadata({"problems": [first, second]})

    assert "duplicate number: 1" in errors


def test_validate_metadata_rejects_invalid_difficulty() -> None:
    problem = valid_problem() | {"difficulty": "Expert"}

    errors = validate_metadata({"problems": [problem]})

    assert "problem 1 has invalid difficulty: Expert" in errors


def test_validate_metadata_rejects_mismatched_paths() -> None:
    problem = valid_problem() | {"solution_path": "solutions/array/p001_two_sum.py"}

    errors = validate_metadata({"problems": [problem]})

    assert "problem 1 solution_path should be solutions/hash_table/p001_two_sum.py" in errors


def test_load_metadata_rejects_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(MetadataError, match="metadata file does not exist"):
        load_metadata(missing_path)
```

- [ ] **Step 3: Run tests to verify failure**

Run:

```bash
python -m pytest tests/scripts/test_validate_metadata.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.validate_metadata'`.

- [ ] **Step 4: Commit failing tests**

Run:

```bash
git add scripts/__init__.py tests/scripts/test_validate_metadata.py
git commit -m "Specify metadata validation behavior before implementation" -m "Validator tests define the schema, uniqueness checks, and generated path conventions before the CLI exists.\n\nConfidence: high\nScope-risk: narrow\nTested: python -m pytest tests/scripts/test_validate_metadata.py -q fails because implementation is absent\nNot-tested: Validator implementation not added yet"
```

---

### Task 3: Metadata Validator Implementation

**Files:**
- Create: `scripts/validate_metadata.py`

- [ ] **Step 1: Implement validator CLI**

Write `scripts/validate_metadata.py`:

```python
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import yaml

VALID_DIFFICULTIES = {"Easy", "Medium", "Hard"}
REQUIRED_FIELDS = {
    "number",
    "title",
    "slug",
    "difficulty",
    "official_group",
    "pattern_group",
    "patterns",
    "leetcode_url",
    "method_name",
    "signature",
    "solution_path",
    "doc_path",
    "test_path",
    "examples",
}


class MetadataError(RuntimeError):
    """Raised when metadata cannot be loaded."""


def load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise MetadataError(f"metadata file does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise MetadataError("metadata root must be a mapping")
    return data


def pattern_dir(pattern_group: str) -> str:
    normalized = pattern_group.lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return normalized


def file_stem(number: int, slug: str) -> str:
    normalized_slug = slug.replace("-", "_")
    return f"p{number:03d}_{normalized_slug}"


def expected_paths(problem: dict[str, Any]) -> tuple[str, str, str]:
    stem = file_stem(int(problem["number"]), str(problem["slug"]))
    directory = pattern_dir(str(problem["pattern_group"]))
    return (
        f"solutions/{directory}/{stem}.py",
        f"docs/problems/{directory}/{stem}.md",
        f"tests/{directory}/test_{stem}.py",
    )


def validate_problem(problem: dict[str, Any], index: int) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_FIELDS - set(problem))
    if missing:
        errors.append(f"problem at index {index} is missing fields: {', '.join(missing)}")
        return errors

    number = problem["number"]
    if not isinstance(number, int) or number <= 0:
        errors.append(f"problem at index {index} has invalid number: {number}")

    difficulty = problem["difficulty"]
    if difficulty not in VALID_DIFFICULTIES:
        errors.append(f"problem {number} has invalid difficulty: {difficulty}")

    patterns = problem["patterns"]
    if not isinstance(patterns, list) or not patterns or not all(isinstance(item, str) for item in patterns):
        errors.append(f"problem {number} patterns must be a non-empty list of strings")

    examples = problem["examples"]
    if not isinstance(examples, list) or not examples:
        errors.append(f"problem {number} examples must be a non-empty list")

    solution_path, doc_path, test_path = expected_paths(problem)
    if problem["solution_path"] != solution_path:
        errors.append(f"problem {number} solution_path should be {solution_path}")
    if problem["doc_path"] != doc_path:
        errors.append(f"problem {number} doc_path should be {doc_path}")
    if problem["test_path"] != test_path:
        errors.append(f"problem {number} test_path should be {test_path}")

    return errors


def validate_metadata(metadata: dict[str, Any]) -> list[str]:
    problems = metadata.get("problems")
    if not isinstance(problems, list) or not problems:
        return ["metadata must contain a non-empty problems list"]

    errors: list[str] = []
    seen_numbers: set[int] = set()
    seen_slugs: set[str] = set()
    seen_paths: set[str] = set()

    for index, problem in enumerate(problems):
        if not isinstance(problem, dict):
            errors.append(f"problem at index {index} must be a mapping")
            continue

        errors.extend(validate_problem(problem, index))
        number = problem.get("number")
        slug = problem.get("slug")
        if isinstance(number, int):
            if number in seen_numbers:
                errors.append(f"duplicate number: {number}")
            seen_numbers.add(number)
        if isinstance(slug, str):
            if slug in seen_slugs:
                errors.append(f"duplicate slug: {slug}")
            seen_slugs.add(slug)

        for path_key in ("solution_path", "doc_path", "test_path"):
            path = problem.get(path_key)
            if isinstance(path, str):
                if path in seen_paths:
                    errors.append(f"duplicate path: {path}")
                seen_paths.add(path)

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate LeetCode 150 metadata.")
    parser.add_argument("metadata", type=Path, help="Path to top_interview_150.yaml")
    args = parser.parse_args(argv)

    try:
        metadata = load_metadata(args.metadata)
    except MetadataError as error:
        print(error, file=sys.stderr)
        return 2

    errors = validate_metadata(metadata)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"OK: {len(metadata['problems'])} problems validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run validator tests**

Run:

```bash
python -m pytest tests/scripts/test_validate_metadata.py -q
```

Expected: PASS for all tests in `tests/scripts/test_validate_metadata.py`.

- [ ] **Step 3: Commit validator**

Run:

```bash
git add scripts/validate_metadata.py tests/scripts/test_validate_metadata.py scripts/__init__.py
git commit -m "Validate tutorial metadata before scaffold generation" -m "Metadata needs schema, uniqueness, and path checks before any generated files can be trusted.\n\nConfidence: high\nScope-risk: narrow\nTested: python -m pytest tests/scripts/test_validate_metadata.py -q\nNot-tested: Full 150-problem metadata not added yet"
```

---

### Task 4: Scaffold Generator Tests

**Files:**
- Create: `tests/scripts/test_generate_scaffold.py`

- [ ] **Step 1: Write generator tests**

Write `tests/scripts/test_generate_scaffold.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.generate_scaffold import generate_scaffold


def sample_metadata() -> dict:
    return {
        "problems": [
            {
                "number": 1,
                "title": "Two Sum",
                "slug": "two-sum",
                "difficulty": "Easy",
                "official_group": "Array / String",
                "pattern_group": "Hash Table",
                "patterns": ["hash-map", "complement-lookup"],
                "leetcode_url": "https://leetcode.com/problems/two-sum/",
                "method_name": "twoSum",
                "signature": "def twoSum(self, nums: list[int], target: int) -> list[int]",
                "solution_path": "solutions/hash_table/p001_two_sum.py",
                "doc_path": "docs/problems/hash_table/p001_two_sum.md",
                "test_path": "tests/hash_table/test_p001_two_sum.py",
                "examples": [
                    {"input": {"nums": [2, 7, 11, 15], "target": 9}, "output": [0, 1]},
                ],
                "constraints_summary": "Exactly one valid answer exists.",
            }
        ]
    }


def test_generate_scaffold_creates_problem_files(tmp_path: Path) -> None:
    created = generate_scaffold(sample_metadata(), tmp_path, force=False)

    assert tmp_path.joinpath("docs/problems/hash_table/p001_two_sum.md").exists()
    assert tmp_path.joinpath("solutions/hash_table/p001_two_sum.py").exists()
    assert tmp_path.joinpath("tests/hash_table/test_p001_two_sum.py").exists()
    assert tmp_path.joinpath("docs/official-order.md").exists()
    assert tmp_path.joinpath("docs/pattern-roadmap.md").exists()
    assert "solutions/hash_table/p001_two_sum.py" in created


def test_generate_scaffold_does_not_overwrite_existing_file(tmp_path: Path) -> None:
    solution_path = tmp_path / "solutions/hash_table/p001_two_sum.py"
    solution_path.parent.mkdir(parents=True)
    solution_path.write_text("custom content\n", encoding="utf-8")

    generate_scaffold(sample_metadata(), tmp_path, force=False)

    assert solution_path.read_text(encoding="utf-8") == "custom content\n"


def test_generate_scaffold_overwrites_with_force(tmp_path: Path) -> None:
    solution_path = tmp_path / "solutions/hash_table/p001_two_sum.py"
    solution_path.parent.mkdir(parents=True)
    solution_path.write_text("custom content\n", encoding="utf-8")

    generate_scaffold(sample_metadata(), tmp_path, force=True)

    assert "class Solution" in solution_path.read_text(encoding="utf-8")


def test_generated_test_file_is_skipped_until_implemented(tmp_path: Path) -> None:
    generate_scaffold(sample_metadata(), tmp_path, force=False)

    test_content = tmp_path.joinpath("tests/hash_table/test_p001_two_sum.py").read_text(encoding="utf-8")

    assert "pytestmark = pytest.mark.skip" in test_content
    assert "Solution().twoSum" in test_content
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python -m pytest tests/scripts/test_generate_scaffold.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.generate_scaffold'`.

- [ ] **Step 3: Commit failing generator tests**

Run:

```bash
git add tests/scripts/test_generate_scaffold.py
git commit -m "Specify scaffold generation behavior before implementation" -m "Generator tests define deterministic file creation, non-overwrite behavior, force behavior, and skipped problem tests.\n\nConfidence: high\nScope-risk: narrow\nTested: python -m pytest tests/scripts/test_generate_scaffold.py -q fails because implementation is absent\nNot-tested: Generator implementation not added yet"
```

---

### Task 5: Scaffold Generator Implementation

**Files:**
- Create: `scripts/generate_scaffold.py`

- [ ] **Step 1: Implement scaffold generator**

Write `scripts/generate_scaffold.py`:

```python
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.validate_metadata import load_metadata, validate_metadata


def write_file(path: Path, content: str, force: bool) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def module_import_path(solution_path: str) -> str:
    return solution_path.removesuffix(".py").replace("/", ".")


def render_doc(problem: dict[str, Any]) -> str:
    examples = "\n".join(
        f"- Example {index}: input `{example['input']}`, output `{example['output']}`"
        for index, example in enumerate(problem["examples"], start=1)
    )
    patterns = ", ".join(problem["patterns"])
    return f"""# {problem['number']}. {problem['title']}

- Difficulty: {problem['difficulty']}
- LeetCode: {problem['leetcode_url']}
- Official Group: {problem['official_group']}
- Pattern Group: {problem['pattern_group']}
- Patterns: {patterns}

## Core Pattern

TODO: Explain the reusable invariant for this problem before filling the implementation.

## When To Use It

TODO: Describe the interview signals that suggest this pattern.

## Approach

TODO: Write the step-by-step reasoning in English.

## Correctness Sketch

TODO: Explain why the maintained invariant proves correctness.

## Complexity

- Time: TODO
- Space: TODO

## Common Pitfalls

TODO: List edge cases and implementation mistakes for this problem.

## Implementation

See `{problem['solution_path']}`.

## Tests

See `{problem['test_path']}`.

{examples}

## Follow-up Practice

TODO: Add closely related problems after this solution is completed.
"""


def render_solution(problem: dict[str, Any]) -> str:
    signature = problem["signature"]
    return f'''from __future__ import annotations


class Solution:
    """Solution placeholder for {problem['title']}.

    Tutorial: {problem['doc_path']}
    """

    {signature}:
        raise NotImplementedError("Solution not implemented yet")
'''


def render_test(problem: dict[str, Any]) -> str:
    import_path = module_import_path(problem["solution_path"])
    method_name = problem["method_name"]
    examples_repr = repr(problem["examples"])
    return f'''from __future__ import annotations

import pytest

from {import_path} import Solution

pytestmark = pytest.mark.skip(reason="Solution not implemented yet")


EXAMPLES = {examples_repr}


def test_official_examples() -> None:
    solution = Solution()
    for example in EXAMPLES:
        result = solution.{method_name}(**example["input"])
        assert result == example["output"]
'''


def render_official_order(problems: list[dict[str, Any]]) -> str:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for problem in problems:
        groups[problem["official_group"]].append(problem)

    lines = ["# Top Interview 150: Official Order", ""]
    for group, group_problems in groups.items():
        lines.extend([f"## {group}", ""])
        for problem in group_problems:
            lines.append(f"- [{problem['number']}. {problem['title']}]({problem['doc_path']}) ({problem['difficulty']})")
        lines.append("")
    return "\n".join(lines)


def render_pattern_roadmap(problems: list[dict[str, Any]]) -> str:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for problem in problems:
        groups[problem["pattern_group"]].append(problem)

    lines = ["# Pattern Roadmap", ""]
    for group in sorted(groups):
        lines.extend([f"## {group}", ""])
        for problem in groups[group]:
            tags = ", ".join(problem["patterns"])
            lines.append(f"- [{problem['number']}. {problem['title']}]({problem['doc_path']}) — {tags}")
        lines.append("")
    return "\n".join(lines)


def generate_scaffold(metadata: dict[str, Any], root: Path, force: bool = False) -> list[str]:
    problems = metadata["problems"]
    created: list[str] = []

    for problem in problems:
        outputs = {
            problem["doc_path"]: render_doc(problem),
            problem["solution_path"]: render_solution(problem),
            problem["test_path"]: render_test(problem),
        }
        for relative_path, content in outputs.items():
            if write_file(root / relative_path, content, force):
                created.append(relative_path)

    index_outputs = {
        "docs/official-order.md": render_official_order(problems),
        "docs/pattern-roadmap.md": render_pattern_roadmap(problems),
    }
    for relative_path, content in index_outputs.items():
        if write_file(root / relative_path, content, force=True):
            created.append(relative_path)

    return created


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the LeetCode 150 tutorial scaffold.")
    parser.add_argument("metadata", type=Path, help="Path to top_interview_150.yaml")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root")
    parser.add_argument("--force", action="store_true", help="Overwrite existing generated files")
    args = parser.parse_args(argv)

    metadata = load_metadata(args.metadata)
    errors = validate_metadata(metadata)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    created = generate_scaffold(metadata, args.root, force=args.force)
    print(f"Generated or updated {len(created)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run generator tests**

Run:

```bash
python -m pytest tests/scripts/test_generate_scaffold.py -q
```

Expected: PASS for all generator tests.

- [ ] **Step 3: Run all script tests**

Run:

```bash
python -m pytest tests/scripts -q
```

Expected: PASS for validator and generator tests.

- [ ] **Step 4: Commit generator**

Run:

```bash
git add scripts/generate_scaffold.py tests/scripts/test_generate_scaffold.py
git commit -m "Generate tutorial scaffold from metadata" -m "The repository needs deterministic docs, solution, test, and index generation from a single metadata source.\n\nConfidence: high\nScope-risk: moderate\nTested: python -m pytest tests/scripts -q\nNot-tested: Full 150-problem generation awaits verified metadata"
```

---

### Task 6: Official List Verification And Metadata

**Files:**
- Create: `data/top_interview_150.yaml`

- [ ] **Step 1: Verify the current official list**

Open the official LeetCode Top Interview 150 page and record the exact problem list, groups, titles, difficulties, URLs, and method names from LeetCode's current problem pages:

```bash
python - <<'PY'
print('Use the official LeetCode Top Interview 150 page as the source of truth before editing metadata.')
print('Record verification date: 2026-04-24')
PY
```

Expected: a dated verification note is available in the implementation notes or commit body.

- [ ] **Step 2: Create metadata file**

Write `data/top_interview_150.yaml` with this structure and repeat one complete entry per verified official problem:

```yaml
problems:
  - number: 1
    title: Two Sum
    slug: two-sum
    difficulty: Easy
    official_group: Array / String
    pattern_group: Hash Table
    patterns:
      - hash-map
      - complement-lookup
    leetcode_url: https://leetcode.com/problems/two-sum/
    method_name: twoSum
    signature: "def twoSum(self, nums: list[int], target: int) -> list[int]"
    solution_path: solutions/hash_table/p001_two_sum.py
    doc_path: docs/problems/hash_table/p001_two_sum.md
    test_path: tests/hash_table/test_p001_two_sum.py
    examples:
      - input:
          nums: [2, 7, 11, 15]
          target: 9
        output: [0, 1]
      - input:
          nums: [3, 2, 4]
          target: 6
        output: [1, 2]
      - input:
          nums: [3, 3]
          target: 6
        output: [0, 1]
    constraints_summary: Exactly one valid answer exists, and the same element cannot be used twice.
```

For every additional problem, preserve these rules:

```text
number: official LeetCode problem number as an integer
title: official English title
slug: official URL slug
solution_path: solutions/<pattern_dir>/pNNN_<slug_with_underscores>.py
doc_path: docs/problems/<pattern_dir>/pNNN_<slug_with_underscores>.md
test_path: tests/<pattern_dir>/test_pNNN_<slug_with_underscores>.py
method_name: LeetCode Python method name
signature: full Python method signature string including `self`
examples: official examples normalized into input/output mappings
```

- [ ] **Step 3: Validate metadata while filling it**

Run repeatedly while editing:

```bash
python scripts/validate_metadata.py data/top_interview_150.yaml
```

Expected after the final edit:

```text
OK: 150 problems validated
```

- [ ] **Step 4: Commit verified metadata**

Run:

```bash
git add data/top_interview_150.yaml
git commit -m "Record verified Top Interview 150 metadata" -m "The scaffold generator needs a canonical metadata source for all official Top Interview 150 problems.\n\nConstraint: Problem list verified against the official LeetCode Top Interview 150 page on 2026-04-24\nConfidence: medium\nScope-risk: moderate\nTested: python scripts/validate_metadata.py data/top_interview_150.yaml\nNot-tested: Generated scaffold not created in this commit"
```

---

### Task 7: Generate Full Scaffold

**Files:**
- Generate: `docs/official-order.md`
- Generate: `docs/pattern-roadmap.md`
- Generate: `docs/problems/<pattern>/pNNN_slug.md`
- Generate: `solutions/<pattern>/pNNN_slug.py`
- Generate: `tests/<pattern>/test_pNNN_slug.py`

- [ ] **Step 1: Run scaffold generation**

Run:

```bash
python scripts/generate_scaffold.py data/top_interview_150.yaml --root .
```

Expected output:

```text
Generated or updated 452 files
```

The exact count is 150 docs + 150 solutions + 150 tests + 2 indexes when no generated files existed before.

- [ ] **Step 2: Validate generated metadata paths still match files**

Run:

```bash
python scripts/validate_metadata.py data/top_interview_150.yaml
```

Expected:

```text
OK: 150 problems validated
```

- [ ] **Step 3: Run full pytest suite**

Run:

```bash
python -m pytest
```

Expected: script tests pass and problem tests are skipped because generated solutions raise `NotImplementedError`.

- [ ] **Step 4: Inspect representative generated files**

Run:

```bash
sed -n '1,80p' docs/official-order.md
sed -n '1,80p' docs/pattern-roadmap.md
sed -n '1,80p' docs/problems/hash_table/p001_two_sum.md
sed -n '1,80p' solutions/hash_table/p001_two_sum.py
sed -n '1,80p' tests/hash_table/test_p001_two_sum.py
```

Expected: files contain English tutorial sections, solution placeholder, skipped pytest examples, and working links.

- [ ] **Step 5: Commit generated scaffold**

Run:

```bash
git add docs/official-order.md docs/pattern-roadmap.md docs/problems solutions tests
git commit -m "Generate the Top Interview 150 tutorial scaffold" -m "The verified metadata now drives the complete docs, solutions, tests, and study indexes for the tutorial.\n\nConfidence: high\nScope-risk: broad\nTested: python scripts/validate_metadata.py data/top_interview_150.yaml; python -m pytest\nNot-tested: Individual problem solutions are intentionally unimplemented"
```

---

### Task 8: Final Documentation Polish

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README with generated paths and counts**

Modify `README.md` so it includes this section after `Study Routes`:

```markdown
## Repository Map

- `data/top_interview_150.yaml`: canonical metadata for all 150 problems.
- `docs/problems/`: English tutorial pages grouped by pattern.
- `solutions/`: Python `Solution` placeholders grouped by pattern.
- `tests/`: skipped pytest examples for unimplemented problems.
- `scripts/validate_metadata.py`: validates metadata consistency.
- `scripts/generate_scaffold.py`: regenerates missing scaffold files.
```

- [ ] **Step 2: Run validation commands**

Run:

```bash
python scripts/validate_metadata.py data/top_interview_150.yaml
python -m pytest
```

Expected: metadata validates 150 problems; pytest passes with skipped unimplemented problem tests.

- [ ] **Step 3: Commit documentation polish**

Run:

```bash
git add README.md
git commit -m "Explain how to use the generated tutorial scaffold" -m "The repository needs concise user-facing guidance after the full scaffold exists.\n\nConfidence: high\nScope-risk: narrow\nTested: python scripts/validate_metadata.py data/top_interview_150.yaml; python -m pytest\nNot-tested: No additional manual rendering check performed"
```

---

### Task 9: Completion Verification

**Files:**
- Read-only verification of repository state.

- [ ] **Step 1: Check git status**

Run:

```bash
git status --short leetcode150
```

Expected: no uncommitted changes under `leetcode150`.

- [ ] **Step 2: Run final validation**

Run:

```bash
cd /Users/yangyang/ai_projs/math/leetcode150
python scripts/validate_metadata.py data/top_interview_150.yaml
python -m pytest
```

Expected: `OK: 150 problems validated`; pytest reports script tests passing and generated problem tests skipped.

- [ ] **Step 3: Summarize final evidence**

Report:

```text
Changed files: README.md, pyproject.toml, .gitignore, data/top_interview_150.yaml, scripts/, docs/, solutions/, tests/
Verification: metadata validator passed for 150 problems; pytest passed with unimplemented problem tests skipped
Remaining risks: Official LeetCode examples may need future adjustment for mutation or multiple valid output cases as individual solutions are implemented
```
