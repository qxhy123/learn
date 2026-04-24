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
