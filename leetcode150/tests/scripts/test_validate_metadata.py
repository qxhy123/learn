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


def test_validate_metadata_rejects_non_int_number_without_crashing() -> None:
    problem = valid_problem() | {"number": "one"}

    errors = validate_metadata({"problems": [problem]})

    assert "problem at index 0 has invalid number: one" in errors


def test_load_metadata_rejects_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(MetadataError, match="metadata file does not exist"):
        load_metadata(missing_path)


def test_load_metadata_rejects_malformed_yaml(tmp_path: Path) -> None:
    metadata_path = tmp_path / "broken.yaml"
    metadata_path.write_text("problems: [\n", encoding="utf-8")

    with pytest.raises(MetadataError, match="failed to parse metadata YAML"):
        load_metadata(metadata_path)
