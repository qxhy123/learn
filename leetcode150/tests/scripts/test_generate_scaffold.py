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


def test_generate_scaffold_preserves_existing_index_without_force(tmp_path: Path) -> None:
    official_order_path = tmp_path / "docs/official-order.md"
    official_order_path.parent.mkdir(parents=True)
    official_order_path.write_text("custom index\n", encoding="utf-8")

    generate_scaffold(sample_metadata(), tmp_path, force=False)

    assert official_order_path.read_text(encoding="utf-8") == "custom index\n"


def test_generate_scaffold_overwrites_existing_index_with_force(tmp_path: Path) -> None:
    official_order_path = tmp_path / "docs/official-order.md"
    official_order_path.parent.mkdir(parents=True)
    official_order_path.write_text("custom index\n", encoding="utf-8")

    generate_scaffold(sample_metadata(), tmp_path, force=True)

    official_order_content = official_order_path.read_text(encoding="utf-8")
    assert official_order_content != "custom index\n"
    assert "# Top Interview 150 in Official Order" in official_order_content


def test_generated_test_file_is_skipped_until_implemented(tmp_path: Path) -> None:
    generate_scaffold(sample_metadata(), tmp_path, force=False)

    test_content = tmp_path.joinpath("tests/hash_table/test_p001_two_sum.py").read_text(encoding="utf-8")

    assert "pytestmark = pytest.mark.skip" in test_content
    assert "Solution().twoSum" in test_content
