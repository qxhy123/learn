from __future__ import annotations

import pytest

from solutions.linked_list.p082_remove_duplicates_from_sorted_list_ii import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'head': [1, 2, 3, 3, 4, 4, 5]}, 'output': [1, 2, 5]}, {'input': {'head': [1, 1, 1, 2, 3]}, 'output': [2, 3]}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().deleteDuplicates(**example["input"])
    for example in EXAMPLES:
        result = solution.deleteDuplicates(**example["input"])
        assert result == example["output"]
