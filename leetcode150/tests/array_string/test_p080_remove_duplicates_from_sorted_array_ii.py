from __future__ import annotations

import pytest

from solutions.array_string.p080_remove_duplicates_from_sorted_array_ii import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'nums': [1, 1, 1, 2, 2, 3]}, 'output': '5, nums = [1,1,2,2,3,_]'}, {'input': {'nums': [0, 0, 1, 1, 1, 1, 2, 3, 3]}, 'output': '7, nums = [0,0,1,1,2,3,3,_,_]'}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().removeDuplicates(**example["input"])
    for example in EXAMPLES:
        result = solution.removeDuplicates(**example["input"])
        assert result == example["output"]
