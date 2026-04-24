from __future__ import annotations

import pytest

from solutions.array_string.p026_remove_duplicates_from_sorted_array import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'nums': [1, 1, 2]}, 'output': '2, nums = [1,2,_]'}, {'input': {'nums': [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]}, 'output': '5, nums = [0,1,2,3,4,_,_,_,_,_]'}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().removeDuplicates(**example["input"])
    for example in EXAMPLES:
        result = solution.removeDuplicates(**example["input"])
        assert result == example["output"]
