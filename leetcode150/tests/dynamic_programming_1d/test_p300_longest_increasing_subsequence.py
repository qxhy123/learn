from __future__ import annotations

import pytest

from solutions.dynamic_programming_1d.p300_longest_increasing_subsequence import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'nums': [10, 9, 2, 5, 3, 7, 101, 18]}, 'output': 4}, {'input': {'nums': [0, 1, 0, 3, 2, 3]}, 'output': 4}, {'input': {'nums': [7, 7, 7, 7, 7, 7, 7]}, 'output': 1}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().lengthOfLIS(**example["input"])
    for example in EXAMPLES:
        result = solution.lengthOfLIS(**example["input"])
        assert result == example["output"]
