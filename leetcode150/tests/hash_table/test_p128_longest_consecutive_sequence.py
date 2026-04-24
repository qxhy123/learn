from __future__ import annotations

import pytest

from solutions.hash_table.p128_longest_consecutive_sequence import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'nums': [100, 4, 200, 1, 3, 2]}, 'output': 4}, {'input': {'nums': [0, 3, 7, 2, 5, 8, 4, 6, 0, 1]}, 'output': 9}, {'input': {'nums': [1, 0, 1, 2]}, 'output': 3}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().longestConsecutive(**example["input"])
    for example in EXAMPLES:
        result = solution.longestConsecutive(**example["input"])
        assert result == example["output"]
