from __future__ import annotations

import pytest

from solutions.divide_conquer.p148_sort_list import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'head': [4, 2, 1, 3]}, 'output': [1, 2, 3, 4]}, {'input': {'head': [-1, 5, 3, 4, 0]}, 'output': [-1, 0, 3, 4, 5]}, {'input': {'head': []}, 'output': []}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().sortList(**example["input"])
    for example in EXAMPLES:
        result = solution.sortList(**example["input"])
        assert result == example["output"]
