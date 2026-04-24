from __future__ import annotations

import pytest

from solutions.two_pointers.p167_two_sum_ii_input_array_is_sorted import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'numbers': [2, 7, 11, 15], 'target': 9}, 'output': [1, 2]}, {'input': {'numbers': [2, 3, 4], 'target': 6}, 'output': [1, 3]}, {'input': {'numbers': [-1, 0], 'target': -1}, 'output': [1, 2]}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().twoSum(**example["input"])
    for example in EXAMPLES:
        result = solution.twoSum(**example["input"])
        assert result == example["output"]
