from __future__ import annotations

import pytest

from solutions.linked_list.p002_add_two_numbers import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'l1': [2, 4, 3], 'l2': [5, 6, 4]}, 'output': [7, 0, 8]}, {'input': {'l1': [0], 'l2': [0]}, 'output': [0]}, {'input': {'l1': [9, 9, 9, 9, 9, 9, 9], 'l2': [9, 9, 9, 9]}, 'output': [8, 9, 9, 9, 0, 0, 0, 1]}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().addTwoNumbers(**example["input"])
    for example in EXAMPLES:
        result = solution.addTwoNumbers(**example["input"])
        assert result == example["output"]
