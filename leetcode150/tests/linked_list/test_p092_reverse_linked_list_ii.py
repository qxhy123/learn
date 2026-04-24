from __future__ import annotations

import pytest

from solutions.linked_list.p092_reverse_linked_list_ii import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'head': [1, 2, 3, 4, 5], 'left': 2, 'right': 4}, 'output': [1, 4, 3, 2, 5]}, {'input': {'head': [5], 'left': 1, 'right': 1}, 'output': [5]}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().reverseBetween(**example["input"])
    for example in EXAMPLES:
        result = solution.reverseBetween(**example["input"])
        assert result == example["output"]
