from __future__ import annotations

import pytest

from solutions.linked_list.p086_partition_list import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'head': [1, 4, 3, 2, 5, 2], 'x': 3}, 'output': [1, 2, 2, 4, 3, 5]}, {'input': {'head': [2, 1], 'x': 2}, 'output': [1, 2]}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().partition(**example["input"])
    for example in EXAMPLES:
        result = solution.partition(**example["input"])
        assert result == example["output"]
