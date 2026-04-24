from __future__ import annotations

import pytest

from solutions.linked_list.p061_rotate_list import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'head': [1, 2, 3, 4, 5], 'k': 2}, 'output': [4, 5, 1, 2, 3]}, {'input': {'head': [0, 1, 2], 'k': 4}, 'output': [2, 0, 1]}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().rotateRight(**example["input"])
    for example in EXAMPLES:
        result = solution.rotateRight(**example["input"])
        assert result == example["output"]
