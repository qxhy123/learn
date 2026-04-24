from __future__ import annotations

import pytest

from solutions.array_string.p055_jump_game import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'nums': [2, 3, 1, 1, 4]}, 'output': True}, {'input': {'nums': [3, 2, 1, 0, 4]}, 'output': False}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().canJump(**example["input"])
    for example in EXAMPLES:
        result = solution.canJump(**example["input"])
        assert result == example["output"]
