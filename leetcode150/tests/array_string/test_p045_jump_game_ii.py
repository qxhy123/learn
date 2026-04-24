from __future__ import annotations

import pytest

from solutions.array_string.p045_jump_game_ii import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'nums': [2, 3, 1, 1, 4]}, 'output': 2}, {'input': {'nums': [2, 3, 0, 1, 4]}, 'output': 2}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().jump(**example["input"])
    for example in EXAMPLES:
        result = solution.jump(**example["input"])
        assert result == example["output"]
