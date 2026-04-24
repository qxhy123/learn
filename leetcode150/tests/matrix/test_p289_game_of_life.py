from __future__ import annotations

import pytest

from solutions.matrix.p289_game_of_life import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'board': [[0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 0, 0]]}, 'output': [[0, 0, 0], [1, 0, 1], [0, 1, 1], [0, 1, 0]]}, {'input': {'board': [[1, 1], [1, 0]]}, 'output': [[1, 1], [1, 1]]}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().gameOfLife(**example["input"])
    for example in EXAMPLES:
        result = solution.gameOfLife(**example["input"])
        assert result == example["output"]
