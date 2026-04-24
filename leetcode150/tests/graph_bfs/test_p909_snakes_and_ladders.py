from __future__ import annotations

import pytest

from solutions.graph_bfs.p909_snakes_and_ladders import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'board': [[-1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1], [-1, 35, -1, -1, 13, -1], [-1, -1, -1, -1, -1, -1], [-1, 15, -1, -1, -1, -1]]}, 'output': 4}, {'input': {'board': [[-1, -1], [-1, 3]]}, 'output': 1}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().snakesAndLadders(**example["input"])
    for example in EXAMPLES:
        result = solution.snakesAndLadders(**example["input"])
        assert result == example["output"]
