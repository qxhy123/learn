from __future__ import annotations

import pytest

from solutions.dynamic_programming_multidimensional.p063_unique_paths_ii import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'obstacleGrid': [[0, 0, 0], [0, 1, 0], [0, 0, 0]]}, 'output': 2}, {'input': {'obstacleGrid': [[0, 1], [0, 0]]}, 'output': 1}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().uniquePathsWithObstacles(**example["input"])
    for example in EXAMPLES:
        result = solution.uniquePathsWithObstacles(**example["input"])
        assert result == example["output"]
