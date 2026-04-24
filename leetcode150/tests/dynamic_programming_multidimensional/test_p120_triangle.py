from __future__ import annotations

import pytest

from solutions.dynamic_programming_multidimensional.p120_triangle import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'triangle': [[2], [3, 4], [6, 5, 7], [4, 1, 8, 3]]}, 'output': 11}, {'input': {'triangle': [[-10]]}, 'output': -10}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().minimumTotal(**example["input"])
    for example in EXAMPLES:
        result = solution.minimumTotal(**example["input"])
        assert result == example["output"]
