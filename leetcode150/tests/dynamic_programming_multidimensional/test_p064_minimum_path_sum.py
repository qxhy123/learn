from __future__ import annotations

import pytest

from solutions.dynamic_programming_multidimensional.p064_minimum_path_sum import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'grid': [[1, 3, 1], [1, 5, 1], [4, 2, 1]]}, 'output': 7}, {'input': {'grid': [[1, 2, 3], [4, 5, 6]]}, 'output': 12}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().minPathSum(**example["input"])
    for example in EXAMPLES:
        result = solution.minPathSum(**example["input"])
        assert result == example["output"]
