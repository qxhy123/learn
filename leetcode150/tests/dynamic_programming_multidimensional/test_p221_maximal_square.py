from __future__ import annotations

import pytest

from solutions.dynamic_programming_multidimensional.p221_maximal_square import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'matrix': [['1', '0', '1', '0', '0'], ['1', '0', '1', '1', '1'], ['1', '1', '1', '1', '1'], ['1', '0', '0', '1', '0']]}, 'output': 4}, {'input': {'matrix': [['0', '1'], ['1', '0']]}, 'output': 1}, {'input': {'matrix': [['0']]}, 'output': 0}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().maximalSquare(**example["input"])
    for example in EXAMPLES:
        result = solution.maximalSquare(**example["input"])
        assert result == example["output"]
