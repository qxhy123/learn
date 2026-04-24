from __future__ import annotations

import pytest

from solutions.backtracking.p046_permutations import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'nums': [1, 2, 3]}, 'output': [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]}, {'input': {'nums': [0, 1]}, 'output': [[0, 1], [1, 0]]}, {'input': {'nums': [1]}, 'output': [[1]]}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().permute(**example["input"])
    for example in EXAMPLES:
        result = solution.permute(**example["input"])
        assert result == example["output"]
