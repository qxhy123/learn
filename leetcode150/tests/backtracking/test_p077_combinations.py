from __future__ import annotations

import pytest

from solutions.backtracking.p077_combinations import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'n': 4, 'k': 2}, 'output': [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]}, {'input': {'n': 1, 'k': 1}, 'output': [[1]]}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().combine(**example["input"])
    for example in EXAMPLES:
        result = solution.combine(**example["input"])
        assert result == example["output"]
