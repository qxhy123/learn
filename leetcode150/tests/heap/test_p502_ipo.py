from __future__ import annotations

import pytest

from solutions.heap.p502_ipo import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'k': 2, 'w': 0, 'profits': [1, 2, 3], 'capital': [0, 1, 1]}, 'output': 4}, {'input': {'k': 3, 'w': 0, 'profits': [1, 2, 3], 'capital': [0, 1, 2]}, 'output': 6}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().findMaximizedCapital(**example["input"])
    for example in EXAMPLES:
        result = solution.findMaximizedCapital(**example["input"])
        assert result == example["output"]
