from __future__ import annotations

import pytest

from solutions.backtracking.p039_combination_sum import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'candidates': [2, 3, 6, 7], 'target': 7}, 'output': [[2, 2, 3], [7]]}, {'input': {'candidates': [2, 3, 5], 'target': 8}, 'output': [[2, 2, 2, 2], [2, 3, 3], [3, 5]]}, {'input': {'candidates': [2], 'target': 1}, 'output': []}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().combinationSum(**example["input"])
    for example in EXAMPLES:
        result = solution.combinationSum(**example["input"])
        assert result == example["output"]
