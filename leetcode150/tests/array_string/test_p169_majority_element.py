from __future__ import annotations

import pytest

from solutions.array_string.p169_majority_element import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'nums': [3, 2, 3]}, 'output': 3}, {'input': {'nums': [2, 2, 1, 1, 1, 2, 2]}, 'output': 2}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().majorityElement(**example["input"])
    for example in EXAMPLES:
        result = solution.majorityElement(**example["input"])
        assert result == example["output"]
