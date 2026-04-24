from __future__ import annotations

import pytest

from solutions.binary_search.p162_find_peak_element import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'nums': [1, 2, 3, 1]}, 'output': 2}, {'input': {'nums': [1, 2, 1, 3, 5, 6, 4]}, 'output': 5}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().findPeakElement(**example["input"])
    for example in EXAMPLES:
        result = solution.findPeakElement(**example["input"])
        assert result == example["output"]
