from __future__ import annotations

import pytest

from solutions.intervals.p228_summary_ranges import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'nums': [0, 1, 2, 4, 5, 7]}, 'output': ['0->2', '4->5', '7']}, {'input': {'nums': [0, 2, 3, 4, 6, 8, 9]}, 'output': ['0', '2->4', '6', '8->9']}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().summaryRanges(**example["input"])
    for example in EXAMPLES:
        result = solution.summaryRanges(**example["input"])
        assert result == example["output"]
