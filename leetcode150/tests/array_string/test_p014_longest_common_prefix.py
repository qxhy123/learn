from __future__ import annotations

import pytest

from solutions.array_string.p014_longest_common_prefix import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'strs': ['flower', 'flow', 'flight']}, 'output': 'fl'}, {'input': {'strs': ['dog', 'racecar', 'car']}, 'output': ''}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().longestCommonPrefix(**example["input"])
    for example in EXAMPLES:
        result = solution.longestCommonPrefix(**example["input"])
        assert result == example["output"]
