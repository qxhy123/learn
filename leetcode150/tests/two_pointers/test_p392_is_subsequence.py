from __future__ import annotations

import pytest

from solutions.two_pointers.p392_is_subsequence import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'s': 'abc', 't': 'ahbgdc'}, 'output': True}, {'input': {'s': 'axc', 't': 'ahbgdc'}, 'output': False}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().isSubsequence(**example["input"])
    for example in EXAMPLES:
        result = solution.isSubsequence(**example["input"])
        assert result == example["output"]
