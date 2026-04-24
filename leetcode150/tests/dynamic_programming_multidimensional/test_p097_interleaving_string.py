from __future__ import annotations

import pytest

from solutions.dynamic_programming_multidimensional.p097_interleaving_string import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'s1': 'aabcc', 's2': 'dbbca', 's3': 'aadbbcbcac'}, 'output': True}, {'input': {'s1': 'aabcc', 's2': 'dbbca', 's3': 'aadbbbaccc'}, 'output': False}, {'input': {'s1': '', 's2': '', 's3': ''}, 'output': True}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().isInterleave(**example["input"])
    for example in EXAMPLES:
        result = solution.isInterleave(**example["input"])
        assert result == example["output"]
