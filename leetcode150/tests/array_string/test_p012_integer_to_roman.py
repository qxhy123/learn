from __future__ import annotations

import pytest

from solutions.array_string.p012_integer_to_roman import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'raw': '3749\n58\n1994'}, 'output': 'See official examples'}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().intToRoman(**example["input"])
    for example in EXAMPLES:
        result = solution.intToRoman(**example["input"])
        assert result == example["output"]
