from __future__ import annotations

import pytest

from solutions.sliding_window.p076_minimum_window_substring import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'s': 'ADOBECODEBANC', 't': 'ABC'}, 'output': 'BANC'}, {'input': {'s': 'a', 't': 'a'}, 'output': 'a'}, {'input': {'s': 'a', 't': 'aa'}, 'output': ''}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().minWindow(**example["input"])
    for example in EXAMPLES:
        result = solution.minWindow(**example["input"])
        assert result == example["output"]
