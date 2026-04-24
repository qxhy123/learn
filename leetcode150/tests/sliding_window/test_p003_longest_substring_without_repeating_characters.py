from __future__ import annotations

import pytest

from solutions.sliding_window.p003_longest_substring_without_repeating_characters import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'s': 'abcabcbb'}, 'output': 3}, {'input': {'s': 'bbbbb'}, 'output': 1}, {'input': {'s': 'pwwkew'}, 'output': 3}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().lengthOfLongestSubstring(**example["input"])
    for example in EXAMPLES:
        result = solution.lengthOfLongestSubstring(**example["input"])
        assert result == example["output"]
