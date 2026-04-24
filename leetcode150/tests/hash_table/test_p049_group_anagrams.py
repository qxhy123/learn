from __future__ import annotations

import pytest

from solutions.hash_table.p049_group_anagrams import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'raw': '["eat","tea","tan","ate","nat","bat"]\n[""]\n["a"]'}, 'output': 'See official examples'}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().groupAnagrams(**example["input"])
    for example in EXAMPLES:
        result = solution.groupAnagrams(**example["input"])
        assert result == example["output"]
