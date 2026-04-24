from __future__ import annotations

import pytest

from solutions.dynamic_programming_1d.p139_word_break import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'s': 'leetcode', 'wordDict': ['leet', 'code']}, 'output': True}, {'input': {'s': 'applepenapple', 'wordDict': ['apple', 'pen']}, 'output': True}, {'input': {'s': 'catsandog', 'wordDict': ['cats', 'dog', 'sand', 'and', 'cat']}, 'output': False}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().wordBreak(**example["input"])
    for example in EXAMPLES:
        result = solution.wordBreak(**example["input"])
        assert result == example["output"]
