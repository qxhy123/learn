from __future__ import annotations

import pytest

from solutions.hash_table.p290_word_pattern import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'raw': '"abba"\n"dog cat cat dog"\n"abba"\n"dog cat cat fish"\n"aaaa"\n"dog cat cat dog"'}, 'output': 'See official examples'}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().wordPattern(**example["input"])
    for example in EXAMPLES:
        result = solution.wordPattern(**example["input"])
        assert result == example["output"]
