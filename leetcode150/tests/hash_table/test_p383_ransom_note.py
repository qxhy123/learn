from __future__ import annotations

import pytest

from solutions.hash_table.p383_ransom_note import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'ransomNote': 'a', 'magazine': 'b'}, 'output': False}, {'input': {'ransomNote': 'aa', 'magazine': 'ab'}, 'output': False}, {'input': {'ransomNote': 'aa', 'magazine': 'aab'}, 'output': True}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().canConstruct(**example["input"])
    for example in EXAMPLES:
        result = solution.canConstruct(**example["input"])
        assert result == example["output"]
