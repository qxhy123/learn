from __future__ import annotations

import pytest

from solutions.hash_table.p242_valid_anagram import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'raw': '"anagram"\n"nagaram"\n"rat"\n"car"'}, 'output': 'See official examples'}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().isAnagram(**example["input"])
    for example in EXAMPLES:
        result = solution.isAnagram(**example["input"])
        assert result == example["output"]
