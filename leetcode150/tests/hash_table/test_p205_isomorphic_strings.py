from __future__ import annotations

import pytest

from solutions.hash_table.p205_isomorphic_strings import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'raw': '"egg"\n"add"\n"foo"\n"bar"\n"paper"\n"title"'}, 'output': 'See official examples'}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().isIsomorphic(**example["input"])
    for example in EXAMPLES:
        result = solution.isIsomorphic(**example["input"])
        assert result == example["output"]
