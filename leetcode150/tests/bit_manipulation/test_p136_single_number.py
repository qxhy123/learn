from __future__ import annotations

import pytest

from solutions.bit_manipulation.p136_single_number import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'raw': '[2,2,1]\n[4,1,2,1,2]\n[1]'}, 'output': 'See official examples'}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().singleNumber(**example["input"])
    for example in EXAMPLES:
        result = solution.singleNumber(**example["input"])
        assert result == example["output"]
