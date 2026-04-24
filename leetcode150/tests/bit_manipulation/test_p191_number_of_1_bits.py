from __future__ import annotations

import pytest

from solutions.bit_manipulation.p191_number_of_1_bits import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'raw': '11\n128\n2147483645'}, 'output': 'See official examples'}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().hammingWeight(**example["input"])
    for example in EXAMPLES:
        result = solution.hammingWeight(**example["input"])
        assert result == example["output"]
