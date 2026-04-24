from __future__ import annotations

import pytest

from solutions.bit_manipulation.p190_reverse_bits import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'raw': '43261596\n2147483644'}, 'output': 'See official examples'}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().reverseBits(**example["input"])
    for example in EXAMPLES:
        result = solution.reverseBits(**example["input"])
        assert result == example["output"]
