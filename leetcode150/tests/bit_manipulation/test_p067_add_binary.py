from __future__ import annotations

import pytest

from solutions.bit_manipulation.p067_add_binary import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'a': '11', 'b': '1'}, 'output': '100'}, {'input': {'a': '1010', 'b': '1011'}, 'output': '10101'}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().addBinary(**example["input"])
    for example in EXAMPLES:
        result = solution.addBinary(**example["input"])
        assert result == example["output"]
