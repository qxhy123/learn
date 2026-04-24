from __future__ import annotations

import pytest

from solutions.array_string.p006_zigzag_conversion import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'s': 'PAYPALISHIRING', 'numRows': 3}, 'output': 'PAHNAPLSIIGYIR'}, {'input': {'s': 'PAYPALISHIRING', 'numRows': 4}, 'output': 'PINALSIGYAHRPI'}, {'input': {'s': 'A', 'numRows': 1}, 'output': 'A'}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().convert(**example["input"])
    for example in EXAMPLES:
        result = solution.convert(**example["input"])
        assert result == example["output"]
