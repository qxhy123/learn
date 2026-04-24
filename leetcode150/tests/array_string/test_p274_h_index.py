from __future__ import annotations

import pytest

from solutions.array_string.p274_h_index import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'citations': [3, 0, 6, 1, 5]}, 'output': 3}, {'input': {'citations': [1, 3, 1]}, 'output': 1}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().hIndex(**example["input"])
    for example in EXAMPLES:
        result = solution.hIndex(**example["input"])
        assert result == example["output"]
