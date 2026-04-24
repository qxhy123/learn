from __future__ import annotations

import pytest

from solutions.array_string.p380_insert_delete_getrandom_o1 import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'raw': '["RandomizedSet","insert","remove","insert","getRandom","remove","insert","getRandom"]\n[[],[1],[2],[2],[],[1],[2],[]]'}, 'output': 'See official examples'}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().__init__(**example["input"])
    for example in EXAMPLES:
        result = solution.__init__(**example["input"])
        assert result == example["output"]
