from __future__ import annotations

import pytest

from solutions.linked_list.p146_lru_cache import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'raw': '["LRUCache","put","put","get","put","get","put","get","get","get"]\n[[2],[1,1],[2,2],[1],[3,3],[2],[4,4],[1],[3],[4]]'}, 'output': 'See official examples'}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().__init__(**example["input"])
    for example in EXAMPLES:
        result = solution.__init__(**example["input"])
        assert result == example["output"]
