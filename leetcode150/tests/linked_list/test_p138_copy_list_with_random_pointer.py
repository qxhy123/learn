from __future__ import annotations

import pytest

from solutions.linked_list.p138_copy_list_with_random_pointer import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'head': [[7, None], [13, 0], [11, 4], [10, 2], [1, 0]]}, 'output': [[7, None], [13, 0], [11, 4], [10, 2], [1, 0]]}, {'input': {'head': [[1, 1], [2, 1]]}, 'output': [[1, 1], [2, 1]]}, {'input': {'head': [[3, None], [3, 0], [3, None]]}, 'output': [[3, None], [3, 0], [3, None]]}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().__init__(**example["input"])
    for example in EXAMPLES:
        result = solution.__init__(**example["input"])
        assert result == example["output"]
