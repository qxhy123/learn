from __future__ import annotations

import pytest

from solutions.array_string.p058_length_of_last_word import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'s': 'Hello World'}, 'output': 5}, {'input': {'s': ' fly me to the moon '}, 'output': 4}, {'input': {'s': 'luffy is still joyboy'}, 'output': 6}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().lengthOfLastWord(**example["input"])
    for example in EXAMPLES:
        result = solution.lengthOfLastWord(**example["input"])
        assert result == example["output"]
