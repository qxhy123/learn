from __future__ import annotations

import pytest

from solutions.array_string.p151_reverse_words_in_a_string import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'s': 'the sky is blue'}, 'output': 'blue is sky the'}, {'input': {'s': ' hello world '}, 'output': 'world hello'}, {'input': {'s': 'a good example'}, 'output': 'example good a'}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().reverseWords(**example["input"])
    for example in EXAMPLES:
        result = solution.reverseWords(**example["input"])
        assert result == example["output"]
