from __future__ import annotations

import pytest

from solutions.array_string.p068_text_justification import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'words': ['This', 'is', 'an', 'example', 'of', 'text', 'justification.'], 'maxWidth': 16}, 'output': '[\n\xa0 \xa0"This \xa0 \xa0is \xa0 \xa0an",\n\xa0 \xa0"example \xa0of text",\n\xa0 \xa0"justification. \xa0"\n]'}, {'input': {'words': ['What', 'must', 'be', 'acknowledgment', 'shall', 'be'], 'maxWidth': 16}, 'output': '[\n\xa0 "What \xa0 must \xa0 be",\n\xa0 "acknowledgment \xa0",\n\xa0 "shall be \xa0 \xa0 \xa0 \xa0"\n]'}, {'input': {'words': ['Science', 'is', 'what', 'we', 'understand', 'well', 'enough', 'to', 'explain', 'to', 'a', 'computer.', 'Art', 'is', 'everything', 'else', 'we', 'do'], 'maxWidth': 20}, 'output': '[\n\xa0 "Science \xa0is \xa0what we",\n  "understand \xa0 \xa0 \xa0well",\n\xa0 "enough to explain to",\n\xa0 "a \xa0computer. \xa0Art is",\n\xa0 "everything \xa0else \xa0we",\n\xa0 "do \xa0 \xa0 \xa0 \xa0 \xa0 \xa0 \xa0 \xa0 \xa0"\n]'}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().fullJustify(**example["input"])
    for example in EXAMPLES:
        result = solution.fullJustify(**example["input"])
        assert result == example["output"]
