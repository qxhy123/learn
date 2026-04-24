from __future__ import annotations

import pytest

from solutions.graph_general.p207_course_schedule import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'numCourses': 2, 'prerequisites': [[1, 0]]}, 'output': True}, {'input': {'numCourses': 2, 'prerequisites': [[1, 0], [0, 1]]}, 'output': False}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().canFinish(**example["input"])
    for example in EXAMPLES:
        result = solution.canFinish(**example["input"])
        assert result == example["output"]
