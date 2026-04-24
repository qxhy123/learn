from __future__ import annotations

import pytest

from solutions.graph_general.p210_course_schedule_ii import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'numCourses': 2, 'prerequisites': [[1, 0]]}, 'output': [0, 1]}, {'input': {'numCourses': 4, 'prerequisites': [[1, 0], [2, 0], [3, 1], [3, 2]]}, 'output': [0, 2, 1, 3]}, {'input': {'numCourses': 1, 'prerequisites': []}, 'output': [0]}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().findOrder(**example["input"])
    for example in EXAMPLES:
        result = solution.findOrder(**example["input"])
        assert result == example["output"]
