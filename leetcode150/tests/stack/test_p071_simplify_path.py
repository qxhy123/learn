from __future__ import annotations

import pytest

from solutions.stack.p071_simplify_path import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'raw': '"/home/"\n"/home//foo/"\n"/home/user/Documents/../Pictures"\n"/../"\n"/.../a/../b/c/../d/./"'}, 'output': 'See official examples'}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().simplifyPath(**example["input"])
    for example in EXAMPLES:
        result = solution.simplifyPath(**example["input"])
        assert result == example["output"]
