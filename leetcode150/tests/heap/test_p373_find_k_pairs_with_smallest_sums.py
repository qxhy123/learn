from __future__ import annotations

import pytest

from solutions.heap.p373_find_k_pairs_with_smallest_sums import Solution

pytestmark = pytest.mark.skip(reason="Scaffold placeholder until the solution is implemented.")

EXAMPLES = [{'input': {'nums1': [1, 7, 11], 'nums2': [2, 4, 6], 'k': 3}, 'output': [[1, 2], [1, 4], [1, 6]]}, {'input': {'nums1': [1, 1, 2], 'nums2': [1, 2, 3], 'k': 2}, 'output': [[1, 1], [1, 1]]}]


def test_official_examples() -> None:
    solution = Solution()
    # Equivalent direct call form: Solution().kSmallestPairs(**example["input"])
    for example in EXAMPLES:
        result = solution.kSmallestPairs(**example["input"])
        assert result == example["output"]
