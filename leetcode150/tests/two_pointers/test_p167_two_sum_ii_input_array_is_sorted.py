from __future__ import annotations

from solutions.two_pointers.p167_two_sum_ii_input_array_is_sorted import Solution


def test_official_examples() -> None:
    solution = Solution()

    assert solution.twoSum([2, 7, 11, 15], 9) == [1, 2]
    assert solution.twoSum([2, 3, 4], 6) == [1, 3]
    assert solution.twoSum([-1, 0], -1) == [1, 2]


def test_negative_numbers_and_positive_target() -> None:
    solution = Solution()

    assert solution.twoSum([-5, -2, 1, 4, 9], 7) == [2, 5]


def test_minimal_two_element_input() -> None:
    solution = Solution()

    assert solution.twoSum([1, 2], 3) == [1, 2]


def test_duplicate_values_can_form_answer() -> None:
    solution = Solution()

    assert solution.twoSum([1, 2, 2, 5], 4) == [2, 3]
