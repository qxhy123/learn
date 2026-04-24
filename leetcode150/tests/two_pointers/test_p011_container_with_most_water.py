from __future__ import annotations

from solutions.two_pointers.p011_container_with_most_water import Solution


def test_official_examples() -> None:
    solution = Solution()

    assert solution.maxArea([1, 8, 6, 2, 5, 4, 8, 3, 7]) == 49
    assert solution.maxArea([1, 1]) == 1


def test_two_bars_only() -> None:
    solution = Solution()

    assert solution.maxArea([4, 9]) == 4


def test_monotonic_heights() -> None:
    solution = Solution()

    assert solution.maxArea([1, 2, 3, 4, 5]) == 6
    assert solution.maxArea([5, 4, 3, 2, 1]) == 6


def test_equal_heights() -> None:
    solution = Solution()

    assert solution.maxArea([5, 5, 5, 5]) == 15
