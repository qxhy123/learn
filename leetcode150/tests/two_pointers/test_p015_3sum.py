from __future__ import annotations

from solutions.two_pointers.p015_3sum import Solution


def normalized(triplets: list[list[int]]) -> list[list[int]]:
    return sorted(sorted(triplet) for triplet in triplets)


def assert_triplets_equal(actual: list[list[int]], expected: list[list[int]]) -> None:
    assert normalized(actual) == normalized(expected)


def test_official_examples() -> None:
    solution = Solution()

    assert_triplets_equal(solution.threeSum([-1, 0, 1, 2, -1, -4]), [[-1, -1, 2], [-1, 0, 1]])
    assert_triplets_equal(solution.threeSum([0, 1, 1]), [])
    assert_triplets_equal(solution.threeSum([0, 0, 0]), [[0, 0, 0]])


def test_empty_and_too_short_inputs() -> None:
    solution = Solution()

    assert_triplets_equal(solution.threeSum([]), [])
    assert_triplets_equal(solution.threeSum([1, -1]), [])


def test_all_zeroes_return_one_triplet() -> None:
    solution = Solution()

    assert_triplets_equal(solution.threeSum([0, 0, 0, 0, 0]), [[0, 0, 0]])


def test_duplicate_heavy_input_returns_unique_triplets() -> None:
    solution = Solution()

    assert_triplets_equal(solution.threeSum([-2, 0, 0, 2, 2]), [[-2, 0, 2]])
    assert_triplets_equal(solution.threeSum([-2, -2, 0, 0, 2, 2]), [[-2, 0, 2]])


def test_multiple_distinct_triplets() -> None:
    solution = Solution()

    assert_triplets_equal(
        solution.threeSum([-4, -2, -2, -1, 0, 1, 2, 2, 3]),
        [[-4, 1, 3], [-4, 2, 2], [-2, -1, 3], [-2, 0, 2], [-1, 0, 1]],
    )
