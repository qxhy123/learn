from __future__ import annotations

from solutions.two_pointers.p392_is_subsequence import Solution


def test_official_examples() -> None:
    solution = Solution()

    assert solution.isSubsequence("abc", "ahbgdc") is True
    assert solution.isSubsequence("axc", "ahbgdc") is False


def test_empty_subsequence_always_matches() -> None:
    solution = Solution()

    assert solution.isSubsequence("", "anything") is True
    assert solution.isSubsequence("", "") is True


def test_non_empty_subsequence_cannot_match_empty_target() -> None:
    solution = Solution()

    assert solution.isSubsequence("a", "") is False


def test_repeated_characters_require_enough_ordered_matches() -> None:
    solution = Solution()

    assert solution.isSubsequence("aaa", "baaac") is True
    assert solution.isSubsequence("aaaa", "baaac") is False


def test_order_matters() -> None:
    solution = Solution()

    assert solution.isSubsequence("ace", "abcde") is True
    assert solution.isSubsequence("aec", "abcde") is False
