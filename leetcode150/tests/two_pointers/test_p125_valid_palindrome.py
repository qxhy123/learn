from __future__ import annotations

from solutions.two_pointers.p125_valid_palindrome import Solution


def test_official_examples() -> None:
    solution = Solution()

    assert solution.isPalindrome("A man, a plan, a canal: Panama") is True
    assert solution.isPalindrome("race a car") is False
    assert solution.isPalindrome(" ") is True


def test_empty_after_filtering_is_palindrome() -> None:
    solution = Solution()

    assert solution.isPalindrome(".,,   :;") is True


def test_mixed_case_and_digits() -> None:
    solution = Solution()

    assert solution.isPalindrome("No 'x' in Nixon") is True
    assert solution.isPalindrome("1A2a1") is True


def test_detects_mismatch_after_filtering() -> None:
    solution = Solution()

    assert solution.isPalindrome("0P") is False
    assert solution.isPalindrome("ab@c") is False
