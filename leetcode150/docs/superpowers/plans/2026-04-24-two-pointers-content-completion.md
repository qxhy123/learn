# Two Pointers Content Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the `Two Pointers` topic with final English tutorials, recommended Python solutions, active tests, and visible completion status.

**Architecture:** Each problem is completed independently across its doc, solution, and test file. Metadata and roadmap updates are handled after the five problem implementations pass so progress markers reflect verified work.

**Tech Stack:** Python 3.11+, `pytest`, existing Markdown docs, existing YAML metadata.

---

## File Structure

- Modify `solutions/two_pointers/p125_valid_palindrome.py`: implement `isPalindrome` using raw-string two pointers.
- Modify `tests/two_pointers/test_p125_valid_palindrome.py`: activate tests and add broad palindrome cases.
- Modify `docs/problems/two_pointers/p125_valid_palindrome.md`: replace scaffold TODOs with detailed teaching content.
- Modify `solutions/two_pointers/p392_is_subsequence.py`: implement `isSubsequence` using two scan pointers.
- Modify `tests/two_pointers/test_p392_is_subsequence.py`: activate tests and add empty/repeated/order cases.
- Modify `docs/problems/two_pointers/p392_is_subsequence.md`: replace scaffold TODOs with detailed teaching content.
- Modify `solutions/two_pointers/p167_two_sum_ii_input_array_is_sorted.py`: implement sorted two-sum with left/right pointers and built-in generics.
- Modify `tests/two_pointers/test_p167_two_sum_ii_input_array_is_sorted.py`: activate tests and add negative/minimal/duplicate cases.
- Modify `docs/problems/two_pointers/p167_two_sum_ii_input_array_is_sorted.md`: replace scaffold TODOs with detailed teaching content.
- Modify `solutions/two_pointers/p011_container_with_most_water.py`: implement max-area two pointers and built-in generics.
- Modify `tests/two_pointers/test_p011_container_with_most_water.py`: activate tests and add monotonic/equal-height cases.
- Modify `docs/problems/two_pointers/p011_container_with_most_water.md`: replace scaffold TODOs with detailed teaching content.
- Modify `solutions/two_pointers/p015_3sum.py`: implement sort + fixed index + inner two pointers.
- Modify `tests/two_pointers/test_p015_3sum.py`: activate tests and normalize triplet output.
- Modify `docs/problems/two_pointers/p015_3sum.md`: replace scaffold TODOs with detailed teaching content.
- Modify `data/top_interview_150.yaml`: add `status: complete` and `completed_at: 2026-04-24` to the five Two Pointers entries only.
- Modify `docs/pattern-roadmap.md`: mark Two Pointers complete and show recommended order.
- Modify `README.md`: add completed-topic status section.

---

### Task 1: Valid Palindrome Completion

**Files:**
- Modify: `tests/two_pointers/test_p125_valid_palindrome.py`
- Modify: `solutions/two_pointers/p125_valid_palindrome.py`
- Modify: `docs/problems/two_pointers/p125_valid_palindrome.md`

- [ ] **Step 1: Replace the skipped test file with active coverage**

Write `tests/two_pointers/test_p125_valid_palindrome.py`:

```python
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
    assert solution.isPalindrome("ab@a") is False
```

- [ ] **Step 2: Run the test to verify it fails before implementation**

Run:

```bash
python -m pytest tests/two_pointers/test_p125_valid_palindrome.py -q
```

Expected: FAIL with `NotImplementedError: Implement the solution described in the tutorial.`

- [ ] **Step 3: Implement the recommended solution**

Write `solutions/two_pointers/p125_valid_palindrome.py`:

```python
from __future__ import annotations


class Solution:
    """See `docs/problems/two_pointers/p125_valid_palindrome.md`."""

    def isPalindrome(self, s: str) -> bool:
        left = 0
        right = len(s) - 1

        while left < right:
            while left < right and not s[left].isalnum():
                left += 1
            while left < right and not s[right].isalnum():
                right -= 1

            if s[left].lower() != s[right].lower():
                return False

            left += 1
            right -= 1

        return True
```

- [ ] **Step 4: Run the problem test to verify it passes**

Run:

```bash
python -m pytest tests/two_pointers/test_p125_valid_palindrome.py -q
```

Expected: PASS with `4 passed`.

- [ ] **Step 5: Replace the tutorial page with final content**

Write `docs/problems/two_pointers/p125_valid_palindrome.md`:

```markdown
# 125. Valid Palindrome

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/valid-palindrome/
- Official Group: Two Pointers
- Pattern Group: Two Pointers
- Patterns: two-pointers

## Core Pattern

Use symmetric pointers when a condition must hold between the beginning and the end of a sequence. Move each pointer inward after the current pair has been validated.

## Why Two Pointers Fits

A palindrome is defined by mirrored positions. After ignoring non-alphanumeric characters and normalizing case, the first meaningful character must match the last meaningful character, the second must match the second-to-last, and so on. Two pointers let us check those mirrored pairs without building a filtered copy of the string.

## Recommended Approach

1. Put `left` at the start of `s` and `right` at the end.
2. Move `left` forward while it points at a non-alphanumeric character.
3. Move `right` backward while it points at a non-alphanumeric character.
4. Compare the lowercase forms of the two meaningful characters.
5. If they differ, return `False` immediately.
6. Otherwise move both pointers inward and continue until they cross.
7. If every mirrored pair matches, return `True`.

## Alternative Approaches

A simple alternative is to build a filtered lowercase list and compare it with its reverse. That is easier to read but uses `O(n)` extra space. The in-place two-pointer scan is the interview-preferred version because it keeps the same linear time while using constant auxiliary space.

## Correctness Sketch

Maintain the invariant that every meaningful character pair outside the current `[left, right]` window has already been checked and matched. The skip loops remove characters that the problem says should not affect the palindrome decision. When both pointers reference meaningful characters, a mismatch proves no valid palindrome exists because those mirrored positions must be equal. If they match, shrinking the window preserves the invariant. When the pointers cross, every required mirrored pair has matched, so the string is a valid palindrome.

## Trace

For `"A man, a plan, a canal: Panama"`:

| Step | Left char | Right char | Action |
| --- | --- | --- | --- |
| 1 | `A` | `a` | Compare lowercase `a == a`, move inward |
| 2 | skip spaces/punctuation | `m` | Skip ignored characters until both sides are alphanumeric |
| 3 | `m` | `m` | Match and move inward |
| 4 | later pairs | later pairs | Continue matching mirrored letters |
| End | pointers cross |  | All meaningful pairs matched |

## Complexity

- Time: `O(n)` because each pointer moves across the string at most once.
- Space: `O(1)` because the algorithm stores only pointer indices.

## Common Pitfalls

- Filtering only spaces but not punctuation.
- Comparing characters without lowercasing them.
- Moving a pointer past the other pointer inside a skip loop.
- Allocating a filtered string when the interviewer asks for constant extra space.

## Implementation Notes

See `solutions/two_pointers/p125_valid_palindrome.py`. The key detail is that each skip loop checks `left < right` before reading the character.

## Tests

See `tests/two_pointers/test_p125_valid_palindrome.py`. The tests cover official examples, strings that become empty after filtering, mixed case, digits, punctuation, and mismatch detection after filtering.

## Interview Script

"I use two pointers because palindrome validity is symmetric. I skip non-alphanumeric characters on both sides, compare lowercase meaningful characters, and move inward. A mismatch immediately proves the string is not a palindrome; if the pointers cross, every mirrored pair matched."

## Review Questions

1. Why is it safe to ignore non-alphanumeric characters before comparing?
2. What invariant is preserved after each successful comparison?
3. Why does this solution use `O(1)` extra space?
4. What happens when the string contains no alphanumeric characters?

## Follow-up Practice

- Check if a string can become a palindrome after deleting at most one character.
- Validate palindromes in Unicode-aware text where normalization rules matter.
- Apply the same symmetric scan idea to arrays instead of strings.
```

- [ ] **Step 6: Verify no scaffold TODO remains for this problem**

Run:

```bash
rg -n "TODO|NotImplementedError|pytestmark" docs/problems/two_pointers/p125_valid_palindrome.md solutions/two_pointers/p125_valid_palindrome.py tests/two_pointers/test_p125_valid_palindrome.py
```

Expected: no matches.

- [ ] **Step 7: Commit Valid Palindrome**

Run:

```bash
git add docs/problems/two_pointers/p125_valid_palindrome.md solutions/two_pointers/p125_valid_palindrome.py tests/two_pointers/test_p125_valid_palindrome.py
git commit -m "Complete the Valid Palindrome tutorial entry" -m "The Two Pointers topic needs its first completed example to establish the detailed teaching and active-test style.\n\nConstraint: Keep one recommended implementation while documenting alternatives\nConfidence: high\nScope-risk: narrow\nTested: python -m pytest tests/two_pointers/test_p125_valid_palindrome.py -q\nNot-tested: Full test suite deferred until the topic batch is complete"
```

---

### Task 2: Is Subsequence Completion

**Files:**
- Modify: `tests/two_pointers/test_p392_is_subsequence.py`
- Modify: `solutions/two_pointers/p392_is_subsequence.py`
- Modify: `docs/problems/two_pointers/p392_is_subsequence.md`

- [ ] **Step 1: Replace the skipped test file with active coverage**

Write `tests/two_pointers/test_p392_is_subsequence.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails before implementation**

Run:

```bash
python -m pytest tests/two_pointers/test_p392_is_subsequence.py -q
```

Expected: FAIL with `NotImplementedError: Implement the solution described in the tutorial.`

- [ ] **Step 3: Implement the recommended solution**

Write `solutions/two_pointers/p392_is_subsequence.py`:

```python
from __future__ import annotations


class Solution:
    """See `docs/problems/two_pointers/p392_is_subsequence.md`."""

    def isSubsequence(self, s: str, t: str) -> bool:
        s_index = 0

        for char in t:
            if s_index == len(s):
                return True
            if s[s_index] == char:
                s_index += 1

        return s_index == len(s)
```

- [ ] **Step 4: Run the problem test to verify it passes**

Run:

```bash
python -m pytest tests/two_pointers/test_p392_is_subsequence.py -q
```

Expected: PASS with `5 passed`.

- [ ] **Step 5: Replace the tutorial page with final content**

Write `docs/problems/two_pointers/p392_is_subsequence.md`:

```markdown
# 392. Is Subsequence

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/is-subsequence/
- Official Group: Two Pointers
- Pattern Group: Two Pointers
- Patterns: two-pointers

## Core Pattern

When one sequence must appear inside another while preserving order, scan the larger sequence once and advance the smaller-sequence pointer only when a match is found.

## Why Two Pointers Fits

A subsequence does not require contiguous positions, but it does require relative order. That means each character in `s` must be matched by a later character in `t`. A pointer into `s` tracks the next required character, while the scan through `t` offers candidates in order.

## Recommended Approach

1. Start `s_index` at `0`, meaning the next needed character is `s[0]`.
2. Iterate through every character in `t` from left to right.
3. If all characters in `s` have already matched, return `True`.
4. When the current `t` character equals `s[s_index]`, advance `s_index`.
5. Ignore non-matching `t` characters because subsequences may skip characters.
6. At the end, return whether `s_index == len(s)`.

## Alternative Approaches

A recursive solution can express the same idea but uses call-stack space and is unnecessary. For many repeated queries against the same `t`, a preprocessing approach can store character positions and binary-search the next valid position for each character of `s`. For one query, the linear two-pointer scan is simpler and optimal.

## Correctness Sketch

Maintain the invariant that `s[:s_index]` has been matched as a subsequence of the part of `t` already scanned. If the next `t` character does not match `s[s_index]`, skipping it cannot hurt because a subsequence is allowed to ignore characters. If it matches, consuming it is safe because it is the earliest available match for the next required character. When the scan ends, all of `s` is a subsequence exactly when every required character has been consumed.

## Trace

For `s = "abc"`, `t = "ahbgdc"`:

| `t` char | Needed char | Action |
| --- | --- | --- |
| `a` | `a` | Match, advance to need `b` |
| `h` | `b` | Skip |
| `b` | `b` | Match, advance to need `c` |
| `g` | `c` | Skip |
| `d` | `c` | Skip |
| `c` | `c` | Match, all of `s` consumed |

## Complexity

- Time: `O(len(t))` for a single query because each character in `t` is scanned once.
- Space: `O(1)` because only the pointer into `s` is stored.

## Common Pitfalls

- Sorting either string, which destroys order information.
- Requiring characters to be contiguous, which would solve substring matching instead.
- Forgetting that the empty string is a subsequence of every string.
- Advancing the `s` pointer on a mismatch.

## Implementation Notes

See `solutions/two_pointers/p392_is_subsequence.py`. The implementation uses one explicit pointer for `s`; the loop over `t` acts as the second pointer.

## Tests

See `tests/two_pointers/test_p392_is_subsequence.py`. The tests cover official examples, empty strings, repeated characters, insufficient repeated matches, and order-sensitive false cases.

## Interview Script

"I scan the target string once while keeping a pointer to the next character needed from the subsequence. A mismatch is ignored, and a match advances the subsequence pointer. If that pointer reaches the end of `s`, every character appeared in order."

## Review Questions

1. Why does skipping a non-matching character in `t` never remove a needed solution?
2. Why is the empty string always a subsequence?
3. How would the approach change for many different `s` queries against one fixed `t`?
4. What invariant does `s_index` represent?

## Follow-up Practice

- Preprocess `t` for many subsequence queries.
- Count how many words in a list are subsequences of one string.
- Compare subsequence matching with substring matching.
```

- [ ] **Step 6: Verify no scaffold TODO remains for this problem**

Run:

```bash
rg -n "TODO|NotImplementedError|pytestmark" docs/problems/two_pointers/p392_is_subsequence.md solutions/two_pointers/p392_is_subsequence.py tests/two_pointers/test_p392_is_subsequence.py
```

Expected: no matches.

- [ ] **Step 7: Commit Is Subsequence**

Run:

```bash
git add docs/problems/two_pointers/p392_is_subsequence.md solutions/two_pointers/p392_is_subsequence.py tests/two_pointers/test_p392_is_subsequence.py
git commit -m "Complete the Is Subsequence tutorial entry" -m "The Two Pointers batch needs a sequence-scan example that teaches ordered matching without contiguity.\n\nConstraint: Keep the implementation optimized for a single query\nRejected: Preprocess target positions | useful for follow-up but unnecessary for this problem instance\nConfidence: high\nScope-risk: narrow\nTested: python -m pytest tests/two_pointers/test_p392_is_subsequence.py -q\nNot-tested: Full test suite deferred until the topic batch is complete"
```

---

### Task 3: Two Sum II Completion

**Files:**
- Modify: `tests/two_pointers/test_p167_two_sum_ii_input_array_is_sorted.py`
- Modify: `solutions/two_pointers/p167_two_sum_ii_input_array_is_sorted.py`
- Modify: `docs/problems/two_pointers/p167_two_sum_ii_input_array_is_sorted.md`

- [ ] **Step 1: Replace the skipped test file with active coverage**

Write `tests/two_pointers/test_p167_two_sum_ii_input_array_is_sorted.py`:

```python
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

    assert solution.twoSum([1, 2, 2, 3], 4) == [2, 3]
```

- [ ] **Step 2: Run the test to verify it fails before implementation**

Run:

```bash
python -m pytest tests/two_pointers/test_p167_two_sum_ii_input_array_is_sorted.py -q
```

Expected: FAIL with `NotImplementedError: Implement the solution described in the tutorial.`

- [ ] **Step 3: Implement the recommended solution**

Write `solutions/two_pointers/p167_two_sum_ii_input_array_is_sorted.py`:

```python
from __future__ import annotations


class Solution:
    """See `docs/problems/two_pointers/p167_two_sum_ii_input_array_is_sorted.md`."""

    def twoSum(self, numbers: list[int], target: int) -> list[int]:
        left = 0
        right = len(numbers) - 1

        while left < right:
            current_sum = numbers[left] + numbers[right]
            if current_sum == target:
                return [left + 1, right + 1]
            if current_sum < target:
                left += 1
            else:
                right -= 1

        raise ValueError("Input must contain exactly one solution")
```

- [ ] **Step 4: Run the problem test to verify it passes**

Run:

```bash
python -m pytest tests/two_pointers/test_p167_two_sum_ii_input_array_is_sorted.py -q
```

Expected: PASS with `4 passed`.

- [ ] **Step 5: Replace the tutorial page with final content**

Write `docs/problems/two_pointers/p167_two_sum_ii_input_array_is_sorted.md`:

```markdown
# 167. Two Sum II - Input Array Is Sorted

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/
- Official Group: Two Pointers
- Pattern Group: Two Pointers
- Patterns: two-pointers, sum

## Core Pattern

When a sorted array asks for a pair with a target sum, compare the smallest and largest remaining candidates. Move the side that is guaranteed not to help.

## Why Two Pointers Fits

The array is sorted in nondecreasing order. If `numbers[left] + numbers[right]` is too small, every pair using `numbers[left]` with an index smaller than `right` is also too small, so `left` can move right. If the sum is too large, every pair using `numbers[right]` with an index larger than `left` is also too large, so `right` can move left.

## Recommended Approach

1. Set `left = 0` and `right = len(numbers) - 1`.
2. Compute `current_sum = numbers[left] + numbers[right]`.
3. If the sum equals `target`, return the 1-indexed pair `[left + 1, right + 1]`.
4. If the sum is smaller than `target`, increment `left` to increase the sum.
5. If the sum is larger than `target`, decrement `right` to decrease the sum.
6. Continue until the answer is found.

## Alternative Approaches

A hash map can solve the unsorted version in linear time, but it uses extra space and ignores the sorted input. Binary-searching the complement for every index gives `O(n log n)` time. The two-pointer method uses the sorted order directly and achieves `O(n)` time with `O(1)` space.

## Correctness Sketch

At each step, the answer must lie within the current `[left, right]` window. If the current sum is too small, `numbers[left]` cannot pair with any remaining value at or left of `right` to reach the target, because those values are no larger than `numbers[right]`. Therefore discarding `left` is safe. The too-large case symmetrically proves that discarding `right` is safe. Since each move discards only impossible candidates and the problem guarantees one answer, the algorithm eventually returns the correct pair.

## Trace

For `numbers = [2, 7, 11, 15]`, `target = 9`:

| Left | Right | Sum | Action |
| --- | --- | --- | --- |
| `2` at index 1 | `15` at index 4 | `17` | Too large, move `right` left |
| `2` at index 1 | `11` at index 3 | `13` | Too large, move `right` left |
| `2` at index 1 | `7` at index 2 | `9` | Return `[1, 2]` |

## Complexity

- Time: `O(n)` because each pointer moves inward at most `n` times total.
- Space: `O(1)` because no auxiliary data structure is needed.

## Common Pitfalls

- Returning zero-based indices instead of 1-indexed positions.
- Using a hash map and missing the constant-space advantage of sorted input.
- Moving both pointers when the sum is not equal to the target.
- Forgetting that duplicate values can be the correct pair.

## Implementation Notes

See `solutions/two_pointers/p167_two_sum_ii_input_array_is_sorted.py`. The implementation raises `ValueError` only as defensive code; valid LeetCode inputs contain exactly one solution.

## Tests

See `tests/two_pointers/test_p167_two_sum_ii_input_array_is_sorted.py`. The tests cover official examples, negative values, minimal input, and duplicate values forming the answer.

## Interview Script

"Because the array is sorted, I start with the smallest and largest values. If their sum is too small, the smaller value cannot work with anything else, so I move left. If the sum is too large, the larger value cannot work with anything else, so I move right. That discards impossible pairs until the guaranteed answer is found."

## Review Questions

1. Why does sorted order make it safe to move only one pointer?
2. Why must the returned indices be shifted by one?
3. How does this differ from the unsorted Two Sum problem?
4. Why do duplicates not require special handling here?

## Follow-up Practice

- Solve the unsorted Two Sum problem with a hash map.
- Count pairs with sum less than a target in a sorted array.
- Extend the idea to 3Sum by fixing one number and scanning the rest.
```

- [ ] **Step 6: Verify no scaffold TODO remains for this problem**

Run:

```bash
rg -n "TODO|NotImplementedError|pytestmark|List\[" docs/problems/two_pointers/p167_two_sum_ii_input_array_is_sorted.md solutions/two_pointers/p167_two_sum_ii_input_array_is_sorted.py tests/two_pointers/test_p167_two_sum_ii_input_array_is_sorted.py
```

Expected: no matches.

- [ ] **Step 7: Commit Two Sum II**

Run:

```bash
git add docs/problems/two_pointers/p167_two_sum_ii_input_array_is_sorted.md solutions/two_pointers/p167_two_sum_ii_input_array_is_sorted.py tests/two_pointers/test_p167_two_sum_ii_input_array_is_sorted.py
git commit -m "Complete the Two Sum II tutorial entry" -m "The Two Pointers batch needs a sorted-pair example that demonstrates monotonic candidate elimination.\n\nConstraint: Preserve LeetCode's 1-indexed return contract\nRejected: Hash map implementation | ignores the sorted-input constant-space advantage\nConfidence: high\nScope-risk: narrow\nTested: python -m pytest tests/two_pointers/test_p167_two_sum_ii_input_array_is_sorted.py -q\nNot-tested: Full test suite deferred until the topic batch is complete"
```

---

### Task 4: Container With Most Water Completion

**Files:**
- Modify: `tests/two_pointers/test_p011_container_with_most_water.py`
- Modify: `solutions/two_pointers/p011_container_with_most_water.py`
- Modify: `docs/problems/two_pointers/p011_container_with_most_water.md`

- [ ] **Step 1: Replace the skipped test file with active coverage**

Write `tests/two_pointers/test_p011_container_with_most_water.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails before implementation**

Run:

```bash
python -m pytest tests/two_pointers/test_p011_container_with_most_water.py -q
```

Expected: FAIL with `NameError: name 'List' is not defined` or `NotImplementedError: Implement the solution described in the tutorial.`

- [ ] **Step 3: Implement the recommended solution**

Write `solutions/two_pointers/p011_container_with_most_water.py`:

```python
from __future__ import annotations


class Solution:
    """See `docs/problems/two_pointers/p011_container_with_most_water.md`."""

    def maxArea(self, height: list[int]) -> int:
        left = 0
        right = len(height) - 1
        best_area = 0

        while left < right:
            width = right - left
            current_height = min(height[left], height[right])
            best_area = max(best_area, width * current_height)

            if height[left] < height[right]:
                left += 1
            else:
                right -= 1

        return best_area
```

- [ ] **Step 4: Run the problem test to verify it passes**

Run:

```bash
python -m pytest tests/two_pointers/test_p011_container_with_most_water.py -q
```

Expected: PASS with `4 passed`.

- [ ] **Step 5: Replace the tutorial page with final content**

Write `docs/problems/two_pointers/p011_container_with_most_water.md`:

```markdown
# 11. Container With Most Water

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/container-with-most-water/
- Official Group: Two Pointers
- Pattern Group: Two Pointers
- Patterns: two-pointers

## Core Pattern

When the score depends on two ends and a shrinking distance, evaluate the widest remaining pair and move the side that limits the score.

## Why Two Pointers Fits

The container area is `width * min(left_height, right_height)`. Starting with the widest possible width gives the best distance. Once a pair is evaluated, moving the taller side cannot improve the limiting height, because the shorter side still caps the area and the width gets smaller. Only moving the shorter side can possibly find a taller limiting wall.

## Recommended Approach

1. Put `left` at the first bar and `right` at the last bar.
2. Compute the area between them.
3. Update the best area seen so far.
4. Move the pointer at the shorter bar inward.
5. If both bars have equal height, moving either side is safe; this implementation moves `right`.
6. Continue until the pointers meet.

## Alternative Approaches

The brute-force approach checks every pair of bars, which takes `O(n^2)` time. A dynamic-programming table is unnecessary because the key decision is local and monotonic: once width shrinks, the only hope is to improve the limiting height. The two-pointer method captures that directly.

## Correctness Sketch

Consider a pair `(left, right)` where `height[left] <= height[right]`. Any container using `left` with a smaller right index has less width and a height no greater than `height[left]`, so it cannot beat the current pair. Therefore, after evaluating `(left, right)`, it is safe to discard `left`. The symmetric argument applies when the right bar is shorter. The algorithm evaluates one representative before discarding each impossible boundary, so the maximum area is never skipped.

## Trace

For `[1, 8, 6, 2, 5, 4, 8, 3, 7]`:

| Left height | Right height | Width | Area | Move |
| --- | --- | --- | --- | --- |
| `1` | `7` | `8` | `8` | Move left, shorter side |
| `8` | `7` | `7` | `49` | Move right, shorter side |
| `8` | `3` | `6` | `18` | Move right |
| `8` | `8` | `5` | `40` | Move right on tie |

The best area remains `49`.

## Complexity

- Time: `O(n)` because one pointer moves on every iteration.
- Space: `O(1)` because only a few integers are stored.

## Common Pitfalls

- Moving the taller side and losing the dominance argument.
- Forgetting that width shrinks as pointers move inward.
- Trying to sort heights, which destroys the original positions and widths.
- Using the taller height instead of the shorter height in the area formula.

## Implementation Notes

See `solutions/two_pointers/p011_container_with_most_water.py`. The implementation keeps the area formula explicit: `width * min(height[left], height[right])`.

## Tests

See `tests/two_pointers/test_p011_container_with_most_water.py`. The tests cover official examples, two-bar input, monotonic height arrays, and equal-height arrays.

## Interview Script

"I start with the widest container. The shorter wall limits the current area, and keeping that wall while reducing width cannot improve the answer. So after checking a pair, I move the shorter side inward and keep the best area seen."

## Review Questions

1. Why is moving the shorter side the only move that can improve the answer?
2. Why would sorting the heights break the problem?
3. What does the width represent in the area formula?
4. Why is the brute-force solution `O(n^2)`?

## Follow-up Practice

- Trapping Rain Water, which also reasons about boundary heights.
- Maximize a score formed by two boundary values and distance.
- Prove dominance arguments for other two-pointer problems.
```

- [ ] **Step 6: Verify no scaffold TODO remains for this problem**

Run:

```bash
rg -n "TODO|NotImplementedError|pytestmark|List\[" docs/problems/two_pointers/p011_container_with_most_water.md solutions/two_pointers/p011_container_with_most_water.py tests/two_pointers/test_p011_container_with_most_water.py
```

Expected: no matches.

- [ ] **Step 7: Commit Container With Most Water**

Run:

```bash
git add docs/problems/two_pointers/p011_container_with_most_water.md solutions/two_pointers/p011_container_with_most_water.py tests/two_pointers/test_p011_container_with_most_water.py
git commit -m "Complete the Container With Most Water tutorial entry" -m "The Two Pointers batch needs a dominance-argument example where moving the limiting side is the core idea.\n\nConstraint: Preserve original bar positions; sorting is invalid\nConfidence: high\nScope-risk: narrow\nTested: python -m pytest tests/two_pointers/test_p011_container_with_most_water.py -q\nNot-tested: Full test suite deferred until the topic batch is complete"
```

---

### Task 5: 3Sum Completion

**Files:**
- Modify: `tests/two_pointers/test_p015_3sum.py`
- Modify: `solutions/two_pointers/p015_3sum.py`
- Modify: `docs/problems/two_pointers/p015_3sum.md`

- [ ] **Step 1: Replace the skipped test file with active normalized coverage**

Write `tests/two_pointers/test_p015_3sum.py`:

```python
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
        [[-4, 1, 3], [-4, 2, 2], [-2, -1, 3], [-2, 0, 2]],
    )
```

- [ ] **Step 2: Run the test to verify it fails before implementation**

Run:

```bash
python -m pytest tests/two_pointers/test_p015_3sum.py -q
```

Expected: FAIL with `NotImplementedError: Implement the solution described in the tutorial.`

- [ ] **Step 3: Implement the recommended solution**

Write `solutions/two_pointers/p015_3sum.py`:

```python
from __future__ import annotations


class Solution:
    """See `docs/problems/two_pointers/p015_3sum.md`."""

    def threeSum(self, nums: list[int]) -> list[list[int]]:
        nums.sort()
        triplets: list[list[int]] = []

        for fixed_index in range(len(nums) - 2):
            fixed_value = nums[fixed_index]
            if fixed_index > 0 and fixed_value == nums[fixed_index - 1]:
                continue
            if fixed_value > 0:
                break

            left = fixed_index + 1
            right = len(nums) - 1

            while left < right:
                current_sum = fixed_value + nums[left] + nums[right]
                if current_sum == 0:
                    triplets.append([fixed_value, nums[left], nums[right]])
                    left += 1
                    right -= 1

                    while left < right and nums[left] == nums[left - 1]:
                        left += 1
                    while left < right and nums[right] == nums[right + 1]:
                        right -= 1
                elif current_sum < 0:
                    left += 1
                else:
                    right -= 1

        return triplets
```

- [ ] **Step 4: Run the problem test to verify it passes**

Run:

```bash
python -m pytest tests/two_pointers/test_p015_3sum.py -q
```

Expected: PASS with `5 passed`.

- [ ] **Step 5: Replace the tutorial page with final content**

Write `docs/problems/two_pointers/p015_3sum.md`:

```markdown
# 15. 3Sum

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/3sum/
- Official Group: Two Pointers
- Pattern Group: Two Pointers
- Patterns: two-pointers, sum

## Core Pattern

Reduce a k-sum problem by fixing one value, then solve the remaining sorted two-sum problem with inward-moving pointers.

## Why Two Pointers Fits

After sorting, the remaining two values for each fixed number live in a sorted suffix. If the three-number sum is too small, increasing the left pointer is the only move that can raise it. If the sum is too large, decreasing the right pointer is the only move that can lower it. Sorting also groups duplicates so they can be skipped deterministically.

## Recommended Approach

1. Sort `nums` in place.
2. Iterate `fixed_index` from left to right.
3. Skip a fixed value if it equals the previous fixed value.
4. Stop early when the fixed value is positive, because all later values are also positive.
5. Set `left = fixed_index + 1` and `right = len(nums) - 1`.
6. Compare `nums[fixed_index] + nums[left] + nums[right]` with zero.
7. Record a triplet on zero, then move both pointers and skip duplicate left/right values.
8. Move `left` on a negative sum and `right` on a positive sum.

## Alternative Approaches

A brute-force triple loop is `O(n^3)` and too slow. Fixing one value and using a hash set for the remaining two values can reach `O(n^2)` time, but duplicate handling becomes more awkward and extra space is needed per fixed value. Sorting plus two pointers keeps the same `O(n^2)` time while making uniqueness easier to enforce.

## Correctness Sketch

For each distinct fixed value, the inner two-pointer scan considers all viable pairs in the sorted suffix. If the sum is too small, every pair using the current `left` with a smaller or equal `right` is also too small, so advancing `left` is safe. If the sum is too large, every pair using the current `right` with a larger or equal `left` is also too large, so decrementing `right` is safe. When a zero-sum triplet is found, skipping equal neighboring values removes only duplicate triplets, not new value combinations. Since every fixed value is considered once and every suffix pair is either evaluated or safely discarded, all unique triplets are returned.

## Trace

For `[-1, 0, 1, 2, -1, -4]`, sort to `[-4, -1, -1, 0, 1, 2]`.

| Fixed | Left | Right | Sum | Action |
| --- | --- | --- | --- | --- |
| `-4` | `-1` | `2` | `-3` | Move left |
| `-4` | `-1` | `2` | `-3` | Move left |
| `-4` | `0` | `2` | `-2` | Move left until pointers cross |
| `-1` | `-1` | `2` | `0` | Record `[-1, -1, 2]`, skip duplicates |
| `-1` | `0` | `1` | `0` | Record `[-1, 0, 1]` |
| next `-1` |  |  |  | Skip duplicate fixed value |

## Complexity

- Time: `O(n^2)` after sorting. Sorting is `O(n log n)`, and the nested fixed-index plus two-pointer scan dominates.
- Space: `O(1)` auxiliary space beyond the output list, ignoring the implementation details of sorting.

## Common Pitfalls

- Forgetting to sort before using two pointers.
- Returning duplicate triplets because fixed, left, or right duplicates were not skipped.
- Skipping duplicates before recording a valid triplet.
- Comparing output order directly in tests even though triplet order is flexible.
- Continuing after the fixed value becomes positive.

## Implementation Notes

See `solutions/two_pointers/p015_3sum.py`. The implementation mutates `nums` by sorting it, which is acceptable for this LeetCode problem. If callers need the original order preserved, sort a copy instead.

## Tests

See `tests/two_pointers/test_p015_3sum.py`. The tests normalize triplet order and cover official examples, empty inputs, all zeroes, duplicate-heavy arrays, and multiple distinct triplets.

## Interview Script

"I sort the array, then fix one number and solve the remaining two-sum problem with left and right pointers. Sorting lets me move left when the sum is too small and right when it is too large. I skip duplicate fixed values and duplicate pointer values after recording a triplet so each value combination appears once."

## Review Questions

1. Why does sorting make the inner two-pointer scan valid?
2. Which duplicate cases must be skipped to avoid repeated triplets?
3. Why can the loop stop once the fixed value is positive?
4. Why is the time complexity `O(n^2)` instead of `O(n^3)`?
5. When would sorting a copy be preferable to sorting `nums` in place?

## Follow-up Practice

- 4Sum and general k-sum recursion.
- Count triplets with sum smaller than a target.
- Solve Two Sum II as the inner subproblem directly.
```

- [ ] **Step 6: Verify no scaffold TODO remains for this problem**

Run:

```bash
rg -n "TODO|NotImplementedError|pytestmark" docs/problems/two_pointers/p015_3sum.md solutions/two_pointers/p015_3sum.py tests/two_pointers/test_p015_3sum.py
```

Expected: no matches.

- [ ] **Step 7: Commit 3Sum**

Run:

```bash
git add docs/problems/two_pointers/p015_3sum.md solutions/two_pointers/p015_3sum.py tests/two_pointers/test_p015_3sum.py
git commit -m "Complete the 3Sum tutorial entry" -m "The Two Pointers batch needs a duplicate-aware k-sum example with normalized tests for flexible output ordering.\n\nConstraint: Return unique value triplets, not index triplets\nRejected: Hash-set inner loop | duplicate handling is less direct for teaching this topic\nConfidence: high\nScope-risk: narrow\nTested: python -m pytest tests/two_pointers/test_p015_3sum.py -q\nNot-tested: Full test suite deferred until the topic batch is complete"
```

---

### Task 6: Topic Progress Metadata And Reader Docs

**Files:**
- Modify: `data/top_interview_150.yaml`
- Modify: `docs/pattern-roadmap.md`
- Modify: `README.md`

- [ ] **Step 1: Add completion metadata to the five Two Pointers entries**

In `data/top_interview_150.yaml`, add these two fields to the entries with numbers `125`, `392`, `167`, `11`, and `15`:

```yaml
    status: complete
    completed_at: 2026-04-24
```

Place them near the existing metadata fields after `constraints_summary` for each touched problem. Do not add `status` to any other problem.

- [ ] **Step 2: Update the README completed-topic section**

Modify `README.md` so the top section contains this block after `Study Routes` and before `Repository Map`:

```markdown
## Completed Topics

Completed topics have final English tutorials, implemented Python solutions, and active pytest coverage.

- ✅ Two Pointers: 5 / 5 problems complete
```

- [ ] **Step 3: Replace the Two Pointers section in the pattern roadmap**

In `docs/pattern-roadmap.md`, replace the current `## Two Pointers` section with:

```markdown
## Two Pointers ✅ Complete

Recommended order:

1. [125. Valid Palindrome](docs/problems/two_pointers/p125_valid_palindrome.md) — symmetry scan, filtering, case normalization
2. [392. Is Subsequence](docs/problems/two_pointers/p392_is_subsequence.md) — ordered matching across two sequences
3. [167. Two Sum II - Input Array Is Sorted](docs/problems/two_pointers/p167_two_sum_ii_input_array_is_sorted.md) — sorted pair elimination
4. [11. Container With Most Water](docs/problems/two_pointers/p011_container_with_most_water.md) — dominance argument with shrinking width
5. [15. 3Sum](docs/problems/two_pointers/p015_3sum.md) — fixed value plus inner two-pointer scan
```

Do not change other roadmap sections.

- [ ] **Step 4: Validate metadata**

Run:

```bash
python scripts/validate_metadata.py data/top_interview_150.yaml
```

Expected: `OK: 150 problems validated`.

- [ ] **Step 5: Verify only five complete statuses exist**

Run:

```bash
python - <<'PY'
import yaml
with open('data/top_interview_150.yaml', encoding='utf-8') as file:
    problems = yaml.safe_load(file)['problems']
complete = [(p['number'], p['title']) for p in problems if p.get('status') == 'complete']
print(complete)
assert {number for number, _ in complete} == {11, 15, 125, 167, 392}
PY
```

Expected: prints the five completed Two Pointers problems and exits successfully.

- [ ] **Step 6: Commit progress docs**

Run:

```bash
git add data/top_interview_150.yaml docs/pattern-roadmap.md README.md
git commit -m "Mark the Two Pointers topic as complete" -m "The first completed topic needs metadata and reader-facing progress markers after its docs, solutions, and tests are active.\n\nConstraint: Mark only the five Two Pointers problems complete\nConfidence: high\nScope-risk: narrow\nTested: python scripts/validate_metadata.py data/top_interview_150.yaml; completion-status assertion script\nNot-tested: Full test suite deferred to final verification"
```

---

### Task 7: Topic And Full-Suite Verification

**Files:**
- Read-only verification across `docs/problems/two_pointers/`, `solutions/two_pointers/`, `tests/two_pointers/`, `data/top_interview_150.yaml`, `README.md`, and `docs/pattern-roadmap.md`.

- [ ] **Step 1: Run Two Pointers tests**

Run:

```bash
python -m pytest tests/two_pointers -q
```

Expected: all five active Two Pointers test files pass with no skips in this directory.

- [ ] **Step 2: Run full metadata validation**

Run:

```bash
python scripts/validate_metadata.py data/top_interview_150.yaml
```

Expected: `OK: 150 problems validated`.

- [ ] **Step 3: Verify no incomplete markers remain in completed topic files**

Run:

```bash
rg -n "TODO|NotImplementedError|pytestmark|List\[" docs/problems/two_pointers solutions/two_pointers tests/two_pointers
```

Expected: no matches.

- [ ] **Step 4: Run full test suite**

Run:

```bash
python -m pytest
```

Expected: full test suite passes. Two Pointers tests are active; unfinished topic tests remain skipped.

- [ ] **Step 5: Check repository status for this project**

Run:

```bash
git status --short .
```

Expected: no uncommitted changes in `leetcode150` after all commits.

- [ ] **Step 6: Final report**

Report this evidence:

```text
Changed files: five Two Pointers docs, five Two Pointers solutions, five Two Pointers tests, data/top_interview_150.yaml, docs/pattern-roadmap.md, README.md
Verification: python -m pytest tests/two_pointers -q; python scripts/validate_metadata.py data/top_interview_150.yaml; python -m pytest
Remaining risks: other 145 problems remain scaffolded and skipped by design
```
