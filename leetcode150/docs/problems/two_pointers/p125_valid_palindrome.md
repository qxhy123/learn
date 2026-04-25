# 125. Valid Palindrome

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/valid-palindrome/
- Official Group: Two Pointers
- Pattern Group: Two Pointers
- Patterns: two-pointers

## First-Principles Explanation

### What The Problem Asks

The input is a string `s`. The string may contain letters, digits, spaces,
punctuation, symbols, and mixed casing. The question is not whether the raw
string reads the same forward and backward. The question is whether the string
reads the same forward and backward after applying two rules:

1. Ignore every character that is not alphanumeric.
2. Compare letters case-insensitively.

So the problem is really about a hidden sequence inside the input string: the
sequence of meaningful characters, all compared in the same case.

For example:

```text
s = "A man, a plan, a canal: Panama"
meaningful lowercase sequence = "amanaplanacanalpanama"
```

That normalized sequence is a palindrome, so the answer is `True`.

By contrast:

```text
s = "race a car"
meaningful lowercase sequence = "raceacar"
```

The normalized sequence is not the same from both ends, so the answer is
`False`.

The important first-principles move is to separate the physical input from the
logical sequence being tested. Spaces and punctuation can appear anywhere in the
physical string, but they do not exist in the logical palindrome check.

### Brute-Force Baseline

A direct baseline is to build the normalized sequence explicitly, then compare
it with its reverse:

```python
def is_palindrome_baseline(s: str) -> bool:
    cleaned = []

    for ch in s:
        if ch.isalnum():
            cleaned.append(ch.lower())

    return cleaned == cleaned[::-1]
```

This is a good way to understand the problem because it says exactly what is
being checked:

1. Scan the original string.
2. Keep only alphanumeric characters.
3. Lowercase those characters.
4. Ask whether the resulting list equals its reverse.

The baseline is correct and simple. Its cost is that it stores a second sequence
of up to `n` characters. If the input is long, this extra storage is unnecessary
because a palindrome check never needs the whole normalized sequence at once. It
only needs to compare mirrored pairs.

### Key Observation

A palindrome is determined by pairs from opposite ends:

```text
first character  == last character
second character == second-to-last character
third character  == third-to-last character
...
```

For this problem, those pairs are not necessarily the raw first and raw last
characters of `s`. They are the first and last meaningful characters after
filtering.

That gives the useful observation:

> Instead of constructing the full normalized string, find the next meaningful
> character from the left and the next meaningful character from the right, then
> compare just that pair.

If the pair differs, the answer is immediately `False`. No later character can
repair a mismatched mirrored pair. If the pair matches, those two characters are
fully proven and can be removed from future consideration.

This is why two pointers fit the problem. One pointer searches forward for the
next meaningful character. The other searches backward for the next meaningful
character. Together they simulate comparing the normalized string from both ends
without ever building that normalized string.

### Two-Pointer Filtering Invariant

Let `left` point somewhere near the front of the unresolved part of the original
string, and let `right` point somewhere near the back.

The invariant is:

> At the start of each main loop iteration, every meaningful mirrored pair
> outside the current `s[left:right + 1]` window has already been compared and
> matched. Therefore, if a mismatch still exists, it must be inside the current
> window.

The filtering steps preserve this invariant:

- If `s[left]` is not alphanumeric, it cannot be part of any mirrored pair, so
  moving `left` rightward discards no meaningful evidence.
- If `s[right]` is not alphanumeric, it cannot be part of any mirrored pair, so
  moving `right` leftward discards no meaningful evidence.
- After both pointers stop on meaningful characters, they represent the next
  unresolved mirrored pair in the normalized sequence.
- If those lowercase characters match, both are proven correct and the window
  can safely shrink inward.
- If they do not match, the normalized sequence cannot be a palindrome.

This is the central proof idea. The algorithm is not guessing which characters
to ignore. It ignores exactly the characters that the problem statement says do
not participate in the logical sequence.

### Detailed Algorithm

1. Initialize `left = 0` and `right = len(s) - 1`.
2. While `left < right`, process the current unresolved window.
3. Move `left` rightward while `left < right` and `s[left]` is not
   alphanumeric.
4. Move `right` leftward while `left < right` and `s[right]` is not
   alphanumeric.
5. Now either the pointers have met/crossed, or both point to meaningful
   characters.
6. Compare `s[left].lower()` with `s[right].lower()`.
7. If they differ, return `False`.
8. If they match, move both pointers inward: `left += 1`, `right -= 1`.
9. If the loop finishes, every meaningful mirrored pair matched, so return
   `True`.

The `left < right` guard matters in both skip loops. A string such as
`".,,   :;"` has no meaningful characters. The pointers may meet or cross while
skipping punctuation. Once that happens, there is no pair left to compare.

### Pseudocode

```text
function isPalindrome(s):
    left = 0
    right = length(s) - 1

    while left < right:
        while left < right and s[left] is not alphanumeric:
            left = left + 1

        while left < right and s[right] is not alphanumeric:
            right = right - 1

        if lowercase(s[left]) != lowercase(s[right]):
            return false

        left = left + 1
        right = right - 1

    return true
```

Equivalent Python code:

```python
class Solution:
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

### Detailed Example Walkthrough

Consider the official example:

```text
s = "A man, a plan, a canal: Panama"
```

The logical normalized sequence is:

```text
amanaplanacanalpanama
```

But the algorithm does not build that sequence. It discovers only the next pair
it needs.

1. `left` starts at `A`, `right` starts at the final `a`.
   - Both are alphanumeric.
   - Compare `a` with `a`: match.
   - Move both pointers inward.
2. `left` reaches a space after `A`.
   - Space is ignored, so advance `left` until it reaches `m`.
   - `right` reaches `m` near the end.
   - Compare `m` with `m`: match.
3. Continue inward.
   - Whenever `left` lands on a space, comma, or colon, skip it.
   - Whenever `right` lands on punctuation or a space, skip it.
   - Each comparison is between the next meaningful character from the front
     and the next meaningful character from the back.
4. Every meaningful pair matches.
5. The pointers eventually meet or cross, so the algorithm returns `True`.

Now consider a failing example:

```text
s = "0P"
```

1. `left` points to `0`; `right` points to `P`.
2. Both characters are alphanumeric, so neither is skipped.
3. Compare lowercase forms: `0` versus `p`.
4. They differ, so return `False` immediately.

This early return is valid because the first meaningful character and the last
meaningful character are required to match in any palindrome. There are no other
characters that can change that fact.

Finally, consider a punctuation-only input:

```text
s = ".,,   :;"
```

All characters are ignored. The skip loops move the pointers inward until there
is no unresolved pair. The normalized sequence is empty, and an empty sequence
is a palindrome, so the result is `True`.

### Correctness

We prove that the algorithm returns `True` exactly when the normalized sequence
is a palindrome.

#### Lemma 1: Skipping non-alphanumeric characters is safe.

A non-alphanumeric character is excluded from the sequence that the problem asks
us to check. Therefore, it cannot be the left side or right side of any required
mirrored pair. Moving past such a character does not remove any character that
could affect the answer.

#### Lemma 2: After the skip loops, the two pointers identify the next
unresolved mirrored pair, if such a pair exists.

The left skip loop stops only when `left` reaches a meaningful character or
there is no pair left. Thus `s[left]` is the earliest remaining meaningful
character in the unresolved window. Similarly, the right skip loop stops only
when `right` reaches the latest remaining meaningful character or there is no
pair left. These are exactly the next two characters that must be equal in the
normalized palindrome check.

#### Lemma 3: If the algorithm returns `False`, the normalized sequence is not a
palindrome.

The algorithm returns `False` only after finding two meaningful characters that
occupy mirrored positions in the remaining normalized sequence and differ after
case normalization. A palindrome requires every mirrored pair to match. Since
this required pair does not match, the normalized sequence is not a palindrome.

#### Lemma 4: If a compared pair matches, shrinking both pointers preserves the
invariant.

When the lowercase characters at `left` and `right` match, that mirrored pair is
proven correct. Future comparisons only need to consider meaningful characters
strictly inside that pair. Moving `left` rightward and `right` leftward removes
only a verified pair from the unresolved window, so all meaningful pairs outside
the new window have been checked and matched.

#### Theorem: The algorithm is correct.

If the algorithm returns `False`, Lemma 3 shows that the normalized sequence is
not a palindrome. If the algorithm finishes the loop and returns `True`, then
all meaningful mirrored pairs have either been skipped as irrelevant
non-alphanumeric characters or compared and matched. By the invariant and Lemma
4, no unmatched required pair remains. Therefore the normalized sequence is a
palindrome.

### Complexity

- Time: `O(n)`, where `n` is the length of `s`. Each pointer moves only inward,
  so each character is skipped or compared a constant number of times.
- Space: `O(1)`. The algorithm stores only two indices and temporary lowercase
  comparisons. It does not build a cleaned copy of the string.

### Common Pitfalls

- Comparing before filtering. Punctuation and spaces must be skipped before any
  equality check.
- Forgetting case normalization. `"A"` and `"a"` should compare equal.
- Treating digits as ignorable. Digits are alphanumeric, so `"0P"` fails
  because `0` and `p` do not match.
- Moving only one pointer after a match. A successful comparison proves both
  sides of the mirrored pair, so both pointers should move inward.
- Using skip loops without `left < right`. Punctuation-only inputs can make the
  pointers cross while filtering.
- Thinking the raw string must be symmetric. The punctuation layout does not
  need to mirror; only the filtered lowercase sequence matters.

### First-Principles Summary

The problem defines a logical sequence hidden inside a noisy string. The brute
force solution materializes that sequence and reverses it. The optimized
solution keeps the same logic but changes when the normalized characters are
produced: it generates only the next meaningful character from each end.

The invariant is that everything outside the current pointer window has already
been resolved. Non-alphanumeric characters are safe to discard because they are
not part of the logical sequence. Matching meaningful characters are safe to
discard because their mirrored relationship has been proven. A single mismatch
is enough to reject the string because every palindrome requires every mirrored
pair to match.

That is the whole algorithm: filter lazily, compare mirrored meaningful
characters, and shrink the unresolved window until either a mismatch appears or
nothing remains to prove.

## Implementation

See `solutions/two_pointers/p125_valid_palindrome.py`.

## Tests

See `tests/two_pointers/test_p125_valid_palindrome.py`.

## Examples

- `"A man, a plan, a canal: Panama"` returns `True` because the normalized
  sequence is `"amanaplanacanalpanama"`.
- `"race a car"` returns `False` because the normalized sequence
  `"raceacar"` has a mismatched mirrored pair.
- `" "` returns `True` because the normalized sequence is empty.
- `"No 'x' in Nixon"` returns `True` because punctuation and case do not affect
  the mirrored alphanumeric sequence.
- `"0P"` returns `False` because digits are meaningful and `0` does not match
  `p`.
- See `tests/two_pointers/test_p125_valid_palindrome.py` for executable
  examples and edge cases.
