# 5. Longest Palindromic Substring

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/longest-palindromic-substring/
- Official Group: Multidimensional DP
- Pattern Group: Dynamic Programming Multidimensional
- Patterns: dynamic-programming-multidimensional, window-or-prefix

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given one string `s`, return the longest contiguous substring of `s` that reads the same forward and backward.

A substring is not the same as a subsequence.

```text
s = "babad"
```

Valid substrings include:

```text
"b"
"ba"
"bab"
"aba"
"babad"
```

But the answer must be a palindrome. In this input:

```text
"bab" reads the same forward and backward
"aba" also reads the same forward and backward
```

Both have length `3`, so either is a valid longest palindromic substring. The example output uses `"bab"`.

The real problem is:

> Among all contiguous intervals `s[left:right + 1]`, find the longest one whose characters mirror perfectly around its middle.

---

### 2. Start From the Brute Force Baseline

The most direct idea is to try every substring and check whether it is a palindrome.

A substring is determined by two indices:

```text
left  = starting index
right = ending index
```

So the brute force algorithm is:

1. Choose every `left`.
2. Choose every `right >= left`.
3. Check whether `s[left:right + 1]` is a palindrome.
4. Keep the longest palindromic substring seen so far.

Pseudocode:

```python
best = ""

for left in range(len(s)):
    for right in range(left, len(s)):
        candidate = s[left:right + 1]
        if is_palindrome(candidate) and len(candidate) > len(best):
            best = candidate

return best
```

The palindrome check itself compares characters from the outside inward:

```python
def is_palindrome(text):
    left = 0
    right = len(text) - 1

    while left < right:
        if text[left] != text[right]:
            return False
        left += 1
        right -= 1

    return True
```

This is correct, but expensive.

There are `O(n^2)` substrings. Checking one substring can take `O(n)` time. Therefore the brute force time complexity is `O(n^3)`.

The first-principles goal is to avoid rechecking the same inner characters again and again.

---

### 3. Key Observation: A Palindrome Is Determined From Its Edges Inward

Consider a substring:

```text
s[left:right + 1]
```

For it to be a palindrome, two things must be true:

1. The outer characters match:

```text
s[left] == s[right]
```

2. The inside substring is also a palindrome:

```text
s[left + 1:right]
```

For example:

```text
s = "racecar"

left  = 0  -> 'r'
right = 6  -> 'r'
```

The outer characters match. Now the question becomes whether the inside substring is a palindrome:

```text
"aceca"
```

That inner substring is also a palindrome, so the whole string is a palindrome.

This gives the dynamic programming relationship:

```text
s[left:right + 1] is a palindrome
if s[left] == s[right]
and s[left + 1:right] is a palindrome
```

There is one important boundary case.

If the substring length is `1`, it is always a palindrome:

```text
"a"
```

If the substring length is `2`, it is a palindrome exactly when both characters match:

```text
"aa" -> palindrome
"ab" -> not a palindrome
```

So for length `1` or `2`, there is no meaningful inner substring that needs to be checked.

---

### 4. The DP State

Because each candidate substring is identified by two endpoints, one index is not enough. We need a two-dimensional state.

Define:

```text
dp[left][right] = True if s[left:right + 1] is a palindrome
                  False otherwise
```

This state is complete because it answers the exact question we need for every possible interval.

Once `dp[left][right]` is known, we can use it to update the best answer:

```text
if dp[left][right] is True and right - left + 1 is longer than best:
    best = s[left:right + 1]
```

The transition is:

```text
dp[left][right] =
    s[left] == s[right]
    and (right - left < 3 or dp[left + 1][right - 1])
```

Why `right - left < 3`?

The length of the substring is:

```text
right - left + 1
```

So `right - left < 3` covers lengths `1`, `2`, and `3`.

For those lengths, matching outer characters are enough:

```text
length 1: "a"   -> always palindrome
length 2: "aa"  -> outer match is enough
length 3: "aba" -> outer match is enough because the middle is one character
```

For length `4` or more, we must also know whether the inside substring is a palindrome.

---

### 5. The Fill Order Invariant

The transition for `dp[left][right]` may read:

```text
dp[left + 1][right - 1]
```

That means the inner interval must already be computed before the outer interval.

There are two common ways to guarantee this.

#### Option A: Fill by substring length

Compute all substrings of length `1`, then length `2`, then length `3`, and so on.

When processing a substring of length `L`, its inner substring has length `L - 2`, which has already been processed.

#### Option B: Move `left` backward and `right` forward

Loop `left` from the end of the string down to the beginning. For each `left`, loop `right` from `left` to the end.

When computing `dp[left][right]`, the inner state is `dp[left + 1][right - 1]`. Since `left + 1` is a later row, it has already been computed by the backward `left` loop.

Both are correct. The length-based version is often easiest to explain; the backward-left version is compact in code.

The invariant is:

```text
Before computing dp[left][right], every inner interval it depends on is already known.
```

---

### 6. Detailed DP Algorithm

1. Let `n = len(s)`.
2. If `n == 0`, return `""`.
3. Create an `n x n` boolean table initialized to `False`.
4. Set the best answer to the first character, because every single character is a palindrome.
5. Process substrings from shorter to longer.
6. For each interval `[left, right]`:
   - If `s[left] != s[right]`, the interval cannot be a palindrome.
   - If the characters match and the interval length is at most `3`, mark it as a palindrome.
   - If the characters match and the interval length is at least `4`, use the inner state `dp[left + 1][right - 1]`.
7. Whenever a palindromic interval is longer than the current best, record it.
8. Return the recorded substring.

Pseudocode:

```python
def longestPalindrome(s):
    n = len(s)
    if n == 0:
        return ""

    dp = [[False] * n for _ in range(n)]
    best_start = 0
    best_length = 1

    for length in range(1, n + 1):
        for left in range(0, n - length + 1):
            right = left + length - 1

            if s[left] != s[right]:
                dp[left][right] = False
                continue

            if length <= 3:
                dp[left][right] = True
            else:
                dp[left][right] = dp[left + 1][right - 1]

            if dp[left][right] and length > best_length:
                best_start = left
                best_length = length

    return s[best_start:best_start + best_length]
```

---

### 7. Center Expansion: The Same Invariant Without the Table

There is also a very natural first-principles approach that does not store `O(n^2)` states.

A palindrome mirrors around a center.

Odd-length palindromes have one center character:

```text
"racecar"
   ^
 center
```

Even-length palindromes have a center gap between two characters:

```text
"abba"
  ^^
 center gap
```

Instead of asking whether every interval is a palindrome, we can start from each possible center and expand outward while the mirrored characters match.

For a center, maintain this invariant:

```text
s[left + 1:right] is already known to be a palindrome.
```

Then try to grow it by one character on both sides:

```text
if left >= 0 and right < n and s[left] == s[right]:
    s[left:right + 1] is also a palindrome
```

Once the characters differ, no larger palindrome with that same center is possible, because every larger palindrome would still need those mismatched characters to mirror.

Center-expansion pseudocode:

```python
def longestPalindrome(s):
    best_start = 0
    best_length = 0

    def expand(left, right):
        nonlocal best_start, best_length

        while left >= 0 and right < len(s) and s[left] == s[right]:
            length = right - left + 1
            if length > best_length:
                best_start = left
                best_length = length
            left -= 1
            right += 1

    for center in range(len(s)):
        expand(center, center)       # odd length
        expand(center, center + 1)   # even length

    return s[best_start:best_start + best_length]
```

This center method is not usually called DP, but it is based on the same palindrome invariant:

```text
A larger palindrome is created by adding equal characters around a smaller palindrome.
```

The DP table remembers the truth value for every interval. Center expansion discovers only the intervals that are palindromes around each center.

---

### 8. Walkthrough: `s = "babad"`

Index the string:

```text
index:  0 1 2 3 4
char:   b a b a d
```

Using center expansion, check each center.

#### Center at index `0`

Odd center:

```text
left = 0, right = 0
s[0] == s[0] -> "b"
```

Best becomes:

```text
"b"
```

Try expanding:

```text
left = -1, right = 1
```

Out of bounds, so stop.

Even center between `0` and `1`:

```text
s[0] = 'b'
s[1] = 'a'
```

They do not match, so no even palindrome here.

#### Center at index `1`

Odd center:

```text
left = 1, right = 1
s[1] == s[1] -> "a"
```

Length `1` does not beat the current best length `1`.

Expand outward:

```text
left = 0, right = 2
s[0] = 'b'
s[2] = 'b'
```

They match, so:

```text
s[0:3] = "bab"
```

Best becomes:

```text
"bab"
```

Expand again:

```text
left = -1, right = 3
```

Out of bounds, so stop.

Even center between `1` and `2`:

```text
s[1] = 'a'
s[2] = 'b'
```

They do not match.

#### Center at index `2`

Odd center:

```text
left = 2, right = 2
s[2] == s[2] -> "b"
```

Expand outward:

```text
left = 1, right = 3
s[1] = 'a'
s[3] = 'a'
```

They match, so:

```text
s[1:4] = "aba"
```

This has length `3`, equal to the current best `"bab"`. If the code only updates on strictly greater length, it keeps `"bab"`.

Expand again:

```text
left = 0, right = 4
s[0] = 'b'
s[4] = 'd'
```

They differ, so no larger palindrome centered at `2` exists.

#### Remaining centers

Centers at `3` and `4` produce only single-character palindromes or mismatches when expanded.

The final answer is:

```text
"bab"
```

`"aba"` is also a correct answer for the same input, because the problem accepts any longest palindromic substring.

---

### 9. Walkthrough: `s = "cbbd"`

Index the string:

```text
index:  0 1 2 3
char:   c b b d
```

Single characters are palindromes, so the best starts as length `1`.

The important center is the even center between indices `1` and `2`:

```text
left = 1, right = 2
s[1] = 'b'
s[2] = 'b'
```

They match, so:

```text
s[1:3] = "bb"
```

Best becomes:

```text
"bb"
```

Expand again:

```text
left = 0, right = 3
s[0] = 'c'
s[3] = 'd'
```

They differ, so the even palindrome cannot grow.

No other center produces a longer palindrome, so the answer is:

```text
"bb"
```

This example is the reason the algorithm must check even centers. If we only checked odd centers, we would miss `"bb"`.

---

### 10. Correctness Argument

We can prove correctness for the center-expansion algorithm, because it is the simplest implementation of the palindrome invariant.

#### Lemma 1: Every substring reported by expansion is a palindrome.

Expansion starts either from:

```text
(left, right) = (center, center)
```

or:

```text
(left, right) = (center, center + 1)
```

The odd start is a one-character palindrome. The even start becomes a palindrome only if the two characters match.

Each successful expansion adds one character to the left and one character to the right, and it only does so when those two characters are equal. Adding equal outer characters around a palindrome preserves the palindrome property.

Therefore every substring considered valid by the expansion loop is a palindrome.

#### Lemma 2: For a fixed center, expansion finds the longest palindrome with that center.

For a fixed center, any larger palindrome must include all smaller mirrored pairs around that same center.

The algorithm expands as long as:

```text
left and right are in bounds
s[left] == s[right]
```

It stops only when it reaches the string boundary or finds a mismatched mirrored pair. If a mismatch occurs, no larger palindrome with that same center can exist, because that larger palindrome would still need the mismatched pair to be equal.

Therefore expansion finds the longest palindrome for that center.

#### Lemma 3: Every palindrome has one of the checked centers.

Every palindrome has either:

- one middle character, for odd length, or
- one middle gap, for even length.

The algorithm checks both possibilities at every index:

```text
expand(center, center)
expand(center, center + 1)
```

So the center of every possible palindromic substring is checked.

#### Theorem: The algorithm returns a longest palindromic substring of `s`.

By Lemma 3, every palindromic substring has a center considered by the algorithm. By Lemma 2, the algorithm finds the longest palindrome for each center. By Lemma 1, every candidate used to update the answer is a valid palindrome.

Since the algorithm records the longest valid palindrome found across all centers, the returned substring is a longest palindromic substring of `s`.

---

### 11. Complexity

For the DP table approach:

- Time: `O(n^2)`, because there are `n * n` possible endpoint pairs and each state is computed in constant time.
- Space: `O(n^2)`, because the table stores whether each interval is a palindrome.

For the center-expansion approach:

- Time: `O(n^2)`, because there are `2n - 1` centers and each expansion can scan outward up to `O(n)` characters.
- Space: `O(1)`, ignoring the returned substring, because it only stores a few indices.

Both are a major improvement over the `O(n^3)` brute force baseline.

---

### 12. Common Pitfalls

- Forgetting even-length palindromes. `"cbbd"` requires checking the gap between the two `b` characters.
- Treating subsequences as substrings. The answer must be contiguous.
- Updating the best answer after moving the pointers instead of before moving them, which can create off-by-one slice errors.
- Using `>=` instead of `>` when updating the best answer if you want to preserve the first longest palindrome found, such as returning `"bab"` for `"babad"`.
- In the DP approach, filling the table in an order that reads `dp[left + 1][right - 1]` before it has been computed.
- Mishandling short substrings in DP. Length `1`, `2`, and `3` need special treatment because their inner substring is empty or one character.
- Returning the length instead of the substring. This problem asks for the actual longest palindromic substring.

---

### 13. First-Principles Summary

A palindrome is not arbitrary. It has mirror symmetry.

That symmetry gives the core rule:

```text
A substring is a palindrome when its outer characters match
and its inside is also a palindrome.
```

From that rule, two efficient approaches follow naturally:

1. Dynamic programming stores `dp[left][right]`, the palindrome truth of every interval.
2. Center expansion grows palindromes outward from every possible middle.

The DP view is useful because the problem is officially grouped as multidimensional dynamic programming: the subproblem is defined by two endpoints. The center view is useful because it keeps the same invariant while avoiding the full table.

In both cases, the algorithm is not guessing substrings. It is exploiting the only structure that matters: matching mirrored characters around a center.

## Implementation
See `solutions/dynamic_programming_multidimensional/p005_longest_palindromic_substring.py`.

## Tests
See `tests/dynamic_programming_multidimensional/test_p005_longest_palindromic_substring.py`.

## Examples

### Example 1
- Input: `{'s': 'babad'}`
- Output: `'bab'`

### Example 2
- Input: `{'s': 'cbbd'}`
- Output: `'bb'`

## Follow-up Practice
- Draw the DP table for `"babad"` and mark every `True` interval.
- Run center expansion manually on `"abba"` to see why even centers matter.
- Explain why a mismatch during expansion proves that no larger palindrome with the same center can exist.
