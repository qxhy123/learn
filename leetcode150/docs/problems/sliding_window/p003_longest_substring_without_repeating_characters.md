# 3. Longest Substring Without Repeating Characters

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/longest-substring-without-repeating-characters/
- Official Group: Sliding Window
- Pattern Group: Sliding Window
- Patterns: sliding-window, window-or-prefix

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given a string `s`, find the length of the longest substring that contains no repeated characters.

Two words matter:

```text
substring
without repeating characters
```

A **substring** must be contiguous.

For example, in:

```text
s = "abcabcbb"
```

`"abc"` is a substring.

But `"acb"` is not, because those characters are not taken as one continuous block.

So every candidate answer has the form:

```text
s[left ... right]
```

The problem becomes:

> Among all contiguous intervals, find the longest one whose characters are all unique.

### 2. Start From the Brute Force Idea

The most direct method is:

1. Enumerate every substring.
2. Check whether it has duplicate characters.
3. Keep the maximum valid length.

Conceptually:

```python
best = 0

for left in range(n):
    for right in range(left, n):
        if s[left:right + 1] has no duplicate:
            best = max(best, right - left + 1)
```

This is correct, but inefficient.

There are `O(n^2)` substrings, and checking each substring can cost up to `O(n)`.

So the deeper question is:

> What repeated work are we doing?

We keep checking overlapping substrings from scratch, even though adjacent substrings differ by only one character.

### 3. A Substring Is Just a Window

Because substrings are contiguous, we can represent a candidate substring using two boundaries:

```text
left
right
```

The current substring is:

```text
s[left:right + 1]
```

This is a **window**.

Instead of rebuilding every substring from scratch, we maintain one moving window.

The key question becomes:

> How should `left` and `right` move so that the window always represents a valid substring?

### 4. The Core Constraint

The window is valid if:

```text
Every character inside the window appears at most once.
```

So we want to maintain this invariant:

```text
s[left:right + 1] contains no duplicate characters
```

Now suppose the current window is already valid:

```text
s[left:right]
```

Then we try to add a new character:

```text
s[right]
```

What can break?

Only one thing:

```text
s[right] already exists inside the current window
```

Why?

Because before adding `s[right]`, the window had no duplicates. Therefore, the only possible duplicate must involve the newly added character.

This is the first-principles insight:

> If the current window is valid, adding one new character can only violate validity through that new character.

### 5. How to Repair the Window

If the new character is not already in the window:

```text
Add it.
The window remains valid.
```

If the new character is already in the window:

```text
Move left forward and remove characters
until the old copy of that character is gone.
```

Then add the new character.

So the two pointers have clear roles:

```text
right explores new characters
left removes old characters to restore validity
```

### 6. Example: `abcabcbb`

Let:

```text
s = "abcabcbb"
```

We maintain:

```text
window = characters inside s[left:right + 1]
best = longest valid length seen so far
```

#### Step 1: Add `a`

```text
window = "a"
best = 1
```

#### Step 2: Add `b`

```text
window = "ab"
best = 2
```

#### Step 3: Add `c`

```text
window = "abc"
best = 3
```

#### Step 4: Add `a`

Current window:

```text
"abc"
```

Adding `a` would create:

```text
"abca"
```

That has duplicate `a`.

So move `left` until the old `a` is removed:

```text
remove 'a'
window = "bca"
best = 3
```

#### Step 5: Add `b`

Current window:

```text
"bca"
```

Adding `b` would duplicate `b`.

Remove from the left until old `b` is gone:

```text
remove 'b'
window = "cab"
best = 3
```

#### Step 6: Add `c`

Current window:

```text
"cab"
```

Adding `c` duplicates `c`.

Remove old `c`:

```text
window = "abc"
best = 3
```

#### Step 7: Add `b`

Current window:

```text
"abc"
```

Adding `b` duplicates `b`.

Remove from the left until old `b` disappears:

```text
remove 'a'
remove 'b'
window = "cb"
best = 3
```

#### Step 8: Add `b`

Current window:

```text
"cb"
```

Adding `b` duplicates `b`.

Remove from the left until old `b` disappears:

```text
remove 'c'
remove 'b'
window = "b"
best = 3
```

Final answer:

```text
3
```

Valid longest substrings include:

```text
"abc"
"bca"
"cab"
```

### 7. Set-Based Sliding Window Code

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        seen = set()
        left = 0
        best = 0

        for right, ch in enumerate(s):
            while ch in seen:
                seen.remove(s[left])
                left += 1

            seen.add(ch)
            best = max(best, right - left + 1)

        return best
```

### 8. Why This Code Is Correct

The invariant is:

```text
seen contains exactly the characters in s[left:right + 1]
and the window has no duplicate characters
```

Before adding `ch`, the window is valid.

If `ch` is not in `seen`, adding it keeps the window valid.

If `ch` is in `seen`, adding it would create a duplicate. So we move `left` forward, removing characters, until the old `ch` is gone. Then adding `ch` restores the invariant.

After each iteration, the window is a valid substring ending at `right`.

For each `right`, the algorithm keeps the longest valid window ending at that `right`, because it only moves `left` as far as necessary to remove the duplicate.

Every substring has some right endpoint, so by checking every `right`, the algorithm eventually considers the best possible answer.

Therefore, `best` is the length of the longest substring without repeating characters.

### 9. Why It Is `O(n)`

At first glance, the nested `while` loop may look expensive.

But each pointer only moves forward:

```text
right moves from 0 to n - 1
left moves from 0 to n - 1
```

Each character is:

```text
added to the window at most once
removed from the window at most once
```

So the total work is linear.

Complexity:

```text
Time:  O(n)
Space: O(k)
```

where `k` is the number of distinct characters stored in the window.

### 10. Faster Last-Seen Index Version

Instead of moving `left` one step at a time, we can jump directly.

Store the most recent index of each character:

```text
last_seen[char] = latest index where char appeared
```

When we see `ch` at index `right`:

```text
If ch appeared inside the current window,
the new left boundary must be after its old position.
```

Code:

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        last_seen = {}
        left = 0
        best = 0

        for right, ch in enumerate(s):
            if ch in last_seen and last_seen[ch] >= left:
                left = last_seen[ch] + 1

            last_seen[ch] = right
            best = max(best, right - left + 1)

        return best
```

### 11. Why `last_seen[ch] >= left` Matters

Consider:

```text
s = "abba"
```

Walkthrough:

```text
right = 0, ch = 'a'
window = "a"

right = 1, ch = 'b'
window = "ab"

right = 2, ch = 'b'
old 'b' is at index 1
move left to 2
window = "b"
```

Now:

```text
right = 3, ch = 'a'
```

The old `a` is at index `0`.

But index `0` is outside the current window, because:

```text
left = 2
```

So that old `a` no longer matters.

If we blindly did:

```python
left = last_seen['a'] + 1
```

then:

```text
left = 1
```

That would move `left` backward, which is invalid.

So we only update `left` when the previous occurrence is still inside the current window:

```python
if last_seen[ch] >= left:
    left = last_seen[ch] + 1
```

Or equivalently:

```python
left = max(left, last_seen[ch] + 1)
```

### 12. First-Principles Summary

This problem follows from five basic facts:

```text
1. A substring is a contiguous interval.
2. A contiguous interval can be represented by left and right boundaries.
3. The validity constraint is: no character appears twice inside the interval.
4. If a valid interval becomes invalid after adding one character, the violation must involve that new character.
5. Moving left forward removes old characters until validity is restored.
```

So the whole algorithm is:

> Keep a window that always has no duplicate characters. Move `right` to explore, move `left` only when needed to repair the constraint, and record the largest valid window length seen.

## Implementation

See `solutions/sliding_window/p003_longest_substring_without_repeating_characters.py`.

## Tests

See `tests/sliding_window/test_p003_longest_substring_without_repeating_characters.py`.

## Examples

### Example 1
- Input: `{'s': 'abcabcbb'}`
- Output: `3`

### Example 2
- Input: `{'s': 'bbbbb'}`
- Output: `1`

### Example 3
- Input: `{'s': 'pwwkew'}`
- Output: `3`

## Follow-up Practice
- Implement both the set-based and last-seen-index versions.
- Trace the invariant on `abba` to understand why `left` must never move backward.
- Generalize the idea to "at most K distinct characters".
