# 76. Minimum Window Substring

- Difficulty: Hard
- LeetCode: https://leetcode.com/problems/minimum-window-substring/
- Official Group: Sliding Window
- Pattern Group: Sliding Window
- Patterns: sliding-window, window-or-prefix

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given two strings:

```text
s = the source string
t = the required characters
```

Find the shortest substring of `s` that contains every character in `t`, including duplicates.

For example:

```text
s = "ADOBECODEBANC"
t = "ABC"
```

A valid window must contain:

```text
A, B, and C
```

The substring:

```text
"ADOBEC"
```

is valid because it contains `A`, `B`, and `C`.

But it is not the shortest valid substring.

The shortest one is:

```text
"BANC"
```

So the answer is:

```text
"BANC"
```

The important detail is that `t` is not just a set of characters. It is a multiset.

For example:

```text
t = "AABC"
```

Then a valid window must contain:

```text
A twice
B once
C once
```

So the real problem is:

> Find the shortest contiguous interval in `s` whose character counts cover the character counts required by `t`.

---

### 2. Start From the Brute Force Idea

The most direct solution is:

1. Enumerate every substring `s[left:right + 1]`.
2. Count its characters.
3. Check whether it covers all characters in `t`.
4. Keep the shortest valid substring.

Conceptually:

```python
best = infinity
need = Counter(t)

for left in range(len(s)):
    for right in range(left, len(s)):
        window = Counter(s[left:right + 1])
        if window covers need:
            best = min(best, s[left:right + 1])
```

This is correct, but inefficient.

There are `O(n^2)` substrings, and checking counts can cost more work.

The deeper question is:

> Adjacent substrings differ by only one character. Can we update validity incrementally instead of recounting from scratch?

Yes. That leads to a sliding window.

---

### 3. A Substring Is a Window

Any candidate substring is a contiguous interval:

```text
s[left:right + 1]
```

So the problem can be seen as moving two boundaries:

```text
left  = start of current window
right = end of current window
```

The current window has character counts:

```text
window_count
```

A window is valid if it covers the required counts:

```text
for every character c in t:
    window_count[c] >= need[c]
```

This is the central condition.

---

### 4. The Core Constraint: Covering a Multiset

Build:

```text
need = counts of characters in t
```

For:

```text
t = "AABC"
```

we have:

```text
need = {
  'A': 2,
  'B': 1,
  'C': 1
}
```

A window is valid only if:

```text
window['A'] >= 2
window['B'] >= 1
window['C'] >= 1
```

Extra characters are allowed.

For example, if `t = "ABC"`, then this is valid:

```text
"AAADOBEC"
```

because it contains at least one `A`, one `B`, and one `C`.

But we want the shortest such window, so extra characters are usually something we try to remove.

---

### 5. Why Sliding Window Works

There are two natural actions:

```text
expand right
shrink left
```

If the current window is not valid, shrinking it cannot help.

Why?

Because removing characters cannot create missing required characters.

So when the window is invalid, the only useful direction is:

```text
move right forward to include more characters
```

If the current window is valid, expanding it cannot make it shorter.

So when the window is valid, the useful direction is:

```text
move left forward to see whether we can keep validity with fewer characters
```

This is the first-principles logic:

> Expand until the window becomes valid. Then shrink until it stops being valid. Repeat.

---

### 6. Avoid Checking Every Required Character Each Time

A naive validity check would inspect every character in `need` every time:

```python
all(window[c] >= need[c] for c in need)
```

That works, but it adds repeated work.

Instead, track how many distinct required characters are currently satisfied.

Let:

```text
required = number of distinct characters in t
formed = number of distinct characters c where window_count[c] >= need[c]
```

When:

```text
formed == required
```

then the window is valid.

Many implementations increment `formed` exactly when a character count reaches its needed amount:

```text
window_count[c] becomes need[c]
```

and decrement `formed` exactly when a character count falls below its needed amount:

```text
window_count[c] becomes need[c] - 1
```

This lets us know validity in constant time.

---

### 7. The Window Invariant

Maintain these facts:

```text
window_count stores the exact counts of required characters inside s[left:right + 1]
formed is the number of required characters whose needed count is currently satisfied
```

Then:

```text
formed == required
```

is equivalent to:

```text
the current window contains all characters required by t
```

This invariant is the engine of the algorithm.

---

### 8. Algorithm

1. Count the required characters in `t`.
2. Set `left = 0`.
3. Move `right` across `s` one character at a time.
4. When `s[right]` is a required character, add it to `window_count`.
5. If that character's count just reached the required count, increment `formed`.
6. While `formed == required`, the window is valid:
   - Update the best answer if the current window is shorter.
   - Remove `s[left]` from the window.
   - If removing it makes a required count fall below what is needed, decrement `formed`.
   - Move `left` forward.
7. Return the best window found, or `""` if none exists.

---

### 9. Example: `ADOBECODEBANC`, `ABC`

Let:

```text
s = "ADOBECODEBANC"
t = "ABC"
need = {'A': 1, 'B': 1, 'C': 1}
required = 3
```

We scan `s` with `right`.

#### Expand until valid

Start:

```text
left = 0
formed = 0
```

Read `A`:

```text
window has A
formed = 1
```

Read `D`, `O`:

```text
not required
formed remains 1
```

Read `B`:

```text
window has A and B
formed = 2
```

Read `E`:

```text
not required
formed remains 2
```

Read `C`:

```text
window has A, B, and C
formed = 3
```

Now the window is:

```text
"ADOBEC"
```

It is valid, so record it:

```text
best = "ADOBEC"
```

#### Shrink while valid

Try moving `left`.

Remove `A`:

```text
window no longer has A
formed = 2
```

Now the window is invalid, so stop shrinking and expand again.

#### Expand again

Continue scanning until another `A` appears:

```text
s = "ADOBECODEBA..."
```

When `A` is included again, the window becomes valid.

Then shrink from the left to remove unnecessary characters.

Eventually the window becomes:

```text
"BANC"
```

It contains:

```text
B, A, C
```

and has length `4`, which is shorter than `"ADOBEC"`.

So update:

```text
best = "BANC"
```

No shorter valid window exists, so the final answer is:

```text
"BANC"
```

---

### 10. Code

```python
from collections import Counter, defaultdict


class Solution:
    def minWindow(self, s: str, t: str) -> str:
        if not s or not t:
            return ""

        need = Counter(t)
        window = defaultdict(int)

        required = len(need)
        formed = 0

        left = 0
        best_len = float("inf")
        best_left = 0

        for right, ch in enumerate(s):
            if ch in need:
                window[ch] += 1

                if window[ch] == need[ch]:
                    formed += 1

            while formed == required:
                current_len = right - left + 1
                if current_len < best_len:
                    best_len = current_len
                    best_left = left

                left_char = s[left]
                if left_char in need:
                    window[left_char] -= 1

                    if window[left_char] < need[left_char]:
                        formed -= 1

                left += 1

        if best_len == float("inf"):
            return ""

        return s[best_left:best_left + best_len]
```

---

### 11. Why This Code Is Correct

The algorithm maintains the invariant that `window` contains the counts of required characters inside the current window `s[left:right + 1]`.

The variable `formed` counts how many required characters currently meet their required counts.

Therefore:

```text
formed == required
```

if and only if the current window covers all characters in `t`.

Whenever the window is invalid, the algorithm expands `right`. This is necessary because removing characters from an invalid window cannot make it contain missing required characters.

Whenever the window is valid, the algorithm records it as a candidate and then moves `left` forward to search for a shorter valid window with the same `right` boundary.

For each `right`, the inner loop shrinks the window until removing one more character would make it invalid. Therefore, it considers the shortest valid window ending at that `right`.

Every possible answer has some right boundary. Since the algorithm processes every `right` and records the best valid window found while shrinking, it cannot miss the global minimum.

Because it records only valid windows, the final recorded minimum is exactly the shortest substring of `s` that contains all characters of `t`.

---

### 12. Why It Is `O(n)`

Although there is a `while` loop inside the `for` loop, both pointers only move forward.

```text
right moves from 0 to len(s) - 1
left moves from 0 to len(s) - 1
```

Each character enters the window at most once and leaves the window at most once.

So the total work is linear in the length of `s`.

Complexity:

```text
Time:  O(len(s) + len(t))
Space: O(k)
```

where `k` is the number of distinct characters in `t` and the tracked window counts.

---

### 13. Common Pitfalls

#### Pitfall 1: Treating `t` as a set

This fails when `t` has duplicates.

For example:

```text
s = "aa"
t = "aa"
```

The answer is:

```text
"aa"
```

A single `"a"` is not enough.

#### Pitfall 2: Incrementing `formed` too often

You should increment `formed` only when a character count reaches the required count exactly:

```python
if window[ch] == need[ch]:
    formed += 1
```

Not every time you see a required character.

#### Pitfall 3: Decrementing `formed` too early

You should decrement `formed` only when removing a character makes its count fall below the required amount:

```python
if window[left_char] < need[left_char]:
    formed -= 1
```

Extra copies do not matter.

#### Pitfall 4: Forgetting to shrink while valid

Using `if formed == required` instead of `while formed == required` misses shorter windows.

Once a window is valid, you should keep shrinking until it becomes invalid.

#### Pitfall 5: Returning a window when none exists

If no valid window was recorded, return:

```text
""
```

---

### 14. First-Principles Summary

This problem follows from these basic facts:

```text
1. A substring is a contiguous interval.
2. A valid interval must cover the character counts required by t.
3. If a window is invalid, shrinking cannot help because it only removes characters.
4. If a window is valid, expanding cannot make it shorter, so we should shrink from the left.
5. Character counts let us update validity incrementally.
6. Tracking how many required characters are fully satisfied lets us test validity in O(1).
```

In one sentence:

> Expand the window until it covers `t`, then shrink it as much as possible while it still covers `t`, recording the shortest valid window seen.

## Implementation

See `solutions/sliding_window/p076_minimum_window_substring.py`.

## Tests

See `tests/sliding_window/test_p076_minimum_window_substring.py`.

## Examples

### Example 1
- Input: `{'s': 'ADOBECODEBANC', 't': 'ABC'}`
- Output: `'BANC'`

### Example 2
- Input: `{'s': 'a', 't': 'a'}`
- Output: `'a'`

### Example 3
- Input: `{'s': 'a', 't': 'aa'}`
- Output: `''`

## Follow-up Practice
- Re-implement validity with a single `missing` counter instead of `formed`.
- Trace a case where `t` contains duplicate characters, such as `AABC`.
- Compare this covering-window pattern with exact-count anagram windows.
