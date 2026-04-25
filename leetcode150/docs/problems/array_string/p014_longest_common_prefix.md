# 14. Longest Common Prefix

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/longest-common-prefix/
- Official Group: Array / String
- Pattern Group: Array / String
- Patterns: string-scanning

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given an array of strings `strs`, return the longest string that is a prefix of every string in the array.

A **prefix** starts at index `0`.

For example:

```text
"flower"
```

has prefixes:

```text
""
"f"
"fl"
"flo"
"flow"
"flowe"
"flower"
```

The prefix does not get to skip characters, start in the middle, or choose different characters from different strings.

For:

```text
strs = ["flower", "flow", "flight"]
```

all three strings start with:

```text
"f"
```

They also all start with:

```text
"fl"
```

But they do not all start with:

```text
"flo"
```

because `"flight"` has:

```text
index 2 = "i"
```

while `"flower"` and `"flow"` have:

```text
index 2 = "o"
```

So the answer is:

```text
"fl"
```

The real problem is:

> Find the longest initial run of character positions where every string has the same character.

---

### 2. Start From the Brute Force Idea

The most direct way is to try possible prefixes and test whether each one appears at the start of every string.

One version is:

```python
best = ""

for length in range(1, len(strs[0]) + 1):
    candidate = strs[0][:length]
    if every string starts with candidate:
        best = candidate
    else:
        break
```

This is correct because every common prefix must be a prefix of the first string.

But the test:

```text
Does every string start with candidate?
```

rechecks characters that were already proven equal for shorter candidates.

For example, after proving that every string starts with `"fl"`, testing `"flo"` by comparing all three characters from the beginning repeats the work for `"f"` and `"l"`.

The first-principles question is:

> Can we examine each relevant character position only once?

Yes. Since prefixes grow from left to right, we can check one column of characters at a time.

---

### 3. Turn Prefixes Into Columns

Line the strings up by index:

```text
index:    0   1   2   3   4   5

flower   f   l   o   w   e   r
flow     f   l   o   w
flight   f   l   i   g   h   t
```

At index `0`, every string has `f`.

At index `1`, every string has `l`.

At index `2`, the strings disagree:

```text
flower -> o
flow   -> o
flight -> i
```

Once a column disagrees, the common prefix must stop immediately before that column.

Why?

Because a prefix of length `3` would have to include the character at index `2`. If not every string has the same character at index `2`, no prefix of length `3` can be common to all strings. And if length `3` is impossible, every longer prefix is impossible too, because longer prefixes include index `2` as well.

This gives the key monotonic structure:

```text
If columns 0 through k are all equal, a common prefix of length k + 1 exists.
If column k fails, no prefix of length k + 1 or greater exists.
```

---

### 4. Key Observation

A string `prefix` is common to every string exactly when every character position inside it passes two tests:

```text
1. The position exists in every string.
2. Every string has the same character at that position.
```

So for each index `i`, there are only two possible reasons to stop:

```text
Some string is too short to have index i.
Some string has a different character at index i.
```

If neither reason happens, then `strs[0][i]` safely belongs to the answer.

This is why the first string can be used as the reference. The answer cannot contain any character that is not in the first string, because the answer must be a prefix of the first string too.

---

### 5. Invariant and State

Use:

```text
i = current character position being checked
```

The invariant before checking index `i` is:

```text
strs[0][:i] is a prefix of every string in strs.
```

That means all positions before `i` have already been proven safe.

At index `i`, compare every other string against the first string's character:

```text
expected = strs[0][i]
```

For each string `s`, require:

```text
i < len(s)
s[i] == expected
```

If every string passes, the invariant extends:

```text
strs[0][:i + 1] is a prefix of every string.
```

If any string fails, return:

```text
strs[0][:i]
```

That is the longest answer, not just some valid answer, because index `i` is the first position where extension becomes impossible.

---

### 6. Detailed Algorithm

Handle the input as a set of strings that must agree from left to right.

1. If the array is empty, return `""`.
2. Let `first = strs[0]`.
3. For each index `i` in `first`:
   - Let `expected = first[i]`.
   - For each string `s` in `strs[1:]`:
     - If `i == len(s)`, then `s` ended before this character, so return `first[:i]`.
     - If `s[i] != expected`, then the strings disagree at this column, so return `first[:i]`.
   - If every string passes, continue to the next index.
4. If the loop finishes, then the entire first string is a prefix of every string, so return `first`.

The stopping rule is the heart of the algorithm:

```text
Return the characters before the first failed column.
```

---

### 7. Pseudocode

```python
def longest_common_prefix(strs):
    if not strs:
        return ""

    first = strs[0]

    for i in range(len(first)):
        expected = first[i]

        for s in strs[1:]:
            if i == len(s):
                return first[:i]

            if s[i] != expected:
                return first[:i]

    return first
```

The LeetCode constraints usually provide at least one string, but the empty-array guard makes the function complete and avoids indexing `strs[0]` when no first string exists.

---

### 8. Detailed Example Walkthrough

Use the official example:

```text
strs = ["flower", "flow", "flight"]
```

Start with:

```text
first = "flower"
```

Before checking any character, the known common prefix is:

```text
""
```

The empty string is always a prefix of every string.

Now check index `0`:

```text
expected = first[0] = "f"
```

Compare against the other strings:

```text
flow[0]   = "f"  matches
flight[0] = "f"  matches
```

All strings have `"f"` at index `0`, so the known common prefix becomes:

```text
"f"
```

Check index `1`:

```text
expected = first[1] = "l"
```

Compare:

```text
flow[1]   = "l"  matches
flight[1] = "l"  matches
```

All strings match again, so the known common prefix becomes:

```text
"fl"
```

Check index `2`:

```text
expected = first[2] = "o"
```

Compare:

```text
flow[2]   = "o"  matches
flight[2] = "i"  does not match
```

The first failed column is index `2`.

Any prefix of length `3` would need all strings to share the same character at index `2`, but they do not. Therefore the longest common prefix is exactly the portion before index `2`:

```text
first[:2] = "fl"
```

Return:

```text
"fl"
```

---

### 9. Walkthrough With a Shorter String

Consider:

```text
strs = ["cart", "car", "carbon"]
```

Line them up:

```text
cart     c   a   r   t
car      c   a   r
carbon   c   a   r   b   o   n
```

Indexes `0`, `1`, and `2` match:

```text
"c"
"ca"
"car"
```

At index `3`, the string `"car"` has already ended.

That means no common prefix can include index `3`, because a prefix of length `4` would require every string to have four characters.

Return:

```text
"car"
```

This case shows why the length check is just as important as the character check.

---

### 10. Correctness

We prove that the algorithm returns the longest common prefix.

**Invariant.** Before each outer-loop iteration at index `i`, `first[:i]` is a prefix of every string in `strs`.

**Initialization.** Before the first iteration, `i = 0`, so `first[:0]` is `""`. The empty string is a prefix of every string, so the invariant holds.

**Maintenance.** Assume the invariant holds before checking index `i`. The algorithm compares `first[i]` with every other string at the same index and also verifies that every string is long enough to contain index `i`.

If every string has index `i` and every character equals `first[i]`, then every string starts with `first[:i]` by the invariant and also has the same next character `first[i]`. Therefore every string starts with `first[:i + 1]`, so the invariant holds for the next iteration.

**Stopping on failure.** If some string does not have index `i`, then no common prefix can have length `i + 1`, because that string is too short. If some string has a different character at index `i`, then no common prefix can have length `i + 1`, because common prefixes require the same character at every included position. In both cases, every longer prefix is also impossible because it would include index `i`. Since the invariant says `first[:i]` is common, `first[:i]` is exactly the longest common prefix.

**Termination after the loop.** If the loop finishes without failure, then every character in `first` was verified in every string. Therefore `first` is a prefix of every string. No common prefix can be longer than `first`, because any common prefix must also be a prefix of `first`. So returning `first` is correct.

Thus, in every case, the algorithm returns exactly the longest common prefix.

---

### 11. Complexity

Let:

```text
n = number of strings
m = length of the first string
```

The algorithm checks at most `m` character positions. For each position, it may compare against up to `n - 1` other strings.

So the worst-case time complexity is:

```text
O(n * m)
```

This happens when all strings share the entire first string as a prefix, or when the mismatch is near the end.

More precisely, the work is bounded by the number of characters examined before the first mismatch or shortest-string end. It stops early when the common prefix is short.

The extra space complexity is:

```text
O(1)
```

excluding the returned slice. The algorithm stores only the current index and expected character.

---

### 12. Common Pitfalls

**Forgetting the shorter-string case.**

If `strs = ["abc", "ab"]`, the answer is `"ab"`. Accessing `s[2]` for `"ab"` would be out of bounds. Check length before reading `s[i]`.

**Thinking a mismatch can be repaired later.**

For prefixes, a mismatch at index `i` is final. Later characters do not matter because any longer prefix must include the failed index.

**Using substring logic instead of prefix logic.**

The answer must start at index `0` in every string. A shared sequence in the middle does not count.

For example:

```text
["abcxyz", "defxyz"]
```

share `"xyz"` as a suffix, but their longest common prefix is `""`.

**Returning the candidate after including the failed character.**

When index `i` fails, return `first[:i]`, not `first[:i + 1]`.

**Building the answer character by character when slicing is simpler.**

It is fine to append matching characters to a result list, but it is not necessary. The first failed index already identifies the correct slice of the first string.

**Overcomplicating with sorting.**

A sorting-based solution can compare only the lexicographically smallest and largest strings after sorting, but sorting costs extra time and changes the shape of the reasoning. The direct column scan is simpler and stops as soon as the answer is known.

---

### 13. First-Principles Summary

This problem follows from five basic facts:

```text
1. A common prefix must start at index 0 in every string.
2. Any common prefix must be a prefix of the first string.
3. A prefix of length k is valid only if every column 0 through k - 1 matches across all strings.
4. The first failed column makes that length and every longer length impossible.
5. Therefore, scan columns left to right and return the characters before the first failure.
```

The whole algorithm is:

> Treat each character position as a column across all strings. As long as every string has the column and every character matches, the common prefix grows by one. At the first missing or mismatching column, return the prefix built before that point.

## Implementation

See `solutions/array_string/p014_longest_common_prefix.py`.

## Tests

See `tests/array_string/test_p014_longest_common_prefix.py`.

## Examples

### Example 1
- Input: `{'strs': ['flower', 'flow', 'flight']}`
- Output: `'fl'`

### Example 2
- Input: `{'strs': ['dog', 'racecar', 'car']}`
- Output: `''`

## Follow-up Practice
- Trace the invariant on `["interview", "internet", "internal"]`.
- Test a case where one string is the full answer, such as `["car", "cart", "carbon"]`.
- Compare the vertical-scan solution with a sorting-based solution and explain why both are correct.
