# 128. Longest Consecutive Sequence

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/longest-consecutive-sequence/
- Official Group: Hashmap
- Pattern Group: Hash Table
- Patterns: hash-table

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given an integer array `nums`, return the length of the longest run of values that appear consecutively by value.

Consecutive by value means numbers like:

```text
1, 2, 3, 4
```

or:

```text
-2, -1, 0, 1
```

The numbers do **not** need to be adjacent in the input array.

For example:

```text
nums = [100, 4, 200, 1, 3, 2]
```

The longest consecutive sequence is:

```text
1, 2, 3, 4
```

Its length is `4`.

The original order of the array is not important. The array is only telling us which values exist.

So the problem is really asking:

> Among all integer intervals whose every value appears in `nums`, what is the maximum interval length?

An interval such as:

```text
[start, start + 1, start + 2, ..., end]
```

is valid only if every value in that interval exists in the input.

Duplicates do not extend a consecutive sequence. For example:

```text
nums = [1, 0, 1, 2]
```

The values that exist are:

```text
0, 1, 2
```

The longest consecutive sequence has length `3`, not `4`, because the second `1` does not create a new value.

### 2. Start From the Brute Force Idea

The most direct way to solve the problem is to try building a sequence from every number.

For each `num`:

1. Start with `current = num`.
2. Check whether `current + 1` exists in the array.
3. Keep extending while the next value exists.
4. Record the length.

Conceptually:

```python
best = 0

for num in nums:
    length = 1
    current = num

    while current + 1 exists somewhere in nums:
        current += 1
        length += 1

    best = max(best, length)
```

This idea is logically correct because every consecutive sequence has some first value, and eventually the loop that starts from that first value will measure it.

But the inefficient part is hidden in this phrase:

```text
exists somewhere in nums
```

If we search the array linearly each time, a membership check costs `O(n)`.

In the worst case, for an input such as:

```text
[1, 2, 3, 4, 5, ..., n]
```

starting at `1` scans almost the whole sequence, starting at `2` scans almost the whole remaining sequence, starting at `3` does the same, and so on. With linear membership checks, this can become `O(n^3)` in the most literal implementation, or `O(n^2)` if membership is optimized but every position still redundantly walks a long suffix.

The deeper question is:

> Which work is repeated, and how do we avoid doing it?

Two repeated costs appear:

1. Checking whether a value exists.
2. Recounting the same sequence from the middle.

The final algorithm fixes both.

### 3. Turn the Array Into a Set of Existing Values

The problem does not care how many times a value appears or where it appears in the array.

It only cares whether each value exists.

That suggests the first transformation:

```python
num_set = set(nums)
```

Now membership questions become expected `O(1)`:

```python
x in num_set
```

This immediately removes the need to scan the array to answer:

```text
Does x exist?
```

It also automatically handles duplicates:

```text
nums    = [1, 0, 1, 2]
set     = {0, 1, 2}
answer  = 3
```

The set is the exact representation we need because a consecutive sequence is determined entirely by value membership.

### 4. The Key Observation: Only Start Counting at Starts

A valid consecutive sequence has one first number.

For example, the sequence:

```text
1, 2, 3, 4
```

has first number `1`.

How can we recognize that `1` is the first number?

Because the previous value does not exist:

```text
0 is not in the set
```

In general, a number `x` is the start of a consecutive sequence exactly when:

```text
x - 1 is not in the set
```

If `x - 1` does exist, then `x` is not the beginning. It belongs to a sequence that started earlier.

For example, with:

```text
num_set = {1, 2, 3, 4, 100, 200}
```

- `1` is a start because `0` is missing.
- `2` is not a start because `1` exists.
- `3` is not a start because `2` exists.
- `4` is not a start because `3` exists.
- `100` is a start because `99` is missing.
- `200` is a start because `199` is missing.

So instead of counting from every number, count only from numbers that are sequence starts.

This is the central first-principles insight:

> Every sequence should be measured exactly once, from its first value, and skipped from all interior values.

### 5. The Set/Start Invariant

The algorithm maintains this invariant while iterating through the set:

```text
A number x is expanded only if x is the first value of its maximal consecutive sequence.
```

The condition for that invariant is:

```python
if num - 1 not in num_set:
    # num is a start
```

Once `num` is known to be a start, we can safely walk forward:

```text
num, num + 1, num + 2, ...
```

until the first missing value.

That walk measures the full maximal sequence beginning at `num` because:

- It starts at a value that exists.
- It includes each next value while that value exists.
- It stops exactly when the next required value is missing.

Numbers that are not starts are skipped because their sequence will already be measured when the loop reaches the true start.

This is what prevents repeated work.

### 6. Detailed Algorithm

1. If the input is empty, the set will also be empty and the answer remains `0`.
2. Put every number into a set called `num_set`.
3. Initialize `best = 0`.
4. Iterate over each distinct number `num` in `num_set`.
5. If `num - 1` exists in `num_set`, skip `num` because it is inside a longer sequence that starts earlier.
6. Otherwise, `num` is the start of a sequence:
   - Set `current = num`.
   - Set `length = 1`.
   - While `current + 1` exists in `num_set`, move `current` forward and increase `length`.
7. Update `best` with the longest length found.
8. Return `best`.

Pseudocode:

```python
def longestConsecutive(nums):
    num_set = set(nums)
    best = 0

    for num in num_set:
        if num - 1 in num_set:
            continue

        current = num
        length = 1

        while current + 1 in num_set:
            current += 1
            length += 1

        best = max(best, length)

    return best
```

A Python implementation matching the intended solution shape is:

```python
from typing import List


class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        num_set = set(nums)
        best = 0

        for num in num_set:
            if num - 1 in num_set:
                continue

            current = num
            length = 1

            while current + 1 in num_set:
                current += 1
                length += 1

            best = max(best, length)

        return best
```

The iteration order of a set is irrelevant. We are not relying on visiting numbers in sorted order. We are only using membership checks.

### 7. Detailed Example Walkthrough

Use the first example:

```text
nums = [100, 4, 200, 1, 3, 2]
```

Build the set:

```text
num_set = {1, 2, 3, 4, 100, 200}
```

The order in which the set is iterated does not matter, so think about each value by its role.

#### Check `1`

Ask whether `1` is a start:

```text
1 - 1 = 0
0 is not in num_set
```

So `1` starts a sequence.

Now walk forward:

```text
current = 1, length = 1
2 exists -> current = 2, length = 2
3 exists -> current = 3, length = 3
4 exists -> current = 4, length = 4
5 missing -> stop
```

The sequence is:

```text
1, 2, 3, 4
```

Update:

```text
best = 4
```

#### Check `2`

Ask whether `2` is a start:

```text
2 - 1 = 1
1 is in num_set
```

So `2` is not a start. Skip it.

This is important: without this skip, we would recount:

```text
2, 3, 4
```

which cannot beat the already measured sequence starting at `1`.

#### Check `3`

```text
3 - 1 = 2
2 is in num_set
```

Skip `3` because it is an interior value.

#### Check `4`

```text
4 - 1 = 3
3 is in num_set
```

Skip `4` because it is an interior value.

#### Check `100`

```text
100 - 1 = 99
99 is not in num_set
```

So `100` starts a sequence.

Walk forward:

```text
101 missing
```

The sequence is only:

```text
100
```

Its length is `1`, so `best` stays `4`.

#### Check `200`

```text
200 - 1 = 199
199 is not in num_set
```

So `200` starts a sequence.

Walk forward:

```text
201 missing
```

The length is `1`, so `best` stays `4`.

Return:

```text
4
```

### 8. Walkthrough With Duplicates

Use the third example:

```text
nums = [1, 0, 1, 2]
```

The set is:

```text
num_set = {0, 1, 2}
```

The duplicate `1` disappears because it does not create an additional integer value.

Now identify starts:

- `0` is a start because `-1` is missing.
- `1` is not a start because `0` exists.
- `2` is not a start because `1` exists.

Expand from `0`:

```text
0 exists, length = 1
1 exists, length = 2
2 exists, length = 3
3 missing, stop
```

Return:

```text
3
```

This example shows why the algorithm must reason about distinct values, not array positions.

### 9. Correctness

We need to prove that the algorithm returns the length of the longest consecutive sequence.

#### Lemma 1: Every maximal consecutive sequence has exactly one start.

Consider any maximal consecutive sequence:

```text
a, a + 1, a + 2, ..., b
```

Because it is maximal, `a - 1` is not in the set. If `a - 1` existed, the sequence could extend left, contradicting that `a` is the first value.

Also, every other value in the sequence is not a start. For any value `x > a`, the previous value `x - 1` is also in the sequence, so `x - 1` is in the set.

Therefore, the sequence has exactly one start: `a`.

#### Lemma 2: When the algorithm expands from a start, it measures the entire maximal sequence beginning there.

The algorithm begins with a start value `a` that exists in the set.

Then it repeatedly checks:

```text
current + 1 in num_set
```

As long as the next value exists, it belongs to the same consecutive sequence and the algorithm includes it. The first time the next value is missing, the sequence cannot continue to the right.

Therefore, the measured length is exactly the length of the maximal consecutive sequence starting at `a`.

#### Lemma 3: The algorithm measures every maximal consecutive sequence at least once.

By Lemma 1, every maximal consecutive sequence has a start `a` where `a - 1` is missing.

The algorithm iterates over every value in the set, so it eventually considers `a`. Since `a - 1` is missing, the algorithm expands from `a`.

So every maximal sequence is measured.

#### Lemma 4: The algorithm never needs to expand from a non-start.

If `x - 1` exists, then `x` belongs to a sequence that began at some earlier value. By Lemma 2, that earlier start will measure the full sequence including `x`.

Expanding from `x` would only measure a suffix of an already measured sequence, so skipping it cannot lose the optimal answer.

#### Theorem: The algorithm returns the length of the longest consecutive sequence.

By Lemma 3, every maximal consecutive sequence is measured. By Lemma 2, each measured length is correct. By Lemma 4, skipped values cannot produce a longer sequence than their true start would produce. Since the algorithm keeps the maximum measured length in `best`, the final value of `best` is exactly the longest consecutive sequence length.

### 10. Complexity

Let `n` be the length of `nums`.

Building the set costs expected `O(n)` time and uses `O(n)` space in the worst case.

The outer loop visits each distinct value once.

The inner `while` loop can look scary because it is nested, but across the entire algorithm each distinct value is advanced through as part of a forward walk only when walking from the start of its sequence. Interior values are skipped by the start check.

So the total expected time is:

```text
O(n)
```

The auxiliary space is:

```text
O(n)
```

because the set may contain every input value.

The `O(n)` time bound is expected because hash set operations are expected `O(1)`.

### 11. Common Pitfalls

#### Pitfall 1: Sorting when the follow-up asks for linear time

Sorting gives a valid alternative idea:

```text
sort nums, then count consecutive runs
```

But sorting costs `O(n log n)`, which misses the usual LeetCode follow-up requirement to solve the problem in `O(n)` time.

The set approach avoids sorting entirely.

#### Pitfall 2: Counting from every number

If you write:

```python
for num in num_set:
    while num + 1 in num_set:
        ...
```

without checking whether `num - 1` exists, you may recount the same long sequence many times.

The start check is what turns the algorithm from repeated suffix counting into one measurement per sequence.

#### Pitfall 3: Treating duplicates as part of the sequence length

For:

```text
[1, 0, 1, 2]
```

The answer is `3`, not `4`.

A consecutive sequence is about distinct integer values. Duplicate copies do not extend it.

#### Pitfall 4: Depending on set iteration order

A set does not promise sorted order.

The algorithm is still correct because it never needs sorted iteration. It only asks local membership questions:

```text
Does num - 1 exist?
Does current + 1 exist?
```

#### Pitfall 5: Forgetting the empty input case

If `nums` is empty, the set is empty, the loop never runs, and `best` should remain `0`.

Initialize:

```python
best = 0
```

not `1`.

#### Pitfall 6: Removing values while iterating over the same set

Some versions of this problem remove visited values from the set to avoid repeated work. That can be correct if done carefully, but mutating a set while iterating over it directly is unsafe in Python.

The start-invariant solution does not need removal, so it avoids that issue.

### 12. First-Principles Summary

The problem looks like it might require arranging the numbers, but arrangement is not the core issue.

The core issue is value existence.

A consecutive sequence is fully determined by questions of the form:

```text
Does this integer exist in the input?
```

A hash set gives constant-time answers to those questions.

Then the decisive observation is that a sequence should be counted from its beginning, not from every value inside it.

The beginning of a sequence is exactly a value whose predecessor is missing:

```text
num - 1 not in num_set
```

Once we count only from starts, every maximal sequence is measured once, all interior starts are skipped, duplicates disappear naturally, and the longest measured length is the answer.

## Implementation
See `solutions/hash_table/p128_longest_consecutive_sequence.py`.

## Tests
See `tests/hash_table/test_p128_longest_consecutive_sequence.py`.

## Examples

### Example 1
- Input: `{'nums': [100, 4, 200, 1, 3, 2]}`
- Output: `4`

### Example 2
- Input: `{'nums': [0, 3, 7, 2, 5, 8, 4, 6, 0, 1]}`
- Output: `9`

### Example 3
- Input: `{'nums': [1, 0, 1, 2]}`
- Output: `3`

## Follow-up Practice
- Explain why `num - 1 not in num_set` identifies exactly the beginning of a sequence.
- Trace the algorithm on an input with negative numbers, such as `[-1, -2, 0, 10]`.
- Trace an input with duplicates and confirm that duplicates do not change the answer.
- Compare the set approach with sorting and identify where the `O(n)` expected-time improvement comes from.
