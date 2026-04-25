# 1. Two Sum

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/two-sum/
- Official Group: Hashmap
- Pattern Group: Hash Table
- Patterns: hash-table, sum

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given:

```text
nums:   a list of integers
target: one integer
```

You must return the indices of two different elements whose values add up to `target`.

So if:

```text
nums = [2, 7, 11, 15]
target = 9
```

then the answer is:

```text
[0, 1]
```

because:

```text
nums[0] + nums[1] = 2 + 7 = 9
```

The problem is not asking for the two values. It is asking for the two indices.

That detail matters because duplicate values can appear:

```text
nums = [3, 3]
target = 6
```

The answer is:

```text
[0, 1]
```

Both values are `3`, but they are two different elements at two different indices.

The core requirement is therefore:

```text
Find i and j such that:
1. i != j
2. nums[i] + nums[j] == target
```

LeetCode's version guarantees that exactly one valid answer exists, so the algorithm can return as soon as it finds the pair.

### 2. Start From the Brute Force Baseline

The most direct way to solve the problem is to try every pair of indices.

Conceptually:

```python
for i in range(len(nums)):
    for j in range(i + 1, len(nums)):
        if nums[i] + nums[j] == target:
            return [i, j]
```

This is correct because every possible pair appears somewhere in the nested loops.

For example, for:

```text
nums = [2, 7, 11, 15]
```

The checked pairs are:

```text
(0, 1): 2 + 7
(0, 2): 2 + 11
(0, 3): 2 + 15
(1, 2): 7 + 11
(1, 3): 7 + 15
(2, 3): 11 + 15
```

If the answer exists, brute force will eventually find it.

But the cost is high. For each index, it may scan many later indices. The number of pairs grows roughly like:

```text
n * (n - 1) / 2
```

So brute force is:

```text
Time:  O(n^2)
Space: O(1)
```

The deeper question is:

> Why are we repeatedly searching for the second number from scratch?

### 3. The Key Observation: Every Number Has One Needed Partner

Suppose we are looking at one value:

```text
current = nums[i]
```

For `current` to be part of a valid answer, the other value must be:

```text
target - current
```

This value is called the complement.

For example:

```text
target = 9
current = 2
complement = 9 - 2 = 7
```

So when we see `2`, we do not need to ask:

```text
Which of all other numbers might work with 2?
```

We can ask the much sharper question:

```text
Have I seen a 7 already?
```

That is the whole problem in one sentence:

> For each number, check whether its complement has already appeared.

This turns a pair-search problem into a lookup problem.

### 4. Why a Hash Map Fits the Problem

A hash map lets us remember previously seen values and retrieve their indices quickly.

We want a table with this meaning:

```text
seen[value] = index where that value appeared earlier
```

Then, when processing `nums[i]`, we compute:

```text
complement = target - nums[i]
```

and ask:

```text
Is complement in seen?
```

If yes, then we already have an earlier index `seen[complement]`, and the current index is `i`.

So the answer is:

```text
[seen[complement], i]
```

If no, then the current number cannot yet complete a pair with any earlier number. But it might be the complement for a future number, so we store it:

```text
seen[nums[i]] = i
```

### 5. The Hash-Map Invariant

The invariant is the exact fact that makes the algorithm correct:

```text
Before processing index i, seen contains every value from nums[0:i]
paired with an index where that value appeared.
```

In other words, `seen` represents only the past.

That matters because the two indices must be different.

At index `i`, the algorithm asks:

```text
Can nums[i] form the target sum with one earlier element?
```

It does not ask whether `nums[i]` can pair with itself.

That is why the order is important:

```text
1. Check complement first.
2. Store current value second.
```

If we stored first and then checked, this case would be dangerous:

```text
nums = [3]
target = 6
```

A careless implementation might store `3` at index `0`, then find complement `3`, and incorrectly use the same element twice.

Checking before storing prevents that because `seen` contains only earlier indices.

### 6. Detailed Algorithm

For each index and value in `nums`:

```text
1. Let complement = target - value.
2. If complement is already in seen:
      return [seen[complement], current index]
3. Otherwise, store value with its current index:
      seen[value] = current index
```

The table grows as the scan moves left to right.

Each step uses the current value only once and asks whether the required earlier partner already exists.

Python implementation:

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        seen = {}

        for index, value in enumerate(nums):
            complement = target - value

            if complement in seen:
                return [seen[complement], index]

            seen[value] = index

        raise ValueError("No two-sum solution exists")
```

On LeetCode, the final error is usually unnecessary because the problem guarantees one solution. It is included here only to make the control flow explicit.

### 7. Walkthrough: Example 1

Input:

```text
nums = [2, 7, 11, 15]
target = 9
```

Start with an empty table:

```text
seen = {}
```

#### Step 1: index `0`, value `2`

Compute the complement:

```text
9 - 2 = 7
```

Ask whether `7` has appeared before:

```text
seen = {}
7 is not in seen
```

So store the current value:

```text
seen = {2: 0}
```

Meaning:

```text
value 2 appeared at index 0
```

#### Step 2: index `1`, value `7`

Compute the complement:

```text
9 - 7 = 2
```

Ask whether `2` has appeared before:

```text
seen = {2: 0}
2 is in seen
```

So we found:

```text
seen[2] = 0
current index = 1
```

Return:

```text
[0, 1]
```

And indeed:

```text
nums[0] + nums[1] = 2 + 7 = 9
```

### 8. Walkthrough: Duplicate Values

Input:

```text
nums = [3, 3]
target = 6
```

Start:

```text
seen = {}
```

#### Step 1: index `0`, value `3`

Complement:

```text
6 - 3 = 3
```

`3` is not in `seen`, so store the current value:

```text
seen = {3: 0}
```

#### Step 2: index `1`, value `3`

Complement:

```text
6 - 3 = 3
```

Now `3` is in `seen`:

```text
seen[3] = 0
```

Return:

```text
[0, 1]
```

This is valid because the two `3`s are at different indices.

The duplicate case is exactly why the invariant says `seen` contains earlier values, not the current value.

### 9. Why This Code Is Correct

We prove correctness using the invariant:

```text
Before processing index i, seen contains the values and indices from nums[0:i].
```

At index `i`, the algorithm computes:

```text
complement = target - nums[i]
```

If `complement` is in `seen`, then there is an earlier index `j` such that:

```text
nums[j] = complement
```

Therefore:

```text
nums[j] + nums[i]
= complement + nums[i]
= target
```

Because `j` came from `seen`, it is earlier than `i`, so:

```text
j != i
```

Thus returning `[j, i]` is valid.

If `complement` is not in `seen`, then no earlier element can pair with `nums[i]` to make `target`. The algorithm stores `nums[i]` so that future elements can use it as their earlier partner. This restores the invariant for the next index.

Since the problem guarantees that a valid pair exists, consider the valid pair with indices:

```text
j < i
```

When the scan reaches `i`, index `j` has already been processed and stored in `seen`. The complement of `nums[i]` is exactly `nums[j]`, so the algorithm finds it and returns the pair.

Therefore, the algorithm returns two distinct indices whose values add to `target`.

### 10. Complexity

The algorithm scans the array once.

For each element, it performs a constant number of hash-map operations:

```text
lookup
insert
```

Hash-map operations are expected `O(1)`, so the total expected time is:

```text
Time: O(n)
```

In the worst case, the table may store almost every number before the answer is found:

```text
Space: O(n)
```

This is the tradeoff:

> Use extra memory to avoid repeatedly scanning the array for complements.

### 11. Common Pitfalls

#### Checking After Storing

This can accidentally allow the same element to be used twice when `value * 2 == target`.

Correct order:

```text
check complement first
then store current value
```

#### Returning Values Instead of Indices

The problem asks for:

```text
indices
```

not:

```text
values
```

For `[2, 7, 11, 15]`, return `[0, 1]`, not `[2, 7]`.

#### Mishandling Duplicates

Duplicates are allowed. The input:

```text
[3, 3], target = 6
```

requires two different indices.

A value-to-index hash map handles this naturally as long as the current element is stored only after the complement check.

#### Assuming the Pair Must Be Adjacent

The two numbers can be anywhere in the array.

For example:

```text
nums = [5, 1, 9, 4]
target = 9
```

The answer uses indices `0` and `3`, not adjacent elements.

#### Overthinking the Return Order

The usual one-pass implementation returns the earlier index first and the current index second:

```text
[seen[complement], index]
```

This matches the examples and keeps the result easy to reason about.

### 12. First-Principles Summary

Two Sum starts as a pair-search problem:

```text
Find two indices whose values add to target.
```

The brute-force method checks every pair because it has no memory.

The key observation is that once one value is fixed, the other value is forced:

```text
other = target - value
```

So instead of searching all pairs, scan once and remember the values already seen.

The invariant is:

```text
seen stores values from earlier indices only.
```

That invariant gives each step a simple local decision:

```text
If the needed complement is in seen, return the earlier index and current index.
Otherwise, store the current value for future elements.
```

The result is a direct tradeoff:

```text
O(n) extra space buys O(n) expected time.
```

## Implementation

See `solutions/hash_table/p001_two_sum.py`.

## Tests

See `tests/hash_table/test_p001_two_sum.py`.

## Examples

### Example 1
- Input: `{'nums': [2, 7, 11, 15], 'target': 9}`
- Output: `[0, 1]`

### Example 2
- Input: `{'nums': [3, 2, 4], 'target': 6}`
- Output: `[1, 2]`

### Example 3
- Input: `{'nums': [3, 3], 'target': 6}`
- Output: `[0, 1]`

## Follow-up Practice

- Explain why the complement of `value` is `target - value`.
- Trace the contents of `seen` after each index.
- Check why duplicates like `[3, 3]` work.
- Compare the number of pair checks in brute force with the number of hash lookups in the one-pass solution.
