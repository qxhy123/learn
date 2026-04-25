# 219. Contains Duplicate II

- Difficulty: Easy
- LeetCode: https://leetcode.com/problems/contains-duplicate-ii/
- Official Group: Hashmap
- Pattern Group: Hash Table
- Patterns: hash-table

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given:

```text
nums = an array of integers
k    = a maximum allowed distance between indices
```

The task is to decide whether there are two different indices `i` and `j` such that:

```text
nums[i] == nums[j]
abs(i - j) <= k
```

There are two separate requirements:

```text
same value
nearby indices
```

A duplicate value alone is not enough.

For example:

```text
nums = [1, 2, 3, 1]
k = 2
```

The value `1` appears twice, but its indices are `0` and `3`.

Their distance is:

```text
3 - 0 = 3
```

Since `3 > 2`, this duplicate is too far away, so it does not satisfy the problem.

But if:

```text
nums = [1, 2, 3, 1]
k = 3
```

then the same pair is valid because:

```text
3 - 0 = 3 <= k
```

So the problem is really asking:

> While scanning the array, can we ever find the same value again within the last `k` positions?

That sentence is the entire shape of the solution.

### 2. Start From the Brute Force Baseline

The most direct solution is to try every pair of indices:

```python
for i in range(len(nums)):
    for j in range(i + 1, len(nums)):
        if nums[i] == nums[j] and j - i <= k:
            return True

return False
```

This is correct because it explicitly checks every possible pair.

However, it is inefficient.

There are `O(n^2)` pairs in the worst case. For each index `i`, the inner loop may scan many later indices. Most of that work is repeated searching: when we stand at position `j`, we repeatedly ask whether this value appeared before nearby.

The important question is:

> Do we really need to compare the current number against every previous number?

No.

For the current index `j`, only previous indices with the same value matter. All different values can never form a valid pair with `nums[j]`.

So instead of remembering every previous number as a list to scan, we want direct access to the previous index where each value appeared.

### 3. The Key Observation

Suppose we are scanning left to right and currently stand at index `i`.

If there is a valid pair ending at `i`, then there must be some earlier index `j` such that:

```text
nums[j] == nums[i]
i - j <= k
```

Because we scan left to right, `i > j`, so `abs(i - j)` becomes simply:

```text
i - j
```

Now consider all previous occurrences of `nums[i]`.

Do we need all of them?

No. The most useful one is the most recent previous occurrence.

If the most recent previous index of this value is `last`, then any older occurrence is even farther away:

```text
older < last < i
```

Therefore:

```text
i - older > i - last
```

So if the most recent occurrence is too far away, every older occurrence is also too far away.

This gives the central first-principles insight:

> For each value, it is enough to remember only its most recent index.

That is exactly what a hash map gives us:

```text
value -> most recent index where this value was seen
```

### 4. The Hash-Map Distance Invariant

Maintain a dictionary called `last_seen`.

After processing indices before `i`, the invariant is:

```text
last_seen[x] is the greatest index < i where value x appeared
```

In other words, just before we process `nums[i]`, the map answers this question in expected `O(1)` time:

```text
Where did this value appear most recently before now?
```

Then the local decision is simple:

1. Let `x = nums[i]`.
2. If `x` has appeared before at index `last_seen[x]`, compute the distance:

   ```text
   i - last_seen[x]
   ```

3. If that distance is at most `k`, we found a valid pair and can return `True`.
4. Otherwise, update `last_seen[x] = i` because the current occurrence is now the most recent one.

The update is not optional.

Even if the previous occurrence was too far away, the current occurrence might help a later index form a valid pair.

For example:

```text
nums = [1, 2, 1, 1]
k = 1
```

At index `2`, the previous `1` is at index `0`:

```text
2 - 0 = 2 > 1
```

That pair is not valid.

But we must update the most recent index of `1` to `2`, because index `3` forms a valid nearby duplicate with it:

```text
3 - 2 = 1 <= 1
```

### 5. Sliding Window Interpretation

This problem can also be viewed as a sliding-window membership problem.

At index `i`, the only earlier indices that can pair with `i` are:

```text
i - k, i - k + 1, ..., i - 1
```

So the question is:

> Does `nums[i]` already exist inside the previous window of size at most `k`?

One valid implementation keeps a set of the last `k` values and removes values that fall out of range.

The hash-map implementation with most recent indices is slightly more direct. Instead of physically maintaining the window contents, it stores the latest index for each value and checks whether that index is still inside the allowed window:

```text
last_seen[nums[i]] >= i - k
```

This is equivalent to:

```text
i - last_seen[nums[i]] <= k
```

So the invariant can be described in two complementary ways:

```text
Hash-map view: value -> most recent index
Sliding-distance view: most recent same value must be no more than k steps back
```

Both viewpoints lead to the same algorithm.

### 6. Detailed Algorithm

Use a dictionary `last_seen`.

For each index `i` and value `num` in `nums`:

1. Check whether `num` is already in `last_seen`.
2. If it is, let `previous = last_seen[num]`.
3. Compute the distance:

   ```text
   i - previous
   ```

4. If the distance is at most `k`, return `True`.
5. Store the current index as the most recent occurrence:

   ```python
   last_seen[num] = i
   ```

6. If the loop finishes without finding such a pair, return `False`.

The check happens before the update because we need to compare the current index with a previous index. If we updated first, we would overwrite the previous occurrence and compare the index with itself.

### 7. Pseudocode

```python
def containsNearbyDuplicate(nums, k):
    last_seen = {}

    for i, num in enumerate(nums):
        if num in last_seen:
            previous = last_seen[num]
            if i - previous <= k:
                return True

        last_seen[num] = i

    return False
```

A Python implementation for the repository would also need the usual type import if using `List[int]` in the signature:

```python
from typing import List
```

### 8. Walkthrough: Example 1

```text
nums = [1, 2, 3, 1]
k = 3
```

Start with an empty map:

```text
last_seen = {}
```

#### Index 0, value 1

`1` has not appeared before.

Update:

```text
last_seen = {1: 0}
```

#### Index 1, value 2

`2` has not appeared before.

Update:

```text
last_seen = {1: 0, 2: 1}
```

#### Index 2, value 3

`3` has not appeared before.

Update:

```text
last_seen = {1: 0, 2: 1, 3: 2}
```

#### Index 3, value 1

`1` appeared before at index `0`.

Compute the distance:

```text
3 - 0 = 3
```

Since:

```text
3 <= k
```

we found two equal values at indices `0` and `3` whose distance is at most `3`.

Return:

```text
True
```

### 9. Walkthrough: Example 3

```text
nums = [1, 2, 3, 1, 2, 3]
k = 2
```

Track the most recent index of each number.

#### First three values

After processing indices `0`, `1`, and `2`:

```text
last_seen = {1: 0, 2: 1, 3: 2}
```

#### Index 3, value 1

Previous `1` was at index `0`.

```text
3 - 0 = 3
```

But:

```text
3 > 2
```

So this pair is too far away.

Update the most recent `1`:

```text
last_seen = {1: 3, 2: 1, 3: 2}
```

#### Index 4, value 2

Previous `2` was at index `1`.

```text
4 - 1 = 3 > 2
```

Too far.

Update:

```text
last_seen = {1: 3, 2: 4, 3: 2}
```

#### Index 5, value 3

Previous `3` was at index `2`.

```text
5 - 2 = 3 > 2
```

Too far.

Update:

```text
last_seen = {1: 3, 2: 4, 3: 5}
```

The scan ends with no nearby duplicate.

Return:

```text
False
```

This example shows why merely detecting duplicates is insufficient. Every value appears twice, but every matching pair is distance `3`, while `k` is only `2`.

### 10. Correctness Argument

We prove that the algorithm returns `True` exactly when a valid nearby duplicate exists.

#### Invariant

Before processing index `i`, for every value already seen, `last_seen` stores the greatest index less than `i` where that value appeared.

#### Why the Invariant Holds

Initially, before processing any index, the map is empty, so the invariant is true.

When processing index `i`, the algorithm may inspect `last_seen[nums[i]]`. After the inspection, it sets:

```python
last_seen[nums[i]] = i
```

This makes `i` the greatest seen index for that value. All other values keep their previous greatest seen index. Therefore the invariant is preserved for the next iteration.

#### If the Algorithm Returns `True`, a Valid Pair Exists

The algorithm returns `True` only when it finds a current index `i` and a previous index `previous = last_seen[nums[i]]` such that:

```text
nums[previous] == nums[i]
i - previous <= k
```

Because `previous < i`, this is the same as:

```text
abs(i - previous) <= k
```

So the two indices satisfy exactly the problem's requirements.

#### If a Valid Pair Exists, the Algorithm Returns `True`

Assume there is a valid pair `(j, i)` with `j < i`, `nums[j] == nums[i]`, and:

```text
i - j <= k
```

Consider the moment when the algorithm processes index `i`.

By the invariant, `last_seen[nums[i]]` stores the most recent previous index where `nums[i]` appeared. Call it `previous`.

Since `j` is one previous occurrence of the same value, the most recent previous occurrence cannot be earlier than `j`:

```text
previous >= j
```

Therefore:

```text
i - previous <= i - j <= k
```

So the algorithm will detect that `nums[i]` appeared within distance `k` and return `True`.

Thus, if any valid nearby duplicate exists, the algorithm finds one.

Together, both directions prove correctness.

### 11. Complexity

Let `n = len(nums)`.

Each array element is processed once. Each dictionary lookup and update is expected `O(1)` time.

So the time complexity is:

```text
O(n)
```

The dictionary may store one entry per distinct value. In the worst case, all values are distinct, so the space complexity is:

```text
O(n)
```

If using the sliding-window set variant, the space can be bounded by `O(min(n, k))`, because values older than `k` positions are removed. The most-recent-index map is still accepted and simple, with worst-case `O(n)` space.

### 12. Common Pitfalls

- Checking only whether a duplicate exists and forgetting the distance constraint.
- Comparing indices with `< k` instead of `<= k`; the problem allows distance exactly equal to `k`.
- Updating `last_seen[num]` before checking the old index, which loses the previous occurrence.
- Failing to update the index after a duplicate is too far away; later duplicates may be close to the current one.
- Using `abs(i - previous)` unnecessarily in the left-to-right scan. It is not wrong, but `i - previous` is enough because `previous` is always earlier.
- Treating `k = 0` as a special success case. Two distinct indices cannot have distance `0`, so the algorithm naturally returns `False` unless the input constraints differ from the standard problem.

### 13. First-Principles Summary

A valid answer needs two equal values whose indices are close enough.

The brute-force solution checks every pair, but for a fixed current index, only previous copies of the same value matter. Among those previous copies, only the most recent one matters, because it gives the smallest possible distance to the current index.

So the whole problem collapses to one invariant:

```text
For each value, remember the most recent index where it appeared.
```

At each new index, ask:

```text
Have I seen this value before, and was that most recent occurrence within k steps?
```

If yes, return `True`. If no, record the current index and continue.

That is the first-principles reason a hash map solves the problem in one pass.

## Implementation
See `solutions/hash_table/p219_contains_duplicate_ii.py`.

## Tests
See `tests/hash_table/test_p219_contains_duplicate_ii.py`.

## Examples

### Example 1
- Input: `{'nums': [1, 2, 3, 1], 'k': 3}`
- Output: `True`

### Example 2
- Input: `{'nums': [1, 0, 1, 1], 'k': 1}`
- Output: `True`

### Example 3
- Input: `{'nums': [1, 2, 3, 1, 2, 3], 'k': 2}`
- Output: `False`

## Follow-up Practice
- Trace the map as `value -> most recent index`, not just as a set of seen values.
- For every duplicate, compute the exact index distance and compare it with `k`.
- Test the boundary case where the distance is exactly `k`.
- Test duplicates that are too far apart before a later nearby duplicate appears.
