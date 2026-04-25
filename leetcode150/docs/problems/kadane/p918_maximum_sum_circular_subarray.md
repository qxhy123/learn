# 918. Maximum Sum Circular Subarray

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/maximum-sum-circular-subarray/
- Official Group: Kadane's Algorithm
- Pattern Group: Kadane
- Patterns: kadane, window-or-prefix, sum

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given an integer array `nums`.

A normal subarray is a non-empty contiguous slice:

```text
nums[left:right + 1]
```

For example, in:

```text
nums = [5, -3, 5]
```

normal subarrays include:

```text
[5]
[5, -3]
[5, -3, 5]
[-3, 5]
```

This problem adds one twist: the array is circular.

Circular means the element after the last element is the first element again. So a valid subarray is still contiguous, but it may wrap around the end of the array.

For:

```text
nums = [5, -3, 5]
```

we may take:

```text
last 5 + first 5 = [5, 5]
```

That wrapped subarray has sum `10`, even though those two elements are not adjacent in the normal linear view.

The problem asks:

> Among all non-empty contiguous subarrays in the circular array, return the maximum possible sum.

The phrase non-empty matters. We are not allowed to choose no elements just to get sum `0`.

---

### 2. Start From the Brute Force Baseline

The most direct way to reason about the problem is:

1. Pick a starting index.
2. Pick a length from `1` to `n`.
3. Walk forward that many elements, using modulo arithmetic to wrap around.
4. Compute the sum.
5. Keep the largest sum seen.

Conceptually:

```python
best = -infinity
n = len(nums)

for start in range(n):
    total = 0
    for length in range(1, n + 1):
        index = (start + length - 1) % n
        total += nums[index]
        best = max(best, total)
```

This is correct because every circular subarray has exactly one starting index and a length between `1` and `n`.

But it is too slow for large inputs:

```text
n starting positions * n possible lengths = O(n^2)
```

The goal is to avoid enumerating every wrapped interval. We need a way to describe all optimal possibilities with a small amount of state.

---

### 3. Split the Problem Into Two Shapes

Any optimal circular subarray has one of two shapes.

#### Shape A: It Does Not Wrap

It lies completely inside the linear array:

```text
[ ... chosen chosen chosen ... ]
```

Example:

```text
nums = [1, -2, 3, -2]
```

The best answer is `[3]`, sum `3`. It does not use the circular boundary.

This is exactly the classic maximum subarray problem.

#### Shape B: It Wraps

It takes some suffix from the end and some prefix from the beginning:

```text
chosen chosen ... skipped skipped ... chosen chosen
```

Example:

```text
nums = [5, -3, 5]
```

The best circular subarray takes:

```text
prefix [5] + suffix [5]
```

and skips the middle `[-3]`.

That observation is the key:

> A wrapping maximum subarray is the total array sum minus one contiguous middle subarray that we choose not to take.

For `[5, -3, 5]`:

```text
total sum      = 7
skipped middle = -3
wrapped sum    = 7 - (-3) = 10
```

So the best wrapped answer is found by removing the minimum-sum non-empty subarray from the middle.

That gives two candidates:

```text
non_wrapped_best = maximum subarray sum
wrapped_best     = total_sum - minimum subarray sum
answer           = max(non_wrapped_best, wrapped_best)
```

There is one important exception: if the minimum subarray is the entire array, then `total_sum - minimum_subarray_sum` chooses nothing. That is invalid because the answer must be non-empty.

We will handle that case explicitly.

---

### 4. Kadane's Maximum-Subarray Invariant

For the non-wrapping case, use Kadane's algorithm.

When scanning from left to right, define:

```text
current_max = best sum of a non-empty subarray that must end at the current index
best_max    = best sum of any non-empty subarray seen so far
```

When we read a new value `x`, a subarray ending at `x` has only two possible forms:

1. Extend the best subarray that ended at the previous index.
2. Start fresh at `x`.

So:

```text
current_max = max(x, current_max + x)
best_max    = max(best_max, current_max)
```

This works because a bad prefix should not be carried forward. If the previous ending sum is negative, adding it to `x` only makes the new subarray worse than starting at `x`.

---

### 5. Kadane's Minimum-Subarray Invariant

For the wrapping case, we need the minimum-sum contiguous subarray to skip.

The reasoning is the mirror image.

Define:

```text
current_min = smallest sum of a non-empty subarray that must end at the current index
best_min    = smallest sum of any non-empty subarray seen so far
```

When we read a new value `x`, a minimum subarray ending at `x` also has only two possible forms:

1. Extend the previous minimum-ending subarray.
2. Start fresh at `x`.

So:

```text
current_min = min(x, current_min + x)
best_min    = min(best_min, current_min)
```

This gives the worst middle segment to remove.

The wrapped candidate is:

```text
wrapped_best = total_sum - best_min
```

The more negative `best_min` is, the larger the wrapped sum becomes.

---

### 6. Why the All-Negative Case Is Special

Consider:

```text
nums = [-3, -2, -3]
```

The maximum non-empty subarray is:

```text
[-2]
```

so `best_max = -2`.

The minimum subarray is the entire array:

```text
[-3, -2, -3]
```

so:

```text
total_sum = -8
best_min  = -8
wrapped_best = total_sum - best_min = 0
```

But `0` represents taking every element except the entire array, which means taking an empty subarray.

That is not allowed.

So when the best normal maximum is negative, all numbers are negative. In that situation the correct answer is simply the largest single element, which Kadane already stored in `best_max`.

Practical rule:

```text
if best_max < 0:
    return best_max
else:
    return max(best_max, total_sum - best_min)
```

Some implementations use `best_max <= 0`; `best_max < 0` is enough for LeetCode's constraints because if there is a `0`, choosing `[0]` is valid and returning `0` is still correct.

---

### 7. Detailed Algorithm

Initialize all running values from the first element, not from `0`:

```text
total_sum   = nums[0]
current_max = nums[0]
best_max    = nums[0]
current_min = nums[0]
best_min    = nums[0]
```

Then scan the rest of the array.

For each `x`:

1. Add `x` to `total_sum`.
2. Update the best maximum subarray ending here:

   ```text
   current_max = max(x, current_max + x)
   ```

3. Update the global maximum subarray:

   ```text
   best_max = max(best_max, current_max)
   ```

4. Update the best minimum subarray ending here:

   ```text
   current_min = min(x, current_min + x)
   ```

5. Update the global minimum subarray:

   ```text
   best_min = min(best_min, current_min)
   ```

After the scan:

```text
if best_max < 0:
    return best_max

return max(best_max, total_sum - best_min)
```

The algorithm stores exactly the information needed for the two possible shapes of the answer: best non-wrapping sum and best wrapping sum.

---

### 8. Pseudocode

```python
def maxSubarraySumCircular(nums):
    total_sum = nums[0]

    current_max = nums[0]
    best_max = nums[0]

    current_min = nums[0]
    best_min = nums[0]

    for x in nums[1:]:
        total_sum += x

        current_max = max(x, current_max + x)
        best_max = max(best_max, current_max)

        current_min = min(x, current_min + x)
        best_min = min(best_min, current_min)

    if best_max < 0:
        return best_max

    return max(best_max, total_sum - best_min)
```

A compact Python implementation often computes all five values in one loop, but the meaning is the same: track the best subarray to keep and the worst subarray to exclude.

---

### 9. Detailed Walkthrough: `[5, -3, 5]`

The circular answer should be `10`, because we can take the last `5` and the first `5` while skipping `-3`.

Start with the first value:

```text
total_sum   = 5
current_max = 5
best_max    = 5
current_min = 5
best_min    = 5
```

Now read `-3`.

Maximum-ending decision:

```text
extend previous: 5 + (-3) = 2
start fresh:     -3
current_max = 2
best_max = max(5, 2) = 5
```

Minimum-ending decision:

```text
extend previous: 5 + (-3) = 2
start fresh:     -3
current_min = -3
best_min = min(5, -3) = -3
```

Total:

```text
total_sum = 2
```

Now read `5`.

Maximum-ending decision:

```text
extend previous: 2 + 5 = 7
start fresh:     5
current_max = 7
best_max = max(5, 7) = 7
```

Minimum-ending decision:

```text
extend previous: -3 + 5 = 2
start fresh:      5
current_min = 2
best_min = min(-3, 2) = -3
```

Total:

```text
total_sum = 7
```

Now compare the two shapes:

```text
non_wrapped_best = best_max = 7          # [5, -3, 5]
wrapped_best     = total_sum - best_min
                 = 7 - (-3)
                 = 10                   # skip [-3], take [5] + [5]
```

Return:

```text
max(7, 10) = 10
```

---

### 10. Walkthrough: `[1, -2, 3, -2]`

Here the best answer is `3`.

The best non-wrapping subarray is:

```text
[3]
```

The total sum is:

```text
1 + (-2) + 3 + (-2) = 0
```

The minimum subarray is:

```text
[-2]
```

so the wrapped candidate is:

```text
total_sum - best_min = 0 - (-2) = 2
```

That corresponds to taking everything except one `-2`, such as:

```text
[3, -2, 1]
```

with sum `2`.

The non-wrapped candidate `3` is better, so the answer is `3`.

This example shows why we cannot assume the best answer wraps just because the array is circular. Circularity adds another candidate; it does not replace the ordinary maximum-subarray case.

---

### 11. Correctness

We prove that the algorithm returns the maximum sum of any non-empty circular subarray.

#### Lemma 1: `best_max` is the maximum sum of any non-empty non-wrapping subarray.

At each index, `current_max` stores the maximum sum of a non-empty subarray that ends exactly at that index.

A subarray ending at the current index either starts at the current element or extends a subarray ending at the previous index. The recurrence:

```text
current_max = max(x, current_max + x)
```

chooses the better of those two complete possibilities.

Then `best_max` is updated with every `current_max`, so after the scan it is the best among all ending positions. Therefore, `best_max` is the maximum sum of any non-empty linear subarray.

#### Lemma 2: `best_min` is the minimum sum of any non-empty non-wrapping subarray.

The same argument applies with `min` instead of `max`.

At each index, `current_min` stores the minimum sum of a non-empty subarray ending exactly at that index. Such a subarray either starts at the current element or extends the previous minimum-ending subarray.

So after scanning all positions, `best_min` is the minimum sum of any non-empty linear subarray.

#### Lemma 3: Every maximum wrapping subarray has sum `total_sum - removed_sum` for some non-empty linear subarray removed from the middle.

A wrapping subarray contains a suffix of the array and a prefix of the array. The elements not chosen form one contiguous block in the middle of the linear array.

Therefore, its sum equals:

```text
total_sum - sum(unchosen middle block)
```

To maximize this wrapped sum, we must minimize the sum of the unchosen middle block. By Lemma 2, the smallest possible middle-block sum is `best_min`.

So the best valid wrapping candidate is `total_sum - best_min`, except when the removed block is the entire array.

#### Lemma 4: If `best_max < 0`, all numbers are negative and no wrapping candidate is valid.

If any number were zero or positive, `best_max` would be at least that number and therefore not negative.

So `best_max < 0` means every number is negative. In that case, the minimum subarray is the entire array, and `total_sum - best_min` would choose the empty subarray. The problem forbids empty subarrays, so the correct answer is the largest single element, which is `best_max`.

#### Theorem: The algorithm returns the correct answer.

Every valid circular subarray either does not wrap or wraps.

- By Lemma 1, the best non-wrapping sum is `best_max`.
- By Lemmas 2 and 3, the best wrapping sum is `total_sum - best_min` when that represents a non-empty selection.
- By Lemma 4, the all-negative case must return `best_max` instead of the empty wrapped candidate.

The algorithm returns the maximum of the only two possible valid shapes, with the invalid all-negative wrapped case excluded. Therefore, it returns the maximum sum of any non-empty circular subarray.

---

### 12. Complexity

Let `n` be the length of `nums`.

- Time: `O(n)` because each element is processed once.
- Space: `O(1)` because the algorithm stores only a fixed number of sums.

---

### 13. Common Pitfalls

- Initializing Kadane values to `0`. This incorrectly allows an empty subarray and breaks all-negative inputs like `[-3, -2, -3]`.
- Returning `total_sum - best_min` unconditionally. If `best_min` is the whole array, the wrapped candidate is empty and invalid.
- Thinking a circular subarray can skip multiple separated blocks. A valid wrapped subarray skips exactly one contiguous middle block.
- Forgetting the non-wrapping case. Some arrays, such as `[1, -2, 3, -2]`, are best solved by an ordinary linear subarray.
- Updating only maximum Kadane state. The circular case also needs minimum-subarray state.
- Treating modulo simulation as necessary. Modulo is useful for brute force, but the optimized solution avoids physically duplicating or wrapping the array.

---

### 14. First-Principles Summary

A circular maximum subarray looks complicated because it may cross the boundary between the end and beginning of the array.

But from first principles, there are only two shapes:

```text
1. Do not wrap: solve ordinary maximum subarray.
2. Wrap: take total sum minus one minimum-sum middle block.
```

Kadane's algorithm gives the first shape by tracking the best subarray ending at each index.

The mirrored Kadane recurrence gives the second shape by tracking the worst subarray ending at each index.

The only subtlety is non-emptiness: when all numbers are negative, removing the minimum subarray removes the entire array, so the wrapped candidate must be rejected.

Once these ideas are separated, the algorithm is just one scan that maintains:

```text
total_sum
current_max, best_max
current_min, best_min
```

and then compares the non-wrapping and wrapping candidates.

## Implementation
See `solutions/kadane/p918_maximum_sum_circular_subarray.py`.

## Tests
See `tests/kadane/test_p918_maximum_sum_circular_subarray.py`.

## Examples

### Example 1
- Input: `{'nums': [1, -2, 3, -2]}`
- Output: `3`

### Example 2
- Input: `{'nums': [5, -3, 5]}`
- Output: `10`

### Example 3
- Input: `{'nums': [-3, -2, -3]}`
- Output: `-2`

## Follow-up Practice
- Trace an all-negative input and explain why the wrapped candidate is invalid.
- Trace an input where wrapping wins, such as `[5, -3, 5]`.
- Trace an input where the ordinary maximum subarray wins, such as `[1, -2, 3, -2]`.
- Write both Kadane recurrences in words before writing code.
