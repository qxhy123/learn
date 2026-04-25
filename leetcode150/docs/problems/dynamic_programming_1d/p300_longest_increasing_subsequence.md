# 300. Longest Increasing Subsequence

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/longest-increasing-subsequence/
- Official Group: 1D DP
- Pattern Group: Dynamic Programming 1D
- Patterns: dynamic-programming-1d

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given an integer array `nums`, find the length of the longest subsequence whose values are strictly increasing.

Two words matter:

```text
subsequence
strictly increasing
```

A **subsequence** keeps the original left-to-right order, but it does not need to use adjacent elements.

For example, in:

```text
nums = [10, 9, 2, 5, 3, 7, 101, 18]
```

`[2, 5, 7, 101]` is a subsequence because those numbers appear in that order.

`[2, 3, 7, 18]` is also a subsequence.

But `[2, 7, 5]` is not a valid increasing subsequence, because although those values exist, choosing `7` before `5` would make the values decrease.

The problem asks only for the length, not the actual subsequence:

```text
[2, 3, 7, 18] has length 4
```

So the output for the example is:

```text
4
```

The core question is:

> Among all ways to delete zero or more elements while preserving order, what is the longest remaining sequence whose values strictly increase?

### 2. Start From the Brute Force Recursion

At each index, there are two natural choices:

```text
skip nums[i]
take nums[i], if it is larger than the previous taken value
```

That gives a direct recursive search:

```python
def dfs(i, previous_value):
    if i == len(nums):
        return 0

    best = dfs(i + 1, previous_value)  # skip nums[i]

    if nums[i] > previous_value:
        best = max(best, 1 + dfs(i + 1, nums[i]))

    return best
```

This is correct because every subsequence is formed by making exactly one take-or-skip decision at each index.

But it is too slow.

There are up to two choices per element, so the recursion can explore roughly:

```text
2^n
```

paths.

The repeated work is not subtle: many recursive branches arrive at the same kind of question again:

```text
Starting from this position, what is the best increasing continuation after the last chosen element?
```

That observation suggests dynamic programming.

### 3. A Simple DP State: Best Subsequence Ending Here

Instead of asking every possible take-or-skip question, reverse the viewpoint.

Suppose an increasing subsequence ends exactly at index `i`.

Then its last value is:

```text
nums[i]
```

The previous element, if any, must come from an earlier index `j < i`, and it must be smaller:

```text
nums[j] < nums[i]
```

So define:

```text
dp[i] = length of the longest increasing subsequence that ends at index i
```

This state is precise because it fixes the final element.

Every single element can stand alone, so the base value is:

```text
dp[i] = 1
```

To compute `dp[i]`, try all earlier positions `j`:

```text
if nums[j] < nums[i], then a subsequence ending at j can be extended by nums[i]
```

The transition is:

```text
dp[i] = max(dp[i], dp[j] + 1)
```

The final answer is not necessarily `dp[n - 1]`, because the best subsequence may end anywhere. Therefore:

```text
answer = max(dp)
```

### 4. Bottom-Up DP Algorithm

The direct DP algorithm is:

1. Create an array `dp` of length `n`, filled with `1`.
2. Scan `i` from left to right.
3. For each `i`, scan every earlier index `j`.
4. If `nums[j] < nums[i]`, extend the best subsequence ending at `j`.
5. Return the largest value in `dp`.

Code:

```python
class Solution:
    def lengthOfLIS(self, nums: list[int]) -> int:
        n = len(nums)
        dp = [1] * n

        for i in range(n):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)

        return max(dp)
```

This version is often the easiest way to understand the problem because every `dp[i]` has a concrete meaning: the best increasing subsequence whose last chosen element is `nums[i]`.

### 5. DP Example Walkthrough

Use the first example:

```text
nums = [10, 9, 2, 5, 3, 7, 101, 18]
```

Initialize:

```text
dp = [1, 1, 1, 1, 1, 1, 1, 1]
```

Each number alone is an increasing subsequence of length `1`.

#### Index 0: `10`

No earlier values exist.

```text
dp[0] = 1
```

#### Index 1: `9`

Earlier value:

```text
10
```

`10 < 9` is false, so `9` cannot extend a subsequence ending at `10`.

```text
dp[1] = 1
```

#### Index 2: `2`

Earlier values:

```text
10, 9
```

Neither is smaller than `2`.

```text
dp[2] = 1
```

#### Index 3: `5`

Earlier values smaller than `5`:

```text
2
```

So `5` can extend `[2]`:

```text
dp[3] = dp[2] + 1 = 2
```

Now:

```text
dp = [1, 1, 1, 2, 1, 1, 1, 1]
```

#### Index 4: `3`

Earlier values smaller than `3`:

```text
2
```

So `3` can extend `[2]`:

```text
dp[4] = 2
```

Now:

```text
dp = [1, 1, 1, 2, 2, 1, 1, 1]
```

#### Index 5: `7`

Earlier values smaller than `7`:

```text
2, 5, 3
```

Possible extensions:

```text
[2] + 7       -> length 2
[2, 5] + 7    -> length 3
[2, 3] + 7    -> length 3
```

So:

```text
dp[5] = 3
```

Now:

```text
dp = [1, 1, 1, 2, 2, 3, 1, 1]
```

#### Index 6: `101`

Every earlier value is smaller than `101`.

The best earlier `dp` value is `3`, from a subsequence like `[2, 5, 7]` or `[2, 3, 7]`.

So:

```text
dp[6] = 4
```

Now:

```text
dp = [1, 1, 1, 2, 2, 3, 4, 1]
```

#### Index 7: `18`

Earlier values smaller than `18`:

```text
10, 9, 2, 5, 3, 7
```

`101` is not smaller than `18`, so it cannot be used before `18`.

The best earlier extendable subsequence has length `3`, ending at `7`.

So:

```text
dp[7] = 4
```

Final:

```text
dp = [1, 1, 1, 2, 2, 3, 4, 4]
answer = 4
```

### 6. Why the `O(n^2)` DP Is Correct

The invariant is:

```text
After processing index i, dp[i] is the length of the longest increasing subsequence that ends exactly at nums[i].
```

For a fixed `i`, any increasing subsequence ending at `i` has only two possible shapes:

```text
[nums[i]]
```

or:

```text
some increasing subsequence ending at j, followed by nums[i]
```

where:

```text
j < i
nums[j] < nums[i]
```

The algorithm checks every such earlier `j`, so it considers every possible previous element of a valid subsequence ending at `i`.

For each valid `j`, `dp[j]` is already correct because `j < i` and the array is filled left to right. Therefore `dp[j] + 1` is the best length obtainable by using `j` as the previous index.

Taking the maximum over all valid `j` gives exactly the best subsequence ending at `i`.

The global longest increasing subsequence must end at some index, so taking `max(dp)` gives the full answer.

### 7. The Faster First-Principles Question

The `O(n^2)` DP is clear, but it stores one answer per final index.

There is a different way to think about the same problem:

> For each possible subsequence length, what is the smallest ending value we can achieve?

Why would a smaller ending value help?

Because smaller endings are easier to extend.

Compare two increasing subsequences of the same length:

```text
length 3 ending at 7
length 3 ending at 18
```

The one ending at `7` is at least as useful for the future, because any future value greater than `18` is also greater than `7`, and some values between `8` and `18` can extend the `7` version but not the `18` version.

So for each length, we only need to remember the smallest possible tail value.

### 8. The Patience Sorting Invariant

Maintain an array called `tails`:

```text
tails[k] = the smallest possible ending value of an increasing subsequence of length k + 1 seen so far
```

For example:

```text
tails[0] = smallest tail of a length-1 subsequence
tails[1] = smallest tail of a length-2 subsequence
tails[2] = smallest tail of a length-3 subsequence
```

Important: `tails` is not necessarily an actual subsequence from the input.

It is a compact summary of the best tail values for each length.

The invariant is:

```text
For every index k, there exists an increasing subsequence of length k + 1 ending at tails[k],
and no seen subsequence of length k + 1 has a smaller ending value.
```

Because the best tail values increase with length, `tails` is sorted in increasing order.

That sorted property allows binary search.

### 9. How Each Number Updates `tails`

When reading a new number `x`, find the first position in `tails` whose value is greater than or equal to `x`.

Call that position `pos`.

There are two cases.

#### Case 1: `pos` is at the end

`x` is larger than every tail value.

That means `x` can extend the longest subsequence found so far.

Append it:

```text
tails.append(x)
```

The known LIS length grows by one.

#### Case 2: `pos` is inside `tails`

There is already a subsequence of length `pos + 1`, but its tail is at least `x`.

Replace it:

```text
tails[pos] = x
```

This does not mean we found a longer subsequence immediately.

It means we found a better, smaller tail for a subsequence of that same length, which may help future numbers extend farther.

### 10. Why Use Greater Than or Equal?

The subsequence must be strictly increasing.

Equal values cannot extend each other.

So when `x` equals an existing tail, it should replace that tail, not create a longer subsequence.

That is why the binary search looks for:

```text
first tails[pos] >= x
```

In Python, this is `bisect_left`.

If the problem asked for a non-decreasing subsequence instead, the search rule would change, but this problem is strictly increasing.

### 11. Patience Algorithm Code

```python
from bisect import bisect_left

class Solution:
    def lengthOfLIS(self, nums: list[int]) -> int:
        tails = []

        for x in nums:
            pos = bisect_left(tails, x)

            if pos == len(tails):
                tails.append(x)
            else:
                tails[pos] = x

        return len(tails)
```

Equivalent pseudocode:

```text
tails = empty array

for x in nums:
    pos = first index where tails[pos] >= x

    if pos is past the end of tails:
        append x to tails
    else:
        replace tails[pos] with x

return length of tails
```

### 12. Patience Example Walkthrough

Use the same input:

```text
nums = [10, 9, 2, 5, 3, 7, 101, 18]
```

Start:

```text
tails = []
```

#### Read `10`

No tail exists, so append:

```text
tails = [10]
```

Meaning:

```text
best length-1 tail is 10
```

#### Read `9`

First tail greater than or equal to `9` is `10`.

Replace it:

```text
tails = [9]
```

We still only have length `1`, but ending at `9` is better than ending at `10`.

#### Read `2`

Replace `9`:

```text
tails = [2]
```

Length is still `1`, with an even better tail.

#### Read `5`

`5` is larger than every tail, so append:

```text
tails = [2, 5]
```

Now we know there is an increasing subsequence of length `2`, such as `[2, 5]`.

#### Read `3`

First tail greater than or equal to `3` is `5`.

Replace it:

```text
tails = [2, 3]
```

This does not say the input subsequence is literally `[2, 3]` for every future purpose, though in this case it is. The important fact is that length `2` can now end as low as `3`, which is better than ending at `5`.

#### Read `7`

`7` is larger than every tail, so append:

```text
tails = [2, 3, 7]
```

Now length `3` is possible.

#### Read `101`

Append:

```text
tails = [2, 3, 7, 101]
```

Now length `4` is possible.

#### Read `18`

First tail greater than or equal to `18` is `101`.

Replace it:

```text
tails = [2, 3, 7, 18]
```

Length stays `4`, but the best length-4 ending value improves from `101` to `18`.

Final answer:

```text
len(tails) = 4
```

### 13. Why the Patience Algorithm Is Correct

The invariant is:

```text
tails[k] is the smallest possible tail value of any increasing subsequence
of length k + 1 found in the processed prefix.
```

When processing a new value `x`, binary search finds the first `pos` where:

```text
tails[pos] >= x
```

All earlier tails are smaller than `x`:

```text
tails[0], tails[1], ..., tails[pos - 1] < x
```

So if `pos > 0`, `x` can extend a subsequence of length `pos` into a subsequence of length `pos + 1`.

Replacing `tails[pos]` with `x` is safe because `x` is no larger than the previous tail at that length. A smaller or equal tail is always at least as good for extending future subsequences.

The replacement does not destroy the fact that a subsequence of that length exists, because the new number `x` itself forms one by extending the shorter tail before it.

If `pos == len(tails)`, then `x` is larger than the tail of the longest known subsequence, so it extends that subsequence and increases the maximum length by one.

Because `tails` records exactly one best tail per length, its length is the largest subsequence length that has been proven possible.

No longer subsequence can exist without creating another tail entry: when the algorithm sees the final element of any increasing subsequence of length `L`, that element must be able to extend a length `L - 1` tail and therefore produce or maintain an entry at index `L - 1`.

Therefore, after all numbers are processed, `len(tails)` is exactly the length of the longest increasing subsequence.

### 14. Complexity

For the bottom-up DP version:

```text
Time:  O(n^2)
Space: O(n)
```

There are `n` states, and each state scans all earlier states.

For the patience sorting / binary search version:

```text
Time:  O(n log n)
Space: O(n)
```

Each number performs one binary search over `tails`, whose length is at most `n`.

### 15. Common Pitfalls

#### Confusing Subsequence With Subarray

The chosen numbers do not need to be adjacent.

For:

```text
[10, 9, 2, 5, 3, 7, 101, 18]
```

`[2, 3, 7, 18]` is valid even though the numbers are separated in the original array.

#### Allowing Equal Values

The problem says increasing, not non-decreasing.

So this is invalid:

```text
[7, 7]
```

That is why Example 3 returns `1`, not `6`.

#### Returning `dp[-1]` in the `O(n^2)` DP

The best subsequence does not have to end at the final element.

Return:

```python
max(dp)
```

not:

```python
dp[-1]
```

#### Using the Wrong Binary Search

For the strict LIS version, use the first index where:

```text
tails[pos] >= x
```

In Python, that is:

```python
bisect_left(tails, x)
```

Using `bisect_right` would allow equal values to increase the length, which solves a different problem.

#### Thinking `tails` Is the Actual Answer Sequence

Sometimes `tails` happens to be a valid subsequence, but the algorithm only guarantees tail values, not reconstruction.

For this problem, that is fine because only the length is required.

### 16. First-Principles Summary

The problem begins as an exponential take-or-skip search over subsequences.

The `O(n^2)` DP works by fixing the last chosen element:

```text
dp[i] = best increasing subsequence ending at i
```

That turns the problem into checking which earlier elements can legally come before `nums[i]`.

The faster `O(n log n)` algorithm compresses the same idea further:

```text
For each length, keep the smallest possible tail value.
```

A smaller tail is better because it leaves more room for future numbers to extend the subsequence.

So the whole optimized algorithm is:

> Scan left to right, place each number into the first tail position where it can improve the ending value, append only when it extends the longest known subsequence, and return the number of tail positions.

## Implementation

See `solutions/dynamic_programming_1d/p300_longest_increasing_subsequence.py`.

## Tests

See `tests/dynamic_programming_1d/test_p300_longest_increasing_subsequence.py`.

## Examples

### Example 1
- Input: `{'nums': [10, 9, 2, 5, 3, 7, 101, 18]}`
- Output: `4`

### Example 2
- Input: `{'nums': [0, 1, 0, 3, 2, 3]}`
- Output: `4`

### Example 3
- Input: `{'nums': [7, 7, 7, 7, 7, 7, 7]}`
- Output: `1`
