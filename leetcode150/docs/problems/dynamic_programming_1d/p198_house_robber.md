# 198. House Robber

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/house-robber/
- Official Group: 1D DP
- Pattern Group: Dynamic Programming 1D
- Patterns: dynamic-programming-1d

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given an array:

```text
nums[i] = amount of money in house i
```

The houses are arranged in a straight line, and you may rob any subset of them with one rule:

```text
You cannot rob two adjacent houses.
```

Your goal is to return the maximum total money you can rob.

For example:

```text
nums = [1, 2, 3, 1]
```

You cannot take both `1` at index `0` and `2` at index `1`, because they are adjacent.

You also cannot take both `2` at index `1` and `3` at index `2`.

Valid choices include:

```text
house 0 + house 2 = 1 + 3 = 4
house 0 + house 3 = 1 + 1 = 2
house 1 + house 3 = 2 + 1 = 3
house 2           = 3
```

The best valid choice is:

```text
1 + 3 = 4
```

So the answer is `4`.

The real problem is:

> Among all subsets of house indices with no adjacent indices, find the maximum sum of their values.

---

### 2. Start From the Brute-Force Recursion

At each house, there are only two possible decisions:

```text
rob this house
skip this house
```

Suppose we are deciding what to do starting at index `i`.

If we rob house `i`, then we are not allowed to rob house `i + 1`, so the next available house is `i + 2`:

```text
nums[i] + best_from(i + 2)
```

If we skip house `i`, then the next available house is `i + 1`:

```text
best_from(i + 1)
```

So a direct recursive definition is:

```python
def best_from(i):
    if i >= len(nums):
        return 0

    rob_current = nums[i] + best_from(i + 2)
    skip_current = best_from(i + 1)
    return max(rob_current, skip_current)
```

This is a correct description of the decision tree.

For `nums = [2, 7, 9, 3, 1]`, the top of the tree is:

```text
best_from(0)
= max(
    rob house 0  -> 2 + best_from(2),
    skip house 0 -> best_from(1)
  )
```

But this recursion repeats work. For example, `best_from(2)` can be reached by:

```text
rob house 0
skip house 0, then skip house 1
```

Both paths ask the same question:

```text
What is the best answer using houses from index 2 onward?
```

The brute-force recursion may explore exponentially many decision paths, even though there are only `n` distinct starting indices.

That repeated work is the reason dynamic programming fits this problem.

---

### 3. The Key Observation

The adjacency rule is local.

When deciding whether to include house `i`, the only house that directly conflicts with it is house `i - 1` or house `i + 1`.

This means the best answer for a prefix of houses can be built from smaller prefixes.

Consider the first `i + 1` houses:

```text
nums[0 ... i]
```

In any optimal valid robbery plan for this prefix, exactly one of these two high-level cases is true:

#### Case A: Do not rob house `i`

Then the best total is simply the best total from the previous prefix:

```text
best using nums[0 ... i - 1]
```

#### Case B: Rob house `i`

Then house `i - 1` cannot be robbed.

So the best compatible total is:

```text
best using nums[0 ... i - 2] + nums[i]
```

There is no third case. Every valid plan either uses house `i` or does not use house `i`.

Therefore:

```text
best through i = max(
    best through i - 1,
    best through i - 2 + nums[i]
)
```

This is the entire problem.

---

### 4. DP State and Invariant

Define:

```text
dp[i] = maximum money that can be robbed from houses nums[0 ... i]
        without robbing adjacent houses
```

This definition is important because it is not vague. It says:

- which part of the input is being considered: `nums[0 ... i]`
- what restriction still applies: no adjacent robbed houses
- what value is stored: the maximum money possible

The invariant is:

```text
After computing dp[i], dp[i] is the optimal answer for the prefix ending at house i.
```

For each `i`, the transition follows from the final decision about house `i`:

```text
dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
```

Interpretation:

```text
dp[i - 1]          = skip house i
dp[i - 2] + nums[i] = rob house i
```

Base cases:

```text
dp[0] = nums[0]
```

With only one house, the best plan is to rob it.

For two houses:

```text
dp[1] = max(nums[0], nums[1])
```

You cannot rob both adjacent houses, so choose the larger one.

A slightly cleaner implementation avoids special cases by keeping only two rolling values:

```text
prev2 = best answer for the prefix ending two houses before the current house
prev1 = best answer for the prefix ending one house before the current house
```

When reading a new house value `money`:

```text
current = max(prev1, prev2 + money)
```

Then shift the window of DP states:

```text
prev2 = prev1
prev1 = current
```

The rolling-state invariant is:

```text
Before processing the next house:
prev1 is the best total for all houses already processed.
prev2 is the best total for all houses processed except the most recent one.
```

---

### 5. Detailed Algorithm

Use the space-optimized DP version.

1. Start with no houses processed:

   ```text
   prev2 = 0
   prev1 = 0
   ```

   With no houses, the best total is `0`.

2. Scan houses from left to right.

3. For the current house with value `money`, compute the best total after considering it:

   ```text
   rob_current  = prev2 + money
   skip_current = prev1
   current = max(skip_current, rob_current)
   ```

4. Move the rolling states forward:

   ```text
   prev2 = prev1
   prev1 = current
   ```

5. After all houses are processed, return `prev1`.

Why left to right?

Because the answer for the current house depends only on two earlier answers:

```text
i - 1 and i - 2
```

Those answers are already known if we scan from left to right.

---

### 6. Walkthrough: Example 1

Input:

```text
nums = [1, 2, 3, 1]
```

Start:

```text
prev2 = 0
prev1 = 0
```

No houses have been processed yet.

#### House 0: money = 1

Options:

```text
skip = prev1 = 0
rob  = prev2 + 1 = 0 + 1 = 1
```

So:

```text
current = max(0, 1) = 1
```

Shift:

```text
prev2 = 0
prev1 = 1
```

Best after `[1]` is `1`.

#### House 1: money = 2

Options:

```text
skip = prev1 = 1
rob  = prev2 + 2 = 0 + 2 = 2
```

So:

```text
current = max(1, 2) = 2
```

Shift:

```text
prev2 = 1
prev1 = 2
```

Best after `[1, 2]` is `2`.

You choose house `1`, not both houses.

#### House 2: money = 3

Options:

```text
skip = prev1 = 2
rob  = prev2 + 3 = 1 + 3 = 4
```

So:

```text
current = max(2, 4) = 4
```

Shift:

```text
prev2 = 2
prev1 = 4
```

Best after `[1, 2, 3]` is `4`.

This corresponds to robbing houses `0` and `2`.

#### House 3: money = 1

Options:

```text
skip = prev1 = 4
rob  = prev2 + 1 = 2 + 1 = 3
```

So:

```text
current = max(4, 3) = 4
```

Shift:

```text
prev2 = 4
prev1 = 4
```

Final answer:

```text
4
```

---

### 7. Walkthrough: Example 2

Input:

```text
nums = [2, 7, 9, 3, 1]
```

Track each step:

```text
Before start: prev2 = 0, prev1 = 0
```

| House | Money | Rob Current (`prev2 + money`) | Skip Current (`prev1`) | Current Best |
|---:|---:|---:|---:|---:|
| 0 | 2 | 0 + 2 = 2 | 0 | 2 |
| 1 | 7 | 0 + 7 = 7 | 2 | 7 |
| 2 | 9 | 2 + 9 = 11 | 7 | 11 |
| 3 | 3 | 7 + 3 = 10 | 11 | 11 |
| 4 | 1 | 11 + 1 = 12 | 11 | 12 |

Final answer:

```text
12
```

One optimal choice is:

```text
house 0 + house 2 + house 4 = 2 + 9 + 1 = 12
```

Notice what happens at house `3`:

```text
robbing it gives 10
skipping it keeps 11
```

So the best plan deliberately skips house `3`.

This is why a greedy rule like "always take a large-looking house" is not reliable. The decision must account for the best compatible prefix, not just the current local value.

---

### 8. Code

Space-optimized Python implementation:

```python
from typing import List


class Solution:
    def rob(self, nums: List[int]) -> int:
        prev2 = 0
        prev1 = 0

        for money in nums:
            current = max(prev1, prev2 + money)
            prev2 = prev1
            prev1 = current

        return prev1
```

Equivalent tabulation version:

```python
def rob(nums):
    if not nums:
        return 0

    if len(nums) == 1:
        return nums[0]

    dp = [0] * len(nums)
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])

    for i in range(2, len(nums)):
        dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])

    return dp[-1]
```

Both versions use the same recurrence. The rolling version is just the tabulation version with the unused older `dp` entries removed.

---

### 9. Correctness

We prove the algorithm returns the maximum money that can be robbed without robbing adjacent houses.

#### Lemma 1: The recurrence considers every possible optimal plan for a prefix.

For any prefix `nums[0 ... i]`, every valid robbery plan falls into exactly one of two cases:

```text
it does not rob house i
it does rob house i
```

If it does not rob house `i`, then its value is at most the optimal value for `nums[0 ... i - 1]`, which is `dp[i - 1]`.

If it does rob house `i`, then it cannot rob house `i - 1`, so the rest of the plan must come from `nums[0 ... i - 2]`. Its value is at most `dp[i - 2] + nums[i]`.

Therefore the best valid value for `nums[0 ... i]` must be:

```text
max(dp[i - 1], dp[i - 2] + nums[i])
```

#### Lemma 2: Each computed state is optimal for its prefix.

Base cases:

- With zero processed houses, the best total is `0`.
- With one processed house, the rolling update computes `max(0, nums[0])`, which is `nums[0]` under the problem constraints.

Inductive step:

Assume the rolling values before processing house `i` correctly represent:

```text
prev1 = best answer through house i - 1
prev2 = best answer through house i - 2
```

The algorithm computes:

```text
current = max(prev1, prev2 + nums[i])
```

By Lemma 1, this is exactly the optimal answer through house `i`.

Then it shifts:

```text
prev2 = prev1
prev1 = current
```

So the invariant is preserved for the next house.

#### Theorem: The returned value is correct.

After the loop has processed every house, `prev1` is the optimal answer for the prefix containing all houses. That prefix is the whole input array, so returning `prev1` gives the required maximum amount of money.

---

### 10. Complexity

Let `n = len(nums)`.

Time complexity:

```text
O(n)
```

Each house is processed once, and each step does constant work.

Space complexity:

```text
O(1)
```

The optimized version stores only `prev2`, `prev1`, and `current`, not the full DP array.

The tabulation version uses `O(n)` space, but the recurrence only needs the previous two states, so the extra array is not necessary unless you want to inspect or reconstruct choices.

---

### 11. Common Pitfalls

#### Pitfall 1: Greedy selection

A tempting strategy is:

```text
At each step, rob the larger of nearby houses.
```

This fails because a local choice can block a better combination later.

For example:

```text
nums = [2, 7, 9, 3, 1]
```

Choosing `7` because it is larger than `2` misses the better combination:

```text
2 + 9 + 1 = 12
```

#### Pitfall 2: Defining the state as "best if we rob house i" but using the wrong transition

If `dp[i]` means "best total ending by robbing house `i`", then the transition is not simply:

```text
dp[i] = dp[i - 2] + nums[i]
```

because the best previous robbed house could be farther back than `i - 2`.

The prefix definition is simpler:

```text
dp[i] = best total using houses 0 through i
```

Then the transition needs only `dp[i - 1]` and `dp[i - 2]`.

#### Pitfall 3: Off-by-one base cases

For a full `dp` array, remember:

```text
dp[0] = nums[0]
dp[1] = max(nums[0], nums[1])
```

Trying to compute `dp[1]` with `dp[-1]` accidentally may work in some languages poorly or produce the wrong meaning in Python.

The rolling version avoids most of this by starting from:

```text
prev2 = 0
prev1 = 0
```

#### Pitfall 4: Updating rolling variables in the wrong order

This is wrong:

```python
prev1 = current
prev2 = prev1
```

After the first line, the old `prev1` is lost.

Use:

```python
prev2 = prev1
prev1 = current
```

or tuple assignment:

```python
prev2, prev1 = prev1, current
```

#### Pitfall 5: Forgetting the empty input shape

LeetCode's constraints normally provide at least one house, but the rolling implementation naturally returns `0` for an empty list. This makes the code robust without extra branching.

---

### 12. First-Principles Summary

The problem is not about simulating a robber. It is about choosing a maximum-sum subset with one structural restriction:

```text
chosen indices cannot be adjacent
```

At any house, the final decision is binary:

```text
skip it, keeping the best answer so far
rob it, adding its money to the best answer from two houses back
```

That gives the recurrence:

```text
best_through_current = max(
    best_through_previous,
    best_through_two_back + current_house_money
)
```

The only information needed from the past is the best answer through the previous house and the best answer through two houses back. Therefore the full decision tree collapses into a one-pass dynamic program with constant space.

## Implementation
See `solutions/dynamic_programming_1d/p198_house_robber.py`.

## Tests
See `tests/dynamic_programming_1d/test_p198_house_robber.py`.

## Examples

### Example 1
- Input: `{'nums': [1, 2, 3, 1]}`
- Output: `4`

### Example 2
- Input: `{'nums': [2, 7, 9, 3, 1]}`
- Output: `12`

## Follow-up Practice
- Write the brute-force recursion before writing the DP transition.
- Explain why the two final cases are "rob current" and "skip current".
- Implement both the full `dp` array version and the `O(1)` rolling-state version.
- Test small arrays by hand: one house, two houses, all equal values, and values where skipping a large-looking local choice is necessary.
