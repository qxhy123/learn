# 55. Jump Game

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/jump-game/
- Official Group: Array / String
- Pattern Group: Array / String
- Patterns: greedy, reachability

## First-Principles Explanation

### What The Problem Is Asking

You are given an array `nums`.

If you stand on index `i`, then `nums[i]` tells you the maximum jump length from that position.

That means from index `i` you may move to any index:

```text
i + 1, i + 2, ..., i + nums[i]
```

as long as that new index stays inside the array.

The question is:

> Starting at index `0`, is there at least one sequence of jumps that reaches the last index?

This is a yes/no reachability problem.

We are not asked:

- for the minimum number of jumps,
- for the actual path,
- or for how many ways there are to finish.

We only need to know whether the last index is reachable at all.

That changes what information matters.

---

### Start From The Most Direct Baseline

The most literal way to think about the problem is:

At each index, try every jump length that is allowed there.

For example, if:

```text
nums[i] = 3
```

then you could try:

```text
i + 1
i + 2
i + 3
```

and recursively ask whether any of those positions can eventually reach the end.

Conceptually:

```python
def can_reach(i):
    if i >= last_index:
        return True

    for step in range(1, nums[i] + 1):
        if can_reach(i + step):
            return True

    return False
```

This matches the problem statement exactly, so it is a good baseline for understanding.

But it is inefficient.

Why?

Because many different jump sequences can land on the same index, and then the recursion repeats the same work.

So the plain brute-force search can become exponential.

---

### A Better Baseline: Mark Reachable Positions

If we want something simpler than brute force but still very direct, we can think in terms of reachability:

1. Mark index `0` as reachable.
2. For every reachable index `i`, mark all positions up to `i + nums[i]` as reachable.
3. At the end, check whether the last index was ever marked reachable.

That gives an `O(n^2)` style baseline in the worst case, because from one index we may try to mark many later indices.

Conceptually:

```python
reachable = [False] * n
reachable[0] = True

for i in range(n):
    if not reachable[i]:
        continue

    for j in range(i + 1, min(n, i + nums[i] + 1)):
        reachable[j] = True

return reachable[n - 1]
```

This is already much better conceptually:

- we no longer explore the same suffix over and over,
- and we focus on which positions can be reached.

But it is still doing more work than necessary.

The key question becomes:

> Do we really need to remember every reachable index separately?

---

### Key Observation: Reachability Forms A Continuous Prefix

Suppose you already know that index `i` is reachable.

Then from there you may jump to any index up to:

```text
i + nums[i]
```

Now imagine that while scanning from left to right, you have already established that every index from `0` through some boundary `farthest` is reachable.

That means the set of reachable indices so far is not scattered randomly.
It is a continuous prefix:

```text
[0, 1, 2, ..., farthest]
```

Why is that enough?

Because if you can stand on every index in that prefix, then any future progress must come from one of those indices.

So instead of remembering:

```text
which exact indices are reachable
```

we only need to remember:

```text
how far the reachable prefix extends
```

That single number is the entire state we need.

---

### The Invariant

Maintain:

```text
farthest = the greatest index reachable using jumps from reachable positions
           among the indices we have already processed
```

More concretely, after processing indices from `0` through `i`, the invariant is:

```text
Every index <= farthest is reachable.
No information from indices > i is needed yet.
```

This leads to two critical facts:

1. If we are currently at index `i` and:

```text
i > farthest
```

then index `i` itself is unreachable.
If we cannot even stand on `i`, we cannot use `nums[i]` to extend anything.
So the process is stuck, and the answer is `False`.

2. If index `i` is reachable, then jumping from it can extend the reachable prefix to:

```text
i + nums[i]
```

So we update:

```text
farthest = max(farthest, i + nums[i])
```

This is the whole greedy idea.

We do not choose a detailed path.
We only keep the best reach discovered so far.

---

### Why This Greedy View Is Safe

At first glance, greedy solutions often feel suspicious because they seem to commit too early.

But here, we are not committing to one jump path.

When we store:

```text
farthest
```

we are summarizing all jump paths discovered so far.

If one reachable index can get us to `7` and another can get us to `10`, then for a yes/no question there is no reason to remember both separately.

Keeping only:

```text
max(7, 10) = 10
```

loses nothing important, because any index up to `10` is at least as good for future reachability as any smaller boundary.

That is why the greedy compression is valid.

---

### Detailed Algorithm

Let:

```text
n = len(nums)
last = n - 1
```

Algorithm:

1. If the array has one element, we are already at the last index, so return `True`.
2. Initialize:

```text
farthest = 0
```

3. Scan indices from left to right.
4. For each index `i`:

If:

```text
i > farthest
```

return `False`, because we have reached a gap we cannot cross.

Otherwise index `i` is reachable, so update:

```text
farthest = max(farthest, i + nums[i])
```

5. If at any moment:

```text
farthest >= last
```

return `True`, because the last index is reachable.

6. If the loop finishes without getting stuck, return `True`.

In practice, step 5 often lets us stop early.

---

### Pseudocode

```python
def canJump(nums):
    last = len(nums) - 1
    farthest = 0

    for i in range(len(nums)):
        if i > farthest:
            return False

        farthest = max(farthest, i + nums[i])

        if farthest >= last:
            return True

    return True
```

Equivalent compact version:

```python
def canJump(nums):
    farthest = 0

    for i, jump in enumerate(nums):
        if i > farthest:
            return False
        farthest = max(farthest, i + jump)

    return True
```

Both rely on exactly the same invariant.

---

### Walkthrough: `nums = [2, 3, 1, 1, 4]`

Last index is `4`.

Start:

```text
farthest = 0
```

#### Index 0

We are at:

```text
i = 0
```

Since:

```text
0 <= farthest
```

index `0` is reachable.

From index `0`, we can jump at most `2` steps, so:

```text
i + nums[i] = 0 + 2 = 2
```

Update:

```text
farthest = max(0, 2) = 2
```

Now we know every index up to `2` is reachable.

#### Index 1

Now:

```text
i = 1
```

Since:

```text
1 <= farthest
```

index `1` is reachable.

From here:

```text
1 + nums[1] = 1 + 3 = 4
```

Update:

```text
farthest = max(2, 4) = 4
```

Now:

```text
farthest = 4
```

which is the last index.

So we can already return:

```text
True
```

Notice what happened:

- We did not explicitly choose the exact jump sequence.
- We simply observed that some reachable index can reach the end.

That is enough.

---

### Walkthrough: `nums = [3, 2, 1, 0, 4]`

Last index is `4`.

Start:

```text
farthest = 0
```

#### Index 0

Reach:

```text
0 + 3 = 3
```

Update:

```text
farthest = 3
```

So indices `0, 1, 2, 3` are all reachable.

#### Index 1

Reach from here:

```text
1 + 2 = 3
```

Update:

```text
farthest = max(3, 3) = 3
```

No improvement.

#### Index 2

Reach from here:

```text
2 + 1 = 3
```

Update:

```text
farthest = 3
```

Still no improvement.

#### Index 3

Reach from here:

```text
3 + 0 = 3
```

Update:

```text
farthest = 3
```

Still stuck.

#### Index 4

Now:

```text
i = 4
```

But:

```text
4 > farthest = 3
```

So index `4` is unreachable.

That means there is a gap after index `3` that no earlier reachable position can cross.

Return:

```text
False
```

This example shows why a `0` can be dangerous:

- a zero is not automatically bad,
- but if the reachable prefix ends on that zero and no earlier index can jump beyond it, progress stops.

---

### Correctness

We prove the algorithm is correct using the invariant.

#### Claim 1

Before processing index `i`, if `i <= farthest`, then index `i` is reachable.

Reason:

`farthest` is defined as the farthest index reachable from the already processed reachable positions.
So every index up to that boundary lies inside the reachable prefix.

#### Claim 2

When processing a reachable index `i`, updating

```text
farthest = max(farthest, i + nums[i])
```

preserves the invariant.

Reason:

If index `i` is reachable, then every jump destination from `i` up to `i + nums[i]` is also reachable.
Taking the maximum with the old boundary extends the reachable prefix exactly as far as any processed reachable index allows.

#### Claim 3

If the algorithm encounters an index `i` with:

```text
i > farthest
```

then the last index is unreachable.

Reason:

All progress must come from previously reachable indices.
But `farthest` already records the greatest index any such position can reach.
If `i` is beyond that boundary, then `i` and everything to its right are unreachable from the processed prefix.
So no valid jump sequence can continue through this gap.

#### Claim 4

If the algorithm returns `True` because:

```text
farthest >= last
```

then the last index is reachable.

Reason:

By definition, `farthest` is an actually reachable boundary.
So if it reaches or passes the last index, the last index is reachable.

Combining these claims:

- the algorithm never reports `True` unless the end is reachable,
- and it never reports `False` unless progress is impossible.

Therefore the algorithm is correct.

---

### Complexity

For an array of length `n`:

- Time: `O(n)`
- Extra space: `O(1)`

Why `O(n)` time?

Because each index is processed at most once, and each step does only constant work.

Why `O(1)` space?

Because we store only a few variables such as:

- `farthest`
- `i`
- `last`

No extra array or recursion stack is needed.

---

### Common Pitfalls

#### 1. Confusing This With Jump Game II

This problem asks:

```text
Can we reach the end?
```

It does not ask:

```text
What is the minimum number of jumps?
```

That is a different problem with different state.

#### 2. Thinking You Must Choose The Best Next Jump Explicitly

You do not need to decide:

```text
"Jump to index 1" or "jump to index 2"
```

The algorithm never commits to a single path.
It only tracks the best reachable boundary produced by all reachable positions seen so far.

#### 3. Forgetting The Unreachable-Index Check

This line is essential:

```python
if i > farthest:
    return False
```

Without it, the code might incorrectly use `nums[i]` from an index you cannot actually stand on.

#### 4. Mishandling A Single-Element Array

If `nums` has length `1`, you start on the last index already.

The answer is:

```text
True
```

The greedy loop naturally handles this, but it is worth keeping in mind.

#### 5. Treating Every Zero As Failure

A zero only causes failure if the reachable prefix gets trapped there.

For example:

```text
[2, 0, 0]
```

is still solvable, because from index `0` you can jump directly to the end.

---

### First-Principles Summary

The problem looks like it might require exploring many jump paths, but the yes/no nature of the question lets us compress the entire search into one number.

The first-principles insight is:

> After processing a prefix of the array, the only thing that matters is the farthest index reachable from that prefix.

Once you know that boundary:

- any index at or before it is usable,
- any index beyond it is currently impossible,
- and every reachable index can only try to push that boundary farther right.

So the solution is:

1. scan left to right,
2. stop immediately if you reach an index outside the reachable boundary,
3. otherwise extend the boundary with `i + nums[i]`,
4. succeed as soon as that boundary reaches the last index.

That is why the greedy algorithm is both simple and sufficient.

## Implementation

See `solutions/array_string/p055_jump_game.py`.

## Tests

See `tests/array_string/test_p055_jump_game.py`.

## Examples

### Example 1
- Input: `{'nums': [2, 3, 1, 1, 4]}`
- Output: `True`

### Example 2
- Input: `{'nums': [3, 2, 1, 0, 4]}`
- Output: `False`

## Follow-up Practice

- Write the exponential recursive version first, then identify exactly what repeated information the greedy solution compresses.
- Trace `farthest` after every index on both solvable and unsolvable arrays.
- Compare this problem with Jump Game II and state clearly why "reachability" and "minimum jumps" require different stored information.
