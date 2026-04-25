# 77. Combinations

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/combinations/
- Official Group: Backtracking
- Pattern Group: Backtracking
- Patterns: backtracking, combinations

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given two integers:

```text
n = the largest available number
k = how many numbers to choose
```

Return every possible way to choose exactly `k` distinct numbers from:

```text
1, 2, 3, ..., n
```

The order inside a choice does not matter.

That means:

```text
[1, 2]
```

and:

```text
[2, 1]
```

represent the same combination. We should output it only once.

For example, if:

```text
n = 4
k = 2
```

we are choosing 2 numbers from:

```text
1, 2, 3, 4
```

The valid combinations are:

```text
[1, 2]
[1, 3]
[1, 4]
[2, 3]
[2, 4]
[3, 4]
```

The real problem is:

> Generate each size-`k` subset of `{1, 2, ..., n}` exactly once.

---

### 2. Why This Is Not a Permutation Problem

If the problem asked for ordered arrangements, then `[1, 2]` and `[2, 1]` would be different results.

But combinations ignore order.

So the main danger is accidentally generating duplicate descriptions of the same set.

For `n = 4`, `k = 2`, a naive ordered search might generate:

```text
[1, 2]
[2, 1]
[1, 3]
[3, 1]
[1, 4]
[4, 1]
...
```

That search is doing extra work because it lets each number appear before and after every other number.

To solve combinations cleanly, we need a representation rule that gives each set exactly one order.

The natural rule is:

```text
Always build combinations in increasing order.
```

Then `{1, 2}` can only appear as:

```text
[1, 2]
```

It can never appear as `[2, 1]`, because once we choose `2`, we are not allowed to go backward and choose `1`.

---

### 3. Start From the Brute Force Idea

The most direct baseline is to look at every subset of `{1, ..., n}` and keep the subsets whose size is `k`.

For each number, there are two choices:

```text
take it
skip it
```

So the decision tree has `2^n` leaves.

Conceptually:

```python
answers = []

def dfs(number, chosen):
    if number == n + 1:
        if len(chosen) == k:
            answers.append(chosen.copy())
        return

    chosen.append(number)
    dfs(number + 1, chosen)
    chosen.pop()

    dfs(number + 1, chosen)
```

This is correct because it considers every subset.

But it also explores many branches that cannot possibly become valid.

For example, if `k = 2` and we already chose 4 numbers, there is no reason to continue.

Or if we still need 3 numbers but only 2 numbers remain, there is no reason to continue.

We can do better by only building partial combinations that are still plausible.

---

### 4. Key Observation: A Combination Has a Next Smallest Candidate

Suppose the partial combination is:

```text
[1, 3]
```

The next number cannot be `1`, `2`, or `3`.

- `1` and `3` are already used.
- `2` would make `[1, 3, 2]`, which describes the same set as `[1, 2, 3]` but in the wrong order.

So the next number must be greater than `3`.

This gives us the core recursive state:

```text
path  = numbers chosen so far, always increasing
start = smallest number that may still be chosen next
```

If the current path is:

```text
[1, 3]
```

then:

```text
start = 4
```

The next recursive call only considers:

```text
4, 5, ..., n
```

This single `start` value prevents duplicates.

---

### 5. Recursion State and Invariant

At every recursive call, maintain this invariant:

```text
path is an increasing list of chosen numbers,
and every future number must be at least start.
```

More concretely:

- `path` contains no duplicates.
- `path` is sorted in strictly increasing order.
- all numbers in `path` are between `1` and `n`.
- if we append a number `x`, then `x >= start`.
- after appending `x`, the next call uses `start = x + 1`.

Why this matters:

```text
Increasing order is not just cosmetic.
It is the rule that makes each combination appear exactly once.
```

The base case is also simple:

```text
if len(path) == k:
    record path
```

At that moment the path is a complete size-`k` combination, and the invariant guarantees it is valid.

---

### 6. Choosing the Next Number

From a state `(start, path)`, try each possible next number:

```text
start, start + 1, start + 2, ..., n
```

For each candidate `value`:

1. Append `value` to `path`.
2. Recurse with `value + 1` as the new `start`.
3. Remove `value` from `path` so the next candidate can be tried.

The removal step is essential.

The same `path` list is reused across sibling branches of the recursion tree. If we do not undo the append, then choices from one branch leak into the next branch.

This is the meaning of backtracking here:

```text
make a choice
explore everything under that choice
undo the choice
try the next choice
```

---

### 7. Pruning: Stop When There Are Not Enough Numbers Left

The basic loop:

```python
for value in range(start, n + 1):
```

is already correct.

But we can avoid calls that cannot finish.

Suppose:

```text
n = 5
k = 3
path = [2]
```

We still need:

```text
k - len(path) = 2 numbers
```

If we choose `5` next, then the path becomes `[2, 5]`, but no larger number remains to supply the third number. That branch is impossible.

In general, before choosing the next number, compute:

```text
remaining_slots = k - len(path)
```

If we choose `value`, then the numbers available including `value` are:

```text
value, value + 1, ..., n
```

There are:

```text
n - value + 1
```

such numbers.

We need at least `remaining_slots` numbers, so:

```text
n - value + 1 >= remaining_slots
```

Rearrange:

```text
value <= n - remaining_slots + 1
```

So the loop only needs to go up to:

```text
limit = n - remaining_slots + 1
```

This pruning does not change the answer. It only skips starting choices that cannot possibly lead to a complete combination.

---

### 8. Detailed Algorithm

1. Create an empty result list `answers`.
2. Create an empty current combination `path`.
3. Define a recursive function `backtrack(start)`.
4. If `len(path) == k`, append a copy of `path` to `answers` and return.
5. Compute how many numbers are still needed:

```text
remaining_slots = k - len(path)
```

6. Compute the largest useful next choice:

```text
limit = n - remaining_slots + 1
```

7. For each `value` from `start` through `limit`:
   - append `value` to `path`.
   - recurse with `backtrack(value + 1)`.
   - pop `value` from `path`.
8. Start the recursion with `backtrack(1)`.
9. Return `answers`.

---

### 9. Code

```python
from typing import List


class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        answers: List[List[int]] = []
        path: List[int] = []

        def backtrack(start: int) -> None:
            if len(path) == k:
                answers.append(path.copy())
                return

            remaining_slots = k - len(path)
            limit = n - remaining_slots + 1

            for value in range(start, limit + 1):
                path.append(value)
                backtrack(value + 1)
                path.pop()

        backtrack(1)
        return answers
```

The important details are:

- `path.copy()` records the current contents, not the mutable list object that will later change.
- `backtrack(value + 1)` enforces increasing order.
- `path.pop()` restores the state for the next loop iteration.
- `limit` avoids branches that do not have enough remaining numbers.

---

### 10. Example Walkthrough: `n = 4`, `k = 2`

Start with:

```text
path = []
start = 1
```

We need 2 numbers. The first value can be at most `3`, because starting with `4` would leave no second number.

So the first choices are:

```text
1, 2, 3
```

#### Choose `1`

```text
path = [1]
start = 2
```

We still need 1 number, so try:

```text
2, 3, 4
```

Choose `2`:

```text
path = [1, 2]
```

Length is `k`, so record:

```text
[1, 2]
```

Backtrack to:

```text
path = [1]
```

Choose `3`:

```text
path = [1, 3]
```

Record:

```text
[1, 3]
```

Backtrack again, choose `4`, and record:

```text
[1, 4]
```

All combinations starting with `1` are complete.

#### Choose `2`

Return to the root and choose `2` as the first number:

```text
path = [2]
start = 3
```

The second number can be:

```text
3, 4
```

Record:

```text
[2, 3]
[2, 4]
```

#### Choose `3`

Return to the root and choose `3` as the first number:

```text
path = [3]
start = 4
```

The only possible second number is `4`, so record:

```text
[3, 4]
```

The final result is:

```text
[[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
```

Notice what never happens:

```text
[2, 1]
[3, 1]
[4, 1]
```

Those are not missing. They are intentionally excluded because they are duplicate orderings of combinations already represented in increasing order.

---

### 11. Correctness

We prove that the algorithm returns exactly all size-`k` combinations from `{1, ..., n}`.

#### Lemma 1: Every recorded path is a valid combination.

The algorithm records a path only when `len(path) == k`.

The recursion starts with `start = 1`. Whenever it appends `value`, the next recursive call uses `value + 1`, so all later appended numbers must be larger than `value`.

Therefore every recorded path contains `k` distinct numbers in strictly increasing order, all within `1` through `n`. So every recorded path is a valid size-`k` combination.

#### Lemma 2: No valid combination is recorded more than once.

Every path is built in strictly increasing order.

A given combination has exactly one strictly increasing ordering.

For example, the set `{1, 3, 4}` can only be represented by:

```text
[1, 3, 4]
```

It cannot also be generated as `[3, 1, 4]` or `[4, 3, 1]`, because the recursion never chooses a smaller number after a larger number.

So no combination can be recorded more than once.

#### Lemma 3: Every valid combination is eventually recorded.

Take any valid combination:

```text
c1 < c2 < ... < ck
```

At the root, the loop considers `c1` unless pruning would exclude it. Pruning excludes only values for which there are fewer than `k` numbers available from that value through `n`; but a valid combination starting with `c1` proves enough numbers exist.

After choosing `c1`, the recursive call starts at `c1 + 1`, so the loop can consider `c2`. The same argument repeats for `c3` through `ck`.

Thus the recursion has a branch that chooses exactly:

```text
c1, c2, ..., ck
```

When the path reaches length `k`, the algorithm records it.

#### Theorem

By Lemma 1, everything recorded is valid. By Lemma 2, nothing is recorded twice. By Lemma 3, every valid combination is recorded. Therefore the algorithm returns exactly all size-`k` combinations from `{1, ..., n}`.

---

### 12. Complexity

There are exactly:

```text
C(n, k)
```

valid combinations.

Each output combination has length `k`, and copying it into the answer costs `O(k)`.

So the output cost alone is:

```text
O(C(n, k) * k)
```

The backtracking work is proportional to the number of useful partial combinations explored. With the pruning shown above, the dominant cost is still the size of the generated output.

Therefore:

- Time: `O(C(n, k) * k)` for generating and copying all combinations.
- Auxiliary space: `O(k)` for the recursion path, excluding the returned output.
- Output space: `O(C(n, k) * k)` for the result list.

The exponential-looking nature of the problem is unavoidable because the output itself can be very large.

---

### 13. Common Pitfalls

#### Appending `path` Instead of a Copy

This is wrong:

```python
answers.append(path)
```

`path` is mutated after recording, so every saved result would point to the same list object.

Use:

```python
answers.append(path.copy())
```

#### Recursing With the Wrong Next Start

This is wrong for combinations:

```python
backtrack(start + 1)
```

The next start depends on the value just chosen, not the previous start.

Use:

```python
backtrack(value + 1)
```

#### Forgetting to Pop

If you append a number before recursion, you must remove it after recursion:

```python
path.append(value)
backtrack(value + 1)
path.pop()
```

Without `pop`, sibling branches inherit choices they should not have.

#### Allowing Decreasing Choices

If each recursive call starts again at `1`, the algorithm will generate duplicate orderings such as `[1, 2]` and `[2, 1]`.

The increasing-order invariant is what prevents this.

#### Over-Pruning by One

The loop limit should include the last possible useful value:

```python
limit = n - remaining_slots + 1
for value in range(start, limit + 1):
```

Forgetting the `+ 1` in either place can skip valid combinations near the end.

---

### 14. First-Principles Summary

A combination is a set-like choice, so order should not create new answers.

To avoid duplicates, give every combination one canonical representation:

```text
strictly increasing order
```

Then the recursive state becomes simple:

```text
path  = the increasing numbers chosen so far
start = the smallest number allowed next
```

At each step, choose a next number, recurse only on larger numbers, then undo the choice.

The invariant guarantees validity. The increasing order guarantees uniqueness. The loop over possible next values guarantees completeness. The pruning only removes branches that do not have enough numbers left to reach length `k`.

That is the whole algorithm: not a memorized template, but a direct consequence of representing each combination exactly once.

## Implementation

See `solutions/backtracking/p077_combinations.py`.

## Tests

See `tests/backtracking/test_p077_combinations.py`.

## Examples

### Example 1
- Input: `{'n': 4, 'k': 2}`
- Output: `[[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]`

### Example 2
- Input: `{'n': 1, 'k': 1}`
- Output: `[[1]]`

## Follow-up Practice
- Draw the recursion tree for `n = 4`, `k = 2` and label each `start` value.
- Explain why `[2, 1]` is not a missing answer.
- Derive the pruning limit `n - remaining_slots + 1` from first principles.
- Mark exactly where `path` is changed, copied, and restored.
