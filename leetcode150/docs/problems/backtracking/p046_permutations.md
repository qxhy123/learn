# 46. Permutations

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/permutations/
- Official Group: Backtracking
- Pattern Group: Backtracking
- Patterns: backtracking, permutations

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given an array `nums` containing distinct integers.

You must return every possible ordering of those integers.

For example:

```text
nums = [1, 2, 3]
```

One valid ordering is:

```text
[1, 2, 3]
```

Another valid ordering is:

```text
[2, 1, 3]
```

Another is:

```text
[3, 2, 1]
```

The word **permutation** means:

```text
Use every input element exactly once, but allow the order to change.
```

So the answer for `[1, 2, 3]` has all length-3 arrays that contain `1`, `2`, and `3` exactly once:

```text
[1, 2, 3]
[1, 3, 2]
[2, 1, 3]
[2, 3, 1]
[3, 1, 2]
[3, 2, 1]
```

The problem is not asking whether one permutation exists. It is asking us to **generate all of them**.

That changes the shape of the solution. We are not searching for a single best answer; we must systematically visit every valid ordering without missing any and without using an element twice in the same ordering.

---

### 2. Start From the Brute Force Idea

A very direct way to think about the problem is:

1. Generate every possible length-`n` sequence using values from `nums`.
2. Keep only the sequences that use each value exactly once.

For `nums = [1, 2, 3]`, if we allowed every position to choose any value, then we would generate sequences like:

```text
[1, 1, 1]
[1, 1, 2]
[1, 1, 3]
[1, 2, 1]
...
```

There are `n` choices for each of `n` positions, so this creates:

```text
n^n
```

candidate sequences.

Most of them are invalid because they reuse numbers.

For example:

```text
[1, 1, 2]
```

is not a permutation of `[1, 2, 3]` because `1` appears twice and `3` is missing.

This brute-force idea is correct in spirit because every real permutation appears somewhere among those candidates. But it does much more work than necessary.

The better question is:

> Can we avoid creating invalid partial sequences in the first place?

Yes. At every position, choose only a number that has not already been used.

---

### 3. Build the Answer One Position at a Time

A permutation has positions:

```text
position 0, position 1, position 2, ..., position n - 1
```

Instead of trying to invent a whole permutation at once, build a partial permutation called `path`.

For example, with:

```text
nums = [1, 2, 3]
```

we might have:

```text
path = [2]
```

This means:

```text
We have chosen 2 for the first position.
The remaining positions still need to be filled.
```

The next value cannot be `2` again, because each input value can appear only once.

So from:

```text
path = [2]
```

valid next choices are:

```text
1 or 3
```

If we choose `1`, the path becomes:

```text
path = [2, 1]
```

Now only `3` remains.

If we choose `3`, the path becomes:

```text
path = [2, 1, 3]
```

Now the path length equals `len(nums)`, so we have one complete permutation.

This is the core construction:

```text
At each step, append one unused number to the current path.
When the path contains all numbers, record it.
```

---

### 4. The Key Observation

The key observation is:

> A partial permutation is valid exactly when it contains no repeated input value.

If `nums` contains distinct values, then we can track validity with a simple `used` structure.

For each number or index, `used` answers:

```text
Has this value already been placed in the current path?
```

Then the local rule is simple:

```text
Only append nums[i] if it is not currently used.
```

Why is this enough?

Because a permutation has only two requirements:

1. It has length `n`.
2. It uses every input element exactly once.

If the path is built only from input elements and never repeats one, then any path of length `n` must contain all `n` distinct input elements exactly once.

So we do not need a complicated validity check at the end. We preserve validity while building.

---

### 5. Why Backtracking Fits This Problem

At each position, there are several possible choices.

For `nums = [1, 2, 3]`, the first position can be:

```text
1, 2, or 3
```

If the first position is `1`, the second position can be:

```text
2 or 3
```

If the path is `[1, 2]`, the third position must be:

```text
3
```

This naturally forms a decision tree:

```text
start
├── choose 1
│   ├── choose 2
│   │   └── choose 3  => [1, 2, 3]
│   └── choose 3
│       └── choose 2  => [1, 3, 2]
├── choose 2
│   ├── choose 1
│   │   └── choose 3  => [2, 1, 3]
│   └── choose 3
│       └── choose 1  => [2, 3, 1]
└── choose 3
    ├── choose 1
    │   └── choose 2  => [3, 1, 2]
    └── choose 2
        └── choose 1  => [3, 2, 1]
```

Backtracking is a disciplined way to walk this tree.

The repeated pattern is:

```text
choose
explore everything that follows from that choice
undo the choice
try the next choice
```

The undo step is important. If we append `1` to `path`, explore all permutations starting with `1`, and then want to explore permutations starting with `2`, the old `1` must be removed from the current path first.

---

### 6. Recursive State and Invariant

A clean recursive implementation keeps three pieces of state:

```text
path     = the partial permutation built so far
used     = which input positions are already inside path
result   = all complete permutations found so far
```

The recursion depth equals:

```text
len(path)
```

That depth also means:

```text
The next recursive call is choosing the value for position len(path).
```

The main invariant is:

```text
At the start of every recursive call:
1. path contains only values from nums.
2. path contains no repeated input position.
3. used[i] is true exactly when nums[i] is currently in path.
4. Every completion of path using the unused values is still possible from this state.
```

This invariant is the reason the algorithm is simple.

When we choose an unused index `i`:

```text
path.append(nums[i])
used[i] = true
```

The invariant remains true because `nums[i]` was not already in `path`.

After recursion returns, we undo:

```text
used[i] = false
path.pop()
```

This restores the exact state that existed before trying `nums[i]`, so the next candidate choice starts from a clean partial permutation.

---

### 7. Detailed Algorithm

Use a helper function, often called `backtrack`.

At any call:

1. If `len(path) == len(nums)`, the path is a complete permutation.
2. Append a copy of `path` to `result`.
3. Otherwise, scan every index `i` in `nums`.
4. If `used[i]` is true, skip it because that value is already in the current path.
5. If `used[i]` is false:
   - Mark it used.
   - Append `nums[i]` to `path`.
   - Recurse to fill the next position.
   - Pop it from `path`.
   - Mark it unused again.

The scan over all indices is repeated at each depth. That is fine because the `used` array tells us which choices remain legal for the current branch.

In Python-like pseudocode:

```python
def permute(nums):
    result = []
    path = []
    used = [False] * len(nums)

    def backtrack():
        if len(path) == len(nums):
            result.append(path.copy())
            return

        for i in range(len(nums)):
            if used[i]:
                continue

            used[i] = True
            path.append(nums[i])

            backtrack()

            path.pop()
            used[i] = False

    backtrack()
    return result
```

The copy in `result.append(path.copy())` is necessary because `path` is mutable and will keep changing as the recursion continues.

---

### 8. Walkthrough: `nums = [1, 2, 3]`

Start with:

```text
path = []
used = [false, false, false]
result = []
```

#### Choose the first value

The loop tries index `0`, value `1`:

```text
path = [1]
used = [true, false, false]
```

Now recurse to fill the second position.

#### From `[1]`, choose the second value

The loop sees index `0` is already used, so it skips `1`.

It tries index `1`, value `2`:

```text
path = [1, 2]
used = [true, true, false]
```

Now recurse to fill the third position.

#### From `[1, 2]`, choose the third value

Indices `0` and `1` are used.

The only unused value is `3`:

```text
path = [1, 2, 3]
used = [true, true, true]
```

Now `len(path) == len(nums)`, so record a copy:

```text
result = [[1, 2, 3]]
```

Then return.

#### Undo `3`

Backtracking removes the last choice:

```text
path = [1, 2]
used = [true, true, false]
```

There are no other unused choices at this depth, so return again.

#### Undo `2`

Now the state is back to:

```text
path = [1]
used = [true, false, false]
```

The loop that was choosing the second value continues.

It already tried `2`, so it next tries `3`:

```text
path = [1, 3]
used = [true, false, true]
```

Only `2` remains, so the next complete permutation is:

```text
[1, 3, 2]
```

Record it:

```text
result = [[1, 2, 3], [1, 3, 2]]
```

Then undo back to `path = [1]`, and eventually undo `1` as well.

#### Move to first value `2`

After all branches starting with `1` are done, the original call tries value `2` as the first element:

```text
path = [2]
used = [false, true, false]
```

This branch produces:

```text
[2, 1, 3]
[2, 3, 1]
```

#### Move to first value `3`

Finally, the first element is `3`:

```text
path = [3]
used = [false, false, true]
```

This branch produces:

```text
[3, 1, 2]
[3, 2, 1]
```

The final result is:

```text
[
  [1, 2, 3],
  [1, 3, 2],
  [2, 1, 3],
  [2, 3, 1],
  [3, 1, 2],
  [3, 2, 1]
]
```

---

### 9. Why the Algorithm Does Not Miss Any Permutation

Take any valid permutation, for example:

```text
[2, 3, 1]
```

At depth `0`, the algorithm loops over every unused value, including `2`.

So there is a branch where it chooses:

```text
path = [2]
```

At depth `1`, it loops over every value not already used, including `3`.

So there is a branch where it chooses:

```text
path = [2, 3]
```

At depth `2`, the only remaining value in this target permutation is `1`, and the algorithm tries it.

So it reaches:

```text
path = [2, 3, 1]
```

This reasoning works for any permutation. Whatever its first value is, the algorithm has that branch. Whatever its second value is, the algorithm has that branch under the first. This continues until the whole permutation is built.

Therefore no valid permutation is missed.

---

### 10. Why the Algorithm Does Not Produce Invalid Permutations

The algorithm appends a number only when its corresponding `used[i]` is false.

So a single path can never contain the same input index twice.

The input numbers are distinct, so using each index at most once means using each value at most once.

The algorithm records a path only when:

```text
len(path) == len(nums)
```

A length-`n` path made from `n` distinct input elements with no repeated index must contain every input element exactly once.

So every recorded path is a valid permutation.

---

### 11. Correctness Argument

We prove that the algorithm returns exactly all permutations of `nums`.

**Invariant.** At the start of each call to `backtrack`, `path` contains distinct elements from `nums`, and `used[i]` is true exactly for the indices already represented in `path`.

**Initialization.** Before the first call, `path` is empty and all entries in `used` are false. The invariant holds.

**Maintenance.** During a call, the algorithm only chooses an index `i` with `used[i] == false`. It appends `nums[i]` and sets `used[i]` to true, so the new path still contains distinct input elements and `used` remains accurate. After the recursive call, it pops that value and resets `used[i]` to false, restoring the previous state before trying the next choice.

**Validity.** The algorithm records a path only when its length is `len(nums)`. By the invariant, the path contains distinct elements from `nums`. Since there are exactly `len(nums)` distinct input elements and the path has that length, the path contains every input element exactly once. Therefore every recorded path is a valid permutation.

**Completeness.** Consider any valid permutation `p`. Its first element is some value from `nums`, and the first loop tries that value. Once chosen, its second element is one of the remaining unused values, and the next recursive loop tries that value. Repeating this argument for every position shows that the recursion has a branch that builds exactly `p`. Therefore every valid permutation is recorded.

Because every recorded result is valid and every valid permutation is recorded, the algorithm is correct.

---

### 12. Complexity

Let:

```text
n = len(nums)
```

There are:

```text
n!
```

permutations.

Each complete permutation has length `n`, and copying it into the answer costs `O(n)`.

So the output alone costs:

```text
O(n * n!)
```

time and space.

The recursive search also scans up to `n` indices at each internal node. This is commonly summarized as:

```text
Time: O(n * n!)
```

because generating and copying all `n!` length-`n` permutations dominates the output cost.

Space excluding the returned output:

```text
O(n)
```

for:

```text
path
used
recursion stack
```

Space including the returned output:

```text
O(n * n!)
```

because the result stores `n!` permutations, each of length `n`.

---

### 13. Common Pitfalls

#### Appending `path` Instead of a Copy

This is wrong:

```python
result.append(path)
```

`path` is mutated later by `pop()` and `append()`. If you store the same list object repeatedly, all entries in `result` can end up pointing to the same changing list.

Use:

```python
result.append(path.copy())
```

#### Forgetting to Undo the Choice

After recursion returns, both pieces of state must be restored:

```python
path.pop()
used[i] = False
```

If you forget `path.pop()`, later branches start with extra values.

If you forget `used[i] = False`, later branches incorrectly believe the value is unavailable.

#### Recording Too Early

Only record when:

```text
len(path) == len(nums)
```

A shorter path like `[1, 2]` is not a permutation of `[1, 2, 3]` because it does not use all input values.

#### Reusing a Value in One Branch

If you do not track `used`, you may generate invalid sequences like:

```text
[1, 1, 1]
```

The entire point of the state is to prevent this.

#### Confusing This Problem With Unique Permutations With Duplicates

LeetCode 46 says the input numbers are distinct.

If the input could contain duplicates, such as:

```text
[1, 1, 2]
```

then using indices alone would produce duplicate output rows unless additional duplicate-skipping logic is added.

That is a different problem: LeetCode 47, Permutations II.

For this problem, no duplicate-skipping rule is needed.

---

### 14. First-Principles Summary

A permutation is not mysterious. It is just a complete sequence formed by choosing every input element exactly once.

The first-principles representation is:

```text
path = positions already filled
used = input elements already placed in path
```

The invariant is:

```text
path is always a valid partial permutation
```

The safe local move is:

```text
choose one unused element and append it
```

The base case is:

```text
if path length equals nums length, record a copy
```

Backtracking works because it explores one choice, recursively explores everything that follows from that choice, and then undoes the choice so the next branch starts from the same clean state.

For `nums = [1, 2, 3]`, this means:

```text
choose first element
choose second unused element
choose third unused element
record
undo and try the next unused choice
```

That process walks the full decision tree of possible orderings and records exactly the leaves of that tree.

## Implementation

See `solutions/backtracking/p046_permutations.py`.

## Tests

See `tests/backtracking/test_p046_permutations.py`.

## Examples

### Example 1
- Input: `{'nums': [1, 2, 3]}`
- Output: `[[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]`

### Example 2
- Input: `{'nums': [0, 1]}`
- Output: `[[0, 1], [1, 0]]`

### Example 3
- Input: `{'nums': [1]}`
- Output: `[[1]]`

## Follow-up Practice

- Draw the decision tree for `[1, 2, 3]` and label each edge with the chosen value.
- Write the invariant in your own words before writing code.
- Trace exactly when `path.append`, `path.pop`, `used[i] = True`, and `used[i] = False` happen.
- Explain why `path.copy()` is necessary when storing a completed permutation.
