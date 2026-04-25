# 39. Combination Sum

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/combination-sum/
- Official Group: Backtracking
- Pattern Group: Backtracking
- Patterns: backtracking, sum

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given:

```text
candidates = a list of distinct positive integers
target     = a positive integer
```

Return every unique combination of numbers from `candidates` whose sum is exactly `target`.

There are two important rules:

1. A candidate number may be used more than once.
2. The answer should not contain duplicate combinations.

For example:

```text
candidates = [2, 3, 6, 7]
target = 7
```

The combination:

```text
[7]
```

is valid because it sums to `7`.

The combination:

```text
[2, 2, 3]
```

is also valid because:

```text
2 + 2 + 3 = 7
```

But these are not different combinations:

```text
[2, 2, 3]
[2, 3, 2]
[3, 2, 2]
```

They contain the same multiset of chosen values, just in a different order. The problem asks for combinations, not permutations, so those three arrangements count as one answer.

The real problem is:

> Choose zero or more copies of each candidate so that the total is exactly `target`, and list each distinct multiset once.

---

### 2. Start From the Brute Force Idea

A very direct way to think about the problem is:

```text
At each step, choose one candidate and add it to the current partial combination.
```

For the first example, one branch might look like this:

```text
[]
[2]
[2, 2]
[2, 2, 2]
[2, 2, 2, 2]
```

The sum has now become `8`, which is larger than `target = 7`, so that branch cannot lead to a valid answer.

Another branch:

```text
[]
[2]
[2, 2]
[2, 2, 3]
```

has sum `7`, so it should be recorded.

This brute-force decision tree is correct in spirit: try possibilities, stop when the sum is too large, and record the ones that hit the target.

But there is a serious duplicate problem. If every recursive call is allowed to choose any candidate, then the same combination is generated in many orders:

```text
2 -> 2 -> 3
2 -> 3 -> 2
3 -> 2 -> 2
```

All three are the same combination. If we generate ordered sequences and try to remove duplicates afterward, the algorithm becomes messier and wastes work.

The main challenge is therefore not only finding sums. It is finding sums while giving every combination exactly one representation.

---

### 3. The Key Observation: Enforce an Order

A combination does not care about order, but our recursion needs an order so it can avoid duplicates.

The simplest rule is:

```text
Once we choose candidates from position i or later, future choices may only come from position i or later.
```

That means the path of chosen candidate indexes is nondecreasing.

If `candidates = [2, 3, 6, 7]`, then:

```text
[2, 2, 3]
```

is allowed because its indexes are:

```text
0, 0, 1
```

They never go backward.

But:

```text
[3, 2, 2]
```

would require indexes:

```text
1, 0, 0
```

That goes backward from index `1` to index `0`, so the recursion never generates it.

This does not lose any valid combination, because every combination can be written in nondecreasing candidate order. We simply choose that sorted-by-candidate-order representation as the one canonical version to generate.

---

### 4. Why Reuse Is Allowed Without Creating Duplicates

The problem allows using the same number multiple times.

So after choosing candidate `i`, the next recursive call should still allow candidate `i` again.

That is the difference between:

```text
recurse(i, ...)
```

and:

```text
recurse(i + 1, ...)
```

For this problem, after choosing `candidates[i]`, we recurse with the same starting index `i`:

```text
path.append(candidates[i])
backtrack(i, remaining - candidates[i])
path.pop()
```

This permits:

```text
2 -> 2 -> 2 -> 2
```

when `2` is useful.

But the loop inside that recursive call still starts at `i`, not at `0`, so it does not go backward and generate reordered duplicates.

---

### 5. Recursion State and Invariant

Use a recursive function with this state:

```text
backtrack(start, remaining)
```

where:

```text
start     = the first candidate index still allowed to be chosen
remaining = how much sum is still needed to reach target
path      = the current partial combination
```

The central invariant is:

```text
sum(path) + remaining == target
```

and:

```text
all candidate indexes used in path are nondecreasing
```

The first invariant tells us that `remaining` accurately represents the unfinished part of the sum.

The second invariant tells us that `path` is the canonical representation of its combination. It prevents the same multiset from appearing in multiple orders.

At every recursive call:

- If `remaining == 0`, then `path` is a complete valid combination.
- If `remaining < 0`, then `path` is too large and cannot be repaired because all candidates are positive.
- Otherwise, try each candidate from `start` onward.

Because candidates are positive, adding more numbers can only increase the path sum and decrease `remaining`. Once `remaining` becomes negative, that branch is impossible.

---

### 6. Detailed Algorithm

1. Initialize an empty answer list:

```text
result = []
```

2. Initialize an empty current path:

```text
path = []
```

3. Define `backtrack(start, remaining)`.

4. If `remaining == 0`, append a copy of `path` to `result` and return.

5. If `remaining < 0`, return because the path already exceeds the target.

6. For every index `i` from `start` to the end of `candidates`:

```text
candidate = candidates[i]
```

7. Choose it:

```text
path.append(candidate)
```

8. Recurse with the same `i`, because this candidate may be reused:

```text
backtrack(i, remaining - candidate)
```

9. Undo the choice so the next loop iteration starts from the previous state:

```text
path.pop()
```

10. Start the search with:

```text
backtrack(0, target)
```

11. Return `result`.

The important detail is the recursive call:

```text
backtrack(i, remaining - candidate)
```

not:

```text
backtrack(0, remaining - candidate)
```

Starting again from `0` would allow smaller-index candidates after larger-index candidates and would create duplicate permutations.

---

### 7. Pseudocode

```python
def combinationSum(candidates, target):
    result = []
    path = []

    def backtrack(start, remaining):
        if remaining == 0:
            result.append(path.copy())
            return

        if remaining < 0:
            return

        for i in range(start, len(candidates)):
            value = candidates[i]
            path.append(value)
            backtrack(i, remaining - value)  # reuse value by keeping i
            path.pop()

    backtrack(0, target)
    return result
```

A common optimization is to sort `candidates` first. Then, inside the loop, if `candidates[i] > remaining`, all later candidates are also too large and the loop can stop early:

```python
candidates.sort()

for i in range(start, len(candidates)):
    value = candidates[i]
    if value > remaining:
        break
    path.append(value)
    backtrack(i, remaining - value)
    path.pop()
```

Sorting is not required for correctness if we keep the `remaining < 0` check, but it makes pruning sharper and gives a predictable output order for typical examples.

---

### 8. Example Walkthrough: `candidates = [2, 3, 6, 7]`, `target = 7`

Start with:

```text
path = []
remaining = 7
start = 0
```

The loop can choose `2`, `3`, `6`, or `7`.

#### Choose `2`

```text
path = [2]
remaining = 5
start = 0
```

We pass `start = 0` again because `2` can be reused.

Choose `2` again:

```text
path = [2, 2]
remaining = 3
start = 0
```

Choose `2` again:

```text
path = [2, 2, 2]
remaining = 1
start = 0
```

Choosing another `2` would make `remaining = -1`, so that branch fails.

Backtrack to:

```text
path = [2, 2]
remaining = 3
```

Now try `3`:

```text
path = [2, 2, 3]
remaining = 0
```

`remaining == 0`, so record:

```text
[2, 2, 3]
```

Then undo `3` and continue. Trying `6` or `7` from `path = [2, 2]` would exceed the target.

Backtrack again to:

```text
path = [2]
remaining = 5
```

The next choices from index `0` onward eventually try `3`:

```text
path = [2, 3]
remaining = 2
start = 1
```

Notice `start = 1`, because the last chosen index was the index of `3`. From here, future choices may be `3`, `6`, or `7`, but not `2`.

That is why the recursion does not generate:

```text
[2, 3, 2]
```

which would duplicate `[2, 2, 3]`.

#### Choose `3` as the first number

After fully exploring branches beginning with `2`, the top-level call tries `3`:

```text
path = [3]
remaining = 4
start = 1
```

It may choose `3` again:

```text
path = [3, 3]
remaining = 1
```

No later candidate can complete the sum, so this path fails. It never tries to add `2` after `3`, because that would go backward and create a duplicate ordering.

#### Choose `6` as the first number

```text
path = [6]
remaining = 1
start = 2
```

No valid continuation exists.

#### Choose `7` as the first number

```text
path = [7]
remaining = 0
```

Record:

```text
[7]
```

The final answer is:

```text
[[2, 2, 3], [7]]
```

The exact ordering of the outer list may vary depending on implementation, but each valid combination appears once.

---

### 9. Correctness

We prove that the algorithm returns exactly the valid combinations.

#### Every recorded path is valid

The algorithm records a path only when:

```text
remaining == 0
```

The invariant says:

```text
sum(path) + remaining == target
```

Therefore, when `remaining == 0`:

```text
sum(path) == target
```

Every value in `path` came from `candidates`, and reuse is allowed because the algorithm may choose the same index again. So every recorded path is a valid combination.

#### No duplicate combination is recorded

The algorithm only recurses from index `i` to future choices starting at index `i` or later.

Therefore, the candidate indexes in every path are nondecreasing.

For any multiset of chosen candidates, there is only one nondecreasing ordering by candidate index. Reordered versions would require some later index to be followed by an earlier index, which the algorithm never allows.

So the same combination cannot be recorded twice in different orders.

#### Every valid combination is eventually recorded

Take any valid combination. Write its candidates in nondecreasing index order.

At the top level, the algorithm's loop can choose the first value in that ordered combination. Because the recursive call keeps the same index available and also permits larger indexes, the next value in the ordered combination is still available. Repeating this argument, the algorithm can follow exactly that ordered sequence of choices.

Since the combination is valid, its sum is `target`, so after the last value is chosen, `remaining` becomes `0`. The algorithm records it.

Therefore, every valid combination appears in the output.

Since every recorded path is valid, no duplicate is recorded, and every valid combination is recorded, the algorithm is correct.

---

### 10. Complexity

Let:

```text
n = len(candidates)
T = target
m = min(candidates)
```

The maximum recursion depth is at most:

```text
T / m
```

because each chosen number decreases `remaining` by at least `m`.

The search tree can still be exponential in that depth, because many partial combinations may need to be explored. A useful output-sensitive way to describe the cost is:

```text
O(number of explored states * average path operation cost)
```

When a valid path is recorded, copying it costs proportional to its length.

So the practical time complexity is commonly described as exponential in the maximum combination length, with additional cost for writing the output:

```text
O(number_of_results * average_result_length)
```

plus the cost of failed partial states explored before pruning.

Space complexity excluding the returned output is:

```text
O(T / m)
```

for the recursion stack and current `path`.

Including the returned output, space also includes all generated combinations.

---

### 11. Common Pitfalls

#### Passing `i + 1` after choosing a candidate

If the recursive call is:

```python
backtrack(i + 1, remaining - value)
```

then each candidate can be used at most once. That solves a different problem. For Combination Sum, reuse is allowed, so the call should keep `i`:

```python
backtrack(i, remaining - value)
```

#### Restarting from index `0`

If the recursive call is:

```python
backtrack(0, remaining - value)
```

then the recursion can generate the same combination in many orders, such as `[2, 2, 3]`, `[2, 3, 2]`, and `[3, 2, 2]`.

#### Appending `path` instead of a copy

This is wrong:

```python
result.append(path)
```

because `path` is mutated later by `pop()` and future `append()` calls.

Use:

```python
result.append(path.copy())
```

#### Forgetting to undo the choice

Every append must be paired with a pop:

```python
path.append(value)
backtrack(i, remaining - value)
path.pop()
```

Without the `pop`, choices from one branch leak into sibling branches.

#### Pruning incorrectly when candidates are unsorted

This is safe only after sorting:

```python
if value > remaining:
    break
```

If candidates are not sorted, `break` may skip a later smaller value that could still work. Without sorting, use `continue` or rely on the `remaining < 0` base case.

#### Assuming output order is the main issue

The core issue is uniqueness of combinations, not the printed order of the answer list. The `start` index prevents duplicate combinations by controlling which future choices are allowed.

---

### 12. First-Principles Summary

The problem asks for all multisets of candidate values whose sum equals `target`.

The brute-force idea is to build partial sums by choosing candidates recursively. The duplicate problem appears because the same multiset can be chosen in many orders.

The key first-principles move is to give every combination one canonical representation: candidate indexes must never decrease.

That leads to the recursive state:

```text
backtrack(start, remaining)
```

where `start` prevents going backward and `remaining` tracks how much sum is still needed.

When we choose `candidates[i]`, we recurse with `i`, not `i + 1`, because the same candidate may be reused. But we do not recurse with `0`, because that would allow reordered duplicates.

The whole algorithm is the disciplined exploration of this decision tree:

```text
choose a candidate at or after start
subtract it from remaining
keep indexes nondecreasing
record the path exactly when remaining reaches zero
undo the choice before trying the next candidate
```

## Implementation

See `solutions/backtracking/p039_combination_sum.py`.

## Tests

See `tests/backtracking/test_p039_combination_sum.py`.

## Examples

### Example 1
- Input: `{'candidates': [2, 3, 6, 7], 'target': 7}`
- Output: `[[2, 2, 3], [7]]`

### Example 2
- Input: `{'candidates': [2, 3, 5], 'target': 8}`
- Output: `[[2, 2, 2, 2], [2, 3, 3], [3, 5]]`

### Example 3
- Input: `{'candidates': [2], 'target': 1}`
- Output: `[]`

## Follow-up Practice
- Draw the decision tree for `candidates = [2, 3]`, `target = 6`.
- Write down the exact meaning of `start` and `remaining` before coding.
- Explain why `backtrack(i, ...)` allows reuse but still avoids duplicate permutations.
- Mark every `append` and the matching `pop` in the recursion.
