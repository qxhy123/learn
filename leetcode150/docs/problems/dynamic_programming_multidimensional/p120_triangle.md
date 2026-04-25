# 120. Triangle

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/triangle/
- Official Group: Multidimensional DP
- Pattern Group: Dynamic Programming Multidimensional
- Patterns: dynamic-programming-multidimensional

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given a triangle of numbers:

```text
row 0:        2
row 1:      3   4
row 2:    6   5   7
row 3:  4   1   8   3
```

A path starts at the top number.

From a number at position `j` in row `i`, the next step must go to one of the two adjacent numbers in the row below:

```text
triangle[i + 1][j]
triangle[i + 1][j + 1]
```

So from `3` in row `1`, position `0`, you may move to `6` or `5`.

From `4` in row `1`, position `1`, you may move to `5` or `7`.

The task is to choose one valid top-to-bottom path with the smallest possible sum.

For the first example:

```text
triangle = [
  [2],
  [3, 4],
  [6, 5, 7],
  [4, 1, 8, 3]
]
```

One minimum path is:

```text
2 -> 3 -> 5 -> 1
```

Its sum is:

```text
2 + 3 + 5 + 1 = 11
```

So the answer is `11`.

The real problem is:

> Among all valid paths from the top row to the bottom row, find the minimum possible path sum.

The path constraint is local: each move only goes to one of two adjacent children. The optimization goal is global: minimize the total sum over the whole path.

---

### 2. Start From the Brute Force Recursion

The most direct way to think about the problem is recursive choice.

At any cell `(row, col)`, there are only two possible next moves:

```text
down-left-ish:  (row + 1, col)
down-right-ish: (row + 1, col + 1)
```

So define a recursive function:

```text
min_path(row, col) = minimum path sum starting at triangle[row][col]
                     and ending somewhere in the bottom row
```

If we are already on the last row, there is nowhere else to go:

```text
min_path(last_row, col) = triangle[last_row][col]
```

Otherwise, the path must include the current value plus the cheaper of the two child paths:

```text
min_path(row, col) = triangle[row][col] + min(
    min_path(row + 1, col),
    min_path(row + 1, col + 1)
)
```

The answer is:

```text
min_path(0, 0)
```

Conceptually:

```python
def dfs(row, col):
    if row == len(triangle) - 1:
        return triangle[row][col]

    left_child = dfs(row + 1, col)
    right_child = dfs(row + 1, col + 1)

    return triangle[row][col] + min(left_child, right_child)

answer = dfs(0, 0)
```

This is correct because it tries both choices at every non-bottom cell.

But it is inefficient.

A triangle with `n` rows has paths that branch downward. Without caching, the recursion recomputes the same subproblems many times.

For example, in the sample triangle, the cell `5` at `(2, 1)` can be reached by two different prefixes:

```text
2 -> 3 -> 5
2 -> 4 -> 5
```

If recursive search reaches `(2, 1)` through both prefixes, it recomputes the best suffix from `5` twice.

That repeated suffix work is exactly what dynamic programming removes.

---

### 3. The Key Observation

A path sum can be split into two parts:

```text
prefix already chosen + best possible suffix from the current cell
```

Once you are standing on a specific cell `(row, col)`, the best way to continue downward does not depend on how you got there.

For example, if you are at the `5` in:

```text
      5
    1   8
```

then the best continuation from `5` is:

```text
5 + min(1, 8) = 6
```

This is true whether the prefix was `2 -> 3 -> 5` or `2 -> 4 -> 5`.

That gives the core first-principles insight:

> The identity of the current cell is enough information to describe the remaining optimization problem.

So the dynamic programming state should be attached to a cell, not to an entire path.

Because a cell in the triangle is identified by two coordinates, one index is not enough. We need a two-dimensional state:

```text
(row, col)
```

---

### 4. DP State and Invariant

Use this state:

```text
dp[row][col] = minimum path sum from triangle[row][col]
               down to the bottom row
```

This is a suffix definition: it answers the question, “If I start at this cell, what is the cheapest total cost to finish?”

The invariant is:

```text
After processing row r, every dp cell in rows r through the bottom
stores the true minimum suffix path sum from that cell to the bottom.
```

The bottom row is the base case.

For any bottom-row cell, the minimum path sum from that cell to the bottom is just the cell itself:

```text
dp[last][col] = triangle[last][col]
```

For an interior cell, the next step must be one of its two children. Therefore:

```text
dp[row][col] = triangle[row][col] + min(
    dp[row + 1][col],
    dp[row + 1][col + 1]
)
```

This recurrence has a very specific dependency direction:

```text
row depends on row + 1
```

So the natural fill order is bottom-up.

---

### 5. Why Bottom-Up Works So Cleanly

When computing `dp[row][col]`, we need the answers for the two child cells below it:

```text
dp[row + 1][col]
dp[row + 1][col + 1]
```

If we fill from top to bottom, those child answers are not known yet.

If we fill from bottom to top, they are already finalized.

So bottom-up DP mirrors the recurrence exactly:

1. Start with the bottom row.
2. Move one row upward at a time.
3. Replace each cell's future uncertainty with the already-known cheaper child suffix.
4. When the top cell is processed, it contains the best path sum for the whole triangle.

This can be implemented with a full `dp` triangle, but we do not actually need to keep every row.

Each row only depends on the row directly below it.

So we can keep a one-dimensional array:

```text
dp[col] = minimum suffix path sum from the cell at the current processed row and column
```

Initially, `dp` is a copy of the bottom row.

Then each row above rewrites the relevant entries:

```text
dp[col] = triangle[row][col] + min(dp[col], dp[col + 1])
```

Before the assignment:

```text
dp[col]     means best suffix from lower-left child
dp[col + 1] means best suffix from lower-right child
```

After the assignment:

```text
dp[col] means best suffix from the current cell
```

That is the whole algorithm.

---

### 6. Detailed Algorithm

Given `triangle`:

1. Let `n = len(triangle)`.
2. Copy the last row into `dp`.
3. Iterate `row` from `n - 2` down to `0`.
4. For each column `col` in that row:
   - The two possible child suffix costs are `dp[col]` and `dp[col + 1]`.
   - Choose the smaller one.
   - Add the current triangle value.
   - Store the result back into `dp[col]`.
5. Return `dp[0]`.

Pseudocode:

```python
def minimumTotal(triangle):
    dp = triangle[-1][:]

    for row in range(len(triangle) - 2, -1, -1):
        for col in range(len(triangle[row])):
            dp[col] = triangle[row][col] + min(dp[col], dp[col + 1])

    return dp[0]
```

A full-table version is also possible:

```python
def minimumTotal(triangle):
    dp = [row[:] for row in triangle]

    for row in range(len(triangle) - 2, -1, -1):
        for col in range(len(triangle[row])):
            dp[row][col] = triangle[row][col] + min(
                dp[row + 1][col],
                dp[row + 1][col + 1],
            )

    return dp[0][0]
```

The one-dimensional version is usually preferred because it has the same logic with less memory.

---

### 7. Walk Through the Main Example

Start with:

```text
triangle = [
  [2],
  [3, 4],
  [6, 5, 7],
  [4, 1, 8, 3]
]
```

Initialize `dp` to the bottom row:

```text
dp = [4, 1, 8, 3]
```

Meaning:

```text
from 4 to bottom costs 4
from 1 to bottom costs 1
from 8 to bottom costs 8
from 3 to bottom costs 3
```

Now process row `2`:

```text
row values = [6, 5, 7]
```

For cell `6` at column `0`, children are `4` and `1`:

```text
6 + min(4, 1) = 7
```

Update:

```text
dp = [7, 1, 8, 3]
```

For cell `5` at column `1`, children are `1` and `8`:

```text
5 + min(1, 8) = 6
```

Update:

```text
dp = [7, 6, 8, 3]
```

For cell `7` at column `2`, children are `8` and `3`:

```text
7 + min(8, 3) = 10
```

Update:

```text
dp = [7, 6, 10, 3]
```

After row `2`, the meaningful prefix of `dp` says:

```text
best suffix from 6 is 7
best suffix from 5 is 6
best suffix from 7 is 10
```

Now process row `1`:

```text
row values = [3, 4]
```

For cell `3`, children suffixes are `7` and `6`:

```text
3 + min(7, 6) = 9
```

Update:

```text
dp = [9, 6, 10, 3]
```

For cell `4`, children suffixes are `6` and `10`:

```text
4 + min(6, 10) = 10
```

Update:

```text
dp = [9, 10, 10, 3]
```

Now process row `0`:

```text
row values = [2]
```

For cell `2`, children suffixes are `9` and `10`:

```text
2 + min(9, 10) = 11
```

Update:

```text
dp = [11, 10, 10, 3]
```

The answer is the top suffix cost:

```text
dp[0] = 11
```

This corresponds to path:

```text
2 -> 3 -> 5 -> 1
```

---

### 8. Why The In-Place One-Dimensional Update Is Safe

The line:

```python
dp[col] = triangle[row][col] + min(dp[col], dp[col + 1])
```

overwrites `dp[col]`.

That is safe because, when processing a row from left to right:

- `dp[col]` before overwrite is the lower-left child for the current cell.
- `dp[col + 1]` before overwrite is the lower-right child for the current cell.
- Once `dp[col]` has been overwritten, the old value is no longer needed by any cell to the right.

For the next cell, `col + 1`, the needed children are:

```text
old dp[col + 1]
old dp[col + 2]
```

The algorithm has not overwritten `dp[col + 1]` yet, so those values remain available.

This is why left-to-right works for the bottom-up one-dimensional version.

---

### 9. Correctness

We prove that the algorithm returns the minimum valid top-to-bottom path sum.

Define:

```text
dp[col] after processing row r = minimum path sum from triangle[r][col]
                                 to the bottom row
```

#### Base Case

Before processing any upper row, `dp` is initialized as a copy of the bottom row.

For every bottom-row column `col`, the only path starting at that cell is the cell itself.

Therefore:

```text
dp[col] = triangle[last][col]
```

is the correct minimum suffix path sum for every bottom-row cell.

So the invariant holds for the bottom row.

#### Inductive Step

Assume that before processing row `r`, `dp[col]` and `dp[col + 1]` are already the correct minimum suffix path sums for the two children of `triangle[r][col]`.

Any valid path starting at `triangle[r][col]` must next move to exactly one of those two children:

```text
triangle[r + 1][col]
triangle[r + 1][col + 1]
```

By the induction hypothesis, the cheapest possible continuation through the first child costs `dp[col]`, and the cheapest possible continuation through the second child costs `dp[col + 1]`.

Therefore the cheapest path starting at `triangle[r][col]` is:

```text
triangle[r][col] + min(dp[col], dp[col + 1])
```

The algorithm stores exactly that value in `dp[col]`.

So after row `r` is processed, the invariant holds for row `r`.

#### Final State

The algorithm processes rows upward until row `0`.

After row `0` is processed, `dp[0]` is the minimum path sum from the top cell to the bottom row.

That is exactly the value the problem asks for.

Therefore the algorithm is correct.

---

### 10. Complexity

Let `n` be the number of rows.

The triangle contains:

```text
1 + 2 + ... + n = n(n + 1) / 2
```

cells.

The algorithm performs `O(1)` work per cell.

So the time complexity is:

```text
O(n^2)
```

More precisely, it is linear in the number of triangle cells.

The one-dimensional `dp` array has length equal to the bottom row, which is `n`.

So the space complexity is:

```text
O(n)
```

If the input triangle itself may be modified, it is also possible to update `triangle` in place and use `O(1)` extra space. But the separate `dp` array is often cleaner because it leaves the input unchanged.

---

### 11. Common Pitfalls

- **Using greedy choice from the top.** Choosing the smaller immediate child does not always give the globally smallest path. A slightly larger child may lead to much cheaper values later.
- **Forgetting that adjacent means same column or next column.** From `(row, col)`, the only children are `(row + 1, col)` and `(row + 1, col + 1)`.
- **Returning the minimum of the bottom row after top-down accumulation when using the wrong state.** That approach can work, but only if the state means “minimum cost to reach this cell from the top.” It is a different invariant from the bottom-up suffix DP described here.
- **Overwriting values in the wrong direction for a top-down rolling array.** Direction matters when reusing one array. The bottom-up version shown here is safe left-to-right because each cell reads `dp[col]` and `dp[col + 1]` before `dp[col + 1]` is overwritten.
- **Assuming values are positive.** Triangle values can be negative. The DP recurrence still works because it compares complete suffix costs, not local positive growth.
- **Mutating the input accidentally.** If using `dp = triangle[-1]` instead of `dp = triangle[-1][:]`, updates to `dp` also mutate the last row of `triangle`.

---

### 12. First-Principles Summary

The problem asks for the cheapest constrained path through a triangular grid.

Brute-force recursion is natural because each cell has two choices, but it repeats the same suffix computations.

The essential observation is that once a path reaches a cell, the best remaining cost depends only on that cell, not on the prefix used to reach it.

That makes `(row, col)` a complete DP state.

Define each state as the cheapest path from that cell to the bottom. The bottom row is known immediately, and every row above can be computed from the two adjacent states below it.

The algorithm is therefore just the recurrence written bottom-up:

```text
current cell value + cheaper child suffix
```

When the top cell is computed, it contains the minimum total path sum for the whole triangle.

## Implementation
See `solutions/dynamic_programming_multidimensional/p120_triangle.py`.

## Tests
See `tests/dynamic_programming_multidimensional/test_p120_triangle.py`.

## Examples

### Example 1
- Input: `{'triangle': [[2], [3, 4], [6, 5, 7], [4, 1, 8, 3]]}`
- Output: `11`

### Example 2
- Input: `{'triangle': [[-10]]}`
- Output: `-10`

## Follow-up Practice
- Rewrite the recurrence using a top-down memoized function and compare it to the bottom-up version.
- Trace the one-dimensional `dp` array on a triangle that contains negative numbers.
- Explain why the smaller immediate child is not enough to make a correct greedy algorithm.
