# 64. Minimum Path Sum

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/minimum-path-sum/
- Official Group: Multidimensional DP
- Pattern Group: Dynamic Programming Multidimensional
- Patterns: dynamic-programming-multidimensional, sum

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given an `m x n` grid of non-negative numbers.

You start at the top-left cell:

```text
(0, 0)
```

You want to reach the bottom-right cell:

```text
(m - 1, n - 1)
```

At each step, you may move only:

```text
right
or
down
```

The cost of a path is the sum of all cell values visited, including both the start cell and the destination cell.

The problem asks:

> Among all valid paths from the top-left cell to the bottom-right cell, what is the minimum possible path sum?

For example:

```text
grid = [
  [1, 3, 1],
  [1, 5, 1],
  [4, 2, 1]
]
```

One valid path is:

```text
1 -> 3 -> 1 -> 1 -> 1
```

This path moves:

```text
right, right, down, down
```

Its sum is:

```text
1 + 3 + 1 + 1 + 1 = 7
```

Another valid path is:

```text
1 -> 1 -> 5 -> 1 -> 1
```

with sum:

```text
1 + 1 + 5 + 1 + 1 = 9
```

The answer is not asking for the path itself. It asks only for the smallest sum achievable by any valid path.

---

### 2. Start From the Brute Force Idea

The most direct way to think about the problem is:

1. Start at `(0, 0)`.
2. Try every possible sequence of right/down moves.
3. Compute the sum of each complete path.
4. Return the smallest sum.

A recursive brute-force version would say:

```text
minimum cost from cell (r, c)
= grid[r][c] + min(
    minimum cost from the cell below,
    minimum cost from the cell to the right
  )
```

Conceptually:

```python
def dfs(row, col):
    if row == m - 1 and col == n - 1:
        return grid[row][col]

    best_next = infinity

    if row + 1 < m:
        best_next = min(best_next, dfs(row + 1, col))

    if col + 1 < n:
        best_next = min(best_next, dfs(row, col + 1))

    return grid[row][col] + best_next
```

This is correct because every valid path from `(row, col)` must choose either the next down cell or the next right cell.

But it repeats a lot of work.

For example, in a `3 x 3` grid, the cell `(1, 1)` can be reached by two different partial paths:

```text
right, down
down, right
```

From `(1, 1)` onward, the remaining problem is identical both times. A brute-force search recomputes that identical suffix cost again and again.

That repeated work is the signal that dynamic programming should help.

---

### 3. The Key Observation: Every Cell Has Only Two Possible Predecessors

Instead of asking:

```text
From this cell, where can I go next?
```

turn the question around:

```text
If I am standing on this cell, where could I have come from?
```

Because moves are only right or down, a path can enter cell `(row, col)` only from:

```text
(row - 1, col)   the cell above
(row, col - 1)   the cell to the left
```

There are no other possibilities.

You cannot come from below, because that would require moving up.
You cannot come from the right, because that would require moving left.
You cannot jump from a non-adjacent cell.

So once we know the cheapest way to reach the cell above and the cheapest way to reach the cell on the left, the cheapest way to reach the current cell is forced:

```text
cost of current cell + cheaper of those two previous costs
```

That is the whole dynamic programming idea for this problem.

---

### 4. DP State and Invariant

Define:

```text
dp[row][col] = minimum path sum needed to reach grid[row][col]
               from grid[0][0]
               using only right and down moves
```

This definition is important. `dp[row][col]` is not the cheapest path from that cell to the end. It is the cheapest path from the start to that cell.

The invariant we maintain is:

```text
After dp[row][col] is computed, it stores the true minimum cost
of any valid path from (0, 0) to (row, col).
```

If this invariant is true for the cells above and left of the current cell, then the transition is:

```text
dp[row][col] = grid[row][col] + min(dp[row - 1][col], dp[row][col - 1])
```

That formula is valid for interior cells, where both predecessors exist.

Boundary cells need special handling:

- The top-left cell has no predecessor.
- Cells in the first row can only come from the left.
- Cells in the first column can only come from above.

So:

```text
dp[0][0] = grid[0][0]

dp[0][col] = grid[0][col] + dp[0][col - 1]
dp[row][0] = grid[row][0] + dp[row - 1][0]
```

---

### 5. Why the Fill Order Matters

The transition for `(row, col)` reads:

```text
dp[row - 1][col]
dp[row][col - 1]
```

So before computing `(row, col)`, we must already know:

```text
the cell above
the cell on the left
```

A natural order is row by row, from left to right:

```text
for row from 0 to m - 1:
    for col from 0 to n - 1:
        compute dp[row][col]
```

When processing this way:

- The cell above is in a previous row, so it has already been computed.
- The cell on the left is earlier in the same row, so it has already been computed.

This is why a simple nested loop works.

---

### 6. Detailed Algorithm

1. Let `m` be the number of rows and `n` be the number of columns.

2. Create an `m x n` table `dp`.

3. Initialize the starting cell:

```text
dp[0][0] = grid[0][0]
```

4. Fill the first row.

Since cells in the first row can only be reached by repeatedly moving right:

```text
dp[0][col] = dp[0][col - 1] + grid[0][col]
```

5. Fill the first column.

Since cells in the first column can only be reached by repeatedly moving down:

```text
dp[row][0] = dp[row - 1][0] + grid[row][0]
```

6. Fill the remaining cells.

For each interior cell, choose the cheaper predecessor:

```text
dp[row][col] = grid[row][col] + min(dp[row - 1][col], dp[row][col - 1])
```

7. Return the bottom-right value:

```text
dp[m - 1][n - 1]
```

---

### 7. Example Walkthrough

Use the first example:

```text
grid = [
  [1, 3, 1],
  [1, 5, 1],
  [4, 2, 1]
]
```

Create a DP table with the same shape:

```text
dp = [
  [?, ?, ?],
  [?, ?, ?],
  [?, ?, ?]
]
```

#### Start Cell

The only way to stand on `(0, 0)` is to start there:

```text
dp[0][0] = grid[0][0] = 1
```

```text
dp = [
  [1, ?, ?],
  [?, ?, ?],
  [?, ?, ?]
]
```

#### First Row

Cell `(0, 1)` can only come from `(0, 0)`:

```text
dp[0][1] = dp[0][0] + grid[0][1]
         = 1 + 3
         = 4
```

Cell `(0, 2)` can only come from `(0, 1)`:

```text
dp[0][2] = dp[0][1] + grid[0][2]
         = 4 + 1
         = 5
```

Now:

```text
dp = [
  [1, 4, 5],
  [?, ?, ?],
  [?, ?, ?]
]
```

#### First Column

Cell `(1, 0)` can only come from `(0, 0)`:

```text
dp[1][0] = dp[0][0] + grid[1][0]
         = 1 + 1
         = 2
```

Cell `(2, 0)` can only come from `(1, 0)`:

```text
dp[2][0] = dp[1][0] + grid[2][0]
         = 2 + 4
         = 6
```

Now:

```text
dp = [
  [1, 4, 5],
  [2, ?, ?],
  [6, ?, ?]
]
```

#### Interior Cell `(1, 1)`

The current grid value is `5`.

Possible predecessors:

```text
from above: dp[0][1] = 4
from left:  dp[1][0] = 2
```

The cheaper predecessor is `2`, so:

```text
dp[1][1] = grid[1][1] + min(dp[0][1], dp[1][0])
         = 5 + min(4, 2)
         = 7
```

```text
dp = [
  [1, 4, 5],
  [2, 7, ?],
  [6, ?, ?]
]
```

#### Interior Cell `(1, 2)`

The current grid value is `1`.

Possible predecessors:

```text
from above: dp[0][2] = 5
from left:  dp[1][1] = 7
```

The cheaper predecessor is `5`, so:

```text
dp[1][2] = 1 + min(5, 7)
         = 6
```

```text
dp = [
  [1, 4, 5],
  [2, 7, 6],
  [6, ?, ?]
]
```

#### Interior Cell `(2, 1)`

The current grid value is `2`.

Possible predecessors:

```text
from above: dp[1][1] = 7
from left:  dp[2][0] = 6
```

The cheaper predecessor is `6`, so:

```text
dp[2][1] = 2 + min(7, 6)
         = 8
```

```text
dp = [
  [1, 4, 5],
  [2, 7, 6],
  [6, 8, ?]
]
```

#### Destination Cell `(2, 2)`

The current grid value is `1`.

Possible predecessors:

```text
from above: dp[1][2] = 6
from left:  dp[2][1] = 8
```

The cheaper predecessor is `6`, so:

```text
dp[2][2] = 1 + min(6, 8)
         = 7
```

Final table:

```text
dp = [
  [1, 4, 5],
  [2, 7, 6],
  [6, 8, 7]
]
```

The answer is:

```text
dp[2][2] = 7
```

This corresponds to the path:

```text
1 -> 3 -> 1 -> 1 -> 1
```

---

### 8. Pseudocode

```python
def minPathSum(grid):
    rows = len(grid)
    cols = len(grid[0])

    dp = [[0] * cols for _ in range(rows)]

    dp[0][0] = grid[0][0]

    for col in range(1, cols):
        dp[0][col] = dp[0][col - 1] + grid[0][col]

    for row in range(1, rows):
        dp[row][0] = dp[row - 1][0] + grid[row][0]

    for row in range(1, rows):
        for col in range(1, cols):
            best_previous = min(dp[row - 1][col], dp[row][col - 1])
            dp[row][col] = grid[row][col] + best_previous

    return dp[rows - 1][cols - 1]
```

This is the clearest version because the DP table directly matches the grid.

A space-optimized version can use one row, because each state only needs:

```text
the value above     -> old dp[col]
the value on left   -> current dp[col - 1]
```

But the full table is usually better to understand first.

---

### 9. Correctness

We prove that the algorithm returns the minimum possible path sum from the top-left cell to the bottom-right cell.

#### Lemma 1: The initialization is correct.

For `(0, 0)`, the path starts there, so the only possible path sum is `grid[0][0]`. Therefore `dp[0][0]` is correct.

For any cell in the first row, the only legal way to reach it is to move right from the previous cell in the same row. There is no cell above it. Therefore each first-row value is the cumulative sum from the left, which is exactly the minimum and only possible path sum.

For any cell in the first column, the only legal way to reach it is to move down from the previous cell in the same column. There is no cell to its left. Therefore each first-column value is the cumulative sum from above, which is exactly the minimum and only possible path sum.

#### Lemma 2: The transition is correct for every interior cell.

Consider an interior cell `(row, col)`.

Any valid path that ends at `(row, col)` must enter it from exactly one of two cells:

```text
(row - 1, col)
(row, col - 1)
```

These are the only possible predecessors because the only allowed moves are down and right.

By the DP invariant, `dp[row - 1][col]` is the minimum cost to reach the cell above, and `dp[row][col - 1]` is the minimum cost to reach the cell on the left.

So the cheapest path into `(row, col)` must use the cheaper of those two predecessor costs, then add `grid[row][col]` for the current cell.

Therefore:

```text
dp[row][col] = grid[row][col] + min(dp[row - 1][col], dp[row][col - 1])
```

is correct.

#### Lemma 3: The fill order computes every dependency before it is used.

The algorithm fills the first row and first column before interior cells.

Then it processes interior cells from top to bottom and left to right. For any interior cell `(row, col)`:

- `(row - 1, col)` is in an earlier row.
- `(row, col - 1)` is earlier in the same row.

So both required predecessor values have already been computed.

#### Conclusion

By the initialization lemmas and the transition lemma, every `dp[row][col]` stores the true minimum path sum from `(0, 0)` to `(row, col)`.

The destination is `(m - 1, n - 1)`, so `dp[m - 1][n - 1]` is the minimum path sum from the start to the destination. The algorithm returns exactly that value.

---

### 10. Complexity

Let:

```text
m = number of rows
n = number of columns
```

The algorithm computes one DP value for each grid cell.

There are `m * n` cells, and each cell does `O(1)` work.

So the time complexity is:

```text
O(m * n)
```

The full DP table stores one value per cell, so the space complexity is:

```text
O(m * n)
```

If the input grid may be modified, the grid itself can be used as the DP table, reducing extra space to:

```text
O(1)
```

If the input should not be modified, a one-dimensional DP array can reduce extra space to:

```text
O(n)
```

where `n` is the number of columns.

---

### 11. Common Pitfalls

#### Forgetting to include the starting cell

The path sum includes every visited cell, including `grid[0][0]`.

So the base case is:

```text
dp[0][0] = grid[0][0]
```

not `0`.

#### Treating the first row like an interior row

For the first row, there is no cell above.

This expression is invalid at `row = 0`:

```text
dp[row - 1][col]
```

The first row must come only from the left.

#### Treating the first column like an interior column

For the first column, there is no cell to the left.

This expression is invalid at `col = 0`:

```text
dp[row][col - 1]
```

The first column must come only from above.

#### Using `max` instead of `min`

The problem asks for the minimum path sum, so each transition chooses the smaller predecessor:

```text
min(above, left)
```

Using `max` solves a different problem.

#### Thinking greedily is enough

A tempting greedy idea is:

```text
At each cell, move to the smaller neighboring cell.
```

That is not reliable because a locally cheap cell can lead into an expensive region later.

Dynamic programming works because it compares complete minimum costs to reach each cell, not just the next immediate cell value.

#### Optimizing space too early

A one-dimensional DP array is useful, but it hides the meaning of the state.

Understand the two-dimensional table first:

```text
dp[row][col] = best cost to reach this exact cell
```

Then optimize only after the dependency pattern is clear.

---

### 12. First-Principles Summary

This problem is about paths through a grid with one-way movement.

Because movement is restricted to right and down, every cell has at most two possible predecessors:

```text
above
left
```

So the cheapest way to reach a cell depends only on:

```text
the cheapest way to reach the cell above
the cheapest way to reach the cell on the left
the value of the current cell
```

That gives the state:

```text
dp[row][col] = minimum sum to reach (row, col)
```

and the transition:

```text
dp[row][col] = grid[row][col] + min(dp[row - 1][col], dp[row][col - 1])
```

The DP table is filled in an order where each cell's dependencies are already known.

The final answer is the DP value at the bottom-right corner.

## Implementation
See `solutions/dynamic_programming_multidimensional/p064_minimum_path_sum.py`.

## Tests
See `tests/dynamic_programming_multidimensional/test_p064_minimum_path_sum.py`.

## Examples

### Example 1
- Input: `{'grid': [[1, 3, 1], [1, 5, 1], [4, 2, 1]]}`
- Output: `7`

### Example 2
- Input: `{'grid': [[1, 2, 3], [4, 5, 6]]}`
- Output: `12`

## Follow-up Practice
- Draw the DP table for a `2 x 3` grid and label what each entry means.
- Explain why a first-row cell cannot use an "above" predecessor.
- Explain why a first-column cell cannot use a "left" predecessor.
- Rewrite the solution using a one-dimensional DP array after the full-table version is clear.
