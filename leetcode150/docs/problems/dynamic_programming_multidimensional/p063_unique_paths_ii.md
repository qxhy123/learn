# 63. Unique Paths II

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/unique-paths-ii/
- Official Group: Multidimensional DP
- Pattern Group: Dynamic Programming Multidimensional
- Patterns: dynamic-programming-multidimensional

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given an `m x n` grid called `obstacleGrid`.

Each cell is either:

```text
0 = empty cell, the robot may stand on it
1 = obstacle, the robot may not stand on it
```

The robot starts at the top-left cell:

```text
(0, 0)
```

and wants to reach the bottom-right cell:

```text
(m - 1, n - 1)
```

On every move, the robot may move only:

```text
right or down
```

The task is to count how many different valid paths reach the destination without ever stepping on an obstacle.

For example:

```text
obstacleGrid = [
  [0, 0, 0],
  [0, 1, 0],
  [0, 0, 0]
]
```

The middle cell is blocked. Without obstacles, a `3 x 3` grid has `6` paths. But any path that goes through the center is invalid, leaving only these two routes:

```text
right -> right -> down  -> down
down  -> down  -> right -> right
```

So the answer is:

```text
2
```

The real problem is:

> Count all right/down paths from the start to the finish, while treating obstacle cells as places where the path count is forced to zero.

---

### 2. Start From the Brute Force Idea

The most direct way to think about the problem is recursive exploration.

From a cell `(row, col)`, try both possible next moves:

```text
(row, col + 1)  # move right
(row + 1, col)  # move down
```

If the move leaves the grid or lands on an obstacle, that branch contributes `0` paths. If the branch reaches the destination, it contributes `1` path.

Conceptually:

```python
def count_paths(row, col):
    if row is outside the grid or col is outside the grid:
        return 0

    if obstacleGrid[row][col] == 1:
        return 0

    if (row, col) is the bottom-right cell:
        return 1

    return count_paths(row + 1, col) + count_paths(row, col + 1)
```

This is correct because every valid path from a non-destination cell must begin with exactly one of two moves: down or right.

But it is inefficient. Many branches ask the same question repeatedly.

For example, in a grid with no obstacles, the cell `(2, 2)` can be reached from several different partial paths. A naive recursion recomputes the number of ways from `(2, 2)` to the end each time it arrives there.

That repeated work is the signal that dynamic programming should be used.

---

### 3. The Key Observation

A path that ends at cell `(row, col)` can only arrive from two possible previous cells:

```text
(row - 1, col)  # from above
(row, col - 1)  # from the left
```

There is no other way to enter `(row, col)` because the robot can only move down or right.

So if `(row, col)` is not an obstacle, the number of ways to reach it is:

```text
ways from above + ways from left
```

If `(row, col)` is an obstacle, the number of ways to reach it is:

```text
0
```

because the robot is not allowed to stand there.

This gives the entire recurrence:

```text
if obstacleGrid[row][col] == 1:
    dp[row][col] = 0
else:
    dp[row][col] = dp[row - 1][col] + dp[row][col - 1]
```

with careful handling for the top row and left column, where one of those previous cells may not exist.

---

### 4. DP State and Invariant

Define:

```text
dp[row][col] = number of valid paths from (0, 0) to (row, col)
```

where a valid path:

- starts at `(0, 0)`;
- moves only right or down;
- never enters an obstacle cell;
- ends exactly at `(row, col)`.

The invariant is:

```text
After dp[row][col] is filled, it stores exactly the number of valid paths to that cell.
```

This state is complete because the future does not need to know the exact route used to reach `(row, col)`. It only needs to know how many valid routes reached that cell. Since every future move from `(row, col)` has the same choices regardless of the previous route, the count is enough.

Obstacles fit naturally into the invariant:

```text
An obstacle cell has zero valid paths to it.
```

That means obstacle cells do not need special graph logic. They simply contribute `0` to every cell that might otherwise depend on them.

---

### 5. Base Cases

The starting cell is special.

If the start is blocked:

```text
obstacleGrid[0][0] == 1
```

then the robot cannot even begin, so the answer is:

```text
0
```

If the start is empty, there is exactly one way to be at the start before making any moves:

```text
dp[0][0] = 1
```

The first row also has a special shape. A cell in the first row can only be reached from the left, because there is no row above it.

So for the first row:

```text
if the current cell is empty:
    dp[0][col] = dp[0][col - 1]
else:
    dp[0][col] = 0
```

Once an obstacle appears in the first row, every cell to its right becomes unreachable unless there were another way around it. But on the first row there is no way around it, because the robot cannot move up from a lower row.

The first column is symmetric. A cell in the first column can only be reached from above:

```text
if the current cell is empty:
    dp[row][0] = dp[row - 1][0]
else:
    dp[row][0] = 0
```

Once an obstacle appears in the first column, every cell below it in that column becomes unreachable from the start through that column.

A common implementation avoids separate first-row and first-column loops by treating missing neighbors as `0` and initializing only `dp[0][0]`.

---

### 6. Detailed Algorithm

Use a 2D DP table with the same dimensions as the grid.

1. Let:

```text
rows = len(obstacleGrid)
cols = len(obstacleGrid[0])
```

2. If the start cell is an obstacle, return `0`.

3. Create a `rows x cols` table filled with `0`.

4. Set:

```text
dp[0][0] = 1
```

5. Scan the grid from top to bottom and left to right.

6. For each cell `(row, col)`:

   - If it is the start cell, it is already initialized.
   - If it is an obstacle, set `dp[row][col] = 0`.
   - Otherwise, add the ways from the cell above if `row > 0`.
   - Add the ways from the cell on the left if `col > 0`.

7. Return:

```text
dp[rows - 1][cols - 1]
```

This fill order works because `dp[row][col]` depends only on cells that appear earlier in row-major order:

```text
above: (row - 1, col)
left:  (row, col - 1)
```

Both are already computed before the current cell is computed.

---

### 7. Pseudocode

```python
def uniquePathsWithObstacles(obstacleGrid):
    rows = len(obstacleGrid)
    cols = len(obstacleGrid[0])

    if obstacleGrid[0][0] == 1:
        return 0

    dp = [[0] * cols for _ in range(rows)]
    dp[0][0] = 1

    for row in range(rows):
        for col in range(cols):
            if row == 0 and col == 0:
                continue

            if obstacleGrid[row][col] == 1:
                dp[row][col] = 0
                continue

            from_above = dp[row - 1][col] if row > 0 else 0
            from_left = dp[row][col - 1] if col > 0 else 0

            dp[row][col] = from_above + from_left

    return dp[rows - 1][cols - 1]
```

A space-optimized version can use one row instead of a full table:

```python
def uniquePathsWithObstacles(obstacleGrid):
    rows = len(obstacleGrid)
    cols = len(obstacleGrid[0])

    dp = [0] * cols
    dp[0] = 1

    for row in range(rows):
        for col in range(cols):
            if obstacleGrid[row][col] == 1:
                dp[col] = 0
            elif col > 0:
                dp[col] += dp[col - 1]

    return dp[cols - 1]
```

In the 1D version:

```text
dp[col] before update  = ways from above
dp[col - 1]            = ways from left after the current row has been processed up to col - 1
```

When the current cell is an obstacle, `dp[col]` must be reset to `0`; otherwise, paths from above would incorrectly pass through the obstacle.

---

### 8. Example Walkthrough

Use the first official example:

```text
obstacleGrid = [
  [0, 0, 0],
  [0, 1, 0],
  [0, 0, 0]
]
```

Start with an empty DP table:

```text
[0, 0, 0]
[0, 0, 0]
[0, 0, 0]
```

The start cell is empty, so:

```text
dp[0][0] = 1
```

Table:

```text
[1, 0, 0]
[0, 0, 0]
[0, 0, 0]
```

Cell `(0, 1)` is in the first row. It can only come from the left:

```text
dp[0][1] = dp[0][0] = 1
```

Table:

```text
[1, 1, 0]
[0, 0, 0]
[0, 0, 0]
```

Cell `(0, 2)` also can only come from the left:

```text
dp[0][2] = dp[0][1] = 1
```

Table:

```text
[1, 1, 1]
[0, 0, 0]
[0, 0, 0]
```

Cell `(1, 0)` is in the first column. It can only come from above:

```text
dp[1][0] = dp[0][0] = 1
```

Table:

```text
[1, 1, 1]
[1, 0, 0]
[0, 0, 0]
```

Cell `(1, 1)` is an obstacle, so it has zero paths:

```text
dp[1][1] = 0
```

Table:

```text
[1, 1, 1]
[1, 0, 0]
[0, 0, 0]
```

Cell `(1, 2)` is empty. It can come from above or left:

```text
from above = dp[0][2] = 1
from left  = dp[1][1] = 0

dp[1][2] = 1 + 0 = 1
```

Table:

```text
[1, 1, 1]
[1, 0, 1]
[0, 0, 0]
```

Cell `(2, 0)` can only come from above:

```text
dp[2][0] = dp[1][0] = 1
```

Table:

```text
[1, 1, 1]
[1, 0, 1]
[1, 0, 0]
```

Cell `(2, 1)` is empty:

```text
from above = dp[1][1] = 0
from left  = dp[2][0] = 1

dp[2][1] = 0 + 1 = 1
```

Table:

```text
[1, 1, 1]
[1, 0, 1]
[1, 1, 0]
```

Cell `(2, 2)` is the destination:

```text
from above = dp[1][2] = 1
from left  = dp[2][1] = 1

dp[2][2] = 1 + 1 = 2
```

Final table:

```text
[1, 1, 1]
[1, 0, 1]
[1, 1, 2]
```

The bottom-right value is `2`, so there are two valid paths.

---

### 9. Correctness

We prove that the algorithm returns the number of valid paths from the top-left cell to the bottom-right cell.

#### Lemma 1: Obstacle cells have zero valid paths.

A valid path may not enter a cell whose value is `1`. Therefore no valid path can end at an obstacle cell. The algorithm assigns `0` to every obstacle cell, so it is correct for those cells.

#### Lemma 2: For every empty non-start cell, every valid path to that cell comes from either the cell above or the cell to the left.

The robot can only move right or down. Therefore the final move into `(row, col)` must be:

- a down move from `(row - 1, col)`, or
- a right move from `(row, col - 1)`.

No other final move is possible.

#### Lemma 3: For every empty non-start cell, the recurrence counts exactly all valid paths to that cell.

By Lemma 2, every valid path to `(row, col)` belongs to exactly one of two groups: paths that arrive from above and paths that arrive from the left. These groups are disjoint because a path has only one final previous cell.

By the DP invariant, `dp[row - 1][col]` counts the first group and `dp[row][col - 1]` counts the second group. Adding them gives exactly the number of valid paths to `(row, col)`.

#### Lemma 4: The algorithm computes each state after its dependencies.

The table is filled top to bottom and left to right. The state `dp[row][col]` depends only on `dp[row - 1][col]` and `dp[row][col - 1]`, when those cells exist. The cell above is in an earlier row, and the cell on the left is earlier in the same row. Both have already been computed.

#### Theorem: The returned value is correct.

The start cell is initialized correctly: if it is blocked, there are zero paths; otherwise there is one way to be at the start before moving. By Lemmas 1 through 4, every DP cell is computed as the exact number of valid paths to that cell. Therefore `dp[rows - 1][cols - 1]` is exactly the number of valid paths to the destination, which is the required answer.

---

### 10. Complexity

Let:

```text
m = number of rows
n = number of columns
```

The 2D DP algorithm visits every cell once, and each cell does constant work.

```text
Time:  O(m * n)
Space: O(m * n)
```

With the 1D rolling-row optimization, the time is unchanged, but the extra space becomes:

```text
Time:  O(m * n)
Space: O(n)
```

The 1D optimization is possible because each state needs only the current row's left value and the previous row's same-column value.

---

### 11. Common Pitfalls

- Forgetting that a blocked start cell immediately makes the answer `0`.
- Forgetting that a blocked destination cell also results in `0`; the recurrence handles this automatically by setting that cell to `0`.
- Initializing the first row or first column as all `1`s without stopping after an obstacle.
- Letting paths pass through obstacles by skipping obstacle cells instead of forcing their DP value to `0`.
- In the 1D version, forgetting to reset `dp[col] = 0` when `obstacleGrid[row][col] == 1`.
- Mixing up the meaning of `dp[row][col]`: it counts paths to the cell, not paths from the cell.
- Filling the table from bottom-right to top-left while still using the top/left recurrence.

---

### 12. First-Principles Summary

The problem looks like path enumeration, but enumerating every path repeats the same subproblems. The important question for each cell is simple:

```text
How many valid ways can the robot reach this cell?
```

Because the robot only moves right and down, the answer for a cell depends only on two already-solved neighbors:

```text
above + left
```

Obstacles are not special routes to reason around manually. They are cells with zero ways to enter. Once that idea is accepted, the whole problem becomes a table-filling process:

```text
start with 1 way at the beginning;
put 0 on obstacles;
for every empty cell, add paths from above and left;
return the destination count.
```

That is the core first-principles idea behind Unique Paths II.

## Implementation
See `solutions/dynamic_programming_multidimensional/p063_unique_paths_ii.py`.

## Tests
See `tests/dynamic_programming_multidimensional/test_p063_unique_paths_ii.py`.

## Examples

### Example 1
- Input: `{'obstacleGrid': [[0, 0, 0], [0, 1, 0], [0, 0, 0]]}`
- Output: `2`

### Example 2
- Input: `{'obstacleGrid': [[0, 1], [0, 0]]}`
- Output: `1`

## Follow-up Practice
- Recompute the first example with the 1D rolling array and write the array after each row.
- Try a grid where the first row contains an obstacle and explain why every cell to its right has zero paths.
- Try a grid where the destination is blocked and confirm that the recurrence returns `0` automatically.
