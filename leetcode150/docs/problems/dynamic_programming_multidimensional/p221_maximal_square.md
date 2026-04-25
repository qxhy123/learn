# 221. Maximal Square

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/maximal-square/
- Official Group: Multidimensional DP
- Pattern Group: Dynamic Programming Multidimensional
- Patterns: dynamic-programming-multidimensional

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

You are given a binary matrix whose cells are strings:

```text
"0" or "1"
```

You need to find the area of the largest square that contains only `"1"` cells.

Two details matter:

```text
square
all 1s
```

A square is not just any rectangle. Its height and width must be the same.

For example, this shape has six `1` cells:

```text
1 1 1
1 1 1
```

but it is a `2 x 3` rectangle, not a square, so its area is not a valid answer for a `3 x 3` square.

The output is also the **area**, not the side length.

So if the largest all-`1` square has side length `2`, the answer is:

```text
2 * 2 = 4
```

The real problem is:

> Among all possible square submatrices, find the largest side length whose every cell is `"1"`, then return side length squared.

---

### 2. Start From the Brute Force Idea

The most direct approach is to try every square.

A square can be described by:

```text
top-left row
top-left column
side length
```

Then we check whether every cell inside that square is `"1"`.

Conceptually:

```python
best_side = 0

for top in range(rows):
    for left in range(cols):
        for side in range(1, min(rows - top, cols - left) + 1):
            if every cell in matrix[top:top + side][left:left + side] is "1":
                best_side = max(best_side, side)

return best_side * best_side
```

This is correct because it explicitly considers every candidate square.

But it repeats a huge amount of work.

If we already know this `3 x 3` region is valid:

```text
1 1 1
1 1 1
1 1 1
```

then checking a nearby `4 x 4` square from scratch rechecks many of the same cells.

In the worst case, the matrix is full of `1`s. Then there are many possible squares, and large squares are expensive to verify cell by cell.

The brute-force question becomes:

> Can we determine whether a square ending at a cell is valid using smaller squares that we already understand?

That question leads directly to dynamic programming.

---

### 3. The Key Observation

Instead of thinking about a square by its top-left corner, think about it by its bottom-right corner.

Suppose cell `(r, c)` is the bottom-right corner of an all-`1` square.

If the square has side length `k`, then three smaller neighboring regions must also support side length at least `k - 1`:

```text
top-left neighbor: (r - 1, c - 1)
top neighbor:      (r - 1, c)
left neighbor:     (r, c - 1)
```

Why exactly these three?

To grow a square ending at `(r, c)`, we need:

1. A smaller square above-left to supply the interior.
2. Enough `1`s above `(r, c)` to supply the right edge.
3. Enough `1`s left of `(r, c)` to supply the bottom edge.

For a `3 x 3` square ending at `X`, the shape is:

```text
? ? ?
? ? ?
? ? X
```

The `2 x 2` square ending up-left covers the interior corner:

```text
1 1 .
1 1 .
. . X
```

The `2 x 2` square ending above proves the cells above the right side can be part of a square:

```text
. 1 1
. 1 1
. . X
```

The `2 x 2` square ending left proves the cells on the bottom side can be part of a square:

```text
. . .
1 1 .
1 1 X
```

All three constraints must hold. If any one of them can only support side length `1`, then the new square cannot have side length `3`.

That gives the central formula:

```text
if matrix[r][c] == "1":
    dp[r][c] = 1 + min(
        dp[r - 1][c],
        dp[r][c - 1],
        dp[r - 1][c - 1]
    )
else:
    dp[r][c] = 0
```

The `min` is essential because a square can only grow as far as its weakest required neighbor allows.

---

### 4. DP State and Invariant

Define:

```text
dp[r][c] = side length of the largest all-1 square whose bottom-right corner is cell (r, c)
```

This state is deliberately specific.

It does **not** mean:

```text
largest square anywhere in matrix[0:r][0:c]
```

It only describes squares that end exactly at `(r, c)`.

That makes the transition local.

The invariant is:

> After processing cell `(r, c)`, `dp[r][c]` equals the largest possible side length of an all-`1` square with bottom-right corner `(r, c)`.

Once that invariant is true for every cell, the global answer is simply:

```text
max(dp[r][c]) over all cells
```

Then return:

```text
max_side * max_side
```

because the problem asks for area.

---

### 5. Why the Recurrence Works

Consider one cell `(r, c)`.

#### Case 1: `matrix[r][c] == "0"`

No all-`1` square can end at `(r, c)` because the bottom-right cell itself is `0`.

So:

```text
dp[r][c] = 0
```

#### Case 2: `matrix[r][c] == "1"`

At minimum, the single cell square exists:

```text
side length = 1
```

To build something larger, say side length `k`, the square ending at `(r, c)` must include:

```text
the (k - 1) x (k - 1) region above it
the (k - 1) x (k - 1) region to the left
the (k - 1) x (k - 1) region above-left
```

Those are summarized by:

```text
dp[r - 1][c]
dp[r][c - 1]
dp[r - 1][c - 1]
```

The new square can only be one larger than the smallest of those three.

So:

```text
dp[r][c] = 1 + min(top, left, diagonal)
```

This is not just a trick. It is the geometric requirement for a square: all three neighboring directions must have enough supporting `1`s.

---

### 6. Boundary Cells

The first row and first column have no full set of three neighbors.

For those cells, the largest square ending there can only have side length `1` if the cell is `"1"`, otherwise `0`.

For example, in the first row:

```text
1 1 1 1
```

Even four consecutive `1`s do not form a square taller than `1`, because there is no row above them.

There are two common implementation styles:

1. Handle first row and first column with special cases.
2. Use a padded DP table with one extra row and one extra column of zeros.

The padded version is usually cleaner.

If `matrix` has `rows x cols`, create:

```text
dp = (rows + 1) x (cols + 1), filled with 0
```

Then matrix cell `(r - 1, c - 1)` maps to DP cell `(r, c)`.

This lets every real cell read:

```text
dp[r - 1][c]
dp[r][c - 1]
dp[r - 1][c - 1]
```

without falling off the table.

---

### 7. Detailed Algorithm

1. If the matrix is empty, return `0`.
2. Let `rows` and `cols` be the matrix dimensions.
3. Create a DP table of zeros with dimensions `(rows + 1) x (cols + 1)`.
4. Set `max_side = 0`.
5. For every matrix cell from top-left to bottom-right:
   - If the cell is `"0"`, leave the corresponding DP value as `0`.
   - If the cell is `"1"`, compute:

```text
dp[r][c] = 1 + min(dp[r - 1][c], dp[r][c - 1], dp[r - 1][c - 1])
```

   - Update `max_side`.
6. Return `max_side * max_side`.

The traversal order matters. When computing `dp[r][c]`, the top, left, and diagonal states must already be known. A normal row-by-row scan satisfies that requirement.

---

### 8. Pseudocode

```python
def maximalSquare(matrix):
    if not matrix or not matrix[0]:
        return 0

    rows = len(matrix)
    cols = len(matrix[0])
    dp = [[0] * (cols + 1) for _ in range(rows + 1)]
    max_side = 0

    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            if matrix[r - 1][c - 1] == "1":
                dp[r][c] = 1 + min(
                    dp[r - 1][c],
                    dp[r][c - 1],
                    dp[r - 1][c - 1],
                )
                max_side = max(max_side, dp[r][c])

    return max_side * max_side
```

The implementation can also be space-optimized to one row, but the full table is better for learning because it directly matches the invariant.

---

### 9. Example Walkthrough

Use Example 1:

```text
matrix =
1 0 1 0 0
1 0 1 1 1
1 1 1 1 1
1 0 0 1 0
```

We compute `dp[r][c]` as the largest square side length ending at that exact cell.

For readability, this walkthrough shows the unpadded DP values aligned with the matrix.

#### Row 0

```text
matrix row: 1 0 1 0 0
dp row:     1 0 1 0 0
```

Every `1` in the first row can only form a `1 x 1` square.

Current best side:

```text
1
```

#### Row 1

Cell `(1, 0)` is `1`, first column, so:

```text
dp[1][0] = 1
```

Cell `(1, 2)` is `1`. Its top is `1`, left is `0`, and diagonal is `0`:

```text
1 + min(1, 0, 0) = 1
```

Cell `(1, 3)` is `1`. Its top is `0`, so it cannot form a `2 x 2` square:

```text
1 + min(0, 1, 1) = 1
```

Cell `(1, 4)` is `1`. Its top is `0`, so again only side `1`:

```text
1 + min(0, 1, 0) = 1
```

DP so far:

```text
1 0 1 0 0
1 0 1 1 1
```

#### Row 2

The matrix row is:

```text
1 1 1 1 1
```

Compute each cell:

```text
dp[2][0] = 1
```

For `(2, 1)`, top is `0`, so only side `1`:

```text
1 + min(0, 1, 1) = 1
```

For `(2, 2)`, left is `1`, top is `1`, but diagonal is `0`:

```text
1 + min(1, 1, 0) = 1
```

For `(2, 3)`, the three neighbors are all at least `1`:

```text
top      = 1
left     = 1
diagonal = 1

1 + min(1, 1, 1) = 2
```

This means there is a `2 x 2` all-`1` square ending at `(2, 3)`:

```text
1 1
1 1
```

using rows `1..2` and columns `2..3`.

For `(2, 4)`, top is `1`, left is `2`, and diagonal is `1`:

```text
1 + min(1, 2, 1) = 2
```

DP so far:

```text
1 0 1 0 0
1 0 1 1 1
1 1 1 2 2
```

Current best side:

```text
2
```

#### Row 3

The matrix row is:

```text
1 0 0 1 0
```

Cells with `0` produce DP value `0`.

Cell `(3, 0)` is first column and `1`, so:

```text
dp[3][0] = 1
```

Cell `(3, 3)` is `1`, but its left neighbor in DP is `0`, so it cannot extend to side `2`:

```text
1 + min(top=2, left=0, diagonal=1) = 1
```

Final DP table:

```text
1 0 1 0 0
1 0 1 1 1
1 1 1 2 2
1 0 0 1 0
```

The largest side length is `2`, so the returned area is:

```text
2 * 2 = 4
```

---

### 10. Correctness

We prove that the algorithm returns the area of the largest all-`1` square.

#### Lemma 1: If `matrix[r][c] == "0"`, then `dp[r][c] = 0` is correct.

Any square whose bottom-right corner is `(r, c)` must include cell `(r, c)`. Since that cell is `0`, no all-`1` square can end there. Therefore the largest valid side length is `0`.

#### Lemma 2: If `matrix[r][c] == "1"`, then `1 + min(top, left, diagonal)` is the largest valid side length ending at `(r, c)`.

Let:

```text
top      = dp[r - 1][c]
left     = dp[r][c - 1]
diagonal = dp[r - 1][c - 1]
```

The cell `(r, c)` alone forms a square of side `1`.

For any larger square of side `k`, the square must have side `k - 1` supported immediately above, immediately left, and diagonally above-left. Therefore `k - 1` cannot exceed any of `top`, `left`, or `diagonal`. So:

```text
k <= 1 + min(top, left, diagonal)
```

Conversely, if the minimum of those three values is `m`, then each required neighboring region can support side length `m`. Adding the current `"1"` cell completes a square of side `m + 1` ending at `(r, c)`. Thus the largest possible side length is exactly:

```text
1 + min(top, left, diagonal)
```

#### Lemma 3: Every DP value is computed before it is needed.

The algorithm scans rows from top to bottom and columns from left to right. When computing a cell, its top, left, and diagonal neighbors have already been processed. Therefore every transition uses finalized values.

#### Theorem: The algorithm returns the correct answer.

By Lemmas 1 and 2, every `dp` cell stores the largest all-`1` square side length ending at that cell. Every possible square has some bottom-right corner, so taking the maximum over all DP cells gives the largest side length anywhere in the matrix. The algorithm returns that side length squared, which is exactly the requested area.

---

### 11. Complexity

Let:

```text
rows = number of rows in matrix
cols = number of columns in matrix
```

The algorithm visits each cell once, and each visit does constant work.

Time complexity:

```text
O(rows * cols)
```

The full DP table stores one value per cell, plus padding.

Space complexity:

```text
O(rows * cols)
```

With a rolling row, space can be reduced to:

```text
O(cols)
```

but the transition must preserve the previous diagonal value carefully.

---

### 12. Common Pitfalls

- Returning `max_side` instead of `max_side * max_side`. The problem asks for area.
- Treating the matrix values as integers when they are strings. Compare with `"1"`, not `1`, unless the input is converted first.
- Using `max(top, left, diagonal)` instead of `min(...)`. A square is limited by the weakest neighboring support.
- Forgetting the diagonal neighbor. Top and left alone are not enough to prove the interior of the square is filled.
- Defining `dp[r][c]` as the best square anywhere so far. The recurrence works only when `dp[r][c]` means “best square ending exactly here.”
- Mishandling the first row or first column. Padding the DP table avoids most boundary mistakes.
- Optimizing to one-dimensional DP too early. The full table is easier to reason about and less error-prone.

---

### 13. First-Principles Summary

The brute-force method asks, “Is this entire candidate square full of `1`s?” over and over.

The DP method asks a sharper question:

> If a square must end at this cell, how large can it be?

That local question has a local answer. A `"0"` cell supports no square. A `"1"` cell supports a square one larger than the smallest square supported by its top, left, and diagonal neighbors.

So the whole problem reduces to maintaining this invariant:

```text
dp[r][c] = largest all-1 square side length ending at (r, c)
```

Once that invariant is computed for every cell, the largest side length anywhere is known, and squaring it gives the required area.

## Implementation
See `solutions/dynamic_programming_multidimensional/p221_maximal_square.py`.

## Tests
See `tests/dynamic_programming_multidimensional/test_p221_maximal_square.py`.

## Examples

### Example 1
- Input: `{'matrix': [['1', '0', '1', '0', '0'], ['1', '0', '1', '1', '1'], ['1', '1', '1', '1', '1'], ['1', '0', '0', '1', '0']]}`
- Output: `4`

### Example 2
- Input: `{'matrix': [['0', '1'], ['1', '0']]}`
- Output: `1`

### Example 3
- Input: `{'matrix': [['0']]}`
- Output: `0`

## Follow-up Practice
- Draw the DP table for `[["1", "1"], ["1", "1"]]` and confirm why the bottom-right value becomes `2`.
- Change one of those four cells to `"0"` and identify which neighbor blocks the square from growing.
- Re-derive the recurrence using the phrase: “a larger square needs top, left, and diagonal support.”
