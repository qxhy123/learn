# 54. Spiral Matrix

- Difficulty: Medium
- LeetCode: https://leetcode.com/problems/spiral-matrix/
- Official Group: Matrix
- Pattern Group: Matrix
- Patterns: matrix

## First-Principles Explanation

### 1. What Is the Problem Really Asking?

Given an `m x n` matrix, return all of its values in spiral order.

Spiral order means:

```text
start at the top-left cell
move right across the top row
move down the rightmost column
move left across the bottom row
move up the leftmost column
then repeat the same idea on the smaller inner rectangle
```

For example:

```text
matrix = [
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 9]
]
```

The traversal is:

```text
1, 2, 3,   # top row, left to right
6, 9,      # right column, top to bottom
8, 7,      # bottom row, right to left
4,         # left column, bottom to top
5          # remaining inner cell
```

So the answer is:

```text
[1, 2, 3, 6, 9, 8, 7, 4, 5]
```

The important point is that the problem does not ask us to sort, search, rotate, or transform the matrix. It only asks for a particular visiting order.

So the real problem is:

> Visit every matrix cell exactly once, in the order produced by repeatedly peeling off the outer rectangle clockwise.

---

### 2. Start From the Brute Force Idea

A very direct way to think about the problem is to simulate walking through the grid one cell at a time.

We could maintain:

```text
current row
current column
current direction
visited cells
```

At each step:

1. Append the current cell.
2. Mark it visited.
3. Try to move forward in the current direction.
4. If the next cell is outside the matrix or already visited, turn right.
5. Stop after visiting `m * n` cells.

Conceptually:

```python
row = 0
col = 0
direction = right
visited = set()
answer = []

while len(answer) < rows * cols:
    answer.append(matrix[row][col])
    visited.add((row, col))

    next_row, next_col = step(row, col, direction)

    if outside_matrix(next_row, next_col) or (next_row, next_col) in visited:
        direction = turn_right(direction)
        next_row, next_col = step(row, col, direction)

    row, col = next_row, next_col
```

This works, and it is a useful mental model.

But it carries more state than the problem really needs. The spiral is not arbitrary movement. After one full clockwise pass around the outside, the remaining unvisited cells always form a smaller rectangle.

That means we can avoid a `visited` set by tracking the rectangle that remains.

---

### 3. The Key Observation: A Spiral Is Repeated Boundary Peeling

Look at the outer layer of a matrix:

```text
top row
right column
bottom row
left column
```

Once those cells are output, none of them will ever be needed again. The remaining unvisited cells form the inner rectangle:

```text
top boundary moves down
bottom boundary moves up
left boundary moves right
right boundary moves left
```

For a `4 x 5` matrix, the first layer is the perimeter:

```text
[ 1,  2,  3,  4,  5]
[ 6,  7,  8,  9, 10]
[11, 12, 13, 14, 15]
[16, 17, 18, 19, 20]
```

After reading the outer boundary, the remaining problem is exactly the same problem on:

```text
[ 7,  8,  9]
[12, 13, 14]
```

This is the first-principles reduction:

```text
spiral(matrix rectangle)
= outer boundary in clockwise order
+ spiral(inner rectangle)
```

The implementation does not need recursion. It can repeatedly shrink four boundaries.

---

### 4. The Boundary Invariant

Maintain four integers:

```text
top    = first unvisited row
bottom = last unvisited row
left   = first unvisited column
right  = last unvisited column
```

At the start of each loop, the invariant is:

```text
Every unvisited cell lies inside rows top..bottom and columns left..right.
Every cell outside that rectangle has already been appended exactly once.
```

The next layer is read in four directional passes:

```text
1. top row:        (top, left)        -> (top, right)
2. right column:   (top + 1, right)   -> (bottom, right)
3. bottom row:     (bottom, right - 1)-> (bottom, left), if a bottom row remains
4. left column:    (bottom - 1, left) -> (top + 1, left), if a left column remains
```

Then shrink the rectangle:

```text
top += 1
right -= 1
bottom -= 1
left += 1
```

The loop continues while the rectangle is still valid:

```text
top <= bottom and left <= right
```

This invariant is the whole algorithm. It tells us exactly what is still unvisited, which edge to read next, and when to stop.

---

### 5. Why the Last Row and Last Column Need Guards

The tricky part is not the normal rectangular case. The tricky part is when the remaining rectangle has only one row or one column.

#### One Remaining Row

Example:

```text
[1, 2, 3]
```

The top-row pass already reads:

```text
1, 2, 3
```

If we then also perform the bottom-row pass, we would read the same row again in reverse:

```text
2, 1
```

So before reading the bottom row, we must check that the top row and bottom row are different:

```text
top < bottom
```

#### One Remaining Column

Example:

```text
[1]
[2]
[3]
```

The right-column pass already reads the column after the top cell:

```text
2, 3
```

If we then also perform the left-column pass, we would read the same column again upward:

```text
2
```

So before reading the left column, we must check that the left column and right column are different:

```text
left < right
```

These two guards are the difference between a clean boundary solution and a solution with duplicate cells on thin matrices.

---

### 6. Detailed Algorithm

1. If the matrix is empty, return an empty list.
2. Initialize:

```text
top = 0
bottom = number of rows - 1
left = 0
right = number of columns - 1
answer = []
```

3. While the remaining rectangle is valid:

```text
top <= bottom and left <= right
```

4. Traverse the top row from `left` to `right`.
5. Traverse the right column from `top + 1` to `bottom`.
6. If `top < bottom`, traverse the bottom row from `right - 1` down to `left`.
7. If `left < right`, traverse the left column from `bottom - 1` down to `top + 1`.
8. Shrink all four boundaries inward.
9. Return `answer`.

The indexing choices avoid corners being visited twice in the same layer:

```text
The top-right corner is read by the top row, so the right column starts at top + 1.
The bottom-right corner is read by the right column, so the bottom row starts at right - 1.
The bottom-left corner is read by the bottom row, so the left column starts at bottom - 1.
The top-left corner is read by the top row, so the left column stops above top.
```

---

### 7. Walkthrough: `3 x 3` Matrix

Input:

```text
matrix = [
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 9]
]
```

Start:

```text
top = 0, bottom = 2
left = 0, right = 2
answer = []
```

#### First Layer: Top Row

Read row `0`, columns `0..2`:

```text
1, 2, 3
```

Answer:

```text
[1, 2, 3]
```

#### First Layer: Right Column

Read column `2`, rows `1..2`:

```text
6, 9
```

Answer:

```text
[1, 2, 3, 6, 9]
```

#### First Layer: Bottom Row

Because `top < bottom`, there is a distinct bottom row.

Read row `2`, columns `1..0` in reverse:

```text
8, 7
```

Answer:

```text
[1, 2, 3, 6, 9, 8, 7]
```

#### First Layer: Left Column

Because `left < right`, there is a distinct left column.

Read column `0`, rows `1..1` in reverse:

```text
4
```

Answer:

```text
[1, 2, 3, 6, 9, 8, 7, 4]
```

Shrink boundaries:

```text
top = 1, bottom = 1
left = 1, right = 1
```

#### Second Layer: Single Center Cell

The remaining rectangle is just:

```text
matrix[1][1] = 5
```

Top-row pass reads it:

```text
5
```

The right-column pass reads nothing because it would start at row `2` and end at row `1`.

The bottom-row guard fails because:

```text
top < bottom
1 < 1 is false
```

The left-column guard fails because:

```text
left < right
1 < 1 is false
```

Answer:

```text
[1, 2, 3, 6, 9, 8, 7, 4, 5]
```

Shrink boundaries again:

```text
top = 2, bottom = 0
left = 2, right = 0
```

Now the remaining rectangle is invalid, so stop.

---

### 8. Walkthrough: Rectangular `3 x 4` Matrix

Input:

```text
matrix = [
  [ 1,  2,  3,  4],
  [ 5,  6,  7,  8],
  [ 9, 10, 11, 12]
]
```

Start:

```text
top = 0, bottom = 2
left = 0, right = 3
```

Read the outer layer:

```text
top row:      1, 2, 3, 4
right column: 8, 12
bottom row:   11, 10, 9
left column:  5
```

Answer so far:

```text
[1, 2, 3, 4, 8, 12, 11, 10, 9, 5]
```

Shrink:

```text
top = 1, bottom = 1
left = 1, right = 2
```

The remaining rectangle is one row:

```text
[6, 7]
```

Read the top row:

```text
6, 7
```

Do not read a bottom row because `top < bottom` is false.

Final answer:

```text
[1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]
```

---

### 9. Code

```python
from typing import List


class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        if not matrix or not matrix[0]:
            return []

        top = 0
        bottom = len(matrix) - 1
        left = 0
        right = len(matrix[0]) - 1
        answer = []

        while top <= bottom and left <= right:
            for col in range(left, right + 1):
                answer.append(matrix[top][col])

            for row in range(top + 1, bottom + 1):
                answer.append(matrix[row][right])

            if top < bottom:
                for col in range(right - 1, left - 1, -1):
                    answer.append(matrix[bottom][col])

            if left < right:
                for row in range(bottom - 1, top, -1):
                    answer.append(matrix[row][left])

            top += 1
            bottom -= 1
            left += 1
            right -= 1

        return answer
```

Equivalent pseudocode:

```text
answer = []
set top, bottom, left, right to the outer matrix boundaries

while there is still a valid rectangle:
    append top edge from left to right
    append right edge from top + 1 to bottom

    if the bottom edge is distinct from the top edge:
        append bottom edge from right - 1 to left

    if the left edge is distinct from the right edge:
        append left edge from bottom - 1 to top + 1

    move all four boundaries inward

return answer
```

---

### 10. Correctness

We prove that the algorithm returns exactly the matrix elements in spiral order.

#### Invariant

At the start of each loop iteration:

```text
The unvisited cells are exactly the cells inside the rectangle
rows top..bottom and columns left..right.
```

All cells outside this rectangle have already been appended exactly once, in spiral order for the layers already removed.

#### Initialization

Before the first iteration:

```text
top = 0
bottom = m - 1
left = 0
right = n - 1
```

So the rectangle is the entire matrix. No cells have been visited yet. The invariant is true.

#### Maintenance

During one iteration, the algorithm appends the current rectangle's outer boundary clockwise:

1. The top edge is appended left to right.
2. The right edge is appended top to bottom, starting below the top-right corner so that corner is not duplicated.
3. If a distinct bottom edge exists, it is appended right to left, starting left of the bottom-right corner so that corner is not duplicated.
4. If a distinct left edge exists, it is appended bottom to top, excluding both corners so they are not duplicated.

These four passes append exactly the outer layer of the current rectangle in clockwise spiral order.

Then the algorithm moves every boundary inward:

```text
top += 1
bottom -= 1
left += 1
right -= 1
```

After this update, the new rectangle contains exactly the cells that were not part of the removed outer layer. Therefore the invariant remains true for the next iteration.

#### Termination

The loop stops when:

```text
top > bottom or left > right
```

At that point, there is no valid remaining rectangle, so there are no unvisited cells left.

By the invariant, every matrix cell has been appended exactly once. Because each iteration appended the current outer boundary in clockwise order before moving inward, the full answer is exactly the spiral order.

Therefore the algorithm is correct.

---

### 11. Complexity

Every cell is appended once.

Even though there are several loops inside the `while` loop, those loops cover disjoint boundary cells for each layer. Across all layers, the total number of appended cells is exactly `m * n`.

Complexity:

```text
Time:  O(m * n)
Space: O(1) extra space, not counting the output list
```

The output list itself contains `m * n` values, so if output space is counted, space is `O(m * n)`.

---

### 12. Common Pitfalls

#### Duplicating Cells in a Single Row

If the remaining rectangle has one row, the top-row pass already reads all remaining cells.

Without this guard:

```text
if top < bottom
```

the bottom-row pass can read the same row again.

#### Duplicating Cells in a Single Column

If the remaining rectangle has one column, the right-column pass already reads the vertical edge after the top cell.

Without this guard:

```text
if left < right
```

the left-column pass can read the same column again.

#### Reading Corners Twice

Each corner belongs to two edges geometrically, but it should be appended only once.

That is why the passes use asymmetric ranges:

```text
right column starts at top + 1
bottom row starts at right - 1
left column starts at bottom - 1 and stops above top
```

#### Shrinking Too Early

Do not move `top`, `bottom`, `left`, or `right` after each edge unless the rest of the code is designed around that choice.

A simple version is:

```text
read all four edges using the current boundaries
then shrink all four boundaries once
```

This keeps the invariant easy to reason about.

#### Assuming the Matrix Is Square

The matrix can be rectangular:

```text
1 x n
m x 1
m x n
```

The boundary method works for all of them as long as the one-row and one-column guards are present.

#### Confusing Rows and Columns

Rows use `top` and `bottom`.

Columns use `left` and `right`.

A reliable mental check is:

```text
matrix[row][col]
```

So when traversing a row, the row index is fixed and the column changes.

When traversing a column, the column index is fixed and the row changes.

---

### 13. First-Principles Summary

The spiral order is not a complicated path if we look at the shape of the remaining work.

At any moment, the unvisited cells form a rectangle. The next spiral segment is exactly that rectangle's outer boundary, read clockwise. After reading it, the same problem remains on a smaller rectangle.

So the solution is to maintain four boundaries:

```text
top, bottom, left, right
```

and preserve this invariant:

```text
inside the boundaries = not yet visited
outside the boundaries = already output in spiral order
```

The algorithm is just the invariant made concrete:

```text
read top edge
read right edge
read bottom edge if distinct
read left edge if distinct
shrink boundaries
repeat
```

That is why the solution is linear in the number of cells and does not need a `visited` set.

## Implementation
See `solutions/matrix/p054_spiral_matrix.py`.

## Tests
See `tests/matrix/test_p054_spiral_matrix.py`.

## Examples

### Example 1
- Input: `{'matrix': [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}`
- Output: `[1, 2, 3, 6, 9, 8, 7, 4, 5]`

### Example 2
- Input: `{'matrix': [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]}`
- Output: `[1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]`

## Follow-up Practice
- Trace coordinates on a `1x1`, `1xn`, and `mx1` matrix.
- Write boundary updates explicitly.
- Decide whether mutation is allowed.
